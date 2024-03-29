from torch.nn.parameter import Parameter

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    def __init__(self, vocab_size, ehr_adj, device, emb_dim=64):
        super(Model, self).__init__()
        K = len(vocab_size)
        self.K = K
        self.vocab_size = vocab_size
        self.device = device

        self.diag_embeddings = nn.Embedding(vocab_size[0], emb_dim)
        self.pro_embeddings = nn.Embedding(vocab_size[1], emb_dim)
        self.age_embeddings = nn.Embedding(vocab_size[2], emb_dim)
        self.hr_embeddings = nn.Embedding(vocab_size[3], emb_dim)
        self.med_embeddings = nn.Embedding(vocab_size[-1], emb_dim)

        self.dropout = nn.Dropout(p=0.4)

        self.encoders = nn.ModuleList([nn.GRU(
                emb_dim, emb_dim * 2, 
                batch_first=True) for _ in range(K-1)])

        self.query = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 8, emb_dim),
        )

        self.ehr_gcn = GCN(voc_size=vocab_size[-1],
                emb_dim=emb_dim, adj=ehr_adj, device=device)

        self.inter = Parameter(torch.FloatTensor(1))

        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 3, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, vocab_size[-1])
        )

        self.final_output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(vocab_size[-1] * 2 , vocab_size[-1]),
        )

        self.init_weights()

    def forward(self, input, age, heartRate):

        full_user_diagnoses = []

        # generate medical embeddings and queries
        diag_seq = []
        pro_seq = []
        age_seq = []
        hr_seq = []

        def mean_embedding(embedding):
            return embedding.mean(dim=1).unsqueeze(dim=0)  # (1,1,dim)

        for adm in input:

            # Diagnosis
            diag_val = mean_embedding(self.dropout(
                    self.diag_embeddings(torch.LongTensor(adm[0])
                        .unsqueeze(dim=0)
                        .to(self.device)))) # (1,1,dim)

            # Procedures
            pro_val = mean_embedding(self.dropout(
                    self.pro_embeddings(torch.LongTensor(adm[1])
                        .unsqueeze(dim=0)
                        .to(self.device))))

            # Age
            age_val = mean_embedding(self.dropout(
                    self.age_embeddings(torch.LongTensor([age])
                        .unsqueeze(dim=0)
                        .to(self.device))))

            # HeartRate
            hr_val = mean_embedding(self.dropout(
                    self.hr_embeddings(torch.LongTensor([heartRate[0], heartRate[1]])
                        .unsqueeze(dim=0)
                        .to(self.device))))

            full_user_diagnoses = full_user_diagnoses + adm[0]

            diag_seq.append(diag_val)
            pro_seq.append(pro_val)
            age_seq.append(age_val)
            hr_seq.append(hr_val)

        diag_seq = torch.cat(diag_seq, dim=1) #(1,seq,dim)
        pro_seq = torch.cat(pro_seq, dim=1) #(1,seq,dim)
        age_seq = torch.cat(age_seq, dim=1) #(1,seq,dim)
        hr_seq = torch.cat(hr_seq, dim=1) #(1,seq,dim)

        o1, _ = self.encoders[0](
            diag_seq
        ) # o1:(1, seq, dim*2) hi:(1,1,dim*2)

        o2, _ = self.encoders[1](
            pro_seq
        )

        o3, _ = self.encoders[2](
            age_seq
        )

        o4, _ = self.encoders[3](
            hr_seq
        )

        patient_representations = torch.cat([o1, o2, o3, o4], dim=-1).squeeze(dim=0) # (seq, dim*4)
        queries = self.query(patient_representations) # (seq, dim)

        # graph memory module
        '''I:generate current input'''
        query = queries[-1:] # (1,dim)

        #ignore
        history_values = torch.tensor([0])
        history_keys= torch.tensor([0])

        drug_memory = self.ehr_gcn()

        if len(input) > 1:
            history_keys = queries[:(queries.size(0)-1)] # (seq-1, dim)
            history_values = np.zeros((len(input)-1, self.vocab_size[-1]))
            for idx, adm in enumerate(input):
                if idx == len(input)-1:
                    break
                history_values[idx, adm[2]] = 1
            history_values = torch.FloatTensor(history_values).to(self.device) # (seq-1, size)
            
        '''O:read from global memory bank and dynamic memory bank'''
        key_weights1 = F.softmax(torch.mm(query, drug_memory.t()), dim=-1)  # (1, size)
        fact1 = torch.mm(key_weights1, drug_memory)  # (1, dim)

        if len(input) > 1:
            visit_weight = F.softmax(torch.mm(query, history_keys.t()), dim=-1) # (1, seq-1)
            weighted_values = visit_weight.mm(history_values) # (1, size)
            fact2 = torch.mm(weighted_values, drug_memory) # (1, dim)
        else:
            fact2 = fact1
        '''R:convert O and predict'''



        output_1 = self.output(torch.cat([query, fact1, fact2], dim=-1)) # (1, dim)

        '''Reading from Item Collaborative_Filtering Model'''
        diag_embeddings = self.dropout(
                            self.diag_embeddings(torch.LongTensor(full_user_diagnoses)
                                .to(self.device))) # (len(user_diagnoses) ,dim)
        med_embeddings = self.med_embeddings.weight.T.to(self.device)

        matrix_fact = torch.matmul(diag_embeddings,med_embeddings)  # (len(user_diagnoses) , len(medicine))
        output_2 = matrix_fact.mean(dim=0).unsqueeze(dim=0)  # (1,1,dim)

        output = self.final_output(torch.cat([output_1, output_2], dim=-1)) # (1, dim)

        if self.training:
            neg_pred_prob = torch.sigmoid(output)
            neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)
            batch_neg = neg_pred_prob

            return output, batch_neg
        else:
            return output

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1

        self.diag_embeddings.weight.data.uniform_(-initrange, initrange)

        self.pro_embeddings.weight.data.uniform_(-initrange, initrange)

        self.med_embeddings.weight.data.uniform_(-initrange, initrange)

        self.hr_embeddings.weight.data.uniform_(-initrange, initrange)

        self.inter.data.uniform_(-initrange, initrange)



class GCN(nn.Module):
    def __init__(self, voc_size, emb_dim, adj, device=torch.device('cpu:0')):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device

        adj = self.normalize(adj + np.eye(adj.shape[0]))


        self.adj = torch.FloatTensor(adj).to(device)
        self.x = torch.eye(voc_size).to(device)

        self.gcn1 = GraphConvolution(voc_size, emb_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)

    def forward(self):
        node_embedding = self.gcn1(self.x, self.adj)
        node_embedding = F.relu(node_embedding)
        node_embedding = self.dropout(node_embedding)
        node_embedding = self.gcn2(node_embedding, self.adj)
        return node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


