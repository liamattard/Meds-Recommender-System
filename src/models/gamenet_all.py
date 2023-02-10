from torch.nn.parameter import Parameter

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Model(nn.Module):
    def __init__(self, ehr_adj, device, features, voc, emb_dim=64):
        super(Model, self).__init__()

        self.med_voc_len = len(voc["med_voc"].idx2word)

        self.embeddings = []

        x = 0

        if "age" in features:
            self.age_embeddings = nn.Embedding(
                len(voc["age_voc"].idx2word), emb_dim)
            self.embeddings.append(self.age_embeddings)
            self.age_encoder = nn.GRU(
                emb_dim, emb_dim * 2, batch_first=True)
            x += 2

        if "gender" in features:
            self.gender_embeddings = nn.Embedding(
                len(voc["gender_voc"].idx2word), emb_dim)
            self.embeddings.append(self.gender_embeddings)
            self.gender_encoder = nn.GRU(
                emb_dim, emb_dim * 2, batch_first=True)
            x += 2

        if "insurance" in features:
            self.insurance_embeddings = nn.Embedding(
                len(voc["insurance_voc"].idx2word), emb_dim)
            self.embeddings.append(self.insurance_embeddings)
            self.insurance_encoder = nn.GRU(
                emb_dim, emb_dim * 2, batch_first=True)
            x += 2

        if "diagnosis" in features:
            self.diag_embeddings = nn.Embedding(
                len(voc["diag_voc"].idx2word), emb_dim)
            self.embeddings.append(self.diag_embeddings)
            self.diagnosis_encoder = nn.GRU(
                emb_dim, emb_dim * 2, batch_first=True)
            x += 2

        if "procedures" in features:
            self.proc_embeddings = nn.Embedding(
                len(voc["pro_voc"].idx2word), emb_dim)
            self.embeddings.append(self.proc_embeddings)
            self.procedures_encoder = nn.GRU(
                emb_dim, emb_dim * 2, batch_first=True)
            x += 2

        self.device = device
        self.dropout = nn.Dropout(p=0.4)

        #  for _ in range(len(features)-1)

        self.query = nn.Sequential(
            nn.Linear(emb_dim * x, 64),
            nn.LeakyReLU(),
            nn.Linear(64, emb_dim),
        )

        self.ehr_gcn = GCN(voc_size=self.med_voc_len,
                           emb_dim=emb_dim, adj=ehr_adj, device=device)

        self.inter = Parameter(torch.FloatTensor(1))

        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 3, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, self.med_voc_len)
        )

        self.init_weights()

    def forward(self, input):

        def mean_embedding(embedding):
            return embedding.mean(dim=1).unsqueeze(dim=0)  # (1,1,dim)

        def get_encoder_result(input, layer, encoder, item):
            values = []

            for adm in input[item]:
                values.append(get_embedding(layer, adm))

            seq = torch.cat(values, dim=1)  # (1,seq,dim)

            result, _ = encoder(
                seq
            )

            return result

        def get_embedding(layer, value):
            return mean_embedding(self.dropout(
                layer(torch.LongTensor(value)
                      .unsqueeze(dim=0)
                      .to(self.device))))  # (1,1,dim)

        encoders = []

        if "diagnosis" in input:
            encoders.append(get_encoder_result(
                input, self.diag_embeddings, self.diagnosis_encoder, "diagnosis"))

        if "procedures" in input:
            encoders.append(get_encoder_result(
                input, self.proc_embeddings, self.procedures_encoder, "procedures"))

        if "age" in input:
            encoders.append(get_encoder_result(
                input, self.age_embeddings, self.age_encoder, "age"))

        if "insurance" in input:
            encoders.append(get_encoder_result(
                input, self.insurance_embeddings, self.insurance_encoder, "insurance"))

        if "gender" in input:
            encoders.append(get_encoder_result(
                input, self.gender_embeddings, self.gender_encoder, "gender"))

        patient_representations = torch.cat(
            encoders, dim=-1).squeeze(dim=0)  # (seq, dim*4)
        queries = self.query(patient_representations)  # (seq, dim)

        # graph memory module
        '''I:generate current input'''
        query = queries[-1:]  # (1,dim)

        drug_memory = self.ehr_gcn()

        if input["size"] > 1:
            history_keys = queries[:(queries.size(0)-1)]  # (seq-1, dim)
            history_values = np.zeros((input["size"]-1, self.med_voc_len))
            for idx, med in enumerate(input["medicine"][:-1]):
                history_values[idx, med] = 1

            history_values = torch.FloatTensor(
                history_values).to(self.device)  # (seq-1, size)

        '''O:read from global memory bank and dynamic memory bank'''
        key_weights1 = F.softmax(
            torch.mm(query, drug_memory.t()), dim=-1)  # (1, size)
        fact1 = torch.mm(key_weights1, drug_memory)  # (1, dim)

        if input["size"] > 1:
            visit_weight = F.softmax(
                torch.mm(query, history_keys.t()), dim=-1)  # (1, seq-1)
            weighted_values = visit_weight.mm(history_values)  # (1, size)
            fact2 = torch.mm(weighted_values, drug_memory)  # (1, dim)
        else:
            fact2 = fact1
        '''R:convert O and predict'''
        output = self.output(
            torch.cat([query, fact1, fact2], dim=-1))  # (1, dim)

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
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)

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
