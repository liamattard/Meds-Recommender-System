from os import wait
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

        self.g_diagnosis_insurance= {}

        self.diag_embeddings = nn.Embedding(
            len(voc["diag_voc"].idx2word), emb_dim)
        self.diagnosis_encoder = nn.GRU(
            emb_dim, emb_dim * 2, batch_first=True)

        self.proc_embeddings = nn.Embedding(
            len(voc["pro_voc"].idx2word), emb_dim)
        self.procedures_encoder = nn.GRU(
            emb_dim, emb_dim * 2, batch_first=True)

        self.embeddings.append(self.proc_embeddings)
        self.embeddings.append(self.diag_embeddings)

        self.device = device
        self.dropout = nn.Dropout(p=0.4)

        self.query = nn.Sequential(
            nn.Linear(emb_dim * 4, self.med_voc_len),
            nn.LeakyReLU(),
            nn.Linear(self.med_voc_len, self.med_voc_len),
        )

        self.ehr_gcn = GCN(voc_size=self.med_voc_len,
                           emb_dim=self.med_voc_len, adj=ehr_adj, device=device)

        self.knn_output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.med_voc_len *3, self.med_voc_len*2),
            nn.ReLU(),
            nn.Linear(self.med_voc_len *2, self.med_voc_len)
        )

        self.init_weights()

    def forward(self, input):

        def get_KNN(input):
            knn_tensor = torch.zeros(1, self.med_voc_len).to(self.device)

            insurance = input["insurance"]
            age = round(input["age"]/10)*10
            gender = input["gender"]
            g_diagnosis = input["g_diagnosis"]

            if g_diagnosis in self.g_diagnosis_age:
                if insurance in self.g_diagnosis_insurance[g_diagnosis]:
                    if age in self.g_diagnosis_insurance[g_diagnosis][insurance]:
                        if gender in self.g_diagnosis_insurance[g_diagnosis][insurance][age]:
                            medicine = self.g_diagnosis_insurance[g_diagnosis][insurance][age][gender]
                            knn_tensor[:, list(medicine)] = 1

                            self.g_diagnosis_insurance[g_diagnosis][insurance][age][gender].update(
                                input["medicine"][-1])

                            return knn_tensor
                        else:
                            self.g_diagnosis_age[g_diagnosis][insurance][age][gender] = set(input["medicine"][-1])

                    else:
                        self.g_diagnosis_age[g_diagnosis][insurance][age] = {}
                        self.g_diagnosis_age[g_diagnosis][insurance][age][gender] = set(input["medicine"][-1])
                
                else:
                    self.g_diagnosis_insurance[g_diagnosis][insurance] = {}
                    self.g_diagnosis_insurance[g_diagnosis][insurance][age] = {}
                    self.g_diagnosis_insurance[g_diagnosis][insurance][age][gender] = set(input["medicine"][-1])
            else:
                self.g_diagnosis_insurance[g_diagnosis] = {}
                self.g_diagnosis_insurance[g_diagnosis][insurance] = {}
                self.g_diagnosis_insurance[g_diagnosis][insurance][age] = {}
                self.g_diagnosis_insurance[g_diagnosis][insurance][age][gender] = set(input["medicine"][-1])

            return None

        def mean_embedding(embedding):
            return embedding.mean(dim=1).unsqueeze(dim=0)  # (1,1,dim)

        def get_encoder_result(input, layer, encoder):
            values = []

            for adm in input:
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

        encoders.append(get_encoder_result(
            input["diagnosis"], self.diag_embeddings, self.diagnosis_encoder).to(self.device))

        encoders.append(get_encoder_result(
            input["procedures"], self.proc_embeddings, self.procedures_encoder).to(self.device))

        patient_representations = torch.cat(
            encoders, dim=-1).squeeze(dim=0)  # (seq, dim*4)

        queries = self.query(patient_representations).to(self.device)  # (seq, dim)

        # graph memory module
        '''I:generate current input'''
        query = queries[-1:]  # (1,dim)

        drug_memory = self.ehr_gcn()

        '''O:read from global memory bank and dynamic memory bank'''
        key_weights1 = F.softmax(
            torch.mm(query, drug_memory.t()), dim=-1)  # (1, size)
        fact1 = torch.mm(key_weights1, drug_memory)  # (1, dim)

        knn_output = get_KNN(input)

        if knn_output is None:
            knn_output = query.to(self.device)

        temp_output = torch.cat([knn_output, query, fact1])
        final_knn_output = self.knn_output(temp_output.view(-1))

        if self.training:
            return final_knn_output.unsqueeze(dim=0), None
        else:
            return final_knn_output.unsqueeze(dim=0)

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)

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
