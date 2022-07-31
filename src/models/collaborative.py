import torch
import torch.nn as nn

from src.models.gamenet_modified import Model_Modified

class User_Model(nn.Module):
    def __init__(self, vocab_size, ehr_adj, device, len_users, emb_dim):
        super().__init__()

        self.device = device
        self.user_embeddings = nn.Embedding(len_users,
                                       emb_dim, max_norm=1).to(self.device)

        self.game_net = Model_Modified(vocab_size, ehr_adj, device, emb_dim)

    def forward(self, user_id, adm):
        user_embeddings = self.user_embeddings(torch.LongTensor(user_id).to(self.device))
        game_net_result,_ = self.game_net(adm)

        breakpoint()
        return torch.matmul(user_embeddings, game_net_result.T)

class Medicine_Model(nn.Module):
    def __init__(self, len_med, device, emb_dim):
        super().__init__()

        self.device = device
        self.med_embeddings = nn.Embedding(len_med,
                                       emb_dim, max_norm=1).to(self.device)

    def forward(self):
        return self.med_embeddings

class Model(nn.Module):
    def __init__(self, vocab_size, ehr_adj, device, len_users, len_med, emb_dim=64):
        super().__init__()

        self.device = device
        self.med_model = Medicine_Model(len_med, device, emb_dim)
        self.user_model = User_Model(vocab_size, ehr_adj, device, len_users, emb_dim)


    def forward(self, user_id, adm):
        user_embeddings = self.user_model([user_id], adm)
        med_embeddings = self.med_model()

        breakpoint()
        return torch.matmul(user_embeddings,med_embeddings.T)

