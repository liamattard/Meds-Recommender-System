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
        #self.linear = nn.Linear(emb_dim * 2, vocab_size[2])

    def forward(self, user_id, adm, is_eval):
        user_embeddings = self.user_embeddings(torch.LongTensor(user_id).to(self.device))

        if is_eval:
            game_net_result = self.game_net(adm)
        else:
            game_net_result,_ = self.game_net(adm)

        return torch.matmul(game_net_result.T, user_embeddings)
        #output = torch.cat((user_embeddings, game_net_result),1)
        #return self.linear(output)

class Medicine_Model(nn.Module):
    def __init__(self, len_med, device, emb_dim):
        super().__init__()

        self.device = device
        self.med_embeddings = nn.Embedding(len_med,
                                       emb_dim, max_norm=1).to(self.device)
        #self.linear = nn.Linear(emb_dim,1)

    def forward(self):
        output =  self.med_embeddings
        #return self.linear(output.weight)
        return output

class Model(nn.Module):
    def __init__(self, vocab_size, ehr_adj, device, len_users, len_med, emb_dim=64):
        super().__init__()

        self.device = device
        self.med_model = Medicine_Model(len_med, device, emb_dim)
        self.user_model = User_Model(vocab_size, ehr_adj, device, len_users, emb_dim)
        self.output = nn.Sequential(
                nn.ReLU(),
                nn.Linear(64,1))


    def forward(self, user_id, adm, is_eval=False):
        user_embeddings = self.user_model([user_id], adm, is_eval)
        med_embeddings = self.med_model()

        mult = torch.matmul(user_embeddings,med_embeddings.weight.T)
        reduced = self.output(mult.T)
        return torch.swapaxes(reduced,1,0)




