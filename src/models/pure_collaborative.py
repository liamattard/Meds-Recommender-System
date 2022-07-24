import torch
import torch.nn as nn
import torch.nn.functional as F

from torchnlp.encoders import LabelEncoder

class Model(nn.Module):
    def __init__(self, dataset, device):
        super().__init__()


        embedding_size = 32
        self.device = device

        self.user_encoder = LabelEncoder(dataset.unique_user_ids,
                                                   reserved_labels=['unknown'],
                                                   unknown_index=0)

        self.user_model = nn.Embedding(len(dataset.unique_user_ids) + 1,
                                       embedding_size, max_norm=1)

        self.med_encoder = LabelEncoder(dataset.unique_medicine_names,
                                                   reserved_labels=['unknown'],
                                                   unknown_index=0)

        self.med_model = nn.Embedding(len(dataset.unique_medicine_names) + 1,
                                       embedding_size, max_norm=1)


    def forward(self, user_list, medicine_list):
        user_ids = list(map(lambda user: self.user_encoder.encode(user), user_list))
        medicine_ids = list(map(lambda medicine: self.med_encoder.encode(medicine), medicine_list))
        user_embeddings = self.user_model(torch.LongTensor(user_ids).to(self.device))
        med_embeddings = self.med_model(torch.LongTensor(medicine_ids).to(self.device))

        return torch.matmul(user_embeddings,med_embeddings.T)

