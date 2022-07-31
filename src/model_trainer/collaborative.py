import torch
import time
from src.models.collaborative import Model
import src.models.gamenet_modified as gamenet

from collections import defaultdict
from torch.optim import Adam

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def train(dataset, dataset_type):

    '''
    wandb.init(project="GameNet", entity="liam_dratta")
    wandb.config = {
      "learning_rate": 0.0001,
      "epochs": 50,
      "batch_size": 1
    }
    '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diag_voc = dataset.voc[0]['diag_voc']
    pro_voc = dataset.voc[0]['pro_voc']
    med_voc = dataset.voc[0]['med_voc']

    split_point = int(len(dataset.data[0]) * 2 / 3)
    eval_len = int(len(dataset.data[0][split_point:]) / 2)

    data_train = dataset.data[0][:split_point]
    data_eval = dataset.data[0][split_point+eval_len:]


    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    model = Model(voc_size, dataset.ehr_adj[0], device, len(data_train), voc_size[2], emb_dim=64)
    model.to(device=device)
    print('parameters', get_n_params(model))

    optimizer = Adam(list(model.parameters()), lr=1e-4)

    history = defaultdict(list)
    best_epoch, best_ja = 0, 0

    EPOCH = 50
    for epoch in range(EPOCH):
        tic = time.time()
        print ('\nepoch {} --------------------------'.format(epoch + 1))
        model.train()
        loss_array = []
        for step, input in enumerate(data_train):
            for idx, adm in enumerate(input):
                seq_input = input[:idx+1]


                breakpoint()
                target_output1, _ = model(0,seq_input)
                breakpoint()
                

