import torch
import time
import wandb
import pickle
import numpy as np
import src.models.gamenet as gamenet
import src.models.collaborative as collab
import torch.nn.functional as F

import sys
from collections import defaultdict
from torch.optim import Adam
from src.utils.constants.model_types import Model_Type
from src.utils.tools import multi_label_metric
import os

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def train(dataset, dataset_type, model_type):

    wandb.init(project="GameNet", entity="liam_dratta")
    wandb.config = {
      "learning_rate": 0.0001,
      "epochs": 50,
      "batch_size": 1
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diag_voc = dataset.voc[0]['diag_voc']
    pro_voc = dataset.voc[0]['pro_voc']
    med_voc = dataset.voc[0]['med_voc']

    split_point = int(len(dataset.data[0]) * 2 / 3)
    eval_len = int(len(dataset.data[0][split_point:]) / 2)

    data_train = dataset.data[0][:split_point]
    data_eval = dataset.data[0][split_point+eval_len:]


    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))
    if(model_type == Model_Type.game_net):
        model = gamenet.Model(voc_size, dataset.ehr_adj[0], device)
    else:
        model = collab.Model(voc_size, dataset.ehr_adj[0], device, len(dataset.data[0]), voc_size[2])

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
                loss_bce_target = np.zeros((1, voc_size[2]))
                loss_bce_target[:, adm[2]] = 1

                loss_multi_target = np.full((1, voc_size[2]), -1)
                for idx, item in enumerate(adm[2]):
                    loss_multi_target[0][idx] = item

                if(model_type == Model_Type.game_net):
                    target_output1, _ = model(seq_input)
                else:
                    target_output1 = model(idx,seq_input)

                loss_bce = F.binary_cross_entropy_with_logits(target_output1, torch.FloatTensor(loss_bce_target).to(device))

                loss_multi = F.multilabel_margin_loss(torch.sigmoid(target_output1), torch.LongTensor(loss_multi_target).to(device))
                loss = 0.9 * loss_bce + 0.1 * loss_multi
                loss_array.append(loss.item())

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()


            llprint('\rtraining step: {} / {}'.format(step, len(data_train))) 
            tic2 = time.time() 

        ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(model, data_eval, voc_size, epoch, (len(data_train[0]) + 1))
        print ('training time: {}, test time: {}'.format(time.time() - tic, time.time() - tic2))

        train_ja, train_prauc, train_avg_p, train_avg_r, train_avg_f1, train_avg_med = eval(model, data_train, voc_size, epoch,0)

        history['ja'].append(ja)
        history['avg_p'].append(avg_p)
        history['avg_r'].append(avg_r)
        history['avg_f1'].append(avg_f1)
        history['prauc'].append(prauc)
        history['med'].append(avg_med)

        wandb.log({
            "Epoch": epoch,
            "Loss": np.mean(loss_array),
            "Testing Jaccard": ja,
            "Testing f1": avg_f1,
            "Testing recall": avg_r,
            "Testing accuracy": prauc,
            "Testing average medications": avg_med,
            "Testing precision": avg_p,
            "Training Jaccard": train_ja,
            "Training f1": train_avg_f1,
            "Training recall": train_avg_r,
            "Training accuracy": train_prauc,
            "Training average medications": train_avg_med,
            "Training precision": train_avg_p
            })

        if epoch >= 5:
            print ('Med: {}, Ja: {}, F1: {}, PRAUC: {}'.format(
                np.mean(history['med'][-5:]),
                np.mean(history['ja'][-5:]),
                np.mean(history['avg_f1'][-5:]),
                np.mean(history['prauc'][-5:])
                ))

        

        wandb.watch(model)

        torch.save(model.state_dict(), open(os.path.join('saved_models', model_type.name, \
            dataset_type.name, 'Epoch_{}_JA_{:.4}.model'.format(epoch, ja)), 'wb'))

        if epoch != 0 and best_ja < ja:
            best_epoch = epoch
            best_ja = ja

        print ('best_epoch: {}'.format(best_epoch))

    with open('history.pkl', 'wb') as handle:
        pickle.dump(history, handle)



def eval(model, data_eval, voc_size, epoch, data_train_len):
    model.eval()

    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    med_cnt, visit_cnt = 0, 0

    for step, input in enumerate(data_eval):
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
        
        for adm_idx, adm in enumerate(input):
            target_output = model((data_train_len+adm_idx),input[:adm_idx+1])

            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)

            # prediction prod
            target_output = torch.sigmoid(target_output).detach().cpu().numpy()[0]
            y_pred_prob.append(target_output)

            # predioction med set
            y_pred_tmp = target_output.copy()
            y_pred_tmp[y_pred_tmp>=0.5] = 1
            y_pred_tmp[y_pred_tmp<0.5] = 0
            y_pred.append(y_pred_tmp)

            # prediction label
            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(sorted(y_pred_label_tmp))
            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)


        smm_record.append(y_pred_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))
        
        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)



        llprint('\rtest step: {} / {}'.format(step, len(data_eval)))

    llprint('Jaccard: {:.4},  PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_MED: {:.4}\n'.format(
        np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt
    ))

    return np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt


def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()

