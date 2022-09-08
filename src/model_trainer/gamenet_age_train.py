import torch
import time
import wandb
import pickle
import numpy as np
import src.models.gamenet_age as gamenet
import torch.nn.functional as F

from collections import defaultdict
from torch.optim import Adam
from src.utils.tools import multi_label_metric
from src.utils.tools import llprint 
from src.utils.tools import get_n_params
import os

def load_data(dataset):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diag_voc = dataset.voc[0]['diag_voc']
    pro_voc = dataset.voc[0]['pro_voc']
    med_voc = dataset.voc[0]['med_voc']
    age_voc = dataset.voc[0]['age_voc']

    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(age_voc.idx2word), len(med_voc.idx2word) )
    return voc_size, device


def train(dataset, dataset_type, model_type):

    wandb.init(project="GameNet Model", entity="liam_dratta")
    wandb.config = {
     "learning_rate": 0.0001,
     "epochs": 50,
     "batch_size": 1
    }

    split_point = int(len(dataset.data[0]) * 2 / 3)
    eval_len = int(len(dataset.data[0][split_point:]) / 2)
    data_train = dataset.data[0][:split_point]
    data_eval = dataset.data[0][split_point+eval_len:]

    voc_size, device = load_data(dataset)

    model = gamenet.Model(voc_size, dataset.ehr_adj[0], device)

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
            age = input[-1]
            input = input[:-1]
            for idx, adm in enumerate(input):
                seq_input = input[:idx+1]
                loss_bce_target = np.zeros((1, voc_size[-1]))
                loss_bce_target[:, adm[2]] = 1

                loss_multi_target = np.full((1, voc_size[-1]), -1)
                for idx, item in enumerate(adm[2]):
                    loss_multi_target[0][idx] = item

                target_output1, _ = model(seq_input, age)

                loss_bce = F.binary_cross_entropy_with_logits(target_output1, torch.FloatTensor(loss_bce_target).to(device))

                loss_multi = F.multilabel_margin_loss(torch.sigmoid(target_output1), torch.LongTensor(loss_multi_target).to(device))
                loss = 0.9 * loss_bce + 0.1 * loss_multi
                loss_array.append(loss.item())

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()


            llprint('\rtraining step: {} / {}'.format(step, len(data_train))) 
            tic2 = time.time() 

        ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(model, data_eval, voc_size)
        print ('training time: {}, test time: {}'.format(time.time() - tic, time.time() - tic2))

        train_ja, train_prauc, train_avg_p, train_avg_r, train_avg_f1, train_avg_med = eval(model, data_train, voc_size)

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

        dir = 'saved_models/' + model_type.name + '/'+ dataset_type.name 
        path = dir + '/' +  'Epoch_{}_JA_{:.4}.model'.format(epoch, ja)

        if not os.path.exists(dir):
            os.makedirs(dir)

        torch.save(model.state_dict(), open(path, 'wb'))

        if epoch != 0 and best_ja < ja:
            best_epoch = epoch
            best_ja = ja

        print ('best_epoch: {}'.format(best_epoch))

    with open('history.pkl', 'wb') as handle:
        pickle.dump(history, handle)



def eval(model, data_eval, voc_size):
    model.eval()

    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    med_cnt, visit_cnt = 0, 0

    for step, input in enumerate(data_eval):
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []

        age = input[-1]
        input = input[:-1]
        
        for adm_idx, adm in enumerate(input):

            target_output = model(input[:adm_idx+1], age)

            y_gt_tmp = np.zeros(voc_size[-1])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)

            # prediction prod
            target_output = torch.sigmoid(target_output).detach().cpu().numpy()[0]
            y_pred_prob.append(target_output)

            # predioction med set
            y_pred_tmp = target_output.copy()
            y_pred_tmp[y_pred_tmp>=0.7] = 1
            y_pred_tmp[y_pred_tmp<0.7] = 0
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

def test(model_path, dataset):

    split_point = int(len(dataset.data[0]) * 2 / 3)
    eval_len = int(len(dataset.data[0][split_point:]) / 2)
    data_test = dataset.data[0][split_point:split_point + eval_len]

    voc_size, device = load_data(dataset)
    model = gamenet.Model(voc_size, dataset.ehr_adj[0], device)

    model.load_state_dict(torch.load(open(model_path, 'rb'), map_location=device))
    model.to(device=device)

    tic = time.time()
    result = []
    for _ in range(10):
        test_sample = np.random.choice(data_test, round(len(data_test) * 0.8), replace=True)
        eval(model, test_sample, voc_size)
        #result.append([ddi_rate, ja, avg_f1, prauc, avg_med])

    result = np.array(result)
    mean = result.mean(axis=0)
    std = result.std(axis=0)

    outstring = ""
    for m, s in zip(mean, std):
        outstring += "{:.4f} $\pm$ {:.4f} & ".format(m, s)

    print (outstring)
    print ('test time: {}'.format(time.time() - tic))
    return

