import torch
import time
import wandb
import numpy as np
import src.models.gamenet as gamenet
import src.models.gamenet_age as gamenet_age
import torch.nn.functional as F

from sklearn.metrics.pairwise import cosine_similarity
from torch.optim import Adam
from src.utils.classes.results import Results
from src.utils.tools import multi_label_metric
from src.utils.tools import llprint
from src.utils.tools import get_n_params 
from src.utils.tools import get_rec_medicine 
import os

torch.manual_seed(1203)
np.random.seed(1203)

def use_wandb(wandb_name,config):
    if wandb_name != None:
        wandb.init(project=wandb_name, entity="liam_dratta", config=config)

def wandb_config(wandb_name, parameters):
    if wandb_name != None:
        wandb.config = {
          "learning_rate": 0.0002,
          "epochs": 50,
          "parameters":parameters
        }
        return wandb.config

def load_data(dataset, with_age):
    diag_voc = dataset.voc[0]['diag_voc']
    pro_voc = dataset.voc[0]['pro_voc']
    med_voc = dataset.voc[0]['med_voc']

    if with_age:
        age_voc = dataset.voc[0]['age_voc']
        voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(age_voc.idx2word), len(med_voc.idx2word))
    else:
        voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return voc_size, device

def train(dataset, dataset_type, model_type, wandb_name, with_age=False):

    voc_size, device = load_data(dataset, with_age)

    data_train = dataset.data[0][0]
    data_eval = dataset.data[0][1]

    if with_age:
        model = gamenet_age.Model(voc_size, dataset.ehr_adj[0], device)
    else:
        model = gamenet.Model(voc_size, dataset.ehr_adj[0], device)

    model.to(device=device)
    parameters = get_n_params(model)
    print('parameters', parameters)
    config = wandb_config(wandb_name, parameters)
    use_wandb(wandb_name, config)

    optimizer = Adam(list(model.parameters()), lr=0.0002)

    tic2 = 0

    EPOCH = 50
    for epoch in range(EPOCH):
        tic = time.time()
        print ('\nepoch {} --------------------------'.format(epoch + 1))
        model.train()
        loss_array = []
        age = None

        for step, input in enumerate(data_train):

            if with_age:
                age = input[-1]
                input = input[:-1]

            for idx, adm in enumerate(input):

                seq_input = input[:idx+1]

                if with_age and age != None:
                    target_output1, _ = model(seq_input, age)
                else:
                    target_output1, _ = model(seq_input)

                loss_bce_target = np.zeros((1, voc_size[-1]))
                loss_bce_target[:, adm[2]] = 1

                loss_multi_target = np.full((1, voc_size[-1]), -1)
                for idx, item in enumerate(adm[2]):
                    loss_multi_target[0][idx] = item


                loss_bce = F.binary_cross_entropy_with_logits(target_output1,
                        torch.FloatTensor(loss_bce_target).to(device))

                loss_multi = F.multilabel_margin_loss(
                        torch.sigmoid(target_output1),
                        torch.LongTensor(loss_multi_target).to(device))

                loss = 0.9 * loss_bce + 0.1 * loss_multi
                loss_array.append(loss.item())

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()


            llprint('\rtraining step: {} / {}'.format(step, len(data_train))) 
            tic2 = time.time() 

        #ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(model, data_eval, voc_size)
        test_results = eval(model, data_eval, voc_size, with_age)

        print ('training time: {}, test time: {}'.format(time.time() - tic,
            time.time() - tic2))

        train_results =  eval(model, data_train, voc_size, with_age)

        if wandb_name != None:
            wandb.log({
                "Epoch": epoch,
                "Loss": np.mean(loss_array),
                "Testing Jaccard": test_results.jaccard,
                "Testing precision recall AUC": test_results.precision_recall_auc,
                "Testing precision": test_results.precision,
                "Testing recall": test_results.recall,
                "Testing F1": test_results.f1,
                "Testing average medications": test_results.avg_med,
                "Testing coverage": test_results.coverage,
                "Testing personalisation": test_results.personalisation,
                "Testing macro F1": test_results.macro_f1,
                "Testing roc auc": test_results.roc_auc,
                "Testing precision at 1": test_results.top_1,
                "Testing precision at 5": test_results.top_5,
                "Testing precision at 10": test_results.top_10,
                "Testing precision at 20": test_results.top_20,
                "Training Jaccard": train_results.jaccard,
                "Training precision recall AUC": train_results.precision_recall_auc,
                "Training precision": train_results.precision,
                "Training recall": train_results.recall,
                "Training F1": train_results.f1,
                "Training average medications": train_results.avg_med,
                "Training coverage": train_results.coverage,
                "Training personalisation": train_results.personalisation,
                "Training macro F1": train_results.macro_f1,
                "Training roc auc": train_results.roc_auc,
                "Training precision at 1": train_results.top_1,
                "Training precision at 5": train_results.top_5,
                "Training precision at 10": train_results.top_10,
                "Training precision at 20": train_results.top_20,
                })

            wandb.watch(model)

        dir = 'saved_models/' + model_type.name + '/'+ dataset_type.name 
        path = dir + '/' +  'Epoch_{}.model'.format(epoch)

        if not os.path.exists(dir):
            os.makedirs(dir)

        torch.save(model.state_dict(), open(path, 'wb'))

def eval(model, data_eval, voc_size, with_age):
    model.eval()

    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    macro_f1, roc_auc, p_1, p_5, p_10, p_20 = [[] for _ in range(6)]
    med_cnt, visit_cnt = 0, 0
    age = None

    covered_medicine = set()
    patient_medicine_arr = np.zeros((len(data_eval), voc_size[-1]))

    for step, input in enumerate(data_eval):

        if with_age:
            age = input[-1]
            input = input[:-1]

        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
        
        for adm_idx, adm in enumerate(input):

            if with_age and age != None:
                target_output = model(input[:adm_idx+1], age)
            else:
                target_output = model(input[:adm_idx+1])

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
            
            for i in y_pred_label_tmp:
                patient_medicine_arr[step, i] = 1

            covered_medicine.update(get_rec_medicine(y_pred_tmp))


        smm_record.append(y_pred_label)
        adm_results = multi_label_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))

        ja.append(adm_results.jaccard)
        prauc.append(adm_results.precision_recall_auc)
        avg_p.append(adm_results.precision)
        avg_r.append(adm_results.recall)
        avg_f1.append(adm_results.f1)
        macro_f1.append(adm_results.macro_f1)
        roc_auc.append(adm_results.roc_auc)
        p_1.append(adm_results.top_1)
        p_5.append(adm_results.top_5)
        p_10.append(adm_results.top_10)
        p_20.append(adm_results.top_20)

        llprint('\rtest step: {} / {}'.format(step, len(data_eval)))

    llprint('Jaccard: {:.4},  PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_MED: {:.4}\n'.format(
        np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt
    ))

    # calculating the coverage and the personalisation metrics
    x = cosine_similarity(patient_medicine_arr)
    iu1 = np.triu_indices(1016, k=1)
    personalisation = 1 - np.mean(x[iu1])
    coverage = (len(covered_medicine)/voc_size[-1]) * 100

    results = Results()
    results.jaccard = np.mean(ja)
    results.precision_recall_auc = np.mean(prauc)
    results.precision = np.mean(avg_p)
    results.recall = np.mean(avg_r)
    results.f1 = np.mean(avg_f1)
    results.avg_med = med_cnt / visit_cnt
    results.coverage = coverage
    results.personalisation = personalisation
    results.macro_f1 = np.mean(macro_f1)
    results.roc_auc = np.mean(roc_auc)
    results.top_1 = np.mean(p_1)
    results.top_5 = np.mean(p_5)
    results.top_10 = np.mean(p_10)
    results.top_20 = np.mean(p_20)
    
    return results


def test(model_path, dataset):

    data_test = dataset.data[0][1]

    voc_size, device = load_data(dataset)
    model = gamenet.Model(voc_size, dataset.ehr_adj[0], device)

    model.load_state_dict(torch.load(open(model_path, 'rb'), map_location=device))
    model.to(device=device)

    tic = time.time()
    result = []
    for _ in range(10):
        test_sample = np.random.choice(data_test, round(len(data_test) * 0.8), replace=True)
        eval(model, test_sample, voc_size, False)
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
