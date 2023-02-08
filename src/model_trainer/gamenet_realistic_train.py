import torch
import time
import wandb
import numpy as np
import torch.nn.functional as F

import src.models.gamenet as gamenet
import src.models.gamenet_all as gamenet_all

from torch.optim import Adam
from src.utils.classes.results import Results
from src.utils.tools import multi_label_metric
from src.utils.tools import llprint
from src.utils.tools import get_n_params
from src.utils.tools import get_rec_medicine
import os

torch.manual_seed(1203)
np.random.seed(1203)


def use_wandb(wandb_name, config):
    if wandb_name != None:
        wandb.init(project=wandb_name, entity="liam_dratta", config=config)

def wandb_config(wandb_name, parameters, epoch):
    if wandb_name != None:
        wandb.config = {
            "learning_rate": 0.0002,
            "epochs": epoch,
            "parameters": parameters
        }
        return wandb.config


def train(dataset=None, dataset_type=None, wandb_name=None, features=None, threshold=None, num_of_epochs=None, lr=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    med_voc_len = len(dataset.voc[0]["med_voc"].idx2word)

    model = gamenet_all.Model(
        dataset.ehr_adj[0], device, features, med_voc_len)

    data_train = dataset.data[0][0]
    data_eval = dataset.data[0][1]

    model.to(device=device)
    parameters = get_n_params(model)

    print('parameters', parameters)

    config = wandb_config(wandb_name, parameters, num_of_epochs)
    use_wandb(wandb_name, config)

    optimizer = Adam(list(model.parameters()), lr=0.0002)

    tic2 = 0

    longest_feature = calculate_longest_feature(features, dataset.voc[0])

    EPOCH = num_of_epochs
    for epoch in range(EPOCH):
        tic = time.time()
        print('\nepoch {} --------------------------'.format(epoch + 1))
        model.train()
        loss_array = []
        batch_size = 32

        num_batches = range(0, len(data_train), batch_size)
        for i in num_batches:

            batch_users = data_train[i:i+batch_size]

            for patient in batch_users:

                llprint('\rTraining batch (size:{} ): {} / {}'.format(batch_size,
                        int(i/32), len(num_batches)))

                seq_input, attention_mask = calculate_input(
                    features, patient, dataset.voc[0], longest_feature)
                target_output1, _ = model(seq_input, attention_mask) 

                loss_bce_target = np.zeros((1, med_voc))
                loss_bce_target[:, patient[2]] = 1

                loss_multi_target = np.full((1, med_voc), -1)
                for idx, item in enumerate(patient[2]):
                    loss_multi_target[0][idx] = item

                loss_bce = F.binary_cross_entropy_with_logits(target_output1,
                                                              torch.FloatTensor(loss_bce_target).to(device))

                loss_multi = F.multilabel_margin_loss(
                    torch.sigmoid(target_output1),
                    torch.LongTensor(loss_multi_target).to(device))

                loss = 0.9 * loss_bce + 0.1 * loss_multi
                loss_array.append(loss.item())

                #  Back Propogation
                optimizer.zero_grad()
                loss.backward(retain_graph=True)

            optimizer.step()

            # llprint('\rtraining step: {} / {}'.format(step, len(data_train)))
            tic2 = time.time()

        print('training time: {}, test time: {}'.format(time.time() - tic,
                                                        time.time() - tic2))

        eval_full_epoch(model, data_train, data_eval, med_voc,
                        wandb_name, (epoch + 1), loss_array, features, threshold)

        if features == None:
            dir = 'saved_models/gameNet/' + dataset_type.name
        else:
            dir = 'saved_models/gameNet/'+"_".join(list(features))

        path = dir + '/' + 'Epoch_{}.model'.format(epoch)

        if not os.path.exists(dir):
            os.makedirs(dir)

        torch.save(model.state_dict(), open(path, 'wb'))


def eval_full_epoch(model, data_train, data_eval, med_voc, wandb_name, epoch, loss_array, features, threshold):

    test_results = eval(model, data_eval, med_voc, features, threshold)

    train_results = eval(model, data_train, med_voc, features, threshold)

    metrics_dic = {
        "Epoch": epoch,
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
    }

    if loss_array != []:
        metrics_dic["Loss"] = np.mean(loss_array)

    if wandb_name != None:
        wandb.log(metrics_dic)
        wandb.watch(model)


def eval(model, data_eval, med_voc, features, threshold):
    model.eval()

    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    macro_f1, roc_auc, p_1, p_5, p_10, p_20 = [[] for _ in range(6)]
    med_cnt, visit_cnt = 0, 0

    covered_medicine = set()

    for step, input in enumerate(data_eval):

        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []

        seq_input = calculate_input(features, input)
        target_output = model(seq_input)

        y_gt_tmp = np.zeros(med_voc)
        y_gt_tmp[input[2]] = 1
        y_gt.append(y_gt_tmp)

        # prediction prod
        target_output = torch.sigmoid(target_output).detach().cpu().numpy()[0]
        y_pred_prob.append(target_output)

        # predioction med set
        y_pred_tmp = target_output.copy()
        y_pred_tmp[y_pred_tmp >= threshold] = 1
        y_pred_tmp[y_pred_tmp < threshold] = 0
        y_pred.append(y_pred_tmp)

        # prediction label
        y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
        y_pred_label.append(sorted(y_pred_label_tmp))
        visit_cnt += 1
        med_cnt += len(y_pred_label_tmp)

        covered_medicine.update(get_rec_medicine(y_pred_tmp))

        smm_record.append(y_pred_label)
        adm_results = multi_label_metric(
            np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))

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
        np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(
            avg_r), np.mean(avg_f1), med_cnt / visit_cnt
    ))

    # calculating the coverage and the personalisation metrics
    coverage = (len(covered_medicine)/med_voc) * 100

    results = Results()
    results.jaccard = np.mean(ja)
    results.precision_recall_auc = np.mean(prauc)
    results.precision = np.mean(avg_p)
    results.recall = np.mean(avg_r)
    results.f1 = np.mean(avg_f1)
    results.avg_med = med_cnt / visit_cnt
    results.coverage = coverage
    results.macro_f1 = np.mean(macro_f1)
    results.roc_auc = np.mean(roc_auc)
    results.top_1 = np.mean(p_1)
    results.top_5 = np.mean(p_5)
    results.top_10 = np.mean(p_10)
    results.top_20 = np.mean(p_20)

    return results


def test(model_path, model_type, dataset, features):

    data_train = dataset.data[0][0]
    data_eval = dataset.data[0][1]

    voc_size, device = load_data(dataset, model_type)
    model = gamenet.Model(voc_size, dataset.ehr_adj[0], device)

    model.load_state_dict(torch.load(
        open(model_path, 'rb'), map_location=device))
    model.to(device=device)

    tic = time.time()
    result = []

    eval_full_epoch(model, data_train, data_eval,  model_type,
                    None, 0, [], features, torch.threshold)

    result = np.array(result)
    mean = result.mean(axis=0)
    std = result.std(axis=0)

    outstring = ""
    for m, s in zip(mean, std):
        outstring += "{:.4f} $\pm$ {:.4f} & ".format(m, s)

    print(outstring)
    print('test time: {}'.format(time.time() - tic))


def calculate_input(features, input, voc, padding_ammount):

    def generate_mask_diag_pro(feature, seq):
        vector = [torch.zeros(1, padding_ammount, dtype=torch.int32),
                  torch.zeros(1, padding_ammount, dtype=torch.int32),
                  torch.zeros(1, padding_ammount, dtype=torch.int32)]

        attention_layer = [torch.ones(vector[0].shape, dtype=torch.int32),
                           torch.ones(vector[0].shape, dtype=torch.int32),
                           torch.ones(vector[0].shape, dtype=torch.int32)]

        for i, visit in enumerate(seq):
            vector[i][0][visit] = 1
            attention_layer[i][0][range(
                len(voc[feature].idx2word), padding_ammount)] = 0

        for i in range(len(seq), 3):
            attention_layer[i][0][:] = 0

        return vector, attention_layer

    def generate_mask_res(feature, item):
        vector = torch.zeros(1, padding_ammount, dtype=torch.int32)
        attention_layer = torch.ones(vector.shape, dtype=torch.int32)

        vector[0][item] = 1
        attention_layer[0][range(
            len(voc[feature].idx2word), padding_ammount)] = 0

        return vector, attention_layer

    med_seq = []
    for visit in input[-1]:
        med_seq.append(visit[2])
    med_seq.append(input[2])

    diag_seq = []
    proc_seq = []

    for visit in input[-1][:1]:
        diag_seq.append(visit[0])
        proc_seq.append(visit[1])

    diag_seq.append(input[0])
    proc_seq.append(input[1])

    input_mask = []
    attention_mask = []

    if "diagnosis" in features:
        i, a = generate_mask_diag_pro("diag_voc", diag_seq)

        input_mask.extend(i)
        attention_mask.extend(a)

    if "procedures" in features:
        i, a = generate_mask_diag_pro("pro_voc", proc_seq)

        input_mask.extend(i)
        attention_mask.extend(a)

    if "age" in features:
        i, a = generate_mask_res("age_voc", input[3])

        input_mask.append(i)
        attention_mask.append(a)

    if "gender" in features:
        i, a = generate_mask_res("gender_voc", input[5])

        input_mask.append(i)
        attention_mask.append(a)

    if "insurance" in features:
        i, a = generate_mask_res("insurance_voc", input[6])

        input_mask.append(i)
        attention_mask.append(a)

    input_mask_tensor = torch.cat(input_mask)
    attention_mask_tensor = torch.cat(attention_mask)
    return input_mask_tensor, attention_mask_tensor


def calculate_longest_feature(features, voc):

    longest_feature = 0

    if "diagnosis" in features and len(voc["diag_voc"].idx2word) > longest_feature:
        longest_feature = len(voc["diag_voc"].idx2word)

    if "procedures" in features and len(voc["pro_voc"].idx2word) > longest_feature:
        longest_feature = len(voc["pro_voc"].idx2word)

    if "age" in features and len(voc["age_voc"].idx2word) > longest_feature:
        longest_feature = len(voc["age_voc"].idx2word)

    if "insurance" in features and len(voc["insurance_voc"].idx2word) > longest_feature:
        longest_feature = len(voc["insurance_voc"].idx2word)

    if "gender" in features and len(voc["gender_voc"].idx2word) > longest_feature:
        longest_feature = len(voc["gender_voc"].idx2word)

    return longest_feature
