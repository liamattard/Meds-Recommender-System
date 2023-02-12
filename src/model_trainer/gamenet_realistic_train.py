import torch
import wandb
import numpy as np
import torch.nn.functional as F

import src.models.gamenet as gamenet
import src.models.gamenet_all as gamenet_all

import torch.optim as optim
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


def wandb_config(wandb_name, parameters, epoch, lr, batches):
    if wandb_name != None:
        wandb.config = {
            "epochs": epoch,
            "parameters": parameters,
            "batches": batches,
            "learning_rate": lr
        }
        return wandb.config


def train(dataset, dataset_type, wandb_name, features, threshold, num_of_epochs,
          batches, lr):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    med_voc_len = len(dataset.voc[0]["med_voc"].idx2word)

    if features == None:
        diag_voc = dataset.voc[0]['diag_voc']
        pro_voc = dataset.voc[0]['pro_voc']
        voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), med_voc_len)
        model = gamenet.Model(voc_size, dataset.ehr_adj[0], device)
    else:
        model = gamenet_all.Model(
            dataset.ehr_adj[0], device, features, dataset.voc[0])

    data_train = dataset.data[0][0]
    data_eval = dataset.data[0][1]

    model.to(device=device)
    parameters = get_n_params(model)
    print('parameters', parameters)
    config = wandb_config(wandb_name, parameters,
                          num_of_epochs, batches=batches, lr=lr)
    use_wandb(wandb_name, config)

    optimizer = optim.Adam(list(model.parameters()), lr=lr)

    EPOCH = num_of_epochs

    for epoch in range(EPOCH):

        metrics_map = {}
        model.train()
        model.g_diagnosis_age = {}
        print('\nepoch {} --------------------------'.format(epoch + 1))

        num_batches = range(0, len(data_train), batches)
        loss_array = []

        for i in num_batches:

            batch_loss = []

            batch_users = data_train[i:i+batches]

            optimizer.zero_grad()

            for patient in batch_users:

                llprint('\rTraining batch : {} / {} (size:{})'.format(
                        int(i/batches), len(num_batches), batches))

                seq_input = calculate_input(features, patient)
                target_output, _ = model(seq_input)

                loss = loss_function(med_voc_len, patient, target_output, device)

                batch_loss.append(loss)

                metrics_map = calculate_metrics(
                    med_voc_len, patient, target_output, threshold, metrics_map)

            total_loss = torch.mean(torch.stack(batch_loss))
            total_loss.backward(retain_graph=True)
            loss_array.append(total_loss.item())

            optimizer.step()
        results = results_from_metric_map(metrics_map)

        eval_full_epoch(model, data_eval, med_voc_len,
                        wandb_name, (epoch + 1), loss_array, features, threshold, results, device)

        if features == None:
            dir = 'saved_models/gameNet/' + dataset_type.name
        else:
            dir = 'saved_models/gameNet_all/'+"_".join(list(features))

        path = dir + '/' + 'Epoch_{}.model'.format(epoch)

        if not os.path.exists(dir):
            os.makedirs(dir)

        torch.save(model.state_dict(), open(path, 'wb'))


def eval_full_epoch(model, data_eval, med_voc, wandb_name, epoch, loss_array, features, threshold, train_results, device):

    test_results = eval(model, data_eval, med_voc, features, threshold, device)

    metrics_dic = {
        "Epoch": epoch,
        "Testing Jaccard": test_results.jaccard,
        "Testing precision recall AUC": test_results.precision_recall_auc,
        "Testing precision": test_results.precision,
        "Testing recall": test_results.recall,
        "Testing F1": test_results.f1,
        "Testing average medications": test_results.avg_med,
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
        metrics_dic["Training Loss"] = np.mean(loss_array)

    metrics_dic["Testing Loss"] = test_results.loss

    if wandb_name != None:
        wandb.log(metrics_dic)
        wandb.watch(model)


def eval(model, data_eval, med_voc, features, threshold, device):
    model.eval()
    metrics_map = {}
    loss_arr = []

    for step, input in enumerate(data_eval):

        seq_input = calculate_input(features, input)
        target_output = model(seq_input)
        loss_arr.append(loss_function(med_voc, input, target_output, device).item())

        metrics_map = calculate_metrics(
            med_voc, input, target_output, threshold, metrics_map)

        llprint('\rtest step: {} / {}'.format(step, len(data_eval)))

    results = results_from_metric_map(metrics_map)
    results.loss = np.mean(loss_arr)

    llprint('''Jaccard: {:.4},  PRAUC: {:.4}, AVG_PRC: {:.4}, 
               AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_MED: {:.4}\n'''.format(
        results.jaccard, results.precision_recall_auc, results.precision,
            results.recall, results.f1, results.avg_med))

    return results


def test(model_path, model_type, dataset, features):

    data_train = dataset.data[0][0]
    data_eval = dataset.data[0][1]

    voc_size, device = load_data(dataset, model_type)
    model = gamenet.Model(voc_size, dataset.ehr_adj[0], device)

    model.load_state_dict(torch.load(
        open(model_path, 'rb'), map_location=device))
    model.to(device=device)

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


def calculate_input(features, input):
    if features != None:
        input_map = {}
        input_map["size"] = 1 + len(input[-1])

        diag_seq = []
        proc_seq = []
        med_seq = []

        for visit in input[-1]:
            diag_seq.append(visit[0])
            proc_seq.append(visit[1])
            med_seq.append(visit[2])

        diag_seq.append(input[0])
        proc_seq.append(input[1])
        med_seq.append(input[2])

        input_map["medicine"] = med_seq

        if "diagnosis" in features:
            input_map["diagnosis"] = diag_seq

        if "procedures" in features:
            input_map["procedures"] = proc_seq


        if "insurance" in features:
            input_map["insurance"] = [[input[6]]] * input_map["size"]

        input_map["age"] = input[3]
        input_map["g_diagnosis"] = input[7]
        input_map["gender"] = input[5]

        return input_map
    else:
        current_visit = input[:3]
        seq_input = [current_visit]
        if input[-1] != []:
            seq_input = input[-1] + seq_input

        return seq_input


def calculate_metrics(med_voc, input, target_output, threshold, metrics_map):

    y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
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

    adm_results = multi_label_metric(
        np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))

    metrics_map = append_or_create_list(
        "jaccard", adm_results.jaccard, metrics_map)
    metrics_map = append_or_create_list(
        "prauc", adm_results.precision_recall_auc, metrics_map)
    metrics_map = append_or_create_list(
        "avg_p", adm_results.precision, metrics_map)
    metrics_map = append_or_create_list(
        "avg_r", adm_results.recall, metrics_map)
    metrics_map = append_or_create_list("avg_f1", adm_results.f1, metrics_map)
    metrics_map = append_or_create_list(
        "macro_f1", adm_results.macro_f1, metrics_map)
    metrics_map = append_or_create_list(
        "roc_auc", adm_results.roc_auc, metrics_map)
    metrics_map = append_or_create_list("p_1", adm_results.top_1, metrics_map)
    metrics_map = append_or_create_list("p_5", adm_results.top_5, metrics_map)
    metrics_map = append_or_create_list(
        "p_10", adm_results.top_10, metrics_map)
    metrics_map = append_or_create_list(
        "p_20", adm_results.top_20, metrics_map)
    metrics_map = append_or_create_list("med_num",
                                        len(y_pred_label_tmp), metrics_map)

    return metrics_map


def append_or_create_list(key, value, map):
    if key not in map:
        map[key] = []

    map[key].append(value)

    return map


def results_from_metric_map(metrics_map):
    results = Results()
    results.jaccard = np.mean(metrics_map["jaccard"])
    results.precision_recall_auc = np.mean(metrics_map["prauc"])
    results.precision = np.mean(metrics_map["avg_p"])
    results.recall = np.mean(metrics_map["avg_r"])
    results.f1 = np.mean(metrics_map["avg_f1"])
    results.macro_f1 = np.mean(metrics_map["macro_f1"])
    results.roc_auc = np.mean(metrics_map["roc_auc"])
    results.top_1 = np.mean(metrics_map["p_1"])
    results.top_5 = np.mean(metrics_map["p_5"])
    results.top_10 = np.mean(metrics_map["p_10"])
    results.top_20 = np.mean(metrics_map["p_20"])
    results.avg_med = np.mean(metrics_map["med_num"])

    return results


def loss_function(med_voc_len, patient, target_output, device):

    loss_bce_target = np.zeros((1, med_voc_len))
    loss_bce_target[:, patient[2]] = 1

    loss_multi_target = np.full((1, med_voc_len), -1)
    for idx, item in enumerate(patient[2]):
        loss_multi_target[0][idx] = item

    loss_bce = F.binary_cross_entropy_with_logits(target_output,
                                                  torch.FloatTensor(loss_bce_target).to(device))

    loss = torch.nn.MSELoss()
    loss_mse =  loss(torch.sigmoid(target_output),torch.FloatTensor(loss_bce_target).to(device))

    return 0.5 * loss_bce + 0.5 * loss_mse
