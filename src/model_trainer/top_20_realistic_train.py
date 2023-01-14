import torch
import wandb
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from src.utils.classes.results import Results
from src.utils.tools import multi_label_metric
from src.utils.tools import llprint
from src.utils.tools import get_rec_medicine 
import src.utils.tools as tools
from src.utils.constants.dataset_types import Dataset_Type 


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

def load_data(dataset, model_type):
    diag_voc = dataset.voc[0]['diag_voc']
    pro_voc = dataset.voc[0]['pro_voc']
    med_voc = dataset.voc[0]['med_voc']

    if tools.isfinal(model_type):
        age_voc = dataset.voc[0]['age_voc']
        hr_voc = dataset.voc[0]['heartrate_voc']
        voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word),
                len(age_voc.idx2word), len(hr_voc.idx2word),
                len(med_voc.idx2word))
    elif tools.isAge(model_type):
        age_voc = dataset.voc[0]['age_voc']
        voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(age_voc.idx2word), len(med_voc.idx2word))
    elif tools.isCollFil(model_type):
        age_voc = dataset.voc[0]['age_voc']
        patient_voc = dataset.voc[0]['patient_voc']

        voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(patient_voc.idx2word), len(age_voc.idx2word), len(med_voc.idx2word))
    else:
        voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    return voc_size 

def train(dataset, dataset_type, model_type, wandb_name):

    voc_size  = load_data(dataset, model_type)

    data_train = dataset.data[0][0]
    data_eval = dataset.data[0][1]


    config = wandb_config(wandb_name, 0)
    use_wandb(wandb_name, config)

    # Evaluation before the model is trained
    eval_full_epoch(data_eval, voc_size, wandb_name, 0, dataset_type)

    EPOCH = 50
    for epoch in range(EPOCH):
        eval_full_epoch(data_eval, voc_size, wandb_name, (epoch + 1), dataset_type)

def eval_full_epoch(data_eval, voc_size, wandb_name, epoch, dataset_type):


    test_results = eval(data_eval, voc_size, dataset_type)

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
            }

    if wandb_name != None:
        wandb.log(metrics_dic)

def eval(data_eval, voc_size, dataset_type):

    top_k_results = None 
    top_k = None 

    if dataset_type == Dataset_Type.realistic3:
        top_k_results = np.empty((1,189))
        top_k_results.fill(-1.5)
        top_k = [13, 0, 19, 5, 2, 23, 8, 11, 41, 24, 35, 36, 29, 15, 22, 12,
                3, 16, 14, 25]
    elif dataset_type == Dataset_Type.realisticNDC:
        top_k_results = np.empty((1,4171))
        top_k_results.fill(-1.5)
        top_k = [0, 33, 219, 70, 50, 101, 18, 83, 2, 130, 120, 9, 26, 108, 4,
                60, 38, 13, 103, 30, 216, 43, 229, 61, 59, 140, 66, 143, 253,
                372, 129, 237, 198, 149, 133, 113, 93, 102, 32, 383, 167,
                424, 164, 36, 215, 307, 127, 225, 172, 46]
    else:
        top_k_results = np.empty((1,465))
        top_k_results.fill(-1.5)
        top_k = [5, 23, 34, 4, 6, 16, 3, 13, 32, 7, 11, 41, 44, 37, 60, 30,
                78, 12, 39, 94, 19, 33, 40, 18, 10, 36, 26, 15, 58, 67]



    for i in top_k:
        top_k_results[0][i] = 1.5


    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    macro_f1, roc_auc, p_1, p_5, p_10, p_20 = [[] for _ in range(6)]
    med_cnt, visit_cnt = 0, 0

    covered_medicine = set()
    patient_medicine_arr = np.zeros((len(data_eval), voc_size[-1]))

    for step, input in enumerate(data_eval):

        current_visit = input[:3]
        seq_input =  [current_visit]
        if input[-1] != []:
            seq_input = input[-1] + seq_input

        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
        
        target_output = torch.tensor(top_k_results)

        y_gt_tmp = np.zeros(voc_size[-1])
        y_gt_tmp[current_visit[2]] = 1
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


