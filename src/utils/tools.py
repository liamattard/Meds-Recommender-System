import numpy as np
import sys
from sklearn.metrics import jaccard_score, roc_auc_score, precision_score, f1_score, average_precision_score

from src.utils.constants.dataset_types import Dataset_Type
from src.utils.classes.results import Results
from src.utils.constants.model_types import Model_Type

def get_list_dimension(myList):

    x = len(myList)
    y = 0

    for i in myList:
        if (len(i) > y):
            y = len(i)
    return x, y


def generate_med_ids(mySet):
    myList = list(mySet)
    myDict = {}

    for i, med in enumerate(myList):
        myDict[med] = i

    return myDict

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def multi_label_metric(y_gt, y_pred, y_prob):

    def jaccard(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            union = set(out_list) | set(target)
            jaccard_score = 0 if union == 0 else len(inter) / len(union)
            score.append(jaccard_score)
        return np.mean(score)

    def average_prc(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            prc_score = 0 if len(out_list) == 0 else len(inter) / len(out_list)
            score.append(prc_score)
        return score

    def average_recall(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            recall_score = 0 if len(target) == 0 else len(inter) / len(target)
            score.append(recall_score)
        return score

    def average_f1(average_prc, average_recall):
        score = []
        for idx in range(len(average_prc)):
            if average_prc[idx] + average_recall[idx] == 0:
                score.append(0)
            else:
                score.append(2*average_prc[idx]*average_recall[idx] / (average_prc[idx] + average_recall[idx]))
        return score

    def macro_f1(y_gt, y_pred):
        all_micro = []
        for b in range(y_gt.shape[0]):
            all_micro.append(f1_score(y_gt[b], y_pred[b], average='macro'))
        return np.mean(all_micro)

    def roc_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(roc_auc_score(y_gt[b], y_prob[b], average='macro'))
        return np.mean(all_micro)

    def precision_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(average_precision_score(y_gt[b], y_prob[b], average='macro'))
        return np.mean(all_micro)

    def precision_at_k(y_gt, y_prob, k=3):
        precision = 0

        # First we get a list that represents the
        # medicine ids from lowest prob to highest prob

        # x = np.argsort(y_prob, axis=-1)

        # We then reverse the order since we want the
        # medicine with highest probability first
        # x = x[:, ::-1]

        # Then we get the Top K of each guess
        # x = x[:, :k]

        sort_index = np.argsort(y_prob, axis=-1)[:, ::-1][:, :k]

        # In the loop we are checking if the items in the
        # sorted list are marked as 1 in the y_gt (ground truth)

        for i in range(len(y_gt)):
            TP = 0
            for j in range(len(sort_index[i])):
                if y_gt[i, sort_index[i, j]] == 1:
                    TP += 1
            precision += TP / len(sort_index[i])

        return precision / len(y_gt)

    # ------------- START ---------------

    # Precision at K
    p_1 = precision_at_k(y_gt, y_prob, k=1)
    p_5 = precision_at_k(y_gt, y_prob, k=5)
    p_10 = precision_at_k(y_gt, y_prob, k=10)
    p_20 = precision_at_k(y_gt, y_prob, k=20)

    # Averagre Precision
    avg_prc = average_prc(y_gt, y_pred)

    # Roc Auc
    try:
        auc = roc_auc(y_gt, y_prob)
    except:
        auc = 0

    # Recall
    avg_recall = average_recall(y_gt, y_pred)

    # macro f1
    macro_f1 = macro_f1(y_gt, y_pred)
    avg_f1 = average_f1(avg_prc, avg_recall)

    # precision-recall Curve
    prauc = precision_auc(y_gt, y_prob)

    # Jaccard
    ja = jaccard(y_gt, y_pred)

    results = Results()
    results.jaccard = ja
    results.precision_recall_auc = prauc
    results.precision = np.mean(avg_prc)
    results.recall = np.mean(avg_recall)
    results.f1 = np.mean(avg_f1)
    results.macro_f1 = macro_f1
    results.roc_auc = auc
    results.top_1 = p_1
    results.top_5 = p_5
    results.top_10 = p_10
    results.top_20 = p_20

    return results
    #return ja, prauc, np.mean(avg_prc), np.mean(avg_recall), np.mean(avg_f1)

def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()


def get_rec_medicine(y_pred):
    return list(np.where(y_pred == 1)[0])

def is1V(dataset_type):
    return (dataset_type == Dataset_Type.full1VATC4 or 
                dataset_type == Dataset_Type.full1VATC3 or
                    dataset_type == Dataset_Type.full1VNDC)

def isM1V(dataset_type):
    return (dataset_type == Dataset_Type.fullM1VATC4 or 
                dataset_type == Dataset_Type.fullM1VATC3 or
                    dataset_type == Dataset_Type.fullM1VNDC or
                        dataset_type == Dataset_Type.multiRealisticNoPro3 or 
                            dataset_type == Dataset_Type.all)

def isATC3(dataset_type):
    return (dataset_type == Dataset_Type.fullATC3 or 
                dataset_type == Dataset_Type.full1VATC3 or
                    dataset_type == Dataset_Type.fullM1VATC3 or 
                        dataset_type == Dataset_Type.full3Age or
                            dataset_type == Dataset_Type.realistic3 or 
                                dataset_type == Dataset_Type.realisticNoPro3 or 
                                    dataset_type == Dataset_Type.multiRealisticNoPro3 or
                                        dataset_type == Dataset_Type.sota_single_only or
                                            dataset_type == Dataset_Type.sota_with_single or 
                                                dataset_type == Dataset_Type.all)

def isATC4(dataset_type):
    return (dataset_type == Dataset_Type.fullATC4 or 
                dataset_type == Dataset_Type.full1VATC4 or
                    dataset_type == Dataset_Type.fullM1VATC4 or
                    dataset_type == Dataset_Type.realistic4)

def isNDC(dataset_type):
    return (dataset_type == Dataset_Type.fullNDC or 
                dataset_type == Dataset_Type.full1VNDC or
                    dataset_type == Dataset_Type.realisticNDC or
                        dataset_type == Dataset_Type.fullM1VNDC)

def isAge(dataset_model_type):
    return (dataset_model_type == Dataset_Type.full3Age or
             dataset_model_type == Dataset_Type.full4Age or
             dataset_model_type == Dataset_Type.realistic3 or
             dataset_model_type == Dataset_Type.realisticNDC or
             dataset_model_type == Dataset_Type.realisticNoPro3 or
             dataset_model_type == Dataset_Type.realistic4 or 
             dataset_model_type == Dataset_Type.multiRealisticNoPro3 or
             dataset_model_type == Model_Type.game_net_age or 
             dataset_model_type == Model_Type.game_net_age_item_coll or
             dataset_model_type == Dataset_Type.all) 

def isItemCollFil(dataset_model_type):
    return (dataset_model_type == Model_Type.game_net_item_coll)

def isCollFil(dataset_model_type):
    return (dataset_model_type == Model_Type.game_net_coll)

def isSota(dataset_type):
    return (dataset_type == Dataset_Type.sota or
             dataset_type == Dataset_Type.old_sota or 
                dataset_type == Dataset_Type.sota_with_single or 
                    dataset_type == Dataset_Type.sota_single_only)

def isByDate(dataset_type):
    return (dataset_type == Dataset_Type.realistic4 or 
                dataset_type == Dataset_Type.realistic3 or
                dataset_type == Dataset_Type.realisticNDC or
                dataset_type == Dataset_Type.realisticNoPro3 or 
                dataset_type == Dataset_Type.multiRealisticNoPro3 or
                dataset_type == Dataset_Type.all)

def isNoPro(dataset_type):
    return (dataset_type == Dataset_Type.realisticNoPro3 or
                dataset_type == Dataset_Type.multiRealisticNoPro3 or
                    dataset_type == Dataset_Type.all)


