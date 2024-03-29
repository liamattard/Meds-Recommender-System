import numpy as np
import src.utils.tools as tools

def get_visit_arr(visit, visit_diagnoses_map, visit_procedures_map, visit_medicine_map, voc, dataset_type):
        
        current_visit = []
        new_row = []
        containsNone = False
        
        if visit in visit_diagnoses_map:
            diagnoses = list(visit_diagnoses_map[visit])
            current_visit.append(diagnoses)
        else:
            containsNone = True

        if visit in visit_procedures_map:
            procedures = list(visit_procedures_map[visit])
            current_visit.append(procedures)
        else:
            if tools.isNoPro(dataset_type):
                current_visit.append(['empty'])
            else:
                containsNone = True

        if visit in visit_medicine_map:
            medicine = list(visit_medicine_map[visit])
            current_visit.append(medicine)
        else:
            containsNone = True

        if not containsNone:
            new_row, voc = get_final_row(current_visit, voc)
        return new_row, voc


def get_final_row(current_visit, voc):

    new_diag_arr, voc["diag_voc"] = convert_to_id(current_visit[0], voc["diag_voc"])
    new_pro_arr, voc["pro_voc"] = convert_to_id(current_visit[1], voc["pro_voc"])
    new_med_arr, voc["med_voc"] = convert_to_id(current_visit[2], voc["med_voc"])

    return [new_diag_arr, new_pro_arr, new_med_arr], voc

def convert_to_id(row, voc):
    new_item_arr = []
    for item in row:
        voc = append(voc, item)
        new_item_arr.append(voc.word2idx[item])
    return new_item_arr, voc

def append(voc, word):
    if word not in voc.word2idx:
        voc.word2idx[word] = len(voc.word2idx)
        voc.idx2word[voc.word2idx[word]] = word
    return voc



def build_ehr_adj(voc, data, isAge, isRealistic):
    ehr_adj = np.zeros((len(voc["med_voc"].idx2word), len(voc["med_voc"].idx2word)))

    for patient in data:
        x = patient

        if isAge:
            x = patient[:-1]
        elif isRealistic:
            x = [patient[:-2]]

        for visit in x:
            for i,medOne in enumerate(visit[2]):
                for j,medTwo in enumerate(visit[2]):
                    if j<=i:
                        continue
                    ehr_adj[medOne, medTwo] = 1
                    ehr_adj[medTwo, medOne] = 1

    return ehr_adj
