from src.utils.classes.Dataset import Dataset

from src.utils.constants.dataset_types import Dataset_Type 
from src.utils.classes.voc import Voc
import src.utils.query_handler as query_handler
import src.utils.file_utils as file_utils 
import src.utils.tools as tools 

import os
import numpy as np
import logging
import configparser
import logging
import pickle
import pandas as pd

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Data loader")

config = configparser.ConfigParser()
config.sections()
config.read("properties.ini")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Data loader")


def build_pure_col(db):
    user_med_list, med_set, past_medicine_array = query_handler.load(db)
    user_medicine_pd = pd.DataFrame(
                            user_med_list,
                            columns=['subject_id', 'drug', 'drug_id',
                                     'has_past_medicine'])

    with open(config["DATASET"]["medicine_set"], 'wb') as handle:
        pickle.dump(med_set, handle)

    with open(config["DATASET"]["user_med_pd"], 'wb') as handle:
        pickle.dump(user_medicine_pd, handle)

    with open(config["DATASET"]["past_med_arr"], 'wb') as handle:
        pickle.dump(past_medicine_array, handle)

def build_dataset(db, dataset_type):

    visit_diagnoses_map = query_handler.load_visit_diagnoses(db)
    visit_procedures_map = query_handler.load_visit_procedures(db)
    _, user_visit_map = query_handler.load_user_visit_map(db)

    visit_medicine_map = query_handler.load_visit_medicine(db, dataset_type)

    user_age_map = query_handler.load_user_age_map(db)

    data = []

    voc = {'diag_voc': Voc(), 'pro_voc': Voc(), 'med_voc': Voc()}

    isAge = tools.isAge(dataset_type)
    is1V = tools.is1V(dataset_type)
    isM1V = tools.isM1V(dataset_type)

    if isAge:
        voc['age_voc']=  Voc()

    number_of_bad_data = 0

    total = 0
    contains_none_diag= 0
    contains_none_pro= 0
    contains_none_med= 0
    number_of_clashes = 0

    for _, patient in enumerate(user_visit_map):
        patient_arr = []

        if(is1V):
            if len(user_visit_map[patient]) > 1:
                continue
        elif(isM1V):
            if len(user_visit_map[patient]) < 2:
                continue


        for visit in user_visit_map[patient]:
            current_visit = []
            containsNone = False
            clash = False

            if visit in visit_diagnoses_map:
                diagnoses = list(visit_diagnoses_map[visit])
                current_visit.append(diagnoses)
            else:
                contains_none_diag = contains_none_diag + 1
                containsNone = True

            if visit in visit_procedures_map:
                procedures = list(visit_procedures_map[visit])
                current_visit.append(procedures)
            else:
                contains_none_pro = contains_none_pro + 1
                clash = True
                containsNone = True

            if visit in visit_medicine_map:
                medicine = list(visit_medicine_map[visit])
                current_visit.append(medicine)
            else:
                if clash == True:
                    number_of_clashes = number_of_clashes + 1
                contains_none_med= contains_none_med + 1
                containsNone = True


            if not containsNone:
                total = total + len(current_visit[2])
                new_row, voc = get_final_row(current_visit, voc)
                patient_arr.append(new_row)


        if(isM1V):
            if len(patient_arr) > 1:
                data.append(patient_arr)
        elif(is1V):
            if len(patient_arr) == 1:
                data.append(patient_arr)
        else:
            if len(patient_arr) > 0:
                if isAge:
                    user_age = user_age_map[patient]
                    voc["age_voc"] = append(voc["age_voc"], user_age)
                    patient_arr.append(voc["age_voc"].word2idx[user_age])

                    data.append(patient_arr)

        # Check for how many patients with 1 visit were added
        if(is1V):
            if len(patient_arr) > 1:
                number_of_bad_data = number_of_bad_data + 1
        elif(isM1V):
            if len(patient_arr) == 1:
                number_of_bad_data = number_of_bad_data + 1

    print("contains none diag", contains_none_diag)
    print("contains none pro", contains_none_pro)
    print("contains none med", contains_none_med)
    print("clashes", number_of_clashes)
    print("total", total)

    # BUILDING ehr_adj file
    ehr_adj = np.zeros((len(voc["med_voc"].idx2word), len(voc["med_voc"].idx2word)))

    for patient in data:
        x = patient
        if isAge:
            x = patient[:-1]
        for visit in x:
            for medOne in visit[2]:
                for medTwo in visit[2]:
                    if(medOne != medTwo):
                        ehr_adj[medOne, medTwo] = 1

    # removing empty rows
    data = list(filter(lambda x: len(x) > 0, data))

    names = file_utils.file_names(dataset_type)

    if(is1V or isM1V):
        print("number_of_bad_data = ", number_of_bad_data)

    if not os.path.exists(names[3]):
        os.makedirs(names[3])

    with open(names[0], 'w+b') as handle:
        pickle.dump(data, handle)

    with open(names[1], 'w+b') as handle:
        pickle.dump(voc, handle)

    with open(names[2], 'w+b') as handle:
        pickle.dump(ehr_adj, handle)


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
