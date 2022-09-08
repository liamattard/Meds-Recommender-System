from src.utils.classes.Dataset import Dataset

from src.utils.constants.dataset_types import Dataset_Type 
from src.utils.classes.voc import Voc
import src.utils.query_handler as query_handler
import src.utils.file_utils as file_utils 
import src.utils.tools as tools 

import os
import dill
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



def build_realistic_dataset(db, dataset_type):

    visit_diagnoses_map = query_handler.load_visit_diagnoses(db)
    visit_procedures_map = query_handler.load_visit_procedures(db)
    visit_user_map, _= query_handler.load_user_visit_map(db)
    visit_by_time = query_handler.load_user_visit_time(db)
    visit_medicine_map = query_handler.load_visit_medicine(db, dataset_type)
    user_age_map = query_handler.load_user_age_map(db)

    data_train = []
    data_test = []
    voc = {'diag_voc': Voc(), 'pro_voc': Voc(), 'med_voc': Voc(), 'age_voc': Voc(), 'patient_voc': Voc()}

    splitpoint = int(len(visit_by_time)* 80/100)
    train = list(visit_by_time.keys())[0:splitpoint]
    test = list(visit_by_time.keys())[splitpoint:]
    list_of_train_patients = []
    list_of_test_patients = []

    def get_visit(date, voc):
        visits_arr = []
        for visit in visit_by_time[date]:
            visit_arr , voc = get_visit_arr(visit, 
                    visit_diagnoses_map, visit_procedures_map,
                    visit_medicine_map, voc)

            if len(visit_arr) > 0:

                patient_id = visit_user_map[visit]
                user_age = user_age_map[patient_id]
                voc["age_voc"] = append(voc["age_voc"], user_age)
                voc["patient_voc"] = append(voc["patient_voc"], patient_id)
                visit_arr.append(voc["age_voc"].word2idx[user_age])
                visit_arr.append(voc["patient_voc"].word2idx[patient_id])
                visits_arr.append(visit_arr)

        return visits_arr, voc


    for date in train:
        visit_arr, voc = get_visit(date,voc)
        if len(visit_arr) > 0:
            for i in visit_arr:
                list_of_train_patients.append(i[-1])
            data_train = data_train + visit_arr

    for date in test:
        visit_arr, voc = get_visit(date,voc)
        if len(visit_arr) > 0:
            for i in visit_arr:
                list_of_test_patients.append(i[-1])
            data_test= data_test + visit_arr

    ehr_adj = build_ehr_adj(voc, data_train, False, True)

    # Saving files
    names = file_utils.file_names(dataset_type)

    if not os.path.exists(names[3]):
        os.makedirs(names[3])

    with open(names[0], 'w+b') as handle:
        pickle.dump([data_train, data_test], handle)

    with open(names[1], 'w+b') as handle:
        pickle.dump(voc, handle)

    with open(names[2], 'w+b') as handle:
        pickle.dump(ehr_adj, handle)

def build_dataset(db, dataset_type):

    # Loading Data From Database
    visit_diagnoses_map = query_handler.load_visit_diagnoses(db)
    visit_procedures_map = query_handler.load_visit_procedures(db)
    _, user_visit_map = query_handler.load_user_visit_map(db)
    visit_medicine_map = query_handler.load_visit_medicine(db, dataset_type)
    user_age_map = query_handler.load_user_age_map(db)

    list_of_dates = []
    data = []
    voc = {'diag_voc': Voc(), 'pro_voc': Voc(), 'med_voc': Voc()}

    isAge = tools.isAge(dataset_type)
    is1V = tools.is1V(dataset_type)
    isM1V = tools.isM1V(dataset_type)

    if isAge:
        voc['age_voc']=  Voc()

    number_of_bad_data = 0

    for _, patient in enumerate(user_visit_map):

        if(is1V):
            if len(user_visit_map[patient]) > 1:
                continue
        elif(isM1V):
            #TODO: I think here I should have included the 2
            if len(user_visit_map[patient]) < 2:
                continue

        patient_arr = []

        for visit in user_visit_map[patient]:
            visit_arr, voc = get_visit_arr(visit, 
                    visit_diagnoses_map, visit_procedures_map,
                    visit_medicine_map, voc)
            if visit_arr != []:
                list_of_dates.append(visit)
                patient_arr.append(visit_arr)

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

    if(is1V or isM1V):
        print("number_of_bad_data = ", number_of_bad_data)

    ehr_adj = build_ehr_adj(voc, data, isAge, False)

    breakpoint()
    split_point = int(len(data[0]) * 2 / 3)
    eval_len = int(len(dataset.data[0][split_point:]) / 2)

    data_train = dataset.data[0][:split_point]
    data_eval = dataset.data[0][split_point+eval_len:]


    # removing empty rows
    data = list(filter(lambda x: len(x) > 0, data))

    # Saving files
    names = file_utils.file_names(dataset_type)

    if not os.path.exists(names[3]):
        os.makedirs(names[3])

    with open(names[0], 'w+b') as handle:
        pickle.dump(data, handle)

    with open(names[1], 'w+b') as handle:
        pickle.dump(voc, handle)

    with open(names[2], 'w+b') as handle:
        pickle.dump(ehr_adj, handle)


def get_visit_arr(visit, visit_diagnoses_map, visit_procedures_map, visit_medicine_map, voc):
        
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


def build_sota_dataset():
    data = pickle.load(open('data/dataset/sota/data.pkl','rb'))
    voc = dill.load(open('data/dataset/sota/voc.pkl','rb'))

    split_point = int(len(data) * 80/100)

    data_train = data[:split_point]
    data_eval = data[split_point:]

    data = [data_train, data_eval]

    ehr_adj = build_ehr_adj(voc, data_train, False, False)

    names = file_utils.file_names(Dataset_Type.sota)

    if not os.path.exists(names[3]):
        os.makedirs(names[3])

    with open(names[0], 'w+b') as handle:
        pickle.dump(data, handle)

    with open(names[2], 'w+b') as handle:
        pickle.dump(ehr_adj, handle)



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

#deprecated
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
