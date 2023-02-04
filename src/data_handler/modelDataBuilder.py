from operator import index
from src.utils.constants.dataset_types import Dataset_Type 
from src.utils.classes.voc import Voc
import src.utils.query_handler as query_handler
import src.utils.file_utils as file_utils 
import src.utils.tools as tools 
import src.utils.data_load_utils as utils

import os
import dill
import logging
import configparser
import logging
import pickle
import pandas as pd

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Data loader")

def build_realistic_dataset(db, dataset_type):

    visit_heartrate = query_handler.load_user_heartrate(db)
    visit_diagnoses_map = query_handler.load_visit_diagnoses(db)
    visit_procedures_map = query_handler.load_visit_procedures(db)
    visit_user_map, user_visit_map = query_handler.load_user_visit_map(db)
    visit_by_time = query_handler.load_user_visit_time(db)
    visit_medicine_map = query_handler.load_visit_medicine(db, dataset_type)
    user_age_map = query_handler.load_user_age_map(db)
    user_insurance_map = query_handler.load_user_insurance(db)
    user_gender_map = query_handler.load_user_gender_map(db)

    data_train = []
    data_test = []

    voc = {'diag_voc': Voc(), 'pro_voc': Voc(), 'med_voc': Voc(), 'age_voc':
            Voc(), 'patient_voc': Voc(), 'heartrate_voc': Voc(), 
            'insurance_voc': Voc(), 'gender_voc': Voc()}

    if tools.isNoPro(dataset_type):
        voc['pro_voc'].idx2word[0] = 'empty'
        voc['pro_voc'].word2idx['empty'] = 0

    voc['heartrate_voc'].idx2word[0] = 'empty'
    voc['heartrate_voc'].word2idx['empty'] = 0

    splitpoint = int(len(visit_by_time) * 80/100)
    train = list(visit_by_time.keys())[0:splitpoint]
    test = list(visit_by_time.keys())[splitpoint:]
    list_of_train_patients = []
    list_of_test_patients = []

    def get_visit(date, voc):
        visits_arr = []
        for visit in visit_by_time[date]:

            patient_id = visit_user_map[visit]

            if tools.isM1V(dataset_type):
                if len(user_visit_map[patient_id]) < 2:
                    continue

            visit_arr , voc = utils.get_visit_arr(visit, 
                    visit_diagnoses_map, visit_procedures_map,
                    visit_medicine_map, voc, dataset_type)

            if len(visit_arr) > 0:
                #Getting patient ID
                voc["patient_voc"] = utils.append(voc["patient_voc"], patient_id)

                #Getting patient Age
                user_age = user_age_map[patient_id]
                voc["age_voc"] = utils.append(voc["age_voc"], user_age)

                #Getting insurace type
                insurance_type = "other"
                if patient_id in user_insurance_map:
                    insurance_type = user_insurance_map[patient_id]
                voc["insurance_voc"] = utils.append(voc["insurance_voc"], insurance_type)

                #Getting gender
                gender = "other"
                if patient_id in user_gender_map:
                    gender = user_gender_map[patient_id]
                voc["gender_voc"] = utils.append(voc["gender_voc"], gender)

                #Getting patient Heart Rate
                heartrate = None
                if visit in visit_heartrate:
                    heartrate = visit_heartrate[visit]
                    voc["heartrate_voc"] = utils.append(voc["heartrate_voc"], heartrate[0])
                    voc["heartrate_voc"] = utils.append(voc["heartrate_voc"], heartrate[1])

                past_visits = user_visit_map[patient_id]
                past_visits_arr = []

                #Getting past visits
                current_visit_number = past_visits.index(visit)
                if current_visit_number > 0:
                    for past_visit in past_visits[:current_visit_number]:
                        past_visit_arr, voc = utils.get_visit_arr(past_visit, 
                                visit_diagnoses_map, visit_procedures_map,
                                visit_medicine_map, voc, dataset_type)

                        if len(past_visit_arr) > 0:
                            past_visits_arr.append(past_visit_arr)

                visit_arr.append(voc["age_voc"].word2idx[user_age])
                visit_arr.append(voc["patient_voc"].word2idx[patient_id])
                visit_arr.append(voc["gender_voc"].word2idx[gender])
                visit_arr.append(voc["insurance_voc"].word2idx[insurance_type])
                
                if heartrate != None:
                    heartrate_min = voc["heartrate_voc"].word2idx[heartrate[0]]
                    heartrate_max = voc["heartrate_voc"].word2idx[heartrate[1]]
                    visit_arr.append([heartrate_min, heartrate_max])
                else:
                    visit_arr.append([0, 0])

                visit_arr.append(past_visits_arr)

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

    ehr_adj = utils.build_ehr_adj(voc, data_train, False, True)

    # Saving files
    names = file_utils.file_names(dataset_type)

    if tools.isNoPro(dataset_type):
        total_keys = len(voc["pro_voc"].idx2word)
        voc["pro_voc"].idx2word[total_keys] = 0
        voc["pro_voc"].word2idx[0] = total_keys

    if not os.path.exists(names[3]):
        os.makedirs(names[3])

    with open(names[0], 'w+b') as handle:
        pickle.dump([data_train, data_test], handle)

    with open(names[1], 'w+b') as handle:
        pickle.dump(voc, handle)

    with open(names[2], 'w+b') as handle:
        pickle.dump(ehr_adj, handle)

def build_dataset(db, dataset_type):

    visit_diagnoses_map = query_handler.load_visit_diagnoses(db)
    visit_procedures_map = query_handler.load_visit_procedures(db)
    _, user_visit_map = query_handler.load_user_visit_map(db)

    visit_medicine_map = query_handler.load_visit_medicine(db, dataset_type)
    user_age_map = query_handler.load_user_age_map(db)

    data = []
    voc = {}

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
            visit_arr, voc= utils.get_visit_arr(visit, 
                    visit_diagnoses_map, visit_procedures_map,
                    visit_medicine_map, voc, dataset_type)
            if visit_arr != []:
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
                    voc["age_voc"] = utils.append(voc["age_voc"], user_age)
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


    data = list(filter(lambda x: len(x) > 0, data))

    split_point = int(len(data) * 80/100)

    data_train = data[:split_point]
    data_eval = data[split_point:]

    ehr_adj = utils.build_ehr_adj(voc, data_train, isAge, False)

    data = [data_train, data_eval]

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

def build_sota_dataset():
    data = dill.load(open('data/dataset/sota/data.pkl','rb'))
    voc = dill.load(open('data/dataset/sota/voc.pkl','rb'))

    split_point = int(len(data) * 80/100)

    data_train = data[:split_point]
    data_eval = data[split_point:]

    data = [data_train, data_eval]

    ehr_adj = utils.build_ehr_adj(voc, data_train, False, False)

    names = file_utils.file_names(Dataset_Type.sota)

    if not os.path.exists(names[3]):
        os.makedirs(names[3])

    with open(names[0], 'w+b') as handle:
        pickle.dump(data, handle)

    with open(names[2], 'w+b') as handle:
        pickle.dump(ehr_adj, handle)


#deprecated
def build_pure_col(db):

    config = configparser.ConfigParser()
    config.sections()
    config.read("properties.ini")

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
