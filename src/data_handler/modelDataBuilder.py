from src.utils.classes.Dataset import Dataset

from src.utils.constants.dataset_types import Dataset_Type 
import src.utils.query_handler as query_handler
import src.utils.file_utils as file_utils 

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
    visit_diagnoses_map, diag_voc = query_handler.load_visit_diagnoses(db)
    visit_procedures_map, prod_voc = query_handler.load_visit_procedures(db)
    visit_medicine_map, med_voc = query_handler.load_visit_medicine(db)
    _, user_visit_map = query_handler.load_user_visit_map(db)
    data = []
    voc = {'diag_voc': diag_voc, 'pro_voc': prod_voc, 'med_voc': med_voc}
    ehr_adj = np.zeros((len(med_voc.idx2word), len(med_voc.idx2word)))

    for patient_count, patient in enumerate(user_visit_map):
        data.append([])
        for visit in user_visit_map[patient]:

            current_visit = []
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

                for medOne in visit_medicine_map[visit]:
                    for medTwo in visit_medicine_map[visit]:
                        if(medOne != medTwo):
                            ehr_adj[medOne, medTwo] = 1
            else:
                containsNone = True

            if(dataset_type == Dataset_Type.full1V):
                if len(user_visit_map[patient]) == 1:
                    containsNone = True

            if not containsNone:
                data[patient_count].append(current_visit)

    data = list(filter(lambda x: len(x) > 0, data))

    names = file_utils.file_names(dataset_type)

    with open(names[0], 'wb') as handle:
        pickle.dump(data, handle)

    with open(names[1], 'wb') as handle:
        pickle.dump(voc, handle)

    with open(names[2], 'wb') as handle:
        pickle.dump(ehr_adj, handle)
