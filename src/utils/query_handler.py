from src.utils import tools
from src.utils.classes.voc import Voc

import numpy as np

queries_base_path = "sql_queries/"


def load(db):

    user_med_list, med_set = load_medicine_values_by_user(db)
    _, visit_count_map, user_visit_map = load_ordered_visits(db)
    visit_medicine_map = load_visit_medicine(db)

    med_ids = tools.generate_med_ids(med_set)

    final_list = []
    medicine_list = []
    has_past_count = 0

    for row in user_med_list:
        visit_count = visit_count_map[row[1]]
        past_medicine_set = set()
        has_past = False
        if visit_count > 0:
            has_past_count = has_past_count + 1
            has_past = True
            for i in range(visit_count):
                past_visit = user_visit_map[row[0]][i]
                if past_visit in visit_medicine_map:
                    past_medicine = list(visit_medicine_map[past_visit])
                    past_medicine_set.update(past_medicine)

        medicine_list.append(list(past_medicine_set))
        final_list.append([row[0], row[2], med_ids[row[2]], has_past])

    _, y = tools.get_list_dimension(medicine_list)
    past_medicine_array = np.zeros((has_past_count, y), dtype="S4")

    temp_count = 0
    for i, row in enumerate(final_list):
        if row[3]:
            for j, med in enumerate(medicine_list[i]):
                past_medicine_array[temp_count][j] = str(med_ids[med])
            temp_count = temp_count + 1

    # for i, row in enumerate(medicine_list):
        # for j, val in enumerate(row):
            # past_medicine_array[i][j] = med_ids[val]

    # for i, row in enumerate(medicine_list):
        # past_medicine_array = []
        # j = 0

        # for j, val in enumerate(row):
            # past_medicine_array.append(med_ids[val])
        # past_medicine_array.extend('0'*(y-j))
        # final_list[i].append(past_medicine_array)

    return final_list, med_set, past_medicine_array

def load_visit_diagnoses(db):
    visit_diagnoses_query = open(queries_base_path + "getDiagnoses.sql").read()
    visit_diagnoses_list = db.query(visit_diagnoses_query)
    return make_dict(visit_diagnoses_list)

def load_visit_procedures(db):
    visit_procedures_query = open(queries_base_path + "getProcedures.sql").read()
    visit_procedures_list = db.query(visit_procedures_query)
    return make_dict(visit_procedures_list)

def load_visit_medicine(db):
    medicine_query = open(queries_base_path + "getMedicine.sql").read()
    visit_medicine_list = db.query(medicine_query)
    return make_dict(visit_medicine_list)

def load_user_age_map(db):

    age_query = open(queries_base_path + "getAge.sql").read()
    user_age_list = db.query(age_query)
    user_age_map = dict(user_age_list)

    return user_age_map


def load_user_gender_map(db):

    gender_query = open(queries_base_path + "getGender.sql").read()
    user_gender_list = db.query(gender_query)
    user_gender_map = dict(user_gender_list)

    return user_gender_map

def load_user_visit_map(db):

    visit_query = open(queries_base_path + "getOrderedVisits.sql").read()
    user_visit_list = db.query(visit_query)
    user_visit_map = {}

    for visit in user_visit_list:

        if visit[0] in user_visit_map:
            user_visit_map[visit[0]].append(visit[1])
        else:
            user_visit_map[visit[0]] = [visit[1]]

    visit_user_map = dict(map(lambda x: (x[1], x[0]), user_visit_list))

    return visit_user_map, user_visit_map

def load_ordered_visits(db):

    patient_query = open(queries_base_path + "getPatients.sql").read()
    patient_list = db.query(patient_query)
    patient_list = list(map(lambda x: x[0], patient_list))

    patient_visits_count = np.zeros(len(patient_list), dtype='int32')
    patient_count_map = dict(zip(patient_list, patient_visits_count))

    visit_query = open(queries_base_path + "getOrderedVisits.sql").read()
    user_visit_list = db.query(visit_query)
    visit_count_map = {}
    user_visit_map = {}

    for visit in user_visit_list:

        visit_count_map[visit[1]] = patient_count_map[visit[0]]
        patient_count_map[visit[0]] = patient_count_map[visit[0]] + 1

        if visit[0] in user_visit_map:
            user_visit_map[visit[0]].append(visit[1])
        else:
            user_visit_map[visit[0]] = [visit[1]]

    visit_user_map = dict(map(lambda x: (x[1], x[0]), user_visit_list))

    return visit_user_map, visit_count_map, user_visit_map



def load_medicine_values_by_user(db):

    medicine_query = open(queries_base_path + "getUniqueMedicine.sql").read()
    medicine_list = db.query(medicine_query)
    medicine_list = list(map(lambda x: x[0], medicine_list))

    medicine_query = open(queries_base_path +
                          "getMedicineValuesByUser.sql").read()
    user_medicine_list = db.query(medicine_query)

    return user_medicine_list, medicine_list


def make_dict(list_of_items):
    final_map = {}
    word2idx = {}
    idx2word = {}

    for row in list_of_items:
        if row[0] in final_map:
            word2idx, idx2word = append(word2idx, idx2word, row[1])
            final_map[row[0]].add(word2idx[row[1]])
        else:
            word2idx, idx2word = append(word2idx, idx2word, row[1])
            final_map[row[0]] = {word2idx[row[1]]}

    return final_map, Voc(idx2word, word2idx)

def append(word2idx, idx2word, word):
    if word not in word2idx:
        word2idx[word] = len(word2idx)
        idx2word[word2idx[word]] = word
    return word2idx, idx2word

