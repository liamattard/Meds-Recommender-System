from src.utils.constants.model_types import Model_Type
from src.utils.constants.dataset_types import Dataset_Type 

import src.data_handler.start as load_data
import src.model_trainer.start as train_test
import numpy as np

def main():

    wandb_name = None

    dataset_type = Dataset_Type.all

    dataset = load_data.start(dataset_type)

    # dataset.data[0][0], dataset.data[0][1] = filter_visits_with_one(dataset)
    # dataset.data[0][0], dataset.data[0][1] = undersampling(dataset)

    # dataset.data[0][0], dataset.data[0][1] = filter_visits_more_than_one(dataset)

    print_statistics_realistic(dataset)

    # features = {"gender","heartrate","insurance","age", "diagnosis", "procedures"}
    features = {"diagnosis", "procedures"}
    threshold = 0.85

    model_type = Model_Type.game_net
    train_test.start(model_type, dataset, dataset_type, wandb_name, features, threshold)

def test(model_type, dataset):
    model_path = "/home/liam/Documents/Masters/saved_models/realistic/gameNet/0_85_threshold/game_net/realistic3/Epoch_49.model"
    train_test.test(model_path, model_type, dataset)
    

def print_statistics_non_realistic(dataset):

    patient_count = len(dataset.data[0][0]) + len(dataset.data[0][1])
    medicine_count = len(dataset.voc[0]['med_voc'].idx2word)

    visit_count = 0
    prescription_count = 0
    covered_medicine = set()

    patient_medicine_arr = np.zeros((patient_count, medicine_count))

    patient_train = len(dataset.data[0][0])

    for patient_i, patient in enumerate(dataset.data[0][0]):
        medicine_arr = patient[0][2]
        prescription_count += len(medicine_arr)
        covered_medicine.update(medicine_arr)
        for med in medicine_arr:
            patient_medicine_arr[patient_i, med] = 1

    for patient_i, patient in enumerate(dataset.data[0][1]):
        medicine_arr = patient[0][2]
        prescription_count += len(medicine_arr)
        covered_medicine.update(medicine_arr)
        for med in medicine_arr:
            patient_medicine_arr[(patient_i + patient_train), med] = 1


    coverage = (len(covered_medicine)/medicine_count) * 100

    dot_product = np.dot(patient_medicine_arr, patient_medicine_arr.T)
    magnitudes = np.linalg.norm(patient_medicine_arr, axis=1)
    cosine_similarity = dot_product / np.outer(magnitudes, magnitudes)
    x = np.triu_indices(1016, k = 1)

    print("Patient Count: " , patient_count)
    print("Visit Count: " , visit_count)
    print("Medicine Count: " , prescription_count)
    print("Coverage: " , coverage)
    print("Personalisation:" , 1 - np.mean(cosine_similarity[x]))
    # print("HHI: " , hhi(patient_medicine_arr))


def print_statistics_realistic(dataset):

    patient_count = len(dataset.voc[0]['patient_voc'].idx2word)
    medicine_count = len(dataset.voc[0]['med_voc'].idx2word)

    visit_count = 0
    prescription_count = 0
    covered_medicine = set()

    for visit in dataset.data[0][0]:
        prescription_count += len(visit[2])
        covered_medicine.update(visit[2])
        visit_count += 1

    for visit in dataset.data[0][1]:
        prescription_count += len(visit[2])
        covered_medicine.update(visit[2])
        visit_count += 1

    coverage = (len(covered_medicine)/medicine_count) * 100

    print("Patient Count: " , patient_count)
    print("Visit Count: " , visit_count)
    print("Medicine Count: " , prescription_count)
    print("Coverage: " , coverage)

def undersampling(dataset):
    return dataset.data[0][0][:4417], dataset.data[0][1][:1025]


def filter_visits_more_than_one(dataset):

    x = []
    y = []
    for patient in dataset.data[0][0]:
        if len(patient) > 1:
            x.append(patient)

    for patient in dataset.data[0][1]:
        if len(patient) > 1:
            y.append(patient)

    return x,y

def filter_visits_with_one(dataset):

    x = []
    y = []
    for patient in dataset.data[0][0]:
        if len(patient) == 1:
            x.append(patient)

    for patient in dataset.data[0][1]:
        if len(patient) == 1:
            y.append(patient)

    return x,y

if __name__ == "__main__":
    main()
