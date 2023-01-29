from src.utils.constants.model_types import Model_Type
from src.utils.constants.dataset_types import Dataset_Type 

import src.data_handler.start as load_data
import src.model_trainer.start as train_test
import numpy as np
from scipy.sparse import csc_matrix


from sklearn.metrics import jaccard_score

def main():

    wandb_name = None

    dataset_type = Dataset_Type.sota_with_single

    dataset = load_data.start(dataset_type)

    dataset.data[0][0], dataset.data[0][1] = filter_visits_with_one(dataset)
    # dataset.data[0][0], dataset.data[0][1] = filter_visits_more_than_one(dataset)
    # dataset.data[0][0], dataset.data[0][1] = undersampling(dataset)

    print_statistics(dataset)

    #model_type = Model_Type.game_net
    #train_test.start(model_type, dataset, dataset_type, wandb_name)


def test(model_type, dataset):
    model_path = "/home/liam/Documents/Masters/saved_models/realistic/gameNet/0_85_threshold/game_net/realistic3/Epoch_49.model"
    train_test.test(model_path, model_type, dataset)
    

def print_statistics(dataset):
    patient_count = len(dataset.data[0][0]) + len(dataset.data[0][1])
    visit_count = 0
    prescription_count = 0
    covered_medicine = set()
    len_medicine = len(dataset.voc[0]['med_voc'].idx2word)

    patient_medicine_arr = np.zeros((patient_count, len_medicine))

    for patient_i, patient in enumerate(dataset.data[0][0]):
        visit_count += len(patient)

        for visit in patient:
            prescription_count += len(visit)
            covered_medicine.update(visit[2])
            for med in visit[2]:
                patient_medicine_arr[patient_i, med] = 1

    for patient_i, patient in enumerate(dataset.data[0][1]):
        visit_count += len(patient)

        for visit in patient:
            prescription_count += len(visit)
            covered_medicine.update(visit[2])

            lenz = len(dataset.data[0][0]) + patient_i

            for med in visit[2]:
                patient_medicine_arr[lenz, med] = 1


    coverage = (len(covered_medicine)/len_medicine) * 100

    dot_product = np.dot(patient_medicine_arr, patient_medicine_arr.T)

    element_product = np.multiply(patient_medicine_arr, patient_medicine_arr)
    element_product = element_product.sum(axis=1)
    jaccard = dot_product / (element_product[:, None] + element_product - dot_product)

    average_similarity = jaccard[np.triu_indices(jaccard.shape[0], k=1)].mean()

    print("Patient Count: " , patient_count)
    print("Visit Count: " , visit_count)
    print("Medicine Count: " , prescription_count)
    print("Coverage: " , coverage)
    print("Personalisation: " , average_similarity)



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
