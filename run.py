from src.utils.constants.model_types import Model_Type
from src.utils.constants.dataset_types import Dataset_Type

import sys
import src.data_handler.start as load_data
import src.model_trainer.start as train_test
import src.probability_tests.tests as probability_tests
import numpy as np

# All Available Features:
# -gender
# -insurance
# -age
# -diagnosis
# -procedures

def diagrams():
    dataset_type = Dataset_Type.all_3
    dataset = load_data.start(dataset_type)

    probability_tests.start(dataset)

def main():

    dataset_type = Dataset_Type.all_3
    dataset = load_data.start(dataset_type)

    print_statistics_realistic(dataset)

    print("Starting GameNet with Batches and low LR")
    model_type = Model_Type.game_net
    train_test.start(dataset=dataset,
                     dataset_type=dataset_type,
                     wandb="Final",
                     model_type=model_type,
                     threshold=0.85,
                     epochs=50,
                     batches=32,
                     lr=0.0002,
                     original_loss=True,
                     model_name="GameNet_w_Batches_0.0002)")

    print("Starting GameNet with Batches and better LR")
    model_type = Model_Type.game_net
    train_test.start(dataset=dataset,
                     dataset_type=dataset_type,
                     wandb="Final",
                     model_type=model_type,
                     threshold=0.85,
                     epochs=50,
                     batches=32,
                     lr=0.002,
                     original_loss=True,
                     model_name="GameNet_w_Batches_0.002")

    print("Starting GameNet with new loss function")
    model_type = Model_Type.game_net
    train_test.start(dataset=dataset,
                     dataset_type=dataset_type,
                     wandb="Final",
                     model_type=model_type,
                     threshold=0.85,
                     epochs=50,
                     batches=32,
                     lr=0.002,
                     original_loss=False,
                     model_name="GameNet_w_New_Loss")

    print("Starting Demographic GameNet")
    model_type = Model_Type.game_net_knn
    train_test.start(dataset=dataset,
                     dataset_type=dataset_type,
                     wandb="Final",
                     model_type=model_type,
                     threshold=0.50,
                     epochs=50,
                     batches=32,
                     lr=0.002,
                     original_loss=False,
                     model_name="Demographic_GameNet")

    print("Starting Collaborative GameNet")
    model_type = Model_Type.game_net_coll
    train_test.start(dataset=dataset,
                     dataset_type=dataset_type,
                     wandb="Final",
                     model_type=model_type,
                     threshold=0.50,
                     epochs=50,
                     batches=32,
                     lr=0.002,
                     original_loss=False,
                     model_name="Collaborative_GameNet")

    print("Starting Final Model")
    model_type = Model_Type.final_model
    train_test.start(dataset=dataset,
                     dataset_type=dataset_type,
                     wandb="Final",
                     model_type=model_type,
                     threshold=0.50,
                     epochs=50,
                     batches=32,
                     lr=0.002,
                     original_loss=False,
                     model_name="Final_Model")

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
    x = np.triu_indices(1016, k=1)

    print("Patient Count: ", patient_count)
    print("Visit Count: ", visit_count)
    print("Medicine Count: ", prescription_count)
    print("Coverage: ", coverage)
    print("Personalisation:", 1 - np.mean(cosine_similarity[x]))
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

    print("Patient Count: ", patient_count)
    print("Visit Count: ", visit_count)
    print("Medicine Count: ", prescription_count)
    print("Coverage: ", coverage)


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

    return x, y


def filter_visits_with_one(dataset):

    x = []
    y = []
    for patient in dataset.data[0][0]:
        if len(patient) == 1:
            x.append(patient)

    for patient in dataset.data[0][1]:
        if len(patient) == 1:
            y.append(patient)

    return x, y


if __name__ == "__main__":
    # globals()[sys.argv[1]]()
    main()
    
