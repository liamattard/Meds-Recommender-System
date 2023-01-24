from src.utils.constants.model_types import Model_Type
from src.utils.constants.dataset_types import Dataset_Type 

import src.data_handler.start as load_data
import src.model_trainer.start as train_test

def main():

    #wandb_name = "sota-with-our-dataset"
    wandb_name = None

    dataset_type = Dataset_Type.sota

    model_type = Model_Type.game_net


    dataset = load_data.start(dataset_type)

    #dataset.data[0][0], dataset.data[0][1] = filter_visits_with_one(dataset)
    dataset.data[0][0], dataset.data[0][1] = undersampling(dataset)


    train_test.start(model_type, dataset, dataset_type, wandb_name)


def test(model_type, dataset):
    model_path = "/home/liam/Documents/Masters/saved_models/realistic/gameNet/0_85_threshold/game_net/realistic3/Epoch_49.model"
    train_test.test(model_path, model_type, dataset)
    

def print_statistics(dataset):
    patient_count = len(dataset.data[0][0]) + len(dataset.data[0][1])
    visit_count = 0
    prescription_count = 0

    for patient in dataset.data[0][0]:
        visit_count += len(patient)

        for visit in patient:
            prescription_count += len(visit)

    for patient in dataset.data[0][1]:
        visit_count += len(patient)

        for visit in patient:
            prescription_count += len(visit)

    print("Patient Count: " , patient_count)
    print("Visit Count: " , visit_count)
    print("Medicine Count: " , prescription_count)

def undersampling(dataset):
    return dataset.data[0][0][:4417], dataset.data[0][1][:1025]


def filter_visits_with_one(dataset):

    x = []
    y = []
    for patient in dataset.data[0][0]:
        if len(patient) > 1:
            x.append(patient)

    for patient in dataset.data[0][1]:
        if len(patient) > 1:
            y.append(patient)

    return x,y

if __name__ == "__main__":
    main()
