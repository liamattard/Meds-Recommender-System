from src.utils.constants.model_types import Model_Type
from src.utils.constants.dataset_types import Dataset_Type 

import src.data_handler.start as load_data
import src.model_trainer.start as train

def main():

    dataset_type = Dataset_Type.sota
    model_type = Model_Type.game_net

    dataset = load_data.start(dataset_type)
    train.start(model_type, dataset, dataset_type)

    #train.test('saved_models/saved_models/game_net_age/full3Age/Epoch_49_JA_0.5548.model', dataset)

if __name__ == "__main__":
    main()
