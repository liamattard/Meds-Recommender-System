from src.utils.constants.model_types import Model_Type
from src.utils.constants.dataset_types import Dataset_Type 

import numpy as np
import src.data_handler.start as load_data
import src.model_trainer.start as train

def main():

    # wandb_name = "GameNet Model"
    wandb_name = "new-dataset-split"

    dataset_type = Dataset_Type.full3Age
    model_type = Model_Type.game_net_age

    dataset = load_data.start(dataset_type)
    train.start(model_type, dataset, dataset_type, wandb_name)

    #train.test('saved_models/saved_models/game_net/sota/Epoch_49_JA_0.5178.model', dataset, model_type)

if __name__ == "__main__":
    main()
