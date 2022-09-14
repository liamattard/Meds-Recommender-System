from src.utils.constants.model_types import Model_Type
from src.utils.constants.dataset_types import Dataset_Type 

import src.data_handler.start as load_data
import src.model_trainer.start as train

def main():

    #wandb_name = "realistic-dataset"
    wandb_name = None

    dataset_type = Dataset_Type.realistic3
    model_type = Model_Type.game_net_item_coll

    dataset = load_data.start(dataset_type)
    train.start(model_type, dataset, dataset_type, wandb_name)

if __name__ == "__main__":
    main()
