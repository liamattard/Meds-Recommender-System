from src.utils.constants.model_types import Model_Type

import src.model_trainer.gamenet_train as gamenet_train
import src.model_trainer.collaborative as collaborative

def start(model_type, dataset, dataset_type):
    if(model_type == Model_Type.game_net):
        gamenet_train.train(dataset, dataset_type)

    if(model_type == Model_Type.colab_filtering):
        collaborative.train(dataset,dataset_type)


