import src.model_trainer.gamenet_age_train as gamenet_age_train
import src.model_trainer.gamenet_train as gamenet_train
from src.utils.constants.model_types import Model_Type

def start(model_type, dataset, dataset_type, wandb):

    if model_type == Model_Type.game_net_age:
        gamenet_age_train.train(dataset, dataset_type, model_type, wandb)
    else:
        gamenet_train.train(dataset, dataset_type, model_type, wandb)


def test(path, dataset, model_type):
    if model_type == Model_Type.game_net_age:
        gamenet_age_train.test(path, dataset)
    else:
        gamenet_train.test(path, dataset)
