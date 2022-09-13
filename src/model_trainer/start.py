import src.model_trainer.gamenet_train as gamenet_train
import src.model_trainer.gamenet_realistic_train as gamenet_realistic_train
from src.utils.constants.model_types import Model_Type
from src.utils.constants.dataset_types import Dataset_Type
import src.utils.tools as tools

def start(model_type, dataset, dataset_type, wandb):

    if tools.isByDate(dataset_type):
        gamenet_realistic_train.train(dataset, dataset_type, model_type, wandb)

    if model_type == Model_Type.game_net_age:
        gamenet_train.train(dataset, dataset_type, model_type, wandb, with_age=True)
    else:
        gamenet_train.train(dataset, dataset_type, model_type, wandb)


def test(path, dataset, model_type = None):
    gamenet_train.test(path, dataset)
