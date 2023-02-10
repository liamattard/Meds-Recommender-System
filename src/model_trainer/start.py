import src.model_trainer.gamenet_train as gamenet_train
import src.model_trainer.gamenet_realistic_train as gamenet_realistic_train
import src.model_trainer.top_20_realistic_train as gamenet_realistic_train_top
from src.utils.constants.model_types import Model_Type
import src.utils.tools as tools


def start(dataset, dataset_type, wandb, features=None, threshold=None,
          epochs=None, model_type=None, lr=0.0002, batches=None):

    if model_type == Model_Type.top_20:
        gamenet_realistic_train_top.train(
            dataset, dataset_type, model_type, wandb)

    elif tools.isByDate(dataset_type):
        gamenet_realistic_train.train(
            dataset=dataset, dataset_type=dataset_type, wandb_name=wandb,
            features=features, threshold=threshold, num_of_epochs=epochs, lr=lr,
            batches=batches)

    else:
        gamenet_train.train(dataset, dataset_type, model_type, wandb)


def test(path, model_type, dataset):
    # gamenet_train.test(path, dataset)
    #gamenet_realistic_train.test(path, model_type, dataset)
    print("hello")
