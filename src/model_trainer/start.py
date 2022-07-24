from src.utils.constants.model_types import Model_Type

import src.models.pure_collaborative as pure_collaborative
import src.model_trainer.gamenet_train as gamenet_train
import torch
import torch.nn
import torch.optim
import numpy as np

def start(model_type, dataset, dataset_type):
    if(model_type == Model_Type.game_net):
        gamenet_train.train(dataset, dataset_type)

    if(model_type == Model_Type.pure_collaborative):
        model = pure_collaborative.Model(dataset)

