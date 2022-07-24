from src.utils.constants.dataset_types import Dataset_Type
from src.utils.classes.Dataset import Dataset

import src.utils.file_utils as file_utils 
import src.data_handler.modelDataLoader as data_loader
import src.data_handler.modelDataBuilder as data_builder
import src.utils.database_utils as db

import pandas as pd
import configparser
import logging
import pickle
import os

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Data loader")

config = configparser.ConfigParser()
config.sections()
config.read("properties.ini")


def start(dataset_type):

    dataset_exits = file_utils.files_exist(dataset_type)

    if not dataset_exits:
        log.info("Started generating the dataset using PostgreSQL server")
        db.connect()
        save(db, dataset_type)

    log.info("Started loading the dataset")
    return load(dataset_type)


def save(db, dataset_type):
    if(dataset_type == Dataset_Type.full):
        data_builder.build_dataset(db)

def load(dataset_type):
    log.info("Starting loading dataset from directory")
    names = file_utils.file_names(dataset_type)

    data = pickle.load(open(names[0], 'rb'))

    clean_data = data

    for i, patient in enumerate(data):
        for j,visit in enumerate(patient):
            if(len(visit[2]) == 0):
                clean_data[i].remove(visit)
    clean_data = list(filter(lambda x: len(x) > 0, clean_data))
    data = clean_data
    voc = pickle.load(open(names[1], 'rb'))
    ehr_adj = pickle.load(open(names[2], 'rb'))

    return Dataset(data=data, voc=voc, ehr_adj=ehr_adj)

