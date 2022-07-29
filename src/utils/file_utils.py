from src.utils.constants.dataset_types import Dataset_Type

import configparser
import logging
import os

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Data loader")

config = configparser.ConfigParser()
config.sections()
config.read("properties.ini")

def files_exist(dataset_type):
    log.info("Checking if Datasets already generated")
    names = file_names(dataset_type)
    exists = True
    for name in names:
        exists = exists and os.path.exists(name)
    return exists

def file_names(dataset_type):
    if(dataset_type == Dataset_Type.full):
        return config["DATASET"]["full_data"], config["DATASET"]["full_voc"], config["DATASET"]["full_ehr_adj"]

    if(dataset_type == Dataset_Type.sota):
        return config["DATASET"]["sota_data"], config["DATASET"]["sota_voc"], config["DATASET"]["sota_ehr_adj"]
