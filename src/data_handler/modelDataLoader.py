from src.utils.classes.Dataset import Dataset

import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Data loader")


def generate_pure_coll(user_med_pd, med_set):
    user_med_pd = user_med_pd.astype('string')
    user_med_pd = user_med_pd.drop_duplicates()

    unique_user_ids = np.unique(user_med_pd['subject_id'])
    unique_med_names = np.array(med_set)

    log.info("Finished loading dataset from directory")
    dataset = Dataset(unique_user_ids=unique_user_ids,
                      unique_medicine_names=unique_med_names,
                      user_medicine_dataset=user_med_pd)

    log.info("Finished generating dataset for model type: Pure Coll Model ")
    return dataset

def generate_gamenet_full():
    print("hello")
