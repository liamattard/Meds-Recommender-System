#DEPRECATED

class Dataset:
    def __init__(self, 
                 data=None,
                 voc=None,
                 ehr_adj=None,
                 unique_medicine_names=None,
                 unique_user_ids=None,
                 unique_medicine_ids=None,
                 medicine_dataset=None,
                 user_medicine_dataset=None,
                 medicine_titles=None,
                 user_ids=None,
                 train=None,
                 test=None):

        self.data=data,
        self.voc=voc,
        self.ehr_adj=ehr_adj,
        self.unique_medicine_names = unique_medicine_names
        self.unique_medicine_ids = unique_medicine_ids
        self.unique_user_ids = unique_user_ids
        self.medicine_dataset = medicine_dataset
        self.user_medicine_dataset = user_medicine_dataset
        self.medicine_titles = medicine_titles
        self.user_ids = user_ids
        self.train = train
        self.test = test
