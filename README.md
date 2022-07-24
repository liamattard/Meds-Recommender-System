# W.I.P Thesis project

A medicine recommender system based on the MIMIC-III Dataset. The aim
of this project is to train a model that is able to accept a number
of input details about a user and suggest a number of medications
that the patient should be given.

Sample Properties file 'properties.ini':

```dosini
[POSTGRES]
user = sample_user
password = sample_password 
host = 127.0.0.1
port = 5432
database = sample_database


[DATASET]
medicine_set = path/to/medicine_set.pkl
user_med_pd = path/to/user_medicine_pd.pkl
```
