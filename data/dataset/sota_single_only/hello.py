import numpy as np 
import dill
import pickle

def build_ehr_adj(voc, data, isAge, isRealistic):
    ehr_adj = np.zeros((len(voc["med_voc"].idx2word), len(voc["med_voc"].idx2word)))

    for patient in data:
        x = patient

        if isAge:
            x = patient[:-1]
        elif isRealistic:
            x = [patient[:-2]]

        for visit in x:
            for i,medOne in enumerate(visit[2]):
                for j,medTwo in enumerate(visit[2]):
                    if j<=i:
                        continue
                    ehr_adj[medOne, medTwo] = 1
                    ehr_adj[medTwo, medOne] = 1

    return ehr_adj



data = dill.load(open('data.pkl','rb'))
voc = dill.load(open('voc.pkl','rb'))

split_point = int(len(data) * 80/100)

data_train = data[:split_point]
data_eval = data[split_point:]

data = [data_train, data_eval]

ehr_adj = build_ehr_adj(voc, data_train, False, False)


with open("generated_data.pkl", 'w+b') as handle:
    pickle.dump(data, handle)

with open("generated_ehr.pkl", 'w+b') as handle:
    pickle.dump(ehr_adj, handle)


