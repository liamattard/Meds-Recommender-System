from src.utils.constants.model_types import Model_Type
from torch.optim import Adam
from sklearn.model_selection import train_test_split


import os
import wandb
import torch
import torch.nn as nn
import src.data_handler.start as load_data
import src.models.pure_collaborative as pure_col_model 

def eval(model, test_pd,device):
    model.eval()

    BATCH_SIZE = 4096

    dataset_length = len(test_pd)
    number_of_batches = int(dataset_length // BATCH_SIZE) + 1

    for batch in range(number_of_batches):
        print("Eval Batch: {}/{}".format(batch, number_of_batches))

        subject_ids = test_pd.iloc[batch * BATCH_SIZE:(batch + 1) * BATCH_SIZE, 1].tolist()
        medicine_names = test_pd.iloc[batch * BATCH_SIZE:(batch + 1) * BATCH_SIZE, 2].tolist()
        x = model(subject_ids, medicine_names).to(device)
        y = torch.eye(x.shape[0]).to(device)

        loss = nn.CrossEntropyLoss(reduction='sum')
        loss_value = loss(x,torch.argmax(y, dim=1)).to(device)

        wandb.log({"testing loss": loss_value})
        print("Eval Loss: {}".format(loss_value))


def temp():
    wandb.init(project="PyTorch Collaborative Filtering Model", entity="liam_dratta")

    device = torch.device('cuda:{}'.format(1))
    device = torch.device('cpu')

    model_type = Model_Type.pure_collaborative

    # Load Data
    dataset = load_data.start(model_type)
    model = pure_col_model.Model(dataset,device)
    wandb.config = {
      "learning_rate": 0.001,
      "epochs": 10,
      "batch_size": 8192 
    }

    model.to(device=device)
    if dataset.user_medicine_dataset is not None:

        pd = dataset.user_medicine_dataset
        pd = pd.drop_duplicates()
        pd['subject_id'] = pd['subject_id'].astype(str)
        pd = pd.drop(['drug_id','has_past_medicine'], axis=1)
        pd = pd.sample(frac=1, random_state=1).reset_index()

        train_pd, test_pd = train_test_split(
            pd, test_size=0.2, shuffle=False)

        optimizer = Adam(list(model.parameters()), lr=0.1)

        EPOCHS = 10
        BATCH_SIZE = 8192
        
        for epoch in range(EPOCHS):
            print("Epoch: {}".format(epoch))
            model.train()
            dataset_length = len(train_pd)
            number_of_batches = int(dataset_length // BATCH_SIZE) + 1
            for batch in range(number_of_batches):
                print("Batch: {}/{}".format(batch, number_of_batches))
                subject_ids = train_pd.iloc[batch * BATCH_SIZE:(batch + 1) * BATCH_SIZE, 1].tolist()
                medicine_names = train_pd.iloc[batch * BATCH_SIZE:(batch + 1) * BATCH_SIZE, 2].tolist()
                x = model(subject_ids, medicine_names).to(device)
                y = torch.eye(x.shape[0]).to(device)

                loss = nn.CrossEntropyLoss(reduction='sum')
                loss_value = loss(x,torch.argmax(y, dim=1).to(device))

                optimizer.zero_grad()
                loss_value.backward(retain_graph=True)
                optimizer.step()
                wandb.log({"training loss": loss_value})
                print("Train LOSS: {}".format(loss_value))
            wandb.watch(model)
            wandb.log({"epoch": epoch})
            eval(model, test_pd,device)

        path = os.path.join('saved_models/', 'net_{}_{}.pth'.format('pure_col', epoch))
        torch.save(model.state_dict(), path)

