import numpy as np
import pandas as pd
# import dcMinMaxFunctions as dc
# import dcor
from scipy.misc import derivative
from sklearn.model_selection import train_test_split
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
import wandb

from cov_help import *

import argparse

parser = argparse.ArgumentParser(description='Forrest cover private training')

parser.add_argument('--data_path', type=str, default='data/covtype.csv',
                    help='Path to the CSV file containing the forrest cover data')
parser.add_argument('--batch_size', type=int, default=1000,
                    help='Batch size for training the model')
parser.add_argument('--num_epochs', type=int, default=10,
                    help='Number of epochs to train the model')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='Learning rate for the optimizer')
parser.add_argument('--wandb_project', type=str, default='covertype',
                    help='Name of the Weights & Biases project to log metrics to')
parser.add_argument('--train_flag',type=int,default=0,
                    help='Flag to indicate if the model should be trained jointly or not')

args = parser.parse_args()

# You can access the parsed arguments like this:
data_path = args.data_path
batch_size = args.batch_size
num_epochs = args.num_epochs
learning_rate = args.learning_rate
wandb_project = args.wandb_project
train_flag = args.train_flag

def main(data_path ,batch_size,num_epochs,learning_rate,train_flag):
    X,Y = cov_data_loader(data_path)
    max_dist = torch.cdist(X, X).max()
    print(max_dist) 
    train_priv = torch.utils.data.TensorDataset(X,Y)

    trainloader_priv = torch.utils.data.DataLoader(train_priv, batch_size=1000,
                                          shuffle=True, num_workers=2)
    net = Net(4)
    optim = torch.optim.Adam(net.parameters(),lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    run = wandb.init(project=wandb_project)
    # wandb.watch(net)
    lr_schedule = LearnerRateScheduler('step', base_learning_rate=learning_rate, decay_rate = 0.99, decay_steps=1)
    train_model_priv(net,trainloader_priv,optim,num_epochs,h=0.82,rate=10,device=torch.device('cuda'),only_reg_flag=0,lr_schedular=lr_schedule)
    print("Please type y or n if you want to save model: \n")
    input1 = input()
    if(input1 == 'y'):
        print("Please type the name of the model: \n")
        input2 = input()
        model_path = "Models/"+input2
        torch.save(net.state_dict(),model_path)
        print("Model saved successfully")
    # # wandb.log_artifact(net)

#write a script to run the code

if __name__ == "__main__":
    main(data_path=data_path,batch_size=batch_size,num_epochs=num_epochs,learning_rate=learning_rate,train_flag=train_flag)

