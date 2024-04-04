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
from opacus import PrivacyEngine
from cov_help import *
import time
import argparse

parser = argparse.ArgumentParser(description='Forrest cover private testing')
parser.add_argument('--data_path', type=str, default='data/covtype.csv',
                    help='Path to the CSV file containing the forrest cover data')
parser.add_argument('--eps', type=float, default=5.0,
                    help='Set epsilon for the model')
parser.add_argument('--batch_size', type=int, default=4096,
                    help='Batch size for training the model')
parser.add_argument('--num_epochs', type=int, default=100,
                    help='Number of epochs to train the model')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='Learning rate for the optimizer')
parser.add_argument('--wandb_project', type=str, default='covertype test baseline',
                    help='Name of the Weights & Biases project to log metrics to')
parser.add_argument('--norm',type=float,default= 1,
                    help='Normalizing the data by multiplying with this number')

args = parser.parse_args()
# You can access the parsed arguments like this:
data_path = args.data_path
eps = args.eps
batch_size = args.batch_size
num_epochs = args.num_epochs
learning_rate = args.learning_rate
wandb_project = args.wandb_project
norm = args.norm
 # adds all of the arguments as config variables
def main():
    X,Y = cov_data_loader(data_path,norm=norm)
    run = wandb.init(project=wandb_project)
    wandb.config.update(args)
   
    
    #write code for train test split using X_emb and Y
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train,Y_train), batch_size=batch_size,
                                            shuffle=True, num_workers=2,drop_last=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test,Y_test), batch_size=batch_size,
                                            shuffle=True, num_workers=2)
   
    #write code to append Xemb and Xemb_rest
    
    criterion = nn.CrossEntropyLoss()
    # model = nn.Sequential(
    #             nn.Linear(54, 128),
    #             nn.ReLU(),
    #             nn.Linear(128, 256),
    #             nn.ReLU(),
    #             nn.Linear(256, 128),
    #             nn.ReLU(),
    #             nn.Linear(128, 7),
    #             nn.Softmax(dim=1)

    #         )
    model = nn.Sequential(
            nn.Linear(54, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 7),
            nn.Softmax(dim=1)
        )
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=1e-4)
    # optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate,weight_decay=1e-4)
    
  

# enter PrivacyEngine
    privacy_engine = PrivacyEngine()
    model2, optimizer2, data_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        target_epsilon=eps,
        target_delta =0.0001,
        epochs = num_epochs,
        max_grad_norm=1.0,
    )
    time_start = time.time()
    train_emb(model2, data_loader, criterion, optimizer2, num_epochs=num_epochs,device=torch.device('cuda'),test_loader = test_loader)
    time_end = time.time()
    print("Time taken to train the model: ",time_end-time_start)
    
    # test_model(model,test_emb_loader)
    # test_model(model,test_emb_full_loader)
if __name__ == "__main__":
    main()

