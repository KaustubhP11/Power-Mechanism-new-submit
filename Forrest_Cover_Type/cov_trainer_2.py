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
import time
from cov_help import *

import argparse
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='Forrest cover private training')

parser.add_argument('--data_path', type=str, default='data/covtype.csv',
                    help='Path to the CSV file containing the forrest cover data')
parser.add_argument('--batch_size', type=int, default=4096,
                    help='Batch size for training the model')
parser.add_argument('--batch_size_priv', type=int, default=1024,
                    help='Batch size for calculating eps of the model')
parser.add_argument('--num_epochs', type=int, default=1,
                    help='Number of epochs to train the model')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='Learning rate for the optimizer')
parser.add_argument('--wandb_project', type=str, default='covertype',
                    help='Name of the Weights & Biases project to log metrics to')
parser.add_argument('--train_flag',type=int,default=0,
                    help='Flag to indicate if the model should be trained jointly or not')
parser.add_argument('--norm',type=float,default= 1,
                    help='Normalizing the data by multiplying with this number')
parser.add_argument('--net_depth',type=int,default= 1,
                    help='Depth of the network')  
parser.add_argument('--lambda_loss',type=float,default= 1.0,
                    help='Weighing the joint losses')    
                               
args = parser.parse_args()

# You can access the parsed arguments like this:
data_path = args.data_path
batch_size = args.batch_size
num_epochs = args.num_epochs
learning_rate = args.learning_rate
wandb_project = args.wandb_project
train_flag = args.train_flag
norm = args.norm
net_depth = args.net_depth
lambda_loss = args.lambda_loss
batch_size_priv = args.batch_size_priv
run = wandb.init(project=wandb_project)
wandb.config.update(args)  # adds all of the arguments as config variables



def main():
    X,Y = cov_data_loader(data_path,norm=norm)
    
    # Perform train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # max_dist = torch.cdist(X_train, X_train).max()
    # print(max_dist) 
  
    train_priv = torch.utils.data.TensorDataset(X_train, Y_train)
    test_priv = torch.utils.data.TensorDataset(X_test, Y_test)

    trainloader_priv = torch.utils.data.DataLoader(train_priv, batch_size=batch_size_priv,
                                          shuffle=True, num_workers=2, drop_last=True)
    testloader_priv = torch.utils.data.DataLoader(test_priv, batch_size=batch_size_priv,
                                          shuffle=False, num_workers=2, drop_last=True)

    
    net = Net_new(net_depth,torch.device('cuda'))
    print(sum(p.numel() for p in net.X_net.parameters() if p.requires_grad))
    
    optim = torch.optim.Adam(net.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # wandb.watch(net)
    lr_schedule = LearnerRateScheduler('step', base_learning_rate=learning_rate, decay_rate=0.99, decay_steps=1)
    torch.cuda.empty_cache()
    time_start = time.time()
    train_model_priv(net, trainloader_priv, optim, num_epochs, h=0.82, rate=10, device=torch.device('cuda'), only_reg_flag=train_flag, lr_schedular=lr_schedule,lambda_loss=lambda_loss)
    # train_emb(net, trainloader_priv, criterion, optimizer, num_epochs=num_epochs,device=device,test_loader = test_emb_loader,test_total_loader = None,max_steps = 100)
    time_end = time.time()
    print("Time taken for training: ", time_end-time_start)
    # X_emb_train,losses_train = create_model_embs2(net,trainloader_priv,device= torch.device('cuda'),l=len(X_train),h=0.82)
    # X_emb_test,losses_test = create_model_embs2(net,testloader_priv,device= torch.device('cuda'),l=len(X_test),h=0.82)
    # print(losses_train.sum()/len(X_train))
    # print(losses_test.sum()/len(X_test))


    # Evaluate on test set
    # test_model_priv(net, testloader_priv, criterion, device=torch.device('cuda'))
    args.time = time_end-time_start
    wandb.config.update(args)
    print("Please type y or n if you want to save model: \n")
    input1 = input()
    if(input1 == 'y'):
        print("Please type the name of the model: \n")
        input2 = input()
        model_path = "Models/" + input2
        torch.save(net.state_dict(), model_path)
        args.model_name = model_path
        args.time = time_end-time_start
        wandb.config.update(args)
        print("Model saved successfully")
    # # # wandb.log_artifact(net)

#write a script to run the code

if __name__ == "__main__":
    main()

