from utils import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from utils import *
import numpy as np
import pandas as pd
import argparse
import wandb
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:3', help='Device to train the model on')
parser.add_argument('--epochs', type=int, default=1, help='Number of epochs for training')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for training')
parser.add_argument('--batch_size', type=int, default=4096, help='Batch size for training')
parser.add_argument('--lambda_loss', type=float, default=1, help='Lambda for loss function')
parser.add_argument('--only_reg_flag', type=int, default=0, help='Flag for only regularizer')
parser.add_argument('--max_steps', type=int, default=10000, help='Max steps for training')
parser.add_argument('--net_depth',type=int,default= 1,
                    help='Depth of the network')
args = parser.parse_args()
device = torch.device(args.device)
epochs = args.epochs


wandb.init(project="Higgs boson training")
wandb.config.update(args)
print("**Starting data processing** \n \n ")
data=pd.read_csv('data/training.csv')
df_data = data.drop(columns=['EventId','Weight'])
X = df_data.drop(columns=['Label'])  # Features

X = np.asarray(pd.get_dummies(X, columns=['PRI_jet_num'], prefix='PRI_jet_num').values, dtype = np.float32)
X = X/np.linalg.norm(X, axis=1).max()
y = df_data['Label'].values  # Target
y = np.where(y == 's', 1, 0)
y = np.asarray(y, dtype = np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import torch

# Convert X_train and X_test to tensors
x_train_tensor = torch.tensor(X_train)
x_test_tensor = torch.tensor(X_test)

# Convert y_train and y_test to tensors
y_train_tensor = torch.tensor(y_train)
y_test_tensor = torch.tensor(y_test)
import torch
import torch.nn as nn

# Define the model architecture

# Print the model architecture
print("\n \n** Formed Tensors and starting model training** \n \n ")
net = Net_new(args.net_depth,device = device)
trainloader = torch.utils.data.DataLoader(list(zip(x_train_tensor, y_train_tensor)), batch_size=args.batch_size, shuffle=False)
torch.cuda.empty_cache()
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)


train_model_priv(net,trainloader,x_test_tensor,y_test_tensor,optimizer,epochs,0.82,device= device,print_cond = True,only_reg_flag=0,lr_schedular =None,lambda_loss=args.lambda_loss,max_steps=args.max_steps)
wandb.config.update(args)
print("Please type y or n if you want to save model: \n")
input1 = input()
if(input1 == 'y'):
    print("Please type the name of the model: \n")
    input2 = input()
    model_path = "Models/" + input2
    torch.save(net.state_dict(), model_path)
    args.model_name = model_path

    wandb.config.update(args)
    print("Model saved successfully")