import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn import preprocessing
from utils import *
import argparse
import warnings
import wandb

import numpy as np
import pandas as pd
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:3', help='Device to train the model on')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for training')
parser.add_argument('--batch_size', type=int, default=4096, help='Batch size for training')
parser.add_argument('--lambda_loss', type=float, default=1, help='Lambda for loss function')
parser.add_argument('--only_reg_flag', type=int, default=0, help='Flag for only regularizer')
parser.add_argument('--max_steps', type=int, default=10000, help='Max steps for training')

parser.add_argument('--net_depth',type=int,default= 1,
                    help='Depth of the network')
parser.add_argument('--eps', type=float, default=1.0,
                    help='Set epsilon for the model')
                    
args = parser.parse_args()
device = torch.device(args.device)
epochs = args.epochs
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
trainloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor), batch_size=args.batch_size,
                                          shuffle=True, num_workers=2, drop_last=True)
import torch
import torch.nn as nn
num_epochs = epochs
model = nn.Sequential(
    nn.Linear(33, 64),  # Input layer with 100 input features and 64 output features
    nn.ReLU(),  # Activation function
    nn.Linear(64, 128),  # Hidden layer with 64 input features and 32 output features
    nn.ReLU(), 
    nn.Linear(128,32),  # Hidden layer with 64 input features and 32 output features
    nn.ReLU(),# Activation function
    nn.Linear(32, 1),
    nn.Sigmoid()# Output layer with 32 input features and 10 output features
)
from opacus import PrivacyEngine
privacy_engine = PrivacyEngine()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
grad_norm = 10
# model2, optimizer2, data_loader = privacy_engine.make_private_with_epsilon(
#     module=model,
#     optimizer=optimizer,
#     data_loader=trainloader,
#     target_epsilon=args.eps,
#     target_delta =0.0001,
#     epochs = num_epochs,
#     max_grad_norm=grad_norm,
# )
wandb.init(project="higgs priv baseline")
wandb.config.update(args)
# train_emb(model2,data_loader,x_test_tensor,y_test_tensor,nn.BCELoss(),optimizer2,num_epochs,device=device,max_steps =args.max_steps)
train_emb(model,trainloader,x_test_tensor,y_test_tensor,nn.BCELoss(),optimizer,num_epochs,device=device,max_steps =args.max_steps)
args.grad_norm = grad_norm
# Train the model

wandb.config.update(args)

