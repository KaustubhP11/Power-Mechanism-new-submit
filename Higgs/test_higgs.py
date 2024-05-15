from utils import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from utils import *
import numpy as np
import pandas as pd
import argparse
import wandb
import numpy as np
import pandas as pd
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:3', help='Device to train the model on')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for training')
parser.add_argument('--batch_size', type=int, default=4096, help='Batch size for training')
parser.add_argument('--batch_size_emb', type=int, default=4096, help='Batch size for embedding')
parser.add_argument('--lambda_loss', type=float, default=1, help='Lambda for loss function')
parser.add_argument('--only_reg_flag', type=int, default=0, help='Flag for only regularizer')
parser.add_argument('--max_steps', type=int, default=10000, help='Max steps for training')
parser.add_argument('--model_path', type=str, default='Models/higg_4096_1',
                    help='Path to the Model to create embeddings')
parser.add_argument('--net_depth',type=int,default= 1,
                    help='Depth of the network')
parser.add_argument('--eps', type=float, default=1.0,
                    help='Set epsilon for the model')
                    
args = parser.parse_args()
device = torch.device(args.device)
epochs = args.epochs

wandb.init(project="Higgs boson testing")
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
import os
def get_folders(directory):
    folders = []
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            folders.append(dir)
    return folders

# Specify the directory path
directory_path = './Embeddings'

# Call the function to get the directory names
directory_names = get_folders(directory_path)

if(args.model_path[7:] in directory_names):
    dirpth =  './Embeddings/'+args.model_path[7:]
    X_emb_train = torch.load(dirpth+'/X_emb_train.pt')
    X_emb_test = torch.load(dirpth+'/X_emb_test.pt')
    losses_train = torch.load(dirpth+'/losses_train.pt')
    losses_test = torch.load(dirpth+'/losses_test.pt')
    
else:
    print("\n \n ** Creating embeddings ** \n \n ",len(y_train_tensor))
    dirpth =  './Embeddings/'+args.model_path[7:]
    
    os.mkdir(dirpth)
    net = Net_new(args.net_depth,device=device)
    state_dict = torch.load(args.model_path)
    

    # Create an instance of Net
    net = Net_new(args.net_depth,device=device)
    net.load_state_dict(state_dict)
    trainloader = torch.utils.data.DataLoader(list(zip(x_train_tensor, y_train_tensor)), batch_size=args.batch_size_emb, shuffle=False)
    X_emb_train, losses_train = create_model_embs2(net,trainloader,device= device,l=len(y_train_tensor),h=0.8)
    x_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    # x_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    testloader = torch.utils.data.DataLoader(list(zip(x_test_tensor, y_test_tensor)), batch_size=args.batch_size_emb, shuffle=False)
    X_emb_test, losses_test = create_model_embs2(net,testloader,device= device,l=len(y_test_tensor),h=0.8)
  
    torch.save( X_emb_train,dirpth+'/X_emb_train.pt')
    torch.save(X_emb_test,dirpth+'/X_emb_test.pt')
    torch.save(losses_train,dirpth+'/losses_train.pt')
    torch.save(losses_test,dirpth+'/losses_test.pt')
    

# # Create the directory
#     if not os.path.exists(dirpth):
#         os.mkdir(dirpth)
#     X_emb_train = torch.save(dirpth+'/X_emb_train.pt')
#     X_emb_test = torch.save(dirpth+'/X_emb_test.pt')
#     losses_train = torch.save(dirpth+'/losses_train.pt')
#     losses_test = torch.saved(dirpth+'/losses_test.pt')
    
losses_train,indices = torch.sort(losses_train)

set_eps = args.eps
ind = (losses_train < set_eps).sum()



X_emb_train_priv = X_emb_train[indices][:ind]
Y_train = y_train_tensor[indices][:ind]

# input_size = 68
# model = LogisticRegression(input_size)
# Define the model architecture
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
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
trainloader_priv = torch.utils.data.DataLoader(list(zip(X_emb_train_priv, Y_train)), batch_size=args.batch_size, shuffle=True)
print("\n \n ** Starting final training ** \n \n ")
torch.cuda.empty_cache()
train_emb(model,trainloader_priv,X_emb_test,y_test_tensor,nn.BCELoss(),optimizer,num_epochs,device=device,test_total_loader = None,max_steps =args.max_steps)
model = RandomForestClassifier(n_estimators=50)

# Train the model
model.fit(X_emb_train_priv, Y_train)

# Make predictions
predictions = model.predict(X_emb_test)
acc_rf = (predictions == np.array(y_test_tensor)).sum()/(len(y_test_tensor))
model = XGBClassifier()

model.fit(X_emb_train_priv, Y_train)

# Make predictions
predictions = model.predict(X_emb_test)

acc_xg = ((predictions == np.array(y_test_tensor)).sum()/(len(y_test_tensor)))
args.acc_rf = acc_rf
args.acc_xg = acc_xg
print("Accuracy of Random Forest: ", acc_rf, "Accuracy of XGBoost: ", acc_xg)
wandb.config.update(args)