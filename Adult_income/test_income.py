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
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

import numpy as np
import pandas as pd
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:3', help='Device to train the model on')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for training')
parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')
parser.add_argument('--lambda_loss', type=float, default=1, help='Lambda for loss function')
parser.add_argument('--only_reg_flag', type=int, default=0, help='Flag for only regularizer')
parser.add_argument('--max_steps', type=int, default=10000, help='Max steps for training')
parser.add_argument('--model_path', type=str, default='Models/adult_512_1',
                    help='Path to the Model to create embeddings')
parser.add_argument('--net_depth',type=int,default= 1,
                    help='Depth of the network')
parser.add_argument('--eps', type=float, default=1.0,
                    help='Set epsilon for the model')
parser.add_argument('--seed',type=int,default= 58,
                    help='Seed for reproducibility')
args = parser.parse_args()
device = torch.device(args.device)
epochs = args.epochs

df = pd.read_csv('./Data/adult.csv')
wandb.init(project="income_prediction testing")
wandb.config.update(args)
print("**Starting data processing** \n \n ")


df['marital-status'] = df['marital-status'].apply(convert_marital_status)

df['native-country'] = df['native-country'].replace('Outlying-US(Guam-USVI-etc)' , 'US Minor Islands')

df = df.drop(['capital-gain', 'capital-loss', 'fnlwgt'], axis=1)

income_mapping = {'<=50K': 0, '>50K': 1}
df['income'] = df['income'].map(income_mapping)




df = fill_missing_categorical(df, 'native-country')
df = fill_missing_categorical(df, 'occupation')
df = fill_missing_categorical(df, 'workclass')

Q1 = df['age'].quantile(0.25)
Q3 = df['age'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['age'] < Q1 - 1.5 * IQR) | (df['age'] > Q3 + 1.5 * IQR)]

df.drop(outliers.index, inplace=True)
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
df.drop(['age', 'hours-per-week'], axis=1, inplace=True)
df.reset_index(inplace=True)
columns_to_keep = ['workclass', 'educational-num', 'marital-status', 'occupation', 'gender', 'native-country', 'income']
Features = df[columns_to_keep]
X= Features

X = pd.get_dummies(X, columns=['workclass', 'marital-status', 'occupation', 'native-country'])

X = X.drop(columns=['income'])
y = df['income']


scaler = StandardScaler()
X1 = scaler.fit_transform(X)
X1 =X1/np.linalg.norm(X1,axis =1).max()

X_fil = []
Y_fil = []

# ind  = y.values.sum()
ind = len(y.values)
counter = 0
print(ind)
for i in range(len(y.values)):
    if y.values[i] == 0:
        if counter < ind:
            X_fil.append(X1[i])
            Y_fil.append(y.values[i])
            counter+=1
    else:
        X_fil.append(X1[i])
        Y_fil.append(y.values[i])
        

import torch
x_train,x_test,y_train,y_test = train_test_split(X_fil,Y_fil,test_size = 0.2,random_state = 42)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# x_train_tensor = normalize2(x_train_tensor)
# x_test_tensor = normalize2(x_test_tensor)

import torch
import torch.nn as nn

print("\n \n ** Creating embeddings ** \n \n ",len(y_train_tensor))


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
    trainloader = torch.utils.data.DataLoader(list(zip(x_train_tensor, y_train_tensor)), batch_size=args.batch_size, shuffle=False)
    X_emb_train, losses_train = create_model_embs2(net,trainloader,device= device,l=len(y_train_tensor),h=0.8)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    # x_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    testloader = torch.utils.data.DataLoader(list(zip(x_test_tensor, y_test_tensor)), batch_size=args.batch_size, shuffle=False)
    X_emb_test, losses_test = create_model_embs2(net,testloader,device= device,l=len(y_test_tensor),h=0.8)
  
    torch.save( X_emb_train,dirpth+'/X_emb_train.pt')
    torch.save(X_emb_test,dirpth+'/X_emb_test.pt')
    torch.save(losses_train,dirpth+'/losses_train.pt')
    torch.save(losses_test,dirpth+'/losses_test.pt')


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
    nn.Linear(68, 64),  # Input layer with 100 input features and 64 output features
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
model = RandomForestClassifier(n_estimators=50,random_state=args.seed)

# Train the model
model.fit(X_emb_train_priv, Y_train)

# Make predictions
predictions = model.predict(X_emb_test)
acc_rf = (predictions == np.array(y_test_tensor)).sum()/(len(y_test_tensor))
model = XGBClassifier(device = 'cuda',random_state=args.seed,tree_method = 'hist',subsample=0.5)

model.fit(X_emb_train_priv, Y_train)

# Make predictions
predictions = model.predict(X_emb_test)

acc_xg = ((predictions == np.array(y_test_tensor)).sum()/(len(y_test_tensor)))
args.acc_rf = acc_rf
args.acc_xg = acc_xg
print("Accuracy of Random Forest: ", acc_rf, "Accuracy of XGBoost: ", acc_xg)
wandb.config.update(args)

    
