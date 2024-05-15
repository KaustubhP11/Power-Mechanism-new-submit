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
parser.add_argument('--epochs', type=int, default=1, help='Number of epochs for training')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for training')
parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')
parser.add_argument('--lambda_loss', type=float, default=1, help='Lambda for loss function')
parser.add_argument('--only_reg_flag', type=int, default=0, help='Flag for only regularizer')
parser.add_argument('--max_steps', type=int, default=10000, help='Max steps for training')
args = parser.parse_args()
device = torch.device(args.device)
epochs = args.epochs

df = pd.read_csv('./Data/adult.csv')
wandb.init(project="income_prediction training")
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


x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# x_train_tensor = normalize2(x_train_tensor)
# x_test_tensor = normalize2(x_test_tensor)

import torch
import torch.nn as nn

# Define the model architecture

# Print the model architecture
print("\n \n** Formed Tensors and starting model training** \n \n ")
print(torch.cdist(x_train_tensor, x_train_tensor).max())
print(torch.cdist(x_test_tensor, x_test_tensor).max())
net = Net_new(1,device = device)
trainloader = torch.utils.data.DataLoader(list(zip(x_train_tensor, y_train_tensor)), batch_size=args.batch_size, shuffle=False)
torch.cuda.empty_cache()
# optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)


# train_model_priv(net,trainloader,x_test_tensor,y_test_tensor,optimizer,epochs,0.82,device= device,print_cond = True,only_reg_flag=0,lr_schedular =None,lambda_loss=args.lambda_loss,max_steps=args.max_steps)
# wandb.config.update(args)
# print("Please type y or n if you want to save model: \n")
# input1 = input()
# if(input1 == 'y'):
#     print("Please type the name of the model: \n")
#     input2 = input()
#     model_path = "Models/" + input2
#     torch.save(net.state_dict(), model_path)
#     args.model_name = model_path

#     wandb.config.update(args)
#     print("Model saved successfully")




