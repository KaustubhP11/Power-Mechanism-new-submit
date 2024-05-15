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
import xgboost as xgb

from cov_help import *
import argparse

device_name = "cuda:1"
device = torch.device(device_name)
argparser = argparse.ArgumentParser(description='Forrest cover private testing')
argparser.add_argument('--eps', type=float, default=5.0,
                    help='Set epsilon for the model')
argparser.add_argument('--model_path', type=str, default='Models/cov_Net_new_512_100',
                    help='Path to the Model to create embeddings')

args = argparser.parse_args()

data_path = "./data/covtype.csv"
norm = 1
X,Y = cov_data_loader(data_path,norm=norm)
# max_dist = torch.cdist(X, X).max()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model_path = args.model_path
import os
def get_folders(directory):
    folders = []
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            folders.append(dir)
    return folders

directory_path = './Embeddings'

# Call the function to get the directory names
directory_names = get_folders(directory_path)

if(args.model_path[7:] in directory_names):
    print("Loading saved embeddings")
    dirpth =  './Embeddings/'+args.model_path[7:]
    X_emb_train = torch.load(dirpth+'/X_emb_train.pt')
    X_emb_test = torch.load(dirpth+'/X_emb_test.pt')
    losses_train = torch.load(dirpth+'/losses_train.pt')
    losses_test = torch.load(dirpth+'/losses_test.pt')      


losses_train,indices = torch.sort(losses_train)
set_eps = args.eps
ind = (losses_train < set_eps).sum()

    
    

X_emb_train_priv = X_emb_train[indices][:ind]
Y_train = Y_train[indices][:ind]
from sklearn.ensemble import RandomForestClassifier

# X, y = make_classification(n_samples=1000, n_features=4,
#                            n_informative=2, n_redundant=0,
#                            random_state=0, shuffle=False) 
clf = RandomForestClassifier(n_estimators=10,max_depth=10,  random_state=0)
clf.fit(X_emb_train_priv, Y_train)

print((clf.predict(X_emb_test) == np.asarray(Y_test)).sum()/len(Y_test))


from xgboost import XGBClassifier
clf = XGBClassifier()
clf.fit(X_emb_train_priv, Y_train)

print((clf.predict(X_emb_test) == np.asarray(Y_test)).sum()/len(Y_test))