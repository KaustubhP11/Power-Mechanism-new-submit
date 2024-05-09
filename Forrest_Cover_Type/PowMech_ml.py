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
device_name = "cuda:1"
device = torch.device(device_name)
data_path = "./data/covtype.csv"
norm = 1
X,Y = cov_data_loader(data_path,norm=norm)
# max_dist = torch.cdist(X, X).max()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model_path = "Models/cov_Net_new_128_100"
model_embs_path = model_path.replace("Models","Embeddings")
if (model_path in ['Models/cov_full_expt_512_1','Models/cov_full_expt_512_100','Models/cov_full_expt_512_400','Models/cov_Net_new_512_100','Models/cov_Net_new_128_100']):
    X_emb_train = torch.load(model_embs_path + "/X_emb_train.pt")
    X_emb_test = torch.load(model_embs_path + "/X_emb_test.pt")
    losses_train = torch.load(model_embs_path + "/losses_train.pt")
    losses_test = torch.load(model_embs_path + "/losses_test.pt")        


losses_train,indices = torch.sort(losses_train)
set_eps = 0.5
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