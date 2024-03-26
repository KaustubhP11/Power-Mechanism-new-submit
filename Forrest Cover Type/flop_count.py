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


class Net(nn.Module):
    def __init__(self):
        super(Net, self ).__init__()
        
        self.loss_reg = 0
        self.p = 1
        self.x = 0
        self.y = 0
        self.H_net1 = nn.Sequential(
            nn.Linear(54, 128),
            nn.Sigmoid(),
            nn.Linear(128, 256),
            nn.Sigmoid(),
            nn.Linear(256, 54*54).cuda()
        )
        self.X_net = nn.Sequential(
            nn.Linear(54, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 7),
            nn.Softmax(dim=1)

        )
        
    def forward(self, x):
        self.x = x
        self.H = self.H_net1(x)
        self.H = self.H.reshape(54,54)
        x = torch.mm(self.H,torch.transpose(x,0,1))
        x = torch.transpose(x,0,1)
        self.y = self.X_net(x)
        return self.y

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self ).__init__()
        
        self.loss_reg = 0
        self.p =1
        self.x = 0
        self.y = 0
        self.H_net1 = nn.Sequential(
            nn.Linear(18, 128),
            nn.Sigmoid(),
            nn.Linear(128, 256),
            nn.Sigmoid(),
            nn.Linear(256, 18*18).cuda()
        )
        self.H_net2 = nn.Sequential(
            nn.Linear(18, 128),
            nn.Sigmoid(),
            nn.Linear(128, 256),
            nn.Sigmoid(),
            nn.Linear(256, 18*18).cuda()
        )
        self.H_net3 = nn.Sequential(
            nn.Linear(18, 128),
            nn.Sigmoid(),
            nn.Linear(128, 256),
            nn.Sigmoid(),
            nn.Linear(256, 18*18).cuda()
        )
        self.X_net = nn.Sequential(
            nn.Linear(54, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 7),
            nn.Softmax(dim=2)

        )
        
    def forward(self, x):
        
       
        x.requires_grad =True
        # p = self.p
        self.x = x
        d = x.shape[1]
        bs = x.shape[0]
        x= torch.unsqueeze(x,1)
        z = x.cuda()
        z1 = z[:,:,0:18]
        z2 = z[:,:,18:36]
        z3 = z[:,:,36:54]
        loss_reg = torch.zeros(bs,d).cuda()

        H1 = self.H_net1(z1).cuda()
        H2 = self.H_net2(z2).cuda()
        H3 = self.H_net3(z3).cuda()
        H1 = H1.reshape(bs,18,18)
        H2 = H2.reshape(bs,18,18)
        H3 = H3.reshape(bs,18,18)

        z1 = torch.matmul(z1,H1).cuda()
        z2 = torch.matmul(z2,H2).cuda()
        z3 = torch.matmul(z3,H3).cuda()
        z = torch.cat((z1,z2,z3),dim=2).cuda()
        
    
        self.y = z
        y = self.X_net(z)
        return y.squeeze(dim=1)

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
nn.Sequential(
            nn.Linear(54, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 7),
            nn.Softmax(dim=1)

        )
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(54, 64)
        self.fc2 = nn.Linear(64, 128)
        # self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 256)
        self.fc5 = nn.Linear(256, 128)
        # self.fc6 = nn.Linear(64, 64)
        self.fc7 = nn.Linear(128, 7)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        # x = self.relu(self.fc3(x)) 
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x)) 
        # x = self.relu(self.fc6(x))
        x = self.softmax(self.fc7(x))
        return x