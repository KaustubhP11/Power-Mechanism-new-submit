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
from tqdm import tqdm
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

import argparse
import warnings
import wandb



class Net_new(nn.Module):
    def __init__(self,p,device=torch.device('cuda')):
        super(Net_new, self ).__init__()
        self.device = device
        self.loss_reg = 0
        self.p =p 
        self.x = 0
        self.y = 0
        self.H_net1 = nn.Sequential(
            nn.Linear(68, 128),
            nn.Sigmoid(),
            nn.Linear(128, 64),
            nn.Sigmoid(),
            nn.Linear(64, 68*68).to(device)
        )
        self.X_net = nn.Sequential(
            nn.Linear(68, 1),
            # nn.ReLU(),
            # nn.Linear(128, 512),
            # nn.ReLU(),
            # nn.Linear(512, 128),
            # nn.ReLU(),
            # nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        def H_mul(z):
            H12 = self.H_net1(z)
            H12= H12.reshape(z.shape[0],d,d)
            x12 = torch.matmul(z,H12)
            return(x12)
    
        
        def batch_jacobian(func, z, create_graph=False):
            # x in shape (Batch, Length)
            def _func_sum(z):
                return func(z).sum(dim=0)
            return torch.squeeze(torch.autograd.functional.jacobian(_func_sum, z, create_graph=create_graph)).permute(1,0,2)
        
        
        device = self.device
       
        x.requires_grad =True
        p = self.p
        self.x = x
        d = x.shape[1]
        bs = x.shape[0]
        x= torch.unsqueeze(x,1)
        z = x.to(device)

        loss_reg = torch.zeros(bs,d).to(device)
        for i in range(p):
            H = self.H_net1(z).to(device)
            H = H.reshape(bs,d,d)
            z = torch.matmul(z,H).to(device)
            J = batch_jacobian(H_mul, z, create_graph=True)
            J_int =-torch.log(torch.abs(torch.det(J)))
            loss_reg = loss_reg + torch.squeeze(torch.autograd.grad(J_int, x,torch.ones_like(J_int),allow_unused=True,create_graph= True)[0]).to(device)
        self.loss_reg = loss_reg
        self.y = z
        y = self.X_net(z)
        return y



        
        
        



     


def gau_ker(u,device = torch.device('cuda')):
    return torch.pow(2*torch.tensor(torch.pi),u.shape[1]/(-2))*torch.exp(torch.bmm(u.view(u.shape[0], 1, u.shape[1]), u.view(u.shape[0],  u.shape[1],1))/(-2)).to(device)


def py_kde(x,X_t,h,device = torch.device('cuda')):
    norm = X_t.shape[0]*(h**x.shape[1])
    prob = torch.zeros(x.shape[0]).to(device)
    for i in range(len(X_t)):
        prob+= (torch.squeeze(gau_ker((x - X_t[i])/h))/norm).to(device)
    return(prob)


def py_kde_der(p_x,x,device = torch.device('cuda')):
    # x.requires_grad = True
    # p_x = py_kde(x,X_t,h)
    return (torch.autograd.grad(p_x,x,torch.ones_like(p_x),allow_unused=True,create_graph=True)[0]).to(device)


def gau_ker_der(X,h):
    N= X.shape[0]
    d = X.shape[1]
    grad = torch.zeros(X.shape)
    for n in range(N):
        for i in range(d):
            for j in range(N):
                grad[n][i]+= torch.exp(-1*torch.dot((X[n]-X[j]),(X[n]-X[j]))/(2*h*h))*(X[n][i] -X[j][i]) /(N*(h**(d+2))*((2*math.pi)**(d/2)))

    return grad


def normalize2(x,norm=1):
    n = torch.norm(x,dim=1).max()
    x_normed =norm*x /(n)
    return x_normed


import torch
import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear(x)
        out = self.sigmoid(out)
        return out


def train_model_priv(net,trainloader,x_test,y_test,optimizer,epochs,h,rate=10,device= torch.device('cuda'),print_cond = True,only_reg_flag=0,lr_schedular =None,lambda_loss=1,max_steps =10000):
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    lr = lr_schedular
    net = net.to(device)
    net.device = device
    acc = 0
    
    criterion = nn.BCELoss()
    for epoch in (pbar:= tqdm(range(epochs))):  # loop over the dataset multiple times
        # scheduler.step()
        running_loss = 0.0
        running_loss_reg = 0.0
        if(lr):
       
            for groups in optimizer.param_groups:
                groups['lr'] = lr(epoch)
        # optimizer.param_groups[0]['lr'] = lr(epoch)
        
        
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            if(i >max_steps ):
                break
            bs = len(data[0])
            
            inputs = data[0].to(device)
            inputs.requires_grad = True
            labels = data[1].to(device)
            f = py_kde(inputs,inputs,h,device = device)
            f_der = py_kde_der(f,inputs,device = device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            if(only_reg_flag==1):
                loss = torch.norm(f_der/f.view(f.shape[0],1)+ net.loss_reg,dim=1).sum()
            elif(only_reg_flag==2):
                loss = criterion(torch.squeeze(outputs),torch.squeeze(labels))
                
            else:
               
                loss = lambda_loss*bs*criterion(torch.squeeze(outputs),torch.squeeze(labels)) + torch.norm(f_der/f.view(f.shape[0],1)+ net.loss_reg,dim=1).sum()
            loss.backward(retain_graph=True)

            optimizer.step()
            loss = loss.detach().cpu()/len(inputs)

          
            if(epoch ==0 and i==0):
                continue
            loss_reg = torch.norm(f_der/f.view(f.shape[0],1)+ net.loss_reg,dim=1).sum().detach().cpu()/len(inputs)
            
            
            wandb.log({"loss": loss.item(),"loss_reg":loss_reg.item()})
            pbar.set_postfix_str(f"loss: {loss.item()}, loss_reg: {loss_reg.item()},Acc: {acc}")

            # print statistics
            # print(loss.sum().shape)
            
            # if i % 100 == 99:    # print every 2000 mini-batches
            # if((i+1)%rate==0):
        
        outputs = net(x_test)
        acc = ((outputs>0.5).squeeze().cpu() == y_test.squeeze()).sum()/(len(y_test))
    print("Final Accuracy: ",acc)

        
        
        # wandb.log({"test acc": acc})
            
        



        
            
            




def create_model_embs2(net,trainloader,device= torch.device('cpu'),l=0,h=0.82):
    alpha =1/l;
    
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    X_emb = torch.zeros(l,68)
    losses = torch.zeros(l)
    bs = trainloader.batch_size


    net = net.to(device)
    # criterion = nn.CrossEntropyLoss()

        
        
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
   
        inputs = data[0].to(device)
        n = len(inputs)
        d = inputs.shape[1]
        inputs.requires_grad = True
        labels = data[1].to(device)
        f = py_kde(inputs,inputs,h,device=device)


        f_der = py_kde_der(f,inputs,device=device)
      
        ci = CI_KDE(f,n,h,d,alpha,device=device)

        output =  net(inputs)
        
        loss =torch.max(torch.linalg.norm(f_der/(f-ci).view(f.shape[0],1)+net.loss_reg,dim=1),torch.linalg.norm(f_der/(f+ci).view(f.shape[0],1)+net.loss_reg,dim=1)) 
        try:
            losses[i*bs:i*bs+len(loss)] =loss.detach().cpu()
        except:
            print(loss.detach().cpu().shape)
            print(len(loss))
            print(net.y.detach().cpu().shape)
            print(i*bs)
            print(X_emb[i*bs:i*bs+len(loss)].shape)
        X_emb[i*bs:i*bs+len(loss)] = torch.squeeze(net.y.detach().cpu())
    return(X_emb,losses)

def train_emb(model, train_loader, x_test,y_test,loss_fn, optimizer, num_epochs=10,device=torch.device('cpu'),test_total_loader = None,max_steps =10000):
    running_loss = 0.0
    counter = 0
    max_test_acc =0.0
    model = model.to(device)
    steps = 0
    acc = 0
    max_acc =0
    for epoch in (pbar:= tqdm(range(num_epochs))):
        
        
        for i, data in enumerate(train_loader, 0):
            
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = loss_fn(outputs, labels)
            
            loss.backward()
            
            optimizer.step()
            running_loss += loss.item()
            wandb.log({"loss": loss.item()})
            pbar.set_postfix_str(f"loss: {loss.item()}, acc: {acc}")
            
            steps+=1
            
            # counter =0
        # if((epoch+1)%10==0):
            # print('Epoch [%d], loss: %.3f' % (epoch + 1, running_loss /(10* len(train_loader))))
            # running_loss = 0.0
        # for params in model.parameters():
        #     print(params.grad)
        with torch.no_grad():
            x_test = x_test.to(device)
            outputs = model(x_test).cpu()
            
            acc = ((outputs>0.5).squeeze().cpu() == y_test.squeeze()).sum()/(len(y_test))
            
        wandb.log({"test acc": acc})
        if(acc>max_acc):
            max_acc = acc
        wandb.log({"max acc": max_acc})
        
       
        
        



def convert_marital_status(status):
    if status in ['Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse']:
        return 'married'
    elif status in ['Never-married', 'Separated', 'Widowed']:
        return 'single'
    else:
        return 'divorced'
    
def fill_missing_categorical(df, column):
    df[column] = df[column].replace('?', np.nan)

    if df[column].notna().all():
        return df

    known = df[df[column].notna()]
    unknown = df[df[column].isna()]

    le = LabelEncoder()
    known[column] = le.fit_transform(known[column])
    X_known = known.drop(column, axis=1)
    y_known = known[column]

    categorical_cols = X_known.select_dtypes(include=['object']).columns

    le_cat = preprocessing.LabelEncoder()
    X_known[categorical_cols] = X_known[categorical_cols].apply(lambda col: le_cat.fit_transform(col.astype(str)))

    clf = RandomForestClassifier()
    clf.fit(X_known, y_known)

    X_unknown = unknown.drop(column, axis=1)

    X_unknown[categorical_cols] = X_unknown[categorical_cols].apply(lambda col: le_cat.fit_transform(col.astype(str)))

    unknown[column] = clf.predict(X_unknown)

    df = pd.concat([known, unknown], axis=0)

    df[column] = le.inverse_transform(df[column])
    
    return df



def CI_KDE(p_x,n,h,d,alpha,device = torch.device('cuda')):
    return( stats.norm.ppf(1-alpha/2)*torch.sqrt(p_x/((2**d)*math.sqrt(torch.pi**d)*n*h**(d))).to(device) )

def CI_KDE_der(p_x_der,p_x,n,h,d,alpha,device = torch.device('cuda')):
    return( p_x_der*stats.norm.ppf(1-alpha/2)*torch.sqrt(1/(p_x.unsqueeze(dim=1)*(2**d)*math.sqrt(torch.pi**d)*n*h**(d))).to(device) )