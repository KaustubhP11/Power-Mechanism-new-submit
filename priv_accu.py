# Imports
import numpy as np
import pandas as pd
# import dcMinMaxFunctions as dc
# import dcor
from scipy.misc import derivative
from sklearn.model_selection import train_test_split
import math
import torch
from scipy import stats
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F


# class Net(nn.Module):
#     def __init__(self,p):
#         super(Net, self ).__init__()
        
#         self.loss_reg = 0
#         self.p =p 
#         self.x = 0
#         self.y = 0
#         self.H_net1 = nn.Sequential(
#             nn.Linear(12, 48),
#             nn.Sigmoid(),
#             nn.Linear(48, 24),
#             nn.Sigmoid(),
#             nn.Linear(24, 144).cuda()

#         )
#         self.X_net = nn.Sequential(
#             nn.Linear(12, 16),
#             nn.ReLU(),
#             # nn.Linear(48, 48),
#             # nn.ReLU(),
#             nn.Linear(16, 1),
#             nn.Sigmoid()

#         )
        
#     def forward(self, x):
#         def H_mul(z):
#             H12 = self.H_net1(z)
#             H12= H12.reshape(z.shape[0],d,d)
#             x12 = torch.matmul(z,H12)
#             return(x12)
    
        
#         def batch_jacobian(func, z, create_graph=False):
#             # x in shape (Batch, Length)
#             def _func_sum(z):
#                 return func(z).sum(dim=0)
#             return torch.squeeze(torch.autograd.functional.jacobian(_func_sum, z, create_graph=create_graph)).permute(1,0,2)
        
#         x.requires_grad =True
#         p = self.p
#         self.x = x
#         d = x.shape[1]
#         bs = x.shape[0]
#         x= torch.unsqueeze(x,1)
#         z = x.cuda()
#         loss_reg = torch.zeros(bs,d).cuda()
#         for i in range(p):
#             H = self.H_net1(z).cuda()
#             H = H.reshape(bs,d,d)
#             z = torch.matmul(z,H).cuda()
#             J = batch_jacobian(H_mul, z, create_graph=True)
#             J_int =-torch.log(torch.abs(torch.det(J)))
#             loss_reg = loss_reg + torch.squeeze(torch.autograd.grad(J_int, x,torch.ones_like(J_int),allow_unused=True,create_graph= True)[0]).cuda()
#         self.loss_reg = loss_reg
#         self.y = z
#         y = self.X_net(z)
        # return y

class Net(nn.Module):
    def __init__(self,p):
        super(Net, self ).__init__()
        
        self.loss_reg = 0
        self.p =p 
        self.x = 0
        self.y = 0
        self.H_net1 = nn.Sequential(
            nn.Linear(12, 48),
            nn.Sigmoid(),
            nn.Linear(48, 24),
            nn.Sigmoid(),
            nn.Linear(24, 144).cuda()

        )
        self.X_net = nn.Sequential(
            nn.Linear(12, 16),
            nn.ReLU(),
            # nn.Linear(48, 48),
            # nn.ReLU(),
            nn.Linear(16, 1),
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
        
        x.requires_grad =True
        p = self.p
        self.x = x
        d = x.shape[1]
        bs = x.shape[0]
        x= torch.unsqueeze(x,1)
        z = x.cuda()
        loss_reg = torch.zeros(bs,d).cuda()
        for i in range(p):
            H = self.H_net1(z).cuda()
            H = H.reshape(bs,d,d)
            z = torch.matmul(z,H).cuda()
            J = batch_jacobian(H_mul, z, create_graph=True)
            J_int =-torch.log(torch.abs(torch.det(J)))
            loss_reg = loss_reg + torch.squeeze(torch.autograd.grad(J_int, x,torch.ones_like(J_int),allow_unused=True,create_graph= True)[0]).cuda()
        self.loss_reg = loss_reg
        self.y = z
        y = self.X_net(z)
        return y

def OHE(x):
    dim = np.max(x)
    y = np.zeros((len(x),dim+1))
    for i in range(len(x)):
        y[i][x[i]] = 1
    return(y)
def normalize(x):
    x_normed = x / x.max(0, keepdim=True)[0]
    return x_normed

def gau_ker(u):
    return torch.pow(2*torch.tensor(torch.pi),u.shape[1]/(-2))*torch.exp(torch.bmm(u.view(u.shape[0], 1, u.shape[1]), u.view(u.shape[0],  u.shape[1],1))/(-2))


def py_kde(x,X_t,h):
    norm = X_t.shape[0]*(h**x.shape[1])
    prob = torch.zeros(x.shape[0]).cuda()
    for i in range(len(X_t)):
        prob+= (torch.squeeze(gau_ker((x - X_t[i])/h))/norm).cuda()
    return(prob)


def py_kde_der(p_x,x):
    # x.requires_grad = True
    # p_x = py_kde(x,X_t,h)
    return (torch.autograd.grad(p_x,x,torch.ones_like(p_x),allow_unused=True,create_graph=True)[0]).cuda()


def gau_ker_der(X,h):
    N= X.shape[0]
    d = X.shape[1]
    grad = torch.zeros(X.shape)
    for n in range(N):
        for i in range(d):
            for j in range(N):
                grad[n][i]+= torch.exp(-1*torch.dot((X[n]-X[j]),(X[n]-X[j]))/(2*h*h))*(X[n][i] -X[j][i]) /(N*(h**(d+2))*((2*math.pi)**(d/2)))

    return grad
def CI_KDE(p_x,n,h,d,alpha):
    return( stats.norm.ppf(1-alpha/2)*torch.sqrt(p_x/((2**d)*math.sqrt(torch.pi**d)*n*h**(d))) )

def CI_KDE_der(p_x_der,p_x,n,h,d,alpha):
    return( p_x_der*stats.norm.ppf(1-alpha/2)*torch.sqrt(1/(p_x.unsqueeze(dim=1)*(2**d)*math.sqrt(torch.pi**d)*n*h**(d))) )



def main():

    net =Net(5)
    df=pd.read_csv("data/Churn_Modelling.csv")
    # df=df.drop(['duration', 'pdays'],axis=1) # duration gives away the answer, and pdays has too much missing info

    X = df.loc[:, df.columns != 'Exited'].replace(dict(yes=True, no=False))
    Y = df.loc[:, ['Exited']].replace(dict(yes=True, no=False))
    categorical_columns = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']
    numerical_columns = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    outputs = ['Exited']
    for category in categorical_columns:
        df[category] = df[category].astype('category')
    
    geo = OHE(df['Geography'].cat.codes.values)
    gen =  np.asarray(df['Gender'].cat.codes.values)
    hcc =  np.asarray(df['HasCrCard'].cat.codes.values)
    iam =  np.asarray(df['IsActiveMember'].cat.codes.values)

    categorical_data = np.stack(( gen, hcc, iam), axis=1)
    # categorical_data = torch.tensor(categorical_data, dtype=torch.int64)
    numerical_data = np.stack([df[col].values for col in numerical_columns], 1)
    # numerical_data = torch.tensor(numerical_data, dtype=torch.float)
    X = np.concatenate((numerical_data, categorical_data,geo), axis=1)
    Y = df[outputs].values
    # outputs = torch.tensor(df[outputs].values).flatten()
    X = torch.Tensor(X)
    Y = torch.Tensor(Y)
    X = normalize(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    data=X

    m1 = torch.load("Models/net_5_5.pth")

    losses= torch.zeros(X.shape[0])
    bs = 1000
    n = bs
    h = 0.65
    d = X.shape[1]
    alpha= 0.0001
    for ct in range(0,len(X),bs):
        x = data[ct:bs+ct].detach()
        x_hat = m1(x)
        f = py_kde(x,x,0.65)
        f_der = py_kde_der(f,x)
        ci = CI_KDE(f,n,h,d,alpha)
        ci_der = CI_KDE_der(f_der,f,n,h,d,alpha)
        loss =torch.max(torch.linalg.norm(f_der/(f-ci).view(f.shape[0],1)+m1.loss_reg,dim=1),torch.linalg.norm(f_der/(f+ci).view(f.shape[0],1)+m1.loss_reg,dim=1)) + torch.norm(ci_der/f.view(-1,1),dim=1)
        losses[ct:bs+ct] =loss
    losses,indices = torch.sort(losses)
    X = X[indices]
    Y = Y[indices]
    eps_list = [1,1.25,1.5,2,2.5,3,3.5,4.5]
    acc_list =[]
    for eps in eps_list:
        set_eps = eps
        ind =(losses*2.7 < set_eps).sum()
        l = len(X)
        # indices = torch.randperm(X.shape[0])
        # X = X[indices]
        # Y = Y[indices]

        # X_train, X_test, Y_train, Y_test = train_test_split(X[0:ind], Y[0:ind], test_size=0.2)
        # o = m1(X.detach())
        # X_emb = torch.squeeze(m1.y).detach()
        # o = m1(X_train.detach())
        # X_emb_train = torch.squeeze(m1.y).detach()
        # o = m1(X_test.detach())
        # X_emb_test = torch.squeeze(m1.y).detach()
        # o = m1(X[ind:-1].detach())
        # X_emb_test2 = torch.squeeze(m1.y).detach()
        # Y_test2 = Y[ind:-1]

        X_train, X_test, Y_train, Y_test = train_test_split(X[-ind:-1], Y[-ind:-1], test_size=0.2)
        o = m1(X.detach())
        X_emb = torch.squeeze(m1.y).detach()
        o = m1(X_train.detach())
        X_emb_train = torch.squeeze(m1.y).detach()
        o = m1(X_test.detach())
        X_emb_test = torch.squeeze(m1.y).detach()
        o = m1(X[0:len(X) - ind].detach())
        X_emb_test2 = torch.squeeze(m1.y).detach()
        Y_test2 = Y[0:len(X) - ind]
        
        model = nn.Sequential(
            nn.Linear(12, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            # nn.Linear(64, 64),
            # nn.ReLU(),
            # nn.Linear(64, 32),
            # nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(

            )
        )
        model = model.cuda()
        loss_fn = nn.BCELoss()  # binary cross entropy
        # import torch.optim as optim
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay= 1e-4)
        n_epochs = 100
        batch_size = 1000

        for epoch in range(n_epochs):
            for i in range(0, len(X_emb_train), batch_size):
                Xbatch = X_emb_train[i:i+batch_size]
                y_pred = model(Xbatch)
                ybatch = Y_train[i:i+batch_size]
                loss = loss_fn(y_pred.cuda(), ybatch.cuda())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            #     print(loss)
            # print(f'Finished epoch {epoch}, latest loss {loss}')
        # compute accuracy (no_grad is optional)
            # if(epoch%100==99):
            #     with torch.no_grad():
        y_pred = model(X_emb_train)
        accuracy1 = (y_pred.round() == Y_train.cuda()).cpu().float().mean().numpy()
        # print(f"Train Accuracy {accuracy}")
        y_pred = model(X_emb_test)
        accuracy2 = (y_pred.round() == Y_test.cuda()).cpu().float().mean().numpy()
        # print(f"Test Accuracy {accuracy}")
        y_pred = model(X_emb_test2)
        accuracy3 = (y_pred.round() == Y_test2.cuda()).cpu().float().mean().numpy()

        y_pred = model(X_emb)
        accuracy4 = (y_pred.round() == Y.cuda()).cpu().float().mean().numpy()
        # print(f"Test Accuracy 2 {accuracy}")
        acc_list.append([str(eps),str(accuracy1),str(accuracy2),str(accuracy3),str(accuracy4),str(ind)])
        # for row in acc_list:
        #     print('\t'.join(row))
            # try:
            #     print('\t'.join(row))
            # except TypeError:
            #     print(row)
    for row in acc_list:
            print('\t'.join(row))
    # print("Accuracies for epsilons ",eps_list," are ",acc_list)

if __name__ == "__main__":
    main()
    