# Imports
import numpy as np
import pandas as pd
# import dcMinMaxFunctions as dc
# import dcor
from scipy.misc import derivative
from sklearn.model_selection import train_test_split
import math
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus import PrivacyEngine

def normalize(x):
    x_normed = x / x.max(0, keepdim=True)[0]
    return x_normed



def OHE(x):
    dim = max(x)
    y = np.zeros((len(x),dim+1))
    for i in range(len(x)):
        y[i][x[i]] = 1
    return(y)
#write a script to load data from data folder and train model


def gau_ker(u):
    return torch.pow(2*torch.tensor(torch.pi),u.shape[1]/(-2))*torch.exp(torch.bmm(u.view(u.shape[0], 1, u.shape[1]), u.view(u.shape[0],  u.shape[1],1))/(-2))
def py_kde(x,X_t,h):
    norm = (X_t.shape[0]*(h**x.shape[1]))
    prob = torch.zeros(x.shape[0]).to(x.device) 
    for i in range(len(X_t)):
        prob+= torch.squeeze(gau_ker((x - X_t[i])/h))/norm
    return(prob)
def py_kde_der(p_x,x):
    # x.requires_grad = True
    # p_x = py_kde(x,X_t,h)
    return (torch.autograd.grad(p_x,x,torch.ones_like(p_x),allow_unused=True,create_graph=True)[0])

# def accuracy(net,X_test,Y_test):
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         outputs = net(X_test)
#         predicted = (outputs > 0.5).float()
#         total += Y_test.size(0)
#         correct += (predicted == Y_test).sum().item()
#     return(100 * correct / total)

def accuracy(preds, labels):
    return (preds == labels).mean()

def train_model(net,trainloader,optimizer,epochs,rate = 10,device= torch.device('cpu'),print_cond = True,privacy_engine = None):
    criterion = nn.BCELoss(reduction= 'none')
    counter =0
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
    
        # data = data.to(device)
        net = net.to(device)
        with BatchMemoryManager(
        data_loader=trainloader, 
        max_physical_batch_size=1000, 
        optimizer=optimizer
    ) as memory_safe_data_loader:
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                
                inputs = data[0].to(device)
                labels = data[1].to(device)
            
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs,labels)
                loss.backward(torch.ones_like(loss))
                optimizer.step()

                # print statistics
                # print(loss.sum().shape)
                running_loss += loss.sum()
                counter+=1
        DELTA = 1e-4
        if (epoch+1) % rate == 0:
            if(privacy_engine != None):
                epsilon = privacy_engine.get_epsilon(DELTA)
            else:
                epsilon = 0
            print(
                f"\tTrain Epoch: {epoch} \t"
                f"Loss: {(running_loss/counter):.6f} "
                # f"Acc@1: {np.mean(top1_acc) * 100:.6f} "
                f"(ε = {epsilon:.2f}, δ = {DELTA})"
                
                )
            counter = 0
            # if i % 100 == 99:    # print every 2000 mini-batches
        # if(epoch%rate==rate-1):
        #     if(print_cond):
        #         print('[%d, %5d] loss: %.10f' %
        #                 (epoch + 1,0, running_loss / 100))
        #         running_loss = 0.0
            # if(privacy_engine != None):
                
            #     eps_val.append(privacy_engine.get_epsilon(delta))
            #     acc.append(accuracy(net,X_test,Y_test))


    print('Finished Training')



def train(model, train_loader, optimizer, epoch, device,privacy_engine = None):
    model.train()
    model.to(device)
    criterion = nn.BCELoss()

    losses = []
    top1_acc = []
    
    with BatchMemoryManager(
        data_loader=train_loader, 
        max_physical_batch_size=1000, 
        optimizer=optimizer
    ) as memory_safe_data_loader:

        for i, (images, target) in enumerate(memory_safe_data_loader):   
            optimizer.zero_grad()
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            preds = output.detach().cpu().numpy()
            labels = target.detach().cpu().numpy()
            # print(preds.shape,labels.shape )
            # measure accuracy and record loss
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            # top1_acc.append(acc)

            loss.backward()
            optimizer.step()
            DELTA = 1e-4
    # if (+1) % 10 == 0:
    #     epsilon = privacy_engine.get_epsilon(DELTA)
    #     print(
    #         f"\tTrain Epoch: {epoch} \t"
    #         f"Loss: {np.mean(losses):.6f} "
    #         f"Acc@1: {np.mean(top1_acc) * 100:.6f} "
    #         f"(ε = {epsilon:.2f}, δ = {DELTA})"
    #     )

def test(model, test_loader, device):
    model.to(device)
    model.eval()
    criterion = nn.BCELoss()
    losses = []
    top1_acc = []

    with torch.no_grad():
        for images, target in test_loader:
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)
            preds = output.detach().cpu().numpy() > 0.5
            labels = target.detach().cpu().numpy()
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc)

    top1_avg = np.mean(top1_acc)

    print(
        f"\tTest set:"
        f"Loss: {np.mean(losses):.6f} "
        f"Acc: {top1_avg * 100:.6f} "
    )
    return np.mean(top1_acc)

def load_data(path):
    df=pd.read_csv("data/Churn_Modelling.csv")
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
    return X,Y




def main():
    accs =[]
    import warnings
    warnings.filterwarnings("ignore")
    
    # print(ModuleValidator.validate(model2, strict=False))
    eps_list =np.arange(0.5,6,0.1)
    for target_eps in eps_list:
        X,Y = load_data("data/Churn_Modelling.csv")
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        trainer = torch.utils.data.TensorDataset(X_train,Y_train)
        tester  = torch.utils.data.TensorDataset(X_test,Y_test)

        trainloader = torch.utils.data.DataLoader(trainer, batch_size=1000,
                                                shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(tester, batch_size=1000,
                                                shuffle=False, num_workers=2)   
    
        
        
        from opacus.validators import ModuleValidator

        
    
        privacy_engine = PrivacyEngine()
        
        # model2 = nn.Sequential(
        #     nn.Linear(12, 32),
        #     nn.ReLU(),
        #     # nn.Linear(48, 32),
        #     # nn.ReLU(),
        #     nn.Linear(32, 1),
        #     nn.Sigmoid(

        #     )
        # )
        # optim2 = torch.optim.Adam(model2.parameters(),lr=0.01,weight_decay=1e-4)

        # train_model(model2,trainloader,optim2,100,10,device=torch.device('cuda'))
        # print(accuracy(model2,X_test.to('cuda'),Y_test.to('cuda')))
        # print(accuracy(model2,X_train.to('cuda'),Y_train.to('cuda')))
        # model2= nn.Sequential(
        #     nn.Linear(12, 32),
        #     nn.ReLU(),
        #     # nn.Linear(48, 32),
        #     # nn.ReLU(),
        #     nn.Linear(32, 1),
        #     nn.Sigmoid(

        #     )
        # )
        
        
        model2= nn.Sequential(
            nn.Linear(12, 32),
            nn.ReLU(),
            # nn.Linear(48, 32),
            # nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(

            )
        )
        # print(test(model,testloader,torch.device('cuda')))
        # model.train()
        optim = torch.optim.Adam(model2.parameters(),lr=0.01,weight_decay=1e-4)
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            
            module=model2,
            optimizer=optim,
            data_loader=trainloader,
            epochs=1000,
            target_epsilon=target_eps,
            target_delta= 1e-4,
            max_grad_norm=3.0,
        )

        # train_model(model,train_loader,optimizer,100,10,device=torch.device('cuda'),privacy_engine=privacy_engine)
        train(model,train_loader,optimizer,1000,torch.device('cuda'),privacy_engine=privacy_engine)
        # # print(accuracy(model,X_test.to('cuda'),Y_test.to('cuda')))
        # # print(accuracy(model,X_train.to('cuda'),Y_train.to('cuda')))
        accs.append(test(model,testloader,torch.device('cuda')))
        # print(target_eps)
        del(model)
        del(privacy_engine)
        del(optimizer)
        del(train_loader)

    import matplotlib.pyplot as plt

    x = eps_list
    y = accs

    plt.plot(x, y)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Plotting x and y')
    plt.show()   
        # del(test)
# Main body of code. Other functions and class methods are called from main.

if __name__ == "__main__":
    main()
    
        