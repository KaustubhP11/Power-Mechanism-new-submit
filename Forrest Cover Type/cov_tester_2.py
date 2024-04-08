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
import time
import argparse
parser = argparse.ArgumentParser(description='Forrest cover private testing')
parser.add_argument('--data_path', type=str, default='data/covtype.csv',
                    help='Path to the CSV file containing the forrest cover data')
parser.add_argument('--eps', type=float, default=5.0,
                    help='Set epsilon for the model')
parser.add_argument('--model_path', type=str, default='Models/net_1_cov',
                    help='Path to the Model to create embeddings')
parser.add_argument('--batch_size', type=int, default=4096,
                    help='Batch size for training the model')
parser.add_argument('--batch_size_priv', type=int, default=1024,
                    help='Batch size for calculating eps of the model')
parser.add_argument('--num_epochs', type=int, default=100,
                    help='Number of epochs to train the model')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='Learning rate for the optimizer')
parser.add_argument('--wandb_project', type=str, default='covertype test',
                    help='Name of the Weights & Biases project to log metrics to')
parser.add_argument('--norm',type=float,default= 1,
                    help='Normalizing the data by multiplying with this number')
parser.add_argument('--net_depth',type=int,default= 1,
                    help='Depth of the network')
parser.add_argument('--device', type=str, default='cuda',
                    help='Device to run the model on')
# parser.add_argument('--hist_flag', type=bool, default=False,
#                     help='If histogram should be plotted or not')
     
args = parser.parse_args()

# You can access the parsed arguments like this:

data_path = args.data_path
eps = args.eps
batch_size = args.batch_size
num_epochs = args.num_epochs
learning_rate = args.learning_rate
wandb_project = args.wandb_project
model_path = args.model_path
norm = args.norm
net_depth = args.net_depth
batch_size_priv = args.batch_size_priv
device_name = args.device
# hist_flag = args.hist_flag  
 # adds all of the arguments as config variables
def main(data_path ,batch_size,num_epochs,learning_rate,model_path):
    device = torch.device(device_name)
    X,Y = cov_data_loader(data_path,norm=norm)
    # max_dist = torch.cdist(X, X).max()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    # if(model_path=="Models/cov_full_expt"):
    #     print("Loading saved embeddings")
    #     X_emb_train = torch.load("Embeddings/cov_full_expt/X_emb_train.pt")
    #     X_emb_test = torch.load("Embeddings/cov_full_expt/X_emb_test.pt")
    #     losses_train = torch.load("Embeddings/cov_full_expt/losses_train.pt")
    #     losses_test = torch.load("Embeddings/cov_full_expt/losses_test.pt")
    # if(model_path=="Models/cov_full"):
    #     print("Loading saved embeddings")
    #     X_emb_train = torch.load("Embeddings/cov_full/X_emb_train.pt")
    #     X_emb_test = torch.load("Embeddings/cov_full/X_emb_test.pt")
    #     losses_train = torch.load("Embeddings/cov_full/losses_train.pt")
    #     losses_test = torch.load("Embeddings/cov_full/losses_test.pt")
    # else:
    model_embs_path = model_path.replace("Models","Embeddings")
    if (model_path in ['Models/cov_full_expt_512_1','Models/cov_full_expt_512_100','Models/cov_full_expt_512_400']):
        X_emb_train = torch.load(model_embs_path + "/X_emb_train.pt")
        X_emb_test = torch.load(model_embs_path + "/X_emb_test.pt")
        losses_train = torch.load(model_embs_path + "/losses_train.pt")
        losses_test = torch.load(model_embs_path + "/losses_test.pt")        
    
    else:
        
    
        train_priv = torch.utils.data.TensorDataset(X_train, Y_train)
        test_priv = torch.utils.data.TensorDataset(X_test, Y_test)

        trainloader_priv = torch.utils.data.DataLoader(train_priv, batch_size=batch_size_priv,
                                            shuffle=False, num_workers=4)
        testloader_priv = torch.utils.data.DataLoader(test_priv, batch_size=batch_size_priv,
                                            shuffle=False, num_workers=4)

        
            
        #     # net = torch.load("../Code/Models/net_1_cov")
        state_dict = torch.load(model_path)

        # Create an instance of Net
        net = Net(net_depth,device=device)
        net.load_state_dict(state_dict)
        

        
        X_emb_train,losses_train = create_model_embs2(net,trainloader_priv,device= device,l=len(X_train),h=0.82)
        X_emb_test,losses_test = create_model_embs2(net,testloader_priv,device= device,l=len(X_test),h=0.82)
        torch.save(X_emb_train, model_embs_path + "/X_emb_train.pt")
        torch.save(X_emb_test, model_embs_path + "/X_emb_test.pt")
        torch.save(losses_train, model_embs_path + "/losses_train.pt")
        torch.save(losses_test, model_embs_path + "/losses_test.pt")
        
        
        
        
        

    
    
    
    max_dist = 1
    losses_train,indices = torch.sort(losses_train*max_dist)


    
     
    X_train = X_train[indices]
    X_emb_train = X_emb_train[indices]
    Y_train = Y_train[indices]
 
    run = wandb.init(project=wandb_project)
    wandb.config.update(args)
   
    set_eps = eps
    ind = (losses_train < set_eps).sum()
    print(ind)
    # num_epochs_eps = int(len(X)*num_epochs/ind)
    batch_size_eps = batch_size
    print(batch_size_eps)
    #write code for train test split using X_emb and Y

# Remove all things from cuda that were generated till now
    torch.cuda.empty_cache()
    
    
    
    
    train_emb_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_emb_train[0:ind],Y_train[0:ind]), batch_size=batch_size,
                                            shuffle=True, num_workers=2,drop_last=True)
    test_emb_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_emb_test,Y_test), batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    
    #write code to append Xemb and Xemb_rest
    
    criterion = nn.CrossEntropyLoss()
    # model = nn.Sequential(
    #             nn.Linear(54, 128),
    #             nn.ReLU(),
    #             nn.Linear(128, 256),
    #             nn.ReLU(),
    #             nn.Linear(256, 128),
    #             nn.ReLU(),
    #             nn.Linear(128, 7),
    #             nn.Softmax(dim=1)

    #         )
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
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=1e-4)
    time_start = time.time()

    train_emb(model, train_emb_loader, criterion, optimizer, num_epochs=num_epochs,device=device,test_loader = test_emb_loader,test_total_loader = None)
    time_end = time.time()
    print("Time taken to train the model: ",time_end-time_start)
    model.to(torch.device('cpu'))
    # test_model(model,test_emb_loader)
    # test_model(model,test_emb_full_loader)
if __name__ == "__main__":
    main(data_path=data_path,batch_size=batch_size,num_epochs=num_epochs,learning_rate=learning_rate,model_path=model_path)

