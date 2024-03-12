import argparse
# apply admm matrix factorization -> how to get the first order easily?
from src.preprocess.ml100k import ML100k
from src.models.mf import MF
from src.utils.trainer import Trainer
import torch
import torch.nn as nn
import copy

import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
import torch

class MlDataLoader(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
import pandas as pd
# labelencdoer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np


argparser = argparse.ArgumentParser()
argparser.add_argument('--latent_dim', type=int, default=10)
argparser.add_argument('--dropout', type=float, default=0.1)
argparser.add_argument('--batch_size', type=int, default=500)
argparser.add_argument('--tau_t', type=int, default=0.0001)
argparser.add_argument('--lambda_1', type=float, default=0.05)
argparser.add_argument('--lambda_2', type=float, default=0.05)
argparser.add_argument('--max_iter', type=int, default=100)
argparser.add_argument('--alpha', type=float, default=0.002)
argparser.add_argument('--beta', type=float, default=0.7)
argparser.add_argument('--rho', type=float, default=0.001)
args = argparser.parse_args()


def distribute_users(data, num_block=100):
    # this function is to distribute the users into num_block blocks
    unique_users = data.user.unique()
    num_users = len(unique_users)
    user_blocks=[]
    blocks = []
    for i in range(num_block):
        start = i * num_users//num_block
        end = (i+1) * num_users//num_block
        users = unique_users[start:end]
        blocks.append(data.loc[data['user'].isin(users)])
        user_blocks.append(users)
 
    return blocks, user_blocks



def blocks_to_tensor(blocks):
    
    blocks_tensors = []
    for df in blocks:
        df['user'] = df['user'].astype(int)
        df['item'] = df['item'].astype(int)
        df['rating'] = df['rating'].astype(float)
        x = torch.tensor(df[['user', 'item']].values, dtype=torch.long)
        rating=torch.tensor(df['rating'].values, dtype=torch.float)
        dataloader=DataLoader(MlDataLoader(x, rating), batch_size=args.batch_size, shuffle=True)
        blocks_tensors.append(dataloader)


    return blocks



    pass

def allocate_memory(num_items,blocks, usersblocks, num_block=100):
    # this function is to allocate memory for each block
    # for each block, there needs to be corresponding embeddings of users and items.  
    # note that blocks contain the data of each block and userblocks contain the users of each block
    # for implementation i will use numpy instead of pytorch for simplicity

    #user_embeddings = []
    #user_embeddings_idx = []
    item_embeddings = []
    dual_variables = []
    for i in range(num_block):
        block = blocks[i]
        users = usersblocks[i]
        num_users = len(users)
        unique_users=np.unique(block.user)

        item_embeddings.append(nn.Embedding(num_items,args.latent_dim))
        #item_embeddings.append(np.random.rand(num_items, args.latent_dim))
        dual_variables.append(nn.Parameter(torch.zeros((num_items, args.latent_dim))))


    # I also want global  item embeddings that needs to be updated every ADMM iteration
    global_item_embeddings = nn.Embedding(num_items,args.latent_dim)
    

    return item_embeddings, global_item_embeddings, dual_variables

# I think call data with df and convert it later to torch tensor

def admm_update(user_embedding, item_embedding, global_item_embeddings, block, users, num_users, num_items, latent_dim, dual_variable, rho=0.01, max_iter=100):
    # this function is responsible for updating the user embeddings and item embeddings for each block using ADMM
    # block will be the whole datasets of each block

    pass



ml100k=ML100k(args)
data=ml100k.get_dataframe()
user_label_encoder = LabelEncoder()
item_label_encoder = LabelEncoder()
data['user'] = user_label_encoder.fit_transform(data['user'])
data['item'] = item_label_encoder.fit_transform(data['item'])
num_items = data.item.nunique()
num_users = data.user.nunique()


user_embedding=nn.Embedding(num_users,args.latent_dim) 
# this is global user embedding note that in the real situation we need to create user embedding for each block however, 
# I will just use global user embedding for simplicity

train,test=train_test_split(data, test_size=0.2, random_state=42) # this is dataframe
blocks,user_blocks=distribute_users(train) # now this contains blocks..  for each distributed data, we need to convert it to torch tensor
item_embeddings, global_item_embeddings, dual_variables=allocate_memory(num_items,blocks, user_blocks, num_block=100) # now we've got item embeddings, global item embeddings, and dual variables
blocks_tensor=blocks_to_tensor(blocks) # now this blocks tensor is ready to be used in the training process

# convert all the data to cuda to make batch operation possible
global_item_embeddings.to('cuda')
user_embedding.to('cuda')
for item_embedding in item_embeddings:
    item_embedding.to('cuda')
for dual_variable in dual_variables:
    dual_variable.to('cuda')

# initialize global item embedding as a mean of item embeddings

#global_item_embeddings.weight.copy_(torch.mean(torch.stack([item_embedding.weight for item_embedding in item_embeddings]), dim=0))


for iter in range(args.max_iter):

# for every block apply admm
    fp=0
    with torch.no_grad():
        global_item_embeddings.weight.copy_(torch.mean(torch.stack([item_embedding.weight for item_embedding in item_embeddings]), dim=0))
    for p in range(len(blocks)):
        block=blocks_tensor[p]
        item_embedding=item_embeddings[p]
        dual_variable=dual_variables[p]



    
        user_embedding.requires_grad_(True)
        item_embedding.requires_grad_(True)

        # for autograd, i need to define loss
        criterion = torch.nn.MSELoss()
        fp=0
        temp=0
        user_embedding=user_embedding.to('cuda')
        batch_size=32
    

        for i  in range(0, len(block), batch_size):
            info=block.iloc[i:i+batch_size,:]
            user=info['user'].values
            item=info['item'].values
            rating=info['rating'].values
            
            
            
            user=torch.tensor(user, dtype=torch.long)
            item=torch.tensor(item, dtype=torch.long)
            rating=torch.tensor(rating, dtype=torch.float)
            user=user.to('cuda')
            item=item.to('cuda')
            rating=rating.to('cuda')
            epsilon=rating-torch.sum(user_embedding(user)*item_embedding(item), dim=1)
            l2_user=torch.norm(user_embedding(user))
            l2_item=torch.norm(item_embedding(item))
            temp+=torch.sum(epsilon**2)+args.lambda_1*l2_user+args.lambda_2*l2_item
            
            # first order derivative for fp
            fp+=temp
            fp+=args.tau_t/2*torch.norm(item_embedding.weight-global_item_embeddings.weight)
            item_embedding=item_embedding.to('cuda')
            global_item_embeddings=global_item_embeddings.to('cuda')
            dual_variable=dual_variable.to('cuda')
            dual_term=torch.matmul(dual_variable.T, item_embedding.weight-global_item_embeddings.weight)

            # trace of the dual term
            trace=torch.trace(dual_term)
            fp+=trace
            fp.backward(retain_graph=True)


            with torch.no_grad():
                user_embedding.weight-= args.tau_t*user_embedding.weight.grad
                # update Vt
                item_embedding.weight.copy_(((args.tau_t)/(1+args.rho*args.tau_t))*((item_embedding.weight/args.tau_t) +args.rho*global_item_embeddings.weight-(dual_variable)-item_embedding.weight.grad))
            
            user_embedding.weight.grad.zero_()
            item_embedding.weight.grad.zero_() 

    #update global item embeddings
    with torch.no_grad():
        for p in range(len(blocks)):
            item_embedding=item_embeddings[p]
            dual_variable=dual_variables[p]
            item_embedding=item_embedding.to('cuda')
            global_item_embeddings=global_item_embeddings.to('cuda')
            dual_variable=dual_variable.to('cuda')
            dual_variable.copy_(dual_variable+args.rho*(item_embedding.weight-global_item_embeddings.weight))


    # calculate test loss
    test_user_id=test['user'].values
    test_item_id=test['item'].values
    test_y = test['rating'].values
    test_loss = 0
    test_user_id=torch.tensor(test_user_id, dtype=torch.long)
    test_item_id=torch.tensor(test_item_id, dtype=torch.long)
    test_y=torch.tensor(test_y, dtype=torch.float)
    test_user_id=test_user_id.to('cuda')
    test_item_id=test_item_id.to('cuda')
    test_y=test_y.to('cuda')

    user_embedding.eval()
    with torch.no_grad():

        for j in range(len(test_y)):
            user = test_user_id[j]
            item = test_item_id[j]
            user_emb=user_embedding(user)
            item_emb=global_item_embeddings(item)
            rating = test_y[j]
            eij = rating-torch.dot(user_emb[:],item_emb[:])
            #eij = rating-np.dot(user_embedding.weight[uidx[0],1:],global_item_embeddings[item,:])
            test_loss += eij**2


    print("iter"+str(iter))
    print("temp: "+str(temp))
    print("test loss: "+str(test_loss/len(test_y)))
print('done')















