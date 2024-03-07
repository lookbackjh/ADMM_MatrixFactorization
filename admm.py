import argparse
# apply admm matrix factorization -> how to get the first order easily?
from src.preprocess.ml100k import ML100k
from src.models.mf import MF
from src.utils.trainer import Trainer
import torch
import pandas as pd
# labelencdoer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

def distribute_users(data, num_block=10):
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

def allocate_memory(num_items,blocks, usersblocks, num_block=10):
    # this function is to allocate memory for each block
    # for each block, there needs to be corresponding embeddings of users and items.  
    # note that blocks contain the data of each block and userblocks contain the users of each block
    # for implementation i will use numpy instead of pytorch for simplicity

    user_embeddings = []
    user_embeddings_idx = []
    item_embeddings = []
    dual_variables = []
    for i in range(num_block):
        block = blocks[i]
        users = usersblocks[i]
        num_users = len(users)
        unique_users=np.unique(block.user)
        #num_items = block.item.nunique()
        user_embeddings.append(np.random.rand(num_users, args.latent_dim))
        item_embeddings.append(np.random.rand(num_items, args.latent_dim))
        dual_variables.append(np.zeros((num_items, args.latent_dim)))

        # want to make user_embeddings with frist column as user index
        user_embedding_idx=np.random.rand(num_users, args.latent_dim+1)
        user_embedding_idx[:,0] = unique_users 
        user_embeddings_idx.append(user_embedding_idx)

    # I also want global  item embeddings that needs to be updated every ADMM iteration
    global_item_embeddings = np.random.rand(num_items, args.latent_dim)
    

    return user_embeddings, item_embeddings, global_item_embeddings, dual_variables, user_embeddings_idx



def get_epsilon(ratings, user_embedding, item_embedding, user, item):
    # this function is responsible for getting the epsilon
    return ratings-np.dot(user_embedding,item_embedding[item,:])

def admm_update(user_embedding, item_embedding, global_item_embeddings, block, users, num_users, num_items, latent_dim, dual_variable,user_labelencoder,item_labelencoder, rho=0.01, max_iter=100):
    # this function is responsible for updating the user embeddings and item embeddings for each block using ADMM
    # block will be the whole datasets of each block
    for i in range(len(block)):
        user = block.iloc[i].user
        item = block.iloc[i]['item']
        rating = block.iloc[i]['rating']
        user=user_labelencoder.transform([user])[0]
        item=item_labelencoder.transform([item])[0]

        # want user embedding row where first column is user index
        temp=np.where(user_embedding[:,0]==user)
        uembedding=user_embedding[temp[0],1:]

        #dual_variable = np.random.rand(num_items,args.latent_dim)
        # update user embedding
        # user=user_labelencoder.transform([user])[0]
        # item=item_labelencoder.transform([item])[0]
        eij=get_epsilon(rating,user_embedding[temp[0],1:],item_embedding,user,item)
        eij=eij[0]
        if eij>1000000:

            print("eij is too large")
        user_embedding[temp[0],1:]=user_embedding[temp[0],1:]+args.tau_t*(eij*item_embedding[item,:]-args.lambda_1*user_embedding[temp[0],1:])

        # update item embedding
        item_embedding[item,:]=(args.tau_t/(1+rho*args.tau_t))*(((1-args.lambda_2*args.tau_t)/args.tau_t)*item_embedding[item,:]+eij*user_embedding[temp[0],1:]+rho*global_item_embeddings[item,:]-dual_variable[item,:])
    
    
    pass 

argparser = argparse.ArgumentParser()
argparser.add_argument('--latent_dim', type=int, default=10)
argparser.add_argument('--dropout', type=float, default=0.1)
argparser.add_argument('--batch_size', type=int, default=500)
argparser.add_argument('--tau_t', type=int, default=0.01)
argparser.add_argument('--lambda_1', type=float, default=0.001)
argparser.add_argument('--lambda_2', type=float, default=0.001)
argparser.add_argument('--max_iter', type=int, default=100)
args = argparser.parse_args()

ml100k = ML100k(args)
data=ml100k.get_dataframe()
user_labelencoder = LabelEncoder()
user_labelencoder.fit(data['user'].values)
item_labelencoder = LabelEncoder()
item_labelencoder.fit(data['item'].values)
num_items = data.item.nunique()


# train test split data
train, test = train_test_split(data, test_size=0.2, random_state=42)


# we want to distribute the users into num_block blocks and allocate memory for each block 
# and those two functions are responsible for that
blocks,user_blocks=distribute_users(train)
user_embeddings, item_embeddings, global_item_embeddings , dual_variables,user_embeddings_idx= allocate_memory(ml100k.num_items,blocks, user_blocks, num_block=10)

#create labelencoder that transforms user into given index. 

for i in range(args.max_iter):

    global_item_embeddings = np.mean(item_embeddings, axis=0)
    for p in range(len(blocks)):
        block = blocks[p]
        users = user_blocks[p]
        num_users = len(users)
        num_items = block.item.nunique()
        # we want to update the user embeddings and item embeddings for each block
        admm_update(user_embeddings_idx[p], item_embeddings[p], global_item_embeddings, block, users, num_users, num_items, args.latent_dim, dual_variables[p],user_labelencoder,item_labelencoder,rho=0.01, max_iter=100)

    # updatae global item embeddings
    
      


    for p in range(len(blocks)):
        # update dual
        dual_variables[p] = dual_variables[p] + args.tau_t*(item_embeddings[p]-global_item_embeddings)
    


    print("iter"+str(i))  
    # calculate test loss
    test_x = test[['user', 'item']].values
    test_y = test['rating'].values
    test_loss = 0

    
    # check which block the user is in


    for i in range(len(test_x)):
        user = test_x[i][0]
        item = test_x[i][1]
        for idx,arr in enumerate(user_blocks):
            if user in arr:
                ub = idx
                break

        user=user_labelencoder.transform([user])[0]
        item=item_labelencoder.transform([item])[0]
        rating = test_y[i]
        uidx=np.where(user_embeddings_idx[ub][:,0]==user)


        eij = rating-np.dot(user_embeddings_idx[ub][uidx[0],1:],global_item_embeddings[item,:])
        test_loss += eij**2
    print("test loss: "+str(test_loss/len(test_x)))





