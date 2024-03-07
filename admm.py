import argparse
# apply admm matrix factorization -> how to get the first order easily?
from src.preprocess.ml100k import ML100k
from src.models.mf import MF
from src.utils.trainer import Trainer
import torch
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
    item_embeddings = []
    for i in range(num_block):
        block = blocks[i]
        users = usersblocks[i]
        num_users = len(users)
        num_items = block.item.nunique()
        user_embeddings.append(np.random.rand(num_users, args.latent_dim))
        item_embeddings.append(np.random.rand(num_items, args.latent_dim))

    # I also want global  item embeddings that needs to be updated every ADMM iteration
    global_item_embeddings = np.random.rand(num_items, args.latent_dim)

    return user_embeddings, item_embeddings, global_item_embeddings

def get_epsilon(ratings, user_embedding, item_embedding, user, item):
    # this function is responsible for getting the epsilon
    return ratings[user, item]-np.multiply(user_embedding[user,:],item_embedding[item,:])

def admm_update(user_embedding, item_embedding, global_item_embeddings, block, users, num_users, num_items, latent_dim, rho=0.1, max_iter=100):
    # this function is responsible for updating the user embeddings and item embeddings for each block using ADMM
    # block will be the whole datasets of each block
    for i in range(len(block)):
        user = block.iloc[i].user
        item = block.iloc[i].item
        rating = block.iloc[i].rating
        dual_variable = np.zeros((latent_dim,))
        # update user embedding
        eij=get_epsilon(rating,user_embedding,item_embedding,user,item)
        user_embedding[user,:]=user_embedding[user,:]+args.tau_t*(eij*item_embedding[item,:]-args.lambda_1*user_embedding[user,:])

        # update item embedding
        item_embedding[item,:]=(args.tau_t/(1+rho*args.tau_t))*((1-args.lambda_2*args.tau_t)/args.taut)*item_embedding[item,:]+\
            eij*user_embedding[user,:]+rho*global_item_embeddings[item,:]-dual_variable
    
    
    pass 

argparser = argparse.ArgumentParser()
argparser.add_argument('--latent_dim', type=int, default=10)
argparser.add_argument('--dropout', type=float, default=0.1)
argparser.add_argument('--batch_size', type=int, default=500)
argparser.add_argument('--tau_t', type=int, default=0.01)
argparser.add_argument('--lambda_1', type=float, default=0.1)
argparser.add_argument('--lambda_2', type=float, default=0.1)
args = argparser.parse_args()

ml100k = ML100k(args)
data=ml100k.get_dataframe()

# we want to distribute the users into num_block blocks and allocate memory for each block 
# and those two functions are responsible for that
blocks,user_blocks=distribute_users(data)
user_embeddings, item_embeddings, global_item_embeddings = allocate_memory(ml100k.num_items,blocks, user_blocks, num_block=10)


for p in range(len(blocks)):
    block = blocks[p]
    users = user_blocks[p]
    num_users = len(users)
    num_items = block.item.nunique()
    # we want to update the user embeddings and item embeddings for each block
    admm_update(user_embeddings[p], item_embeddings[p], global_item_embeddings, block, users, num_users, num_items, args.latent_dim, rho=0.1, max_iter=100)





