import argparse
# apply admm matrix factorization -> how to get the first order easily?
from src.preprocess.ml100k import ML100k
from src.models.mf import MF
from src.utils.trainer import Trainer
import torch

argparser = argparse.ArgumentParser()
argparser.add_argument('--latent_dim', type=int, default=10)
argparser.add_argument('--dropout', type=float, default=0.1)
argparser.add_argument('--batch_size', type=int, default=500)
args = argparser.parse_args()

ml100k = ML100k(args)
train_data,test_data=ml100k.load_torch_data()
mf_model=MF(args,ml100k.num_users,ml100k.num_items)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(mf_model.parameters(), lr=0.001)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mf_model.to(device) # just note that to is in-place operation
trainer=Trainer(mf_model,optimizer,criterion, device)
trainer.train(train_data, test_data,100)









