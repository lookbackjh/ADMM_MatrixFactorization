import torch
import torch.nn as nn

class MF(nn.Module):

    def __init__(self, args ,num_users,num_items) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = args.latent_dim
        self.user_embedding = nn.Embedding(self.num_users, self.latent_dim)
        self.item_embedding = nn.Embedding(self.num_items, self.latent_dim)
        self.dropout = nn.Dropout(p=args.dropout)
        #self.sigmoid = nn.Sigmoid()


    def forward(self, x) -> None:
        user = x[:, 0]
        item = x[:, 1]
        user = self.user_embedding(user)
        item = self.item_embedding(item)
        out = torch.sum(user * item, dim=1)
        return out



