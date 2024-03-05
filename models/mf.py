import torch
import torch.nn as nn

class MF(nn.Module):

    def __init__(self, args ) -> None:
        super().__init__()
        self.num_users = args.num_users
        self.num_items = args.num_items
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

    def predict(self, *args, **kwargs) -> None:
        pass

