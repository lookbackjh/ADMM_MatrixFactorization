import pandas as pd
# traintest split
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from src.preprocess.mldataloader import MlDataLoader


class ML100k():

    def __init__(self,args) -> None:
        self.args=args
        self.call_data(args)
        pass

    def call_data(self, args) -> None:
        self.args = args
        data_path="data/ml-100k/u.data"
        self.data= pd.read_csv(data_path, sep='\t', header=None, names=['user', 'item', 'rating', 'timestamp'])
        self.num_users = self.data.user.nunique()
        self.num_items = self.data.item.nunique()
        self.num_ratings = len(self.data)

        # make user and item column as x
        # make user and item column add 1


    
        self.data.user = self.data.user -1
        self.data.item = self.data.item -1
        self.x = self.data[['user', 'item']].values

        self.y= self.data.rating.values

        # split data into train and test
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=42)
        
        
    def load_data(self,args):
        self.call_data(args)
        sorted_idx = self.x_train[:, 0].argsort()
        self.x_train = self.x_train[sorted_idx]
        self.y_train = self.y_train[sorted_idx]
        sorted_idx = self.x_test[:, 0].argsort()
        self.x_test = self.x_test[sorted_idx]
        self.y_test = self.y_test[sorted_idx]

        return self.x_train, self.y_train, self.x_test, self.y_test
    
    def load_torch_data(self):
        train_dataset = MlDataLoader(self.x_train, self.y_train)
        test_dataset = MlDataLoader(self.x_test, self.y_test)

        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False)

        return train_loader, test_loader


    def get_dataframe(self):
        # sort dataframe by user
        self.data = self.data.sort_values(by=['user'])


        return self.data[['user', 'item','rating']]

        


