import pandas as pd
# traintest split
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from src.preprocess.mldataloader import MlDataLoader


class ML100k():

    def __init__(self) -> None:
        pass

    def load_data(self, args) -> None:
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
        self.x_train=torch.LongTensor(self.x_train)
        self.x_test=torch.LongTensor(self.x_test)
        self.y_train=torch.FloatTensor(self.y_train)
        self.y_test=torch.FloatTensor(self.y_test)
        
        ml100_train = MlDataLoader(self.x_train, self.y_train)
        ml100_test = MlDataLoader(self.x_test, self.y_test)
        ml100_train = DataLoader(ml100_train, batch_size=self.args.batch_size, shuffle=True)
        ml100_test = DataLoader(ml100_test, batch_size=self.args.batch_size, shuffle=False)

        return ml100_train, ml100_test


        


