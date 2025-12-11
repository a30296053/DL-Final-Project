from  torch.utils.data import Dataset
import torch
import numpy as np

class DRIAM(Dataset):
    def __init__(self, X ,  Y):
        self.X = torch.tensor(X,  dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype= torch.float32)

    def __len__(self):
        return len(self.X)
    def __getitem__(self, indx):
        return self.X[indx] , self.Y[indx]
def get_data():
    path = './data/'
    X = np.load(path+'X_norm.npy')
    Y = np.load(path+ 'Y.npy')
    print(type(X))
    return X, Y
