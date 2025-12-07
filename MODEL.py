import torch.nn as nn
class Funnel(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.relu = nn.ReLU()
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList() 
        for i in range(len(sizes) - 2):
            self.layers.append(nn.Linear(sizes[i], sizes[i+1]))
            self.bns.append(nn.BatchNorm1d(sizes[i+1]))
        self.output = nn.Linear(sizes[-2], sizes[-1])

    def forward(self, x):
        for i,layer  in enumerate(self.layers):
            x = self.relu(self.bns[i](layer(x)))
        x = self.output(x)
        return x
        ...
