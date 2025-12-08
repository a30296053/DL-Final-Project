import torch.nn as nn
class Funnel(nn.Module):
    def __init__(self, sizes,p_dropout = 0.3):
        super().__init__()
        self.relu = nn.ReLU()
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList() 
        self.dropout = nn.ModuleList()
        for i in range(len(sizes) - 2):
            self.layers.append(nn.Linear(sizes[i], sizes[i+1]))
            self.bns.append(nn.BatchNorm1d(sizes[i+1]))

            if(sizes[i] < 2000):
                self.dropout.append(nn.Identity())
            else:
                self.dropout.append(nn.Dropout(p_dropout))


        self.output = nn.Linear(sizes[-2], sizes[-1])


    def forward(self, x):
        for i,layer  in enumerate(self.layers):
            x = self.relu(self.bns[i](layer(x)))
            x = self.dropout[i](x)
        x = self.output(x)
        return x
        ...