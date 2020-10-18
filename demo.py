import os

import torch
import torch.nn as nn
from torch import optim


a = torch.Tensor([1,2,3,4,5,6,7,8,9,0]).cuda()
b = torch.Tensor([2,3,4,5,6,7,8,9,0,13]).cuda()

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(10, 10),
            nn.LeakyReLU(inplace=True),
            nn.Linear(10, 10),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = self.model(x)
        return x
model = MLP().cuda()
cer = torch.nn.L1Loss()
optimizer = optim.SGD(model.parameters(),lr = 0.001)
for i in range(100):
    optimizer.zero_grad()
    c =model(a)
    loss = cer(c,b)
    loss.backward()
    optimizer.step()


