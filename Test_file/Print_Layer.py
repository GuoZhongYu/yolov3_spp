import torch
import torch.nn as nn
from torch.autograd import Variable


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        print(x)  # print(x.shape)
        return x


model = nn.Sequential(
    nn.Linear(1, 5),
    nn.ReLU(),
    PrintLayer(),  # Add Print layer for debug
    nn.Linear(5, 1),
    nn.LogSigmoid(),
)

x = Variable(torch.ones(10, 1))
output = model(x)