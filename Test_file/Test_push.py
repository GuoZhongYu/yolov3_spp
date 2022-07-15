import torch
import torch.nn as nn

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        print(x)
        global print_x
        print_x = x
        # print(x.shape)
        return x

in_chnls = 128
ratio = 16
modules = nn.Sequential()
modules.add_module("Linear", nn.Conv2d(in_chnls, in_chnls//ratio, 1, 1, 0))
modules.add_module("Print_Layer", PrintLayer())
modules.add_module("activation", nn.Sigmoid())

x = torch.ones(1,128,128,128)
y = modules(x)
print(y.size())
print(y)
print("-----------------------------------")
y_new = torch.sigmoid(print_x)
print(y_new.size())
print(y_new)