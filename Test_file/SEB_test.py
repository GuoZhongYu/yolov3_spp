# import torch
# import torch.nn as nn
#
# modules = nn.Sequential()
#
# filters = 128
# bn = 1
# k = 3
# stride = 1
# reduce = 16
#
# x = torch.rand(30,128,128,128)
#
# modules.add_module("Conv2d", nn.Conv2d(in_channels=filters,out_channels=filters,kernel_size=k,stride=stride,padding=k // 2,bias=not bn))
# modules.add_module("BatchNorm2d", nn.BatchNorm2d(filters))
# modules.add_module("activation", nn.LeakyReLU(0.1, inplace=True))
# modules.add_module("AvgPool2d", nn.AdaptiveAvgPool2d(1))
# modules.add_module("Linear_1", nn.Linear(filters, filters // reduce))
# modules.add_module("Relu", nn.ReLU(inplace=True))
# modules.add_module("Linear_2", nn.Linear(filters // reduce, filters))
# modules.add_module("Sigmoid", nn.Sigmoid())
#
# print(modules)
#
# for i in range(len(modules)):
#     result = modules[i](x)
#     print("modules.index:{},result.size:{}".format(i,result.size()))
# #
# # y = modules(x)
# #
# # print(y.shape)
# # print(y)
#

import torch
import torch.nn as nn

class SE(nn.Module):

    def __init__(self, in_chnls=128, ratio=16):
        super(SE, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Conv2d(in_chnls, in_chnls//ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls//ratio, in_chnls, 1, 1, 0)

    def forward(self, x):
        out = self.squeeze(x)
        out = self.compress(out)
        out = torch.relu(out)
        out = self.excitation(out)
        out = torch.sigmoid(out)
        out = x * out.expand_as(x)
        out = out + x
        return out

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        print(x)  # print(x.shape)
        return x

# net = SE()
x = torch.rand(1,128,128,128)
print(x)
y = SE(x)

print(y.size())
print(y)