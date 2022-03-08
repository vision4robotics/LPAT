import torch.nn as nn
import torch.nn.functional as F
import torch as t


class Cattention(nn.Module):

    def __init__(self, in_dim):
        super(Cattention, self).__init__()
        self.chanel_in = in_dim
        self.conv1=nn.Sequential(
                nn.ConvTranspose2d(in_dim*2, in_dim,  kernel_size=1, stride=1),
                )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear1=nn.Conv2d(in_dim, in_dim // 6, 1, bias=False)
        self.linear2=nn.Conv2d(in_dim // 6, in_dim, 1, bias=False)
        self.gamma = nn.Parameter(t.zeros(1))
        self.activation=nn.ReLU(inplace=True)
        self.dropout=nn.Dropout()
    def forward(self, x,y):
        ww=self.linear2(self.dropout(self.activation(self.linear1(self.avg_pool(y)))))
        weight=self.conv1(t.cat((x,y),1))*ww
        
        
        return x+self.gamma*weight*x