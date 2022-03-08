from thop import profile

from utilec import Transformer 
import torch
from torchstat import stat
from torchsummary import summary
import torchvision.models as models

model=Transformer(192)
# summary(model,(121,1,192))

input = torch.randn(121,1,192)
flops, params = profile(model, inputs=(input,))
print('flops',flops)			## 打印计算量
print('params',params)			## 打印参数量
