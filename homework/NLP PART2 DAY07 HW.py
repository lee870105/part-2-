#!/usr/bin/env python
# coding: utf-8

# In[1]:


#第一部分：了解 nn.Module的基本操作
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np

model = models.resnet18()
#打印出 model底下所有 parameters 的 name 以及對應的 shape
for name, param in model.named_parameters():
    print(name,param.requires_grad)


# In[2]:


#為了使 forward propagation 加速 並降低 memory 使用量，請將所有 parameters 的requires_grad 關閉，關閉之後執行 forward propagation
input_ = torch.randn(1, 3, 128, 128)
for param in model.parameters():
    param.requires_grad = False
output = model(input_)
print(output.shape)


# In[3]:


#第二部分：依照指令，以較簡潔的方式搭建出模型
#input_shape = torch.Size([10, 12])
#先經過一層 nn.Linear(12, 10)
#經過10層 nn.Linear(10, 10)
#最後經過 nn.Linear(10, 3) 輸出
#每一個 nn.Linear過完後要先經過 nn.BatchNorm1d 才能到下一層，輸出層不用
## 示範
#Linear = nn.Linear(12,10)
#BN = nn.BatchNorm1d(10)


# In[4]:


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.sequential = nn.Sequential(nn.Linear(12,10), nn.BatchNorm1d(10))
        self.repeat_linear = nn.ModuleList([nn.Sequential(nn.Linear(10,10), nn.BatchNorm1d(10)) for _ in range(10)])
        self.output = nn.Linear(10, 3)

    def forward(self, x):
        x = self.sequential(x)
        for module in self.repeat_linear:
            x = module(x)
        x = self.output(x)
        return x
model=Model()
model


# In[5]:


input_ = torch.randn(10,12)
output = model(input_)
output


# In[ ]:




