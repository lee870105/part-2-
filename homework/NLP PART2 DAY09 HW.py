#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
import torch.optim as optim


# In[2]:


class LinearBNAC(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, dropout=0.3, is_output=False):
        super(LinearBNAC, self).__init__()
        if is_output and out_channels==1:
            self.linear = nn.Sequential(
                nn.Linear(in_channels, out_channels, bias=bias),
                nn.Sigmoid()
            )
        elif is_output:
            self.linear = nn.Sequential(
                nn.Linear(in_channels, out_channels, bias=bias),
                nn.Softmax(dim=1)
            )   
        else:
            self.linear = nn.Sequential(
                nn.Linear(in_channels, out_channels, bias=bias),
                nn.Dropout(dropout),
                nn.BatchNorm1d(out_channels),
                nn.LeakyReLU(inplace=True)
            )
            
    def forward(self, x):
        out=self.linear(x)
        return out
class Model(nn.Module):
    def __init__(self, input_dimention, output_classes=1):
        super(Model, self).__init__()
        self.layer1 = LinearBNAC(input_dimention, 128)
        self.layer2 = LinearBNAC(128,64)
        self.layer3 = LinearBNAC(64,32)
        self.output = LinearBNAC(32, output_classes, is_output=True)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.output(x)
        return x 


# In[3]:


#準備輸入資料、優化器、標籤資料、模型輸出
model = Model(input_dimention=256,output_classes=10)
optimizer = optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=1e-3)


# In[4]:


batch_size = 4
input_features = 256
dummy_input = torch.randn(batch_size, input_features,)
print(dummy_input)
#target = torch.empty(4, dtype=torch.float).random_(10)
target = torch.tensor([9., 5., 4., 4.], dtype=torch.long)
print(target)


# In[5]:


output = model(dummy_input)
print(output)


# In[6]:


from torch.nn import CrossEntropyLoss
criterion = CrossEntropyLoss()
loss = criterion(torch.log(output), target)
print(loss)


# In[7]:


loss.backward()
print('weight: {}'.format(model.layer1.linear[0].weight))
print('\n')
print('grad: {}'.format(model.layer1.linear[0].weight.grad))


# In[8]:


optimizer.step()
print('weight: {}'.format(model.layer1.linear[0].weight))
print('\n')
print('grad: {}'.format(model.layer1.linear[0].weight.grad))


# In[9]:


optimizer.zero_grad()
print('weight : {}'.format(model.layer1.linear[0].weight))
print('\n')
print('grad : {}'.format(model.layer1.linear[0].weight.grad))

