#!/usr/bin/env python
# coding: utf-8

# In[1]:


#依照指示取出模型特定層的資訊
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np


# In[8]:


# Define model
class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16* 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model
model = TheModelClass()
for name,_ in model.named_children():
    print(name)


# In[11]:


#取出 self.pool層兩次的輸出，包含：
#x = self.pool(F.relu(self.conv1(x)))
#x = self.pool(F.relu(self.conv2(x)))
outputs=[]
def layerpool_hook(module, input_, output):
    outputs.append(output)
model.pool.register_forward_hook(layerpool_hook)


# In[12]:


input_ = torch.randn(1, 3, 32, 32)
output = model(input_)
outputs


# In[13]:


print(outputs[0].shape)
print(outputs[1].shape)


# In[20]:


#加入自定義 initialization fuction
#對所有Conv2D層使用自定義initialization function
#weight : nn.init.kaiming_normal_
#bias : 全部輸入1
from torch.nn import init
def weights_init(m):
    if isinstance(m,nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.ones_(m.bias)
model.apply(weights_init)


# In[21]:


#查看 conv層的bias是否皆為1
for name, parameters in model.named_parameters():
    if ('conv' in name) and ('bias' in name):
        print(name, parameters)
        print('\n')


# In[ ]:




