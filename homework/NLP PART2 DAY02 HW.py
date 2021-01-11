#!/usr/bin/env python
# coding: utf-8

# In[1]:


#torch.from_array() / tensor.numpy()
#torch.unsqueeze() / torch.squeeze()
#tensor.transpose() / tensor.permute()
#torch.reshape() / tensor.view()
#torch.randn() / torch.rand() / torch.randint()
import torch
import numpy as np


# In[7]:


#Function 1 - torch.from_array() / tensor.numpy()
a = np.random.rand(1,2,3,3)
print(f'a: {type(a)}, {a.dtype}')
b = torch.from_numpy(a)
print(f'b: {type(b)}, {b.dtype}')
c = torch.tensor(a)
print(f'c: {type(c)}, {c.dtype}')
d = c.numpy()
print(f'd: {type(d)}, {d.dtype}')
a


# In[10]:


# Example 2 - we can see that the transformed dtype will in accordance with the original one
a = np.random.randint(low=0, high=10, size=(2,2))
print(f'a: {type(a)}, {a.dtype}')
b = torch.from_numpy(a)
print(f'b: {type(b)}, {b.dtype}')
c = torch.tensor(a)
print(f'c: {type(c)}, {c.dtype}')
d = c.numpy()
print(f'd: {type(d)}, {d.dtype}')
b


# In[11]:


#Function 2 - torch.unsqueeze() / torch.squeeze()
# Example 1 - expanding/squeeze the dimension
a = torch.tensor([[[1,2],[2,3]]], dtype=torch.float32)
print(f'Before unsqueeze/squeeze: {a.size()}')

b = torch.unsqueeze(a, dim=0)
print(f'After unsqueeze: {b.size()}')

c = torch.squeeze(a, dim=0)
print(f'After squeeze: {c.size()}')


# In[12]:


# Example 2 - expanding/squeeze the dimension
a = torch.tensor([[1,2],[2,3]], dtype=torch.float32)
print(f'Before unsqueeze/squeeze: {a.size()}')

b = torch.unsqueeze(a, dim=1)
print(f'After unsqueeze: {b.size()}')

c = torch.squeeze(a, dim=1)
print(f'After squeeze: {c.size()}')


# In[13]:


#Function 3 - tensor.transpose() / tensor.permute()
# Example 1 - transpose and permute the tensor dimension
a = torch.tensor([[[1,2,3], [4,5,6]], [[7,8,9], [10,11,12]]])
print(f'Original shape: {a.size()}')

b = a.transpose(dim0=0, dim1=2)
print(f'Transpose shape: {b.size()}')

c = a.permute((1,2,0))
print(f'Permute shape: {c.size()}')


# In[14]:


# Example 2 - sharing memory
a = torch.tensor([[1,2],[3,4]])
print(f'Original shape: {a.size()}')
print(a)

b = a.transpose(dim0=0, dim1=1)
print(f'Transpose shape: {b.size()}')
print(b)

c = a.permute((0,1))
print(f'Permute shape: {c.size()}')
print(c)

print('\n')
# change a[0][0] to 0
a[0][0] = 0

# check the value of a, b, c
print(f'a: {a}')
print(f'b: {b}')
print(f'c: {c}')


# In[15]:


#Function 4 - torch.reshape() / tensor.view()
# Example 1 - reshape and view to change tensor's dimension

a = torch.tensor([[[1,2,3], [4,5,6]], [[7,8,9], [10,11,12]]])
print(f'Original shape: {a.size()}')

b = torch.reshape(a, shape=(-1, 3))
print(f'Reshape shape: {b.size()}')

c = a.view((-1,3))
print(f'View shape: {c.size()}')


# In[16]:


# Example 2 - sharing memory and continuous data
a = torch.tensor([[1,2],[3,4]])
print(f'Original shape: {a.size()}')
print(a)

b = torch.reshape(a, (-1,))
print(f'Transpose shape: {b.size()}')
print(b)

c = a.view((-1))
print(f'Permute shape: {c.size()}')
print(c)

print('\n')
# change a[0][0] to 0
a[0][0] = 0

# check the value of a, b, c
print(f'a: {a}')
print(f'b: {b}')
print(f'c: {c}')


# In[ ]:




