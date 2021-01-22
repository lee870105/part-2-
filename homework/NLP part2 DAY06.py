#!/usr/bin/env python
# coding: utf-8

# In[1]:


#作業目的: 熟練以Torchtext進行文本資料讀取
import torch
import pandas as pd
import numpy as np
from torchtext import data, datasets


# In[2]:


# 探索資料
# 可以發現資料為文本與類別，而類別即為正評與負評
input_data = pd.read_csv('Z:\研究所\自然語言馬拉松 深度學習\DAY06\polarity.tsv', delimiter='\t', header=None, names=['text', 'label'])
input_data


# In[3]:


## 建立Field與Dataset
text_field = data.Field(lower=True, tokenize='spacy',dtype=torch.float64)
label_field = data.Field(sequential=False)
input_data = data.TabularDataset(path='Z:\研究所\自然語言馬拉松 深度學習\DAY06\polarity.tsv', format='tsv',fields=[('text',text_field),('label',label_field)])
#取的examples並打亂順序
examples =  input_data.examples
np.random.shuffle(examples)

# 以8:2的比例切分examples
train_ex = examples[:int(len(examples)*0.8)]
test_ex = examples[int(len(examples)*0.8):]

# 建立training與testing dataset
train_data = data.Dataset(examples=train_ex, fields={'text':text_field, 'label':label_field})
test_data = data.Dataset(examples=test_ex, fields={'text':text_field, 'label':label_field})

train_data[0].label, train_data[0].text


# In[20]:


# 建立字典
#text_field.vocab.stoi
text_field.build_vocab(train_data,test_data) 
label_field.build_vocab(train_data,test_data)

print(f"Vocabularies of index 0-5: {text_field.vocab.itos[:10]} \n")
print(f"words to index {text_field.vocab.stoi}")
print(label_field.vocab.stoi)
print(f'Total {len(text_field.vocab)} unique words')


# In[31]:


#最後就是要透過 iterator 來取得 batch data。
train_iter, test_iter = data.Iterator.splits(datasets=(train_data, test_data),
                                             batch_sizes = (3,3),
                                             repeat=False,  
                                             sort_key = lambda ex:len(ex.text)) 


# In[32]:


#透過迴圈取得 batch data
for train_batch in train_iter:
    print(train_batch.text, train_batch.text.shape)
    print(train_batch.label, train_batch.label.shape)
    break


# In[ ]:




