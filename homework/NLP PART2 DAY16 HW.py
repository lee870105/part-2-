#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 載入 gensim 與 word2vec 模型
import gensim
from gensim.models import word2vec

# 忽略警告訊息
import warnings
warnings.filterwarnings("ignore")
sentences = word2vec.Text8Corpus('text8/text8')
model = word2vec.Word2Vec(sentences, size=10, min_count=3, window=5)
#size : 詞向量的維度
#min_count : 最小次數，一個詞出現的次數若小於 min_count，則拋棄不參與訓練。
#window : 訓練窗格大小，也就是一個詞在看上下文關係時，上下應該各看幾個字的意思。


# In[2]:


model.save('text8.model')
model1 = word2vec.Word2Vec.load('text8.model')


# In[3]:


# 顯示字彙間的相似性
model.wv.similarity('woman', 'man')


# In[4]:


# 顯示字彙的詞向量
model['computer']


# In[ ]:




