#!/usr/bin/env python
# coding: utf-8

# In[1]:


#將 glove.6B.300d.txt 複製到本程式同一執行目錄中, 再執行後續程式
# 載入 gensim 與 GloVe 模型容器
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

# 忽略警告訊息
import warnings
warnings.filterwarnings("ignore")


# In[2]:


# 設定模型
input_file = 'glove.6B.300d.txt'
output_file = 'gensim_glove.6B.300d.txt'
glove2word2vec(input_file, output_file)


# In[3]:


# 轉換並讀取模型
model = KeyedVectors.load_word2vec_format(output_file, binary=False)


# In[4]:


# 顯示最相近的字彙
model.most_similar(['woman'])


# In[ ]:




