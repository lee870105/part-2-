#!/usr/bin/env python
# coding: utf-8

# In[1]:


#熟練Pytorch Dataset與DataLoader進行資料讀取
# Import torch and other required modules
import glob
import torch
import re
import nltk
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_svmlight_file
from nltk.corpus import stopwords

nltk.download('stopwords') #下載stopwords
nltk.download('punkt') #下載word_tokenize需要的corpus


# In[2]:


#探索資料與資料前處理
#在train資料中，有分成pos(positive)與neg(negative)，分別為正評價與負評價，此評價即為label。
# 讀取字典，這份字典為review內所有出現的字詞
with open(r'C:\Users\lab506\Desktop\aclImdb\imdb.vocab', encoding="utf-8") as f:
    vocab = f.read()

vocab = vocab.split('\n')
# 以nltk stopwords移除贅字，過多的贅字無法提供有用的訊息，也可能影響模型的訓練
print(f"vocab length before removing stopwords: {len(vocab)}")
vocab = list(set(vocab).difference(set(stopwords.words('english'))))
print(f"vocab length after removing stopwords: {len(vocab)}")
# 將字典轉換成dictionary
vocab_dic = dict(zip(vocab, range(len(vocab))))


# In[3]:


# 將資料打包成(x, y)配對，其中x為review的檔案路徑，y為正評(1)或負評(0)
# 這裡將x以檔案路徑代表的原因是讓同學練習不一次將資料全讀取進來，若電腦記憶體夠大(所有資料檔案沒有很大)
# 可以將資料全一次讀取，可以減少在訓練時I/O時間，增加訓練速度

review_pos = glob.glob(r"C:\Users\lab506\Desktop\aclImdb\train\pos\*.txt")#glob為抓路徑
review_neg = glob.glob(r"C:\Users\lab506\Desktop\aclImdb\test\neg\*.txt")
review_all = review_pos + review_neg
y = [1]*len(review_pos) + [0]*len(review_neg)
review_pairs = list(zip(review_all, y))
print(review_pairs[:2])
print(f"Total reviews: {len(review_pairs)}")


# In[4]:


#建立Dataset與DataLoader讀取資料
#這裡我們會需要兩個helper functions，其中一個是讀取資料與清洗資料的函式(load_review)，另外一個是生成詞向量BoW的函式 (generate_bow)
def load_review(review_path):
    
    with open(review_path, 'r') as f:
        review = f.read()
        
    #移除non-alphabet符號、贅字與tokenize
    review = re.sub('[^a-zA-Z]',' ',review)
    review = nltk.word_tokenize(review)
    review = list(set(review).difference(set(stopwords.words('english'))))
    
    return review
def generate_bow(review, vocab_dic):
    bag_vector = np.zeros(len(vocab_dic))
    for word in review:
        if vocab_dic.get(word):
            bag_vector[vocab_dic.get(word)] += 1
            
    return bag_vector
class dataset(Dataset):
    '''custom dataset to load reviews and labels
    Parameters
    ----------
    data_pairs: list
        directory of all review-label pairs
    vocab: list
        list of vocabularies
    '''
    def __init__(self, data_dirs, vocab):
        self.data_dirs = data_dirs
        self.vocab = vocab

    def __len__(self):
        return len(self.data_dirs)

    def __getitem__(self, idx):
        pair = self.data_dirs[idx]
        review = pair[0]
        review = load_review(review)
        review = generate_bow(review, self.vocab)
        
        return idx, review, pair[1]


# In[5]:


custom_dst = dataset(review_pairs, vocab_dic)
custom_dst[400]


# In[6]:


custom_dataloader = DataLoader(dataset=custom_dst, batch_size=4, shuffle=True)#shuffle 為是否打散資料
next(iter(custom_dataloader))#我們也可以透過 next() 來個別讀取資料


# In[ ]:




