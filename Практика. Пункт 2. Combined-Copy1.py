#!/usr/bin/env python
# coding: utf-8

# In[43]:


import numpy as np
from matplotlib import pyplot as plt

import pandas as pd

from PIL import Image

from os import listdir
from os.path import isfile, join


# In[115]:


mypath = r'D:\practis\video_combined\photo' #берем фото путь ____ меняем цифру в пути для обработки фото с камер
myoutpath = r'D:\practis\video_combined\hist256' #путь, куда кладем ____ меняем цифру в пути для обработки фото с камер и номер папки
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ] #будущее название считываемого файла


# In[116]:


b = 256 #колличество столбцов в диаграмме


# In[117]:


matX = np.zeros((len(onlyfiles),b), dtype = np.double) #создаю пустой массив нужных размеров


# In[118]:


for n in range(0, len(onlyfiles)):
  images = Image.open(join(mypath,onlyfiles[n])).convert('L') #считываем изображение
  pix = np.array(images)
  num_rows, num_cols = pix.shape #определяем его размер
  hist = images.histogram()
  np_hist = np.array(hist) #строим гистограмму изображения
  norm_hist = np_hist/(num_rows*num_cols) #нормируем гистограмму
  f = np.sum(norm_hist.reshape(b, 256//b), axis=1)
  #print(f, np.shape(f))
  x = list(range(0,b)) #задаю ось абсцисс
  fig = plt.bar(x, f) #делаем граффик гистограммы
  matX[n] = f #построково заполняю массив
  plt.savefig(join(myoutpath,onlyfiles[n])) #сохраняем гистограммы под тем же именем
  #plt.show()
  plt.cla() #разделяет диаграммы

#пункт 2.1
print(matX) #веткор признаков 
#print(np.shape(matX))  


# In[119]:


matY = np.zeros((1, len(onlyfiles)), dtype = np.float) #метка класса, НЕ УВЕРЕН, ЧТО ЗАДАЛ ПРАВИЛЬНО, МАКСИМУМ НУЖНО БРАТЬ НА ГЛАЗ видимо
for i in range(0,824):
    matY[0, i] = 1 #солнечно
for i in range(824,len(onlyfiles)):
    matY[0, i] = -1 #пасмурно

#matY = matY.transpose()
print(matY)
print(np.shape(matY))


# In[121]:


#перехожу к пункту 2.2: делим выборку на обучающую и тестовую


# In[122]:


from sklearn.model_selection import train_test_split
X = matX
Y = np.ravel(matY) #нужно ли транспонировать?
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify = Y, test_size=0.30, random_state=45) #добавил stratify, нужно ли


# In[123]:


X_train


# In[124]:


X_test


# In[125]:


Y_train


# In[126]:


Y_test


# In[127]:


#перехожу к пункту 2.3: сохраняю выборки


# In[128]:


np.savetxt(r'D:\practis\video_combined\matrices256\X_train.csv', X_train, delimiter=',', fmt= '%.15f') #заменить цифру в пути
np.savetxt(r'D:\practis\video_combined\matrices256\X_test.csv', X_test, delimiter=',', fmt= '%.15f')
np.savetxt(r'D:\practis\video_combined\matrices256\Y_train.csv', Y_train, delimiter=',', fmt= '%.15f')
np.savetxt(r'D:\practis\video_combined\matrices256\Y_test.csv', Y_test, delimiter=',', fmt= '%.15f')


# In[ ]:




