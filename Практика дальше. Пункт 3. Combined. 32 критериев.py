#!/usr/bin/env python
# coding: utf-8

# In[1]:


#пункт 3.1: считываем обучающие и тестовые выборки


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


X_train = pd.read_csv(r'D:\practis\video_combined\matrices32\X_train.csv') #считываем полученные выборки из прошлого скрипта
X_test = pd.read_csv(r'D:\practis\video_combined\matrices32\X_test.csv')
Y_train = pd.read_csv(r'D:\practis\video_combined\matrices32\Y_train.csv').values.ravel() #добавил .values.ravel() и ошибка ушла
Y_test = pd.read_csv(r'D:\practis\video_combined\matrices32\Y_test.csv').values.ravel()


# In[3]:


X_train = np.array(X_train)
#print(X_train)
#print(np.shape(X_train))
X_test = np.array(X_test)
#print(X_test)
Y_train = np.array(Y_train)
#print(Y_train)
Y_test = np.array(Y_test)
#print(Y_test)


# In[4]:


x = np.zeros(6, dtype=float) #пустые массивы для создания графика
y = np.zeros(6, dtype=float)


# 1. ДЕРЕВО РЕШЕНИЙ

# 1.1. Дерево решений для классификации

# In[5]:


from sklearn.tree import DecisionTreeClassifier  
classifier = DecisionTreeClassifier() #создаем дерево решений
classifier.fit(X_train, Y_train) #обучаем классификатор


# In[6]:


y_pred = classifier.predict(X_test) #делаем прогноз по тестовым данным


# In[7]:


#set(Y_test) - set(y_pred) #проверяю, какую метку не видит
#print(X_test,Y_test)


# In[8]:


from sklearn.metrics import classification_report, confusion_matrix #оценка алгоритма
print(confusion_matrix(Y_test, y_pred))
print(classification_report(Y_test, y_pred))


# In[9]:


x[0] = precision_score(Y_test, y_pred, average='macro')
y[0] = recall_score(Y_test, y_pred, average='macro')
#print(x, y)


# 1.2. Дерево решений для регрессии

# In[10]:


from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X_train, Y_train)


# In[11]:


y_pred = regressor.predict(X_test)


# In[12]:


from sklearn import metrics
print('Средняя абсолютная ошибка:', metrics.mean_absolute_error(Y_test, y_pred))
print('Среднеквадратичная ошибка:', metrics.mean_squared_error(Y_test, y_pred))
print('Ошибка корневого квадрата:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))


# 2. SVM (support vector machine)

# 2.1. Линейный способ

# In[13]:


from sklearn.svm import SVC #обучение алгоритма
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, Y_train) 


# In[14]:


y_pred = svclassifier.predict(X_test) #делаем прогноз


# In[15]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(Y_test,y_pred)) #матрица неточностей
print(classification_report(Y_test,y_pred))


# In[16]:


x[1] = precision_score(Y_test, y_pred, average='macro')
y[1] = recall_score(Y_test, y_pred, average='macro')
#print(x, y)


# 2.2. Реализация SVM ядра

# 2.2.1. Полиномиальное ядро

# In[17]:


from sklearn.svm import SVC
svclassifier = SVC(kernel='poly', degree=2)
svclassifier.fit(X_train, Y_train)


# In[18]:


y_pred = svclassifier.predict(X_test) #прогноз


# In[19]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(Y_test, y_pred))
print(classification_report(Y_test, y_pred))


# In[20]:


x[2] = precision_score(Y_test, y_pred, average='macro')
y[2] = recall_score(Y_test, y_pred, average='macro')
#print(x, y)


# 2.2.2. Гауссово ядро

# In[21]:


from sklearn.svm import SVC
svclassifier = SVC(kernel='rbf')
svclassifier.fit(X_train, Y_train)


# In[22]:


y_pred = svclassifier.predict(X_test)


# In[23]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(Y_test, y_pred))
print(classification_report(Y_test, y_pred))


# In[24]:


x[3] = precision_score(Y_test, y_pred, average='macro')
y[3] = recall_score(Y_test, y_pred, average='macro')
#print(x, y)


# 2.2.3. Cигмоидное ядро

# In[25]:


from sklearn.svm import SVC
svclassifier = SVC(kernel='sigmoid')
svclassifier.fit(X_train, Y_train)


# In[26]:


y_pred = svclassifier.predict(X_test)


# In[27]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(Y_test, y_pred))
print(classification_report(Y_test, y_pred))


# In[28]:


x[4] = precision_score(Y_test, y_pred, average='macro')
y[4] = recall_score(Y_test, y_pred, average='macro')
#print(x, y)


# 3. KNN

# 3.1. Метод K-ближайших соседей

# In[29]:


from sklearn.neighbors import KNeighborsClassifier


# In[30]:


model = KNeighborsClassifier(n_neighbors = 1) #передаем параметр k


# In[31]:


model.fit(X_train, Y_train) #обучаем модель


# In[32]:


predictions = model.predict(X_test) #делаем предсказание


# In[33]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[34]:


print(classification_report(Y_test, predictions)) #оценка точности


# In[35]:


x[5] = precision_score(Y_test, predictions, average='macro')
y[5] = recall_score(Y_test, predictions, average='macro')
#print(x, y)


# In[36]:


print(confusion_matrix(Y_test, predictions)) #матрица ошибок


# Метод локтя для определения оптимального k

# In[37]:


error_rates = [] #в этот список будем добавлять частоту ошибок


# In[38]:


for i in np.arange(1, 101): #перебираем различные значения k
    new_model = KNeighborsClassifier(n_neighbors = i)
    new_model.fit(X_train, Y_train)
    new_predictions = new_model.predict(X_test)
    error_rates.append(np.mean(new_predictions != Y_test))


# In[39]:


plt.plot(error_rates) #визуализируем частоту ошибок


# Сравнение результатов

# In[40]:


plt.scatter(x[0], y[0], label = 'Дерево решений', alpha=0.5)
plt.scatter(x[1], y[1], label = 'SVM: Линейный способ', alpha=0.5)
plt.scatter(x[2], y[2], label = 'SVM: Полиномиальное ядро', alpha=0.5)
plt.scatter(x[3], y[3], label = 'SVM: Гауссово ядро', alpha=0.5)
plt.scatter(x[4], y[4], label = 'SVM: Cигмоидное ядро', alpha=0.5)
plt.scatter(x[5], y[5], label = 'KNN', alpha=0.5)
#plt.title('Сравнение различных методов', fontsize=14)
plt.xlabel('Precision', color='gray', fontsize=13)
plt.ylabel('Recall', color='gray', fontsize=13)


plt.legend(fontsize=10, loc = 'upper left')
plt.grid()
plt.xlim([0.2, 1.02])
plt.ylim([0.45, 1.02])
plt.savefig(r'D:\practis\video_combined\Сравнение различных способов (32).jpg', dpi = 100) #поменять адрес
plt.show()


# In[ ]:




