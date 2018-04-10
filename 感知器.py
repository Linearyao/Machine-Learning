
# coding: utf-8

# In[37]:


import numpy as np
class Perceptron(object):
    def __init__(self,eta = 0.01,n_iter=10):
        """
        eta:学习率
        n_iter:权重向量的训练次数
        w_:神经分叉权重向量
        errors_:用于记录神经元判断出错次数
        """
        self.eta = eta;
        self.n_iter = n_iter
        pass
    def fit(self,x,y):
        """
        训练神经元，x输入样本向量，y对应样本分类
        """
        self.w_ = np.zero(1 + x.shape[1]);
        self.error_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi,target in zip(x,y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update;
                
                errors += int(update !=0.0)
                self.errors_.append(errors)
                pass
            
            pass
        def net_input(self,x):
            return np.dot(x,self.w_[1:] + self.w_[0])
            pass
        def predict(self,x):
            return np.where(self.net_input(x) >= 0.0 , 1, -1)
            pass
        pass


# In[38]:


file = "F:\jumpy\iris.csv"
import pandas as pd
df = pd.read_csv(file,header=None)
df.head(10)


# In[39]:


import matplotlib.pyplot as plt
import numpy as np

y = df.loc[0:100,4].values
y = np.where(y == 'Iris-setosa',-1,1)

x = df.iloc[0:100,[0,2]].values

plt.scatter(x[:50,0],x[:50,1],color='red',marker='o',label='setosa')
plt.scatter(x[50:100,0],x[50:100,1],color='blue',marker='x',label='sersicolor')
plt.xlabel('花瓣长度')
plt.ylabel('花茎长度')
plt.legend(loc='upper left')
plt.show()


# In[42]:


ppn = Perceptron(eta=0.1,n_iter=10)
ppn.fit(x,y)
plt.plot(range(1,len(ppn.errors_) + 1),ppn.errors_,marker='0')
plt.xlabel('Epochs')
plt.ylabel('错误分类次数')


# In[25]:


from matplotlib.colors import ListedColormap
def plot_decision_regions(x,y,classifier,resolution=0.02):
    marker = ('s','x','o','v')
    colors = ('red','blue','lightgreen','gray','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    x1_min,x1_max = x[:,0].min() - 1,x[:,0].max()
    x2_min,x2_max = x[:,1].min() - 1,x[:,1].max()
    
#     print(x1_min,x1_max)
#     print(x2_min,x2_max)
#     xxl,xx2 = np.meshgrid(np.arrage(x1_min,x1_max,resolution),np.arrage(x2_min,x2_max,resolution))
    
#     print(np.arange(x1_min,x1_max,resolution).shape)
#     print(np.arange(x1_min,x1_max,resolution))
#     print(xxl.shape)
#     print(xxl)


# In[26]:

plot_decision_regions(x,y,ppn,resolution=0.02)

# 适用于样本为可分割样本空间的数集，例如0-1分布
