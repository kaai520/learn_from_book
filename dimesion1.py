import numpy as np
import matplotlib.pyplot as plt


#一维随机漫步算法
n_person = 2000
n_times=500
t=np.arange(n_times)
#只产生-1 和 1的随机数，random_integers 与randint最大的差别就是前者是全闭，后者是左闭右开
steps=2*np.random.random_integers(0,1,size=(n_person,n_times))-1
amount = np.cumsum(steps,axis=1)
mean_sd_amount=np.mean(amount**2,axis=0)
plt.plot(np.sqrt(mean_sd_amount),'b')
plt.plot(np.sqrt(t),'r')
plt.show()
