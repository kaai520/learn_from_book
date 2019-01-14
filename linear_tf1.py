import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#一元线性回归,数据集
train_X = np.linspace(-1,1,100)
train_Y = 2*train_X+np.random.randn(*train_X.shape)*0.3


#占位符
X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)

#模型参数
W=tf.Variable(tf.random_normal([1]),name='weight')
b=tf.Variable(tf.zeros([1]),name='bias')
#前向结构
z=tf.multiply(X,W)+b

#反向优化
cost=tf.reduce_mean(tf.square(Y-z))
rate=1e-3
optimizer=tf.train.GradientDescentOptimizer(rate).minimize(cost)

init=tf.global_variables_initializer()
train_epochs=80
display_step=1
#真正运行
with tf.Session() as sess:
    sess.run(init)
    plotdata={'epoch':[],'loss':[]}
    for epoch in range(train_epochs):
        for (x,y) in zip(train_X,train_Y):
            sess.run(optimizer,feed_dict={X:x,Y:y})

        if epoch%display_step==0:
            loss=sess.run(cost,feed_dict={X:train_X,Y:train_Y})
            print('Epoch:',epoch+1,'cost=',loss,'W=',sess.run(W),'b=',sess.run(b))
            if not (loss=='NA'):
                plotdata['epoch'].append(epoch)
                plotdata['loss'].append(loss)
    print('Finished!')
    print('cost=',sess.run(cost,feed_dict={X:train_X,Y:train_Y}),'W=',sess.run(W),'b=',sess.run(b))
    # 可视化
    plt.subplot(121)
    plt.plot(train_X, train_Y, 'ro', label='original_data')
    plt.plot(train_X, train_X *sess.run(W)+sess.run(b),'b-',label='Fittedline')
    plt.legend()
    plt.grid()
    plt.subplot(122)
    plt.plot(plotdata['epoch'],plotdata['loss'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()




