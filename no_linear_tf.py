import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs,in_size,out_size,activation_function=None):
    #inputs.shape=(n_samples,in_size)
    W=tf.Variable(tf.random_normal([in_size,out_size]))
    b=tf.Variable(tf.zeros([1,out_size])+1e-2)
    Wx_plus_b=tf.matmul(inputs,W)+b
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    return outputs

X_train=np.linspace(-1, 1, 300).reshape(-1, 1)
y_train= np.square(X_train) + 0.5 * X_train + 0.5

X=tf.placeholder(tf.float32, [None, 1])
Y=tf.placeholder(tf.float32, [None, 1])

#两层隐藏层节点都为10，激活函数选的是softplus
l1=add_layer(X, X_train.shape[1], 10, tf.nn.softplus)
l2=add_layer(l1,10,10,tf.nn.softplus)
prediction=add_layer(l2,10,1)
loss=tf.reduce_mean(tf.square(Y - prediction))

learning_rate=1e-2
train_epochs=300
display_step=10
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    plotdata={'epoch':[],'loss':[]}
    for epoch in range(train_epochs):
        #全集梯度下降
        sess.run(optimizer, feed_dict={X:X_train, Y:y_train})
        if epoch%display_step==0:
            train_loss=sess.run(loss, feed_dict={X:X_train, Y:y_train})
            if not (train_loss== 'NA'):
                plotdata['epoch'].append(epoch)
                plotdata['loss'].append(train_loss)
                print('Epoch:', epoch,'loss=', sess.run(loss, feed_dict={X: X_train, Y: y_train}))
    print('Finished!')
    print('loss=', sess.run(loss, feed_dict={X:X_train, Y:y_train}))

    plt.subplot(121)
    plt.plot(X_train, y_train, 'r-', label='original_data')
    plt.plot(X_train, sess.run(prediction, feed_dict={X:X_train}), 'b-', label='Fittedline')
    plt.legend()
    plt.grid()
    plt.subplot(122)
    plt.plot(plotdata['epoch'], plotdata['loss'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
