import tensorflow as tf
import numpy as np


(X_train,y_train),(X_test,y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1,28*28)/255.0
X_test = X_test.astype(np.float32).reshape(-1,28*28)/255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]



from itertools import count
from collections import defaultdict
from scipy.sparse import csr
import numpy as np
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
import tensorflow as tf
from tqdm import tqdm_notebook as tqdm

#
# def vectorize_dic(dic,ix=None,p=None,n=0,g=0):
#     """
#     dic -- dictionary of feature lists. Keys are the name of features
#     ix -- index generator (default None)
#     p -- dimension of feature space (number of columns in the sparse matrix) (default None)
#     """
#     if ix==None:
#         ix = dict()
#
#     nz = n * g
#
#     col_ix = np.empty(nz,dtype = int)
#
#     i = 0
#     for k,lis in dic.items():
#         for t in range(len(lis)):
#             ix[str(lis[t]) + str(k)] = ix.get(str(lis[t]) + str(k),0) + 1
#             col_ix[i+t*g] = ix[str(lis[t]) + str(k)]
#         i += 1
#
#     row_ix = np.repeat(np.arange(0,n),g)
#     data = np.ones(nz)
#     if p == None:
#         p = len(ix)
#
#     ixx = np.where(col_ix < p)
#     return csr.csr_matrix((data[ixx],(row_ix[ixx],col_ix[ixx])),shape=(n,p)),ix
#
#
# def batcher(X_, y_=None, batch_size=-1):
#     n_samples = X_.shape[0]
#
#     if batch_size == -1:
#         batch_size = n_samples
#     if batch_size < 1:
#        raise ValueError('Parameter batch_size={} is unsupported'.format(batch_size))
#
#     for i in range(0, n_samples, batch_size):
#         upper_bound = min(i + batch_size, n_samples)
#         ret_x = X_[i:upper_bound]
#         ret_y = None
#         if y_ is not None:
#             ret_y = y_[i:i + batch_size]
#             yield (ret_x, ret_y)
#
#
# cols = ['user','item','rating','timestamp']
#
# train = pd.read_csv('data/ua.base',delimiter='\t',names = cols)
# test = pd.read_csv('data/ua.test',delimiter='\t',names = cols)
#
# x_train,ix = vectorize_dic({'users':train['user'].values,
#                             'items':train['item'].values},n=len(train.index),g=2)
#
#
# x_test,ix = vectorize_dic({'users':test['user'].values,
#                            'items':test['item'].values},ix,x_train.shape[1],n=len(test.index),g=2)
#
#
# print(x_train)
# y_train = train['rating'].values
# y_test = test['rating'].values
#
# x_train = x_train.todense()
# x_test = x_test.todense()
#
# print(x_train)
#
# print(x_train.shape)
# print (x_test.shape)

# n_inputs = x_train.shape[1]
# n_hidden1  = 50
# n_hidden2 = 50
# n_outputs =5
#
# base = min(y_train)
#
# y_train = y_train-base
# y_test = y_test-base
# y_train = y_train.astype(np.int32)
# y_test = y_test.astype(np.int32)

n_inputs = 28*28
n_hidden1  = 300
n_hidden2 = 100
n_outputs = 10

X = tf.placeholder(tf.float32,shape=(None,n_inputs),name='X')
y = tf.placeholder(tf.int32,shape = (None),name='y')

def neuron_layer(X,n_neurons,name,activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2/np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs,n_neurons),stddev=stddev)
        W = tf.Variable(init,name='kernel')
        b = tf.Variable(tf.zeros([n_neurons]),name='bias')
        Z = tf.matmul(X,W)+b
        if activation is not None:
            return activation(Z)
        else:
            return Z

with tf.name_scope('dnn'):
    hidden1 = neuron_layer(X,n_hidden1,name='hidden1',activation=tf.nn.relu)
    hidden2 = neuron_layer(hidden1,n_hidden2,name='hidden2',activation=tf.nn.relu)
    logits = neuron_layer(hidden2,n_outputs,name='outputs')

with tf.name_scope('loss'):
    x_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
    loss = tf.reduce_mean(x_entropy,name='loss')

# learning_rate = 0.01
# with tf.name_scope('train'):
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
#     training_op = optimizer.minimize(loss)

########学习率衰减
with tf.name_scope('train'):
    initial_learning_rate = 0.1
    decay_steps = 10000
    decay_rate = 0.1
    global_step = tf.Variable(0,trainable=False,name='global_step')
    learning_rate = tf.train.exponential_decay(initial_learning_rate,global_step,decay_steps,decay_rate)
    optimizer = tf.train.MomentumOptimizer(learning_rate,momentum=0.9)
    training_op = optimizer.minimize(loss,global_step=global_step)


with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits,y,1)
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epoches = 20
batch_size = 200

def shuffle_batch(X,y,batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X)//batch_size
    for batch_idx in np.array_split(rnd_idx,n_batches):
        X_batch,y_batch = X[batch_idx],y[batch_idx]
        yield X_batch,y_batch

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epoches):
        for X_batch,y_batch in shuffle_batch(x_train,y_train,batch_size):
            sess.run(training_op,feed_dict={X:X_batch,y:y_batch})
        acc_batch  =accuracy.eval(feed_dict={X:X_batch,y:y_batch})
        # acc_val = accuracy.eval(feed_dict={X:X_valid,y:y_valid})
        acc_val = accuracy.eval(feed_dict={X: x_test, y: y_test})
        print(epoch,'batch acc',acc_batch,'val acc:',acc_val)

    save_path = saver.save(sess,'./my_model.ckpt')

# with tf.Session() as sess:
#     saver.restore(sess,'./my_model.ckpt')
#     X_new_scaled = X_test[:20]
#     Z = logits.eval(feed_dict={X:X_new_scaled})
#     y_pred = np.argmax(Z,axis=1)
#
# print('pred classes:',y_pred)
# print('actual classes:',y_test[:20])



