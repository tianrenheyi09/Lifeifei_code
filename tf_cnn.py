import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt

from cs231n.data_utils import load_CIFAR10
def get_CIFAR10_data(num_training=49000,num_validation=1000,num_test=10000):
    # Load the raw CIFAR-10 data
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    return X_train, y_train, X_val, y_val, X_test, y_test

X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

##claear old variables
tf.reset_default_graph()

X = tf.placeholder(tf.float32,[None,32,32,3])
y = tf.placeholder(tf.int64,[None])
is_training = tf.placeholder(tf.bool)

def simple_model(X,y):

    Wconv1 = tf.get_variable('Wconv1',shape=[7,7,3,32])
    bconv1 = tf.get_variable('bconv1',shape=[32])
    W1 = tf.get_variable('W1',shape=[5408,10])
    b1 = tf.get_variable('b1',shape=[10])
    ###define graph
    a1 = tf.nn.conv2d(X,Wconv1,strides=[1,2,2,1],padding='VALID')+bconv1###[None,13,13,32]
    h1 = tf.nn.relu(a1)
    h1_hat = tf.reshape(h1,[-1,5408])
    y_out = tf.matmul(h1_hat,W1)+b1
    return y_out

y_out = simple_model(X,y)

####loss
total_loss = tf.losses.hinge_loss(tf.one_hot(y,10),logits=y_out)
mean_loss = tf.reduce_mean(total_loss)
#######optimizer
optimizer = tf.train.AdamOptimizer(5e-4)
train_step = optimizer.minimize(mean_loss)

#######accuracy
correct_prediction = tf.equal(tf.argmax(y_out,1),y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


def shuffle_batch( X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

batch_size = 64
epochs = 1
init = tf.global_variables_initializer()


def run_model(X_train,y_train,X_test,y_test,epochs,batch_size):

    loss_train = []
    loss_test = []
    acc_train = []
    acc_test = []
    for e in range(epochs):
        iter_cnt = 0
        trans_loss = 0
        trans_acc = 0
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            feed = {X: X_batch, y: y_batch, is_training: True}
            loss_tr, acc_tr, _ = sess.run([mean_loss, accuracy, train_step], feed_dict=feed)
            iter_cnt += 1
            trans_loss += loss_tr
            trans_acc += acc_tr
        loss_rt = trans_loss / iter_cnt
        acc_rt = trans_acc / iter_cnt
        loss_train.append(loss_rt)
        acc_train.append(acc_rt)
        ############test  evalutation
        loss_te, acc_te = sess.run([mean_loss, accuracy], feed_dict={X: X_test, y: y_test, is_training: False})
        loss_test.append(loss_te)
        acc_test.append(acc_te)
        # print('epoch:%d batch loss:%.3f test loss:%.3f batch acc:%.3f test acc:%.3f'%(e, loss_rt, loss_te,acc_rt,acc_te))
        print('epoch:{:d} batch loss{:.3f} test loss{:.3f} batch acc{:.3f} test acc:{:.3f}'.format(
            e + 1, loss_rt, loss_te, acc_rt, acc_te))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print('training')
run_model(X_train,y_train,X_val,y_val,epochs,batch_size)


#
# with tf.Session() as sess:
#     init.run()
#     loss_train = []
#     loss_test = []
#     acc_train = []
#     acc_test = []
#
#     for e in range(epochs):
#
#         iter_cnt = 0
#         trans_loss = 0
#         trans_acc = 0
#         for X_batch,y_batch in shuffle_batch(X_train,y_train,batch_size):
#             feed = {X:X_batch,y:y_batch,is_training:True}
#             loss_tr,acc_tr, _ = sess.run([mean_loss,accuracy, train_step], feed_dict=feed)
#             iter_cnt +=1
#             trans_loss +=loss_tr
#             trans_acc +=acc_tr
#         loss_rt = trans_loss/iter_cnt
#         acc_rt = trans_acc/iter_cnt
#         loss_train.append(loss_rt)
#         acc_train.append(acc_rt)
#         ############test  evalutation
#         loss_te,acc_te = sess.run([mean_loss,accuracy],feed_dict={X:X_test,y:y_test,is_training:False})
#         loss_test.append(loss_te)
#         acc_test.append(acc_te)
#         # print('epoch:%d batch loss:%.3f test loss:%.3f batch acc:%.3f test acc:%.3f'%(e, loss_rt, loss_te,acc_rt,acc_te))
#         print('epoch:{:d} batch loss{:.3f} test loss{:.3f} batch acc{:.3f} test acc:{:.3f}'.format (
#         e+1, loss_rt, loss_te, acc_rt, acc_te))
#

# plt.figure()
# plt.plot(loss_train,color='red', linestyle="solid", marker="o",label='loss_train')
# plt.plot(loss_test,color='blue', linestyle="dashed", marker="*",label='loss_test')
# plt.xlabel('epochs')
# plt.legend()

########clear old variables
tf.reset_default_graph()
# define our input (e.g. the data that changes every batch)
# The first dim is None, and gets sets automatically based on batch size fed in
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)

def complex_model(X,y,is_training):
    Wconv1 = tf.get_variable('Wconv1',shape=[7,7,3,32])
    bconv1 = tf.get_variable('bconv1',shape=[32])

    W1 = tf.get_variable('W1',shape=[5408,1024])
    b1 = tf.get_variable('b1',shape=[1024])

    W2 = tf.get_variable('W2',shape=[1024,10])
    b2 = tf.get_variable('b2',shape=[10])

    a1 = tf.nn.conv2d(X,Wconv1,strides=[1,1,1,1],padding='VALID')+bconv1
    h1 = tf.nn.relu(a1)
    num_filters = 32
    axes = [0,1,2]
    mean,var = tf.nn.moments(h1,axes=axes)
    offset = tf.Variable(tf.zeros([num_filters]))
    scale = tf.Variable(tf.ones([num_filters]))
    eps = 0.0001
    bn1 = tf.nn.batch_normalization(h1,mean,var,offset=offset,scale=scale,variance_epsilon=eps)

    pool1 = tf.nn.max_pool(bn1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    p1_flat = tf.reshape(pool1,[-1,5408])
    y1_out = tf.matmul(p1_flat,W1)+b1
    h2 = tf.nn.relu(y1_out)
    h2_flat = tf.reshape(h2,[-1,1024])
    y2_out = tf.matmul(h2_flat,W2)+b2

    return y2_out

y_out = complex_model(X,y,is_training)
x = np.random.randn(64,32,32,3)

import time

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    time1 = time.time()
    ans = sess.run(y_out,feed_dict={X:x,is_training:True})
    time2 = time.time()
    print('take time:%.3f'%(time2-time1))
    print(ans.shape)
    print(np.array_equal(ans.shape, np.array([64, 10])))


##########c尝试下GPU
try:
    with tf.Session() as sess:
        with tf.device("/gpu:0") as dev: #"/cpu:0" or "/gpu:0"
            tf.global_variables_initializer().run()

            ans = sess.run(y_out,feed_dict={X:x,is_training:True})

except tf.errors.InvalidArgumentError:
    print("no gpu found, please use Google Cloud if you want GPU acceleration")
    # rebuild the graph
    # trying to start a GPU throws an exception
    # and also trashes the original graph
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y = tf.placeholder(tf.int64, [None])
    is_training = tf.placeholder(tf.bool)
    y_out = complex_model(X,y,is_training)

total_loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(y,10),logits=y_out)
mean_loss = tf.reduce_mean(total_loss)
optimizer = tf.train.RMSPropOptimizer(1e-3)

extra_updates_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_updates_ops):
    train_step = optimizer.minimize(mean_loss)

correct_prediction = tf.equal(tf.argmax(y_out,1), y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print('training')

def run_model(X_train,y_train,X_test,y_test,epochs,batch_size):

    loss_train = []
    loss_test = []
    acc_train = []
    acc_test = []
    for e in range(epochs):
        iter_cnt = 0
        trans_loss = 0
        trans_acc = 0
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            feed = {X: X_batch, y: y_batch, is_training: True}
            loss_tr, acc_tr, _ = sess.run([mean_loss, accuracy, train_step], feed_dict=feed)
            iter_cnt += 1
            trans_loss += loss_tr
            trans_acc += acc_tr
        loss_rt = trans_loss / iter_cnt
        acc_rt = trans_acc / iter_cnt
        loss_train.append(loss_rt)
        acc_train.append(acc_rt)
        ############test  evalutation
        loss_te, acc_te = sess.run([mean_loss, accuracy], feed_dict={X: X_test, y: y_test, is_training: False})
        loss_test.append(loss_te)
        acc_test.append(acc_te)
        # print('epoch:%d batch loss:%.3f test loss:%.3f batch acc:%.3f test acc:%.3f'%(e, loss_rt, loss_te,acc_rt,acc_te))
        print('epoch:{:d} batch loss{:.3f} test loss{:.3f} batch acc{:.3f} test acc:{:.3f}'.format(
            e + 1, loss_rt, loss_te, acc_rt, acc_te))

print('validation')
run_model(X_train,y_train,X_val,y_val,1,64)

def conv33_relu_batch(inputs,num_filters):
    regularizer = tf.contrib.layers.l2_regularizer(scale = 0.2)
    conv1 = tf.layers.conv2d(
        inputs = inputs,
        filters=num_filters,
        strides=[1,1],
        kernel_size=[3,3],
        padding='same',
        kernel_initializer=regularizer
    )
    relu1 = tf.nn.relu(conv1)
    batch_norm1 = tf.layers.batch_normalization(relu1,axis=1)
    dropout1 = tf.layers.dropout(inputs=batch_norm1,rate=0.5)
    conv2 = tf.layers.conv2d(
        inputs = dropout1,
        filters=2*num_filters,
        strides=[1,1],
        kernel_size=[3,3],
        padding='same',
        kernel_initializer=regularizer
    )
    relu2 = tf.nn.relu6(conv2)
    batch_norm2 = tf.layers.batch_normalization(relu2,axis=1)
    pool = tf.layers.max_pooling2d(inputs=batch_norm2,pool_size=[2,2],strides=2)
    dropout2 = tf.layers.dropout(pool)
    return dropout2

def my_model(X,y,is_training):
    ######conv-relu-conv-relu->global average pool->softmax
    num_classes = 10
    nn1 = conv33_relu_batch(X,64)
    nn2 = conv33_relu_batch(nn1,128)
    nn3 = conv33_relu_batch(nn2,256)

    pool_size = (nn3.shape[1],nn2.shape[2])
    pool_ave = tf.layers.average_pooling2d(
        inputs=nn3,
        pool_size=pool_size,
        strides=[1,1],
        padding='VALID',
        data_format='channels_last'
    )
    pool_ave_flat_size = pool_ave.shape[1]*pool_ave.shape[2]*pool_ave.shape[3]
    pool_ave_flat = tf.reshape(pool_ave,[-1,pool_ave_flat_size])
    dropout =  tf.layers.dropout(inputs=pool_ave_flat,rate=0.5,training=is_training)

    y_out = tf.layers.dense(inputs=dropout,units=num_classes)
    return y_out

tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)

y_out = my_model(X,y,is_training)
total_loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(y,10),logits=y_out)
mean_loss = tf.reduce_mean(total_loss)
########学习率衰减
global_step = tf.Variable(0,trainable=False)
initial_learning_rate = 1e-2
learning_rate = tf.train.exponential_decay(initial_learning_rate,global_step=global_step,
                                           decay_steps=50,decay_rate=0.9)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

extra_updates_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_updates_ops):
    train_step = optimizer.minimize(mean_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
run_model(X_train,y_train,X_val,y_val,5,64)


