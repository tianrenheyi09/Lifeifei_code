import tensorflow as tf
import numpy as np
from functools import partial

(X_train,y_train),(X_test,y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1,28*28)/255.0
X_test = X_test.astype(np.float32).reshape(-1,28*28)/255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]



n_inputs = 28*28
n_hidden1  = 300
n_hidden2 = 100
n_outputs = 10

X = tf.placeholder(tf.float32,shape=(None,n_inputs),name='X')
y = tf.placeholder(tf.int32,shape = (None),name='y')

batch_norm_momentum = 0.9
training = tf.placeholder_with_default(False,shape=(),name='training')

with tf.name_scope('dnn'):
    he_init = tf.variance_scaling_initializer()
    my_batch_norm_layer = partial(
        tf.layers.batch_normalization,
        training = training,
        momentum = batch_norm_momentum
    )
    my_dense_layer = partial(
        tf.layers.dense,
        kernel_initializer=he_init
    )
    hidden1 = my_dense_layer(X,n_hidden1,name='hidden1')
    bn1 = tf.nn.elu(my_batch_norm_layer(hidden1))
    hidden2 = my_dense_layer(bn1,n_hidden2,name='hidden2')
    bn2 = tf.nn.elu(my_batch_norm_layer(hidden2))
    logits_before_bn = my_dense_layer(bn2,n_outputs,name='outputs')
    logits = my_batch_norm_layer(logits_before_bn)

with tf.name_scope('loss'):
    x_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
    loss = tf.reduce_mean(x_entropy,name='loss')

learning_rate = 0.01

with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    extra_updates_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_updates_ops):
        training_op = optimizer.minimize(loss)


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

def get_batch( Xi,  y, batch_size, index):
    start = index * batch_size
    end = (index + 1) * batch_size
    end = end if end < len(y) else len(y)
    return Xi[start:end], [y_ for y_ in y[start:end]]

def shuffle_in_unison_scary( a, c):
    res = list(zip(a,c))
    np.random.shuffle(res)
    a,c = zip(*res)

acc_train = []
acc_valid = []


sess = tf.Session()

sess.run(tf.global_variables_initializer())

# for epoch in range(n_epoches):
#     for X_batch,y_batch in shuffle_batch(X_train,y_train,batch_size):
#         sess.run(training_op,feed_dict={training:True,X:X_batch,y:y_batch})
#     acc_batch  =sess.run(accuracy,feed_dict={X:X_batch,y:y_batch})
#     acc_train.append(acc_batch)
#     acc_val = sess.run(accuracy,feed_dict={X:X_valid,y:y_valid})
#     acc_valid.append(acc_val)
#     print(epoch,'batch acc',acc_batch,'val acc:',acc_val)

for epoch in range(n_epoches):
    shuffle_in_unison_scary(X_train, y_train)
    total_batch = int((len(y_train) - 1) / batch_size) + 1
    for i in range(total_batch):
        X_batch, y_batch = get_batch(X_train, y_train, batch_size, i)
        feed_dict = {training: True, X: X_batch, y: y_batch}
        opt = sess.run((training_op), feed_dict=feed_dict)

    acc_batch = sess.run(accuracy, feed_dict={X: X_batch, y: y_batch})
    acc_train.append(acc_batch)
    acc_val = sess.run(accuracy, feed_dict={X: X_valid, y: y_valid})
    acc_valid.append(acc_val)
    print(epoch + 1, 'batch acc', acc_batch, 'val acc:', acc_val)


save_path = saver.save(sess,'./my_model.ckpt')


import matplotlib.pyplot as plt
plt.figure()
plt.plot(acc_train,'-o',label='train')
plt.plot(acc_valid,'-*',label='val')
plt.xlabel('epoch')
plt.legend(loc='upper left')#图例位置