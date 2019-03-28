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

dropout_rate = 0.5#####1-keep_prob
training = tf.placeholder_with_default(False,shape=(),name='training')

he_init = tf.variance_scaling_initializer()
# def dnn(inputs,n_hidden_layerss=5,n_neurons=100,name=None,activation=tf.nn.relu,initializer=he_init):
#     with tf.name_scope('dnn'):
#         for layer in range(n_hidden_layerss):
#             inputs = tf.layers.dense(inputs,n_neurons,activation=activation,
#                                      kernel_initializer=initializer,
#                                      name='hidden'+str(layer+1))
#
#         return inputs

def dnn_one(inputs,hidden_dims,name=None,activation=tf.nn.relu,initializer=he_init):
    with tf.name_scope('dnn1'):
        for i in range(len(hidden_dims)):
            inputs = tf.layers.dense(inputs,hidden_dims[i],activation=activation,
                                     kernel_initializer=initializer,
                                     name='hidden'+str(i+1))

        return inputs

dnn_outputs = dnn_one(X,[200,100,50])
logits = tf.layers.dense(dnn_outputs,n_outputs,kernel_initializer=he_init,name='logits')
y_prob = tf.nn.softmax(logits,name='y_prob')



with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")


learning_rate = 0.01

with tf.name_scope('train'):
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
    training_op = optimizer.minimize(loss)


with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits,y,1)
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epoches = 30
batch_size = 200

max_checks_without_progress = 5
checks_without_progress = 0
best_loss = np.infty


def shuffle_batch(X,y,batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X)//batch_size
    for batch_idx in np.array_split(rnd_idx,n_batches):
        X_batch,y_batch = X[batch_idx],y[batch_idx]
        yield X_batch,y_batch

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epoches):
        for X_batch,y_batch in shuffle_batch(X_train,y_train,batch_size):
            sess.run(training_op,feed_dict={X:X_batch,y:y_batch})

        loss_val,acc_val = sess.run([loss,accuracy],feed_dict={X:X_valid,y:y_valid})
        if loss_val<best_loss:
            save_path = saver.save(sess,'./my_mnist_model_early.ckpt')
            best_loss = loss_val
            checks_without_progress = 0
        else:
            checks_without_progress += 1
            if checks_without_progress>max_checks_without_progress:
                print('early stopping!')
                break

        print('{}\tValidation loss {:.6f}\tBest loss:{:.6f}\tAccuracy:{:.2f}%'.format(
            epoch,loss_val,best_loss,acc_val*100
        ))


with tf.Session() as sess:
    saver.restore(sess,'./my_mnist_model_early.ckpt')
    acc_test = accuracy.eval(feed_dict={X:X_test,y:y_test})
    print('fina test acc:{:.2f}%'.format(acc_test*100))
