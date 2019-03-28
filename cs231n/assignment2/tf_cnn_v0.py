import numpy as np
import tensorflow as tf
import os
from __future__ import division, print_function, unicode_literals
from io import open
# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

# To plot pretty figures

import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "cnn"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

tf.reset_default_graph()

filter_primes = np.array([2., 3., 5., 7., 11., 13.], dtype=np.float32)
x = tf.constant(np.arange(1, 13+1, dtype=np.float32).reshape([1, 1, 13, 1]))
filters = tf.constant(filter_primes.reshape(1, 6, 1, 1))

valid_conv = tf.nn.conv2d(x, filters, strides=[1, 1, 5, 1], padding='VALID')
same_conv = tf.nn.conv2d(x, filters, strides=[1, 1, 5, 1], padding='SAME')

with tf.Session() as sess:
    print("VALID:\n", valid_conv.eval())
    print('valid_conv shape:',valid_conv.get_shape())
    print("SAME:\n", same_conv.eval())
    print('same_conv:',same_conv.get_shape())

########----------mnist cnn
tf.reset_default_graph()

height = 28
width = 28
channels = 1
n_inputs = height * width

conv1_fmaps = 32
conv1_ksize = 3
conv1_stride = 1
conv1_pad = "SAME"

conv2_fmaps = 64
conv2_ksize = 3
conv2_stride = 2
conv2_pad = "SAME"

pool3_fmaps = conv2_fmaps

n_fc1 = 64
n_outputs = 10

X = tf.placeholder(tf.float32,shape=[None,n_inputs],name='X')
X_reshaped = tf.reshape(X,shape=[-1,height,width,channels])
y = tf.placeholder(tf.int32,shape=[None],name='y')
training = tf.placeholder_with_default(False,shape=[],name='training')


conv1 = tf.layers.conv2d(
    inputs=X_reshaped,
    filters=conv1_fmaps,
    kernel_size=conv1_ksize,
    strides=conv1_stride,
    padding=conv1_pad,
    activation=tf.nn.relu,
    name='conv1'
)
conv2 = tf.layers.conv2d(
    inputs=conv1,
    filters=conv2_fmaps,
    kernel_size=conv2_ksize,
    strides=conv2_stride,
    padding=conv2_pad,
    activation=tf.nn.relu,
    name='conv2'
)

pool3 = tf.layers.max_pooling2d(
    inputs=conv2,
    pool_size=(2,2),
    strides=(2,2),
    padding='VALID',
    name='pool3'
)
pool3_flat = tf.reshape(pool3,shape=[-1,pool3.shape[1]*pool3.shape[2]*pool3.shape[3]])

fc1 = tf.layers.dense(pool3_flat,n_fc1,activation=tf.nn.relu,name='fc1')
logits = tf.layers.dense(fc1,n_outputs,name='output')
Y_proba = tf.nn.softmax(logits,name='Y_prob')

xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(y,n_outputs),logits=logits)
loss = tf.reduce_mean(xentropy)

optimizer = tf.train.AdamOptimizer()
training_op = optimizer.minimize(loss)

correct = tf.equal(tf.argmax(logits,1),tf.to_int64(y))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

def get_model_params():
    gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    return {gvar.op.name for gvar,value in zip(gvars,tf.get_default_session().run(gvars))}

def restore_model_params(model_params):
    gvar_names = list(model_params.keys())
    assign_ops = {gvar_name: tf.get_default_graph().get_operation_by_name(gvar_name + "/Assign")
                  for gvar_name in gvar_names}
    init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
    feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
    tf.get_default_session().run(assign_ops, feed_dict=feed_dict)

n_epochs = 20
batch_size = 50
iteration = 0

best_loss_val = np.infty
check_interval = 500
checks_since_last_progress = 0
max_check_without_progress = 20
best_model_params = None

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            iteration += 1
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch,training:True})
            if iteration % check_interval ==0:
                loss_val = loss.eval(feed_dict={X:X_valid,y:y_valid})
                if loss_val<best_loss_val:
                    best_loss_val = loss_val
                    checks_since_last_progress = 0
                    best_model_params = get_model_params()
                else:
                    checks_since_last_progress +=1
        acc_batch = accuracy.eval(feed_dict={X:X_batch,y:y_batch})
        acc_val = accuracy.eval(feed_dict={X:X_valid,y:y_valid})
        print('epoch %d ,last batch acc %.4f ,valid acc %.4f ,valid best loss %.6f'
              %(epoch+1,acc_batch,acc_val,best_loss_val))
        if checks_since_last_progress>max_check_without_progress:
            print('early stpping')
            break

    if best_model_params:
        restore_model_params(best_model_params)
    acc_test = accuracy.eval(feed_dict={X:X_test,y:y_test})
    print('final acc on test:',acc_test)
    save_path = saver.save(sess,'./my_mnist_nodel')





# with tf.Session() as sess:
#     init.run()
#     for epoch in range(n_epochs):
#         for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
#             sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
#         acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
#         acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
#         print(epoch, "Last batch accuracy:", acc_batch, "Test accuracy:", acc_test)
#
#         save_path = saver.save(sess, "./my_mnist_model")