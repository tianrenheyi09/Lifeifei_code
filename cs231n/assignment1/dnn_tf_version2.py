import tensorflow as tf
import numpy as np



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


with tf.name_scope('dnn'):
    hidden1 = tf.layers.dense(X,n_hidden1,name='hidden1',activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1,n_hidden2,name='hidden2',activation=tf.nn.relu)
    logits = tf.layers.dense(hidden2,n_outputs,name='outputs')

with tf.name_scope('loss'):
    x_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
    loss = tf.reduce_mean(x_entropy,name='loss')

learning_rate = 0.01

with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits,y,1)
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epoches = 40
batch_size = 50

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
        acc_batch  =accuracy.eval(feed_dict={X:X_batch,y:y_batch})
        acc_val = accuracy.eval(feed_dict={X:X_valid,y:y_valid})
        print(epoch,'batch acc',acc_batch,'val acc:',acc_val)

    save_path = saver.save(sess,'./my_model.ckpt')

with tf.Session() as sess:
    saver.restore(sess,'./my_model.ckpt')
    X_new_scaled = X_test[:20]
    Z = logits.eval(feed_dict={X:X_new_scaled})
    y_pred = np.argmax(Z,axis=1)

print('pred classes:',y_pred)
print('actual classes:',y_test[:20])



