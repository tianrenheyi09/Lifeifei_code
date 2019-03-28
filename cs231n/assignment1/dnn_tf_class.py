from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
import tensorflow as tf
import numpy as np


class DNNClassifier(BaseEstimator,ClassifierMixin):
    def __init__(self,n_hidden_layers=5,n_neurons=100,optimizer_class=tf.train.AdamOptimizer,
                 learning_rate=0.01,batch_size=20,activation=tf.nn.relu,initializer=tf.variance_scaling_initializer(),
                 batch_norm_momentum=None,dropout_rate=None,random_state=None):

        self.n_hidden_layers = n_hidden_layers
        self.n_neurons = n_neurons
        self.optimizer_class = optimizer_class
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.activation = activation
        self.initializer = initializer
        self.batch_norm_momentum = batch_norm_momentum
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        self._session = None
        # self._graph = tf.Graph()
        # self._training = tf.placeholder_with_default(False, shape=(), name='training')

    def dnn(self,inputs):
        for layer in range(self.n_hidden_layers):
            if self.dropout_rate:
                inputs = tf.layers.dropout(inputs,self.dropout_rate,training=self._training)
            inputs = tf.layers.dense(inputs,self.n_neurons,activation=self.activation,
                                     kernel_initializer=self.initializer,
                                     name='hidden'+str(layer+1))
            if self.batch_norm_momentum:
                inputs = tf.layers.batch_normalization(inputs,momentum=self.batch_norm_momentum,
                                                       training=self._training)
            inputs = self.activation(inputs,name='hidden'+str(layer+1))

        return inputs

    def build_graph(self,n_inputs,n_outputs):
        if self.random_state is not None:
            tf.set_random_seed(self.random_state)
            np.random.seed(self.random_state)

        X = tf.placeholder(tf.float32,shape=(None,n_inputs),name='X')
        y = tf.placeholder(tf.int32,shape=(None),name='y')

        if self.batch_norm_momentum or self.dropout_rate:
            self._training = tf.placeholder_with_default(False, shape=(), name='training')
        else:
            self._training = None

        dnn_outputs = self.dnn(X)
        logits = tf.layers.dense(dnn_outputs, n_outputs, kernel_initializer=tf.variance_scaling_initializer(), name='logits')
        y_prob = tf.nn.softmax(logits, name='y_prob')

        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
        loss = tf.reduce_mean(xentropy,name='loss')

        optimizer = self.optimizer_class(learning_rate=self.learning_rate)
        training_op = optimizer.minimize(loss)

        correct = tf.nn.in_top_k(logits,y,1)
        accuracy = tf.reduce_mean(tf.cast(correct,tf.float32),name='accuracy')

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        self._X,self._y = X,y
        self._Y_prob,self._loss = y_prob,loss
        self._training_op,self._accuracy = training_op,accuracy
        self._init,self._saver = init,saver

    def close_session(self):
        if self._session:
            self._session.close()

    def get_model_params(self):
        with self._graph.as_default():
            gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        return {gvar.op.name:value for gvar,value in zip(gvars,self._session.run(gvars))}
    def restore_model_params(self,model_parmas):
        gvars_names = list(model_parmas.keys())
        assign_ops = {gvar_name:self._graph.get_operation_by_name(gvar_name+'/Assign') for gvar_name in gvars_names}
        init_values = {gvar_name:assign_op.inputs[1] for gvar_name,assign_op in assign_ops.items()}
        feed_dict = {init_values[gvar_name]:model_parmas[gvar_name] for gvar_name in gvars_names}
        self._session.run(assign_ops,feed_dict=feed_dict)

    def shuffle_batch(self,X, y, batch_size):
        rnd_idx = np.random.permutation(len(X))
        n_batches = len(X) // batch_size
        for batch_idx in np.array_split(rnd_idx, n_batches):
            X_batch, y_batch = X[batch_idx], y[batch_idx]
            yield X_batch, y_batch


    def fit(self,X,y,n_epochs=100,X_valid=None,y_valid=None):
        self.close_session()

        n_inputs = X.shape[1]
        self.classes_ = np.unique(y)
        n_outputs = len(self.classes_)

        self.class_to_index = {label:index for index,label in enumerate(self.classes_)}
        y = np.array([self.class_to_index[label] for label in y],dtype=np.int32)

        self._graph = tf.Graph()
        with self._graph.as_default():
            self.build_graph(n_inputs,n_outputs)

            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        max_checks_without_progress = 20
        checks_without_progress = 0
        best_loss = np.infty
        #######--train the model
        self._session = tf.Session(graph=self._graph)
        with self._session.as_default() as sess:
            self._init.run()
            for epoch in range(n_epochs):
                for X_batch,y_batch in self.shuffle_batch(X,y,self.batch_size):
                    feed_dict = {self._X:X_batch,self._y:y_batch}
                    if self._training is not None:
                        feed_dict[self._training] = True
                    sess.run(self._training_op,feed_dict=feed_dict)
                    if extra_update_ops:
                        sess.run(extra_update_ops,feed_dict=feed_dict)

                if X_valid is not None and y_valid is not None:
                    loss_val,acc_val = sess.run([self._loss,self._accuracy],
                                                feed_dict={self._X:X_valid,
                                                           self._y:y_valid})

                    if loss_val<best_loss:
                        best_params = self.get_model_params()
                        best_loss = loss_val
                        checks_without_progress = 0
                    else:
                        checks_without_progress +=1

                    print('{}\tValidation loss {:.6f}\tBest loss:{:.6f}\tAccuracy:{:.2f}%'.format(
                        epoch, loss_val, best_loss, acc_val * 100
                    ))

                    if checks_without_progress > max_checks_without_progress:
                        print('early stopping!')
                        break

                else:
                    loss_train,acc_train = sess.run([self._loss,self._accuracy],
                                                    feed_dict={self._X:X_batch,
                                                               self._y:y_batch})
                    print("{}\tLast training batch loss: {:.6f}\tAccuracy: {:.2f}%".format(
                        epoch, loss_train, acc_train * 100))

            if best_params:
                self.restore_model_params(best_params)

            return self

    def predict_prob(self,X):
        if not self._session:
            raise NotFittedError("This %s instance is not fitted yet" % self.__class__.__name__)
        with self._session.as_default() as sess:
            return self._Y_prob.eval(feed_dict={self._X:X})

    def predict(self,X):
        class_indices = np.argmax(self.predict_prob(X),axis=1)
        return np.array([self.classes_[class_index] for class_index in class_indices],np.int32)

    def save(self,path):
        self._saver.save(self._session,path)


##########主函数
(X_train,y_train),(X_test,y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1,28*28)/255.0
X_test = X_test.astype(np.float32).reshape(-1,28*28)/255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

X_train1 = X_train[y_train < 5]
y_train1 = y_train[y_train < 5]
X_valid1 = X_valid[y_valid < 5]
y_valid1 = y_valid[y_valid < 5]
X_test1 = X_test[y_test < 5]
y_test1 = y_test[y_test < 5]

dnn_clf = DNNClassifier(random_state=42)
dnn_clf.fit(X_train1,y_train1,n_epochs=100,X_valid=X_valid1,y_valid=y_valid1)

from sklearn.metrics import accuracy_score
y_pred = dnn_clf.predict(X_test)
accuracy_score(y_test,y_pred)

###########3寻找超参数
from sklearn.model_selection import RandomizedSearchCV
def leaky_relu(alpha=0.01):
    def parametrized_leaky_relu(z, name=None):
        return tf.maximum(alpha * z, z, name=name)
    return parametrized_leaky_relu

param_distribs = {
    "n_neurons": [10, 30, 50, 70, 90, 100, 120, 140, 160],
    "batch_size": [10, 50, 100, 500],
    "learning_rate": [0.01, 0.02, 0.05, 0.1],
    "activation": [tf.nn.relu, tf.nn.elu, leaky_relu(alpha=0.01), leaky_relu(alpha=0.1)],
    # you could also try exploring different numbers of hidden layers, different optimizers, etc.
    #"n_hidden_layers": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #"optimizer_class": [tf.train.AdamOptimizer, partial(tf.train.MomentumOptimizer, momentum=0.95)],
}

rnd_search = RandomizedSearchCV(DNNClassifier(random_state=42),param_distribs,n_iter=50,cv=3,
                                random_state=42,verbose=2)
rnd_search.fit(X_train1,y_train1,X_valid=X_valid1,y_valid=y_valid1,n_epochs=1000)

###########最优分数以及最优参数
print('best training score:',rnd_search.best_score_)
print('best params of dnn:',rnd_search.best_params_)

##########33333带入测试数据到最好的模型中
y_pred = rnd_search.predict(X_test1)
accuracy_score(y_test1,y_pred)
#########保存超参数后的模型
rnd_search.best_estimator_.save('./my_best_mnist_model_o_to_4')

###########----------------------batch_norm的影响
dnn_clf = DNNClassifier(activation=leaky_relu(alpha=0.1),batch_size=500,learning_rate=0.01,n_neurons=140,random_state=42)
dnn_clf.fit(X_train1,y_train1,n_epochs=1000,X_valid=X_valid1,y_valid=y_valid1)

y_pred = dnn_clf.predict(X_test1)
accuracy_score(y_test1,y_pred)

dnn_clf_bn = DNNClassifier(activation=leaky_relu(alpha=0.1),batch_size=500,
                        learning_rate=0.01,n_neurons=140,random_state=42,batch_norm_momentum=0.95)
dnn_clf_bn.fit(X_train1,y_train1,n_epochs=1000,X_valid=X_valid1,y_valid=y_valid1)

y_pred = dnn_clf_bn.predict(X_test1)
accuracy_score(y_test1, y_pred)
############超参数的选取
param_distribs = {
    "n_neurons": [10, 30, 50, 70, 90, 100, 120, 140, 160],
    "batch_size": [10, 50, 100, 500],
    "learning_rate": [0.01, 0.02, 0.05, 0.1],
    "activation": [tf.nn.relu, tf.nn.elu, leaky_relu(alpha=0.01), leaky_relu(alpha=0.1)],
    # you could also try exploring different numbers of hidden layers, different optimizers, etc.
    #"n_hidden_layers": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #"optimizer_class": [tf.train.AdamOptimizer, partial(tf.train.MomentumOptimizer, momentum=0.95)],
    "batch_norm_momentum": [0.9, 0.95, 0.98, 0.99, 0.999],
}

rnd_search_bn = RandomizedSearchCV(DNNClassifier(random_state=42), param_distribs, n_iter=50, cv=3,
                                   random_state=42, verbose=2)
rnd_search_bn.fit(X_train1, y_train1, X_valid=X_valid1, y_valid=y_valid1, n_epochs=1000)

y_pred = dnn_clf.predict(X_train1)
accuracy_score(y_train1, y_pred)

############drop_out

dnn_clf_dropout = DNNClassifier(activation=leaky_relu(alpha=0.1), batch_size=500, learning_rate=0.01,
                                n_neurons=90, random_state=42,
                                dropout_rate=0.5)
dnn_clf_dropout.fit(X_train1, y_train1, n_epochs=1000, X_valid=X_valid1, y_valid=y_valid1)

y_pred = dnn_clf_dropout.predict(X_test1)
accuracy_score(y_test1, y_pred)


param_distribs = {
    "n_neurons": [10, 30, 50, 70, 90, 100, 120, 140, 160],
    "batch_size": [10, 50, 100, 500],
    "learning_rate": [0.01, 0.02, 0.05, 0.1],
    "activation": [tf.nn.relu, tf.nn.elu, leaky_relu(alpha=0.01), leaky_relu(alpha=0.1)],
    # you could also try exploring different numbers of hidden layers, different optimizers, etc.
    #"n_hidden_layers": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #"optimizer_class": [tf.train.AdamOptimizer, partial(tf.train.MomentumOptimizer, momentum=0.95)],
    "dropout_rate": [0.2, 0.3, 0.4, 0.5, 0.6],
}

rnd_search_dropout = RandomizedSearchCV(DNNClassifier(random_state=42), param_distribs, n_iter=50,
                                        cv=3, random_state=42, verbose=2)
rnd_search_dropout.fit(X_train1, y_train1, X_valid=X_valid1, y_valid=y_valid1, n_epochs=1000)

y_pred = rnd_search_dropout.predict(X_test1)
accuracy_score(y_test1, y_pred)