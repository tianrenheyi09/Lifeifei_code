from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

class TwoLayerNet(object):

    def __init__(self,input_size,hidden_size,output_size,std=1e-4):
        """
           Initialize the model. Weights are initialized to small random values and
           biases are initialized to zero. Weights and biases are stored in the
           variable self.params, which is a dictionary with the following keys:
           W1: First layer weights; has shape (D, H)
           b1: First layer biases; has shape (H,)
           W2: Second layer weights; has shape (H, C)
           b2: Second layer biases; has shape (C,)
           Inputs:
           - input_size: The dimension D of the input data.
           - hidden_size: The number of neurons H in the hidden layer.
           - output_size: The number of classes C.
           """
        self.params = {}
        self.params['W1'] = std*np.random.rand(input_size,hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std*np.random.rand(hidden_size,output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self,X,y=None,reg=0.0):
        W1,b1 = self.params['W1'],self.params['b1']
        W2,b2 = self.params['W2'],self.params['b2']
        N,D = X.shape

        scores = None

        h1 = np.maximum(0,np.dot(X,W1)+b1)
        h2 = np.dot(h1,W2)+b2
        scores = h2

        if y is None:
            return scores

        loss = None
        shift_scores = scores-np.max(scores,axis=1).reshape((-1,1))#######减去每一行的最大值
        softmax_output = np.exp(shift_scores)/np.sum(np.exp(shift_scores),axis=1).reshape((-1,1))
        loss =-np.sum(np.log(softmax_output[np.arange(N),y]))
        loss /= N
        loss +=0.5*reg*(np.sum(W1*W1)+np.sum(W2*W2))

        grads = {}
        ##第二层梯度计算
        dscores = softmax_output.copy()
        dscores[np.arange(N),y] -= 1
        dscores /=N
        grads['W2'] = h1.T.dot(dscores)+reg*W2
        grads['b2'] = np.sum(dscores,axis=0)
        ######第一层梯度计算
        dh = dscores.dot(W2.T) ###N*H
        dh_relu = (h1>0)*dh####N*H
        grads['W1'] = X.T.dot(dh_relu)+reg*W1 ####D*H
        grads['b1'] = np.sum(dh_relu,axis=0)####h

        return loss,grads

    def train(self,X,y,X_val,y_val,learning_rate=1e-3,learning_rate_decay=0.95,
              reg=1e-5,num_iters=100,batch_size=200,verbose=False):

        num_train = X.shape[0]
        iteration_per_epoch = max(num_train/batch_size,1)
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None
            idx = np.random.choice(num_train,batch_size,replace=True)
            X_bacth = X[idx]
            y_batch = y[idx]
            loss,grads = self.loss(X_bacth,y=y_batch,reg=reg)
            loss_history.append(loss)
            ########参数更新
            self.params['W2'] += -learning_rate*grads['W2']
            self.params['b2'] += -learning_rate*grads['b2']
            self.params['W1'] += -learning_rate*grads['W1']
            self.params['b1'] += -learning_rate*grads['b1']

            if verbose and it%50 ==0:
                print('iteration %d /%d :loss %f'%(it,num_iters,loss))

            if it %iteration_per_epoch==0:
                train_acc = (self.predict(X_bacth)==y_batch).mean()
                val_acc = (self.predict(X_val)==y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)
                learning_rate  = learning_rate*learning_rate_decay

        return {'loss_history':loss_history,
                'train_acc_hiatory':train_acc_history,
                'val_acc_history':val_acc_history
                }

    def predict(self,X):
        y_pred = None
        h = np.maximum(0,X.dot(self.params['W1'])+self.params['b1'])
        scores = h.dot(self.params['W2'])+self.params['b2']
        y_pred = np.argmax(scores,axis=1)

        return y_pred










