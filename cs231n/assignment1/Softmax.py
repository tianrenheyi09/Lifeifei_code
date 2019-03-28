import numpy as np
from random import shuffle

def softmax_loss_naive(W,X,y,reg):
    loss = 0.0
    dW = np.zeros_like(W)
    num_classes = W.shape[1]
    num_train = X.shape[0]

    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class = y[i]
        exp_scores = np.zeros_like(scores)
        row_sum = 0
        for j in range(num_classes):
            exp_scores[j] = np.exp(scores[j])
            row_sum += exp_scores[j]
        loss += -np.log(exp_scores[correct_class]/row_sum)
        for k in range(num_classes):
            if k != correct_class:
                dW[:,k] += exp_scores[k]/row_sum*X[i]
            else:
                dW[:,k] += (exp_scores[correct_class]/row_sum-1)*X[i]

    loss /= num_train
    reg_loss = 0.5*reg*np.sum(W**2)
    loss += reg_loss
    dW /=num_train
    dW +=reg*W
    return loss,dW

def softmax_loss_vectorized(W,X,y,reg):

    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    scores = X.dot(W)
    # ###########
    # correct_class_score = scores[np.arange(num_train),y].reshape(num_train,1)
    # exp_sum = np.sum(np.exp(scores),axis=1).reshape(num_train,1)
    # loss += np.sum(np.log(exp_sum)-correct_class_score)
    # #########
    shift_scores = scores-np.max(scores,axis=1).reshape((-1,1))
    softmax_out = np.exp(shift_scores)/np.sum(np.exp(shift_scores),axis=1).reshape((-1,1))
    loss =-np.sum(np.log(softmax_out[np.arange(num_train),y]))
    loss /=num_train
    loss += 0.5*reg*np.sum(W*W)

    # margin = np.exp(scores)/exp_sum
    margin = softmax_out.copy()
    margin[np.arange(num_train),y] += -1
    dW = X.T.dot(margin)
    dW /= num_train
    dW +=reg*W

    return loss,dW


class Softmax(object):

    def __init__(self):
        self.W = None

    def train(self,X,y,learning_rate=1e-3,reg=1e-5,num_iters=100,batch_size=20,print_flag=False):

        loss_history = []
        num_train = X.shape[0]
        dim = X.shape[1]
        num_classes =np.max(y)+1
        if self.W == None:
            self.W = 0.001*np.random.randn(dim,num_classes)

        for t in range(num_iters):
            idx_batch = np.random.choice(num_train,batch_size,replace=True)
            X_batch = X[idx_batch]
            y_batch = y[idx_batch]
            loss, dW = softmax_loss_vectorized(self.W,X_batch, y_batch, reg)
            loss_history.append(loss)
            self.W += -learning_rate * dW

            if print_flag and t % 100 == 0:
                print('iteration %d / %d: loss %f' % (t, num_iters, loss))

        return loss_history

    def predict(self,X):
        y_pred = np.zeros(X.shape[0])
        scores = np.dot(X,self.W)
        y_pred = np.argmax(scores,axis=1)

        return y_pred







