import numpy as np


def loss_naive(W,X, y, reg):
    loss = 0.0
    dw = np.zeros(W.shape)

    num_train = X.shape[0]
    num_classes = W.shape[1]
    for i in range(num_train):
        scores = np.dot(X[i], W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j==y[i]:
                continue
            margin = scores[j]-correct_class_score+1
            if margin>0:
                loss += margin
                dw[:,j] += X[i].T
                dw[:,y[i]] -= X[i].T

    loss /= num_train
    dw /= num_train

    loss += 0.5*reg*np.sum(W*W)
    dw += reg*W

    return loss,dw

def loss_vectorized(W,X,y,reg):
    loss = 0.0
    dw = np.zeros(W.shape)

    num_train = X.shape[0]
    scores = np.dot(X,W)
    correct_score = scores[range(num_train),list(y)].reshape(-1,1)
    margin = np.maximum(0,scores-correct_score+1)
    margin[range(num_train),list(y)] = 0
    loss = np.sum(margin)/num_train+0.5*reg*np.sum(W*W)

    num_classes = W.shape[1]
    mask = np.zeros((num_train,num_classes))
    mask[margin>0] = 1
    mask[range(num_train),list(y)] = 0
    mask[range(num_train),list(y)] = -np.sum(mask,axis=1)
    dw = np.dot(X.T,mask)
    dw = dw/num_train+reg*W

    return loss,dw



class linear_svm(object):

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
            loss, dW = loss_vectorized(self.W,X_batch, y_batch, reg)
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






