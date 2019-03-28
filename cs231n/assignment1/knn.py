import numpy as np

class knn(object):

    def __init__(self):
        pass
    def train(self,X,y):
        self.X_train = X
        self.y_train = y

    def predict(self,X,k=1,num_loops=0):
        if num_loops == 0:
            dists = self.compute_dist_no_loops(X)
        elif num_loops==1:
            dists = self.compute_dist_one_loops(X)
        elif num_loops==2:
            dists = self.compute_dist_two_loops(X)
        else:
            raise ValueError("invalid value %d fro num_loops"%num_loops)

        return self.predict_labels(dists,k=k)

    def predict_labels(self,dists,k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            closest_y = []
            y_indices = np.argsort(dists[i,:],axis=0)
            closest_y = self.y_train[y_indices[:k]]
            y_pred[i] = np.argmax(np.bincount(closest_y))

        return y_pred

    def compute_dist_two_loops(self,X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test,num_train))
        for i in range(num_test):
            for j in range(num_train):
                dists[i][j] = np.sqrt(np.sum(np.square(X[i]-self.X_train[j])))

        return dists
    def compute_dist_one_loops(self,X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        print(X.shape,self.X_train.shape)
        for i in range(num_test):
            dists[i,:] = np.sqrt(np.sum(np.square(self.X_train-X[i,:]),axis=1))

        return dists
    def compute_dist_no_loops(self,X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test,num_train))
        test_sum = np.sum(np.square(X),axis=1)#####axis=1是对行进行求和，axis=0是对列进行求和
        train_sum = np.sum(np.square(self.X_train),axis=1)
        inner_product = np.dot(X,self.X_train.T)
        dists = np.sqrt(-2*inner_product+test_sum.reshape(-1,1)+train_sum)
        return dists

def compute_dist_no_loops(X,X_train):
    num_test = X.shape[0]
    num_train = X_train.shape[0]
    dists = np.zeros((num_test,num_train))
    test_sum = np.sum(np.square(X),axis=1)#####axis=1是对行进行求和，axis=0是对列进行求和
    print(test_sum.shape)
    train_sum = np.sum(np.square(X_train),axis=1)
    print(train_sum.shape)
    inner_product = np.dot(X,X_train.T)
    print(inner_product.shape)
    dists = np.sqrt(-2*inner_product+test_sum.reshape(-1,1)+train_sum)
    return dists

##3测试代码
# X = np.array([[0,1,2],[2,3,4]])
# X_train = np.array([[0,1,2],[1,2,3],[2,3,4],[3,4,5]])
# test_sum = np.sum(np.square(X),axis=1)#####axis=1是对行进行求和，axis=0是对列进行求和
# print(test_sum.shape)
# train_sum = np.sum(np.square(X_train),axis=1)
# print(train_sum.shape)
# inner_product = np.dot(X,X_train.T)
# print(inner_product.shape)
# dists = np.sqrt(-2*inner_product+test_sum.reshape(-1,1)+train_sum)



