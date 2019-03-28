import tensorflow as tf
import numpy as np
import pandas as pd
import random
from data_load import load_CIFAR10
import matplotlib.pyplot as plt
from knn import knn

plt.rcParams['figure.figsize'] = (10,8)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

filepath = 'data/cifar-10-batches-py'
X_train,y_train,X_test,y_test = load_CIFAR10(filepath)

print('train data shape:',X_train.shape)
print('train labels shape:',y_train.shape)
print('test data shape:',X_test.shape)
print('test labels shape:',y_test.shape)

#########显示部分数据集
classes = ['plane','car','bird','cat','deer','dog','drog','horse','ship','truck']
num_classes = len(classes)
samples_per_class = 7
for y,cls in enumerate(classes):
    idxs = np.flatnonzero(y_train ==y)
    idxs = np.random.choice(idxs,samples_per_class,replace = False)
    for i,idx in enumerate(idxs):
        plt_idx = i*num_classes+y+1
        plt.subplot(samples_per_class,num_classes,plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()

###调整数据集大小
num_training = 5000
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]


X_train = np.reshape(X_train,(X_train.shape[0],-1))
X_test = np.reshape(X_test,(X_test.shape[0],-1))
print(X_train.shape,X_test.shape)

######测试集预测
model = knn()
model.train(X_train,y_train)
dists = model.compute_dist_two_loops(X_test)
print(dists)
y_test_pred = model.predict_labels(dists,k=1)

###评价
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct)/num_test
print("got %d /%d correct==>accuracy:%f"%(num_correct,num_test,accuracy))

dists_one = model.compute_dist_one_loops(X_test)
diff = np.linalg.norm(dists-dists_one)
print('diff was :%f'%diff)

import time
def time_function(f,*args):
    tic = time.time()
    f(*args)
    toc = time.time()
    return toc-tic

two_loop_time = time_function(model.compute_dist_two_loops,X_test)
one_loop_time = time_function(model.compute_dist_one_loops,X_test)
no_loop_time = time_function(model.compute_dist_no_loops,X_test)
print('two loop %f seconds,one loop %f seconds,no loop %f seonds'%(two_loop_time,one_loop_time,no_loop_time))


#####交叉验证
##随机打乱

num_folds = 5
num_val_samples = int(len(X_train)/num_folds)

per = np.random.permutation(X_train.shape[0])
new_train = X_train[per]
new_y_train = y_train[per]


k_choices = [1,3,5,8,10]

# Split up the training data into folds
X_train_folds = []
y_train_folds = []
X_train_folds = np.split(new_train, num_folds)
y_train_folds = np.split(new_y_train, num_folds)

# A dictionary holding the accuracies for different values of k
k_accuracy = {}
for k in k_choices:
    accuracies = []
    #knn = KNearestNeighbor()
    model = knn()
    for i in range(num_folds):
        Xtr = np.concatenate(X_train_folds[:i] + X_train_folds[i+1:])
        ytr = np.concatenate(y_train_folds[:i] + y_train_folds[i+1:])
        Xcv = X_train_folds[i]
        ycv = y_train_folds[i]
        model.train(Xtr, ytr)
        ycv_pred = model.predict(Xcv, k=k, num_loops=0)
        num_correct = np.sum(ycv_pred == ycv)
        accuracy = float(num_correct) / len(ycv_pred)
        # accuracy = np.mean(ycv_pred == ycv)
        accuracies.append(accuracy)
    k_accuracy[k] = accuracies

# Print the accuracy
for k in k_choices:
    for i in range(num_folds):
        print('k = %d, fold = %d, accuracy: %f' % (k, i+1, k_accuracy[k][i]))


#######another k validation
for k in k_choices:
    accuracy = []

    for i in range(num_folds):
        model = knn()
        val_data = new_train[i*num_val_samples:(i+1)*num_val_samples]
        val_targets = new_y_train[i*num_val_samples:(i+1)*num_val_samples]
        train_data = np.concatenate([new_train[:i*num_val_samples],new_train[(i+1)*num_val_samples:]],axis=0)
        train_targets = np.concatenate([new_y_train[:i*num_val_samples],new_y_train[(i+1)*num_val_samples:]],axis=0)
        model.train(train_data,train_targets)
        y_pred = model.predict(val_data)
        num_correct = np.sum(y_pred==val_targets)
        accuracy = float(num_correct) / len(y_pred)
        # accuracy = np.mean(ycv_pred == ycv)
        accuracies.append(accuracy)
    k_accuracy[k] = accuracies

# Print the accuracy
for k in k_choices:
    for i in range(num_folds):
        print('k = %d, fold = %d, accuracy: %f' % (k, i+1, k_accuracy[k][i]))


for k in k_choices:
    plt.scatter([k]*num_folds,k_accuracy[k])

accuracies_mean = [np.mean(k_accuracy[k]) for k in k_accuracy]
accuracies_std = [np.std(k_accuracy[k]) for k in k_accuracy]
plt.errorbar(k_choices,accuracies_mean,yerr=accuracies_std)
plt.title('cross-validation on k')
plt.xlabel('k')
plt.ylabel('cross-validation acc')
plt.show()
########选出最好的参数
best_k  = k_choices[np.argmax(accuracies_mean)]
model = knn()
model.train(X_train,y_train)
y_pred = model.predict(X_test,k=best_k)
num_correct = np.sum(y_pred==y_test)
accuracy = num_correct/len(y_pred)
print('corrrect%d/%d: accuracy is :%f'%(num_correct,len(y_pred),accuracy))
