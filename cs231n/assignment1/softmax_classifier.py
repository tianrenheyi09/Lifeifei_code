import tensorflow as tf
import numpy as np
import pandas as pd
import random
from data_load import load_CIFAR10
import matplotlib.pyplot as plt
from linear_svm import linear_svm,loss_vectorized,loss_naive
from Softmax import softmax_loss_naive,softmax_loss_vectorized,Softmax
from gradient_check import grad_check_sparse

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

# Split the data into train, val, and test sets
num_train = 49000
num_val = 1000
num_test = 1000
num_dev = 500

# Validation set
mask = range(num_train, num_train + num_val)
X_val = X_train[mask]
y_val = y_train[mask]

# Train set
mask = range(num_train)
X_train = X_train[mask]
y_train = y_train[mask]

# Test set
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

mask = range(num_dev)
x_dev = X_train[mask]
y_dev = y_train[mask]
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

# Preprocessing: reshape the images data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
x_dev = np.reshape(x_dev,(x_dev.shape[0],-1))

print('Train data shape: ', X_train.shape)
print('Validation data shape: ', X_val.shape)
print('Test data shape: ', X_test.shape)

# Processing: subtract the mean images
mean_image = np.mean(X_train, axis=0)
plt.figure(figsize=(4,4))
plt.imshow(mean_image.reshape((32,32,3)).astype('uint8'))
plt.show()

X_train -= mean_image
X_val -= mean_image
X_test -= mean_image
x_dev -= mean_image

# append the bias dimension of ones (i.e. bias trick)
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
x_dev = np.hstack([x_dev,np.ones((x_dev.shape[0],1))])
print('Train data shape: ', X_train.shape)
print('Validation data shape: ', X_val.shape)
print('Test data shape: ', X_test.shape)

#####梯度检验
W=np.random.randn(3073,10)*0.0001
loss,grad=softmax_loss_naive(W,x_dev,y_dev,0.01)
print('loss is : %f '% loss)
f = lambda w:softmax_loss_naive(w, x_dev, y_dev, 1e2)[0]
grad_numerical = grad_check_sparse(f,W,grad,10)

#########计算loss和梯度
import time
tic = time.time()
loss_naive,grad_naive = softmax_loss_naive(W,x_dev,y_dev,0.00001)
toc = time.time()
print('naive loss: %e computed in %fs' % (loss_naive, toc - tic))

tic = time.time()
loss_vec,grad_vec = softmax_loss_vectorized(W,x_dev,y_dev,0.00001)
toc = time.time()
print('vec loss: %e computed in %fs' % (loss_vec ,toc - tic))

grad_difference = np.linalg.norm(grad_naive-grad_vec,ord='fro')
print('loss difference %f'%np.abs(loss_naive-loss_vec))
print('gradient difference :%f'%grad_difference)




soft = Softmax()
loss_history = soft.train(X_train,y_train,learning_rate=1e-7,reg=2.5e4,num_iters=2000,batch_size=200,print_flag=True)

# Plot the loss_history
plt.plot(loss_history)
plt.xlabel('Iteration number')
plt.ylabel('Loss value')
plt.show()

y_pred = soft.predict(X_train)
num_correct = np.sum(y_pred == y_train)
accuracy = np.mean(y_pred == y_train)
print('Training correct %d/%d: The accuracy is %f' % (num_correct, X_train.shape[0], accuracy))




y_pred = soft.predict(X_test)
num_correct = np.sum(y_pred == y_test)
accuracy = np.mean(y_pred == y_test)
print('Test correct %d/%d: The accuracy is %f' % (num_correct, X_test.shape[0], accuracy))




learning_rates = [1e-7,5e-7]
regularization_strengths = [5e4,1e8]

results = {}
best_lr = None
best_reg = None
best_val = -1   # The highest validation accuracy that we have seen so far.

best_softmax = None


for lr in learning_rates:
    for reg in regularization_strengths:
        soft = Softmax()
        history = soft.train(X_train,y_train,learning_rate=lr,reg=reg,num_iters=2000)
        y_train_pred = soft.predict(X_train)
        accuracy_train = np.mean(y_train_pred == y_train)
        y_val_pred = soft.predict(X_val)
        accuracy_val = np.mean(y_val_pred == y_val)
        results[(lr, reg)] = accuracy_train, accuracy_val

        if accuracy_val > best_val:
            best_lr = lr
            best_reg = reg
            best_val = accuracy_val
            best_softmax = soft

        print('lr: %e reg: %e train accuracy: %f val accuracy: %f' %
              (lr, reg, results[(lr, reg)][0], results[(lr, reg)][1]))
print('Best validation accuracy during cross-validation:\nlr = %e, reg = %e, best_val = %f' %
      (best_lr, best_reg, best_val))



# Visualize the cross-validation results
import math

x_scatter = [math.log10(x[0]) for x in results]
y_scatter = [math.log10(x[1]) for x in results]

# Plot training accuracy
plt.figure(figsize=(10,10))
make_size = 100
colors = [results[x][0] for x in results]
plt.subplot(2, 1, 1)
plt.scatter(x_scatter, y_scatter, make_size, c = colors)
plt.colorbar()
plt.xlabel('log learning rate')
plt.ylabel('log regularization strength')
plt.title('Training accuracy')

# Plot validation accuracy
colors = [results[x][1] for x in results]
plt.subplot(2, 1, 2)
plt.scatter(x_scatter, y_scatter, make_size, c = colors)
plt.colorbar()
plt.xlabel('log learning rate')
plt.ylabel('log regularization strength')
plt.title('Validation accuracy')
plt.show()


# Use the best svm to test
# Evaluate the best softmax on test set
y_test_pred = best_softmax.predict(X_test)
test_accuracy = np.mean(y_test == y_test_pred)
print ('softmax on raw pixels final test set accuracy: %f' % (test_accuracy, ))


W = best_softmax.W[:-1, :]    # delete the bias
W = W.reshape(32, 32, 3, 10)
W_min, W_max = np.min(W), np.max(W)
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for i in range(10):
    plt.subplot(2, 5, i+1)
    imgW = 255.0 * ((W[:, :, :, i].squeeze() - W_min) / (W_max - W_min))
    plt.imshow(imgW.astype('uint8'))
    plt.axis('off')
    plt.title(classes[i])
plt.show()
