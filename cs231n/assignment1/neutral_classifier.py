import tensorflow as tf
import numpy as np
import pandas as pd
import random
from data_load import load_CIFAR10
import matplotlib.pyplot as plt
from gradient_check import grad_check_sparse
from TwoLayerNet import TwoLayerNet

plt.rcParams['figure.figsize'] = (10,8)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


filepath = 'data/cifar-10-batches-py'
X_train,y_train,X_test,y_test = load_CIFAR10(filepath)

print('train data shape:',X_train.shape)
print('train labels shape:',y_train.shape)
print('test data shape:',X_test.shape)
print('test labels shape:',y_test.shape)


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

X_train -= mean_image
X_val -= mean_image
X_test -= mean_image
x_dev -= mean_image

print('Train data shape: ', X_train.shape)
print('Validation data shape: ', X_val.shape)
print('Test data shape: ', X_test.shape)

##########模型训练
input_size = 32*32*3
hidden_size = 50
num_classes = 10
net = TwoLayerNet(input_size,hidden_size,num_classes)
stats = net.train(X_train,y_train,X_val,y_val,num_iters=1000,batch_size=200,
                  learning_rate=1e-4,learning_rate_decay=0.95,reg=0.5,verbose=True)
val_acc = (net.predict(X_val)==y_val).mean()
print('valiadation accuracy:',val_acc)

###########loss和ACC可视化
plt.subplot(211)
plt.plot(stats['loss_history'])
plt.title('loss_history')
plt.xlabel('iteration')
plt.ylabel('loss')

plt.subplot(212)
plt.plot(stats['train_acc_hiatory'],label='train',color='red')
plt.plot(stats['val_acc_history'],label='val',color='blue')
plt.title('class acc history')
plt.xlabel('epoch')
plt.ylabel('class acc')
plt.legend()
plt.show()

########可视化权重
from vis_utils import visualize_grid
def show_net_weights(net):
    W1 = net.params['W1']
    W1 = W1.reshape(32,32,3,-1).transpose(3,0,1,2)
    plt.imshow(visualize_grid(W1,padding=3).astype('uint8'))
    plt.axis('off')
    plt.show()

show_net_weights(net)

##########交叉验证选择超参数
input_size = 32*32*3
num_classes = 10
hidden_size = [75,100]
results = {}
best_val_acc = 0
best_net = None
best_lr = -1
best_reg = None
best_hidden_size = 0

learning_rates = np.array([0.8,1])*1e-3
regularization_strengths = [0.75,1]
print('starting')
for hs in hidden_size:
    for lr in learning_rates:
        for reg in regularization_strengths:
            net = TwoLayerNet(input_size,hs,num_classes)
            stats = net.train(X_train, y_train, X_val, y_val, num_iters=1500, batch_size=200,
                              learning_rate=lr, learning_rate_decay=0.95, reg=reg, verbose=False)
            val_acc = (net.predict(X_val) == y_val).mean()
            train_acc = (net.predict(X_train)==y_train).mean()
            if val_acc>best_val_acc:
                best_val_acc = val_acc
                best_net = net
                best_lr = lr
                best_reg = reg
                best_hidden_size = hs
            results[(hs,lr,reg)] = train_acc,val_acc
            print('hidden_size: %d lr: %e reg: %e train accuracy: %f val accuracy: %f' %
                  (hs, lr, reg, results[(hs,lr, reg)][0], results[(hs,lr, reg)][1]))

print('finshed')

print('Best hidden_size: %d\nBest lr: %e\nBest reg: %e\ntrain accuracy: %f\nval accuracy: %f' %
     (best_hidden_size, best_lr, best_reg, results[(best_hidden_size,best_lr, best_reg)][0], results[(best_hidden_size,best_lr, best_reg)][1]))

print('betst validaton acc is :',best_val_acc)


show_net_weights(best_net)

#####测试
test_acc = (best_net.predict(X_test)==y_test).mean()
print('test acc:',test_acc)
