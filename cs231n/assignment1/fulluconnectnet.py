import tensorflow as tf
import numpy as np
import pandas as pd
import random
from data_load import load_CIFAR10
import matplotlib.pyplot as plt
from fc_net import *

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

# Processing: subtract the mean images
mean_image = np.mean(X_train, axis=0)
# plt.figure(figsize=(4,4))
# plt.imshow(mean_image.reshape((32,32,3)).astype('uint8'))
# plt.show()

X_train -= mean_image
X_val -= mean_image
X_test -= mean_image
x_dev -= mean_image
#
#
# print('Train data shape: ', X_train.shape)
# print('Validation data shape: ', X_val.shape)
# print('Test data shape: ', X_test.shape)

###########-------------solve

from Solver import Solver

data = {}
data = {'X_train':X_train,'y_train':y_train,'X_val':X_val,'y_val':y_val,'X_test':X_test,'y_test':y_test}


model = TwoLayerNet(reg=0.1)
solver = Solver(model,data,update_rule='sgd',optim_config={'learning_rate':1e-3},
                lr_decay=0.8,num_epochs=10,batch_size=100,print_every=100)
solver.train()

scores = model.loss(data['X_test'])
y_pred = np.argmax(scores,axis=1)
acc = np.mean(y_pred==data['y_test'])
print('test acc is :%f'%acc)

plt.subplot(2,1,1)
plt.title('training loss')
plt.plot(solver.loss_history,'o')
plt.xlabel('iteration')
plt.subplot(2,1,2)
plt.title('accuracy')
plt.plot(solver.train_acc_history,'-o',label='train')
plt.plot(solver.val_acc_history,'-o',label='val')
plt.plot([0.5]*len(solver.val_acc_history),'k--')
plt.xlabel('epoch')
plt.legend(loc='lower right')#图例位置
plt.gcf().set_size_inches(15,12)
plt.show()

#########多层网络的测试代码

np.random.seed(231)
N, D, H1, H2, C = 2, 15, 20, 30, 10
X = np.random.randn(N, D)
y = np.random.randint(C, size=(N,))

for reg in [0, 3.14]:
  print('Running check with reg = ', reg)
  model = FullyConnectedNet([H1, H2], input_dim=D, num_classes=C,
                            reg=reg, weight_scale=5e-2, dtype=np.float64)

  loss, grads = model.loss(X, y)
  print('Initial loss: ', loss)

  for name in sorted(grads):
    f = lambda _: model.loss(X, y)[0]
    grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
    print('%s relative error: %.2e' % (name, rel_error(grad_num, grads[name])))

num_train = 50
small_data = {
  'X_train': data['X_train'][:num_train],
  'y_train': data['y_train'][:num_train],
  'X_val': data['X_val'],
  'y_val': data['y_val'],
}

weight_scale = 1e-2
learning_rate = 1e-2
model = FullyConnectedNet([140, 100],
              weight_scale=weight_scale, dtype=np.float64)
solver = Solver(model, small_data,
                print_every=10, num_epochs=20, batch_size=15,
                update_rule='sgd',
                optim_config={
                  'learning_rate': learning_rate,
                }
         )
solver.train()

plt.plot(solver.loss_history, 'o')
plt.title('Training loss history')
plt.xlabel('Iteration')
plt.ylabel('Training loss')
plt.show()

#################多个优化器的对比
##检验sgd+momentum
from optim import *
N,D = 4,5
w = np.linspace(-0.4,0.6,num=N*D).reshape(N,D)
dw = np.linspace(-0.6,0.4,num=N*D).reshape(N,D)
v = np.linspace(0.6,0.9,num=N*D).reshape(N,D)
config = {'learning_rate':1e-3,'velocity':v}
next_w,_ = sgd_momentum(w,dw,config = config)

expected_next_w = np.asarray([
[ 0.1406, 0.20738947, 0.27417895, 0.34096842, 0.40775789],
[ 0.47454737, 0.54133684, 0.60812632, 0.67491579, 0.74170526],
[ 0.80849474, 0.87528421, 0.94207368, 1.00886316, 1.07565263],
[ 1.14244211, 1.20923158, 1.27602105, 1.34281053, 1.4096 ]])

print('rel error:',rel_error(next_w,expected_next_w))
##########--sgd+momentum
num_train = 4000
small_data = {
  'X_train': data['X_train'][:num_train],
  'y_train': data['y_train'][:num_train],
  'X_val': data['X_val'],
  'y_val': data['y_val'],
}

solvers = {}

for update_rule in ['sgd', 'sgd_momentum']:
  print('running with ', update_rule)
  model = FullyConnectedNet([100, 100, 100, 100, 100], weight_scale=5e-2, dtype=np.float64)

  solver = Solver(model, small_data,
                  num_epochs=5, batch_size=100,
                  update_rule=update_rule,
                  optim_config={
                    'learning_rate': 1e-2,
                  },
                  verbose=True)
  solvers[update_rule] = solver
  solver.train()

plt.subplot(3, 1, 1)
plt.title('Training loss')
plt.xlabel('Iteration')

plt.subplot(3, 1, 2)
plt.title('Training accuracy')
plt.xlabel('Epoch')

plt.subplot(3, 1, 3)
plt.title('Validation accuracy')
plt.xlabel('Epoch')

for update_rule,solver in solvers.items():
    plt.subplot(3,1,1)
    plt.plot(solver.loss_history,'o',label=update_rule)
    plt.subplot(3, 1, 2)
    plt.plot(solver.train_acc_history, '-o', label=update_rule)
    plt.subplot(3, 1, 3)
    plt.plot(solver.val_acc_history, '-o', label=update_rule)

for i in [1,2,3]:
    plt.subplot(3,1,i)
    plt.legend(loc='upper center',ncol=4)
plt.gcf().set_size_inches(15,15)
plt.show()



num_train = 4000
small_data = {
  'X_train': data['X_train'][:num_train],
  'y_train': data['y_train'][:num_train],
  'X_val': data['X_val'],
  'y_val': data['y_val'],
}

solvers = {}
learning_rates = {'sgd':1e-2,'sgd_momentum':1e-2,'rmsprop':1e-4,'adam':1e-3}
for update_rule in ['sgd','sgd_momentum','rmsprop','adam']:
    print('runing with',update_rule)
    model = FullyConnectedNet([100,100,100,100,100],weight_scale=5e-2)
    solver = Solver(model,small_data,num_epochs=5,batch_size=100,
                    update_rule=update_rule,
                    optim_config={'learning_rate':learning_rates[update_rule]},
                    verbose=True)
    solvers[update_rule] = solver
    solver.train()

plt.subplot(3, 1, 1)
plt.title('Training loss')
plt.xlabel('Iteration')

plt.subplot(3, 1, 2)
plt.title('Training accuracy')
plt.xlabel('Epoch')

plt.subplot(3, 1, 3)
plt.title('Validation accuracy')
plt.xlabel('Epoch')

for update_rule, solver in solvers.items():
    plt.subplot(3, 1, 1)
    plt.plot(solver.loss_history, 'o', label=update_rule)

    plt.subplot(3, 1, 2)
    plt.plot(solver.train_acc_history, '-o', label=update_rule)

    plt.subplot(3, 1, 3)
    plt.plot(solver.val_acc_history, '-o', label=update_rule)

for i in [1, 2, 3]:
    plt.subplot(3, 1, i)
    plt.legend(loc='upper center', ncol=4)
plt.gcf().set_size_inches(15, 15)
plt.show()

############训练做好的模型
best_model = None
################################################################################
# TODO: Train the best FullyConnectedNet that you can on CIFAR-10. You might   #
# batch normalization and dropout useful. Store your best model in the         #
# best_model variable.                                                         #
################################################################################
X_val= data['X_val']
y_val= data['y_val']
X_test= data['X_test']
y_test= data['y_test']

learning_rate = 3.1e-4
weight_scale = 2.5e-2 #1e-5
model = FullyConnectedNet([600, 500, 400, 300, 200, 100],
                weight_scale=weight_scale, dtype=np.float64, dropout=0.25, use_batchnorm=True, reg=1e-2)
solver = Solver(model, data,
                print_every=500, num_epochs=30, batch_size=100,
                update_rule='adam',
                optim_config={
                  'learning_rate': learning_rate,
                },
                lr_decay=0.9
         )
solver.train()
scores = model.loss(data['X_test'])
y_pred = np.argmax(scores, axis = 1)
acc = np.mean(y_pred == data['y_test'])
print('test acc: %f' %(acc))
best_model = model

plt.subplot(2, 1, 1)
plt.plot(solver.loss_history)
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
plt.plot(solver.train_acc_history, label='train')
plt.plot(solver.val_acc_history, label='val')
plt.title('Classification accuracy history')
plt.xlabel('Epoch')
plt.ylabel('Clasification accuracy')
plt.show()

############test our model
y_test_pred = np.argmax(best_model.loss(X_test), axis=1)
y_val_pred = np.argmax(best_model.loss(X_val), axis=1)
print('Validation set accuracy: ', (y_val_pred == y_val).mean())
print('Test set accuracy: ', (y_test_pred == y_test).mean())