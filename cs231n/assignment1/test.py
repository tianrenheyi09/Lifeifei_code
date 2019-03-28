import numpy as np

import keras.backend as K
import tensorflow as tf


def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
    """
    a naive implementation of numerical gradient of f at x
    - f should be a function that takes a single argument
    - x is the point (numpy array) to evaluate the gradient at
    """

    fx = f(x) # evaluate function value at original point
    grad = np.zeros_like(x)
    # iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:

        # evaluate function at x+h
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h # increment by h
        fxph = f(x) # evalute f(x + h)
        x[ix] = oldval - h
        fxmh = f(x) # evaluate f(x - h)
        x[ix] = oldval # restore

        # compute the partial derivative with centered formula
        grad[ix] = (fxph - fxmh) / (2 * h) # the slope
        if verbose:
            print(ix, grad[ix])
        it.iternext() # step to next dimension

    return grad
def rel_error(x, y):
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
def sigmoid_forward(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache
def sigmoid_loss(x, y):#######只针对2分类的函数，多分类用softmax_loss,输入y为标签列表，不是矩阵
    probs, _ = sigmoid_forward(x)
    N = x.shape[0]
    y_ = np.zeros((N,2))
    y_[np.arange(N),y] = 1
    loss = -y_*np.log(probs)-(1-y_)*np.log(1-probs)
    loss = np.mean(loss)

    da = -(y_/probs-(1-y_)/(1-probs))
    dx = da*probs * (1 - probs)/N/2########因为y标签算了2次，相当于2N个数据
    ###与吴恩达视频中梯度计算有差异因为视频中标签算了1次
    return loss,dx


num_classes,num_inputs=2,5
x=0.001*np.random.randn(num_inputs,num_classes)
y=np.random.randint(num_classes,size=num_inputs)

loss_sig,_= sigmoid_loss(x,y)
print('loss of my sigmoid:',loss_sig)
print()

##########tf 用来对比
probs, _ = sigmoid_forward(x)
N = x.shape[0]
y_ = np.zeros((N,2))
y_[np.arange(N),y] = 1
#######tensorflow的sigmoid——entropy的label标签是one_hor形式，输入logis为不是概率，直接就是X，因为内部已经计算了sigmoid
loss_tf = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=x)
loss_tf_ave = tf.reduce_mean(loss_tf)
print("the loss tf  is: ", K.eval(loss_tf))
print("the loss  tf ave is: ", K.eval(loss_tf_ave))
#######梯度检验
dx_num = eval_numerical_gradient(lambda x:sigmoid_loss(x,y)[0],x,verbose=False)
loss,dx=sigmoid_loss(x,y)
print('\ntesting softmax_loss:')
print('loss:',loss)
print('dx error:',rel_error(dx_num,dx))

#########其他情况下的sigmoid计算
labels = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
logits = np.array([[11., 8., 7.], [10., 14., 3.], [1., 2., 4.]])
y_pred,_= sigmoid_forward(logits)
prob_error1 = -labels * np.log(y_pred) - (1 - labels) * np.log(1 - y_pred)
print(prob_error1)
print(np.mean(prob_error1))


print(".............")
labels1 = np.array([[0., 1., 0.], [1., 1., 0.], [0., 0., 1.]])  # 不一定只属于一个类别
logits1 = np.array([[1., 8., 7.], [10., 14., 3.], [1., 2., 4.]])
y_pred1,_= sigmoid_forward(logits1)
prob_error11 = -labels1 * np.log(y_pred1) - (1 - labels1) * np.log(1 - y_pred1)
print(prob_error11)
print(np.mean(prob_error11))



