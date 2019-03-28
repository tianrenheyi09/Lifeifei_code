import numpy as np
from layers import *
from gradient_check import eval_numerical_gradient
def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))



num_inputs = 2
input_shape = (4,5,6)
output_dim = 3
input_size = num_inputs*np.prod(input_shape)
weight_size = output_dim*np.prod(input_shape)
x = np.linspace(-0.1,0.5,num=input_size).reshape(num_inputs,*input_shape)
w = np.linspace(-0.2,0.3,num=weight_size).reshape((np.prod(input_shape),output_dim))
b = np.linspace(-0.3,0.1,num=output_dim)

out,_ = affine_forward(x,w,b)
correct_out = np.array([[ 1.49834967, 1.70660132, 1.91485297],
[ 3.25553199, 3.5141327, 3.77273342]])
print('test affien function:')
print('difference:',out-correct_out)



from gradient_check import eval_numerical_gradient_array,grad_check_sparse
x  = np.random.randn(10,2,3)
w = np.random.randn(6,5)
b = np.random.randn(5)
dout = np.random.randn(10,5)

dx_num = eval_numerical_gradient_array(lambda x:affine_forward(x,w,b)[0],x,dout)
dw_num = eval_numerical_gradient_array(lambda w:affine_forward(x,w,b)[0],w,dout)
db_num = eval_numerical_gradient_array(lambda b:affine_forward(x,w,b)[0],b,dout)

_,cache = affine_forward(x,w,b)
dx,dw,db = affine_backward(dout,cache)
print('test affint backward:')
print('dx error',dx_num-dx)


##########定义relu层


x = np.linspace(-0.5,0.5,num=12).reshape(3,4)
out,_ = relu_forward(x)
correct_out = np.array([[ 0., 0., 0., 0.,
],
[ 0., 0., 0.04545455,
0.13636364,],
[ 0.22727273, 0.31818182, 0.40909091, 0.5,]])
print('test relu_forward:')
print('difference:',np.max(out-correct_out))




x = np.random.randn(10,10)
dout = np.random.randn(*x.shape)

dx_num = eval_numerical_gradient_array(lambda x:relu_forward(x)[0],x,dout)
_,cache = relu_forward(x)
dx = relu_backward(dout,cache)
print('test relu_backeard')
print('dx error:',np.max(np.abs(dx_num-dx)))

#########检验sigmoid的后向传播

dx_num = eval_numerical_gradient_array(lambda x:sigmoid_forward(x)[0],x,dout)
_,cache =sigmoid_forward(x)
dx = sigmoid_backward(dout,cache)
print('test relu_backeard')
print('dx error:',np.max(np.abs(dx_num-dx)))



x=np.random.randn(2,3,4)
w=np.random.randn(12,10)
b=np.random.randn(10)
dout=np.random.randn(2,10)
out,cache=affine_relu_forward(x,w,b)
dx,dw,db=affine_relu_backward(dout,cache)
dx_num=eval_numerical_gradient_array(lambda x:affine_relu_forward(x,w,b)
[0],x,dout)
dw_num=eval_numerical_gradient_array(lambda w:affine_relu_forward(x,w,b)
[0],w,dout)
db_num=eval_numerical_gradient_array(lambda b:affine_relu_forward(x,w,b)
[0],b,dout)
print('testing affine_relu_forward:')
print('dx_error:',rel_error(dx_num,dx))
print('dw error:',rel_error(dw_num,dw))
print('db error:',rel_error(db_num,db))

print('dx_error:',np.max(np.abs(dx_num-dx)))
print('dw error:',np.max(np.abs(dw_num-dw)))
print('db error:',np.max(np.abs(db_num-db)))




num_classes,num_inputs=2,50
x=0.001*np.random.randn(num_inputs,num_classes)
y=np.random.randint(num_classes,size=num_inputs)



dx_num=eval_numerical_gradient(lambda x:svm_loss(x,y)[0],x,verbose=False)
loss,dx=svm_loss(x,y)
print('testing svm_loss:')
print('loss:',loss)
print('dx error:',rel_error(dx_num,dx))


dx_num = eval_numerical_gradient(lambda x:softmax_loss(x,y)[0],x,verbose=False)
loss,dx=softmax_loss(x,y)
print('\ntesting softmax_loss:')
print('loss:',loss)
print('dx error:',rel_error(dx_num,dx))


##########loss函数的各种验证
import tensorflow as tf

labels = [[0, 0, 1], [0, 1, 0]]
logits = [[2, 0.5, 6], [0.1, 0, 3]]


logits_scaled = tf.nn.softmax(logits)

result1 = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
result3 = -tf.reduce_sum(labels*tf.log(logits_scaled), 1)
loss_one = tf.reduce_mean(result3)

label_spar = [2,1]
logits = [[2, 0.5, 6], [0.1, 0, 3]]
result_spar = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_spar,logits=logits)
result_spar_mean = tf.reduce_mean(result_spar)

print(tf.Session().run(result_spar))
print(tf.Session().run(result_spar_mean))


num_classes,num_inputs=3,3
x = np.array(logits)
y =label_spar
loss_my_soft,_= softmax_loss(x,y)
print(loss_my_soft)

############sigmoid函数的验证
import numpy as np
from gradient_check import *
import keras.backend as K
import tensorflow as tf

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


dx_num = eval_numerical_gradient(lambda x:sigmoid_loss(x,y)[0],x,verbose=False)
loss,dx=sigmoid_loss(x,y)
print('\ntesting softmax_loss:')
print('loss:',loss)
print('dx error:',rel_error(dx_num,dx))





labels=np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
logits=np.array([[-800,8,7],[-10.,-700.,-13.],[-1.,-2.,0]])
logits = np.array([[11., 8., 7.], [10., 14., 3.], [1., 2., 4.]])
y_pred=sigmoid(logits)
prob_error1=-labels*np.log(y_pred)-(1-labels)*np.log(1-y_pred)
print("未优化的代码：",prob_error1)
prob_error2=np.greater_equal(logits,0)-logits*labels+np.log(1+np.exp(-np.abs(logits)))
prob_error3=logits-logits*labels+np.log(1+np.exp(-logits))
print("优化的结果：",prob_error2)
print("未优化的结果：",prob_error3)
print(K.eval(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)))