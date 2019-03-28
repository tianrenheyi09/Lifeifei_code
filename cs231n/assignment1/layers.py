import numpy as np

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def affine_forward(x,w,b):

    out = None
    N = x.shape[0]
    x_rsp = x.reshape(N,-1)
    out = x_rsp.dot(w)+b
    cache = (x,w,b)
    return out,cache

def affine_backward(dout,cache):
    x,w,b = cache
    dx,dw,db = None,None,None
    N = x.shape[0]
    x_rsp = x.reshape((N,-1))
    dx = dout.dot(w.T)
    dx = dx.reshape(*x.shape)
    dw = x_rsp.T.dot(dout)
    db = np.sum(dout,axis=0)
    return dx,dw,db


##########定义relu层

def relu_forward(x):
    out = None
    out = x*(x>=0)
    cache = x
    return out,cache



def relu_backward(dout,cache):
    dx,x = None,cache
    dx = (x>=0)*dout
    return dx



##########映射层和relu层合为一个层
def affine_relu_forward(x,w,b):
    a,fc_cache = affine_forward(x,w,b)
    out,relu_cache = relu_forward(a)
    cache = (fc_cache,relu_cache)
    return out,cache

def affine_relu_backward(dout,cache):
    fc_cache,relu_cache = cache
    da = relu_backward(dout,relu_cache)
    dx,dw,db = affine_backward(da,fc_cache)
    return dx,dw,db

def sigmoid_forward(Z):
    """
     Implements the sigmoid activation in numpy

     Arguments:
     Z -- numpy array of any shape

     Returns:
     A -- output of sigmoid(z), same shape as Z
     cache -- returns Z as well, useful during backpropagation
     """

    A = 1 / (1 + np.exp(-Z))
    cache = Z

    return A, cache

def sigmoid_backward(dA,cache):
    """
        Implement the backward propagation for a single SIGMOID unit.

        Arguments:
        dA -- post-activation gradient, of any shape
        cache -- 'Z' where we store for computing backward propagation efficiently

        Returns:
        dZ -- Gradient of the cost with respect to Z
        """

    Z = cache

    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    assert (dZ.shape == Z.shape)

    return dZ


def sigmoid_loss(x, y):
    probs, _ = sigmoid_forward(x)
    N = x.shape[0]
    # loss = (-np.dot(y.T, np.log(probs)) - np.dot((1 - y).T, np.log(1 - probs))) / N
    y_ = np.zeros((N,2))
    y_[np.arange(N),y] = 1
    loss = -y_*np.log(probs)-(1-y_)*np.log(1-probs)
    loss = np.mean(loss)

    da = -(y_/probs-(1-y_)/(1-probs))
    dx = da*probs * (1 - probs)/N
    dx = dx/2########因为多算了一类
    return loss,dx


def softmax_loss(x,y):
    probs = np.exp(x-np.max(x,axis=1).reshape((-1,1)))
    softmax_out = probs/np.sum(probs,axis=1).reshape((-1,1))
    N = x.shape[0]
    loss = -np.sum(np.log(softmax_out[np.arange(N), y]))/N

    dx = softmax_out.copy()
    dx[np.arange(N), y] -= 1
    dx = dx/N
    return loss, dx





def svm_loss(x,y):
    loss = 0.0
    num_train = x.shape[0]
    correct_score = x[range(num_train), list(y)].reshape(-1, 1)
    margin = np.maximum(0,x - correct_score + 1)
    margin[range(num_train), list(y)] = 0
    loss = np.sum(margin) / num_train

    num_pos = np.sum(margin>0,axis=1)
    dx = np.zeros_like(x)
    dx[margin>0] = 1
    dx[np.arange(num_train),y] -= num_pos
    dx = dx/num_train
    return loss,dx




def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the mean
    and variance of each feature, and these averages are used to normalize data
    at test-time.
    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:
    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var
    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7 implementation
    of batch normalization also uses running averages.
    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features
    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #############################################################################
        # TODO: Implement the training-time forward pass for batch normalization.   #
        # Use minibatch statistics to compute the mean and variance, use these      #
        # statistics to normalize the incoming data, and scale and shift the        #
        # normalized data using gamma and beta.                                     #
        #                                                                           #
        # You should store the output in the variable out. Any intermediates that   #
        # you need for the backward pass should be stored in the cache variable.    #
        #                                                                           #
        # You should also use your computed sample mean and variance together with  #
        # the momentum variable to update the running mean and running variance,    #
        # storing your result in the running_mean and running_var variables.        #
        #############################################################################
        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)
        x_hat = (x - sample_mean) / (np.sqrt(sample_var + eps))
        out = gamma * x_hat + beta
        cache = (x,eps,gamma,beta,x_hat,sample_mean,sample_var)
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        # pass
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
    elif mode == 'test':
        #############################################################################
        # TODO: Implement the test-time forward pass for batch normalization. Use   #
        # the running mean and variance to normalize the incoming data, then scale  #
        # and shift the normalized data using gamma and beta. Store the result in   #
        # the out variable.                                                         #
        #############################################################################

        x_ = (x-running_mean)/np.sqrt(running_var+eps)
        out = gamma*x_+beta
        # pass
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    #############################################################################
    # TODO: Implement the backward pass for batch normalization. Store the      #
    # results in the dx, dgamma, and dbeta variables.                           #
    #############################################################################
    x, eps, gamma, beta, x_, running_mean, running_var = cache
    N, D = x.shape
    dx_ = dout * gamma
    d_running_var = np.sum(dx_ * (x - running_mean), axis=0) * (-0.5) * (running_var + eps) ** (-3.0 / 2.0)
    d_running_mean = np.sum(dx_ * (-1.0 / np.sqrt(running_var + eps)), axis=0) + d_running_var * (
    np.sum(-2 * (x - running_mean), axis=0) / N)
    dx = dx_ * (1.0 / np.sqrt(running_var + eps)) + d_running_var * 2 * (
    x - running_mean) / N + d_running_mean * 1.0 / N
    dgamma = np.sum(dout * x_, axis=0)
    dbeta = np.sum(dout, axis=0)
    # pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.
    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.
    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    x, eps, gamma, beta, x_, running_mean, running_var = cache
    N, D = x.shape

    dgamma=np.sum(dout*x_,axis=0)
    dbeta=np.sum(dout,axis=0)

    dx = (1. / N) * gamma * (running_var + eps) ** (-1. / 2.) * (N * dout - np.sum(dout, axis=0)- (x - running_mean) * (running_var + eps) ** (-1.0) * np.sum(dout * (x - running_mean), axis=0))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.
    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.
    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        retain_prob=1.0-p
        mask = np.random.binomial(n=1, p=retain_prob, size=x.shape)
        out=x*mask
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out=x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.
    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx=dout*mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.
    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.
    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.
    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    stride=conv_param['stride']
    pad = conv_param['pad']
    HH, WW = w.shape[2], w.shape[3]
    H, W = x.shape[2], x.shape[3]

    H_,W_=int(1 + (H + 2 * pad - HH) / stride),int(1 + (W + 2 * pad - WW) / stride)
    out=np.random.randn(x.shape[0],w.shape[0],H_,W_)
    x_pad=np.pad(x, ((0,0),(0,0),(1,1),(1,1)), 'constant', constant_values=((0,0),(0,0),(0,0),(0,0)))

    for ni in range(x.shape[0]):
        for fi in range(w.shape[0]):
            for xi in range(H_):
                for yi in range(W_):
                    out[ni, fi, xi, yi] = np.sum(x_pad[ni, :, xi * stride:xi * stride + HH, yi * stride:yi * stride + WW] * w[fi, :, :, :])

            out[ni,fi,:,:]+=b[fi]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.
    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive
    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    x,w,b,conv_param=cache
    stride = conv_param['stride']
    pad = conv_param['pad']
    HH, WW = w.shape[2], w.shape[3]
    H, W = x.shape[2], x.shape[3]

    H_, W_ = int(1 + (H + 2 * pad - HH) / stride), int(1 + (W + 2 * pad - WW) / stride)
    x_pad = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), 'constant', constant_values=((0, 0), (0, 0), (0, 0), (0, 0)))

    dx_pad = np.zeros_like(x_pad)
    dw=np.zeros_like(w)
    db = np.zeros_like(b)

    for ni in range(x.shape[0]):
        for fi in range(w.shape[0]):
            for xi in range(H_):
                for yi in range(W_):
                    dw[fi,:,:,:]+=dout[ni,fi,xi,yi]*x_pad[ni, :, xi * stride:xi * stride + HH, yi * stride:yi * stride + WW]
                    dx_pad[ni, :, xi * stride:xi * stride + HH, yi * stride:yi * stride + WW] += dout[ni, fi, xi, yi] * w[fi,:,:,:]
            db[fi]+=np.sum(dout[ni,fi,:,:])
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    dx=dx_pad[:,:,pad:pad+H,pad:pad+W]
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.
    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions
    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################

    pool_height=pool_param['pool_height']
    pool_width=pool_param['pool_width']
    stride = pool_param['stride']

    H, W = x.shape[2], x.shape[3]

    H_, W_ = int(1 + (H - pool_height) / stride), int(1 + (W - pool_width) / stride)
    out = np.random.randn(x.shape[0], x.shape[1], H_, W_)

    for ni in range(x.shape[0]):
        for ci in range(x.shape[1]):
            for xi in range(H_):
                for yi in range(W_):
                    out[ni, ci, xi, yi] = np.max(x[ni,ci,xi * stride:xi * stride + pool_height,yi * stride:yi * stride + pool_width])
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.
    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.
    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    x, pool_param=cache
    pool_height=pool_param['pool_height']
    pool_width=pool_param['pool_width']
    stride = pool_param['stride']

    H, W = x.shape[2], x.shape[3]

    H_, W_ = dout.shape[2], dout.shape[3]
    dx = np.zeros_like(x)

    for ni in range(x.shape[0]):
        for ci in range(x.shape[1]):
            for xi in range(H_):
                for yi in range(W_):
                    search_range=x[ni, ci, xi * stride:xi * stride + pool_height, yi * stride:yi * stride + pool_width]
                    max_id=np.argmax(np.mat(search_range))
                    max_idx=int(max_id/pool_height)+xi * stride
                    max_idy=max_id%pool_width+yi * stride
                    dx[ni, ci,max_idx,max_idy] +=dout[ni,ci,xi,yi]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.
    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features
    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    N, C, H, W=x.shape
    x_=x.transpose(0,2,3,1).reshape(-1,C)
    out,cache=batchnorm_forward(x_,gamma,beta,bn_param)
    out =out.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.
    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass
    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    N, C, H, W = dout.shape
    dout_ = dout.transpose(0, 2, 3, 1).reshape(-1, C)
    dx,dgamma,dbeta=batchnorm_backward(dout_, cache)
    dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta