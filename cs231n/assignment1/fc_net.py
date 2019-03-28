import numpy as np
from layers import rel_error,affine_relu_backward,affine_backward,affine_forward,relu_forward,relu_backward,affine_relu_forward
from layers import softmax_loss
from layers import *
from gradient_check import eval_numerical_gradient

class TwoLayerNet(object):
    """
        A two-layer fully-connected neural network with ReLU nonlinearity and
        softmax loss that uses a modular layer design. We assume an input dimension
        of D, a hidden dimension of H, and perform classification over C classes.
        The architecure should be affine - relu - affine - softmax.
        Note that this class does not implement gradient descent; instead, it
        will interact with a separate Solver object that is responsible for running
        optimization.
        The learnable parameters of the model are stored in the dictionary
        self.params that maps parameter names to numpy arrays.
        """
    def __init__(self,input_dim=3*32*32,hidden_dim=100,num_classes=10,weight_scale=1e-3,reg=0.0):
        """
                Initialize a new network.
                Inputs:
                - input_dim: An integer giving the size of the input
                - hidden_dim: An integer giving the size of the hidden layer
                - num_classes: An integer giving the number of classes to classify
                - dropout: Scalar between 0 and 1 giving dropout strength.
                - weight_scale: Scalar giving the standard deviation for random
                  initialization of the weights.
                - reg: Scalar giving L2 regularization strength.
                """
        self.params = {}
        self.reg = reg
        self.params['W1'] = weight_scale*np.random.randn(input_dim,hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = weight_scale*np.random.randn(hidden_dim,num_classes)
        self.params['b2'] = np.zeros(num_classes)

    def loss(self,X,y=None):
        """
                Compute loss and gradient for a minibatch of data.
                Inputs:
                - X: Array of input data of shape (N, d_1, ..., d_k)
                - y: Array of labels, of shape (N,). y[i] gives the label for X[i].
                Returns:
                If y is None, then run a test-time forward pass of the model and return:
                - scores: Array of shape (N, C) giving classification scores, where
                  scores[i, c] is the classification score for X[i] and class c.
                If y is not None, then run a training-time forward and backward pass and
                return a tuple of:
                - loss: Scalar value giving the loss
                - grads: Dictionary with the same keys as self.params, mapping parameter
                  names to gradients of the loss with respect to those parameters.
                """
        scores = None
        W1,b1 = self.params['W1'],self.params['b1']
        W2,b2 = self.params['W2'],self.params['b2']

        ar1_out,ar1_cache = affine_relu_forward(X,W1,b1)
        ar2_out,ar2_cache = affine_forward(ar1_out,W2,b2)

        scores = ar2_out

        if y is None:
            return scores
        loss,grads = 0,{}
        loss,dout = softmax_loss(scores,y)
        loss = loss+0.5*self.reg*np.sum(W1*W1)+0.5*self.reg*np.sum(W2*W2)
        dx2,dw2,db2 = affine_backward(dout,ar2_cache)
        grads['W2'] = dw2 +self.reg*W2
        grads['b2'] = db2
        dx1,dw1,db1 = affine_relu_backward(dx2,ar1_cache)
        grads['W1'] = dw1+self.reg*W1
        grads['b1'] = db1

        return loss,grads

class FullyConnectedNet(object):
    """
        A fully-connected neural network with an arbitrary number of hidden layers,
        ReLU nonlinearities, and a softmax loss function. This will also implement
        dropout and batch normalization as options. For a network with L layers,
        the architecture will be
        {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
        where batch normalization and dropout are optional, and the {...} block is
        repeated L - 1 times.
        Similar to the TwoLayerNet above, learnable parameters are stored in the
        self.params dictionary and will be learned using the Solver class.
        """
    def __init__(self,hidden_dims,input_dim=3*32*32,num_classes=10,dropout=0,
                 use_batchnorm=False,reg=0.0,weight_scale=1e-2,dtype=np.float32,seed=None):
        """
                Initialize a new FullyConnectedNet.
                Inputs:
                - hidden_dims: A list of integers giving the size of each hidden layer.
                - input_dim: An integer giving the size of the input.
                - num_classes: An integer giving the number of classes to classify.
                - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
                  the network should not use dropout at all.
                - use_batchnorm: Whether or not the network should use batch normalization.
                - reg: Scalar giving L2 regularization strength.
                - weight_scale: Scalar giving the standard deviation for random
                  initialization of the weights.
                - dtype: A numpy datatype object; all computations will be performed using
                  this datatype. float32 is faster but less accurate, so you should use
                  float64 for numeric gradient checking.
                - seed: If not None, then pass this random seed to the dropout layers. This
                  will make the dropout layers deteriminstic so we can gradient check the
                  model.
                """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout>0
        self.reg = reg
        self.num_layers = 1+len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        for i in range(self.num_layers-1):
            self.params['W'+str(i+1)] = np.random.normal(0,weight_scale,[input_dim,hidden_dims[i]])
            self.params['b'+str(i+1)] = np.zeros([hidden_dims[i]])

            if self.use_batchnorm:
                self.params['beta'+str(i+1)] = np.zeros([hidden_dims[i]])
                self.params['gamma'+str(i+1)] = np.ones([hidden_dims[i]])

            input_dim = hidden_dims[i]

        self.params['W'+str(self.num_layers)] = np.random.normal(0,weight_scale,[input_dim,num_classes])
        self.params['b'+str(self.num_layers)] = np.zeros([num_classes])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode':'train','p':dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed
                # With batch normalization we need to keep track of running means and
                # variances, so we need to pass a special bn_param object to each batch
                # normalization layer. You should pass self.bn_params[0] to the forward pass
                # of the first batch normalization layer, self.bn_params[1] to the forward
                # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode':'train'} for i in range(self.num_layers-1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self,X,y=None):
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        if self.use_dropout :
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None


        inputi = X
        batch_size = X.shape[0]
        X = np.reshape(X,[batch_size,-1])

        fc_cache_list = []
        relu_cache_list = []
        bn_cache_list = []
        dropout_cache_list = []


        for i in range(self.num_layers-1):
            fc_act,fc_cache= affine_forward(X,self.params['W'+str(i+1)],self.params['b'+str(i+1)])
            fc_cache_list.append(fc_cache)
            if self.use_batchnorm:
                bn_act,bn_cache = batchnorm_forward(fc_act,self.params['gamma'+str(i+1)],self.params['beta'+str(i+1)],self.bn_params[i])
                bn_cache_list.append(bn_cache)
                relu_act,relu_cache = relu_forward(bn_act)
                relu_cache_list.append(relu_cache)
            else:
                relu_act,relu_cache = relu_forward(fc_act)
                relu_cache_list.append(relu_cache)
            if self.use_dropout:
                relu_act,dropout_cache = dropout_forward(relu_act,self.dropout_param)
                dropout_cache_list.append(dropout_cache)

            X = relu_act.copy()
        ########最后一层
        scores,final_cache = affine_forward(X,self.params['W'+str(self.num_layers)],self.params['b'+str(self.num_layers)])
        #
        # for layer in range(self.num_layers):
        #     Wi,bi = self.params['W%d'%(layer+1)],self.params['b%d'%(layer+1)]
        #     outi,fc_cachei = affine_forward(inputi,Wi,bi)
        #     fc_cache_list.append(fc_cachei)
        #
        #     if self.use_batchnorm and layer!=self.num_layers-1:
        #         gammai,betai = self.params['gamma%d'%(layer+1)],self.params['beta%d'%(layer+1)]
        #
        #         outi,bn_cachei = batchnorm_forward(outi,gammai,betai,self.bn_params[layer])
        #         bn_cache_list.append(bn_cachei)
        #     outi,relu_cachei = relu_forward(outi)
        #     relu_cache_list.append(relu_cachei)
        #
        #     if self.use_dropout:
        #         outi,dropout_cachei = dropout_forward(outi,self.dropout_param)
        #         dropout_cache_list.append(dropout_cachei)
        #
        #     inputi = outi
        #
        # scores = outi

        if mode == 'test':
            return scores

        loss,grads = 0.0,{}

        loss,dsoft = softmax_loss(scores,y)
        loss += 0.5*self.reg*(np.sum(np.square(self.params['W'+str(self.num_layers)])))
        #########最后一层的反向传播
        dx_last,dw_last,db_last = affine_backward(dsoft,final_cache)
        grads['W'+str(self.num_layers)] = dw_last+self.reg*self.params['W'+str(self.num_layers)]
        grads['b'+str(self.num_layers)] = db_last

        for i in range(self.num_layers-1,0,-1):

            if self.use_dropout:
                dx_last = dropout_backward(dx_last,dropout_cache_list[i-1])

            drelu = relu_backward(dx_last,relu_cache_list[i-1])
            if self.use_batchnorm:
                dbatchnorm,dgamma,dbeta = batchnorm_backward(drelu,bn_cache_list[i-1])
                dx_last,dw_last,db_last = affine_backward(dbatchnorm,fc_cache_list[i-1])
                grads['beta'+str(i)] = dbeta
                grads['gamma'+str(i)] = dgamma
            else:
                dx_last,dw_last,db_last = affine_backward(drelu,fc_cache_list[i-1])

            grads['W'+str(i)] = dw_last+self.reg*self.params['W'+str(i)]
            grads['b'+str(i)] = db_last

            loss += 0.5*self.reg*(np.sum(np.square(self.params['W'+str(i)])))

        return loss,grads
        #
        #
        #
        # data_loss,dout = softmax_loss(scores,y)
        # W_square_sum = 0
        # for layer in range (self.num_layers):
        #     Wi = self.params['W%d'%(layer+1)]
        #     W_square_sum += (np.sum(Wi**2))
        # reg_loss = 0.5*self.reg*W_square_sum
        # loss = data_loss+reg_loss
        #
        # for layer in list(range(self.num_layers,0,-1)):
        #
        #     if self.use_dropout:
        #         dout = dropout_backward(dout,dropout_cache_list[layer-1])
        #     ##relu
        #     dout = relu_backward(dout,relu_cache_list[layer-1])
        #     ###vatch_norm
        #     if self.use_batchnorm and layer!=self.num_layers:
        #         dout,dgamma,dbeta = batchnorm_backward(dout,bn_cache_list[layer-1])
        #         grads['gamma%d'%(layer)] = dgamma
        #         grads['beta%d'%(layer)] = dbeta
        #     #####bachforward
        #     dxi,dWi,dbi = affine_backward(dout,fc_cache_list[layer-1])
        #     dWi += self.reg*self.params['W%d'%(layer)]
        #     grads['W%d'%(layer)] = dWi
        #     grads['b%d'%(layer)] = dbi
        #     dout = np.dot(dout,self.params['W%d'%(layer)].T)#######比较重要
        #
        # return loss,grads

    class FullyConnectedNet(object):
        """
        A fully-connected neural network with an arbitrary number of hidden layers,
        ReLU nonlinearities, and a softmax loss function. This will also implement
        dropout and batch normalization as options. For a network with L layers,
        the architecture will be

        {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

        where batch normalization and dropout are optional, and the {...} block is
        repeated L - 1 times.

        Similar to the TwoLayerNet above, learnable parameters are stored in the
        self.params dictionary and will be learned using the Solver class.
        """

        def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10,
                     dropout=0, use_batchnorm=False, reg=0.0,
                     weight_scale=1e-2, dtype=np.float32, seed=None):
            """
            Initialize a new FullyConnectedNet.

            Inputs:
            - hidden_dims: A list of integers giving the size of each hidden layer.
            - input_dim: An integer giving the size of the input.
            - num_classes: An integer giving the number of classes to classify.
            - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
              the network should not use dropout at all.
            - use_batchnorm: Whether or not the network should use batch normalization.
            - reg: Scalar giving L2 regularization strength.
            - weight_scale: Scalar giving the standard deviation for random
              initialization of the weights.
            - dtype: A numpy datatype object; all computations will be performed using
              this datatype. float32 is faster but less accurate, so you should use
              float64 for numeric gradient checking.
            - seed: If not None, then pass this random seed to the dropout layers. This
              will make the dropout layers deteriminstic so we can gradient check the
              model.
            """
            self.use_batchnorm = use_batchnorm
            self.use_dropout = dropout > 0
            self.reg = reg
            self.num_layers = 1 + len(hidden_dims)
            self.dtype = dtype
            self.params = {}

            ############################################################################
            # TODO: Initialize the parameters of the network, storing all values in    #
            # the self.params dictionary. Store weights and biases for the first layer #
            # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
            # initialized from a normal distribution with standard deviation equal to  #
            # weight_scale and biases should be initialized to zero.                   #
            #                                                                          #
            # When using batch normalization, store scale and shift parameters for the #
            # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
            # beta2, etc. Scale parameters should be initialized to one and shift      #
            # parameters should be initialized to zero.                                #
            ############################################################################
            layer_input_dim = input_dim
            for i, hd in enumerate(hidden_dims):
                self.params['W%d' % (i + 1)] = weight_scale * np.random.randn(layer_input_dim, hd)
                self.params['b%d' % (i + 1)] = weight_scale * np.zeros(hd)
                if self.use_batchnorm:
                    self.params['gamma%d' % (i + 1)] = np.ones(hd)
                    self.params['beta%d' % (i + 1)] = np.zeros(hd)
                layer_input_dim = hd
            self.params['W%d' % (self.num_layers)] = weight_scale * np.random.randn(layer_input_dim, num_classes)
            self.params['b%d' % (self.num_layers)] = weight_scale * np.zeros(num_classes)
            # pass
            ############################################################################
            #                             END OF YOUR CODE                             #
            ############################################################################

            # When using dropout we need to pass a dropout_param dictionary to each
            # dropout layer so that the layer knows the dropout probability and the mode
            # (train / test). You can pass the same dropout_param to each dropout layer.
            self.dropout_param = {}
            if self.use_dropout:
                self.dropout_param = {'mode': 'train', 'p': dropout}
                if seed is not None:
                    self.dropout_param['seed'] = seed

            # With batch normalization we need to keep track of running means and
            # variances, so we need to pass a special bn_param object to each batch
            # normalization layer. You should pass self.bn_params[0] to the forward pass
            # of the first batch normalization layer, self.bn_params[1] to the forward
            # pass of the second batch normalization layer, etc.
            self.bn_params = []
            if self.use_batchnorm:
                self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]

            # Cast all parameters to the correct datatype
            for k, v in self.params.iteritems():
                self.params[k] = v.astype(dtype)

        def loss(self, X, y=None):
            """
            Compute loss and gradient for the fully-connected net.
            Input / output: Same as TwoLayerNet above.
            """
            X = X.astype(self.dtype)
            mode = 'test' if y is None else 'train'

            # Set train/test mode for batchnorm params and dropout param since they
            # behave differently during training and testing.
            if self.dropout_param is not None:
                self.dropout_param['mode'] = mode
            if self.use_batchnorm:
                for bn_param in self.bn_params:
                    bn_param['mode'] = mode

            scores = None
            ############################################################################
            # TODO: Implement the forward pass for the fully-connected net, computing  #
            # the class scores for X and storing them in the scores variable.          #
            #                                                                          #
            # When using dropout, you'll need to pass self.dropout_param to each       #
            # dropout forward pass.                                                    #
            #                                                                          #
            # When using batch normalization, you'll need to pass self.bn_params[0] to #
            # the forward pass for the first batch normalization layer, pass           #
            # self.bn_params[1] to the forward pass for the second batch normalization #
            # layer, etc.                                                              #
            ############################################################################
            layer_input = X
            ar_cache = {}
            dp_cache = {}

            for lay in xrange(self.num_layers - 1):
                if self.use_batchnorm:
                    layer_input, ar_cache[lay] = affine_bn_relu_forward(layer_input,
                                                                        self.params['W%d' % (lay + 1)],
                                                                        self.params['b%d' % (lay + 1)],
                                                                        self.params['gamma%d' % (lay + 1)],
                                                                        self.params['beta%d' % (lay + 1)],
                                                                        self.bn_params[lay])
                else:
                    layer_input, ar_cache[lay] = affine_relu_forward(layer_input, self.params['W%d' % (lay + 1)],
                                                                     self.params['b%d' % (lay + 1)])

                if self.use_dropout:
                    layer_input, dp_cache[lay] = dropout_forward(layer_input, self.dropout_param)

            ar_out, ar_cache[self.num_layers] = affine_forward(layer_input, self.params['W%d' % (self.num_layers)],
                                                               self.params['b%d' % (self.num_layers)])
            scores = ar_out
            # pass
            ############################################################################
            #                             END OF YOUR CODE                             #
            ############################################################################

            # If test mode return early
            if mode == 'test':
                return scores

            loss, grads = 0.0, {}
            ############################################################################
            # TODO: Implement the backward pass for the fully-connected net. Store the #
            # loss in the loss variable and gradients in the grads dictionary. Compute #
            # data loss using softmax, and make sure that grads[k] holds the gradients #
            # for self.params[k]. Don't forget to add L2 regularization!               #
            #                                                                          #
            # When using batch normalization, you don't need to regularize the scale   #
            # and shift parameters.                                                    #
            #                                                                          #
            # NOTE: To ensure that your implementation matches ours and you pass the   #
            # automated tests, make sure that your L2 regularization includes a factor #
            # of 0.5 to simplify the expression for the gradient.                      #
            ############################################################################
            loss, dscores = softmax_loss(scores, y)
            dhout = dscores
            loss = loss + 0.5 * self.reg * np.sum(
                self.params['W%d' % (self.num_layers)] * self.params['W%d' % (self.num_layers)])
            dx, dw, db = affine_backward(dhout, ar_cache[self.num_layers])
            grads['W%d' % (self.num_layers)] = dw + self.reg * self.params['W%d' % (self.num_layers)]
            grads['b%d' % (self.num_layers)] = db
            dhout = dx
            for idx in xrange(self.num_layers - 1):
                lay = self.num_layers - 1 - idx - 1
                loss = loss + 0.5 * self.reg * np.sum(self.params['W%d' % (lay + 1)] * self.params['W%d' % (lay + 1)])
                if self.use_dropout:
                    dhout = dropout_backward(dhout, dp_cache[lay])
                if self.use_batchnorm:
                    dx, dw, db, dgamma, dbeta = affine_bn_relu_backward(dhout, ar_cache[lay])
                else:
                    dx, dw, db = affine_relu_backward(dhout, ar_cache[lay])
                grads['W%d' % (lay + 1)] = dw + self.reg * self.params['W%d' % (lay + 1)]
                grads['b%d' % (lay + 1)] = db
                if self.use_batchnorm:
                    grads['gamma%d' % (lay + 1)] = dgamma
                    grads['beta%d' % (lay + 1)] = dbeta
                dhout = dx
            # pass
            ############################################################################
            #                             END OF YOUR CODE                             #
            ############################################################################

            return loss, grads

#
#
# class FullyConnectedNet(object):
#     """
#     A fully-connected neural network with an arbitrary number of hidden layers,
#     ReLU nonlinearities, and a softmax loss function. This will also implement
#     dropout and batch normalization as options. For a network with L layers,
#     the architecture will be
#     {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
#     where batch normalization and dropout are optional, and the {...} block is
#     repeated L - 1 times.
#     Similar to the TwoLayerNet above, learnable parameters are stored in the
#     self.params dictionary and will be learned using the Solver class.
#     """
#
#     def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10,
#                  dropout=0, use_batchnorm=False, reg=0.0,
#                  weight_scale=1e-2, dtype=np.float32, seed=None):
#         """
#         Initialize a new FullyConnectedNet.
#         Inputs:
#         - hidden_dims: A list of integers giving the size of each hidden layer.
#         - input_dim: An integer giving the size of the input.
#         - num_classes: An integer giving the number of classes to classify.
#         - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
#           the network should not use dropout at all.
#         - use_batchnorm: Whether or not the network should use batch normalization.
#         - reg: Scalar giving L2 regularization strength.
#         - weight_scale: Scalar giving the standard deviation for random
#           initialization of the weights.
#         - dtype: A numpy datatype object; all computations will be performed using
#           this datatype. float32 is faster but less accurate, so you should use
#           float64 for numeric gradient checking.
#         - seed: If not None, then pass this random seed to the dropout layers. This
#           will make the dropout layers deteriminstic so we can gradient check the
#           model.
#         """
#         self.use_batchnorm = use_batchnorm
#         self.use_dropout = dropout > 0
#         self.reg = reg
#         self.num_layers = 1 + len(hidden_dims)
#         self.dtype = dtype
#         self.params = {}
#
#         ############################################################################
#         # TODO: Initialize the parameters of the network, storing all values in    #
#         # the self.params dictionary. Store weights and biases for the first layer #
#         # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
#         # initialized from a normal distribution with standard deviation equal to  #
#         # weight_scale and biases should be initialized to zero.                   #
#         #                                                                          #
#         # When using batch normalization, store scale and shift parameters for the #
#         # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
#         # beta2, etc. Scale parameters should be initialized to one and shift      #
#         # parameters should be initialized to zero.                                #
#         ############################################################################
#
#         for layer in range(self.num_layers):
#             if layer == 0:
#                 layer_dim = (input_dim, hidden_dims[layer])
#             elif layer == self.num_layers - 1:
#                 layer_dim = (hidden_dims[layer - 1], num_classes)
#             else:
#                 layer_dim = (hidden_dims[layer - 1], hidden_dims[layer])
#             self.params['W%d' % (layer + 1)] = weight_scale * np.random.randn(layer_dim[0], layer_dim[1])
#             self.params['b%d' % (layer + 1)] = np.zeros(layer_dim[1])
#             # batch normalization intialize
#             if self.use_batchnorm and layer != self.num_layers - 1:
#                 self.params['gamma%d' % (layer + 1)] = np.ones(layer_dim[1])
#                 self.params['beta%d' % (layer + 1)] = np.zeros(layer_dim[1])
#         ############################################################################
#         #                             END OF YOUR CODE                             #
#         ############################################################################
#
#         # When using dropout we need to pass a dropout_param dictionary to each
#         # dropout layer so that the layer knows the dropout probability and the mode
#         # (train / test). You can pass the same dropout_param to each dropout layer.
#         self.dropout_param = {}
#         if self.use_dropout:
#             self.dropout_param = {'mode': 'train', 'p': dropout}
#             if seed is not None:
#                 self.dropout_param['seed'] = seed
#
#         # With batch normalization we need to keep track of running means and
#         # variances, so we need to pass a special bn_param object to each batch
#         # normalization layer. You should pass self.bn_params[0] to the forward pass
#         # of the first batch normalization layer, self.bn_params[1] to the forward
#         # pass of the second batch normalization layer, etc.
#         self.bn_params = []
#         if self.use_batchnorm:
#             self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
#
#         # Cast all parameters to the correct datatype
#         for k, v in self.params.items():
#             self.params[k] = v.astype(dtype)
#
#     def loss(self, X, y=None):
#         """
#         Compute loss and gradient for the fully-connected net.
#         Input / output: Same as TwoLayerNet above.
#         """
#         X = X.astype(self.dtype)
#         mode = 'test' if y is None else 'train'
#
#         # Set train/test mode for batchnorm params and dropout param since they
#         # behave differently during training and testing.
#         if self.use_dropout:
#             self.dropout_param['mode'] = mode
#         if self.use_batchnorm:
#             for bn_param in self.bn_params:
#                 bn_param['mode'] = mode
#
#         scores = None
#         ############################################################################
#         # TODO: Implement the forward pass for the fully-connected net, computing  #
#         # the class scores for X and storing them in the scores variable.          #
#         #                                                                          #
#         # When using dropout, you'll need to pass self.dropout_param to each       #
#         # dropout forward pass.                                                    #
#         #                                                                          #
#         # When using batch normalization, you'll need to pass self.bn_params[0] to #
#         # the forward pass for the first batch normalization layer, pass           #
#         # self.bn_params[1] to the forward pass for the second batch normalization #
#         # layer, etc.                                                              #
#         ############################################################################
#         inputi = X
#         # use for BP
#         fc_cache_list = []
#         relu_cache_list = []
#         bn_cache_list = []
#         dropout_cache_list = []
#         for layer in range(self.num_layers):
#             # forward
#             Wi, bi = self.params['W%d' % (layer + 1)], self.params['b%d' % (layer + 1)]
#             outi, fc_cachei = affine_forward(inputi, Wi, bi)
#             fc_cache_list.append(fc_cachei)
#
#             # batch normalization:the last layer of the network should not be normalized
#             if self.use_batchnorm and layer != self.num_layers - 1:
#                 gammai, betai = self.params['gamma%d' % (layer + 1)], self.params['beta%d' % (layer + 1)]
#                 outi, bn_cachei = batchnorm_forward(outi, gammai, betai, self.bn_params[layer])
#                 bn_cache_list.append(bn_cachei)
#             # relu
#             outi, relu_cachei = relu_forward(outi)
#             relu_cache_list.append(relu_cachei)
#
#             # dropout
#             if self.use_dropout:
#                 outi, dropout_cachei = dropout_forward(outi, self.dropout_param)
#                 dropout_cache_list.append(dropout_cachei)
#
#             inputi = outi
#
#         scores = outi
#         ############################################################################
#         #                             END OF YOUR CODE                             #
#         ############################################################################
#
#         # If test mode return early
#         if mode == 'test':
#             return scores
#
#         loss, grads = 0.0, {}
#         ############################################################################
#         # TODO: Implement the backward pass for the fully-connected net. Store the #
#         # loss in the loss variable and gradients in the grads dictionary. Compute #
#         # data loss using softmax, and make sure that grads[k] holds the gradients #
#         # for self.params[k]. Don't forget to add L2 regularization!               #
#         #                                                                          #
#         # When using batch normalization, you don't need to regularize the scale   #
#         # and shift parameters.                                                    #
#         #                                                                          #
#         # NOTE: To ensure that your implementation matches ours and you pass the   #
#         # automated tests, make sure that your L2 regularization includes a factor #
#         # of 0.5 to simplify the expression for the gradient.                      #
#         ############################################################################
#
#         data_loss, dout = softmax_loss(scores, y)
#         W_square_sum = 0
#         for layer in range(self.num_layers):
#             Wi = self.params['W%d' % (layer + 1)]
#             W_square_sum += (np.sum(Wi ** 2))
#         reg_loss = 0.5 * self.reg * W_square_sum
#         loss = data_loss + reg_loss
#
#         for layer in list(range(self.num_layers, 0, -1)):
#             # dropout
#             if self.use_dropout:
#                 dout = dropout_backward(dout, dropout_cache_list[layer - 1])
#             # relu
#             dout = relu_backward(dout, relu_cache_list[layer - 1])
#             # batch normalization: the last layer of the network should not be normalized
#             if self.use_batchnorm and layer != self.num_layers:
#                 dout, dgamma, dbeta = batchnorm_backward(dout, bn_cache_list[layer - 1])
#                 grads['gamma%d' % (layer)] = dgamma
#                 grads['beta%d' % (layer)] = dbeta
#
#             # backforward
#             dxi, dWi, dbi = affine_backward(dout, fc_cache_list[layer - 1])
#             dWi += self.reg * self.params['W%d' % (layer)]
#
#             grads['W%d' % (layer)] = dWi
#             grads['b%d' % (layer)] = dbi
#
#             dout = np.dot(dout, self.params['W%d' % (layer)].T)
#
#         ############################################################################
#         #                             END OF YOUR CODE                             #
#         ############################################################################
#
#         return loss, grads


# ##########测试代码
# N,D,H,C = 3,5,50,7
# X = np.random.randn(N,D)
# y = np.random.randint(C,size=N)
#
# std = 1e-2
# model = TwoLayerNet(input_dim = D,hidden_dim=H,num_classes=C,weight_scale=std)
# print('testing initialization')
# W1_std = abs(model.params['W1'].std()-std)
# b1 = model.params['b1']
# W2_std = abs(model.params['W2'].std()-std)
# b2 = model.params['b2']
# assert W1_std < std / 10, 'First layer weights do not seem right'
# assert np.all(b1 == 0), 'First layer biases do not seem right'
# assert W2_std < std / 10, 'Second layer weights do not seem right'
# assert np.all(b2 == 0), 'Second layer biases do not seem right'
#
# print('testing forward pass')
# model.params['W1'] = np.linspace(-0.7,0.3,num=D*H).reshape(D,H)
# model.params['b1'] = np.linspace(-0.1,0.9,num=H)
# model.params['W2'] = np.linspace(-0.3,0.4,num=H*C).reshape(H,C)
# model.params['b2'] = np.linspace(-0.9,0.1,num=C)
# X = np.linspace(-5.5,4.5,num=N*D).reshape(D,N).T
# scores = model.loss(X)
# correct_scores = np.asarray(
# [[11.53165108, 12.2917344, 13.05181771, 13.81190102, 14.57198434,
# 15.33206765, 16.09215096],
# [12.05769098, 12.74614105, 13.43459113, 14.1230412, 14.81149128,
# 15.49994135, 16.18839143],
# [12.58373087, 13.20054771, 13.81736455, 14.43418138, 15.05099822,
# 15.66781506, 16.2846319 ]])
# scores_diff = np.abs(scores-correct_scores).sum()
# assert scores_diff < 1e-6, 'Problem with test‐time forward pass'
#
# print('Testing traing loss (no reg)')
# y = np.asarray([0,5,1])
# loss,grads = model.loss(X,y)
# correct_loss = 3.4702243556
# assert abs(loss-correct_loss) < 1e-10, 'Problem with training‐time loss'
#
# model.reg = 1.0
# loss, grads = model.loss(X, y)
# correct_loss = 26.5948426952
# assert abs(loss-correct_loss) < 1e-10, 'Problem with regularization loss'
#
# for reg in [0.0, 0.7]:
#   print('Running numeric gradient check with reg = ', reg)
#   model.reg = reg
#   loss, grads = model.loss(X, y)
#
#   for name in sorted(grads):
#     f = lambda _: model.loss(X, y)[0]
#     grad_num = eval_numerical_gradient(f, model.params[name], verbose=False)
#     print('%s relative error: %.3e' % (name, rel_error(grad_num, grads[name])))
#
