import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


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
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
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
    
    ############################################################################
    # TODO: Initialize the weights and biases of the two-layer net. Weights    #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases using the keys 'W2' and 'b2'.                                 #
    ############################################################################
    
    #pass
    self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
    self.params['b1'] = np.zeros(hidden_dim)
    self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b2'] = np.zeros(num_classes)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
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
    ############################################################################
    # TODO: Implement the forward pass for the two-layer net, computing the    #
    # class scores for X and storing them in the scores variable.              #
    ############################################################################
    
    #pass
    W1 = self.params['W1']
    b1 = self.params['b1']
    W2 = self.params['W2']
    b2 = self.params['b2']

    #h1 = X.dot(W1) + b1
    #h1[h1 < 0] = 0
    #scores = h1.dot(W2) + b2
    h1, cache_1 = affine_forward(X, W1, b1)
    h1, ori_h1 = relu_forward(h1)
    scores ,cache_2 = affine_forward(h1, W2, b2)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the two-layer net. Store the loss  #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    
    #pass
    #num_samples = X.shape[0]
    #scores = np.exp(scores)
    #index = np.array(range(num_samples))
    #correct_class_score = scores[index, y[index]]
    #sum_of_scores = np.sum(scores, axis = 1)
    #loss = - np.sum(np.log(correct_class_score / sum_of_scores))
    #loss /= num_samples

    #loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2))

    #num_classes = b2.shape[0]
    #prob = scores / np.outer(sum_of_scores, np.ones(num_classes).T)
    #dscores = prob
    #dscores[index, y[index]] -= 1
    #dscores /= num_samples

    loss , dscores = softmax_loss(scores, y)

    loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2))

    #grads['W2'] = h1.T.dot(dscores)
    #grads['b2'] = np.sum(dscores, axis=0, keepdims=True)

    dh1, grads['W2'], grads['b2'] = affine_backward(dscores, cache_2)
    grads['b2'] = grads['b2'].reshape(self.params['b2'].shape[0])
    #dh1 = dscores.dot(W2.T)
    #dh1[h1 <= 0] = 0

    dh1 = relu_backward(dh1, h1)

    #grads['W1'] = X.T.dot(dh1)
    #grads['b1'] = np.sum(dh1, axis = 0)

    dx, grads['W1'], grads['b1'] = affine_backward(dh1, cache_1)
    grads['b1'] = grads['b1'].reshape(self.params['b1'].shape[0])

    grads ['W2'] += self.reg * W2
    grads ['W1'] += self.reg * W1

    #reshape b1, b2


    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


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

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
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
    
    #pass
    num_layers = self.num_layers
    self.wei_list = [None] * num_layers
    self.bias_list = [None] * num_layers
    #self.drop_list = [None] * num_layers
    self.gamma_list = [None] * (num_layers - 1)
    self.beta_list = [None] * (num_layers - 1)
    for i in range(num_layers) :
      self.wei_list[i] = 'W%d' %(i + 1)
      self.bias_list[i] = 'b%d' %(i + 1)
      if self.use_batchnorm and i != (num_layers - 1) :
        self.gamma_list[i] = 'gamma%d' %(i + 1)
        self.beta_list[i] = 'beta%d' %(i + 1)
        self.params[self.gamma_list[i]] = np.random.normal( 1 , 1e-3,hidden_dims[i]) ########*********
        self.params[self.beta_list[i]] = np.zeros(hidden_dims[i])
      if i == 0 :
        self.params[self.wei_list[i]] = weight_scale * np.random.randn(input_dim, hidden_dims[i])
        self.params[self.bias_list[i]] = np.zeros(hidden_dims[i])
      elif i == (num_layers - 1) : 
        self.params[self.wei_list[i]] = weight_scale * np.random.randn(hidden_dims[i - 1], num_classes)
        self.params[self.bias_list[i]] = np.zeros(num_classes)
      else :
        self.params[self.wei_list[i]] = weight_scale * np.random.randn(hidden_dims[i - 1], hidden_dims[i])
        self.params[self.bias_list[i]] = np.zeros(hidden_dims[i])

        

    
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
        bn_param[mode] = mode

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
    
    #pass
    hidden_scores = {}
    cache = {}
    num_layers = self.num_layers
    hidden_str = [None] * num_layers
    cache_str = [None] * num_layers
    cache_bn_str = [None] * (num_layers - 1)
    cache_drop_str = [None] * num_layers
    for i in range(num_layers) :
      hidden_str[i] = 'hidden%d' %(i + 1)
      cache_str[i] = 'cache%d' %(i + 1)
      if i == 0 : 
        hidden_scores[hidden_str[i]], cache[cache_str[i]] = affine_forward(X, self.params[self.wei_list[i]], self.params[self.bias_list[i]])
        if self.use_batchnorm :
          cache_bn_str[i] = 'cache_bn%d' %(i + 1)
          hidden_scores[hidden_str[i]], cache[cache_bn_str[i]] = batchnorm_forward(hidden_scores[hidden_str[i]], self.params[self.gamma_list[i]], self.params[self.beta_list[i]], self.bn_params[i])
        hidden_scores[hidden_str[i]], _ = relu_forward(hidden_scores[hidden_str[i]])
        if self.use_dropout :
          cache_drop_str[i] = 'cache_drop%d' %(i + 1)
          hidden_scores[hidden_str[i]], cache[cache_drop_str[i]] = dropout_forward(hidden_scores[hidden_str[i]], self.dropout_param)
      elif i == (num_layers - 1) :
        scores, cache[cache_str[i]] = affine_forward(hidden_scores[hidden_str[i - 1]], self.params[self.wei_list[i]], self.params[self.bias_list[i]])
      else :
        hidden_scores[hidden_str[i]], cache[cache_str[i]] = affine_forward(hidden_scores[hidden_str[i - 1]], self.params[self.wei_list[i]], self.params[self.bias_list[i]])
        if self.use_batchnorm :
          cache_bn_str[i] = 'cache_bn%d' %(i + 1)
          hidden_scores[hidden_str[i]], cache[cache_bn_str[i]] = batchnorm_forward(hidden_scores[hidden_str[i]], self.params[self.gamma_list[i]], self.params[self.beta_list[i]], self.bn_params[i])
        hidden_scores[hidden_str[i]], _ = relu_forward(hidden_scores[hidden_str[i]])
        if self.use_dropout :
          cache_drop_str[i] = 'cache_drop%d' %(i + 1)
          hidden_scores[hidden_str[i]], cache[cache_drop_str[i]] = dropout_forward(hidden_scores[hidden_str[i]], self.dropout_param)
    
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
    
    #pass
    
    loss , dscores = softmax_loss(scores, y)

    #loss reg
    sum_of_lossreg = 0.0
    for i in range(num_layers) :
      W = self.params[self.wei_list[i]]
      sum_of_lossreg += np.sum(W * W)

    loss += 0.5 * self.reg * sum_of_lossreg

    dhidden = {}

    for i in range(num_layers, 0, -1) :
      #print i, hidden_str[i - 1]
      if i == num_layers :
        dhidden[hidden_str[i - 1]], grads[self.wei_list[i - 1]], grads[self.bias_list[i - 1]] = affine_backward(dscores, cache[cache_str[i - 1]])
        grads[self.bias_list[i - 1]] = grads[self.bias_list[i - 1]].reshape(self.params[self.bias_list[i - 1]].shape[0])
      else :
        if self.use_dropout :
          dhidden[hidden_str[i]] = dropout_backward(dhidden[hidden_str[i]], cache[cache_drop_str[i - 1]])
        dhidden[hidden_str[i]] = relu_backward(dhidden[hidden_str[i]], hidden_scores[hidden_str[i - 1]])
        if self.use_batchnorm :
          dhidden[hidden_str[i]], grads[self.gamma_list[i - 1]], grads[self.beta_list[i - 1]] = batchnorm_backward_alt(dhidden[hidden_str[i]], cache[cache_bn_str[i - 1]])
        dhidden[hidden_str[i - 1]], grads[self.wei_list[i - 1]], grads[self.bias_list[i - 1]] = affine_backward(dhidden[hidden_str[i]], cache[cache_str[i - 1]])
        grads[self.bias_list[i - 1]] = grads[self.bias_list[i - 1]].reshape(self.params[self.bias_list[i - 1]].shape[0])

    #weight reg
    for i in range(num_layers) :
      W = self.params[self.wei_list[i]]
      grads[self.wei_list[i]] += self.reg * W

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
