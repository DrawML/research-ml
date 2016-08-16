research-ml
===================
Research about ML Componentization and Experiment Realization Service.

###[Please Read Issues!](https://github.com/DrawML/research-ml/issues)

Research Objectives
---------------------
1. ML Componentization.
2. Specify detail XML format for ML Components.
3. Specify the algorithm which translates from XML to tensorflow object code.
4. Prototyping objectives above.


What we support
---------------------
* Model
    * linear regression   (regularization is supported)
    * logistic regression (regularization is supported)
    * softmax regression  (regularization is supported)
    * neural network
    * convolution neural network
    * recurrent neural network

* Layer type
    * none(normal layer)
    * convolution
    * recurrent
    * lstm
    * gru


* Initializer
    * random uniform
    * random normal

* Activation function
    * relu

* Pooling 
    * max pooling

* Padding
    * same
    * valid

* Optimizer
    * gradient descent
    * adadelta
    * adagrad
    * momentum
    * adam
    * ftrl
    * rmsprop
