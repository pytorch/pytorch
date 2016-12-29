"""
:mod:`torch.optim` is a package for optimizing neural networks.
It provides a wide variety of optimization methods such as SGD, Adam etc.

Currently, the following optimization methods are supported, typically with
options such as weight decay and other bells and whistles.

- SGD
- AdaDelta
- Adagrad
- Adam
- AdaMax
- Averaged SGD
- RProp
- RMSProp


The usage of the Optim package itself is as follows.

1. Construct an optimizer
2. Use ``optimizer.step(...)`` to optimize.
   - Call ``optimizer.zero_grad()`` to zero out the gradient buffers when appropriate

Constructing the optimizer
--------------------------

One first constructs an ``Optimizer`` object by giving it a list of parameters
to optimize, as well as the optimizer options,such as learning rate, weight decay, etc.

Examples::

    optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
    optimizer = optim.Adam([var1, var2], lr = 0.0001)

Per-parameter options
---------------------

In a more advanced usage, one can specify per-layer options by passing each parameter group along with it's custom options.

**Any parameter group that does not have an attribute defined will use the default attributes.**

This is very useful when one wants to specify per-layer learning rates for example.

For example such invocation::

    optim.SGD([
        {'params': model1.parameters()},
        {'params': model2.parameters(), 'lr': 1e-3}],
        lr=1e-2, momentum=0.9)

means that

* ``model1``'s parameters will use the default learning rate of ``1e-2`` and momentum of ``0.9``
* ``model2``'s parameters will use a learning rate of ``1e-3``, and the default momentum of ``0.9``

Then, you can use the optimizer by calling `optimizer.zero_grad()` and `optimizer.step(...)`. Read the next sections.

Taking an optimization step using ``step``
-------------------------------------------------------

``optimizer.step()``
^^^^^^^^^^^^^^^^^^^^

This is a simplified version supported by most optimizers.

The function can be called after computing the gradients with ``backward()``.

Example 2 - training a neural network::

    net = MNISTNet()
    criterion = ClassNLLLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001)

    for data in data_batches:
        input, target = data
            optimizer.zero_grad()
            output = net(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

The step function can be used in two ways.

``optimizer.step(closure)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``step`` function takes a user-defined closure that computes f(x) and returns the loss.

The closure should look somewhat like this::

    def f_closure(x):
        optimizer.zero_grad()
        loss = f(x)
        loss.backward()
        return loss

Example 1 - training a neural network::

    net = MNISTNet()
    criterion = ClassNLLLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001)

    for data in data_batches:
        input, target = data
            def closure():
                optimizer.zero_grad()
                output = net(input)
                    loss = criterion(output, target)
                    loss.backward()
                    return loss
            optimizer.step(closure)

Note:
    **Why is this supported?**
    Some optimization algorithms such as Conjugate Gradient and LBFGS need to evaluate their function
    multiple times. For such optimization methods, the function (i.e. the closure) has to be defined.
"""

from .adadelta import Adadelta
from .adagrad import Adagrad
from .adam import Adam
from .adamax import Adamax
from .asgd import ASGD
from .sgd import SGD
from .rprop import Rprop
from .rmsprop import RMSprop
from .optimizer import Optimizer

del adadelta
del adagrad
del adam
del adamax
del asgd
del sgd
del rprop
del rmsprop
del optimizer
