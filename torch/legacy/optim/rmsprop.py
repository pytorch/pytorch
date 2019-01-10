import torch


def rmsprop(opfunc, x, config, state=None):
    """ An implementation of RMSprop

    ARGS:

    - 'opfunc' : a function that takes a single input (X), the point
                of a evaluation, and returns f(X) and df/dX
    - 'x'      : the initial point
    - 'config` : a table with configuration parameters for the optimizer
    - 'config['learningRate']'      : learning rate
    - 'config['alpha']'             : smoothing constant
    - 'config['epsilon']'           : value with which to initialise m
    - 'config['weightDecay']'       : weight decay
    - 'state'                    : a table describing the state of the optimizer;
                                after each call the state is modified
    - 'state['m']'                  : leaky sum of squares of parameter gradients,
    - 'state['tmp']'                : and the square root (with epsilon smoothing)

    RETURN:
    - `x`     : the new x vector
    - `f(x)`  : the function, evaluated before the update

    """
    # (0) get/update state
    if config is None and state is None:
        raise ValueError("rmsprop requires a dictionary to retain state between iterations")
    state = state if state is not None else config
    lr = config.get('learningRate', 1e-2)
    alpha = config.get('alpha', 0.99)
    epsilon = config.get('epsilon', 1e-8)
    wd = config.get('weightDecay', 0)

    # (1) evaluate f(x) and df/dx
    fx, dfdx = opfunc(x)

    # (2) weight decay
    if wd != 0:
        dfdx.add_(wd, x)

    # (3) initialize mean square values and square gradient storage
    if 'm' not in state:
        state['m'] = x.new().resize_as_(dfdx).zero_()
        state['tmp'] = x.new().resize_as_(dfdx)

    # (4) calculate new (leaky) mean squared values
    state['m'].mul_(alpha)
    state['m'].addcmul_(1.0 - alpha, dfdx, dfdx)

    # (5) perform update
    torch.sqrt(state['m'], out=state['tmp']).add_(epsilon)
    x.addcdiv_(-lr, dfdx, state['tmp'])

    # return x*, f(x) before optimization
    return x, fx
