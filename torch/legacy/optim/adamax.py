import torch


def adamax(opfunc, x, config, state=None):
    """ An implementation of AdaMax http://arxiv.org/pdf/1412.6980.pdf

    ARGS:

    - 'opfunc' : a function that takes a single input (X), the point
                of a evaluation, and returns f(X) and df/dX
    - 'x'      : the initial point
    - 'config` : a table with configuration parameters for the optimizer
    - 'config.learningRate'      : learning rate
    - 'config.beta1'             : first moment coefficient
    - 'config.beta2'             : second moment coefficient
    - 'config.epsilon'           : for numerical stability
    - 'config.weightDecay'       : weight decay
    - 'state'                    : a table describing the state of the optimizer;
                                   after each call the state is modified.

    RETURN:
    - `x`     : the new x vector
    - `f(x)`  : the value of optimized function, evaluated before the update

    """
    # (0) get/update state
    if config is None and state is None:
        raise ValueError("adamax requires a dictionary to retain state between iterations")
    state = state if state is not None else config
    lr = config.get('learningRate', 0.002)
    beta1 = config.get('beta1', 0.9)
    beta2 = config.get('beta2', 0.999)
    epsilon = config.get('epsilon', 1e-38)
    wd = config.get('weightDecay', 0)

    # (1) evaluate f(x) and df/dx
    fx, dfdx = opfunc(x)

    # (2) weight decay
    if wd != 0:
        dfdx.add_(wd, x)

    # Initialization
    if 't' not in state:
        state['t'] = 0
        # Exponential moving average of gradient values
        state['m'] = x.new().resize_as_(dfdx).zero_()
        # Exponential moving average of the infinity norm
        state['u'] = x.new().resize_as_(dfdx).zero_()
        # A tmp tensor to hold the input to max()
        state['max'] = x.new(*((2,) + dfdx.size())).zero_()

    state['t'] += 1

    # Update biased first moment estimate.
    state['m'].mul_(beta1).add_(1 - beta1, dfdx)
    # Update the exponentially weighted infinity norm.
    state['max'][0].copy_(state['u']).mul_(beta2)
    state['max'][1].copy_(dfdx).abs_().add_(epsilon)
    torch.max(state['max'], 0, out=(state['u'], state['u'].new().long()))

    biasCorrection1 = 1 - beta1 ** state['t']
    stepSize = lr / biasCorrection1
    # (2) update x
    x.addcdiv_(-stepSize, state['m'], state['u'])

    # return x*, f(x) before optimization
    return x, fx
