
def adadelta(opfunc, x, config, state=None):
    """ADADELTA implementation http://arxiv.org/abs/1212.5701

    ARGUMENTS:
    - `opfunc` : a function that takes a single input (X), the point of
                evaluation, and returns f(X) and df/dX
    - `x` : the initial point
    - `config` : a table of hyper-parameters
    - `config['rho']` : interpolation parameter
    - `config['eps']` : for numerical stability
    - `config['weightDecay']` : weight decay
    - `state` : a table describing the state of the optimizer; after each
            call the state is modified
    - `state['paramVariance']` : vector of temporal variances of parameters
    - `state['accDelta']` : vector of accummulated delta of gradients
    RETURNS:
    - `x` : the new x vector
    - `f(x)` : the value of optimized function, evaluated before the update
    """
    # (0) get/update state
    if config is None and state is None:
        raise ValueError("adadelta requires a dictionary to retain state between iterations")
    state = state if state is not None else config
    rho = config.get('rho', 0.9)
    eps = config.get('eps', 1e-6)
    wd = config.get('weightDecay', 0)
    state['evalCounter'] = state.get('evalCounter', 0)

    # (1) evaluate f(x) and df/dx
    fx, dfdx = opfunc(x)

    # (2) weight decay
    if wd != 0:
        dfdx.add_(wd, x)

    # (3) parameter update
    if 'paramVariance' not in state:
        state['paramVariance'] = x.new().resize_as_(dfdx).zero_()
        state['paramStd'] = x.new().resize_as_(dfdx).zero_()
        state['delta'] = x.new().resize_as_(dfdx).zero_()
        state['accDelta'] = x.new().resize_as_(dfdx).zero_()

    state['paramVariance'].mul_(rho).addcmul_(1 - rho, dfdx, dfdx)
    state['paramStd'].resize_as_(state['paramVariance']).copy_(state['paramVariance']).add_(eps).sqrt_()
    state['delta'].resize_as_(state['paramVariance']).copy_(
        state['accDelta']).add_(eps).sqrt_().div_(state['paramStd']).mul_(dfdx)
    x.add_(-1, state['delta'])
    state['accDelta'].mul_(rho).addcmul_(1 - rho, state['delta'], state['delta'])

    # (4) update evaluation counter
    state['evalCounter'] += 1

    # return x*, f(x) before optimization
    return x, fx
