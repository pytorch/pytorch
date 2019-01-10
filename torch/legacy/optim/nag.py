
def nag(opfunc, x, config, state=None):
    """
    An implementation of SGD adapted with features of Nesterov's
    Accelerated Gradient method, based on the paper
    On the Importance of Initialization and Momentum in Deep Learning
    Sutsveker et. al., ICML 2013

    ARGS:
    opfunc : a function that takes a single input (X), the point of
            evaluation, and returns f(X) and df/dX
    x      : the initial point
    state  : a table describing the state of the optimizer; after each
            call the state is modified
    state['learningRate']      : learning rate
    state['learningRateDecay'] : learning rate decay
    state['weightDecay']       : weight decay
    state['momentum']          : momentum
    state['learningRates']     : vector of individual learning rates

    RETURN:
    x     : the new x vector
    f(x)  : the function, evaluated before the update

    (Dilip Krishnan, 2013)
    """

    # (0) get/update state
    if config is None and state is None:
        raise ValueError("nag requires a dictionary to retain state between iterations")
    state = state if state is not None else config
    lr = config.get('learningRate', 1e-3)
    lrd = config.get('learningRateDecay', 0)
    wd = config.get('weightDecay', 0)
    mom = config.get('momentum', 0.9)
    damp = config.get('dampening', mom)
    lrs = config.get('learningRates', None)
    state['evalCounter'] = state.get('evalCounter', 0)

    if mom <= 0:
        raise ValueError('Momentum must be positive for Nesterov Accelerated Gradient')

    # (1) evaluate f(x) and df/dx
    # first step in the direction of the momentum vector

    if 'dfdx' in state:
        x.add_(mom, state['dfdx'])

    #: compute gradient at that point
    # comment out the above line to get the original SGD
    fx, dfdx = opfunc(x)

    # (2) weight decay
    if wd != 0:
        dfdx.add_(wd, x)

    # (3) learning rate decay (annealing)
    clr = lr / (1 + state['evalCounter'] * lrd)

    # (4) apply momentum
    if 'dfdx' not in state:
        state['dfdx'] = dfdx.new().resize_as_(dfdx).zero_()
    else:
        state['dfdx'].mul_(mom)

    # (5) parameter update with single or individual learning rates
    if lrs is not None:
        if 'deltaParameters' in state:
            state['deltaParameters'] = x.new().resize_as_(dfdx)

        state['deltaParameters'].copy_(lrs).mul_(dfdx)
        x.add_(-clr, state['deltaParameters'])
        state['dfdx'].add_(-clr, state['deltaParameters'])
    else:
        x.add_(-clr, dfdx)
        state['dfdx'].add_(-clr, dfdx)

    # (6) update evaluation counter
    state['evalCounter'] += 1

    # return x, f(x) before optimization
    return x, fx
