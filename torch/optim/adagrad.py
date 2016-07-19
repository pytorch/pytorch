
def adagrad(opfunc, x, config, state=None):
    """ADAGRAD implementation

    ARGS:
    - `opfunc` : a function that takes a single input (X), the point of
            evaluation, and returns f(X) and df/dX
    - `x` : the initial point
    - `state` : a table describing the state of the optimizer; after each
            call the state is modified
    - `state['learningRate']` : learning rate
    - `state['paramVariance']` : vector of temporal variances of parameters
    - `state['weightDecay']` : scalar that controls weight decay
    RETURN:
    - `x` : the new x vector
    - `f(x)` : the value of optimized function, evaluated before the update

    """
    # (0) get/update state
    if config is None and state is None:
        raise ValueError("adagrad requires a dictionary to retain state between iterations")
    state = state or config
    lr = config.get('learningRate', 1e-3)
    lrd = config.get('learningRateDecay', 0)
    wd = config.get('weightDecay', 0)
    state['evalCounter'] = state.get('evalCounter', 0)

    # (1) evaluate f(x) and df/dx
    fx, dfdx = opfunc(x)

    # (2) weight decay with a single parameter
    if wd != 0:
        dfdx.add(wd, x)


    # (3) learning rate decay (annealing)
    clr = lr / (1 + state['evalCounter'] * lrd)

    # (4) parameter update with single or individual learning rates
    if not 'paramVariance' in state:
        state['paramVariance'] = x.new().resizeAs(dfdx).zero()
        state['paramStd'] = x.new().resizeAs(dfdx)

    state['paramVariance'].addcmul(1, dfdx, dfdx)
    state['paramStd'].resizeAs(state['paramVariance']).copy(state['paramVariance']).sqrt()
    x.addcdiv(-clr, dfdx, state['paramStd'].add(1e-10))

    # (5) update evaluation counter
    state['evalCounter'] += 1

    # return x*, f(x) before optimization
    return x, fx

