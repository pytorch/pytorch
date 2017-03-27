import math


def asgd(opfunc, x, config, state=None):
    """ An implementation of ASGD

    ASGD:

        x := (1 - lambda eta_t) x - eta_t df/dx(z,x)
        a := a + mu_t [ x - a ]

        eta_t = eta0 / (1 + lambda eta0 t) ^ 0.75
        mu_t = 1/max(1,t-t0)

    implements ASGD algoritm as in L.Bottou's sgd-2.0

    ARGS:

    - `opfunc` : a function that takes a single input (X), the point of
            evaluation, and returns f(X) and df/dX
    - `x`      : the initial point
    - `state`  : a table describing the state of the optimizer; after each
            call the state is modified
    - `state['eta0']`   : learning rate
    - `state['lambda']` : decay term
    - `state['alpha']`  : power for eta update
    - `state['t0']`     : point at which to start averaging

    RETURN:
    - `x`     : the new x vector
    - `f(x)`  : the function, evaluated before the update
    - `ax`    : the averaged x vector

    (Clement Farabet, 2012)
    """
    # (0) get/update state
    if config is None and state is None:
        raise ValueError("asgd requires a dictionary to retain state between iterations")
    state = state if state is not None else config
    config['eta0'] = config.get('eta0', 1e-4)
    config['lambda'] = config.get('lambda', 1e-4)
    config['alpha'] = config.get('alpha', 0.75)
    config['t0'] = config.get('t0', 1e6)

    # (hidden state)
    state['eta_t'] = state.get('eta_t', config['eta0'])
    state['mu_t'] = state.get('mu_t', 1)
    state['t'] = state.get('t', 0)

    # (1) evaluate f(x) and df/dx
    fx, dfdx = opfunc(x)

    # (2) decay term
    x.mul_(1 - config['lambda'] * state['eta_t'])

    # (3) update x
    x.add_(-state['eta_t'], dfdx)

    # (4) averaging
    state['ax'] = state.get('ax', x.new().resize_as_(x).zero_())
    state['tmp'] = state.get('tmp', state['ax'].new().resize_as_(state['ax']))
    if state['mu_t'] != 1:
        state['tmp'].copy_(x)
        state['tmp'].add_(-1, state['ax']).mul_(state['mu_t'])
        state['ax'].add_(state['tmp'])
    else:
        state['ax'].copy_(x)

    # (5) update eta_t and mu_t
    state['t'] += 1
    state['eta_t'] = config['eta0'] / math.pow((1 + config['lambda'] * config['eta0'] * state['t']), config['alpha'])
    state['mu_t'] = 1 / max(1, state['t'] - config['t0'])

    # return x*, f(x) before optimization, and average(x_t0,x_t1,x_t2,...)
    return x, fx, state['ax']
