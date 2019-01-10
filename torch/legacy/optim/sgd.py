import torch


def sgd(opfunc, x, config, state=None):
    """A plain implementation of SGD

    ARGS:

    - `opfunc` : a function that takes a single input (X), the point
                of a evaluation, and returns f(X) and df/dX
    - `x`      : the initial point
    - `config` : a table with configuration parameters for the optimizer
    - `config['learningRate']`      : learning rate
    - `config['learningRateDecay']` : learning rate decay
    - `config['weightDecay']`       : weight decay
    - `config['weightDecays']`      : vector of individual weight decays
    - `config['momentum']`          : momentum
    - `config['dampening']`         : dampening for momentum
    - `config['nesterov']`          : enables Nesterov momentum
    - `config['learningRates']`     : vector of individual learning rates
    - `state`  : a table describing the state of the optimizer; after each
                call the state is modified
    - `state['evalCounter']`        : evaluation counter (optional: 0, by default)

    RETURN:
    - `x`     : the new x vector
    - `f(x)`  : the function, evaluated before the update

    (Clement Farabet, 2012)
    """
    # (0) get/update state
    state = state if state is not None else config
    lr = config.get('learningRate', 1e-3)
    lrd = config.get('learningRateDecay', 0)
    wd = config.get('weightDecay', 0)
    mom = config.get('momentum', 0)
    damp = config.get('dampening', mom)
    nesterov = config.get('nesterov', False)
    lrs = config.get('learningRates', None)
    wds = config.get('weightDecays', None)
    if 'evalCounter' not in state:
        state['evalCounter'] = 0
    if nesterov and (mom <= 0 and damp != 0):
        raise ValueError("Nesterov momentum requires a momentum and zero dampening")
    if wd != 0 and wds is not None:
        raise ValueError("Only one of wd and wds can be specified")

    # (1) evaluate f(x) and df/dx
    fx, dfdx = opfunc(x)

    # (2) weight decay with single or individual parameters
    if wd != 0:
        dfdx.add_(wd, x)
    elif wds is not None:
        if not state['decayParameters']:
            state['decayParameters'] = torch.Tensor().type_as(x).resize_as_(dfdx)

        state['decayParameters'].copy_(wds).mul_(x)
        dfdx.add_(state['decayParameters'])

    # (3) apply momentum
    if mom != 0:
        if 'dfdx' not in state:
            state['dfdx'] = torch.Tensor().type_as(dfdx).resize_as_(dfdx).copy_(dfdx)
        else:
            state['dfdx'].mul_(mom).add_(1 - damp, dfdx)

        if nesterov:
            dfdx.add_(mom, state['dfdx'])
        else:
            dfdx = state['dfdx']

    # (4) learning rate decay (annealing)
    clr = lr / (1 + state['evalCounter'] * lrd)

    # (5) parameter update with single or individual learning rates
    if lrs is not None:
        if 'deltaParameters' not in state:
            state['deltaParameters'] = torch.Tensor().type_as(x).resize_as_(dfdx)

        state['deltaParameters'].copy_(lrs).mul_(dfdx)
        x.add_(-clr, state['deltaParameters'])
    else:
        x.add_(-clr, dfdx)

    # (6) update evaluation counter
    state['evalCounter'] += 1

    # return x*, f(x) before optimization
    return x, fx
