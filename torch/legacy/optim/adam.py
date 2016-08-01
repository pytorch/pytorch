import math

def adam(opfunc, x, config, state=None):
    """ An implementation of Adam http://arxiv.org/pdf/1412.6980.pdf

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
    - 'state'                    : a table describing the state of the optimizer; after each
                                call the state is modified

    RETURN:
    - `x`     : the new x vector
    - `f(x)`  : the value of optimized function, evaluated before the update

    """
    # (0) get/update state
    if config is None and state is None:
        raise ValueError("adam requires a dictionary to retain state between iterations")
    state = state or config
    lr = config.get('learningRate', 0.001)
    beta1 = config.get('beta1', 0.9)
    beta2 = config.get('beta2', 0.999)
    epsilon = config.get('epsilon', 1e-8)
    wd = config.get('weightDecay', 0)

    # (1) evaluate f(x) and df/dx
    fx, dfdx = opfunc(x)

    # (2) weight decay
    if wd != 0:
        dfdx.add(wd, x)

    # Initialization
    if not 't' in state:
        state['t'] = 0
        # Exponential moving average of gradient values
        state['m'] = x.new().resizeAs(dfdx).zero()
        # Exponential moving average of squared gradient values
        state['v'] = x.new().resizeAs(dfdx).zero()
        # A tmp tensor to hold the sqrt(v) + epsilon
        state['denom'] = x.new().resizeAs(dfdx).zero()

    state['t'] += 1

    # Decay the first and second moment running average coefficient
    state['m'].mul(beta1).add(1 - beta1, dfdx)
    state['v'].mul(beta2).addcmul(1 - beta2, dfdx, dfdx)

    state['denom'].copy(state['v']).sqrt().add(epsilon)

    biasCorrection1 = 1 - beta1 ** state['t']
    biasCorrection2 = 1 - beta2 ** state['t']
    stepSize = lr * math.sqrt(biasCorrection2)/biasCorrection1
    # (3) update x
    x.addcdiv(-stepSize, state['m'], state['denom'])

    # return x*, f(x) before optimization
    return x, fx
