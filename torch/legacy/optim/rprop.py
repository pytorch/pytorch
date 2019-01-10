import torch


def rprop(opfunc, x, config, state=None):
    """ A plain implementation of RPROP

    ARGS:
    - `opfunc` : a function that takes a single input (X), the point of
                evaluation, and returns f(X) and df/dX
    - `x`      : the initial point
    - `state`  : a table describing the state of the optimizer; after each
                call the state is modified
    - `state['stepsize']`    : initial step size, common to all components
    - `state['etaplus']`     : multiplicative increase factor, > 1 (default 1.2)
    - `state['etaminus']`    : multiplicative decrease factor, < 1 (default 0.5)
    - `state['stepsizemax']` : maximum stepsize allowed (default 50)
    - `state['stepsizemin']` : minimum stepsize allowed (default 1e-6)
    - `state['niter']`       : number of iterations (default 1)

    RETURN:
    - `x`     : the new x vector
    - `f(x)`  : the function, evaluated before the update

    (Martin Riedmiller, Koray Kavukcuoglu 2013)
    """
    if config is None and state is None:
        raise ValueError("rprop requires a dictionary to retain state between iterations")

    # (0) get/update state
    state = state if state is not None else config
    stepsize = config.get('stepsize', 0.1)
    etaplus = config.get('etaplus', 1.2)
    etaminus = config.get('etaminus', 0.5)
    stepsizemax = config.get('stepsizemax', 50.0)
    stepsizemin = config.get('stepsizemin', 1e-06)
    niter = config.get('niter', 1)

    hfx = []

    for i in range(niter):
        # (1) evaluate f(x) and df/dx
        fx, dfdx = opfunc(x)

        # init temp storage
        if 'delta' not in state:
            state['delta'] = dfdx.new(dfdx.size()).zero_()
            state['stepsize'] = dfdx.new(dfdx.size()).fill_(stepsize)
            state['sign'] = dfdx.new(dfdx.size())
            state['bytesign'] = torch.ByteTensor(dfdx.size())
            state['psign'] = torch.ByteTensor(dfdx.size())
            state['nsign'] = torch.ByteTensor(dfdx.size())
            state['zsign'] = torch.ByteTensor(dfdx.size())
            state['dminmax'] = torch.ByteTensor(dfdx.size())
            if str(type(x)).find('Cuda') > -1:
                # Push to GPU
                state['psign'] = state['psign'].cuda()
                state['nsign'] = state['nsign'].cuda()
                state['zsign'] = state['zsign'].cuda()
                state['dminmax'] = state['dminmax'].cuda()

        # sign of derivative from last step to this one
        torch.mul(dfdx, state['delta'], out=state['sign']).sign_()

        # get indices of >0, <0 and ==0 entries
        torch.gt(state['sign'], 0, out=state['psign'])
        torch.lt(state['sign'], 0, out=state['nsign'])
        torch.eq(state['sign'], 0, out=state['zsign'])

        # get step size updates
        state['sign'][state['psign']] = etaplus
        state['sign'][state['nsign']] = etaminus
        state['sign'][state['zsign']] = 1

        # update stepsizes with step size updates
        state['stepsize'].mul_(state['sign'])

        # threshold step sizes
        # >50 => 50
        torch.gt(state['stepsize'], stepsizemax, out=state['dminmax'])
        state['stepsize'][state['dminmax']] = stepsizemax
        # <1e-6 ==> 1e-6
        torch.lt(state['stepsize'], stepsizemin, out=state['dminmax'])
        state['stepsize'][state['dminmax']] = stepsizemin

        # for dir<0, dfdx=0
        # for dir>=0 dfdx=dfdx
        dfdx[state['nsign']] = 0
        torch.sign(dfdx, out=state['sign'])

        # update weights
        x.addcmul_(-1, state['sign'], state['stepsize'])

        # update state['dfdx'] with current dfdx
        state['delta'].copy_(dfdx)

        hfx.append(fx)

    # return x*, table of f(x) values from each step
    return x, hfx
