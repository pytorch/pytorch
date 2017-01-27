import torch


def lbfgs(opfunc, x, config, state=None):
    """
    An implementation of L-BFGS, heavily inspired by minFunc (Mark Schmidt)
    This implementation of L-BFGS relies on a user-provided line
    search function (state.lineSearch). If this function is not
    provided, then a simple learningRate is used to produce fixed
    size steps. Fixed size steps are much less costly than line
    searches, and can be useful for stochastic problems.
    The learning rate is used even when a line search is provided.
    This is also useful for large-scale stochastic problems, where
    opfunc is a noisy approximation of f(x). In that case, the learning
    rate allows a reduction of confidence in the step size.

    Args:
    - `opfunc` : a function that takes a single input (X), the point of
             evaluation, and returns f(X) and df/dX
    - `x` : the initial point
    - `state` : a table describing the state of the optimizer; after each
             call the state is modified
    - `state.maxIter` : Maximum number of iterations allowed
    - `state.maxEval` : Maximum number of function evaluations
    - `state.tolFun` : Termination tolerance on the first-order optimality
    - `state.tolX` : Termination tol on progress in terms of func/param changes
    - `state.lineSearch` : A line search function
    - `state.learningRate` : If no line search provided, then a fixed step size is used

    Returns:
    - `x*` : the new `x` vector, at the optimal point
    - `f`  : a table of all function values:
         `f[1]` is the value of the function before any optimization and
         `f[#f]` is the final fully optimized value, at `x*`

    (Clement Farabet, 2012)
    """

    # (0) get/update state
    if config is None and state is None:
        raise ValueError("lbfgs requires a dictionary to retain state between iterations")
    state = state if state is not None else config
    maxIter = config.get('maxIter', 20)
    maxEval = config.get('maxEval', maxIter * 1.25)
    tolFun = config.get('tolFun', 1e-5)
    tolX = config.get('tolX', 1e-9)
    nCorrection = config.get('nCorrection', 100)
    lineSearch = config.get('lineSearch')
    lineSearchOptions = config.get('lineSearchOptions')
    learningRate = config.get('learningRate', 1)
    isverbose = config.get('verbose', False)

    state.setdefault('funcEval', 0)
    state.setdefault('nIter', 0)

    # verbose function
    if isverbose:
        def verbose(*args):
            args = ('<optim.lbfgs>',) + args
            print(args)
    else:
        def verbose(*args):
            pass

    # evaluate initial f(x) and df/dx
    f, g = opfunc(x)
    f_hist = [f]
    currentFuncEval = 1
    state['funcEval'] += 1
    p = g.size(0)

    # check optimality of initial point
    if 'tmp1' not in state:
        state['tmp1'] = g.new(g.size()).zero_()
    tmp1 = state['tmp1']
    tmp1.copy_(g).abs_()
    if tmp1.sum() <= tolFun:
        verbose('optimality condition below tolFun')
        return x, f_hist

    if 'dir_bufs' not in state:
        # reusable buffers for y's and s's, and their histories
        verbose('creating recyclable direction/step/history buffers')
        state['dir_bufs'] = list(g.new(nCorrection + 1, p).split(1))
        state['stp_bufs'] = list(g.new(nCorrection + 1, p).split(1))
        for i in range(len(state['dir_bufs'])):
            state['dir_bufs'][i] = state['dir_bufs'][i].squeeze(0)
            state['stp_bufs'][i] = state['stp_bufs'][i].squeeze(0)

    # variables cached in state (for tracing)
    d = state.get('d')
    t = state.get('t')
    old_dirs = state.get('old_dirs')
    old_stps = state.get('old_stps')
    Hdiag = state.get('Hdiag')
    g_old = state.get('g_old')
    f_old = state.get('f_old')

    # optimize for a max of maxIter iterations
    nIter = 0
    while nIter < maxIter:
        # keep track of nb of iterations
        nIter += 1
        state['nIter'] += 1

        ############################################################
        # compute gradient descent direction
        ############################################################
        if state['nIter'] == 1:
            d = g.neg()
            old_dirs = []
            old_stps = []
            Hdiag = 1
        else:
            # do lbfgs update (update memory)
            y = state['dir_bufs'].pop()
            s = state['stp_bufs'].pop()
            torch.add(g, -1, g_old, out=y)
            torch.mul(d, t, out=s)
            ys = y.dot(s)  # y*s
            if ys > 1e-10:
                # updating memory
                if len(old_dirs) == nCorrection:
                    # shift history by one (limited-memory)
                    state['dir_bufs'].append(old_dirs.pop(0))
                    state['stp_bufs'].append(old_stps.pop(0))

                # store new direction/step
                old_dirs.append(s)
                old_stps.append(y)

                # update scale of initial Hessian approximation
                Hdiag = ys / y.dot(y)  # (y*y)
            else:
                # put y and s back into the buffer pool
                state['dir_bufs'].append(y)
                state['stp_bufs'].append(s)

            # compute the approximate (L-BFGS) inverse Hessian
            # multiplied by the gradient
            k = len(old_dirs)

            # need to be accessed element-by-element, so don't re-type tensor:
            if 'ro' not in state:
                state['ro'] = torch.Tensor(nCorrection)
            ro = state['ro']

            for i in range(k):
                ro[i] = 1 / old_stps[i].dot(old_dirs[i])

            # iteration in L-BFGS loop collapsed to use just one buffer
            q = tmp1  # reuse tmp1 for the q buffer
            # need to be accessed element-by-element, so don't re-type tensor:
            if 'al' not in state:
                state['al'] = torch.zeros(nCorrection)
            al = state['al']

            torch.mul(g, -1, out=q)
            for i in range(k - 1, -1, -1):
                al[i] = old_dirs[i].dot(q) * ro[i]
                q.add_(-al[i], old_stps[i])

            # multiply by initial Hessian
            r = d  # share the same buffer, since we don't need the old d
            torch.mul(q, Hdiag, out=r)
            for i in range(k):
                be_i = old_stps[i].dot(r) * ro[i]
                r.add_(al[i] - be_i, old_dirs[i])
            # final direction is in r/d (same object)
        if g_old is None:
            g_old = g.clone()
        else:
            g_old.copy_(g)
        f_old = f

        ############################################################
        # compute step length
        ############################################################
        # directional derivative
        gtd = g.dot(d)  # g * d

        # check that progress can be made along that direction
        if gtd > -tolX:
            break

        # reset initial guess for step size
        if state['nIter'] == 1:
            tmp1.copy_(g).abs_()
            t = min(1, 1 / tmp1.sum()) * learningRate
        else:
            t = learningRate

        # optional line search: user function
        lsFuncEval = 0
        if lineSearch is not None:
            # perform line search, using user function
            f, g, x, t, lsFuncEval = lineSearch(opfunc, x, t, d, f, g, gtd, lineSearchOpts)
            f_hist.append(f)
        else:
            # no line search, simply move with fixed-step
            x.add_(t, d)
            if nIter != maxIter:
                # re-evaluate function only if not in last iteration
                # the reason we do this: in a stochastic setting,
                # no use to re-evaluate that function here
                f, g = opfunc(x)
                lsFuncEval = 1
                f_hist.append(f)

        # update func eval
        currentFuncEval += lsFuncEval
        state['funcEval'] += lsFuncEval

        ############################################################
        # check conditions
        ############################################################
        if nIter == maxIter:
            # no use to run tests
            verbose('reached max number of iterations')
            break

        if currentFuncEval >= maxEval:
            # max nb of function evals
            verbose('max nb of function evals')
            break

        tmp1.copy_(g).abs_()
        if tmp1.sum() <= tolFun:
            # check optimality
            verbose('optimality condition below tolFun')
            break

        tmp1.copy_(d).mul_(t).abs_()
        if tmp1.sum() <= tolX:
            # step size below tolX
            verbose('step size below tolX')
            break

        if abs(f - f_old) < tolX:
            # function value changing less than tolX
            verbose('function value changing less than tolX')
            break

    # save state
    state['old_dirs'] = old_dirs
    state['old_stps'] = old_stps
    state['Hdiag'] = Hdiag
    state['g_old'] = g_old
    state['f_old'] = f_old
    state['t'] = t
    state['d'] = d

    # return optimal x, and history of f(x)
    return x, f_hist, currentFuncEval
