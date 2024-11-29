# mypy: allow-untyped-defs
from typing import Optional, Union

import torch
from torch import Tensor

from .optimizer import Optimizer, ParamsT


__all__ = ["BFGS"]


def _cubic_interpolate(x1, f1, g1, x2, f2, g2, bounds=None):
    # ported from https://github.com/torch/optim/blob/master/polyinterp.lua
    # Compute bounds of interpolation area
    if bounds is not None:
        xmin_bound, xmax_bound = bounds
    else:
        xmin_bound, xmax_bound = (x1, x2) if x1 <= x2 else (x2, x1)

    # Code for most common case: cubic interpolation of 2 points
    #   w/ function and derivative values for both
    # Solution in this case (where x2 is the farthest point):
    #   d1 = g1 + g2 - 3*(f1-f2)/(x1-x2);
    #   d2 = sqrt(d1^2 - g1*g2);
    #   min_pos = x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2));
    #   t_new = min(max(min_pos,xmin_bound),xmax_bound);
    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    d2_square = d1**2 - g1 * g2
    if d2_square >= 0:
        d2 = d2_square.sqrt()
        if x1 <= x2:
            min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
        else:
            min_pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
        return min(max(min_pos, xmin_bound), xmax_bound)
    else:
        return (xmin_bound + xmax_bound) / 2.0


def _hager_zhang(
    obj_func, x, t, d, f, g, gtd, c1=1e-4, c2=0.9, tolerance_change=1e-9, max_ls=25
):
    """
    Hager-Zhang line search algorithm for nonmonotone line search with strong Wolfe conditions

    Algorithm parameters from paper https://www.math.lsu.edu/~hozhang/papers/cg_descent.pdf

    Inspired by the _strong_wolfe function.

    Parameters:
    -----------
    obj_func : callable
        Function that returns (f_new, g_new) given (x, t, d)
        - f_new is the new function value
        - g_new is the new gradient
    x : tensor
        Current point in the optimization space
    t : float 
        Initial step length (alpha in the paper)
    d : tensor
        Search direction (should be a descent direction)
    f : float
        Initial function value f(x)
    g : tensor
        Initial gradient ∇f(x)
    gtd : float
        Initial directional derivative g·d (should be negative for descent)
    c1 : float
        Sufficient decrease constant for Wolfe conditions (default: 1e-4)
    c2 : float
        Curvature condition constant for Wolfe conditions (default: 0.9) 
    tolerance_change : float
        Minimum change in t to continue search (default: 1e-9)
    max_ls : int
        Maximum number of line search steps (default: 25)
        
    Returns:
    --------
    f_new : float
        Final function value at the accepted point
    g_new : tensor 
        Final gradient at the accepted point
    t : float
        Final accepted step length
    n_evals : int 
        Number of function evaluations performed
    """

    # Constants for the algorithm
    delta = c1          # Sufficient decrease parameter (same as input c1)
    sigma = c2         # Curvature condition parameter (same as input c2)
    epsilon = 1e-6     # Error tolerance for approximate Wolfe conditions
    
    # Get the maximum absolute value of the search direction for scaling
    d_norm = d.abs().max()
    # Ensure gradient is contiguous in memory for efficient operations
    g = g.clone(memory_format=torch.contiguous_format)
    
    # Initial function and gradient evaluation at the trial point
    f_new, g_new = obj_func(x, t, d)
    ls_func_evals = 1
    gtd_new = g_new.dot(d)  # New directional derivative
    
    # Initialize bracketing phase variables
    t_prev, f_prev = 0.0, f  # Previous step length and function value
    g_prev = g              # Previous gradient
    gtd_prev = gtd          # Previous directional derivative
    done = False            # Flag for successful step length found
    ls_iter = 0            # Line search iteration counter
    
    # Bracketing phase: Find an interval containing acceptable step lengths
    while ls_iter < max_ls:
        # Check standard Wolfe conditions:
        # 1. Sufficient decrease condition (Armijo condition)
        wolfe1 = f_new <= f + delta * t * gtd
        # 2. Curvature condition
        wolfe2 = abs(gtd_new) <= -sigma * gtd
        
        # Check approximate Wolfe conditions (useful when exact ones are hard to satisfy):
        # 1. Approximate sufficient decrease
        approx_wolfe1 = (2*delta - 1) * gtd >= gtd_new
        # 2. Approximate curvature condition
        approx_wolfe2 = gtd_new >= sigma * gtd
        
        # If either set of conditions is satisfied, we're done
        if (wolfe1 and wolfe2) or (approx_wolfe1 and approx_wolfe2 and f_new <= f + epsilon):
            bracket = [t]
            bracket_f = [f_new]
            bracket_g = [g_new]
            bracket_gtd = [gtd_new]
            done = True
            break
            
        # If we've gone too far (function value too high) or non-monotonic behavior
        if f_new > f + delta * t * gtd or (ls_iter > 1 and f_new >= f_prev):
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break
            
        # If curvature condition is satisfied (but not sufficient decrease)
        if abs(gtd_new) <= -sigma * gtd:
            bracket = [t]
            bracket_f = [f_new]
            bracket_g = [g_new]
            bracket_gtd = [gtd_new]
            done = True
            break
            
        # If the directional derivative becomes positive, we've bracketed a minimum
        if gtd_new >= 0:
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        # Use cubic interpolation to choose next trial value
        # Ensure the step length increases by at least 1% and at most 10x
        min_step = t + 0.01 * (t - t_prev)  # Minimum step length
        max_step = t * 10                   # Maximum step length
        tmp = t
        t = _cubic_interpolate(
            t_prev, f_prev, gtd_prev,
            t, f_new, gtd_new,
            bounds=(min_step, max_step)
        )

        # Store current values as previous for next iteration
        t_prev = tmp
        f_prev = f_new
        g_prev = g_new.clone(memory_format=torch.contiguous_format)
        gtd_prev = gtd_new
        
        # Evaluate function at new trial point
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

    # If bracketing phase failed, use the initial and final points
    if ls_iter == max_ls:
        bracket = [0, t]
        bracket_f = [f, f_new]
        bracket_g = [g, g_new]
        bracket_gtd = [gtd, gtd_new]

    # Zoom phase: Narrow the bracket until we find an acceptable point
    insuf_progress = False
    if not done and ls_iter < max_ls:
        # Identify the low (better) and high (worse) points in the bracket
        low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[-1] else (1, 0)
        
        while not done and ls_iter < max_ls:
            # Check if bracket is too small (convergence check)
            if abs(bracket[1] - bracket[0]) * d_norm < tolerance_change:
                break

            # Use cubic interpolation to choose next trial value
            t = _cubic_interpolate(
                bracket[0], bracket_f[0], bracket_gtd[0],
                bracket[1], bracket_f[1], bracket_gtd[1]
            )

            # Handle boundary cases to prevent the algorithm from stalling
            eps = 0.1 * (max(bracket) - min(bracket))
            if min(max(bracket) - t, t - min(bracket)) < eps:
                if insuf_progress or t >= max(bracket) or t <= min(bracket):
                    # If too close to bounds, move away from closest bound
                    if abs(t - max(bracket)) < abs(t - min(bracket)):
                        t = max(bracket) - eps
                    else:
                        t = min(bracket) + eps
                    insuf_progress = False
                else:
                    insuf_progress = True
            else:
                insuf_progress = False

            # Evaluate function at new trial point
            f_new, g_new = obj_func(x, t, d)
            ls_func_evals += 1
            gtd_new = g_new.dot(d)
            ls_iter += 1

            # Update the bracket based on the new point
            if f_new > f + delta * t * gtd or f_new >= bracket_f[low_pos]:
                # New point becomes the high point
                bracket[high_pos] = t
                bracket_f[high_pos] = f_new
                bracket_g[high_pos] = g_new.clone(memory_format=torch.contiguous_format)
                bracket_gtd[high_pos] = gtd_new
                low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)
            else:
                if abs(gtd_new) <= -sigma * gtd:
                    done = True
                elif gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0:
                    # Update high point to match low point if derivatives suggest
                    # minimum is closer to low point
                    bracket[high_pos] = bracket[low_pos]
                    bracket_f[high_pos] = bracket_f[low_pos]
                    bracket_g[high_pos] = bracket_g[low_pos]
                    bracket_gtd[high_pos] = bracket_gtd[low_pos]

                # New point becomes the low point
                bracket[low_pos] = t
                bracket_f[low_pos] = f_new
                bracket_g[low_pos] = g_new.clone(memory_format=torch.contiguous_format)
                bracket_gtd[low_pos] = gtd_new

    # Return the best point found (usually the low point)
    t = bracket[low_pos if len(bracket) > 1 else 0]
    f_new = bracket_f[low_pos if len(bracket) > 1 else 0]
    g_new = bracket_g[low_pos if len(bracket) > 1 else 0]
    return f_new, g_new, t, ls_func_evals


def _strong_wolfe(
    obj_func, x, t, d, f, g, gtd, c1=1e-4, c2=0.9, tolerance_change=1e-9, max_ls=25
):
    # ported from https://github.com/torch/optim/blob/master/lswolfe.lua
    d_norm = d.abs().max()
    g = g.clone(memory_format=torch.contiguous_format)
    # evaluate objective and gradient using initial step
    f_new, g_new = obj_func(x, t, d)
    ls_func_evals = 1
    gtd_new = g_new.dot(d)

    # bracket an interval containing a point satisfying the Wolfe criteria
    t_prev, f_prev, g_prev, gtd_prev = 0, f, g, gtd
    done = False
    ls_iter = 0
    while ls_iter < max_ls:
        # check conditions
        if f_new > (f + c1 * t * gtd) or (ls_iter > 1 and f_new >= f_prev):
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        if abs(gtd_new) <= -c2 * gtd:
            bracket = [t]
            bracket_f = [f_new]
            bracket_g = [g_new]
            done = True
            break

        if gtd_new >= 0:
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        # interpolate
        min_step = t + 0.01 * (t - t_prev)
        max_step = t * 10
        tmp = t
        t = _cubic_interpolate(
            t_prev, f_prev, gtd_prev, t, f_new, gtd_new, bounds=(min_step, max_step)
        )

        # next step
        t_prev = tmp
        f_prev = f_new
        g_prev = g_new.clone(memory_format=torch.contiguous_format)
        gtd_prev = gtd_new
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

    # reached max number of iterations?
    if ls_iter == max_ls:
        bracket = [0, t]
        bracket_f = [f, f_new]
        bracket_g = [g, g_new]

    # zoom phase: we now have a point satisfying the criteria, or
    # a bracket around it. We refine the bracket until we find the
    # exact point satisfying the criteria
    insuf_progress = False
    # find high and low points in bracket
    low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[-1] else (1, 0)  # type: ignore[possibly-undefined]
    while not done and ls_iter < max_ls:
        # line-search bracket is so small
        if abs(bracket[1] - bracket[0]) * d_norm < tolerance_change:  # type: ignore[possibly-undefined]
            break

        # compute new trial value
        t = _cubic_interpolate(
            bracket[0],
            bracket_f[0],
            bracket_gtd[0],  # type: ignore[possibly-undefined]
            bracket[1],
            bracket_f[1],
            bracket_gtd[1],
        )

        # test that we are making sufficient progress:
        # in case `t` is so close to boundary, we mark that we are making
        # insufficient progress, and if
        #   + we have made insufficient progress in the last step, or
        #   + `t` is at one of the boundary,
        # we will move `t` to a position which is `0.1 * len(bracket)`
        # away from the nearest boundary point.
        eps = 0.1 * (max(bracket) - min(bracket))
        if min(max(bracket) - t, t - min(bracket)) < eps:
            # interpolation close to boundary
            if insuf_progress or t >= max(bracket) or t <= min(bracket):
                # evaluate at 0.1 away from boundary
                if abs(t - max(bracket)) < abs(t - min(bracket)):
                    t = max(bracket) - eps
                else:
                    t = min(bracket) + eps
                insuf_progress = False
            else:
                insuf_progress = True
        else:
            insuf_progress = False

        # Evaluate new point
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

        if f_new > (f + c1 * t * gtd) or f_new >= bracket_f[low_pos]:
            # Armijo condition not satisfied or not lower than lowest point
            bracket[high_pos] = t
            bracket_f[high_pos] = f_new
            bracket_g[high_pos] = g_new.clone(memory_format=torch.contiguous_format)  # type: ignore[possibly-undefined]
            bracket_gtd[high_pos] = gtd_new
            low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)
        else:
            if abs(gtd_new) <= -c2 * gtd:
                # Wolfe conditions satisfied
                done = True
            elif gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0:
                # old high becomes new low
                bracket[high_pos] = bracket[low_pos]
                bracket_f[high_pos] = bracket_f[low_pos]
                bracket_g[high_pos] = bracket_g[low_pos]  # type: ignore[possibly-undefined]
                bracket_gtd[high_pos] = bracket_gtd[low_pos]

            # new point becomes new low
            bracket[low_pos] = t
            bracket_f[low_pos] = f_new
            bracket_g[low_pos] = g_new.clone(memory_format=torch.contiguous_format)  # type: ignore[possibly-undefined]
            bracket_gtd[low_pos] = gtd_new

    # return stuff
    t = bracket[low_pos]  # type: ignore[possibly-undefined]
    f_new = bracket_f[low_pos]
    g_new = bracket_g[low_pos]  # type: ignore[possibly-undefined]
    return f_new, g_new, t, ls_func_evals


class BFGS(Optimizer):
    """Implements BFGS algorithm.

    Heavily inspired from the pytorch implementation of L-BFGS.
    
    Note: This version stores the full inverse Hessian approximation,
    unlike L-BFGS which uses limited memory.

    .. warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).

    .. warning::
        Right now all parameters have to be on a single device. This will be
        improved in the future.

    Args:
        params (iterable): iterable of parameters to optimize. Parameters must be real.
        lr (float): learning rate (default: 1)
        max_iter (int): maximal number of iterations per optimization step
            (default: 20)
        max_eval (int): maximal number of function evaluations per optimization
            step (default: max_iter * 1.25).
        tolerance_grad (float): termination tolerance on first order optimality
            (default: 1e-7).
        tolerance_change (float): termination tolerance on function
            value/parameter changes (default: 1e-9).
        line_search_fn (str): either 'strong_wolfe', 'hager_zhang' or None (default: None).
    """

    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1,
        max_iter: int = 20,
        max_eval: Optional[int] = None,
        tolerance_grad: float = 1e-7,
        tolerance_change: float = 1e-9,
        line_search_fn: Optional[str] = None,
    ):
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        defaults = dict(
            lr=lr,
            max_iter=max_iter,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            line_search_fn=line_search_fn,
        )
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError(
                "BFGS doesn't support per-parameter options (parameter groups)"
            )

        self._params = self.param_groups[0]["params"]
        self._numel_cache = None

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = sum(
                2 * p.numel() if torch.is_complex(p) else p.numel()
                for p in self._params
            )
        return self._numel_cache

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            if torch.is_complex(view):
                view = torch.view_as_real(view).view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            if torch.is_complex(p):
                p = torch.view_as_real(p)
            numel = p.numel()
            p.add_(update[offset : offset + numel].view_as(p), alpha=step_size)
            offset += numel
        assert offset == self._numel()

    def _clone_param(self):
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            p.copy_(pdata)

    def _directional_evaluate(self, closure, x, t, d):
        self._add_grad(t, d)
        loss = float(closure())
        flat_grad = self._gather_flat_grad()
        self._set_param(x)
        return loss, flat_grad

    @torch.no_grad()
    def step(self, closure):
        """Perform a single optimization step.

        Args:
            closure (Callable): A closure that reevaluates the model
                and returns the loss.
        """
        assert len(self.param_groups) == 1

        # Make sure the closure is always called with grad enabled
        closure = torch.enable_grad()(closure)

        group = self.param_groups[0]
        lr = group["lr"]
        max_iter = group["max_iter"]
        max_eval = group["max_eval"]
        tolerance_grad = group["tolerance_grad"]
        tolerance_change = group["tolerance_change"]
        line_search_fn = group["line_search_fn"]

        # NOTE: BFGS has only global state, but we register it as state for
        # the first param, because this helps with casting in load_state_dict
        state = self.state[self._params[0]]
        state.setdefault("func_evals", 0)
        state.setdefault("n_iter", 0)

        # evaluate initial f(x) and df/dx
        orig_loss = closure()
        loss = float(orig_loss)
        current_evals = 1
        state["func_evals"] += 1

        flat_grad = self._gather_flat_grad()
        opt_cond = flat_grad.abs().max() <= tolerance_grad

        # optimal condition
        if opt_cond:
            return orig_loss

        # tensors cached in state (for tracing)
        d = state.get("d")
        t = state.get("t")
        H = state.get("H")  # full inverse Hessian matrix
        prev_flat_grad = state.get("prev_flat_grad")
        prev_loss = state.get("prev_loss")

        n_iter = 0
        # optimize for a max of max_iter iterations
        while n_iter < max_iter:
            # keep track of nb of iterations
            n_iter += 1
            state["n_iter"] += 1

            ############################################################
            # compute gradient descent direction
            ############################################################
            if state["n_iter"] == 1:
                d = flat_grad.neg()
                H = torch.eye(flat_grad.size(0), device=flat_grad.device)
            else:
                # do BFGS update (update inverse Hessian)
                y = flat_grad.sub(prev_flat_grad)
                s = d.mul(t)
                ys = y.dot(s)  # y*s
                
                if ys > 1e-10:
                    # BFGS update formula for inverse Hessian
                    Hy = H @ y
                    yHy = y.dot(Hy)
                    
                    # Compute the outer products
                    ss_term = torch.outer(s, s) / ys
                    yH_term = torch.outer(Hy, s) + torch.outer(s, Hy)
                    yHy_term = (yHy + ys) * torch.outer(s, s) / (ys * ys)
                    
                    # Update H using the BFGS formula
                    H.add_(ss_term)
                    H.add_(-yH_term / ys)
                    H.add_(yHy_term)

                # compute search direction using the inverse Hessian
                d = -H @ flat_grad

            if prev_flat_grad is None:
                prev_flat_grad = flat_grad.clone(memory_format=torch.contiguous_format)
            else:
                prev_flat_grad.copy_(flat_grad)
            prev_loss = loss

            ############################################################
            # compute step length
            ############################################################
            # reset initial guess for step size
            if state["n_iter"] == 1:
                t = min(1.0, 1.0 / flat_grad.abs().sum()) * lr
            else:
                t = lr

            # directional derivative
            gtd = flat_grad.dot(d)  # g * d

            # directional derivative is below tolerance
            if gtd > -tolerance_change:
                break

            # optional line search: user function
            ls_func_evals = 0
            if line_search_fn is not None:
                # perform line search, using user function
                if line_search_fn == "strong_wolfe":
                    x_init = self._clone_param()

                    def obj_func(x, t, d):
                        return self._directional_evaluate(closure, x, t, d)

                    loss, flat_grad, t, ls_func_evals = _strong_wolfe(
                        obj_func, x_init, t, d, loss, flat_grad, gtd
                    )
                elif line_search_fn == "hager_zhang":
                    x_init = self._clone_param()

                    def obj_func(x, t, d):
                        return self._directional_evaluate(closure, x, t, d)

                    loss, flat_grad, t, ls_func_evals = _hager_zhang(
                        obj_func, x_init, t, d, loss, flat_grad, gtd
                    )
                else:
                    raise RuntimeError("only 'strong_wolfe' and 'hager_zhang' are supported")

                self._add_grad(t, d)
                opt_cond = flat_grad.abs().max() <= tolerance_grad
            else:
                # no line search, simply move with fixed-step
                self._add_grad(t, d)
                if n_iter != max_iter:
                    # re-evaluate function only if not in last iteration
                    # the reason we do this: in a stochastic setting,
                    # no use to re-evaluate that function here
                    with torch.enable_grad():
                        loss = float(closure())
                    flat_grad = self._gather_flat_grad()
                    opt_cond = flat_grad.abs().max() <= tolerance_grad
                    ls_func_evals = 1

            # update func eval
            current_evals += ls_func_evals
            state["func_evals"] += ls_func_evals

            ############################################################
            # check conditions
            ############################################################
            if n_iter == max_iter:
                break

            if current_evals >= max_eval:
                break

            # optimal condition
            if opt_cond:
                break

            # lack of progress
            if d.mul(t).abs().max() <= tolerance_change:
                break

            if abs(loss - prev_loss) < tolerance_change:
                break

        state["d"] = d
        state["t"] = t
        state["H"] = H
        state["prev_flat_grad"] = prev_flat_grad
        state["prev_loss"] = prev_loss

        return orig_loss
