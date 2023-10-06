"""
=============================
Jacobians, hessians, and more
=============================

Computing jacobians or hessians are useful in a number of non-traditional
deep learning models. It is difficult (or annoying) to compute these quantities
efficiently using a standard autodiff system like PyTorch Autograd; functorch
provides ways of computing various higher-order autodiff quantities efficiently.
"""
from functools import partial

import torch
import torch.nn.functional as F

torch.manual_seed(0)


######################################################################
# Setup: Comparing functorch vs the naive approach
# --------------------------------------------------------------------
# Let's start with a function that we'd like to compute the jacobian of.
# This is a simple linear function with non-linear activation.
def predict(weight, bias, x):
    return F.linear(x, weight, bias).tanh()


# Here's some dummy data: a weight, a bias, and a feature vector.
D = 16
weight = torch.randn(D, D)
bias = torch.randn(D)
x = torch.randn(D)

# Let's think of ``predict`` as a function that maps the input ``x`` from R^D -> R^D.
# PyTorch Autograd computes vector-Jacobian products. In order to compute the full
# Jacobian of this R^D -> R^D function, we would have to compute it row-by-row
# by using a different unit vector each time.
xp = x.clone().requires_grad_()
unit_vectors = torch.eye(D)


def compute_jac(xp):
    jacobian_rows = [
        torch.autograd.grad(predict(weight, bias, xp), xp, vec)[0]
        for vec in unit_vectors
    ]
    return torch.stack(jacobian_rows)


jacobian = compute_jac(xp)

# Instead of computing the jacobian row-by-row, we can use ``vmap`` to get rid
# of the for-loop and vectorize the computation. We can't directly apply vmap
# to PyTorch Autograd; instead, functorch provides a ``vjp`` transform:
from functorch import vjp, vmap

_, vjp_fn = vjp(partial(predict, weight, bias), x)
(ft_jacobian,) = vmap(vjp_fn)(unit_vectors)
assert torch.allclose(ft_jacobian, jacobian)

# In another tutorial a composition of reverse-mode AD and vmap gave us
# per-sample-gradients. In this tutorial, composing reverse-mode AD and vmap
# gives us Jacobian computation! Various compositions of vmap and autodiff
# transforms can give us different interesting quantities.
#
# functorch provides ``jacrev`` as a convenience function that performs
# the vmap-vjp composition to compute jacobians. ``jacrev`` accepts an argnums
# argument that says which argument we would like to compute Jacobians with
# respect to.
from functorch import jacrev

ft_jacobian = jacrev(predict, argnums=2)(weight, bias, x)
assert torch.allclose(ft_jacobian, jacobian)

# Let's compare the performance of the two ways to compute jacobian.
# The functorch version is much faster (and becomes even faster the more outputs
# there are). In general, we expect that vectorization via ``vmap`` can help
# eliminate overhead and give better utilization of your hardware.
from torch.utils.benchmark import Timer

without_vmap = Timer(stmt="compute_jac(xp)", globals=globals())
with_vmap = Timer(stmt="jacrev(predict, argnums=2)(weight, bias, x)", globals=globals())
print(without_vmap.timeit(500))
print(with_vmap.timeit(500))

# It's pretty easy to flip the problem around and say we want to compute
# Jacobians of the parameters to our model (weight, bias) instead of the input.
ft_jac_weight, ft_jac_bias = jacrev(predict, argnums=(0, 1))(weight, bias, x)

######################################################################
# reverse-mode Jacobian (jacrev) vs forward-mode Jacobian (jacfwd)
# --------------------------------------------------------------------
# We offer two APIs to compute jacobians: jacrev and jacfwd:
# - jacrev uses reverse-mode AD. As you saw above it is a composition of our
#   vjp and vmap transforms.
# - jacfwd uses forward-mode AD. It is implemented as a composition of our
#   jvp and vmap transforms.
# jacfwd and jacrev can be subsituted for each other and have different
# performance characteristics.
#
# As a general rule of thumb, if you're computing the jacobian of an R^N -> R^M
# function, if there are many more outputs than inputs (i.e. M > N) then jacfwd is
# preferred, otherwise use jacrev. There are exceptions to this rule, but a
# non-rigorous argument for this follows:

# In reverse-mode AD, we are computing the jacobian row-by-row, while in
# forward-mode AD (which computes Jacobian-vector products), we are computing
# it column-by-column. The Jacobian matrix has M rows and N columns.
from functorch import jacfwd, jacrev

# Benchmark with more inputs than outputs
Din = 32
Dout = 2048
weight = torch.randn(Dout, Din)
bias = torch.randn(Dout)
x = torch.randn(Din)

using_fwd = Timer(stmt="jacfwd(predict, argnums=2)(weight, bias, x)", globals=globals())
using_bwd = Timer(stmt="jacrev(predict, argnums=2)(weight, bias, x)", globals=globals())
print(f"jacfwd time: {using_fwd.timeit(500)}")
print(f"jacrev time: {using_bwd.timeit(500)}")

# Benchmark with more outputs than inputs
Din = 2048
Dout = 32
weight = torch.randn(Dout, Din)
bias = torch.randn(Dout)
x = torch.randn(Din)

using_fwd = Timer(stmt="jacfwd(predict, argnums=2)(weight, bias, x)", globals=globals())
using_bwd = Timer(stmt="jacrev(predict, argnums=2)(weight, bias, x)", globals=globals())
print(f"jacfwd time: {using_fwd.timeit(500)}")
print(f"jacrev time: {using_bwd.timeit(500)}")

######################################################################
# Hessian computation with functorch.hessian
# --------------------------------------------------------------------
# We offer a convenience API to compute hessians: functorch.hessian.
# Hessians are the jacobian of the jacobian, which suggests that one can just
# compose functorch's jacobian transforms to compute one.
# Indeed, under the hood, ``hessian(f)`` is simply ``jacfwd(jacrev(f))``
#
# Depending on your model, you may want to use ``jacfwd(jacfwd(f))`` or
# ``jacrev(jacrev(f))`` instead to compute hessians.
from functorch import hessian

# # TODO: make sure PyTorch has tanh_backward implemented for jvp!!
# hess0 = hessian(predict, argnums=2)(weight, bias, x)
# hess1 = jacfwd(jacfwd(predict, argnums=2), argnums=2)(weight, bias, x)
hess2 = jacrev(jacrev(predict, argnums=2), argnums=2)(weight, bias, x)

######################################################################
# Batch Jacobian (and Batch Hessian)
# --------------------------------------------------------------------
# In the above examples we've been operating with a single feature vector.
# In some cases you might want to take the Jacobian of a batch of outputs
# with respect to a batch of inputs where each input produces an independent
# output. That is, given a batch of inputs of shape (B, N) and a function
# that goes from (B, N) -> (B, M), we would like a Jacobian of shape (B, M, N).
# The easiest way to do this is to sum over the batch dimension and then
# compute the Jacobian of that function:


def predict_with_output_summed(weight, bias, x):
    return predict(weight, bias, x).sum(0)


batch_size = 64
Din = 31
Dout = 33
weight = torch.randn(Dout, Din)
bias = torch.randn(Dout)
x = torch.randn(batch_size, Din)

batch_jacobian0 = jacrev(predict_with_output_summed, argnums=2)(weight, bias, x)

# If you instead have a function that goes from R^N -> R^M but inputs that are
# batched, you compose vmap with jacrev to compute batched jacobians:

compute_batch_jacobian = vmap(jacrev(predict, argnums=2), in_dims=(None, None, 0))
batch_jacobian1 = compute_batch_jacobian(weight, bias, x)
assert torch.allclose(batch_jacobian0, batch_jacobian1)

# Finally, batch hessians can be computed similarly. It's easiest to think about
# them by using vmap to batch over hessian computation, but in some cases the sum
# trick also works.
compute_batch_hessian = vmap(hessian(predict, argnums=2), in_dims=(None, None, 0))
batch_hess = compute_batch_hessian(weight, bias, x)
