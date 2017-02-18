"""
What is PyTorch?
================

It’s a Python based scientific computing package targeted at two sets of
audiences:

-  A replacement for numpy to use the power of GPUs
-  a deep learning research platform that provides maximum flexibility
   and speed

Getting Started
---------------

Tensors
^^^^^^^

Tensors are similar to numpy’s ndarrays, with the addition being that
Tensors can also be used on a GPU to accelerate computing.
"""

from __future__ import print_function
import torch

###############################################################
# Construct a 5x3 matrix, uninitialized:

x = torch.Tensor(5, 3)
print(x)

###############################################################
# Construct a randomly initialized matrix

x = torch.rand(5, 3)
print(x)

###############################################################
# Get its size

print(x.size())

###############################################################
# .. note::
#     ``torch.Size`` is in fact a tuple, so it supports the same operations
# Operations
# ^^^^^^^^^^
# There are multiple syntaxes for operations. Let's see addition as an example
#
# Addition: syntax 1
y = torch.rand(5, 3)
print(x + y)

###############################################################
# Addition: syntax 2

print(torch.add(x, y))

###############################################################
# Addition: giving an output tensor
result = torch.Tensor(5, 3)
torch.add(x, y, out=result)
print(result)

###############################################################
# addition: in-place

# adds x to y
y.add_(x)
print(y)

###############################################################
# .. note::
#     Any operation that mutates a tensor in-place is post-fixed with an ``_``
#     For example: ``x.copy_(y)``, ``x.t_()``, will change ``x``.
#
# You can use standard numpy-like indexing with all bells and whistles!

print(x[:, 1])

###############################################################
# **Read later:**
#
#
# 100+ Tensor operations, including transposing, indexing, slicing,
# mathematical operations, linear algebra, random numbers, etc.
#
# http://pytorch.org/docs/torch.html
#
# Numpy Bridge
# ------------
#
# Converting a torch Tensor to a numpy array and vice versa is a breeze.
#
# The torch Tensor and numpy array will share their underlying memory
# locations, and changing one will change the other.
#
# Converting torch Tensor to numpy Array
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

a = torch.ones(5)
print(a)

###############################################################
#


b = a.numpy()
print(b)

###############################################################
# See how the numpy array changed in value.

a.add_(1)
print(a)
print(b)

###############################################################
# Converting numpy Array to torch Tensor
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# See how changing the np array changed the torch Tensor automatically

import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

###############################################################
# All the Tensors on the CPU except a CharTensor support converting to
# NumPy and back.
#
# CUDA Tensors
# ------------
#
# Tensors can be moved onto GPU using the ``.cuda`` function.

# let us run this cell only if CUDA is available
if torch.cuda.is_available():
    x = x.cuda()
    y = y.cuda()
    x + y
