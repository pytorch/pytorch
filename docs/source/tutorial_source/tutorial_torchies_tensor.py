"""
Tensors
=======

Tensors behave almost exactly the same way in PyTorch as they do in
Torch.

Create a tensor of size (5 x 7) with uninitialized memory:

"""

import torch
a = torch.FloatTensor(5, 7)

###############################################################
# Initialize a tensor randomized with a normal distribution with mean=0, var=1:

a = torch.randn(5, 7)
print(a)
print(a.size())

###############################################################
# .. note::
#     ``torch.Size`` is in fact a tuple, so it supports the same operations
#
# Inplace / Out-of-place
# ----------------------
#
# The first difference is that ALL operations on the tensor that operate
# in-place on it will have an ``_`` postfix. For example, ``add`` is the
# out-of-place version, and ``add_`` is the in-place version.

a.fill_(3.5)
# a has now been filled with the value 3.5

b = a.add(4.0)
# a is still filled with 3.5
# new tensor b is returned with values 3.5 + 4.0 = 7.5

print(a, b)

###############################################################
# Some operations like ``narrow`` do not have in-place versions, and
# hence, ``.narrow_`` does not exist. Similarly, some operations like
# ``fill_`` do not have an out-of-place version, so ``.fill`` does not
# exist.
#
# Zero Indexing
# -------------
#
# Another difference is that Tensors are zero-indexed. (In lua, tensors are
# one-indexed)

b = a[0, 3]  # select 1st row, 4th column from a

###############################################################
# Tensors can be also indexed with Python's slicing

b = a[:, 3:5]  # selects all rows, 4th column and  5th column from a

###############################################################
# No camel casing
# ---------------
#
# The next small difference is that all functions are now NOT camelCase
# anymore. For example ``indexAdd`` is now called ``index_add_``


x = torch.ones(5, 5)
print(x)

###############################################################
#

z = torch.Tensor(5, 2)
z[:, 0] = 10
z[:, 1] = 100
print(z)

###############################################################
#
x.index_add_(1, torch.LongTensor([4, 0]), z)
print(x)

###############################################################
# Numpy Bridge
# ------------
#
# Converting a torch Tensor to a numpy array and vice versa is a breeze.
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
#
a.add_(1)
print(a)
print(b) 	# see how the numpy array changed in value


###############################################################
# Converting numpy Array to torch Tensor
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)  # see how changing the np array changed the torch Tensor automatically

###############################################################
# All the Tensors on the CPU except a CharTensor support converting to
# NumPy and back.
#
# CUDA Tensors
# ------------
#
# CUDA Tensors are nice and easy in pytorch, and they are much more
# consistent as well. Transfering a CUDA tensor from the CPU to GPU will
# retain it’s type.

# let us run this cell only if CUDA is available
if torch.cuda.is_available():
    # creates a LongTensor and transfers it
    # to GPU as torch.cuda.LongTensor
    a = torch.LongTensor(10).fill_(3).cuda()
    print(type(a))
    b = a.cpu()
    # transfers it to CPU, back to
    # being a torch.LongTensor
