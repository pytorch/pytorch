.. _out-notes:

Out Notes
=========

When a user passes one or more tensors to out= the contract is as follows:

* if an out tensor has no elements it will be resized to the shape, stride, and memory format of the output of the computation.
* if an out tensor has a different shape than the result of the computation an error is thrown OR the out tensor is resized to the same shape, stride, and memory format of the output computation, just like a tensor with no elements. (This resizing behavior is deprecated and PyTorch is updating its operators to consistently throw an error.)
* passing out= tensors with the correct shape is numerically equivalent to performing the operation and "safe copying" its results to the (possibly resized) out tensor. In this case strides and memory format are preserved.
* passing out= tensors with grad needed is not supported.
* if multiple tensors are passed to out= then the above behavior applies to each independently.

A "safe copy" is different from PyTorch's regular copy. For operations that do not participate in type promotion the device and dtype of the source and destination tensors must match. For operations that do participate in type promotion the copy can be to a different dtype, but the destination of the copy cannot be a lower "type kind" than the source. PyTorch has four type kinds: boolean, integer, float, and complex, in that order. So, for example, an operation like add (which participates in type promotion) will throw a runtime error if given float inputs but an integer out= tensor.

Note that while the numerics of out= for correctly shaped tensors are that the operation is performed and then its results are "safe copied," behind the scenes operations may reuse the storage of out= tensors and fuse the copy for efficiency. Many operations, like add, perform these optimizations. Also, while PyTorch's "out= contract" is specified above, many operations in PyTorch do not correctly implement the contract and need to be updated.