This directory contains files from CUTLASS 3.1 source tree, modified
for the purpose of `_sparse_semi_structured_linear()` implementation.
This method implements linear operator with weight matrix $W$, bias
vector $b$, and input tensor $x$ as arguments:

$$y=xW^{T}+b$$

where the matrix $W$ is a structured sparse matrix.  Since CUTLASS
support sparse GEMM operation only when the first operand is in
structured sparse format, the operation above is actually implemented
in `_sparse_semi_structured_linear()` as follows:

$$y=(Wx^{T}+b^{T})^{T}$$

which means that bias vector $b$ is added to each column of $Wx^{T}$
product.  As of mentioned version, CUTLASS supports producing sparse
GEMM output only in row-major format, but on the other side for this
format it only provides for adding bias vector to each row of the
product.  For this reason, changed version of
`predicated_tile_iterator.h` file is created, to make it possible to
add bias vector to each column of the product, and then several other
files from CUTLASS source tree had to be modified too in order to
propagate the change up to the instantiation tree of relevant CUTLASS
classes.

Except for `predicated_tile_iterator.h`, changes from the CUTLASS
version of mentioned files are trivial.  However, it may be a slight
incovenience for detecting these changes when files in question get
processed by linters used by PyTorch.
