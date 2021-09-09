# The UCX & UCC binding

Files in this directory will be built as a separate shared object `libtorch_ucc.so`.
PyTorch will not link to `libtorch_ucc.so` directly, but instead, PyTorch will 
dynamicly loaded it using `dlopen` at runtime.

This design allows the UCX & UCC binding to be shipped to a machine without UCX or
UCC. Importing PyTorch on that machine will succeed without causing an `undefined
reference to [some UCX/UCC symbol]` error.
