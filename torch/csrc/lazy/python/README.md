# Lazy Tensor Python Code

Lazy Tensor Core is part of libtorch, which can not depend on python.

Parts of lazy tensor core use python for 2 purposes
A) py bindings let python programs call into lazy tensor c++ code
B) lazy tensor core calls into python to use it (e.g. for grabbing stack traces)

(A) is trivial since the python bindings only depend on libtorch;
(B) requires making libtorch_python register a function with libtorch if loaded, and having a default (no-op) function otherwise.  Any functionality that strictly needs to depend on python should be part of the 'python' folder.
