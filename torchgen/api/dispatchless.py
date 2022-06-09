from torchgen.api import cpp
from torchgen.model import DispatchKey, FunctionSchema, NativeFunction

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                                                                     #
#                 Dispatch-less Composite Kernels                     #
#                                                                     #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
# [Note: Dispatchless Composite]
# Issue: https://github.com/pytorch/pytorch/issues/50953
#
# By definition, CompositeExplicitAutograd are operations that have its
# own differentiation formula. Meaning that they don't really need to go
# through the dispatcher every time they call a different operation.
# That's what dispatch-less composite kernels bring to the table: a way
# for kernels call operations by-passing the dispatcher.
#
# This is accomplished by:
#
#   1. building a name space (struct with static methods) that will serve
#      as proxy to the calls to operations that would go into the dispatcher
#
#   2. turning composite kernel functions into templated functions. Where
#      each operation is called using the templated type as the source name
#      space
#
# Consider, for example, 'add.Scalar' operation. Its implementation
# consists in: (i) a conversion of a scalar into a tensor; and (ii) a call
# to the same operation, but with tensor types. In order to allow it to
# take advantage of dispatch-less kernel generation, we have to:
#
#   1. list the dependent operations (name with overload name) in the
#      'native_functions.yaml' file
#
#     - func: add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
#       device_check: NoCheck   # TensorIterator
#       variants: function, method
#       composite: add.Tensor
#
#   2. move its old implementation to 'aten/src/ATen/native/composite/add.h'.
#      Not only that, but make it a templated function where each operation
#      call uses the type parameter as the source namespace
#
#     namespace at { namespace native {
#     template <typename OPS>
#     Tensor add(const Tensor& self, const Scalar& other, const Scalar& alpha) {
#       return OPS::add(self, wrapped_scalar_tensor(other), alpha);
#     }
#     }} // namespace at::native
#
# After these 2 changes, the codegen will automatically generate a CPU,
# CUDA, and CompositeExplicitAutograd kernel registration, by-passing
# the dispatcher trips (if possible). It will also work with kernels that
# already have a dispatch entry for some of the mentioned dispatch keys.
# In that case, the codegen will not generate a new kernel for that
# specific dispatch key.
#
# Given that we are generating code for a dispatch key (K), finding the
# appropriate function to call for each dependency requires considering
# 2 options, in order:
#
#   1. a function registered for K
#
#   2. the C++ API function that goes through the dispatcher
#
# In other words, if a dependent function like (1) is not found, we
# fallback to calling the dispatcher (see 'dest/dispatchless.py').
#
# Note that if K is CompositeExplicitAutograd, we always pick (2).
# See: https://github.com/pytorch/pytorch/pull/77486#issuecomment-1150992380


# name of the generated kernel.
# it wraps a call to the templated function provided.
def kernel(func: FunctionSchema, dispatch_key: DispatchKey) -> str:
    return f"{func.name.unambiguous_name()}_{dispatch_key.lower()}"


# name of the generated struct.
# it holds all the static methods that re-direct to the kernels
# by-passing the dispatcher.
def struct(f: NativeFunction) -> str:
    return f"dispatchless_struct__{cpp.name(f.func)}"


# name of the templated function + the struct name.
# this is just a convenience for calling the templated function with
# the generated struct.
def call(f: NativeFunction) -> str:
    return f"{cpp.name(f.func)}<{struct(f)}>"
