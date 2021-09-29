#include <ATen/native/ufunc/${name}.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/Dispatch.h>

namespace at {
namespace native {
${native_definitions}
}} // namespace at::native
