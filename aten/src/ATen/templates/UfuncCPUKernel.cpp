#define TORCH_ASSERT_NO_OPERATORS

#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/ufunc/${name}.h>
#include <c10/core/Scalar.h>

namespace at {
namespace native {
$ {
  native_definitions
}
} // namespace native
} // namespace at
