#pragma once

#include <ATen/Context.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>

namespace c10 {
class Scalar;
}
namespace at {
struct Generator;
class Tensor;
struct Type;
} // namespace at

namespace at {
namespace native {
namespace legacy {
namespace cuda {

Tensor & _th_copy_ignoring_overlaps_(Tensor & self, const Tensor & src);

} // namespace th
} // namespace legacy
} // namespace native
} // namespace at
