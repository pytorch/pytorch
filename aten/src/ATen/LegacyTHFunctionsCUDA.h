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

std::tuple<Tensor &,Tensor &> _th_gels_out(const Tensor & self, const Tensor & A, Tensor & res1, Tensor & res2);
std::tuple<Tensor,Tensor> _th_gels(const Tensor & self, const Tensor & A);
Tensor & _th_potri_out(Tensor & output, const Tensor & self, bool upper);
Tensor _th_potri(const Tensor & self, bool upper);
Tensor & _th_copy_ignoring_overlaps_(Tensor & self, const Tensor & src);
Tensor _thnn_rrelu_with_noise_backward(const Tensor & grad_output, const Tensor & self, const Tensor & noise, const Scalar& lower, const Scalar& upper, bool training);

} // namespace th
} // namespace legacy
} // namespace native
} // namespace at
