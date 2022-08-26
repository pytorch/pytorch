#include <ATen/native/special_functions.h>

#include <type_traits>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/ExpandUtils.h>
#include <ATen/RedispatchFunctions.h>
#include <torch/library.h>

namespace at {
namespace meta {
TORCH_META_FUNC (special_cos_pi)   (const Tensor &z) { build_borrowing_unary_float_op (maybe_get_output(), z); }
TORCH_META_FUNC (special_cosh_pi)  (const Tensor &z) { build_borrowing_unary_float_op (maybe_get_output(), z); }
TORCH_META_FUNC (special_sin_pi)   (const Tensor &z) { build_borrowing_unary_float_op (maybe_get_output(), z); }
TORCH_META_FUNC (special_sinc_pi)  (const Tensor &z) { build_borrowing_unary_float_op (maybe_get_output(), z); }
TORCH_META_FUNC (special_sinh_pi)  (const Tensor &z) { build_borrowing_unary_float_op (maybe_get_output(), z); }
TORCH_META_FUNC (special_sinhc)    (const Tensor &z) { build_borrowing_unary_float_op (maybe_get_output(), z); }
TORCH_META_FUNC (special_sinhc_pi) (const Tensor &z) { build_borrowing_unary_float_op (maybe_get_output(), z); }
TORCH_META_FUNC (special_tan_pi)   (const Tensor &z) { build_borrowing_unary_float_op (maybe_get_output(), z); }
TORCH_META_FUNC (special_tanh_pi)  (const Tensor &z) { build_borrowing_unary_float_op (maybe_get_output(), z); }
} // namespace meta

namespace native {
DEFINE_DISPATCH(special_cos_pi_stub);
DEFINE_DISPATCH(special_cosh_pi_stub);
DEFINE_DISPATCH(special_sin_pi_stub);
DEFINE_DISPATCH(special_sinc_pi_stub);
DEFINE_DISPATCH(special_sinh_pi_stub);
DEFINE_DISPATCH(special_sinhc_pi_stub);
DEFINE_DISPATCH(special_sinhc_stub);
DEFINE_DISPATCH(special_tan_pi_stub);
DEFINE_DISPATCH(special_tanh_pi_stub);

TORCH_IMPL_FUNC (special_cos_pi_out)   (const Tensor &z, const Tensor &out) { special_cos_pi_stub   (device_type(), *this); }
TORCH_IMPL_FUNC (special_cosh_pi_out)  (const Tensor &z, const Tensor &out) { special_cosh_pi_stub  (device_type(), *this); }
TORCH_IMPL_FUNC (special_sin_pi_out)   (const Tensor &z, const Tensor &out) { special_sin_pi_stub   (device_type(), *this); }
TORCH_IMPL_FUNC (special_sinc_pi_out)  (const Tensor &z, const Tensor &out) { special_sinc_pi_stub  (device_type(), *this); }
TORCH_IMPL_FUNC (special_sinh_pi_out)  (const Tensor &z, const Tensor &out) { special_sinh_pi_stub  (device_type(), *this); }
TORCH_IMPL_FUNC (special_sinhc_out)    (const Tensor &z, const Tensor &out) { special_sinhc_stub    (device_type(), *this); }
TORCH_IMPL_FUNC (special_sinhc_pi_out) (const Tensor &z, const Tensor &out) { special_sinhc_pi_stub (device_type(), *this); }
TORCH_IMPL_FUNC (special_tan_pi_out)   (const Tensor &z, const Tensor &out) { special_tan_pi_stub   (device_type(), *this); }
TORCH_IMPL_FUNC (special_tanh_pi_out)  (const Tensor &z, const Tensor &out) { special_tanh_pi_stub  (device_type(), *this); }
} // namespace native
} // namespace at
