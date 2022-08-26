#pragma once

#include <ATen/core/TensorBase.h>
#include <ATen/native/DispatchStub.h>
#include <c10/core/Scalar.h>

namespace at {
struct TensorIterator;
struct TensorIteratorBase;
namespace native {
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_cos_pi_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_cosh_pi_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_sin_pi_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_sinc_pi_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_sinh_pi_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_sinhc_pi_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_sinhc_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_tan_pi_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_tanh_pi_stub);
} // namespace native
} // namespace at
