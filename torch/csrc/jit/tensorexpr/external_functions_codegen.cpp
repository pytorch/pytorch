#include <torch/csrc/jit/tensorexpr/external_functions.h>

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/tensorexpr/external_functions_registry.h>

namespace torch {
namespace jit {
namespace tensorexpr {

#ifdef C10_MOBILE
extern "C" {
#endif

#ifndef C10_MOBILE

void nnc_aten_abs(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::abs_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_abs("nnc_aten_abs", nnc_aten_abs);

void nnc_aten_absolute(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::absolute_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_absolute(
    "nnc_aten_absolute",
    nnc_aten_absolute);

void nnc_aten_angle(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::angle_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_angle(
    "nnc_aten_angle",
    nnc_aten_angle);

void nnc_aten_sgn(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::sgn_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_sgn("nnc_aten_sgn", nnc_aten_sgn);

void nnc_aten_conj(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::conj_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_conj(
    "nnc_aten_conj",
    nnc_aten_conj);

void nnc_aten_acos(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::acos_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_acos(
    "nnc_aten_acos",
    nnc_aten_acos);

void nnc_aten_arccos(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::arccos_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_arccos(
    "nnc_aten_arccos",
    nnc_aten_arccos);

void nnc_aten_acosh(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::acosh_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_acosh(
    "nnc_aten_acosh",
    nnc_aten_acosh);

void nnc_aten_arccosh(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::arccosh_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_arccosh(
    "nnc_aten_arccosh",
    nnc_aten_arccosh);

void nnc_aten_asinh(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::asinh_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_asinh(
    "nnc_aten_asinh",
    nnc_aten_asinh);

void nnc_aten_arcsinh(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::arcsinh_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_arcsinh(
    "nnc_aten_arcsinh",
    nnc_aten_arcsinh);

void nnc_aten_atanh(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::atanh_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_atanh(
    "nnc_aten_atanh",
    nnc_aten_atanh);

void nnc_aten_arctanh(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::arctanh_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_arctanh(
    "nnc_aten_arctanh",
    nnc_aten_arctanh);

void nnc_aten_asin(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::asin_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_asin(
    "nnc_aten_asin",
    nnc_aten_asin);

void nnc_aten_arcsin(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::arcsin_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_arcsin(
    "nnc_aten_arcsin",
    nnc_aten_arcsin);

void nnc_aten_atan(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::atan_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_atan(
    "nnc_aten_atan",
    nnc_aten_atan);

void nnc_aten_arctan(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::arctan_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_arctan(
    "nnc_aten_arctan",
    nnc_aten_arctan);

void nnc_aten_bitwise_not(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::bitwise_not_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_bitwise_not(
    "nnc_aten_bitwise_not",
    nnc_aten_bitwise_not);

void nnc_aten_copysign(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  const at::Tensor& other = tensors[2];
  try {
    at::copysign_out(r, self, other);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_copysign(
    "nnc_aten_copysign",
    nnc_aten_copysign);

void nnc_aten_logical_not(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::logical_not_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_logical_not(
    "nnc_aten_logical_not",
    nnc_aten_logical_not);

void nnc_aten_logical_xor(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  const at::Tensor& other = tensors[2];
  try {
    at::logical_xor_out(r, self, other);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_logical_xor(
    "nnc_aten_logical_xor",
    nnc_aten_logical_xor);

void nnc_aten_logical_and(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  const at::Tensor& other = tensors[2];
  try {
    at::logical_and_out(r, self, other);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_logical_and(
    "nnc_aten_logical_and",
    nnc_aten_logical_and);

void nnc_aten_logical_or(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  const at::Tensor& other = tensors[2];
  try {
    at::logical_or_out(r, self, other);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_logical_or(
    "nnc_aten_logical_or",
    nnc_aten_logical_or);

void nnc_aten_bmm(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  const at::Tensor& mat2 = tensors[2];
  try {
    at::bmm_out(r, self, mat2);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_bmm("nnc_aten_bmm", nnc_aten_bmm);

void nnc_aten_ceil(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::ceil_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_ceil(
    "nnc_aten_ceil",
    nnc_aten_ceil);

void nnc_aten_complex(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& real = tensors[1];
  const at::Tensor& imag = tensors[2];
  try {
    at::complex_out(r, real, imag);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_complex(
    "nnc_aten_complex",
    nnc_aten_complex);

void nnc_aten_polar(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& abs = tensors[1];
  const at::Tensor& angle = tensors[2];
  try {
    at::polar_out(r, abs, angle);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_polar(
    "nnc_aten_polar",
    nnc_aten_polar);

void nnc_aten_cos(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::cos_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_cos("nnc_aten_cos", nnc_aten_cos);

void nnc_aten_cosh(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::cosh_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_cosh(
    "nnc_aten_cosh",
    nnc_aten_cosh);

void nnc_aten_div(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  const at::Tensor& other = tensors[2];
  try {
    at::div_out(r, self, other);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_div("nnc_aten_div", nnc_aten_div);

void nnc_aten_divide(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  const at::Tensor& other = tensors[2];
  try {
    at::divide_out(r, self, other);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_divide(
    "nnc_aten_divide",
    nnc_aten_divide);

void nnc_aten_true_divide(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  const at::Tensor& other = tensors[2];
  try {
    at::true_divide_out(r, self, other);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_true_divide(
    "nnc_aten_true_divide",
    nnc_aten_true_divide);

void nnc_aten_dot(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  const at::Tensor& tensor = tensors[2];
  try {
    at::dot_out(r, self, tensor);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_dot("nnc_aten_dot", nnc_aten_dot);

void nnc_aten_vdot(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  const at::Tensor& other = tensors[2];
  try {
    at::vdot_out(r, self, other);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_vdot(
    "nnc_aten_vdot",
    nnc_aten_vdot);

void nnc_aten_erf(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::erf_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_erf("nnc_aten_erf", nnc_aten_erf);

void nnc_aten_erfc(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::erfc_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_erfc(
    "nnc_aten_erfc",
    nnc_aten_erfc);

void nnc_aten_exp(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::exp_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_exp("nnc_aten_exp", nnc_aten_exp);

void nnc_aten_exp2(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::exp2_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_exp2(
    "nnc_aten_exp2",
    nnc_aten_exp2);

void nnc_aten_expm1(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::expm1_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_expm1(
    "nnc_aten_expm1",
    nnc_aten_expm1);

void nnc_aten_floor(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::floor_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_floor(
    "nnc_aten_floor",
    nnc_aten_floor);

void nnc_aten_floor_divide(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  const at::Tensor& other = tensors[2];
  try {
    at::floor_divide_out(r, self, other);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_floor_divide(
    "nnc_aten_floor_divide",
    nnc_aten_floor_divide);

void nnc_aten_frac(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::frac_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_frac(
    "nnc_aten_frac",
    nnc_aten_frac);

void nnc_aten_gcd(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  const at::Tensor& other = tensors[2];
  try {
    at::gcd_out(r, self, other);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_gcd("nnc_aten_gcd", nnc_aten_gcd);

void nnc_aten_lcm(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  const at::Tensor& other = tensors[2];
  try {
    at::lcm_out(r, self, other);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_lcm("nnc_aten_lcm", nnc_aten_lcm);

void nnc_aten_inverse(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::inverse_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_inverse(
    "nnc_aten_inverse",
    nnc_aten_inverse);

void nnc_aten_kron(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  const at::Tensor& other = tensors[2];
  try {
    at::kron_out(r, self, other);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_kron(
    "nnc_aten_kron",
    nnc_aten_kron);

void nnc_aten_ldexp(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  const at::Tensor& other = tensors[2];
  try {
    at::ldexp_out(r, self, other);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_ldexp(
    "nnc_aten_ldexp",
    nnc_aten_ldexp);

void nnc_aten_log(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::log_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_log("nnc_aten_log", nnc_aten_log);

void nnc_aten_log10(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::log10_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_log10(
    "nnc_aten_log10",
    nnc_aten_log10);

void nnc_aten_log1p(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::log1p_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_log1p(
    "nnc_aten_log1p",
    nnc_aten_log1p);

void nnc_aten_log2(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::log2_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_log2(
    "nnc_aten_log2",
    nnc_aten_log2);

void nnc_aten_logaddexp(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  const at::Tensor& other = tensors[2];
  try {
    at::logaddexp_out(r, self, other);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_logaddexp(
    "nnc_aten_logaddexp",
    nnc_aten_logaddexp);

void nnc_aten_logaddexp2(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  const at::Tensor& other = tensors[2];
  try {
    at::logaddexp2_out(r, self, other);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_logaddexp2(
    "nnc_aten_logaddexp2",
    nnc_aten_logaddexp2);

void nnc_aten_matmul(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  const at::Tensor& other = tensors[2];
  try {
    at::matmul_out(r, self, other);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_matmul(
    "nnc_aten_matmul",
    nnc_aten_matmul);

void nnc_aten__compute_linear_combination(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& input = tensors[1];
  const at::Tensor& coefficients = tensors[2];
  try {
    at::_compute_linear_combination_out(r, input, coefficients);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc__compute_linear_combination(
    "nnc_aten__compute_linear_combination",
    nnc_aten__compute_linear_combination);

void nnc_aten_mm(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  const at::Tensor& mat2 = tensors[2];
  try {
    at::mm_out(r, self, mat2);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_mm("nnc_aten_mm", nnc_aten_mm);

void nnc_aten_mul(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  const at::Tensor& other = tensors[2];
  try {
    at::mul_out(r, self, other);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_mul("nnc_aten_mul", nnc_aten_mul);

void nnc_aten_multiply(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  const at::Tensor& other = tensors[2];
  try {
    at::multiply_out(r, self, other);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_multiply(
    "nnc_aten_multiply",
    nnc_aten_multiply);

void nnc_aten_mv(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  const at::Tensor& vec = tensors[2];
  try {
    at::mv_out(r, self, vec);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_mv("nnc_aten_mv", nnc_aten_mv);

void nnc_aten_rad2deg(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::rad2deg_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_rad2deg(
    "nnc_aten_rad2deg",
    nnc_aten_rad2deg);

void nnc_aten_deg2rad(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::deg2rad_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_deg2rad(
    "nnc_aten_deg2rad",
    nnc_aten_deg2rad);

void nnc_aten_reciprocal(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::reciprocal_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_reciprocal(
    "nnc_aten_reciprocal",
    nnc_aten_reciprocal);

void nnc_aten_neg(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::neg_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_neg("nnc_aten_neg", nnc_aten_neg);

void nnc_aten_negative(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::negative_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_negative(
    "nnc_aten_negative",
    nnc_aten_negative);

void nnc_aten_round(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::round_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_round(
    "nnc_aten_round",
    nnc_aten_round);

void nnc_aten_rsqrt(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::rsqrt_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_rsqrt(
    "nnc_aten_rsqrt",
    nnc_aten_rsqrt);

void nnc_aten_silu(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::silu_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_silu(
    "nnc_aten_silu",
    nnc_aten_silu);

void nnc_aten_sigmoid(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::sigmoid_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_sigmoid(
    "nnc_aten_sigmoid",
    nnc_aten_sigmoid);

void nnc_aten_sin(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::sin_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_sin("nnc_aten_sin", nnc_aten_sin);

void nnc_aten_sinc(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::sinc_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_sinc(
    "nnc_aten_sinc",
    nnc_aten_sinc);

void nnc_aten_sinh(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::sinh_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_sinh(
    "nnc_aten_sinh",
    nnc_aten_sinh);

void nnc_aten_sqrt(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::sqrt_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_sqrt(
    "nnc_aten_sqrt",
    nnc_aten_sqrt);

void nnc_aten_square(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::square_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_square(
    "nnc_aten_square",
    nnc_aten_square);

void nnc_aten_tan(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::tan_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_tan("nnc_aten_tan", nnc_aten_tan);

void nnc_aten_tanh(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::tanh_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_tanh(
    "nnc_aten_tanh",
    nnc_aten_tanh);

void nnc_aten_trunc(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::trunc_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_trunc(
    "nnc_aten_trunc",
    nnc_aten_trunc);

void nnc_aten_fix(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::fix_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_fix("nnc_aten_fix", nnc_aten_fix);

void nnc_aten_heaviside(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  const at::Tensor& values = tensors[2];
  try {
    at::heaviside_out(r, self, values);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_heaviside(
    "nnc_aten_heaviside",
    nnc_aten_heaviside);

void nnc_aten_hspmm(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& mat1 = tensors[1];
  const at::Tensor& mat2 = tensors[2];
  try {
    at::hspmm_out(r, mat1, mat2);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_hspmm(
    "nnc_aten_hspmm",
    nnc_aten_hspmm);

void nnc_aten_take(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  const at::Tensor& index = tensors[2];
  try {
    at::take_out(r, self, index);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_take(
    "nnc_aten_take",
    nnc_aten_take);

void nnc_aten_masked_select(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  const at::Tensor& mask = tensors[2];
  try {
    at::masked_select_out(r, self, mask);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_masked_select(
    "nnc_aten_masked_select",
    nnc_aten_masked_select);

void nnc_aten_nonzero(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::nonzero_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_nonzero(
    "nnc_aten_nonzero",
    nnc_aten_nonzero);

void nnc_aten_orgqr(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  const at::Tensor& input2 = tensors[2];
  try {
    at::orgqr_out(r, self, input2);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_orgqr(
    "nnc_aten_orgqr",
    nnc_aten_orgqr);

void nnc_aten_lu_solve(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  const at::Tensor& LU_data = tensors[2];
  const at::Tensor& LU_pivots = tensors[3];
  try {
    at::lu_solve_out(r, self, LU_data, LU_pivots);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_lu_solve(
    "nnc_aten_lu_solve",
    nnc_aten_lu_solve);

void nnc_aten_lgamma(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::lgamma_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_lgamma(
    "nnc_aten_lgamma",
    nnc_aten_lgamma);

void nnc_aten_digamma(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::digamma_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_digamma(
    "nnc_aten_digamma",
    nnc_aten_digamma);

void nnc_aten_erfinv(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::erfinv_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_erfinv(
    "nnc_aten_erfinv",
    nnc_aten_erfinv);

void nnc_aten_i0(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::i0_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_i0("nnc_aten_i0", nnc_aten_i0);

void nnc_aten_sign(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::sign_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_sign(
    "nnc_aten_sign",
    nnc_aten_sign);

void nnc_aten_signbit(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::signbit_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_signbit(
    "nnc_aten_signbit",
    nnc_aten_signbit);

void nnc_aten_atan2(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  const at::Tensor& other = tensors[2];
  try {
    at::atan2_out(r, self, other);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_atan2(
    "nnc_aten_atan2",
    nnc_aten_atan2);

void nnc_aten_hypot(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  const at::Tensor& other = tensors[2];
  try {
    at::hypot_out(r, self, other);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_hypot(
    "nnc_aten_hypot",
    nnc_aten_hypot);

void nnc_aten_igamma(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  const at::Tensor& other = tensors[2];
  try {
    at::igamma_out(r, self, other);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_igamma(
    "nnc_aten_igamma",
    nnc_aten_igamma);

void nnc_aten_igammac(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  const at::Tensor& other = tensors[2];
  try {
    at::igammac_out(r, self, other);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_igammac(
    "nnc_aten_igammac",
    nnc_aten_igammac);

void nnc_aten_nextafter(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  const at::Tensor& other = tensors[2];
  try {
    at::nextafter_out(r, self, other);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_nextafter(
    "nnc_aten_nextafter",
    nnc_aten_nextafter);

void nnc_aten_fmin(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  const at::Tensor& other = tensors[2];
  try {
    at::fmin_out(r, self, other);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_fmin(
    "nnc_aten_fmin",
    nnc_aten_fmin);

void nnc_aten_fmax(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  const at::Tensor& other = tensors[2];
  try {
    at::fmax_out(r, self, other);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_fmax(
    "nnc_aten_fmax",
    nnc_aten_fmax);

void nnc_aten_maximum(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  const at::Tensor& other = tensors[2];
  try {
    at::maximum_out(r, self, other);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_maximum(
    "nnc_aten_maximum",
    nnc_aten_maximum);

void nnc_aten_max(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  const at::Tensor& other = tensors[2];
  try {
    at::max_out(r, self, other);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_max("nnc_aten_max", nnc_aten_max);

void nnc_aten_minimum(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  const at::Tensor& other = tensors[2];
  try {
    at::minimum_out(r, self, other);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_minimum(
    "nnc_aten_minimum",
    nnc_aten_minimum);

void nnc_aten_min(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  const at::Tensor& other = tensors[2];
  try {
    at::min_out(r, self, other);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_min("nnc_aten_min", nnc_aten_min);

void nnc_aten_msort(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::msort_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_msort(
    "nnc_aten_msort",
    nnc_aten_msort);

void nnc_aten_hardsigmoid(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::hardsigmoid_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_hardsigmoid(
    "nnc_aten_hardsigmoid",
    nnc_aten_hardsigmoid);

void nnc_aten_hardswish(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::hardswish_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_hardswish(
    "nnc_aten_hardswish",
    nnc_aten_hardswish);

void nnc_aten_log_sigmoid(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::log_sigmoid_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_log_sigmoid(
    "nnc_aten_log_sigmoid",
    nnc_aten_log_sigmoid);

void nnc_aten_isposinf(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::isposinf_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_isposinf(
    "nnc_aten_isposinf",
    nnc_aten_isposinf);

void nnc_aten_isneginf(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::isneginf_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_isneginf(
    "nnc_aten_isneginf",
    nnc_aten_isneginf);

void nnc_aten_special_entr(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::special_entr_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_special_entr(
    "nnc_aten_special_entr",
    nnc_aten_special_entr);

void nnc_aten_special_expm1(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::special_expm1_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_special_expm1(
    "nnc_aten_special_expm1",
    nnc_aten_special_expm1);

void nnc_aten_special_exp2(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::special_exp2_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_special_exp2(
    "nnc_aten_special_exp2",
    nnc_aten_special_exp2);

void nnc_aten_special_gammaln(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::special_gammaln_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_special_gammaln(
    "nnc_aten_special_gammaln",
    nnc_aten_special_gammaln);

void nnc_aten_special_erf(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::special_erf_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_special_erf(
    "nnc_aten_special_erf",
    nnc_aten_special_erf);

void nnc_aten_special_erfc(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::special_erfc_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_special_erfc(
    "nnc_aten_special_erfc",
    nnc_aten_special_erfc);

void nnc_aten_special_erfinv(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::special_erfinv_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_special_erfinv(
    "nnc_aten_special_erfinv",
    nnc_aten_special_erfinv);

void nnc_aten_special_xlog1py(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  const at::Tensor& other = tensors[2];
  try {
    at::special_xlog1py_out(r, self, other);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_special_xlog1py(
    "nnc_aten_special_xlog1py",
    nnc_aten_special_xlog1py);

void nnc_aten_special_i0e(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::special_i0e_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_special_i0e(
    "nnc_aten_special_i0e",
    nnc_aten_special_i0e);

void nnc_aten_special_expit(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::special_expit_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_special_expit(
    "nnc_aten_special_expit",
    nnc_aten_special_expit);

void nnc_aten_linalg_cholesky(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::linalg_cholesky_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_linalg_cholesky(
    "nnc_aten_linalg_cholesky",
    nnc_aten_linalg_cholesky);

void nnc_aten_linalg_det(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::linalg_det_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_linalg_det(
    "nnc_aten_linalg_det",
    nnc_aten_linalg_det);

void nnc_aten_linalg_eigvals(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::linalg_eigvals_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_linalg_eigvals(
    "nnc_aten_linalg_eigvals",
    nnc_aten_linalg_eigvals);

void nnc_aten_linalg_householder_product(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& input = tensors[1];
  const at::Tensor& tau = tensors[2];
  try {
    at::linalg_householder_product_out(r, input, tau);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_linalg_householder_product(
    "nnc_aten_linalg_householder_product",
    nnc_aten_linalg_householder_product);

void nnc_aten_linalg_inv(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  try {
    at::linalg_inv_out(r, self);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_linalg_inv(
    "nnc_aten_linalg_inv",
    nnc_aten_linalg_inv);

void nnc_aten_inner(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  const at::Tensor& other = tensors[2];
  try {
    at::inner_out(r, self, other);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_inner(
    "nnc_aten_inner",
    nnc_aten_inner);

void nnc_aten_outer(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  const at::Tensor& vec2 = tensors[2];
  try {
    at::outer_out(r, self, vec2);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_outer(
    "nnc_aten_outer",
    nnc_aten_outer);

void nnc_aten_ger(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& self = tensors[1];
  const at::Tensor& vec2 = tensors[2];
  try {
    at::ger_out(r, self, vec2);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_ger("nnc_aten_ger", nnc_aten_ger);

void nnc_aten_linalg_svdvals(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& input = tensors[1];
  try {
    at::linalg_svdvals_out(r, input);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_linalg_svdvals(
    "nnc_aten_linalg_svdvals",
    nnc_aten_linalg_svdvals);

void nnc_aten_linalg_solve(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor& r = tensors[0];
  const at::Tensor& input = tensors[1];
  const at::Tensor& other = tensors[2];
  try {
    at::linalg_solve_out(r, input, other);
  } catch (...) {
  }
}
const static RegisterNNCExternalFunction nnc_linalg_solve(
    "nnc_aten_linalg_solve",
    nnc_aten_linalg_solve);
#endif

#ifdef C10_MOBILE
} // extern "C"
#endif

} // namespace tensorexpr
} // namespace jit
} // namespace torch
