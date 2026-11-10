#include <torch/csrc/stable/cpublas.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/util/Exception.h>

namespace cpublas = torch::stable::cpublas;
using torch::stable::Tensor;

namespace {

void check_matrices(
    const Tensor& a,
    const Tensor& b,
    const Tensor& c) {
  STD_TORCH_CHECK(
      a.is_cpu() && b.is_cpu() && c.is_cpu(),
      "cpublas test inputs must be CPU tensors");
  STD_TORCH_CHECK(
      a.dim() == 2 && b.dim() == 2 && c.dim() == 2,
      "cpublas test inputs must be matrices");
  STD_TORCH_CHECK(
      a.is_contiguous() && b.is_contiguous() && c.is_contiguous(),
      "cpublas test inputs must be contiguous");
  STD_TORCH_CHECK(
      a.size(1) == b.size(0),
      "cpublas test matrix inner dimensions do not match");
  STD_TORCH_CHECK(
      c.size(0) == a.size(0) && c.size(1) == b.size(1),
      "cpublas test output matrix shape does not match");
}

} // namespace

Tensor my_cpublas_brgemm(
    const Tensor& a,
    const Tensor& b,
    Tensor c,
    bool add_c) {
  check_matrices(a, b, c);
  const auto call_brgemm = [&](auto* a_ptr, auto* b_ptr, auto* c_ptr) {
    cpublas::brgemm(
        a.size(0),
        b.size(1),
        a.size(1),
        a.stride(0),
        b.stride(0),
        c.stride(0),
        add_c,
        a_ptr,
        b_ptr,
        c_ptr,
        false);
  };

  const auto dtype_a = a.scalar_type();
  const auto dtype_b = b.scalar_type();
  const auto dtype_c = c.scalar_type();
  if (dtype_a == torch::headeronly::ScalarType::Float &&
      dtype_b == torch::headeronly::ScalarType::Float &&
      dtype_c == torch::headeronly::ScalarType::Float) {
    call_brgemm(
        a.const_data_ptr<float>(),
        b.const_data_ptr<float>(),
        c.mutable_data_ptr<float>());
  } else if (
      dtype_a == torch::headeronly::ScalarType::Half &&
      dtype_b == torch::headeronly::ScalarType::Half &&
      dtype_c == torch::headeronly::ScalarType::Float) {
    call_brgemm(
        a.const_data_ptr<c10::Half>(),
        b.const_data_ptr<c10::Half>(),
        c.mutable_data_ptr<float>());
  } else if (
      dtype_a == torch::headeronly::ScalarType::BFloat16 &&
      dtype_b == torch::headeronly::ScalarType::BFloat16 &&
      dtype_c == torch::headeronly::ScalarType::Float) {
    call_brgemm(
        a.const_data_ptr<c10::BFloat16>(),
        b.const_data_ptr<c10::BFloat16>(),
        c.mutable_data_ptr<float>());
  } else if (
      dtype_a == torch::headeronly::ScalarType::Byte &&
      dtype_b == torch::headeronly::ScalarType::Byte &&
      dtype_c == torch::headeronly::ScalarType::Int) {
    call_brgemm(
        a.const_data_ptr<uint8_t>(),
        b.const_data_ptr<uint8_t>(),
        c.mutable_data_ptr<int32_t>());
  } else if (
      dtype_a == torch::headeronly::ScalarType::Byte &&
      dtype_b == torch::headeronly::ScalarType::Char &&
      dtype_c == torch::headeronly::ScalarType::Int) {
    call_brgemm(
        a.const_data_ptr<uint8_t>(),
        b.const_data_ptr<int8_t>(),
        c.mutable_data_ptr<int32_t>());
  } else if (
      dtype_a == torch::headeronly::ScalarType::Char &&
      dtype_b == torch::headeronly::ScalarType::Char &&
      dtype_c == torch::headeronly::ScalarType::Int) {
    call_brgemm(
        a.const_data_ptr<int8_t>(),
        b.const_data_ptr<int8_t>(),
        c.mutable_data_ptr<int32_t>());
  } else if (
      dtype_a == torch::headeronly::ScalarType::Float8_e4m3fn &&
      dtype_b == torch::headeronly::ScalarType::Float8_e4m3fn &&
      dtype_c == torch::headeronly::ScalarType::Float) {
    call_brgemm(
        a.const_data_ptr<c10::Float8_e4m3fn>(),
        b.const_data_ptr<c10::Float8_e4m3fn>(),
        c.mutable_data_ptr<float>());
  } else if (
      dtype_a == torch::headeronly::ScalarType::Float8_e5m2 &&
      dtype_b == torch::headeronly::ScalarType::Float8_e5m2 &&
      dtype_c == torch::headeronly::ScalarType::Float) {
    call_brgemm(
        a.const_data_ptr<c10::Float8_e5m2>(),
        b.const_data_ptr<c10::Float8_e5m2>(),
        c.mutable_data_ptr<float>());
  } else {
    STD_TORCH_CHECK(false, "unsupported cpublas brgemm input/output dtypes");
  }
  return c;
}

Tensor my_cpublas_gemm_no_downcast_default_real(
    const Tensor& a,
    const Tensor& b,
    Tensor c,
    double alpha,
    double beta) {
  check_matrices(a, b, c);
  STD_TORCH_CHECK(
      a.scalar_type() == torch::headeronly::ScalarType::Float &&
          b.scalar_type() == torch::headeronly::ScalarType::Float &&
          c.scalar_type() == torch::headeronly::ScalarType::Float,
      "cpublas GEMM test inputs must be float32 tensors");
  // The native GEMM stub uses column-major storage. Swap A and B to operate
  // on row-major tensors, as the native brgemm fallback does.
  cpublas::gemm_no_downcast_default_real(
      torch::headeronly::ScalarType::Float,
      torch::headeronly::TransposeType::NoTranspose,
      torch::headeronly::TransposeType::NoTranspose,
      b.size(1),
      a.size(0),
      a.size(1),
      alpha,
      b.const_data_ptr<float>(),
      b.stride(0),
      a.const_data_ptr<float>(),
      a.stride(0),
      beta,
      c.mutable_data_ptr<float>(),
      c.stride(0));
  return c;
}

bool my_cpublas_could_pack(const Tensor& input) {
  return cpublas::could_pack(input.scalar_type());
}

bool my_cpublas_brgemm_release() {
  cpublas::brgemm_release(false);
  return true;
}

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def(
      "my_cpublas_brgemm(Tensor a, Tensor b, Tensor(a!) c, bool add_c) -> Tensor(a!)");
  m.def(
      "my_cpublas_gemm_no_downcast_default_real(Tensor a, Tensor b, Tensor(a!) c, float alpha, float beta) -> Tensor(a!)");
  m.def("my_cpublas_could_pack(Tensor input) -> bool");
  m.def("my_cpublas_brgemm_release() -> bool");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CompositeExplicitAutograd, m) {
  m.impl("my_cpublas_brgemm", TORCH_BOX(&my_cpublas_brgemm));
  m.impl(
      "my_cpublas_gemm_no_downcast_default_real",
      TORCH_BOX(&my_cpublas_gemm_no_downcast_default_real));
  m.impl("my_cpublas_could_pack", TORCH_BOX(&my_cpublas_could_pack));
  m.impl("my_cpublas_brgemm_release", TORCH_BOX(&my_cpublas_brgemm_release));
}
