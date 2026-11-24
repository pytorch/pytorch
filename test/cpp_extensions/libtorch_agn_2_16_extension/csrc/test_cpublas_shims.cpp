#include <torch/csrc/stable/cpublas.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/util/Exception.h>

namespace cpublas = torch::stable::cpublas;
using torch::stable::Tensor;

template <typename AType, typename BType, typename CType>
Tensor my_cpublas_brgemm(
    const Tensor& a,
    const Tensor& b,
    Tensor c,
    bool add_c,
    bool is_vnni) {
  cpublas::brgemm(
      a.size(0),
      b.size(1),
      a.size(1),
      a.stride(0),
      b.stride(0),
      c.stride(0),
      add_c,
      a.const_data_ptr<AType>(),
      b.const_data_ptr<BType>(),
      c.mutable_data_ptr<CType>(),
      is_vnni);
  return c;
}

Tensor my_cpublas_brgemm_dispatch(
    const Tensor& a,
    const Tensor& b,
    Tensor c,
    bool add_c,
    bool is_vnni) {
  using S = torch::headeronly::ScalarType;

  const auto dtype_a = a.scalar_type();
  const auto dtype_b = b.scalar_type();
  const auto dtype_c = c.scalar_type();
  const auto matches = [&](S expected_a, S expected_b, S expected_c) {
    return dtype_a == expected_a && dtype_b == expected_b &&
        dtype_c == expected_c;
  };

  if (matches(S::Float, S::Float, S::Float)) {
    return my_cpublas_brgemm<float, float, float>(a, b, c, add_c, is_vnni);
  }
  if (matches(S::Half, S::Half, S::Float)) {
    return my_cpublas_brgemm<
        torch::headeronly::Half,
        torch::headeronly::Half,
        float>(a, b, c, add_c, is_vnni);
  }
  if (matches(S::BFloat16, S::BFloat16, S::Float)) {
    return my_cpublas_brgemm<
        torch::headeronly::BFloat16,
        torch::headeronly::BFloat16,
        float>(a, b, c, add_c, is_vnni);
  }
  if (matches(S::Byte, S::Byte, S::Int)) {
    return my_cpublas_brgemm<unsigned char, unsigned char, int32_t>(
        a, b, c, add_c, is_vnni);
  }
  if (matches(S::Byte, S::Char, S::Int)) {
    return my_cpublas_brgemm<unsigned char, signed char, int32_t>(
        a, b, c, add_c, is_vnni);
  }
  if (matches(S::Char, S::Char, S::Int)) {
    return my_cpublas_brgemm<signed char, signed char, int32_t>(
        a, b, c, add_c, is_vnni);
  }
  if (matches(S::Float8_e4m3fn, S::Float8_e4m3fn, S::Float)) {
    return my_cpublas_brgemm<
        torch::headeronly::Float8_e4m3fn,
        torch::headeronly::Float8_e4m3fn,
        float>(a, b, c, add_c, is_vnni);
  }
  if (matches(S::Float8_e5m2, S::Float8_e5m2, S::Float)) {
    return my_cpublas_brgemm<
        torch::headeronly::Float8_e5m2,
        torch::headeronly::Float8_e5m2,
        float>(a, b, c, add_c, is_vnni);
  }

  STD_TORCH_CHECK(
      false,
      "Unsupported brgemm dtype combination: ",
      dtype_a,
      ", ",
      dtype_b,
      " -> ",
      dtype_c);
}

Tensor my_cpublas_gemm_no_downcast_stub_cpu(
    const Tensor& a,
    const Tensor& b,
    Tensor c,
    double alpha,
    double beta,
    bool transpose_b) {
  // The native GEMM stub uses column-major storage. Swap A and B to operate
  // on row-major tensors, so transposing B selects native transa.
  const auto transa = transpose_b
      ? torch::headeronly::TransposeType::Transpose
      : torch::headeronly::TransposeType::NoTranspose;
  const auto m = transpose_b ? b.size(0) : b.size(1);
  cpublas::gemm_no_downcast_stub(
      torch::headeronly::DeviceType::CPU,
      torch::headeronly::ScalarType::Float,
      transa,
      torch::headeronly::TransposeType::NoTranspose,
      m,
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

void my_cpublas_brgemm_release() {
  cpublas::brgemm_release(true);
}

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def(
      "my_cpublas_brgemm_dispatch(Tensor a, Tensor b, Tensor(a!) c, bool add_c, bool is_vnni) -> Tensor(a!)");
  m.def(
      "my_cpublas_gemm_no_downcast_stub_cpu(Tensor a, Tensor b, Tensor(a!) c, float alpha, float beta, bool transpose_b) -> Tensor(a!)");
  m.def("my_cpublas_could_pack(Tensor input) -> bool");
  m.def("my_cpublas_brgemm_release() -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CompositeExplicitAutograd, m) {
  m.impl("my_cpublas_brgemm_dispatch", TORCH_BOX(&my_cpublas_brgemm_dispatch));
  m.impl(
      "my_cpublas_gemm_no_downcast_stub_cpu",
      TORCH_BOX(&my_cpublas_gemm_no_downcast_stub_cpu));
  m.impl("my_cpublas_could_pack", TORCH_BOX(&my_cpublas_could_pack));
  m.impl("my_cpublas_brgemm_release", TORCH_BOX(&my_cpublas_brgemm_release));
}
