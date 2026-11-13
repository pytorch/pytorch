#include <torch/csrc/stable/cpublas.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>

namespace cpublas = torch::stable::cpublas;
using torch::stable::Tensor;

template <typename AType, typename BType, typename CType>
Tensor my_cpublas_brgemm(
    const Tensor& a,
    const Tensor& b,
    Tensor c,
    bool add_c) {
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
      false);
  return c;
}

Tensor my_cpublas_gemm_no_downcast_stub_default(
    const Tensor& a,
    const Tensor& b,
    Tensor c,
    double alpha,
    double beta) {
  // The native GEMM stub uses column-major storage. Swap A and B to operate
  // on row-major tensors, as the native brgemm fallback does.
  auto gemm = cpublas::gemm_no_downcast_stub.DEFAULT;
  gemm(
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

Tensor my_cpublas_gemm_no_downcast_stub_cpu(
    const Tensor& a,
    const Tensor& b,
    Tensor c,
    double alpha,
    double beta) {
  // The native GEMM stub uses column-major storage. Swap A and B to operate
  // on row-major tensors, as the native brgemm fallback does.
  cpublas::gemm_no_downcast_stub(
      torch::headeronly::DeviceType::CPU,
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
      "my_cpublas_brgemm_f32(Tensor a, Tensor b, Tensor(a!) c, bool add_c) -> Tensor(a!)");
  m.def(
      "my_cpublas_brgemm_f16(Tensor a, Tensor b, Tensor(a!) c, bool add_c) -> Tensor(a!)");
  m.def(
      "my_cpublas_brgemm_bf16(Tensor a, Tensor b, Tensor(a!) c, bool add_c) -> Tensor(a!)");
  m.def(
      "my_cpublas_brgemm_u8_u8(Tensor a, Tensor b, Tensor(a!) c, bool add_c) -> Tensor(a!)");
  m.def(
      "my_cpublas_brgemm_u8_i8(Tensor a, Tensor b, Tensor(a!) c, bool add_c) -> Tensor(a!)");
  m.def(
      "my_cpublas_brgemm_i8_i8(Tensor a, Tensor b, Tensor(a!) c, bool add_c) -> Tensor(a!)");
  m.def(
      "my_cpublas_brgemm_f8e4m3(Tensor a, Tensor b, Tensor(a!) c, bool add_c) -> Tensor(a!)");
  m.def(
      "my_cpublas_brgemm_f8e5m2(Tensor a, Tensor b, Tensor(a!) c, bool add_c) -> Tensor(a!)");
  m.def(
      "my_cpublas_gemm_no_downcast_stub_default(Tensor a, Tensor b, Tensor(a!) c, float alpha, float beta) -> Tensor(a!)");
  m.def(
      "my_cpublas_gemm_no_downcast_stub_cpu(Tensor a, Tensor b, Tensor(a!) c, float alpha, float beta) -> Tensor(a!)");
  m.def("my_cpublas_could_pack(Tensor input) -> bool");
  m.def("my_cpublas_brgemm_release() -> bool");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CompositeExplicitAutograd, m) {
  m.impl(
      "my_cpublas_brgemm_f32",
      TORCH_BOX((&my_cpublas_brgemm<float, float, float>)));
  m.impl(
      "my_cpublas_brgemm_f16",
      TORCH_BOX((&my_cpublas_brgemm<
                 torch::headeronly::Half,
                 torch::headeronly::Half,
                 float>)));
  m.impl(
      "my_cpublas_brgemm_bf16",
      TORCH_BOX((&my_cpublas_brgemm<
                 torch::headeronly::BFloat16,
                 torch::headeronly::BFloat16,
                 float>)));
  m.impl(
      "my_cpublas_brgemm_u8_u8",
      TORCH_BOX((&my_cpublas_brgemm<unsigned char, unsigned char, int32_t>)));
  m.impl(
      "my_cpublas_brgemm_u8_i8",
      TORCH_BOX((&my_cpublas_brgemm<unsigned char, signed char, int32_t>)));
  m.impl(
      "my_cpublas_brgemm_i8_i8",
      TORCH_BOX((&my_cpublas_brgemm<signed char, signed char, int32_t>)));
  m.impl(
      "my_cpublas_brgemm_f8e4m3",
      TORCH_BOX((&my_cpublas_brgemm<
                 torch::headeronly::Float8_e4m3fn,
                 torch::headeronly::Float8_e4m3fn,
                 float>)));
  m.impl(
      "my_cpublas_brgemm_f8e5m2",
      TORCH_BOX((&my_cpublas_brgemm<
                 torch::headeronly::Float8_e5m2,
                 torch::headeronly::Float8_e5m2,
                 float>)));
  m.impl(
      "my_cpublas_gemm_no_downcast_stub_default",
      TORCH_BOX(&my_cpublas_gemm_no_downcast_stub_default));
  m.impl(
      "my_cpublas_gemm_no_downcast_stub_cpu",
      TORCH_BOX(&my_cpublas_gemm_no_downcast_stub_cpu));
  m.impl("my_cpublas_could_pack", TORCH_BOX(&my_cpublas_could_pack));
  m.impl("my_cpublas_brgemm_release", TORCH_BOX(&my_cpublas_brgemm_release));
}
