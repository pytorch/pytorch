#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Config.h>
#include <ATen/Context.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/zendnn/Matmul.h>
#include <ATen/native/zendnn/ZenDNN_utils.hpp>
#include <ATen/record_function.h>

#if AT_ZENDNN_ENABLED()
#include <zendnnl.hpp>
namespace at::native {

using namespace zendnnl::lowoha;
void zendnn_baddbmm(
    const Tensor& self,
    const Tensor& batch1,
    const Tensor& batch2,
    float beta,
    float alpha) {
  RECORD_FUNCTION(
      "zendnn::zendnn_baddbmm",
      std::vector<c10::IValue>({batch1, batch2, self}));

  Tensor b1 = batch1;
  Tensor b2 = batch2;
  // Infer matrix dimensions from 3D inputs:
  // [B, M, K] x [B, K, N] -> [B, M, N]
  const int64_t M = b1.size(1);
  const int64_t N = b2.size(2);
  const int64_t K = b1.size(2);

  // Check if a 3D tensor is transposed (transposed version of a contiguous
  // tensor) in the last two dimensions.
  // For a transposed tensor
  // [B, M, K] -> [B, K, M]:
  // - stride[0] should be M*K (batch stride unchanged)
  // - stride[1] should be 1 (innermost dimension after transpose)
  // - stride[2] should be M (step size for original rows, now columns)
  auto is_transposed = [](const Tensor& t) {
    const auto sizes = t.sizes();
    const auto strides = t.strides();
    return strides[0] == sizes[1] * sizes[2] && strides[1] == 1 &&
        strides[2] == sizes[1];
  };

  // check if tensor is transposed
  bool transa = is_transposed(b1);
  bool transb = is_transposed(b2);

  // make a copy of tensor when tensor is neither contiguous nor transposed
  b1 = (transa || b1.is_contiguous()) ? b1 : b1.contiguous();
  b2 = (transb || b2.is_contiguous()) ? b2 : b2.contiguous();

  auto strideA = b1.strides();
  auto strideB = b2.strides();
  auto strideC = self.strides();

  const int64_t lda = transa ? strideA[2] : strideA[1];
  const int64_t ldb = transb ? strideB[2] : strideB[1];
  const int64_t ldc = strideC[1];

  data_type_t out_type = get_zendnn_dtype(self);
  data_type_t inp_dtype = get_zendnn_dtype(b1);
  data_type_t wgt_dtype = get_zendnn_dtype(b2);

  TORCH_CHECK(
      (b1.scalar_type() == b2.scalar_type()),
      "zendnn_baddbmm: batch1 and batch2 data types should be same");

  data_types matmul_dtype;
  matmul_dtype.src = inp_dtype;
  matmul_dtype.wei = wgt_dtype;
  matmul_dtype.dst = out_type;
  matmul_dtype.bias = data_type_t::none;
  matmul_dtype.compute = data_type_t::none;

  lowoha_params params;
  params.dtypes = matmul_dtype;

  // Execute batched matmul directly for LoA path
  matmul_direct(
      'r',
      transa,
      transb,
      M,
      N,
      K,
      alpha,
      b1.data_ptr(),
      lda,
      b2.data_ptr(),
      ldb,
      nullptr,
      beta,
      self.data_ptr(),
      ldc,
      params,
      b1.size(0),
      b2.size(0));
  return;
}
} // namespace at::native

#endif // AT_ZENDNN_ENABLED()
