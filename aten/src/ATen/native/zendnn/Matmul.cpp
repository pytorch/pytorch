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

  // Infer matrix dimensions from 3D inputs:
  // [B, M, K] x [B, K, N] -> [B, M, N]
  const int64_t M = batch1.size(1);
  const int64_t N = batch2.size(2);
  const int64_t K = batch1.size(2);

  // Leading dimensions depend on memory layout; swap if transposed.
  const bool a_is_contig = batch1.is_contiguous();
  const bool b_is_contig = batch2.is_contiguous();
  const int64_t lda = a_is_contig ? K : M;
  const int64_t ldb = b_is_contig ? N : K;
  const int64_t ldc = N;
  const bool transa = a_is_contig ? false : true;
  const bool transb = b_is_contig ? false : true;

  data_type_t out_type = get_zendnn_dtype(self);
  data_type_t inp_dtype = get_zendnn_dtype(batch1);
  data_type_t wgt_dtype = get_zendnn_dtype(batch2);

  TORCH_CHECK(
      (batch1.scalar_type() == batch2.scalar_type()),
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
      batch1.data_ptr(),
      lda,
      batch2.data_ptr(),
      ldb,
      nullptr,
      beta,
      self.data_ptr(),
      ldc,
      params,
      batch1.size(0),
      batch2.size(0));
  return;
}
} // namespace at::native

#endif // AT_ZENDNN_ENABLED()
