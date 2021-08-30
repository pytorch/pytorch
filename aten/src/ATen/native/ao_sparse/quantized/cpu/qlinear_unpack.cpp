#include <ATen/ATen.h>
#include <torch/custom_class.h>

#include <ATen/cpp_custom_type_hack.h>
#include <ATen/native/ao_sparse/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/ao_sparse/quantized/cpu/packed_params.h>
#include <ATen/native/ao_sparse/quantized/cpu/qnnpack_utils.h>

namespace ao {
namespace sparse {
torch::class_<LinearPackedParamsBase> register_linear_params();

#ifdef USE_FBGEMM



  // TODO: uncomment once unpack is implemented for BCSRMatrix
  // int8_t* weight_ptr_int8 =
  //     reinterpret_cast<int8_t*>(weight_origin.data_ptr<c10::qint8>());
  // packW->unpack(weight_ptr_int8);
  std::vector<int64_t> block_pattern(
      {out_features_block_size_, in_features_block_size_});

  return std::make_tuple(weight_origin, bias_, std::move(block_pattern));
}

#endif // USE_FBGEMM




TORCH_LIBRARY_IMPL(sparse, QuantizedCPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("sparse::qlinear_unpack"),
      TORCH_FN(QLinearUnpackWeightInt8::run));
}
}  // namespace
}}  // namespace ao::sparse
