#include <ATen/ATen.h>

#include <torch/custom_class.h>

#include <ATen/native/ao_sparse/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/ao_sparse/quantized/cpu/packed_params.h>
#include <ATen/native/ao_sparse/quantized/cpu/qnnpack_utils.h>

namespace ao {
namespace sparse {
torch::class_<LinearPackedParamsBase> register_linear_params() {
  
#ifdef USE_FBGEMM
                if (at::globalContext().qEngine() == at::QEngine::FBGEMM) {
                  if (weight.scalar_type() == at::kQInt8) {
                    return PackedLinearWeight::prepack(
                        weight,
                        bias,
                        out_features_block_size,
                        in_features_block_size);
                  } else {
                    TORCH_CHECK(
                        false,
                        "Unsupported data type",
                        c10::toString(weight.scalar_type()),
                        " in serialized LinearPackedParams object!");
                  }
                }
#endif // USE_FBGEMM
#ifdef USE_PYTORCH_QNNPACK
                if (at::globalContext().qEngine() == at::QEngine::QNNPACK) {
                  if (weight.scalar_type() == at::kQInt8) {
                    return PackedLinearWeightQnnp::prepack(
                        weight,
                        bias,
                        out_features_block_size,
                        in_features_block_size);
                  } else {
                    TORCH_CHECK(
                        false,
                        "Unsupported data type",
                        c10::toString(weight.scalar_type()),
                        " in serialized LinearPackedParams object!");
                  }
                }
#endif // USE_FBGEMM
                TORCH_CHECK(false, "Unknown qengine");
              });
  return register_linear_params;
}

namespace {
static auto linear_params = register_linear_params();
}  // namespace

}}  // namespace ao::sparse
