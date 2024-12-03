#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Context.h>

#include <torch/custom_class.h>

#include <ATen/native/ao_sparse/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/ao_sparse/quantized/cpu/packed_params.h>
#include <ATen/native/ao_sparse/quantized/cpu/qnnpack_utils.h>

namespace ao::sparse {
int register_linear_params() {
  static auto register_linear_params =
      torch::selective_class_<LinearPackedParamsBase>(
          "sparse", TORCH_SELECTIVE_CLASS("LinearPackedParamsBase"))
          .def_pickle(
              [](const c10::intrusive_ptr<LinearPackedParamsBase>& params)
                  -> BCSRSerializationType { // __getstate__
                return params->serialize();
              },
              [](BCSRSerializationType state)
                  -> c10::intrusive_ptr<
                      LinearPackedParamsBase> { // __setstate__
#ifdef USE_FBGEMM
                if (at::globalContext().qEngine() == at::QEngine::FBGEMM) {
                  return PackedLinearWeight::deserialize(state);
                }
#endif // USE_FBGEMM
#ifdef USE_PYTORCH_QNNPACK
                if (at::globalContext().qEngine() == at::QEngine::QNNPACK) {
                  return PackedLinearWeightQnnp::deserialize(state);
                }
#endif // USE_FBGEMM
                TORCH_CHECK(false, "Unknown qengine");
              });
  // (1) we can't (easily) return the static initializer itself because it can have a different type because of selective build
  // (2) we can't return void and be able to call the function in the global scope
  return 0;
}

namespace {
[[maybe_unused]] static auto linear_params = register_linear_params();
} // namespace
}  // namespace ao::sparse
