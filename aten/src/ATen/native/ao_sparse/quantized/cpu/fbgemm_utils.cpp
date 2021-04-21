#include <ATen/ATen.h>

#include <torch/custom_class.h>

#include <ATen/native/ao_sparse/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/ao_sparse/quantized/cpu/packed_params.h>
#include <ATen/native/ao_sparse/quantized/cpu/qnnpack_utils.h>

torch::class_<SparseLinearPackedParamsBase> register_sparse_linear_params() {
  static auto register_linear_params =
      torch::class_<SparseLinearPackedParamsBase>(
          "sparse", "SparseLinearPackedParamsBase")
          .def_pickle(
              [](const c10::intrusive_ptr<SparseLinearPackedParamsBase>& params)
                  -> SerializationTypeSparseLinearPacked { // __getstate__
                return params->unpack();
              },
              [](SerializationTypeSparseLinearPacked state)
                  -> c10::intrusive_ptr<
                      SparseLinearPackedParamsBase> { // __setstate__
                at::Tensor weight;
                c10::optional<at::Tensor> bias;
                int64_t out_features_block_size, in_features_block_size;
                weight = std::move(std::get<0>(state));
                bias = std::move(std::get<1>(state));
                out_features_block_size = std::get<2>(state)[0];
                in_features_block_size = std::get<2>(state)[1];

#ifdef USE_FBGEMM
                if (at::globalContext().qEngine() == at::QEngine::FBGEMM) {
                  if (weight.scalar_type() == at::kQInt8) {
                    return SparsePackedLinearWeight::prepack(
                        weight,
                        bias,
                        out_features_block_size,
                        in_features_block_size);
                  } else {
                    TORCH_CHECK(
                        false,
                        "Unsupported data type",
                        c10::toString(weight.scalar_type()),
                        " in serialized SparseLinearPackedParams object!");
                  }
                }
#endif // USE_FBGEMM
#ifdef USE_PYTORCH_QNNPACK
                if (at::globalContext().qEngine() == at::QEngine::QNNPACK) {
                  if (weight.scalar_type() == at::kQInt8) {
                    return SparsePackedLinearWeightQnnp::prepack(
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
static auto sparse_linear_params = register_sparse_linear_params();
}
