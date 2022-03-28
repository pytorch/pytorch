#pragma once
/*
This file contains some of the auxiliary functions used by both Conv.cpp & Linear.cpp (introduced in a later PR)
*/

#ifdef USE_CUDA
#include <ATen/cuda/CUDAConfig.h>  // for the definition of AT_CUDNN_ENABLED

#if AT_CUDNN_ENABLED()

#include <ATen/native/cudnn/Macros.h>

#if HAS_CUDNN_V8()

#include <ATen/cudnn/Types.h>
#include <ATen/Tensor.h>
#include <ATen/native/quantized/packed_params.h>
#include <c10/core/QScheme.h>
#include <c10/util/ArrayRef.h>
#include <cudnn_frontend.h>

template <int kSpatialDim = 2>
struct TORCH_API PackedConvWeightCudnn : public ConvPackedParamsBase<kSpatialDim> {
  PackedConvWeightCudnn(
      at::Tensor orig_weight,
      c10::optional<at::Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> output_padding,
      torch::List<int64_t> dilation,
      int64_t groups,
      bool transpose,
      c10::QScheme q_scheme)
      : orig_weight_(std::move(orig_weight)),
        bias_(std::move(bias)),
        stride_(std::move(stride)),
        padding_(std::move(padding)),
        output_padding_(std::move(output_padding)),
        dilation_(std::move(dilation)),
        groups_(groups),
        transpose_(transpose),
        q_scheme_(q_scheme) {}

  at::Tensor apply(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point) override;

  at::Tensor apply_relu(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point) override;

  at::Tensor apply_dynamic(
    const at::Tensor& input,
    bool reduce_range) {
    TORCH_CHECK(false, "apply_dynamic is currently not reported");
  }

  at::Tensor apply_dynamic_relu(
    const at::Tensor& input,
    bool reduce_range) {
    TORCH_CHECK(false, "apply_dynamic_relu is currently not reported");
  }

  std::tuple<at::Tensor, c10::optional<at::Tensor>> unpack() override;

  static c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> prepack(
      at::Tensor weight,
      c10::optional<at::Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> output_padding,
      torch::List<int64_t> dilation,
      int64_t groups,
      bool transpose);

  const float* GetBiasData(at::Tensor* bias);

  torch::List<int64_t> stride() const override {
    return stride_;
  }

  torch::List<int64_t> padding() const override {
    return padding_;
  }

  torch::List<int64_t> output_padding() const override {
    return output_padding_;
  }

  torch::List<int64_t> dilation() const override {
    return dilation_;
  }

  int64_t groups() const override {
    return groups_;
  }

  bool transpose() const override {
    return transpose_;
  }

 private:
  at::Tensor orig_weight_;
  c10::optional<at::Tensor> bias_;
  torch::List<int64_t> stride_;
  torch::List<int64_t> padding_;
  torch::List<int64_t> output_padding_;
  torch::List<int64_t> dilation_;
  int64_t groups_;
  bool transpose_;
  c10::QScheme q_scheme_;

  template <bool ReluFused>
  at::Tensor apply_impl(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point);

  template <bool ReluFused>
  void apply_impl_helper(
      const at::Tensor& quantized_output,
      const at::Tensor& input,
      double output_scale);
};

namespace cudnn_utils {
namespace {

uint8_t getAlignment(const at::Tensor &t) {
  // alignment are in bytes
  uint8_t alignment = 1;
  uintptr_t address = reinterpret_cast<uintptr_t>(t.data_ptr());
  while (address % alignment == 0 && alignment < 16) alignment *= 2;
  return alignment;
}

cudnn_frontend::Tensor getTensorDescriptor(const at::Tensor &t, int64_t id, uint8_t alignment) {
  auto shape = t.sizes();
  auto strides = t.strides();
  return cudnn_frontend::TensorBuilder()
    .setDim(shape.size(), shape.data())
    .setStrides(strides.size(), strides.data())
    .setId(id)
    .setAlignment(alignment)
    .setDataType(at::native::getCudnnDataType(t))
    .build();
}

cudnn_frontend::Tensor getTensorDescriptor(const c10::IntArrayRef& shape, const c10::IntArrayRef& strides, cudnnDataType_t cudnn_dtype, int64_t id, uint8_t alignment) {
  return cudnn_frontend::TensorBuilder()
    .setDim(shape.size(), shape.data())
    .setStrides(strides.size(), strides.data())
    .setId(id)
    .setAlignment(alignment)
    .setDataType(cudnn_dtype)
    .build();
}

// TODO: there is a table from input dtype to operator dtype, we can derive
// the operator dtype based on input dtype
cudnn_frontend::PointWiseDesc_v8 getPointWiseMulDescriptor(cudnnDataType_t dataType) {
  return cudnn_frontend::PointWiseDescBuilder()
    .setMode(cudnnPointwiseMode_t::CUDNN_POINTWISE_MUL)
    .setMathPrecision(dataType)
    .build();
}

// TODO: there is a table from input dtype to operator dtype, we can derive
// the operator dtype based on input dtype
cudnn_frontend::PointWiseDesc_v8 getPointWiseAddDescriptor(cudnnDataType_t dataType) {
  return cudnn_frontend::PointWiseDescBuilder()
    .setMode(cudnnPointwiseMode_t::CUDNN_POINTWISE_ADD)
    .setMathPrecision(dataType)
    .build();
}

// TODO: there is a table from input dtype to operator dtype, we can derive
// the operator dtype based on input dtype
cudnn_frontend::PointWiseDesc_v8 getPointWiseReluDescriptor(cudnnDataType_t dataType) {
  return cudnn_frontend::PointWiseDescBuilder()
    .setMode(cudnnPointwiseMode_t::CUDNN_POINTWISE_RELU_FWD)
    .setMathPrecision(dataType)
    .build();
}


void filterEngineConfigs(
  cudnn_frontend::EngineConfigList &from,
  cudnn_frontend::EngineConfigList &to,
  bool deterministic, bool allow_tf32, c10::ScalarType scalar_type)
{
  auto filter = [=](cudnnBackendDescriptor_t c) {
    if (deterministic) {
      if (cudnn_frontend::hasNumericalNote<CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC>(c)) return true;
    }
    if (scalar_type == at::kFloat || scalar_type == at::kChar || !allow_tf32) {
      if (cudnn_frontend::hasNumericalNote<CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS>(c)) return true;
      if (cudnn_frontend::hasNumericalNote<CUDNN_NUMERICAL_NOTE_TENSOR_CORE>(c)) return true;
    }
    return false;
  };
  cudnn_frontend::filter(from, to, filter);
}


cudnn_frontend::ExecutionPlan get_execplan_from_heuristics_else_fall_back(cudnn_frontend::OperationGraph&& opGraph, cudnnHandle_t handle_) {
  auto heuristics = cudnn_frontend::EngineHeuristicsBuilder()
    .setOperationGraph(opGraph)
    .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
    .build();

  // std::cout << "Heuristic has " << heuristics.getEngineConfigCount() << " configurations " << std::endl;
  auto& engine_config = heuristics.getEngineConfig(heuristics.getEngineConfigCount());

  // Try engine configs returned by the heuristics and pick up the first one that works.
  for (auto& ecfg : engine_config) {
    try {
      auto plan = cudnn_frontend::ExecutionPlanBuilder()
        .setHandle(handle_)
        .setEngineConfig(ecfg, opGraph.getTag())
        .build();
      return plan;
    } catch (cudnn_frontend::cudnnException& e) {
      continue;
    }
  }

  {
    auto total_engines = opGraph.getEngineCount();
    // std::cout << opGraph.describe() << " has " << total_engines << " engines." << std::endl;
    auto engine = cudnn_frontend::EngineBuilder().setGlobalEngineIdx(0).setOperationGraph(opGraph).build();
    // std::cout << engine.describe() << std::endl;

    auto engine_config = cudnn_frontend::EngineConfigBuilder().setEngine(engine).build();
    // std::cout << engine_config.describe() << std::endl;

    return cudnn_frontend::ExecutionPlanBuilder().setHandle(handle_).setEngineConfig(engine_config).build();
  }
}
} // anonymous
} // cudnn_utils

#endif  // HAS_CUDNN_V8
#endif  // AT_CUDNN_ENABLED
#endif  // USE_CUDA
