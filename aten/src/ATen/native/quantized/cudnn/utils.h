#pragma once
/*
This file contains some of the auxiliary functions used by both Conv.cpp & Linear.cpp
*/

#ifdef USE_CUDA
#include <ATen/cuda/CUDAConfig.h>  // for the definition of AT_CUDNN_ENABLED

#if AT_CUDNN_ENABLED()

#include <ATen/native/cudnn/Macros.h>
#include <c10/util/ArrayRef.h>
#include <cudnn_frontend.h>

#if HAS_CUDNN_V8()

namespace cudnn_utils {

inline uint8_t getAlignment(const at::Tensor &t) {
  // alignment are in bytes
  uint8_t alignment = 1;
  uintptr_t address = reinterpret_cast<uintptr_t>(t.data_ptr());
  while (address % alignment == 0 && alignment < 16) alignment *= 2;
  return alignment;
}

inline cudnn_frontend::Tensor getTensorDescriptor(const at::Tensor &t, int64_t id, uint8_t alignment) {
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

inline cudnn_frontend::Tensor getTensorDescriptor(const c10::IntArrayRef& shape, const c10::IntArrayRef& strides, cudnnDataType_t cudnn_dtype, int64_t id, uint8_t alignment) {
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
inline cudnn_frontend::PointWiseDesc_v8 getPointWiseMulDescriptor(cudnnDataType_t dataType) {
  return cudnn_frontend::PointWiseDescBuilder()
    .setMode(cudnnPointwiseMode_t::CUDNN_POINTWISE_MUL)
    .setMathPrecision(dataType)
    .build();
}

// TODO: there is a table from input dtype to operator dtype, we can derive
// the operator dtype based on input dtype
inline cudnn_frontend::PointWiseDesc_v8 getPointWiseAddDescriptor(cudnnDataType_t dataType) {
  return cudnn_frontend::PointWiseDescBuilder()
    .setMode(cudnnPointwiseMode_t::CUDNN_POINTWISE_ADD)
    .setMathPrecision(dataType)
    .build();
}

// TODO: there is a table from input dtype to operator dtype, we can derive
// the operator dtype based on input dtype
inline cudnn_frontend::PointWiseDesc_v8 getPointWiseReluDescriptor(cudnnDataType_t dataType) {
  return cudnn_frontend::PointWiseDescBuilder()
    .setMode(cudnnPointwiseMode_t::CUDNN_POINTWISE_RELU_FWD)
    .setMathPrecision(dataType)
    .build();
}


inline void filterEngineConfigs(
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


inline cudnn_frontend::ExecutionPlan
get_execplan_from_heuristics_else_fall_back(cudnn_frontend::OperationGraph&& opGraph, cudnnHandle_t handle_) {
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

}

#endif  // HAS_CUDNN_V8
#endif  // AT_CUDNN_ENABLED
#endif  // USE_CUDA
