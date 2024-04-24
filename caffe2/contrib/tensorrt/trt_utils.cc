#include "caffe2/contrib/tensorrt/trt_utils.h"

#include <NvOnnxParser.h>

namespace caffe2 {
namespace tensorrt {
std::shared_ptr<nvinfer1::ICudaEngine> BuildTrtEngine(
    const std::string& onnx_model_str,
    TrtLogger* logger,
    size_t max_batch_size,
    size_t max_workspace_size,
    bool debug_builder) {
  auto trt_builder = TrtObject(nvinfer1::createInferBuilder(*logger));
#if defined(TENSORRT_VERSION_MAJOR) && (TENSORRT_VERSION_MAJOR >= 6)
  auto trt_builder_cfg = TrtObject(trt_builder->createBuilderConfig());
  // TensorRTOp doesn't support dynamic shapes yet
  auto trt_network = TrtObject(trt_builder->createNetworkV2(
      1U << static_cast<uint32_t>(nvinfer1::
      NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
#else
  auto trt_network = TrtObject(trt_builder->createNetwork());
#endif
  auto trt_parser =
      TrtObject(nvonnxparser::createParser(*trt_network, *logger));
  auto status = trt_parser->parse(onnx_model_str.data(), onnx_model_str.size());
  if (!status) {
    const auto num_errors = trt_parser->getNbErrors();
    if (num_errors > 0) {
      const auto* error = trt_parser->getError(num_errors - 1);
      CAFFE_THROW(
          "TensorRTTransformer ERROR: ",
          error->file(),
          ":",
          error->line(),
          " In function ",
          error->func(),
          ":\n",
          "[",
          static_cast<int>(error->code()),
          "] ",
          error->desc());
    } else {
      CAFFE_THROW("TensorRTTransformer Unknown Error");
    }
  }
  trt_builder->setMaxBatchSize(max_batch_size);
#if defined(TENSORRT_VERSION_MAJOR) && (TENSORRT_VERSION_MAJOR >= 6)
  trt_builder_cfg->setMaxWorkspaceSize(max_workspace_size);
  if (debug_builder) {
    trt_builder_cfg->setFlag(nvinfer1::BuilderFlag::kDEBUG);
  }
  trt_builder_cfg->setDefaultDeviceType(nvinfer1::DeviceType::kGPU);
  return TrtObject(trt_builder->
      buildEngineWithConfig(*trt_network.get(), *trt_builder_cfg));
#else
  trt_builder->setMaxWorkspaceSize(max_workspace_size);
  trt_builder->setDebugSync(debug_builder);
  return TrtObject(trt_builder->buildCudaEngine(*trt_network.get()));
#endif
}
} // namespace tensorrt
} // namespace caffe2
