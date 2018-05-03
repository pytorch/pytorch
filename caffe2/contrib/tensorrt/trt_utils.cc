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
  auto trt_network = TrtObject(trt_builder->createNetwork());
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
  trt_builder->setMaxWorkspaceSize(max_workspace_size);
  trt_builder->setDebugSync(debug_builder);
  return TrtObject(trt_builder->buildCudaEngine(*trt_network.get()));
}
} // namespace tensorrt
} // namespace caffe2
