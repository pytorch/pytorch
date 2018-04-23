#include "caffe2/contrib/tensorrt/trt_utils.h"

#include <onnx2trt.hpp>

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
  auto importer = TrtObject(onnx2trt::createImporter(trt_network.get()));
  auto status =
      importer->import(onnx_model_str.data(), onnx_model_str.size(), false);
  if (status.is_error()) {
    CAFFE_THROW(
        "TensorRTTransformer ERROR: ",
        status.file(),
        ":",
        status.line(),
        " In function ",
        status.func(),
        ":\n",
        "[",
        status.code(),
        "] ",
        status.desc());
  }
  trt_builder->setMaxBatchSize(max_batch_size);
  trt_builder->setMaxWorkspaceSize(max_workspace_size);
  trt_builder->setDebugSync(debug_builder);
  return TrtObject(trt_builder->buildCudaEngine(*trt_network.get()));
}
} // namespace tensorrt
} // namespace caffe2
