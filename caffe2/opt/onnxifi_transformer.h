#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "onnx/onnx_pb.h"

#include "caffe2/core/common.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/workspace.h"
#include "caffe2/onnx/onnxifi_init.h"
#include "caffe2/proto/caffe2.pb.h"

namespace caffe2 {
namespace onnx {
class OnnxExporter;
}

class CAFFE2_API OnnxifiTransformer {
 public:
  explicit OnnxifiTransformer(bool debug);

  void Transform(
      Workspace* ws,
      NetDef* pred_net,
      const std::unordered_map<std::string, TensorShape>& shape_hints);

 private:
  caffe2::NetDef SubnetToOnnxifiOp(
      const caffe2::NetDef& net,
      Workspace* ws,
      onnx::OnnxExporter* exporter,
      std::unordered_map<std::string, TensorShape>* shape_hints);

  OperatorDef BuildOnnxifiOp(
      const std::string& onnx_model_str,
      const std::unordered_map<std::string, std::vector<int>>&
          output_size_hints,
      const std::unordered_set<std::string>& initialization_list,
      const caffe2::NetDef& net);

  CaffeMap<std::string, TensorShape> SsaRewriteAndMapNames(
      Workspace* ws,
      NetDef* pred_net,
      const std::unordered_map<std::string, TensorShape>& input_shape_hints);

  // Dump onnx model for debugging
  bool debug_{false};

  // Pointer to loaded onnxifi library
  onnxifi_library* lib_{nullptr};

  // Number of backends
  size_t num_backends_{0};

  // Backned IDs
  std::vector<onnxBackendID> backend_ids_;
  // Input mapping
  std::unordered_map<std::string, std::string> input_mapping_;
};
} // namespace caffe2
