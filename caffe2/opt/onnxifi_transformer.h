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
#include "caffe2/proto/caffe2_pb.h"

namespace caffe2 {
namespace onnx {
class OnnxExporter;
}

class CAFFE2_API OnnxifiTransformer {
 public:
  explicit OnnxifiTransformer(bool infer_shapes, bool debug);

  void Transform(
      Workspace* ws,
      NetDef* pred_net,
      const std::vector<std::string>& external_inputs,
      const std::unordered_map<std::string, TensorShape>& shape_hints,
      const std::unordered_set<int>& blacklisted_ops);

  const std::unordered_map<std::string, std::string>& input_mapping() const {
    return input_mapping_;
  }

  const std::unordered_map<std::string, std::string>& reverse_input_mapping()
      const {
    return reverse_input_mapping_;
  }

 private:
  // Since we create new tensors during the conversion process, we actually need
  // into inject them into the original workspace
  caffe2::NetDef SubnetToOnnxifiOp(
      const caffe2::NetDef& net,
      const std::unordered_set<std::string>& weights_in_ws,
      Workspace* ws,
      onnx::OnnxExporter* exporter,
      std::unordered_map<std::string, TensorShape>* shape_hints);

  OperatorDef BuildOnnxifiOp(
      const std::string& onnx_model_str,
      const std::unordered_map<std::string, TensorShape>& output_size_hints,
      const std::unordered_set<std::string>& initialization_list,
      const caffe2::NetDef& net);

  CaffeMap<std::string, TensorShape> SsaRewriteAndMapNames(
      Workspace* ws,
      NetDef* pred_net,
      const std::unordered_map<std::string, TensorShape>& input_shape_hints);

  // Run shape inference
  bool infer_shapes_{false};

  // Dump onnx model for debugging
  bool debug_{false};

  // Pointer to loaded onnxifi library
  onnxifi_library* lib_{nullptr};

  // Number of backends
  size_t num_backends_{0};

  // Backned IDs
  std::vector<onnxBackendID> backend_ids_;

  // Input mapping of input name -> original input name
  std::unordered_map<std::string, std::string> input_mapping_;

  // Input mapping of orignal input name -> input name
  std::unordered_map<std::string, std::string> reverse_input_mapping_;
};
} // namespace caffe2
