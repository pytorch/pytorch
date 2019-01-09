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

struct OnnxifiTransformerOptions {
  // Run shape inference
  bool infer_shapes{false};
  // Dump onnx model for debugging
  bool debug{false};
  // Pass serialized onnx model if true, otherwise pass serialized c2 model
  bool use_onnx{true};
};

class CAFFE2_API OnnxifiTransformer final {
 public:
  explicit OnnxifiTransformer(const OnnxifiTransformerOptions& opts);
  ~OnnxifiTransformer();

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
  caffe2::NetDef SubnetToOnnxifiOpViaOnnx(
      const caffe2::NetDef& net,
      const std::unordered_set<std::string>& weights_in_ws,
      Workspace* ws,
      onnx::OnnxExporter* exporter,
      std::unordered_map<std::string, TensorShape>* shape_hints);

  // Convert a cutoff subgraph net to an Onnxifi op
  caffe2::NetDef SubnetToOnnxifiOpViaC2(
      const caffe2::NetDef& net,
      const std::unordered_set<std::string>& weights_in_ws,
      const std::unordered_map<std::string, TensorShape>& shape_hints);

  // We already have all the ops and external inputs and outputs!
  OperatorDef BuildOnnxifiOp(
      const std::string& onnx_model_str,
      const std::unordered_map<std::string, TensorShape>& output_size_hints,
      const std::unordered_set<std::string>& initialization_list,
      const caffe2::NetDef& net);

  CaffeMap<std::string, TensorShape> SsaRewriteAndMapNames(
      Workspace* ws,
      NetDef* pred_net,
      const std::unordered_map<std::string, TensorShape>& input_shape_hints);

  // Transform by passing C2 proto to backend
  NetDef TransformViaC2(
      NetDef* pred_net,
      const std::unordered_set<std::string>& weights,
      const std::unordered_set<int>& blacklisted_ops,
      const std::unordered_map<std::string, TensorShape>& shape_hints);

  // Transform by passing ONNX proto to backend
  NetDef TransformViaOnnx(
      Workspace* ws,
      NetDef* pred_net,
      const std::unordered_set<std::string>& weights,
      const std::unordered_set<int>& blacklisted_ops,
      std::unordered_map<std::string, TensorShape>* shape_hints);

  // Options
  OnnxifiTransformerOptions opts_;

  // Pointer to loaded onnxifi library
  onnxifi_library* lib_{nullptr};

  // Number of backends
  size_t num_backends_{0};

  // backend idx
  int idx_{0};

  // Backned IDs
  std::vector<onnxBackendID> backend_ids_;

  // Input mapping of input name -> original input name
  std::unordered_map<std::string, std::string> input_mapping_;

  // Input mapping of orignal input name -> input name
  std::unordered_map<std::string, std::string> reverse_input_mapping_;
};
} // namespace caffe2
