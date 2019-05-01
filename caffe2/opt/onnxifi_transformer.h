#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "onnx/onnx_pb.h"

#include "caffe2/core/operator.h"
#include "caffe2/onnx/onnxifi_init.h"
#include "caffe2/opt/backend_transformer_base.h"

namespace caffe2 {
namespace onnx {
class OnnxExporter;
}

struct OnnxifiTransformerOptions {
  explicit OnnxifiTransformerOptions() : bound_shape_spec(0, 0) {}

  // Dump onnx model for debugging
  bool debug{false};

  // Pass serialized onnx model if true, otherwise pass serialized c2 model
  bool use_onnx{false};

  // Whether to adjust batch at the ouptuts or not
  bool adjust_batch{true};

  // Minimum number of ops to create an onnxifi op. If the subgraph is too
  // small, it doesn't make sense to lower it to backend.
  size_t min_ops{1};

  // Bound shape spec
  BoundShapeSpec bound_shape_spec;
};

class CAFFE2_API OnnxifiTransformer final : public BackendTransformerBase {
 public:
  explicit OnnxifiTransformer(const OnnxifiTransformerOptions& opts);
  ~OnnxifiTransformer() override;

  void transform(
      Workspace* ws,
      NetDef* pred_net,
      const std::vector<std::string>& weight_names,
      const std::unordered_map<std::string, TensorShape>& shape_hints,
      const std::unordered_set<int>& blacklisted_ops) override;

 private:
  // Since we create new tensors during the conversion process, we actually need
  // into inject them into the original workspace
  // Since our onnx exporter uses std::unordered_map<std::string, TensorShape>
  // as lut, we need to include an extra copy of shape info and maintain them
  // together
  caffe2::NetDef SubnetToOnnxifiOpViaOnnx(
      const caffe2::NetDef& net,
      const std::unordered_set<std::string>& weights_in_ws,
      Workspace* ws,
      onnx::OnnxExporter* exporter,
      ShapeInfoMap* shape_hints);

  // Convert a cutoff subgraph net to an Onnxifi op
  caffe2::NetDef SubnetToOnnxifiOpViaC2(
      const caffe2::NetDef& net,
      const std::unordered_set<std::string>& weights_in_ws,
      const ShapeInfoMap& shape_hints);

  // We already have all the ops and external inputs and outputs!
  OperatorDef BuildOnnxifiOp(
      const std::string& onnx_model_str,
      const std::unordered_map<std::string, TensorShape>& output_size_hints,
      const std::unordered_set<std::string>& initialization_list,
      const std::vector<std::string>& external_inputs,
      const std::vector<std::string>& external_outputs,
      const std::unordered_map<int, std::string>& batch_pos_map);

  // Transform by passing C2 proto to backend
  NetDef TransformViaC2(
      NetDef* pred_net,
      const std::unordered_set<std::string>& weights,
      const std::unordered_set<int>& blacklisted_ops,
      const ShapeInfoMap& shape_hints);

  // Transform by passing ONNX proto to backend
  NetDef TransformViaOnnx(
      Workspace* ws,
      NetDef* pred_net,
      const std::unordered_set<std::string>& weights,
      const std::unordered_set<int>& blacklisted_ops,
      ShapeInfoMap* shape_hints);

  // Query whether an operator is supported by passing C2 protobuf
  bool supportOpC2(
      const caffe2::OperatorDef& op,
      const ShapeInfoMap& shape_hints,
      const std::unordered_set<int>& blacklisted_ops,
      onnxBackendID backend_id) const;

  // Query whether an operator is supported by passing ONNX protobuf
  bool supportOpOnnx(
      const caffe2::OperatorDef& op,
      onnx::OnnxExporter* exporter,
      const std::unordered_set<int>& blacklisted_ops,
      onnxBackendID backend_id) const;

  // Go through the inputs of the onnxifi subgraph and extract their batch size.
  // And use this info to hint how we can cut output batch size from
  // max_batch_size. The returning key/value is max_batch_size/input_name. For
  // example, when OnnxifiOp sees an output with batch size 100, it will lookup
  // the map with key=100. And if we then look into the batch size of the
  // corresponding input for its real batch size, say 50, and use that to shrink
  // the output tensor to real batch size.
  std::unordered_map<int, std::string> generateBatchPaddingHints(
      const NetDef& onnxifi_net,
      const ShapeInfoMap& shape_hints);

  // Tie the output of Gather to the scalar weight input of the
  // SparseLengthsWeighted* op. If the latter is disabled, disable the former
  // too.
  void tieGatherAndSparseLengthsWeightedSumOps(
      const NetDef& net,
      const ShapeInfoMap& shape_hints,
      std::unordered_set<int>* blacklisted_ops) const;

  // Rule based filtering
  void applyFilteringRules(
      const NetDef& net,
      const ShapeInfoMap& shape_hints,
      std::unordered_set<int>* blacklisted_ops) const;

  // Determine backend id
  void getBackendId();

  // Options
  OnnxifiTransformerOptions opts_;

  // Pointer to loaded onnxifi library
  onnxifi_library* lib_{nullptr};

  // Number of backends
  size_t num_backends_{0};

  // backend idx
  int idx_{0};

  // Number of Onnxifi Ops we build so far
  int onnxifi_op_id_{0};

  // Model id
  std::string model_id_;

  // Backned IDs
  std::vector<onnxBackendID> backend_ids_;

  // A cache for ONNX shape hints
  std::unordered_map<std::string, TensorShape> shape_hints_onnx_;
};
} // namespace caffe2
