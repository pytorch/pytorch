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

// Split SparseLengthsSumSparse into SparseLengthsSumSparseLookup +
// SparseLengthsSum
TORCH_API void splitSparseLengthsSumSparse(NetDef* net, const Workspace& ws);

struct OnnxifiTransformerOptions final : public BackendTransformOptions {
  explicit OnnxifiTransformerOptions() : BackendTransformOptions() {}

  // Pass serialized onnx model if true, otherwise pass serialized c2 model
  bool use_onnx{false};

  // Whether to adjust batch at the outputs or not
  bool adjust_batch{true};

  // Whether to lower model blob by blob
  bool load_model_by_blob{false};

  // Whether to enforce fp32 inputs into fp16.
  bool enforce_fp32_inputs_into_fp16{false};

  // Whether to combine fp32 batched inputs into one tensor and convert it to
  // fp16 or not
  bool merge_fp32_inputs_into_fp16{false};

  // Whether the net has been ssaRewritten
  bool predictor_net_ssa_rewritten{false};

  // Inference timeout
  int timeout{0};

  // Mapping of batch sizes to shape infos
  std::unordered_map<int, ShapeInfoMap> shape_hints_per_bs;
};

class TORCH_API OnnxifiOptionHelper final {
 public:
  OnnxifiOptionHelper();

  // Set Onnxifi option
  bool setOnnxifiOption(const std::string& option, const std::string& value);

  //  Get Onnxifi option
  std::string getOnnxifiOption(const std::string& option);

 private:
  // Pointer to loaded onnxifi library
  onnxifi_library* lib_{nullptr};
};

class TORCH_API OnnxifiTransformer final : public BackendTransformerBase {
 public:
  explicit OnnxifiTransformer(const OnnxifiTransformerOptions& opts);
  ~OnnxifiTransformer() override;

  void transform(
      Workspace* ws,
      NetDef* pred_net,
      const std::vector<std::string>& weight_names,
      const ShapeInfoMap& shape_hints,
      const std::unordered_set<int>& blocklisted_ops) override;

  // Query whether an operator is supported by passing C2 protobuf
  bool supportOpC2(
      const caffe2::OperatorDef& op,
      const ShapeInfoMap& shape_hints,
      const std::unordered_set<std::string>& weights,
      const std::unordered_set<int>& blocklisted_ops,
      onnxBackendID backend_id) const;

  // Determine backend id
  std::vector<onnxBackendID> getBackendId();

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
      ShapeInfoMap* shape_hints_max_bs,
      const std::unordered_map<int, ShapeInfoMap>& shape_hints_per_bs);

  // Convert a cutoff subgraph net to an Onnxifi op
  caffe2::NetDef SubnetToOnnxifiOpViaC2(
      const caffe2::NetDef& net,
      const std::unordered_set<std::string>& weights_in_ws,
      const ShapeInfoMap& shape_hints_max_bs,
      const std::unordered_map<int, ShapeInfoMap>& shape_hints_per_bs);

  // Check that output shape hints are present to ensure we can pass them to
  // OnnxifiOp
  bool canPassOutputShapeHintsPerBs(
      const OperatorDef& op,
      const std::unordered_map<int, ShapeInfoMap>& shape_hints_per_bs) const;

  // We already have all the ops and external inputs and outputs!
  OperatorDef buildOnnxifiOp(
      const std::string& onnx_model_str,
      const std::unordered_set<std::string>& initialization_list,
      const std::vector<std::string>& external_inputs,
      const std::vector<std::string>& external_outputs,
      const ShapeInfoMap& shape_hints_max_bs,
      const std::unordered_map<int, ShapeInfoMap>& shape_hints_per_bs);

  // Transform by passing C2 proto to backend
  NetDef TransformViaC2(
      NetDef* pred_net,
      const std::unordered_set<std::string>& weights,
      const std::unordered_set<int>& blocklisted_ops,
      const ShapeInfoMap& shape_hints_max_bs,
      const std::unordered_map<int, ShapeInfoMap>& shape_hints_per_bs);

  // Transform by passing ONNX proto to backend
  NetDef TransformViaOnnx(
      Workspace* ws,
      NetDef* pred_net,
      const std::unordered_set<std::string>& weights,
      const std::unordered_set<int>& blocklisted_ops,
      ShapeInfoMap* shape_hints_max_bs,
      const std::unordered_map<int, ShapeInfoMap>& shape_hints_per_bs);

  // Query whether an operator is supported by passing ONNX protobuf
  bool supportOpOnnx(
      const caffe2::OperatorDef& op,
      onnx::OnnxExporter* exporter,
      const std::unordered_set<int>& blocklisted_ops,
      onnxBackendID backend_id) const;

  // Tie the output of Gather to the scalar weight input of the
  // SparseLengthsWeighted* and SparseLengthsSumSparseLookup (which is split
  // from the SparseLengthsWeighted*Sparse) ops. If the latter is disabled,
  // disable the former too.
  void tieGatherAndSparseLengthsWeightedSumOps(
      const NetDef& net,
      const ShapeInfoMap& shape_hints,
      const std::unordered_set<std::string>& weights,
      std::unordered_set<int>* blocklisted_ops) const;

  // For net with partitioning info, blocklist ops that are supposed to run on
  // CPU, whose partition info will contain empty device_id list.
  void blocklistCpuPartition(
      const NetDef& net,
      std::unordered_set<int>* blocklisted_ops) const;

  // Rule based filtering
  void applyFilteringRules(
      const NetDef& net,
      const ShapeInfoMap& shape_hints,
      const std::unordered_set<std::string>& weights,
      std::unordered_set<int>* blocklisted_ops) const;

  // Extract partition info from the original net
  void extractPartitionInfo(const NetDef& net);

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

  // Partition info
  std::vector<PartitionInfo> partition_infos_;
};
} // namespace caffe2
