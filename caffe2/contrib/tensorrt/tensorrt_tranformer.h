#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "caffe2/core/common.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/workspace.h"
#include "caffe2/onnx/onnx_exporter.h"
#include "caffe2/proto/caffe2_pb.h"
#include "onnx/onnx_pb.h"

namespace caffe2 {

CAFFE2_API void BuildInitializationList(
    Workspace* ws,
    ::ONNX_NAMESPACE::GraphProto* g,
    std::unordered_set<std::string>* initialization_list);

class CAFFE2_API TensorRTTransformer {
 public:
  TensorRTTransformer(
      size_t max_batch_size,
      size_t max_workspace_size,
      int verbosity,
      bool debug_builder,
      bool build_serializable_op = false)
      : build_serializable_op_(build_serializable_op),
        max_batch_size_(max_batch_size),
        max_workspace_size_(max_workspace_size),
        verbosity_(verbosity),
        debug_builder_(debug_builder) {}

  OperatorDef BuildTrtOp(
      const std::string& onnx_model_str,
      const std::unordered_map<std::string, std::vector<int>>&
          output_size_hints);

  void Transform(
      Workspace* ws,
      NetDef* pred_net,
      const std::unordered_map<std::string, TensorShape>& shape_hints);

 private:
  caffe2::NetDef SubnetToTrtOp(
      const caffe2::NetDef& net,
      Workspace* ws,
      onnx::OnnxExporter* exporter,
      std::unordered_map<std::string, TensorShape>* shape_hints);

  void AddTrtOptions(
      caffe2::OperatorDef* op,
      const std::unordered_map<std::string, std::vector<int>>&
          output_size_hints);

  // A lazy version of Trt op building function, where instead of invoking the
  // trt build engine and serialize the trt runtime, we just attach the
  // serialized trt model string. The runtime will be built when trt op is
  // constructed, during which the weights will be pulled from the workspace.
  // The benefit of doing so is that we can avoid serialize/deserialize the
  // weights across OperatorDef.
  OperatorDef BuildTrtOpLazy(
      const std::string& onnx_model_str,
      const std::unordered_map<std::string, std::vector<int>>&
          output_size_hints,
      const std::unordered_set<std::string>& initialization_list,
      const caffe2::NetDef& net);

  CaffeMap<std::string, TensorShape> SsaRewriteAndMapNames(
      Workspace* ws,
      NetDef* pred_net,
      const std::unordered_map<std::string, TensorShape>& input_shape_hints);

  // Prune the unreferenced weights in original workspace to save memory
  void PruneUnusedWeights(Workspace* ws, const NetDef& pred_net);

  // Input mapping
  std::unordered_map<std::string, std::string> input_mapping_;

  // Generate serializable trt op or defer the onnx->trt process to ctor of the
  // Trt op
  bool build_serializable_op_{true};

  // TensorRT params
  size_t max_batch_size_{50};
  size_t max_workspace_size_{1024 * 1024 * 2};
  int verbosity_{2};
  bool debug_builder_{false};
};
} // namespace caffe2
