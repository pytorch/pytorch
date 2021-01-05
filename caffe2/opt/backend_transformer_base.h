#pragma once

#include "caffe2/core/common.h"
#include "caffe2/core/workspace.h"
#include "caffe2/opt/bound_shape_inferencer.h"
#include "caffe2/proto/caffe2_pb.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace caffe2 {
namespace {
constexpr char kNetPos[] = "net_pos";
constexpr char kModelId[] = "model_id";
} // namespace

struct BackendTransformOptions {
  explicit BackendTransformOptions() : bound_shape_spec(0, 0) {}

  // Enable debugging by dumping more intermediate graphs
  bool debug{false};

  // Minimum number of ops to create a backend op. If the subgraph is too
  // small, it doesn't make sense to lower it to backend.
  size_t min_ops{1};

  // Bound shape spec
  BoundShapeSpec bound_shape_spec;
};

// Wrap TensorShape into TensorProto
TensorProto wrapShapeInfoIntoTensorProto(
    const std::string& name,
    const ShapeInfo& shape_info);

// Wrap Quantized TensorShape into QTensorProto
QTensorProto wrapShapeInfoIntoQTensorProto(
    const std::string& name,
    const ShapeInfo& shape_info);

// This class contains some common functions for backend lowering and graph
// cutting
class BackendTransformerBase {
 public:
  BackendTransformerBase() {}
  virtual ~BackendTransformerBase() {}

  const std::unordered_map<std::string, std::string>& input_mapping() const {
    return input_mapping_;
  }

  const std::unordered_map<std::string, std::string>& reverse_input_mapping()
      const {
    return reverse_input_mapping_;
  }

  virtual void transform(
      Workspace* ws,
      NetDef* pred_net,
      const std::vector<std::string>& weight_names,
      const ShapeInfoMap& shape_hints,
      const std::unordered_set<int>& blocklisted_ops) = 0;

  static void annotateOpIndex(NetDef* net);

  // Get model ID from the NetDef
  static std::string getModelId(const NetDef& net);

 protected:
  // add shape info to the net
  void addShapeToNet(NetDef& shape_net, const ShapeInfoMap& shape_hints) const;

  // Dump the net with shape info
  void dumpNet(
      const NetDef& pred_net,
      const ShapeInfoMap& map,
      const std::string& fname) const;

  // SSA rewrite the net and return name mapping
  ShapeInfoMap ssaRewriteAndMapNames(
      Workspace* ws,
      NetDef* pred_net,
      const ShapeInfoMap& input_shape_hints);

  // Do bound shape inference and collect shape infos
  ShapeInfoMap inferShapes(
      Workspace* ws,
      NetDef* pred_net,
      const ShapeInfoMap& shape_hints_mapped,
      const BoundShapeSpec& spec);

  // Input mapping of input name -> original input name
  std::unordered_map<std::string, std::string> input_mapping_;

  // Input mapping of original input name -> input name
  std::unordered_map<std::string, std::string> reverse_input_mapping_;
};
} // namespace caffe2
