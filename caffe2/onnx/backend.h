#pragma once

#include "caffe2/onnx/backend_rep.h"
#include "caffe2/onnx/device.h"
#include "caffe2/proto/caffe2.pb.h"
#include "onnx/onnx_pb.h"

#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace caffe2 {
namespace onnx {

using ::ONNX_NAMESPACE::AttributeProto;
using ::ONNX_NAMESPACE::GraphProto;
using ::ONNX_NAMESPACE::ModelProto;
using ::ONNX_NAMESPACE::NodeProto;
using ::ONNX_NAMESPACE::TensorProto;

// \brief This struct holds the converted ops after the onnx->c2 conversion.
// Notice that for RNN ops, it may create ops in init_net. Hence we have the
// `init_ops` field.
struct Caffe2Ops {
  ::google::protobuf::RepeatedPtrField<caffe2::OperatorDef> init_ops;
  ::google::protobuf::RepeatedPtrField<caffe2::OperatorDef> ops;
  ::google::protobuf::RepeatedPtrField<std::string> interface_blobs;
};

// A convenient class to query attributes of a NodeProto. Note that the
// NodeProto can not be modified during the query of OnnxAttributes object
class OnnxAttributes {
 public:
  OnnxAttributes(const NodeProto& node);

  bool HasAttribute(const std::string& key) const {
    return onnx_attrs_.count(key);
  }

  AttributeProto* AddRewrittenAttibute(const std::string& key) {
    auto tmp = rewritten_onnx_attrs_.emplace(key, AttributeProto());
    auto& attr = tmp.first->second;
    attr.set_name(key);
    return &attr;
  }

  ::google::protobuf::RepeatedPtrField<caffe2::Argument> OnnxAttrToCaffe2Arg(
      std::function<std::string(const std::string&)> mapper) const;

  // Get attribute given attribute name, specialied on data type T. Note that
  // the return value is copied
  template <typename T>
  T get(const std::string& key) const;

  template <typename T>
  T get(const std::string& key, const T& default_value) const {
    if (onnx_attrs_.count(key)) {
      return get<T>(key);
    } else {
      return default_value;
    }
  }

  const AttributeProto* remove(const std::string& key) {
    const AttributeProto* result = nullptr;
    auto iter = onnx_attrs_.find(key);
    if (iter != onnx_attrs_.end()) {
      result = iter->second;
      onnx_attrs_.erase(iter);
    }
    return result;
  }

 private:
  std::unordered_map<std::string, const AttributeProto*> onnx_attrs_;
  std::unordered_map<std::string, AttributeProto> rewritten_onnx_attrs_;
};

template <>
int64_t OnnxAttributes::get(const std::string& key) const;
template <>
float OnnxAttributes::get(const std::string& key) const;

template <>
::google::protobuf::RepeatedPtrField<std::string> OnnxAttributes::get(
    const std::string& key) const;

template <>
::google::protobuf::RepeatedField<::google::protobuf::int64>
OnnxAttributes::get(const std::string& key) const;

template <>
const TensorProto* OnnxAttributes::get(const std::string& key) const;

// convenient class for onnx node
struct OnnxNode {
  OnnxNode(const NodeProto& node_in) : node(node_in), attributes(node_in) {}

  const NodeProto& node;

  OnnxAttributes attributes;
};

class Caffe2Backend {
 public:
  Caffe2BackendRep* Prepare(
      const std::string& onnx_model_str,
      const std::string& device,
      const std::vector<Caffe2Ops>& extras);

  bool SupportOp(const std::string tyep) const;

  Caffe2Ops ConvertNode(const std::string& node_str, int opset_version);

 private:
  using SpecialOpConverter = Caffe2Ops (Caffe2Backend::*)(OnnxNode*, int);

  void OnnxToCaffe2(
      caffe2::NetDef* init_net,
      caffe2::NetDef* pred_net,
      const ModelProto& onnx_model,
      const std::string& device,
      int opset_version,
      bool include_initializers,
      const std::vector<Caffe2Ops>& extras);

  Caffe2Ops OnnxNodeToCaffe2Ops(
      const ModelProto& init_model,
      const ModelProto& pred_model,
      OnnxNode* onnx_node,
      int opset_version);

  std::unordered_set<std::string> AllNamesInGraph(const GraphProto& graph);

  void BuildTensorFillingOp(
      caffe2::OperatorDef* c2_op,
      const TensorProto& onnx_tensor,
      const std::string& name = "");

  Caffe2Ops CommonOnnxNodeToCaffe2Ops(OnnxNode* onnx_node, int opset_version);

  Caffe2Ops CreateConstant(OnnxNode* onnx_node, int opset_version);

  Caffe2Ops CreateConvPoolOpBase(OnnxNode* onnx_node, int opset_version);

  Caffe2Ops CreateReshape(OnnxNode* onnx_node, int opset_version);

  Caffe2Ops CreateGather(OnnxNode* onnx_node, int opset_version);

  Caffe2Ops CreateGemm(OnnxNode* onnx_node, int opset_version);

  Caffe2Ops CreatePad(OnnxNode* onnx_node, int opset_version);

  Caffe2Ops CreateConcat(OnnxNode* onnx_node, int opset_version);

  Caffe2Ops CreateLogSoftmax(OnnxNode* onnx_node, int opset_version);

  Caffe2Ops CreateSlice(OnnxNode* onnx_node, int opset_version);

  Caffe2Ops CreateReciprocal(OnnxNode* onnx_node, int opset_version);

  Caffe2Ops CreateBatchNormalization(OnnxNode* onnx_node, int opset_version);

  Caffe2Ops CreateMatMul(OnnxNode* onnx_node, int opset_version);

  // LUT related getters
  const std::unordered_map<std::string, std::string>& get_renamed_operators()
      const;
  const std::unordered_set<std::string>& get_rnn_operators() const;
  const std::unordered_map<std::string, int>& get_broken_operators() const;
  const std::unordered_map<std::string, std::string>& get_renamed_attrs() const;
  const std::
      unordered_map<std::string, std::unordered_map<std::string, std::string>>&
      get_per_op_renamed_attrs() const;
  const std::unordered_map<std::string, Caffe2Backend::SpecialOpConverter>&
  get_special_operators() const;
};

} // namespace onnx
} // namespace caffe2
