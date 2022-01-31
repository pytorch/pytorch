#include "caffe2/onnx/helper.h"

#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"

namespace caffe2 {
namespace onnx {

std::string DummyName::NewDummyName() {
  while (true) {
    const std::string name = c10::str("OC2_DUMMY_", counter_++);
    auto ret = used_names_.insert(name);
    if (ret.second) {
      // NOLINTNEXTLINE(performance-no-automatic-move)
      return name;
    }
  }
}

void DummyName::Reset(const std::unordered_set<std::string>& used_names) {
  used_names_ = used_names;
  counter_ = 0;
}

::ONNX_NAMESPACE::TypeProto ExtraTypeProto(
    const ::ONNX_NAMESPACE::TensorProto& tensor) {
  ::ONNX_NAMESPACE::TypeProto t;
  auto* tensor_type = t.mutable_tensor_type();
  tensor_type->set_elem_type(tensor.data_type());
  auto* shape = tensor_type->mutable_shape();
  for (const auto d : tensor.dims()) {
    shape->add_dim()->set_dim_value(d);
  }
  return t;
}

NodeProto MakeNode(
    const std::string& type,
    const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs,
    const std::vector<AttributeProto>& attributes,
    const std::string& name) {
  NodeProto node;
  if (!name.empty()) {
    node.set_name(name);
  }
  node.set_op_type(type);
  for (const auto& input : inputs) {
    node.add_input(input);
  }
  for (const auto& output : outputs) {
    node.add_output(output);
  }
  for (const auto& attr : attributes) {
    node.add_attribute()->CopyFrom(attr);
  }
  return node;
}
} // namespace onnx
} // namespace caffe2
