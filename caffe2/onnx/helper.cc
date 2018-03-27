#include "caffe2/onnx/helper.h"

#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"

namespace caffe2 { namespace onnx  {

size_t DummyName::counter_ = 0;

std::unordered_set<std::string>& DummyName::get_used_names() {
  static std::unordered_set<std::string> used_names;
  return used_names;
}

std::string DummyName::NewDummyName() {
  while (true) {
    const std::string name = caffe2::MakeString("OC2_DUMMY_", counter_++);
    auto ret = get_used_names().insert(name);
    if (ret.second) {
      return name;
    }
  }
}

void DummyName::Reset(const std::unordered_set<std::string> &used_names) {
  auto& names = get_used_names();
  names = used_names;
  counter_ = 0;
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
  for (const auto& input: inputs) {
    node.add_input(input);
  }
  for (const auto& output: outputs) {
    node.add_output(output);
  }
  for (const auto& attr: attributes) {
    node.add_attribute()->CopyFrom(attr);
  }
  return node;
}
}}
