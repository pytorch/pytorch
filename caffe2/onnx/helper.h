#pragma once

#include "onnx/onnx_pb.h"

#include <set>
#include <string>
#include <unordered_set>

namespace caffe2 {
namespace onnx {

using ::ONNX_NAMESPACE::AttributeProto;
using ::ONNX_NAMESPACE::NodeProto;

// \brief This class generates unique dummy names
class DummyName {
 public:
  std::string NewDummyName();

  void Reset(const std::unordered_set<std::string>& used_names);

  void AddName(const std::string& new_used) {
    used_names_.insert(new_used);
  }

 private:
  std::unordered_set<std::string> used_names_;
  size_t counter_{0};
};

inline AttributeProto MakeAttribute(
    const std::string& name,
    const std::vector<int64_t>& vals) {
  AttributeProto attr;
  attr.set_name(name);
  for (const auto v : vals) {
    attr.add_ints(v);
  }
  attr.set_type(AttributeProto::INTS);
  return attr;
}

inline AttributeProto MakeAttribute(const std::string& name, int64_t val) {
  AttributeProto attr;
  attr.set_name(name);
  attr.set_i(val);
  attr.set_type(AttributeProto::INT);
  return attr;
}

inline AttributeProto MakeAttribute(
    const std::string& name,
    const std::string& val) {
  AttributeProto attr;
  attr.set_name(name);
  attr.set_s(val);
  attr.set_type(AttributeProto::STRING);
  return attr;
}

NodeProto MakeNode(
    const std::string& type,
    const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs,
    const std::vector<AttributeProto>& attributes,
    const std::string& name = "");

inline NodeProto MakeNode(
    const std::string& type,
    const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs,
    const std::string& name = "") {
  return MakeNode(type, inputs, outputs, {}, name);
}

} // namespace onnx
} // namespace caffe2
