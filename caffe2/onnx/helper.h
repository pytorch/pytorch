#pragma once

#include "caffe2/core/common.h"
#include "onnx/onnx_pb.h"

#include <set>
#include <string>
#include <unordered_set>

namespace caffe2 {
namespace onnx {

using ::ONNX_NAMESPACE::AttributeProto;
using ::ONNX_NAMESPACE::NodeProto;

// \brief This class generates unique dummy names
class CAFFE2_API DummyName {
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

::ONNX_NAMESPACE::TypeProto ExtraTypeProto(
    const ::ONNX_NAMESPACE::TensorProto& tensor);

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

inline AttributeProto MakeAttribute(
    const std::string& name,
    const std::vector<float>& vals) {
  AttributeProto attr;
  attr.set_name(name);
  for (const auto v : vals) {
    attr.add_floats(v);
  }
  attr.set_type(AttributeProto::FLOATS);
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

inline AttributeProto MakeAttribute(
    const std::string& name,
    ::ONNX_NAMESPACE::TensorProto& val) {
  AttributeProto attr;
  attr.set_name(name);
  attr.mutable_t()->CopyFrom(val);
  attr.set_type(AttributeProto::TENSOR);
  return attr;
}

template <class T>
::ONNX_NAMESPACE::TensorProto MakeTensor(
    const string& name,
    const std::vector<T>& v,
    const ::ONNX_NAMESPACE::TensorProto_DataType& data_type_) {
  ::ONNX_NAMESPACE::TensorProto ret;
  ret.set_name(name);
  ret.add_dims(v.size());
  ret.set_data_type(data_type_);
  ret.mutable_raw_data()->assign(
      reinterpret_cast<const char*>(v.data()), v.size() * sizeof(T));
  return ret;
}

CAFFE2_API NodeProto MakeNode(
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
