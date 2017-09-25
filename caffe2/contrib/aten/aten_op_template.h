#pragma once
#include <unordered_map>
#include <string>
#include <ATen/ATen.h>
#include <caffe2/core/context.h>
#include <caffe2/core/operator.h>
#include <google/protobuf/text_format.h>
#include <iostream>

// a map from descriptor strings (see [DESCRIPTORS])
// to the key in the switch statement that implements them
static std::unordered_map<std::string, int> op_to_key = {
  ${mappings}
};

namespace caffe2 {

using at::Half; // for AT_FORALL_SCALAR_TYPES

std::function<void(void*)> deleterFor(at::Tensor t) {
  // return a closure that holds a handle to t until it is called
  // to keep the aten memory alive
  return [t](void * ptr) mutable {
    t.reset();
  };
}

template <class Context>
class ATenOp : public Operator<Context> {
 public:
  ATenOp(const OperatorDef& operator_def, Workspace* ws)
  : Operator<Context>(operator_def, ws) {
    VLOG(2) << "ATen OpDef: " << ProtoDebugString(operator_def) << "\n";
    switch(findImplementation(operator_def)) {
      ${implementations}
      default:
        CAFFE_THROW("Unexpected key value for aten operator");
    }
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    return run_op();
  }
private:
  // actual operator implementation is initialized in ctor.
  std::function<bool()> run_op;
  at::Backend backend() const;

  TypeMeta typeMetaFor(const at::Tensor & t) {
    return typeMetaFor(t.type().scalarType());
  }
  TypeMeta typeMetaFor(at::ScalarType st) {
    #define DEFINE_CASE(ctype,aten_name,_) \
      case at::k##aten_name: \
        return TypeMeta::Make<ctype>();
    switch(st) {
      AT_FORALL_SCALAR_TYPES(DEFINE_CASE)
      default:
        CAFFE_THROW("Unknown ATen Type");
    }
    #undef DEFINE_CASE
  }

  at::Type & typeFor(const Tensor<Context> & ten) {
    return at::getType(backend(), atScalarTypeFor(ten.meta()));
  }
  at::Tensor tensorWrapping(Tensor<Context> & ten) {
    return typeFor(ten).tensorFromBlob(ten.raw_mutable_data(), ten.dims());
  }
  at::ScalarType atScalarTypeFor(const TypeMeta & meta) {
    #define DEFINE_IF(ctype,aten_name,_) \
    if(meta.Match<ctype>()) { \
      return at::k##aten_name; \
    }
    AT_FORALL_SCALAR_TYPES(DEFINE_IF)
    #undef DEFINE_IF
    CAFFE_THROW("Unknown type meta"); // TODO: improve error message...
  }
  void assignTo(Tensor<Context> * dst, const at::Tensor & src_) {
    at::Tensor src = src_.contiguous();
    auto at_sizes = src.sizes();
    std::vector<int64_t> dims(at_sizes.begin(),at_sizes.end());
    dst->Resize(dims);
    dst->ShareExternalPointer(src.data_ptr(), typeMetaFor(src), 0, deleterFor(src));
  }

  // the AT_FORALL_SCALAR_TYPES macro just gives a 'i' or 'd' argument
  // for each type to specify if it is stored as a integer or a double.
  // We need this workaround here to extract the value in the scalar losslessly
  // because in some cases like 'sum' Torch promotes float to double
  // and will complain if we downcast it with toFloat, causing it
  // to lose precision
  double extract_d(const at::Scalar & s) {
    return s.toDouble();
  }
  int64_t extract_i(const at::Scalar & s) {
    return s.toLong();
  }

  void assignTo(Tensor<Context> * dst, at::Type & inferred_type, at::Scalar scalar) {
    switch(inferred_type.scalarType()) {
      #define DEFINE_CASE(ctype,aten_name,native) \
        case at::k##aten_name: { \
          auto value = extract_##native(scalar); \
          assignToValue<ctype>(dst, at::convert<ctype,decltype(value)>(value)); \
        } break;
      AT_FORALL_SCALAR_TYPES(DEFINE_CASE)
      #undef DEFINE_CASE
      default:
        CAFFE_THROW("Unknown ATen Type");
    }
  }
  template<typename T>
  void assignToValue(Tensor<Context> * dst, T v) {
    dst->Resize(std::vector<TIndex>());
    math::Set(1, v, dst->template mutable_data<T>(), &context_);
  }
  int findImplementation(const OperatorDef& operator_def) {
    CAFFE_ENFORCE(HasArgument("operator"));
    std::string op = OperatorBase::GetSingleArgument<std::string>("operator", "");
    // construct descriptor string ([DESCRIPTORS]) given the attributes
    // and inputs of this operator_def, and look up the implementation key
    // for this variant
    std::stringstream descriptor;
    descriptor << op << "-" << InputSize();
    std::vector<std::string> attrs;
    for(size_t i = 0; i < operator_def.arg_size(); i++) {
      auto & attr = operator_def.arg(i);
      if(attr.name() == "operator" || attr.name() == "type" )
        continue;
      attrs.push_back(attr.name());
    }
    std::sort(attrs.begin(), attrs.end());
    for(auto & a : attrs)
      descriptor << "-" << a;
    std::string descriptor_s = descriptor.str();
    if(op_to_key.count(descriptor_s) == 0) {
      std::stringstream ss;
      ss << "Attempting to run unknown ATen operator configuration: "
         << descriptor_s;
      CAFFE_THROW(ss.str());
    }
    return op_to_key.at(descriptor_s);
  }
  at::Scalar readScalarAttribute(const std::string & name) {
    if(OperatorBase::HasSingleArgumentOfType<int64_t>(name)) {
      return OperatorBase::GetSingleArgument<int64_t>(name, 0);
    } else {
      CAFFE_ENFORCE(OperatorBase::HasSingleArgumentOfType<float>(name));
      return OperatorBase::GetSingleArgument<float>(name, 0);
    }
  }
  template<typename T>
  T readAttribute(const std::string & name) {
    CAFFE_ENFORCE(OperatorBase::HasSingleArgumentOfType<int64_t>(name));
    return OperatorBase::GetSingleArgument<T>(name, 0);
  }
  std::vector<int64_t> readIntList(const std::string & name) {
    CAFFE_ENFORCE(OperatorBase::HasArgument(name));
    return OperatorBase::GetRepeatedArgument<int64_t>(name, {});
  }
  at::ScalarType stringToScalarType(const std::string & name) {
    #define DEFINE_IF(type,aten) \
      if(#type == name) \
        return at::k##aten;
    DEFINE_IF(float16, Half)
    DEFINE_IF(float, Float)
    DEFINE_IF(double, Double)
    DEFINE_IF(uint8, Byte)
    DEFINE_IF(int8, Char)
    DEFINE_IF(int16, Short)
    DEFINE_IF(int32, Int)
    DEFINE_IF(int64, Long)
    CAFFE_THROW("unsupported type annotation: ", name);
  }
  at::Type & stringToType(const std::string & name) {
    return at::getType(backend(), stringToScalarType(name));
  }
  at::Type * readTypeAttribute(const std::string & name) {
    CAFFE_ENFORCE(OperatorBase::HasSingleArgumentOfType<std::string>(name));
    return &stringToType(OperatorBase::GetSingleArgument<std::string>(name, ""));
  }
};

}
