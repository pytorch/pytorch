#pragma once

#include <sstream>

#include <ATen/ATen.h>
#include <caffe2/core/context.h>
#include <caffe2/core/operator.h>
#include <torch/csrc/jit/function_schema.h>
#include <torch/csrc/jit/interpreter.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/ivalue.h>
#include <torch/csrc/jit/named_value.h>
#include <torch/csrc/jit/operator.h>
#include <torch/csrc/jit/type.h>

namespace caffe2 {

template <class Context>
class ATenOp2 : public Operator<Context> {
 public:
  ATenOp2(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        schema_str_(
            this->template GetSingleArgument<std::string>("schema", "")) {
    CAFFE_ENFORCE(!schema_str_.empty());
    schema_ = std::make_shared<torch::jit::FunctionSchema>(
        torch::jit::parseSchema(schema_str_));
    torch::jit::Graph g;
    std::vector<torch::jit::NamedValue> inputs;
    CAFFE_ENFORCE_EQ(schema_->arguments.size(), InputSize());
    CAFFE_ENFORCE_EQ(schema_->returns.size(), OutputSize());
    for (auto& argument : schema_->arguments) {
      auto* inp = g.addInput(argument.name);
      inp->setType(argument.type);
      inputs.emplace_back(argument.name, inp);
    }
    auto* n =
        g.insert(torch::jit::Symbol::fromQualString(schema_->name), inputs)
            ->node();
    operation_ = torch::jit::getOperatorFor(n).getOperation();
  }

  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    torch::jit::Stack stack;
    size_t i = 0;
    for (auto& arg : schema_->arguments) {
      stack.push_back(convertToIValue(arg.type, Input(i)));
      ++i;
    }
    operation_(stack);
    i = 0;
    CAFFE_ENFORCE_EQ(stack.size(), schema_->returns.size());
    for (auto& item : stack) {
      convertToCaffe2(item, Output(i));
    }
    return true;
  }

 private:
  torch::jit::IValue convertToIValue(
      torch::jit::TypePtr type,
      const Tensor& t) {
    // PERPETUAL TODO: Support more types
    if (type->isSubtypeOf(torch::jit::DynamicType::get())) {
      auto& ten = const_cast<Tensor&>(t);
      return torch::jit::IValue(
          typeFor(ten).tensorFromBlob(ten.raw_mutable_data(), ten.dims()));
    } else if (type->isSubtypeOf(torch::jit::NumberType::get())) {
      auto& ten = const_cast<Tensor&>(t);
      auto at_tensor =
          typeFor(ten).tensorFromBlob(ten.raw_mutable_data(), ten.dims());
      return torch::jit::IValue(at::_local_scalar(at_tensor));
    } else {
      std::ostringstream oss;
      oss << "Unknown input type ";
      oss << type->str();
      oss << " for operator with schema ";
      oss << *schema_;
      oss << ". Please file a bug report\n";
      CAFFE_THROW(oss.str());
    }
  }
  void convertToCaffe2(torch::jit::IValue ivalue, Tensor* dst) {
    // PERPETUAL TODO: Support more types
    if (ivalue.isTensor()) {
      auto src = ivalue.toTensor().contiguous();
      auto at_sizes = src.sizes();
      caffe2::TypeMeta type_meta = typeMetaFor(src);
      at::Device device = src.device();
      at::TensorImpl* src_impl = src.unsafeReleaseTensorImpl();
      std::vector<int64_t> dims(at_sizes.begin(), at_sizes.end());
      dst->Resize(dims);
      dst->ShareExternalPointer(
          at::DataPtr(
              src_impl->data(),
              static_cast<void*>(src_impl),
              [](void* t_ptr) -> void {
                at::TensorImpl* local_impl =
                    static_cast<at::TensorImpl*>(t_ptr);
                c10::raw::intrusive_ptr::decref(local_impl);
              },
              device),
          type_meta,
          0);
    } else {
      std::ostringstream oss;
      oss << "Unknown output type for operator with schema ";
      oss << *schema_;
      CAFFE_THROW(oss.str());
    }
  }

 private:
  std::shared_ptr<torch::jit::FunctionSchema> schema_;
  std::string schema_str_;
  torch::jit::Operation operation_;

  TypeMeta typeMetaFor(const at::Tensor& t) {
    return typeMetaFor(t.type().scalarType());
  }
  TypeMeta typeMetaFor(at::ScalarType st) {
#define DEFINE_CASE(ctype, aten_name, _) \
  case at::k##aten_name:                 \
    return TypeMeta::Make<ctype>();
    switch (st) {
      AT_FORALL_SCALAR_TYPES(DEFINE_CASE)
      default:
        CAFFE_THROW("Unknown ATen Type");
    }
#undef DEFINE_CASE
  }

  at::Type& typeFor(const Tensor& ten) {
    return at::getNonVariableType(backend(), atScalarTypeFor(ten.meta()));
  }

  at::ScalarType atScalarTypeFor(const TypeMeta& meta) {
#define DEFINE_IF(ctype, aten_name, _) \
  if (meta.Match<ctype>()) {           \
    return at::k##aten_name;           \
  }
    AT_FORALL_SCALAR_TYPES(DEFINE_IF)
#undef DEFINE_IF
    // Special case for bool, since the type in ATen is actually Byte
    if (meta.Match<bool>()) {
      return at::kByte;
    }
    CAFFE_THROW("Unknown type meta"); // TODO: improve error message...
  }

  at::Backend backend() const;
};

} // namespace caffe2
