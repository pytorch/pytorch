#pragma once

#include "torch/csrc/utils/pybind.h"

namespace torch { namespace jit {

inline Stack createStack(const py::tuple& tuple, at::ArrayRef<Value*> inputs, size_t reserve_extra_space = 0) {
  if (tuple.size() != inputs.size()) {
    throw std::runtime_error("expected " + std::to_string(inputs.size()) +
                             " inputs, but got " + std::to_string(tuple.size()));
  }
  static const auto castToIValue = [](const py::object& obj, Type& t) -> IValue{
    switch (t.kind()) {
      case TypeKind::DynamicType:
      case TypeKind::TensorType:
        return py::cast<autograd::Variable>(obj);
      case TypeKind::FloatType:
        return py::cast<double>(obj);
      case TypeKind::IntType:
        return py::cast<int64_t>(obj);
      case TypeKind::NoneType:
        return {};
      case TypeKind::ListType:
      case TypeKind::TupleType:
        throw std::runtime_error("Lists and tuples are not supported yet");
      case TypeKind::NumberType:
        throw std::runtime_error("Insufficient type information to convert input");
    }
    throw std::runtime_error("Missing cases in castToIValue! File a bug report.");
  };
  Stack result;
  result.reserve(tuple.size() + reserve_extra_space);
  for (size_t i = 0; i < inputs.size(); ++i) {
    result.push_back(castToIValue(tuple[i], *inputs[i]->type()));
  }
  return result;
}

inline py::object wrapStack(Stack&& outputs, at::ArrayRef<Value*> output_vals) {
  if (outputs.size() != output_vals.size()) {
    throw std::runtime_error("expected " + std::to_string(output_vals.size()) +
                             " outputs, but got " + std::to_string(outputs.size()));
  }
  static const auto createOutput = [](IValue && ivalue, Value * value) -> py::object {
    switch (value->type()->kind()) {
      case TypeKind::DynamicType:
      case TypeKind::TensorType:
        return py::cast(autograd::Variable(ivalue.toTensor()));
      case TypeKind::FloatType:
        return py::cast(ivalue.toDouble());
      case TypeKind::IntType:
        return py::cast(ivalue.toInt());
      case TypeKind::NoneType:
        return py::none();
      case TypeKind::ListType:
      case TypeKind::TupleType:
        throw std::runtime_error("Lists and tuples are not supported yet");
      case TypeKind::NumberType:
        throw std::runtime_error("Insufficient type information to convert input");
    }
    throw std::runtime_error("Missing cases in createOutput! File a bug report.");
  };
  if (outputs.size() == 0) {
    return py::none();
  } else if (outputs.size() == 1) {
    return createOutput(std::move(outputs[0]), output_vals[0]);
  } else {
    py::tuple tuple(outputs.size());
    for(size_t i = 0; i < outputs.size(); i++) {
      tuple[i] = createOutput(std::move(outputs[i]), output_vals[i]);
    }
    return tuple;
  }
}

} }  // namespace torch::jit
