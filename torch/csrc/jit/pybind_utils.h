#pragma once

#include "torch/csrc/jit/function_schema.h"
#include "torch/csrc/jit/ivalue.h"
#include "torch/csrc/jit/stack.h"
#include "torch/csrc/jit/script/module.h"
#include "torch/csrc/jit/type.h"
#include "torch/csrc/jit/operator.h"
#include "torch/csrc/utils/pybind.h"

#include <ATen/Error.h>

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace torch { namespace jit {
namespace detail {
inline void findErrorInKwargs(
    const FunctionSchema& schema,
    py::kwargs kwargs) {
  const auto& arguments = schema.arguments;
  // First check if any of the kwargs are unknown, i.e. don't match the name of
  // any argument in the schema.
  for (const auto& kwarg : kwargs) {
    const auto key = py::cast<std::string>(kwarg.first);
    AT_CHECK(
        std::count_if(
            arguments.begin(),
            arguments.end(),
            [&key](const Argument& argument) { return argument.name == key; }),
        "Unknown keyword argument '", key, "' for operator '",
        schema.name, "'. Schema: ", schema);
  }
  // If there are unconsumed kwargs but none of them were unknown, the first
  // positional argument present in the kwargs is duplicated.
  for (const auto& argument : arguments) {
    if (kwargs.contains(argument.name.c_str())) {
      AT_ASSERT(!argument.default_value);
      AT_ERROR(
          "Argument '", argument.name, "' specified both as positional and ",
          "keyword argument. Schema: ", schema);
    }
  }
}
} // namespace detail

inline IValue toIValue(py::handle input) {
  if (THPVariable_Check(input.ptr())) {
    return py::cast<at::Tensor>(input);
  } else if (py::isinstance<py::tuple>(input)) {
    py::tuple input_tuple = py::cast<py::tuple>(input);
    Stack s;
    s.reserve(input_tuple.size());
    for (py::handle elem : input_tuple) {
      s.push_back(toIValue(elem));
    }
    return Tuple::create(s);
  } else {
    AT_ERROR("Only tensors and tuples of tensors are supported as inputs to traced functions");
  }
}

inline Stack toStack(const py::tuple& inputs) {
  return toIValue(inputs).toTuple()->elements();
}

inline IValue toIValue(py::handle obj, const TypePtr& type);

inline IValue createGenericList(py::handle obj, const TypePtr& elem_type) {
  std::vector<IValue> elems;
  for(auto elem : obj) {
    elems.push_back(toIValue(elem, elem_type));
  }
  return ConstantList<IValue>::create(std::move(elems));
}

inline IValue toIValue(py::handle obj, const TypePtr& type) {
    switch (type->kind()) {
      case TypeKind::DynamicType:
      case TypeKind::TensorType:
      case TypeKind::CompleteTensorType:
        return py::cast<autograd::Variable>(obj);
      case TypeKind::FloatType:
        return py::cast<double>(obj);
      case TypeKind::IntType:
        return py::cast<int64_t>(obj);
      case TypeKind::NoneType:
        return {};
      case TypeKind::TupleType: {
        py::tuple tuple = py::cast<py::tuple>(obj);
        size_t tuple_size = tuple.size();
        const auto & elem_types = type->cast<TupleType>()->elements();
        if (elem_types.size() != tuple_size) {
          AT_ERROR("Expected ", elem_types.size(), " tuple elements for argument, but got ", tuple_size);
        }
        std::vector<IValue> values;
        values.reserve(tuple_size);
        for (size_t i = 0; i < tuple_size; ++i) {
          values.push_back(toIValue(tuple[i], elem_types[i]));
        }
        return Tuple::create(std::move(values));
      }
      case TypeKind::StringType:
        return ConstantString::create(py::cast<std::string>(obj));
      case TypeKind::ListType: {
        const auto& elem_type = type->expect<ListType>()->getElementType();
        switch(elem_type->kind()) {
          case TypeKind::IntType:
            return py::cast<std::vector<int64_t>>(obj);
          case TypeKind::FloatType:
            return py::cast<std::vector<double>>(obj);
          case TypeKind::TensorType:
          case TypeKind::DynamicType:
            return py::cast<std::vector<at::Tensor>>(obj);
          default:
            return createGenericList(obj, elem_type);
        }
      }
      case TypeKind::NumberType:
        AT_ERROR("Insufficient type information to convert input");
      case TypeKind::GeneratorType:
        AT_ERROR("Generators are not supported yet.");
    }
  AT_ERROR("Missing cases in toIValue! File a bug report.");
}

inline IValue argumentToIValue(
    const FunctionSchema& schema,
    size_t argumentPosition,
    py::handle object) {
  const auto& argument = schema.arguments.at(argumentPosition);
  try {
    return toIValue(object, argument.type);
  } catch (const py::cast_error& error) {
    AT_ERROR(
        schema.name, "() expected value of type ", argument.type->str(),
        " for argument '", argument.name,
        "' in position ", argumentPosition,
        ", but instead got value of type ", object.get_type().attr("__name__").str(),
        ". Declaration: ", schema);
  }
}

inline py::object toPyObject(IValue&& ivalue) {
  if (ivalue.isNone()) {
    return py::none();
  } else if (ivalue.isTensor()) {
    return py::cast(autograd::Variable(ivalue.toTensor()));
  } else if (ivalue.isDouble()) {
    return py::cast(ivalue.toDouble());
  } else if (ivalue.isInt()) {
    return py::cast(ivalue.toInt());
  } else if (ivalue.isIntList()) {
    return py::cast(ivalue.toIntListRef());
  } else if (ivalue.isDoubleList()) {
    return py::cast(ivalue.toDoubleListRef());
  } else if (ivalue.isTensorList()) {
    return py::cast(ivalue.toTensorListRef());
  } else if (ivalue.isTuple()) {
    auto tuple = ivalue.toTuple();
    const auto & elements = tuple->elements();
    py::tuple t { elements.size() };
    for (size_t i = 0; i < elements.size(); ++i) {
      t[i] = toPyObject(IValue{elements[i]});
    }
    return t;
  } else {
    AT_ERROR("Missing cases in 'toPyObject'! File a bug report.");
  }
}

inline Stack createStackForSchema(
    const FunctionSchema& schema,
    py::args args,
    py::kwargs kwargs = py::kwargs()) {
  AT_CHECK(
      args.size() + kwargs.size() <= schema.arguments.size(),
      schema.name, "() expected at most ", schema.arguments.size(),
      " argument(s) but received ",
      args.size() + kwargs.size(), " argument(s). Declaration: ", schema);

  Stack stack;
  stack.reserve(schema.arguments.size());

  // First push all positional args.
  for (size_t i = 0; i < args.size(); ++i) {
    // Use the type information from the schema to convert the PyObject.
    push(stack, argumentToIValue(schema, i, args[i]));
  }

  // Now for every remaining non-positional argument in the schema, look for it
  // in the kwargs dict and push it if found, or use its default value if it
  // has one.
  size_t consumed_kwargs = 0;
  for (size_t i = args.size(); i < schema.arguments.size(); ++i) {
    const auto& arg = schema.arguments[i];
    if (kwargs.contains(arg.name.c_str())) {
      push(stack, argumentToIValue(schema, i, kwargs[arg.name.c_str()]));
      consumed_kwargs += 1;
    } else if (arg.default_value) {
      push(stack, *arg.default_value);
    } else {
      AT_ERROR(
          schema.name, "() is missing value for argument '", arg.name,
          "'. Declaration: ", schema);
    }
  }

  if (consumed_kwargs != kwargs.size()) {
    detail::findErrorInKwargs(schema, kwargs);
  }

  return stack;
}

inline py::object createPyObjectForStack(Stack&& stack) {
  if (stack.empty()) {
    return py::none();
  }

  // Return a simple value and not a single-element tuple if there is only one
  // return value.
  if (stack.size() == 1) {
    return toPyObject(std::move(stack[0]));
  }

  // If there is more than one return value, pop them into a py::tuple.
  py::tuple return_values(stack.size());
  for (size_t ret = 0; ret < return_values.size(); ++ret) {
    return_values[ret] = toPyObject(std::move(stack[ret]));
  }

  return return_values;
}

// TODO: Remove once we clean up the GraphExecutor usage.
inline Stack evilDeprecatedBadCreateStackDoNotUse(const py::tuple& tuple, at::ArrayRef<Value*> inputs, size_t reserve_extra_space = 0) {
  if (tuple.size() != inputs.size()) {
    AT_ERROR("expected " + std::to_string(inputs.size()) +
                             " inputs, but got " + std::to_string(tuple.size()));
  }
  Stack result;
  result.reserve(tuple.size() + reserve_extra_space);
  for (size_t i = 0; i < inputs.size(); ++i) {
    result.push_back(toIValue(std::move(tuple[i]), inputs[i]->type()));
  }
  return result;
}

inline py::object invokeScriptMethodFromPython(
    script::Method& method,
    py::args args, py::kwargs kwargs) {
  auto stack = createStackForSchema(method.getSchema(), std::move(args), std::move(kwargs));
  method.run(stack);
  return createPyObjectForStack(std::move(stack));
}

inline py::object invokeOperatorFromPython(
    const Operator& op,
    py::args args,
    py::kwargs kwargs) {
  try {
    // Create a stack full of the arguments and keyword arguments.
    auto stack =
        createStackForSchema(op.schema(), std::move(args), std::move(kwargs));

    // Invoke the operation, which puts the return values onto the stack.
    op.getOperation()(stack);

    return createPyObjectForStack(std::move(stack));
  } catch (const at::Error& error) {
    // We don't want to show the backtrace in the error message in Python.
    throw std::runtime_error(error.what_without_backtrace());
  }
}
}}  // namespace torch::jit
