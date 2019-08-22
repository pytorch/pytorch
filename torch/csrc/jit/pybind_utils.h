#pragma once

#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/stack.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/Layout.h>
#include <torch/csrc/QScheme.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/operator.h>
#include <torch/csrc/jit/script/module.h>
#include <torch/csrc/jit/tracer.h>
#include <torch/csrc/utils/auto_gil.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/six.h>

#include <ATen/core/function_schema.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

// The visibility attribute is to avoid a warning about storing a field in the
// struct that has a different visibility (from pybind) than the struct.
#ifdef _WIN32
#define VISIBILITY_HIDDEN
#else
#define VISIBILITY_HIDDEN __attribute__((visibility("hidden")))
#endif

namespace torch {
namespace jit {

// error reporting: when reporting user-caused errors, these functions should
// not use AT_ERROR macros, since these macros add stack trace information
// that is confusing to display to the end user since it always reports
// locations in libtorch code rather than user code.


std::shared_ptr<script::CompilationUnit> get_python_cu();

struct TypedIValue : public std::pair<IValue, TypePtr> {
  using pair::pair;

  IValue& ivalue() {
    return this->first;
  }
  TypePtr& type() {
    return this->second;
  }
};

TypedIValue toDictKeyIValue(py::handle key);

c10::optional<TypePtr> unifyOrInitializeType(
    TypePtr accum,
    TypePtr unify);

MatchTypeReturn tryToInferContainerType(py::handle input);

// Try to infer the type of a Python object
// The type cannot be inferred if:
//   input is a None
//   input is an empty container (list, dict)
//   input is an list with element types that cannot be unified
//   input is an dict with key or value types that cannot be unified
MatchTypeReturn tryToInferType(py::handle input);

IValue toIValue(
    py::handle obj,
    const TypePtr& type,
    c10::optional<int32_t> N = c10::nullopt);

bool isTraceableType(TypePtr type);

TypedIValue toTraceableIValue(py::handle input);

IValue toIValue(py::handle input);

Stack toStack(const py::tuple& inputs);

tracer::TypedStack toTypedStack(const py::tuple& inputs);

IValue createGenericList(py::handle obj, const TypePtr& elem_type);

IValue createGenericDict(
    py::handle obj,
    const TypePtr& key_type,
    const TypePtr& value_type);

IValue toIValue(
    py::handle obj,
    const TypePtr& type,
    c10::optional<int32_t> N);

// Small wrapper around getting the type name string from Python to make
// types easier to interpret, e.g. give the structural type for a NamedTuple
std::string friendlyTypeName(py::handle obj);

IValue argumentToIValue(
    const FunctionSchema& schema,
    size_t argumentPosition,
    py::handle object);

IValue returnToIValue(const TypePtr& type, py::handle object);

c10::optional<py::object> tryToConvertToCustomClass(
    const c10::intrusive_ptr<c10::ivalue::Object>& obj);
py::object toPyObject(IValue&& ivalue);

struct VISIBILITY_HIDDEN tuple_slice {
  /*implicit*/ tuple_slice(py::tuple tup_)
      : tup(std::move(tup_)), b(0), e(tup.size()) {}
  tuple_slice(py::tuple tup_, int64_t b_)
      : tup(std::move(tup_)), b(b_), e(tup.size()) {}
  tuple_slice(py::tuple tup_, int64_t b_, int64_t e_)
      : tup(std::move(tup_)), b(b_), e(e_) {}
  py::detail::tuple_iterator begin() const {
    return {tup, static_cast<pybind11::ssize_t>(b)};
  }
  py::detail::tuple_iterator end() const {
    return {tup, static_cast<pybind11::ssize_t>(e)};
  }
  size_t size() const {
    return e - b;
  }
  py::detail::tuple_accessor operator[](size_t index) const {
    return {tup, static_cast<size_t>(b + index)};
  }

 private:
  py::tuple tup;
  int64_t b;
  int64_t e;
};

Stack createStackForSchema(
    const FunctionSchema& schema,
    const tuple_slice& args,
    const py::kwargs& kwargs,
    c10::optional<IValue> self);

py::object createPyObjectForStack(Stack&& stack);

// TODO: Remove once we clean up the GraphExecutor usage.
Stack evilDeprecatedBadCreateStackDoNotUse(
    const py::tuple& tuple,
    at::ArrayRef<Value*> inputs,
    size_t reserve_extra_space = 0);

py::object invokeScriptFunctionFromPython(
    Function& callee,
    tuple_slice args,
    py::kwargs kwargs,
    c10::optional<IValue> self = c10::nullopt);

py::object invokeScriptMethodFromPython(
    script::Method& callee,
    tuple_slice args,
    py::kwargs kwargs);
py::object invokeOperatorFromPython(
    const Operator& op,
    py::args args,
    py::kwargs kwargs);

} // namespace jit
} // namespace torch
