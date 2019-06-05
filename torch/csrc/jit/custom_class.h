
#pragma once

#include <ATen/core/function_schema.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/core/stack.h>
#include <c10/util/C++17.h>
#include <c10/util/Metaprogramming.h>
#include <c10/util/TypeList.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/jit/operator.h>
#include <torch/csrc/jit/script/compilation_unit.h>
#include <torch/csrc/jit/tracer.h>
#include <torch/csrc/utils/variadic.h>
#include <iostream>

namespace py = pybind11;
using namespace std;
namespace torch {
namespace jit {
template <class CurClass>
struct class_ {
  std::string className;
  std::shared_ptr<py::class_<CurClass>> pyClass = nullptr;
  std::shared_ptr<script::CompilationUnit> classCu = nullptr;
  ClassTypePtr classType;

  const std::string parentModule = "torch._C";
  class_(string className_) : className(className_) {
    auto obj = py::module::import(parentModule.c_str());
    pyClass = std::make_shared<py::class_<CurClass>>(obj, className.c_str());
    pyClass->def(py::init<>());
    classCu = std::make_shared<script::CompilationUnit>();
    classType = ClassType::create(
        c10::QualifiedName("__torch__." + parentModule + "." + className),
        classCu);
    script::CompilationUnit::_get_python_cu().register_class(classType);
    auto graph = std::make_shared<Graph>();
    auto input = graph->addInput()->setType(classType);
    graph->registerOutput(input);
    classCu->create_function("__init__", graph);
  }
  template <typename Func>
  class_& def(string name, Func f) {
    pyClass->def(name.c_str(), f);
    auto qualFuncName = className + "::" + name;
    static auto classRegistry = c10::RegisterOperators().op(
        qualFuncName,
        torch::RegisterOperators::options().kernel(
            [f](CurClass* cur) { return std::invoke(f, *cur); }));
    auto graph = std::make_shared<Graph>();
    auto input = graph->addInput()->setType(classType);
    auto& ops = getAllOperatorsFor(Symbol::fromQualString(qualFuncName));
    std::cout << ops.size() << std::endl;

    auto res = graph
                   ->insertNode(graph->create(
                       Symbol::fromQualString(qualFuncName), {input}))
                   ->output();
    graph->registerOutput(res);
    classCu->create_function(name, graph);
    return *this;
  }
};

} // namespace jit

} // namespace torch
