
#pragma once

#include <ATen/core/function_schema.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/core/type_conversion.h>
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
  ClassTypePtr classTypePtr;

  const std::string parentModule = "torch._C";
  class_(string className_) : className(className_) {
    auto obj = py::module::import(parentModule.c_str());
    pyClass = std::make_shared<py::class_<CurClass>>(obj, className.c_str());
    auto newClass = py::module::import("torch.jit").attr("_add_script_class")(*pyClass, ("__torch__."+parentModule+"."+ className_).c_str());

    classCu = std::make_shared<script::CompilationUnit>();
    tmap.put<CurClass*>(ClassType::create(
        c10::QualifiedName("__torch__." + parentModule + "." + className),
        classCu));
    classTypePtr = tmap.find<CurClass*>()->second;
    classTypePtr->addAttribute("capsule", CapsuleType::get());
    script::CompilationUnit::_get_python_cu().register_class(classTypePtr);
  }
  class_& init() {
    pyClass->def(py::init<>());
    auto graph = std::make_shared<Graph>();
    auto qualFuncName = className + "::__init__";
    static auto classRegistry = c10::RegisterOperators().op(
        qualFuncName, [](CurClass* cur) {
              // std::cout<<"initial class ptr: " <<&cur<<std::endl;
              *cur = CurClass();
            });
    auto input = graph->addInput()->setType(classTypePtr);
    auto capsuleNode = graph->insertNode(graph->create(prim::CreateCapsule, {}, 1))->output()->setType(CapsuleType::get());
    // auto capsuleNode = graph->insertConstant(5);
    auto n = graph->insertNode(graph->create(prim::SetAttr, {input, capsuleNode}, 0));
    n->s_(attr::name, "capsule");
    auto res = graph
                   ->insertNode(graph->create(
                       Symbol::fromQualString(qualFuncName), {input}, 0));
    graph->registerOutput(input);
    classCu->create_function("__init__", graph);
    return *this;
  }
  template <typename Func>
  class_& def(string name, Func f) {
    pyClass->def(name.c_str(), f);
    auto qualFuncName = className + "::" + name;
    static auto classRegistry = c10::RegisterOperators().op(
        qualFuncName, [f](CurClass* cur) {
              std::invoke(f, *cur);
            });
    auto graph = std::make_shared<Graph>();
    auto input = graph->addInput()->setType(classTypePtr);
    auto res = graph
                   ->insertNode(graph->create(
                       Symbol::fromQualString(qualFuncName), {input}, 0));
    graph->registerOutput(input);
    classCu->create_function(name, graph);
    return *this;
  }
};

} // namespace jit

} // namespace torch
