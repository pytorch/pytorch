
#pragma once

#include <ATen/core/function_schema.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/core/stack.h>
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
struct Pet {
  Pet(const std::string& name) : name(name) {}
  void setName(const std::string& name_) {
    name = name_;
  }
  const std::string& getName() const {
    return name;
  }

  std::string name;
};
namespace torch {
namespace jit {
template <class CurClass>
struct class_ {
  class_(string className) {
    // std::cout<<"40: "<<object<<std::endl;
    auto obj = py::module::import("torch._C");
    py::class_<CurClass>(obj, className.c_str()).def(py::init<>());
    auto cu = std::make_shared<torch::jit::script::CompilationUnit>();
    auto classType = ClassType::create(
        c10::QualifiedName("__torch__.torch._C." + className), cu);
    torch::jit::script::CompilationUnit::_get_python_cu().register_class(
        classType);
    auto func = []() { return new CurClass; };
    auto graph = std::make_shared<Graph>();
    auto input = graph->addInput()->setType(classType);
    graph->registerOutput(input);
    cu->create_function("__init__", graph);
  }
  // todo: Doesn't work yet
  // template<typename Func>
  // class_& def(string name, Func f) {
  //   auto graph = std::make_shared<Graph>();
  //   cu->create_function(name, make_shared<Graph>());
  // }
};

} // namespace jit

} // namespace torch
