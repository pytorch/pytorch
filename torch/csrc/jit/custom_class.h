
#pragma once

#include <ATen/core/function_schema.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/core/stack.h>
#include <c10/util/Metaprogramming.h>
#include <c10/util/TypeList.h>
#include <torch/csrc/jit/operator.h>
#include <torch/csrc/jit/script/compilation_unit.h>
#include <torch/csrc/jit/tracer.h>
#include <torch/csrc/utils/variadic.h>
#include <iostream>

using namespace std;
namespace torch {
namespace jit {
template <class CurClass>
struct class_ {
  class_(string className) {
    cout << "className: " << className << endl;
    auto cu = std::make_shared<CompilationUnit>();
    std::cout << "class 24: "
              << &torch::jit::script::CompilationUnit::_get_python_cu()
              << std::endl;
    auto classType = ClassType::create(c10::QualifiedName(className), cu);
    cout << "Registering: " << className << endl;
    std::cout << "this thread: " << std::this_thread::get_id() << std::endl;
    // cout<<"python_cu: "<<&CompilationUnit::_get_python_cu()<<endl;
    CompilationUnit::_get_python_cu().register_class(classType);
  }
};

} // namespace jit

} // namespace torch
