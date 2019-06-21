
#pragma once

#include <ATen/core/function_schema.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/core/stack.h>
#include <ATen/core/type_conversion.h>
#include <c10/util/C++17.h>
#include <c10/util/Metaprogramming.h>
#include <c10/util/TypeList.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/jit/operator.h>
#include <torch/csrc/jit/script/compilation_unit.h>
#include <torch/csrc/jit/tracer.h>
#include <torch/csrc/utils/variadic.h>
#include <iostream>
#include <sstream>

namespace py = pybind11;
namespace torch {
namespace jit {

namespace detail {
template <class R, class...>
struct types {
  constexpr static bool hasRet = true;
  using type = types;
};
template <class... args>
struct types<void, args...> {
  constexpr static bool hasRet = false;
  using type = types;
};
template <class Sig>
struct args;
template <class R, class CurClass, class... Args>
struct args<R (CurClass::*)(Args...)> : types<R, Args...> {};
template <class Sig>
using args_t = typename args<Sig>::type;
} // namespace detail
template <class... Types>
detail::types<void, Types...> init() {}

template <class CurClass>
class class_ {
  std::string className;
  std::shared_ptr<py::class_<CurClass>> pyClass = nullptr;
  std::shared_ptr<script::CompilationUnit> classCu = nullptr;
  ClassTypePtr classTypePtr;

  const std::string parentModule = "torch._C";

 public:
  class_(string className_) : className(std::move(className_)) {
    // Currently we register everything as a python class just for convenience.
    // We'll want to remove this at some point to get rid of the python
    // dependency. It would require significant changes to class registration,
    // (I think)?
    auto obj = py::module::import(parentModule.c_str());
    pyClass = std::make_shared<py::class_<CurClass>>(obj, className.c_str());
    auto newClass =
        py::module::import("torch.jit")
            .attr("_add_script_class")(
                *pyClass,
                ("__torch__." + parentModule + "." + className_).c_str());

    // We currently represent custom classes as torchscript classes with a
    // capsule attribute.
    classCu = std::make_shared<script::CompilationUnit>();
    tmap.put<CurClass*>(ClassType::create(
        c10::QualifiedName("__torch__." + parentModule + "." + className),
        classCu));
    classTypePtr = tmap.find<CurClass*>()->second;
    classTypePtr->addAttribute("capsule", CapsuleType::get());
    script::CompilationUnit::_get_python_cu().register_class(classTypePtr);
  }

  template <typename... Types>
  class_& def(detail::types<void, Types...>) { // Used in combination with
                                               // torch::jit::init<...>()
    pyClass->def(py::init<Types...>());
    auto graph = std::make_shared<Graph>();
    auto qualFuncName = className + "::__init__";
    // auto qualFuncName = className + "::__init__." + type_name<int64_t,
    // Types...>();
    auto func = [](CurClass* cur, Types... args) { *cur = CurClass(args...); };
    //  auto func = [](CurClass* cur, Types... args) {
    //     auto res = new Capsule();
    //     res->ptr = (void*)(new CurClass(args...));
    //     return res;
    //   };
    std::vector<Value*> inputs = addInputs(func, graph);
    static auto classRegistry =
        torch::RegisterOperators().op(qualFuncName, std::move(func));
    auto capsuleNode =
        graph->insertNode(graph->create(prim::CreateCapsule, {}, 1))
            ->output()
            ->setType(CapsuleType::get());
    auto n = graph->insertNode(
        graph->create(prim::SetAttr, {inputs[0], capsuleNode}, 0));
    n->s_(attr::name, "capsule");
    auto res = graph->insertNode(
        graph->create(Symbol::fromQualString(qualFuncName), inputs, 0));
    graph->registerOutput(
        graph->insertConstant(IValue())->setType(NoneType::get()));
    classCu->create_function("__init__", graph);
    return *this;
  }
  template <typename Func>
  class_& def(string name, Func f) {
    auto res = def_(name, f, detail::args_t<decltype(f)>{});
    return *this;
  }

 private:
  template <class T>
  struct addInput {
    static Value* call(std::shared_ptr<Graph> graph) {
      return graph->addInput()->setType(getTypePtr<T>());
    }
  };
  template <class Func, size_t... arg_indices>
  std::vector<Value*> addInputs_(
      Func f,
      std::shared_ptr<Graph> graph,
      guts::index_sequence<arg_indices...>) {
    using argTypes =
        typename guts::infer_function_traits_t<Func>::parameter_types;
    std::vector<Value*> res = {
        addInput<guts::typelist::element_t<arg_indices, argTypes>>::call(
            graph)...};
    return res;
  }
  template <class Func>
  std::vector<Value*> addInputs(Func f, std::shared_ptr<Graph> graph) {
    constexpr auto numArgs =
        guts::infer_function_traits_t<Func>::number_of_parameters;
    return addInputs_(f, graph, guts::make_index_sequence<numArgs>());
  }
  template <class T>
  void addType(Value* v) {
    v->setType(getTypePtr<T>());
  }
  template <typename Last>
  std::string type_name () {
      return std::string(typeid(Last).name());
  }
  template <typename First, typename Second, typename ...Rest>
  std::string type_name () {
      return type_name<First>() + "_" + type_name<Second, Rest...>();
  }
  template <typename Func, typename R, typename... Types>
  class_& def_(string name, Func f, detail::types<R, Types...> funcInfo) {
    pyClass->def(name.c_str(), f);
    auto qualFuncName = className + "::" + name;
    auto func = [f](CurClass* cur, Types... args) {
      return guts::invoke(f, *cur, args...);
    };
    static auto classRegistry =
        torch::RegisterOperators().op(qualFuncName, std::move(func));

    auto graph = std::make_shared<Graph>();
    std::vector<Value*> inputs = addInputs(func, graph);
    auto methodCall = graph->insertNode(graph->create(
        Symbol::fromQualString(qualFuncName), inputs, funcInfo.hasRet));
    Value* res;
    if (funcInfo.hasRet) {
      res = methodCall->output();
      addType<R>(res);
    } else {
      res = graph->insertConstant(IValue())->setType(NoneType::get());
    }
    graph->registerOutput(res);
    classCu->create_function(name, graph);
    return *this;
  }
};

} // namespace jit

} // namespace torch
