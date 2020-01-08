
#pragma once

#include <ATen/core/function_schema.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/core/stack.h>
#include <c10/util/C++17.h>
#include <c10/util/Metaprogramming.h>
#include <c10/util/TypeList.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/jit/operator.h>
#include <torch/csrc/jit/pybind_utils.h>
#include <torch/csrc/jit/script/compilation_unit.h>
#include <torch/csrc/jit/tracer.h>
#include <torch/csrc/utils/variadic.h>
#include <iostream>
#include <sstream>


namespace py = pybind11;
namespace torch {
namespace jit {

static std::vector<c10::RegisterOperators> registeredOps;

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
detail::types<void, Types...> init() { return detail::types<void, Types...>{}; }

// To bind custom classes into Torchscript, use an API very similar to Pybind's.
// Currently exposes one class `torch::jit::class_<T>` and 2 methods.
// - Constructing `torch::jit::class_<Foo>` registers `Foo` in Python and
// Torchscript, and puts it under `torch.classes.Foo` in Python.
// - torch::jit::class_<Foo>.def("method1", &Foo::method1) does some template
// metaprogramming to introspect the function types and register the operator
// for use in Torchscript.
// - torch::jit::class_<Foo>.def(torch::jit::init<int64_t, int64_t>()) registers
// the Foo(int, int) constructor.
// see test/custom_operator/classes.cpp and
// test/custom_operator/test_custom_classes.py for example usages

template <class CurClass>
class class_ {
  static_assert(std::is_base_of<CustomClassHolder, CurClass>::value,
    "torch::jit::class_<T> requires T to inherit from CustomClassHolder");

  std::string className;
  std::string qualClassName;
  c10::optional<py::class_<CurClass>> pyClass = c10::nullopt;
  std::shared_ptr<script::CompilationUnit> classCu = nullptr;
  ClassTypePtr classTypePtr;

  const std::string parentModule = "classes";
  const std::string topModule = "__torch__.torch";

 public:
  class_(std::string className_) : className(std::move(className_)) {
    // Currently we register everything as a python class just for convenience.
    // We'll want to remove this at some point to get rid of the python
    // dependency. It would require significant changes to class registration,
    // (I think)?
    qualClassName = topModule + "." + parentModule + "." + className;

    auto obj = py::module::import("torch").attr(parentModule.c_str());
    pyClass = py::class_<CurClass>(obj, className.c_str());
    pyClass->attr("qualified_name") = py::str(qualClassName);
    auto newClass =
        py::module::import("torch.jit")
            .attr("_add_script_class")(*pyClass, qualClassName.c_str());

    auto castToPython = [](void* objPtr) -> PyObject* {
      CurClass x = *static_cast<CurClass*>(objPtr);
      auto py_object = py::cast(x);
      PyObject* rawPyObj = py_object.release().ptr();
      return rawPyObj;
    };
    at::getClassConverter()[qualClassName] = castToPython;

    // We currently represent custom classes as torchscript classes with a
    // capsule attribute
    classCu = torch::jit::get_python_cu();
    classTypePtr =
        ClassType::create(c10::QualifiedName(qualClassName), classCu);
    classTypePtr->addAttribute("capsule", CapsuleType::get());

    c10::getCustomClassTypeMap().insert({typeid(c10::intrusive_ptr<CurClass>).name(),
                              c10::StrongTypePtr(classCu, classTypePtr)});
    c10::getCustomClassTypeMap().insert({typeid(c10::tagged_capsule<CurClass>).name(),
                              c10::StrongTypePtr(classCu, classTypePtr)});

    classCu->register_type(classTypePtr);
  }

  template <typename... Types>
  class_& def(detail::types<void, Types...>) { // Used in combination with
                                               // torch::jit::init<...>()
    pyClass->def(py::init<Types...>());

    auto func = [](c10::tagged_capsule<CurClass> self, Types... args) {
      auto classObj = c10::make_intrusive<CurClass>(args...);
      auto genericPtr = c10::static_intrusive_pointer_cast<torch::jit::CustomClassHolder>(classObj);
      auto capsule = IValue(genericPtr);
      auto object = self.ivalue.toObject();
      object->setSlot(0, capsule);
    };

    defineMethod<void>("__init__", std::move(func), false);
    return *this;
  }
  template <typename Func>
  class_& def(std::string name, Func f) {
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
      at::guts::index_sequence<arg_indices...>) {
    using argTypes =
        typename at::guts::infer_function_traits_t<Func>::parameter_types;
    std::vector<Value*> res = {
        addInput<at::guts::typelist::element_t<arg_indices, argTypes>>::call(
            graph)...};
    return res;
  }
  template <class Func>
  std::vector<Value*> addInputs(Func f, std::shared_ptr<Graph> graph) {
    constexpr auto numArgs =
        at::guts::infer_function_traits_t<Func>::number_of_parameters;
    return addInputs_(f, graph, at::guts::make_index_sequence<numArgs>());
  }

  template <typename Last>
  std::string type_name() {
    return std::string(typeid(Last).name());
  }
  template <typename First, typename Second, typename... Rest>
  std::string type_name() {
    return type_name<First>() + "_" + type_name<Second, Rest...>();
  }

  template <class T>
  void addType(Value* v) {
    v->setType(getTypePtr<T>());
  }
  template<typename R, typename Func>
  void defineMethod(std::string name, Func func, bool hasRet) {
    auto graph = std::make_shared<Graph>();
    auto qualFuncName = className + "::" + name;
    registeredOps.push_back(
        torch::RegisterOperators().op(qualFuncName, std::move(func)));


    std::vector<Value*> inputs = addInputs(func, graph);
    auto methodCall = graph->insertNode(graph->create(
        Symbol::fromQualString(qualFuncName), inputs, hasRet));
    Value* res;
    if (hasRet) {
      res = methodCall->output();
      addType<R>(res);
    } else {
      res = graph->insertConstant(IValue())->setType(NoneType::get());
    }
    graph->registerOutput(res);

    auto method = classCu->create_function(qualClassName + "." + name, graph);
    classTypePtr->addMethod(method);
  }
  template <typename Func, typename R, typename... Types>
  class_& def_(std::string name, Func f, detail::types<R, Types...> funcInfo) {
    pyClass->def(name.c_str(), f);

    auto func = [f](c10::intrusive_ptr<CurClass> cur, Types... args) {
      return at::guts::invoke(f, *cur, args...);
    };
    defineMethod<R>(name, std::move(func), funcInfo.hasRet);
    return *this;
  }
};

} // namespace jit

} // namespace torch
