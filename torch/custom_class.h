
#pragma once

#include <ATen/core/function_schema.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/core/stack.h>
#include <c10/util/C++17.h>
#include <c10/util/Metaprogramming.h>
#include <c10/util/TypeList.h>
#include <c10/util/TypeTraits.h>
#include <torch/custom_class_detail.h>
#include <iostream>
#include <sstream>


namespace torch {
namespace jit {

template <class... Types>
detail::types<void, Types...> init() {
  return detail::types<void, Types...>{};
}

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
  // ClassTypePtr classTypePtr;

  const std::string parentModule = "classes";
  const std::string topModule = "__torch__.torch";

 public:
  class_(std::string className_) : className(std::move(className_)) {
    qualClassName = topModule + "." + parentModule + "." + className;

    TORCH_CHECK(!registeredClasses().count(className),
                "C++ class ", className, " has already been "
                "registered!");

    registeredClasses()[className] = detail::RegisteredClassRecord{
        qualClassName,
        typeid(c10::intrusive_ptr<CurClass>).name(),
        typeid(c10::tagged_capsule<CurClass>).name()
      };
    invokeClassRegistrationCallbacks(registeredClasses()[className]);
  }

  template <typename... Types>
  class_& def(detail::types<void, Types...>) { // Used in combination with
                                               // torch::jit::init<...>()
    auto func = [](c10::tagged_capsule<CurClass> self, Types... args) {
      auto classObj = c10::make_intrusive<CurClass>(args...);
      auto genericPtr = c10::static_intrusive_pointer_cast<torch::jit::CustomClassHolder>(classObj);
      auto capsule = IValue(genericPtr);
      auto object = self.ivalue.toObject();
      object->setSlot(0, capsule);
    };

    defineMethod("__init__", std::move(func));
    return *this;
  }
  template <typename Func>
  class_& def(std::string name, Func f) {
    auto wrapped_f = detail::wrap_func<CurClass, Func>(std::move(f));
    defineMethod(std::move(name), std::move(wrapped_f));
    return *this;
  }

  // Pickle
  template <typename GetStateFn, typename SetStateFn>
  class_& def_pickle(GetStateFn&& get_state, SetStateFn&& set_state) {
    static_assert(
        c10::guts::is_stateless_lambda<std::decay_t<GetStateFn>>::value &&
            c10::guts::is_stateless_lambda<std::decay_t<SetStateFn>>::value,
        "torch::jit::pickle_ currently only supports lambdas as "
        "__getstate__ and __setstate__ arguments.");
    def("__getstate__", std::forward<GetStateFn>(get_state));

    // __setstate__ needs to be registered with some custom handling:
    // We need to wrap the invocation of of the user-provided function
    // such that we take the return value (i.e. c10::intrusive_ptr<CurrClass>)
    // and assign it to the `capsule` attribute.
    using SetStateTraits =
        c10::guts::infer_function_traits_t<std::decay_t<SetStateFn>>;
    using SetStateArg = typename c10::guts::typelist::head_t<
        typename SetStateTraits::parameter_types>;
    auto setstate_wrapper = [set_state = std::move(set_state)](
                                c10::tagged_capsule<CurClass> self,
                                SetStateArg&& arg) {
      c10::intrusive_ptr<CurClass> classObj =
          at::guts::invoke(set_state, std::forward<SetStateArg>(arg));
      auto genericPtr =
          c10::static_intrusive_pointer_cast<torch::jit::CustomClassHolder>(
              classObj);
      auto capsule = IValue(genericPtr);
      auto object = self.ivalue.toObject();
      object->setSlot(0, capsule);
    };
    defineMethod(
        "__setstate__",
        detail::wrap_func<CurClass, decltype(setstate_wrapper)>(
            std::move(setstate_wrapper)));

    return *this;
  }

 private:
  template <typename Func>
  void defineMethod(std::string name, Func func) {
    auto qualFuncName = className + "::" + name;
    registeredOps().push_back(
        torch::RegisterOperators().op(qualFuncName, std::move(func)));

    TORCH_INTERNAL_ASSERT(registeredClasses().count(className));
    auto &class_record = registeredClasses()[className];
    TORCH_CHECK(!class_record.registeredMethods.count(name),
                "Method ", name, " on class ", className,
                " has already been defined!");
    class_record.registeredMethods[name] = qualFuncName;
    invokeMethodRegistrationCallbacks(class_record, name);
  }
};

// APIs for listeners. This allows, for example, TorchScript to listen into these class registrations
// and do futher processing.
using ClassRegistrationCallback = std::function<void(const detail::RegisteredClassRecord& /*class*/)>;
using MethodRegistrationCallback = std::function<void(const detail::RegisteredClassRecord& /*class*/, const std::string& /*method name*/)>;

TORCH_API void registerClassRegistrationCallback(ClassRegistrationCallback cb);
TORCH_API void registerMethodRegistrationCallback(MethodRegistrationCallback cb);

} // namespace jit

} // namespace torch
