#pragma once

#include <ATen/core/builtin_function.h>
#include <ATen/core/function_schema.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/class_type.h>
#include <ATen/core/op_registration/infer_schema.h>
#include <ATen/core/stack.h>
#include <c10/util/C++17.h>
#include <c10/util/Metaprogramming.h>
#include <c10/util/TypeList.h>
#include <c10/util/TypeTraits.h>
#include <torch/custom_class_detail.h>
#include <torch/library.h>
#include <sstream>

namespace torch {

/// This function is used in conjunction with `class_::def()` to register
/// a constructor for a given C++ class type. For example,
/// `torch::init<int, std::string>()` would register a two-argument constructor
/// taking an `int` and a `std::string` as argument.
template <class... Types>
detail::types<void, Types...> init() {
  return detail::types<void, Types...>{};
}

template <typename Func, typename... ParameterTypeList>
struct InitLambda {
  Func f;
};

template <typename Func>
decltype(auto) init(Func&& f) {
  using InitTraits = c10::guts::infer_function_traits_t<std::decay_t<Func>>;
  using ParameterTypeList = typename InitTraits::parameter_types;

  InitLambda<Func, ParameterTypeList> init{std::forward<Func>(f)};
  return init;
}

/// Entry point for custom C++ class registration. To register a C++ class
/// in PyTorch, instantiate `torch::class_` with the desired class as the
/// template parameter. Typically, this instantiation should be done in
/// the initialization of a global variable, so that the class will be
/// made available on dynamic library loading without any additional API
/// calls needed. For example, to register a class named Foo, you might
/// create a global variable like so:
///
///     static auto register_foo = torch::class_<Foo>("myclasses", "Foo")
///       .def("myMethod", &Foo::myMethod)
///       .def("lambdaMethod", [](const c10::intrusive_ptr<Foo>& self) {
///         // Do something with `self`
///       });
///
/// In addition to registering the class, this registration also chains
/// `def()` calls to register methods. `myMethod()` is registered with
/// a pointer to the Foo class's `myMethod()` method. `lambdaMethod()`
/// is registered with a C++ lambda expression.
template <class CurClass>
class class_ : public ::torch::detail::class_base {
  static_assert(
      std::is_base_of_v<CustomClassHolder, CurClass>,
      "torch::class_<T> requires T to inherit from CustomClassHolder");

 public:
  /// This constructor actually registers the class type.
  /// String argument `namespaceName` is an identifier for the
  /// namespace you would like this class to appear in.
  /// String argument `className` is the name you would like to
  /// see this class exposed as in Python and TorchScript. For example, if
  /// you pass `foo` as the namespace name and `Bar` as the className, the
  /// class will appear as `torch.classes.foo.Bar` in Python and TorchScript
  explicit class_(
      const std::string& namespaceName,
      const std::string& className,
      std::string doc_string = "")
      : class_base(
            namespaceName,
            className,
            std::move(doc_string),
            typeid(c10::intrusive_ptr<CurClass>),
            typeid(c10::tagged_capsule<CurClass>)) {}

  /// def() can be used in conjunction with `torch::init()` to register
  /// a constructor for a given C++ class type. For example, passing
  /// `torch::init<int, std::string>()` would register a two-argument
  /// constructor taking an `int` and a `std::string` as argument.
  template <typename... Types>
  class_& def(
      torch::detail::types<void, Types...>,
      std::string doc_string = "",
      std::initializer_list<arg> default_args =
          {}) { // Used in combination with
    // torch::init<...>()
    auto func = [](c10::tagged_capsule<CurClass> self, Types... args) {
      auto classObj = c10::make_intrusive<CurClass>(args...);
      auto object = self.ivalue.toObject();
      object->setSlot(0, c10::IValue::make_capsule(std::move(classObj)));
    };

    defineMethod(
        "__init__",
        std::move(func),
        std::move(doc_string),
        default_args);
    return *this;
  }

  // Used in combination with torch::init([]lambda(){......})
  template <typename Func, typename... ParameterTypes>
  class_& def(
      InitLambda<Func, c10::guts::typelist::typelist<ParameterTypes...>> init,
      std::string doc_string = "",
      std::initializer_list<arg> default_args = {}) {
    auto init_lambda_wrapper = [func = std::move(init.f)](
                                   c10::tagged_capsule<CurClass> self,
                                   ParameterTypes... arg) {
      c10::intrusive_ptr<CurClass> classObj =
          at::guts::invoke(func, std::forward<ParameterTypes>(arg)...);
      auto object = self.ivalue.toObject();
      object->setSlot(0, c10::IValue::make_capsule(classObj));
    };

    defineMethod(
        "__init__",
        std::move(init_lambda_wrapper),
        std::move(doc_string),
        default_args);

    return *this;
  }

  /// This is the normal method registration API. `name` is the name that
  /// the method will be made accessible by in Python and TorchScript.
  /// `f` is a callable object that defines the method. Typically `f`
  /// will either be a pointer to a method on `CurClass`, or a lambda
  /// expression that takes a `c10::intrusive_ptr<CurClass>` as the first
  /// argument (emulating a `this` argument in a C++ method.)
  ///
  /// Examples:
  ///
  ///     // Exposes method `foo` on C++ class `Foo` as `call_foo()` in
  ///     // Python and TorchScript
  ///     .def("call_foo", &Foo::foo)
  ///
  ///     // Exposes the given lambda expression as method `call_lambda()`
  ///     // in Python and TorchScript.
  ///     .def("call_lambda", [](const c10::intrusive_ptr<Foo>& self) {
  ///       // do something
  ///     })
  template <typename Func>
  class_& def(
      std::string name,
      Func f,
      std::string doc_string = "",
      std::initializer_list<arg> default_args = {}) {
    auto wrapped_f = detail::wrap_func<CurClass, Func>(std::move(f));
    defineMethod(
        std::move(name),
        std::move(wrapped_f),
        std::move(doc_string),
        default_args);
    return *this;
  }

  /// Method registration API for static methods.
  template <typename Func>
  class_& def_static(std::string name, Func func, std::string doc_string = "") {
    auto qualMethodName = qualClassName + "." + name;
    auto schema =
        c10::inferFunctionSchemaSingleReturn<Func>(std::move(name), "");

    auto wrapped_func =
        [func = std::move(func)](jit::Stack& stack) mutable -> void {
      using RetType =
          typename c10::guts::infer_function_traits_t<Func>::return_type;
      detail::BoxedProxy<RetType, Func>()(stack, func);
    };
    auto method = std::make_unique<jit::BuiltinOpFunction>(
        std::move(qualMethodName),
        std::move(schema),
        std::move(wrapped_func),
        std::move(doc_string));

    classTypePtr->addStaticMethod(method.get());
    registerCustomClassMethod(std::move(method));
    return *this;
  }

  /// Property registration API for properties with both getter and setter
  /// functions.
  template <typename GetterFunc, typename SetterFunc>
  class_& def_property(
      const std::string& name,
      GetterFunc getter_func,
      SetterFunc setter_func,
      std::string doc_string = "") {
    torch::jit::Function* getter{};
    torch::jit::Function* setter{};

    auto wrapped_getter =
        detail::wrap_func<CurClass, GetterFunc>(std::move(getter_func));
    getter = defineMethod(name + "_getter", wrapped_getter, doc_string);

    auto wrapped_setter =
        detail::wrap_func<CurClass, SetterFunc>(std::move(setter_func));
    setter = defineMethod(name + "_setter", wrapped_setter, doc_string);

    classTypePtr->addProperty(name, getter, setter);
    return *this;
  }

  /// Property registration API for properties with only getter function.
  template <typename GetterFunc>
  class_& def_property(
      const std::string& name,
      GetterFunc getter_func,
      std::string doc_string = "") {
    torch::jit::Function* getter{};

    auto wrapped_getter =
        detail::wrap_func<CurClass, GetterFunc>(std::move(getter_func));
    getter = defineMethod(name + "_getter", wrapped_getter, doc_string);

    classTypePtr->addProperty(name, getter, nullptr);
    return *this;
  }

  /// Property registration API for properties with read-write access.
  template <typename T>
  class_& def_readwrite(const std::string& name, T CurClass::*field) {
    auto getter_func = [field =
                            field](const c10::intrusive_ptr<CurClass>& self) {
      return self.get()->*field;
    };

    auto setter_func = [field = field](
                           const c10::intrusive_ptr<CurClass>& self, T value) {
      self.get()->*field = value;
    };

    return def_property(name, getter_func, setter_func);
  }

  /// Property registration API for properties with read-only access.
  template <typename T>
  class_& def_readonly(const std::string& name, T CurClass::*field) {
    auto getter_func =
        [field = std::move(field)](const c10::intrusive_ptr<CurClass>& self) {
          return self.get()->*field;
        };

    return def_property(name, getter_func);
  }

  /// This is an unsafe method registration API added for adding custom JIT
  /// backend support via custom C++ classes. It is not for general purpose use.
  class_& _def_unboxed(
      const std::string& name,
      std::function<void(jit::Stack&)> func,
      c10::FunctionSchema schema,
      std::string doc_string = "") {
    auto method = std::make_unique<jit::BuiltinOpFunction>(
        qualClassName + "." + name,
        std::move(schema),
        std::move(func),
        std::move(doc_string));
    classTypePtr->addMethod(method.get());
    registerCustomClassMethod(std::move(method));
    return *this;
  }

  /// def_pickle() is used to define exactly what state gets serialized
  /// or deserialized for a given instance of a custom C++ class in
  /// Python or TorchScript. This protocol is equivalent to the Pickle
  /// concept of `__getstate__` and `__setstate__` from Python
  /// (https://docs.python.org/2/library/pickle.html#object.__getstate__)
  ///
  /// Currently, both the `get_state` and `set_state` callables must be
  /// C++ lambda expressions. They should have the following signatures,
  /// where `CurClass` is the class you're registering and `T1` is some object
  /// that encapsulates the state of the object.
  ///
  ///     __getstate__(intrusive_ptr<CurClass>) -> T1
  ///     __setstate__(T2) -> intrusive_ptr<CurClass>
  ///
  /// `T1` must be an object that is convertable to IValue by the same rules
  /// for custom op/method registration.
  ///
  /// For the common case, T1 == T2. T1 can also be a subtype of T2. An
  /// example where it makes sense for T1 and T2 to differ is if __setstate__
  /// handles legacy formats in a backwards compatible way.
  ///
  /// Example:
  ///
  ///     .def_pickle(
  ///         // __getstate__
  ///         [](const c10::intrusive_ptr<MyStackClass<std::string>>& self) {
  ///           return self->stack_;
  ///         },
  ///         [](std::vector<std::string> state) { // __setstate__
  ///            return c10::make_intrusive<MyStackClass<std::string>>(
  ///               std::vector<std::string>{"i", "was", "deserialized"});
  ///         })
  template <typename GetStateFn, typename SetStateFn>
  // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
  class_& def_pickle(GetStateFn&& get_state, SetStateFn&& set_state) {
    static_assert(
        c10::guts::is_stateless_lambda<std::decay_t<GetStateFn>>::value &&
            c10::guts::is_stateless_lambda<std::decay_t<SetStateFn>>::value,
        "def_pickle() currently only supports lambdas as "
        "__getstate__ and __setstate__ arguments.");
    def("__getstate__", std::forward<GetStateFn>(get_state));

    // __setstate__ needs to be registered with some custom handling:
    // We need to wrap the invocation of the user-provided function
    // such that we take the return value (i.e. c10::intrusive_ptr<CurrClass>)
    // and assign it to the `capsule` attribute.
    using SetStateTraits =
        c10::guts::infer_function_traits_t<std::decay_t<SetStateFn>>;
    using SetStateArg = typename c10::guts::typelist::head_t<
        typename SetStateTraits::parameter_types>;
    auto setstate_wrapper = [set_state = std::forward<SetStateFn>(set_state)](
                                c10::tagged_capsule<CurClass> self,
                                SetStateArg arg) {
      c10::intrusive_ptr<CurClass> classObj =
          at::guts::invoke(set_state, std::move(arg));
      auto object = self.ivalue.toObject();
      object->setSlot(0, c10::IValue::make_capsule(classObj));
    };
    defineMethod(
        "__setstate__",
        detail::wrap_func<CurClass, decltype(setstate_wrapper)>(
            std::move(setstate_wrapper)));

    // type validation
    auto getstate_schema = classTypePtr->getMethod("__getstate__").getSchema();
#ifndef STRIP_ERROR_MESSAGES
    auto format_getstate_schema = [&getstate_schema]() {
      std::stringstream ss;
      ss << getstate_schema;
      return ss.str();
    };
#endif
    TORCH_CHECK(
        getstate_schema.arguments().size() == 1,
        "__getstate__ should take exactly one argument: self. Got: ",
        format_getstate_schema());
    auto first_arg_type = getstate_schema.arguments().at(0).type();
    TORCH_CHECK(
        *first_arg_type == *classTypePtr,
        "self argument of __getstate__ must be the custom class type. Got ",
        first_arg_type->repr_str());
    TORCH_CHECK(
        getstate_schema.returns().size() == 1,
        "__getstate__ should return exactly one value for serialization. Got: ",
        format_getstate_schema());

    auto ser_type = getstate_schema.returns().at(0).type();
    auto setstate_schema = classTypePtr->getMethod("__setstate__").getSchema();
    auto arg_type = setstate_schema.arguments().at(1).type();
    TORCH_CHECK(
        ser_type->isSubtypeOf(*arg_type),
        "__getstate__'s return type should be a subtype of "
        "input argument of __setstate__. Got ",
        ser_type->repr_str(),
        " but expected ",
        arg_type->repr_str());

    return *this;
  }

 private:
  template <typename Func>
  torch::jit::Function* defineMethod(
      std::string name,
      Func func,
      std::string doc_string = "",
      std::initializer_list<arg> default_args = {}) {
    auto qualMethodName = qualClassName + "." + name;
    auto schema =
        c10::inferFunctionSchemaSingleReturn<Func>(std::move(name), "");

    // If default values are provided for function arguments, there must be
    // none (no default values) or default values for all function
    // arguments, except for self. This is because argument names are not
    // extracted by inferFunctionSchemaSingleReturn, and so there must be a
    // torch::arg instance in default_args even for arguments that do not
    // have an actual default value provided.
    TORCH_CHECK(
        default_args.size() == 0 ||
            default_args.size() == schema.arguments().size() - 1,
        "Default values must be specified for none or all arguments");

    // If there are default args, copy the argument names and default values to
    // the function schema.
    if (default_args.size() > 0) {
      schema = withNewArguments(schema, default_args);
    }

    auto wrapped_func =
        [func = std::move(func)](jit::Stack& stack) mutable -> void {
      // TODO: we need to figure out how to profile calls to custom functions
      // like this! Currently can't do it because the profiler stuff is in
      // libtorch and not ATen
      using RetType =
          typename c10::guts::infer_function_traits_t<Func>::return_type;
      detail::BoxedProxy<RetType, Func>()(stack, func);
    };
    auto method = std::make_unique<jit::BuiltinOpFunction>(
        qualMethodName,
        std::move(schema),
        std::move(wrapped_func),
        std::move(doc_string));

    // Register the method here to keep the Method alive.
    // ClassTypes do not hold ownership of their methods (normally it
    // those are held by the CompilationUnit), so we need a proxy for
    // that behavior here.
    auto method_val = method.get();
    classTypePtr->addMethod(method_val);
    registerCustomClassMethod(std::move(method));
    return method_val;
  }
};

/// make_custom_class() is a convenient way to create an instance of a
/// registered custom class and wrap it in an IValue, for example when you want
/// to pass the object to TorchScript. Its syntax is equivalent to APIs like
/// `std::make_shared<>` or `c10::make_intrusive<>`.
///
/// For example, if you have a custom C++ class that can be constructed from an
/// `int` and `std::string`, you might use this API like so:
///
///     IValue custom_class_iv = torch::make_custom_class<MyClass>(3,
///     "foobarbaz");
template <typename CurClass, typename... CtorArgs>
c10::IValue make_custom_class(CtorArgs&&... args) {
  auto userClassInstance =
      c10::make_intrusive<CurClass>(std::forward<CtorArgs>(args)...);
  return c10::IValue(std::move(userClassInstance));
}

// Alternative api for creating a torchbind class over torch::class_ this api is
// preffered to prevent size regressions on Edge usecases. Must be used in
// conjunction with TORCH_SELECTIVE_CLASS macro aka
// selective_class<foo>("foo_namespace", TORCH_SELECTIVE_CLASS("foo"))
template <class CurClass>
inline class_<CurClass> selective_class_(
    const std::string& namespace_name,
    detail::SelectiveStr<true> className) {
  auto class_name = std::string(className.operator const char*());
  return torch::class_<CurClass>(namespace_name, class_name);
}

template <class CurClass>
inline detail::ClassNotSelected selective_class_(
    const std::string&,
    detail::SelectiveStr<false>) {
  return detail::ClassNotSelected();
}

// jit namespace for backward-compatibility
// We previously defined everything in torch::jit but moved it out to
// better reflect that these features are not limited only to TorchScript
namespace jit {

using ::torch::class_;
using ::torch::getCustomClass;
using ::torch::init;
using ::torch::isCustomClass;

} // namespace jit

template <class CurClass>
inline class_<CurClass> Library::class_(const std::string& className) {
  TORCH_CHECK(
      kind_ == DEF || kind_ == FRAGMENT,
      "class_(\"",
      className,
      "\"): Cannot define a class inside of a TORCH_LIBRARY_IMPL block.  "
      "All class_()s should be placed in the (unique) TORCH_LIBRARY block for their namespace.  "
      "(Error occurred at ",
      file_,
      ":",
      line_,
      ")");
  TORCH_INTERNAL_ASSERT(ns_.has_value(), file_, ":", line_);
  return torch::class_<CurClass>(*ns_, className);
}

const std::unordered_set<std::string> getAllCustomClassesNames();

template <class CurClass>
inline class_<CurClass> Library::class_(detail::SelectiveStr<true> className) {
  auto class_name = std::string(className.operator const char*());
  TORCH_CHECK(
      kind_ == DEF || kind_ == FRAGMENT,
      "class_(\"",
      class_name,
      "\"): Cannot define a class inside of a TORCH_LIBRARY_IMPL block.  "
      "All class_()s should be placed in the (unique) TORCH_LIBRARY block for their namespace.  "
      "(Error occurred at ",
      file_,
      ":",
      line_,
      ")");
  TORCH_INTERNAL_ASSERT(ns_.has_value(), file_, ":", line_);
  return torch::class_<CurClass>(*ns_, class_name);
}

template <class CurClass>
inline detail::ClassNotSelected Library::class_(detail::SelectiveStr<false>) {
  return detail::ClassNotSelected();
}

} // namespace torch
