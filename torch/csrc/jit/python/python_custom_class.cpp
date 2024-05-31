#include <torch/csrc/jit/python/python_custom_class.h>

#include <torch/csrc/jit/frontend/sugared_value.h>

#include <fmt/format.h>

namespace torch::jit {

struct CustomMethodProxy;
struct CustomObjectProxy;

py::object ScriptClass::__call__(py::args args, py::kwargs kwargs) {
  auto instance =
      Object(at::ivalue::Object::create(class_type_, /*numSlots=*/1));
  Function* init_fn = instance.type()->findMethod("__init__");
  TORCH_CHECK(
      init_fn,
      fmt::format(
          "Custom C++ class: '{}' does not have an '__init__' method bound. "
          "Did you forget to add '.def(torch::init<...>)' to its registration?",
          instance.type()->repr_str()));
  Method init_method(instance._ivalue(), init_fn);
  // NOLINTNEXTLINE(performance-move-const-arg)
  invokeScriptMethodFromPython(init_method, std::move(args), std::move(kwargs));
  return py::cast(instance);
}

/// Variant of StrongFunctionPtr, but for static methods of custom classes.
/// They do not belong to compilation units (the custom class method registry
/// serves that purpose in this case), so StrongFunctionPtr cannot be used here.
/// While it is usually unsafe to carry a raw pointer like this, the custom
/// class method registry that owns the pointer is never destroyed.
struct ScriptClassFunctionPtr {
  ScriptClassFunctionPtr(Function* function) : function_(function) {
    TORCH_INTERNAL_ASSERT(function_);
  }
  Function* function_;
};

void initPythonCustomClassBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  py::class_<ScriptClassFunctionPtr>(
      m, "ScriptClassFunction", py::dynamic_attr())
      .def("__call__", [](py::args args, const py::kwargs& kwargs) {
        auto strongPtr = py::cast<ScriptClassFunctionPtr>(args[0]);
        Function& callee = *strongPtr.function_;
        py::object result = invokeScriptFunctionFromPython(
            callee, tuple_slice(std::move(args), 1), kwargs);
        return result;
      });

  py::class_<ScriptClass>(m, "ScriptClass")
      .def("__call__", &ScriptClass::__call__)
      .def(
          "__getattr__",
          [](ScriptClass& self, const std::string& name) {
            // Define __getattr__ so that static functions of custom classes can
            // be used in regular Python.
            auto type = self.class_type_.type_->castRaw<ClassType>();
            TORCH_INTERNAL_ASSERT(type);
            auto* fn = type->findStaticMethod(name);
            if (fn) {
              return ScriptClassFunctionPtr(fn);
            }

            throw AttributeError("%s does not exist", name.c_str());
          })
      .def_property_readonly("__doc__", [](const ScriptClass& self) {
        return self.class_type_.type_->expectRef<ClassType>().doc_string();
      });

  // This function returns a ScriptClass that wraps the constructor
  // of the given class, specified by the qualified name passed in.
  //
  // This is to emulate the behavior in python where instantiation
  // of a class is a call to a code object for the class, where that
  // code object in turn calls __init__. Rather than calling __init__
  // directly, we need a wrapper that at least returns the instance
  // rather than the None return value from __init__
  m.def(
      "_get_custom_class_python_wrapper",
      [](const std::string& ns, const std::string& qualname) {
        std::string full_qualname =
            "__torch__.torch.classes." + ns + "." + qualname;
        auto named_type = getCustomClass(full_qualname);
        TORCH_CHECK(
            named_type,
            fmt::format(
                "Tried to instantiate class '{}.{}', but it does not exist! "
                "Ensure that it is registered via torch::class_",
                ns,
                qualname));
        c10::ClassTypePtr class_type = named_type->cast<ClassType>();
        return ScriptClass(c10::StrongTypePtr(
            std::shared_ptr<CompilationUnit>(), std::move(class_type)));
      });
}

} // namespace torch::jit
