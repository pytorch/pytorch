#include <torch/csrc/jit/python/python_custom_class.h>
#include <torch/csrc/jit/frontend/sugared_value.h>

namespace torch {
namespace jit {

struct CustomMethodProxy;
struct CustomObjectProxy;

py::object ScriptClass::__call__(py::args args, py::kwargs kwargs) {
  auto instance =
      Object(at::ivalue::Object::create(class_type_, /*numSlots=*/1));
  return invokeScriptMethodFromPython(
      instance, "__init__", std::move(args), std::move(kwargs));
}

void initPythonCustomClassBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  py::class_<ScriptClass>(m, "ScriptClass")
      .def("__call__", &ScriptClass::__call__);

  // This function returns a ScriptClass that wraps the constructor
  // of the given class, specified by the qualified name passed in.
  //
  // This is to emulate the behavior in python where instantiation
  // of a class is a call to a code object for the class, where that
  // code object in turn calls __init__. Rather than calling __init__
  // directly, we need a wrapper that at least returns the instance
  // rather than the None return value from __init__
  m.def("_get_custom_class_python_wrapper", [](const std::string& qualname) {
    std::string full_qualname = "__torch__.torch.classes." + qualname;
    auto named_type = getCustomClass(full_qualname);
    c10::ClassTypePtr class_type = named_type->cast<ClassType>();
    return ScriptClass(c10::StrongTypePtr(
        std::shared_ptr<CompilationUnit>(), std::move(class_type)));
  });
}

} // namespace jit
} // namespace torch
