#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/jit/python/python_resolver.h>
#include <torch/csrc/jit/python/python_sugared_value.h>

namespace torch {
namespace jit {

std::shared_ptr<SugaredValue> PythonResolver::resolveValue(
    const std::string& name,
    GraphFunction& m,
    const SourceRange& loc) {
  pybind11::gil_scoped_acquire ag;
  py::object obj = rcb_(name);
  if (obj.is(py::none())) {
    return nullptr;
  }
  return toSugaredValue(obj, m, loc);
}

bool PythonResolver::isNamedTupleClass(py::object obj) {
  auto tuple_type = reinterpret_cast<PyObject*>(&PyTuple_Type);
  return PyObject_IsSubclass(obj.ptr(), tuple_type) &&
      py::hasattr(obj, "_fields");
}

TypePtr PythonResolver::resolveTypeFromObject(
    const py::object& obj,
    const SourceRange& loc) {
  if (py::isinstance<ScriptClass>(obj)) {
    auto script_class = py::cast<ScriptClass>(obj);
    return script_class.class_type_.type_;
  }

  py::bool_ isClass = py::module::import("inspect").attr("isclass")(obj);
  if (!py::cast<bool>(isClass)) {
    return nullptr;
  }

  if (isNamedTupleClass(obj)) {
    return registerNamedTuple(obj, loc);
  }

  auto qualifiedName = c10::QualifiedName(py::cast<std::string>(
      py::module::import("torch._jit_internal").attr("_qualified_name")(obj)));

  return get_python_cu()->get_type(qualifiedName);
}

TypePtr PythonResolver::resolveType(
    const std::string& name,
    const SourceRange& loc) {
  if (classType_ && name == classname_) {
    return classType_;
  }
  pybind11::gil_scoped_acquire ag;
  py::object obj = rcb_(name);
  if (obj.is(py::none())) {
    return nullptr;
  }

  auto annotation_type = py::module::import("torch.jit.annotations")
                             .attr("try_ann_to_type")(obj, loc);
  if (!annotation_type.is_none()) {
    return py::cast<TypePtr>(annotation_type);
  }
  return resolveTypeFromObject(obj, loc);
}

} // namespace jit
} // namespace torch
