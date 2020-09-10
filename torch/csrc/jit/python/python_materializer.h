#pragma once

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/python/pybind.h>
#include <torch/csrc/jit/serialization/python_print.h>
#include <torch/csrc/jit/serialization/type_name_uniquer.h>

namespace torch {
namespace jit {

class TORCH_API PythonMaterializer {
 public:
  PythonMaterializer(py::object def_cb, py::object obj_cb)
      : def_cb_(std::move(def_cb)), obj_cb_(std::move(obj_cb)) {}

  py::object materialize(const Module& module) {
    class_deps_.push_back(module.type());
    for (size_t i = 0; i < class_deps_.size(); ++i) {
      // note: convertNameType may extend class_deps_, so re-checking
      // .size() is necessary
      convertNamedType(class_deps_[i]);
    }

    py::object rv = pyInstanceFromIValueInstance(module._ivalue());
    return rv;
  }

 private:
  void convertNamedType(const c10::NamedTypePtr& class_type) {
    if (converted_types_.count(class_type)) {
      return;
    }
    converted_types_.insert(class_type);
    auto qualname = type_name_uniquer_.getUniqueName(class_type);
    std::string qualifier = qualname.prefix();

    auto type_printer =
        [&](const c10::ConstTypePtr& t) -> c10::optional<std::string> {
      auto namedType = t->cast<c10::NamedType>();
      if (namedType && namedType->name()) {
        return type_name_uniquer_.getUniqueName(namedType).qualifiedName();
      }
      return c10::nullopt;
    };

    PythonPrint pp(
        constant_table_,
        class_deps_,
        type_printer,
        /*enforce_importable=*/true);

    pp.printNamedType(class_type);
    def_cb_(pp.str(), qualifier);
  }

  py::object pyInstanceFromIValueInstance(ObjectPtr obj) {
    if (obj->type()->findMethod("__setstate__")) {
      throw std::runtime_error("NYI");
    }

    std::unordered_map<std::string, py::object> for_cb;
    for (size_t i = 0; i < obj->type()->numAttributes(); ++i) {
      auto name = obj->type()->getAttributeName(i);
      auto& attr = obj->getSlot(i);
      if (attr.isObject()) {
        for_cb[name] = pyInstanceFromIValueInstance(attr.toObject());
      } else {
        for_cb[name] = toPyObject(attr);
      }
    }

    TORCH_CHECK(obj->type()->name());
    py::object pyobj =
        obj_cb_(obj->type()->name()->qualifiedName(), std::move(for_cb));

    return pyobj;
  }

  py::object def_cb_;
  py::object obj_cb_;

  std::vector<c10::NamedTypePtr> class_deps_;
  std::unordered_set<c10::NamedTypePtr> converted_types_;
  std::vector<at::IValue> constant_table_;
  TypeNameUniquer type_name_uniquer_;
};

} // namespace jit
} // namespace torch
