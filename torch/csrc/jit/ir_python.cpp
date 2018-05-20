#include "torch/csrc/python_headers.h"
#include "ir.h"

#include "torch/csrc/autograd/function.h"

#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <stack>
#include <sstream>
#include <algorithm>
#include <string>

#include "torch/csrc/utils/auto_gil.h"
#include "torch/csrc/utils/python_strings.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

namespace torch { namespace jit {



std::string getPythonName(const PyObject* obj_) {
  AutoGIL gil;
  PyObject* obj = const_cast<PyObject*>(obj_);
  auto v = py::getattr(obj, "__name__", py::str("<python_value>"));
  // if this was a autograd.Function recover the name of the class
  return py::str(v);
}

std::ostream& printPyObject_python(std::ostream & out, const THPObjectPtr& obj) {
  AutoGIL gil;
  auto pyobj = py::handle(const_cast<PyObject*>(obj.get()));
  if (py::isinstance<py::tuple>(pyobj)) {
    // This special-case for printing tuples handles a problem where
    // str((2L, 3L)) outputs "(2L, 3L)" in Python 2 but "(2, 3)"
    // in Python 3.  In order to suppress the L-suffix, we must
    // manually print the string ourselves, calling str() on the
    // sub-elements.
    //
    // This is a fairly fragile fix (What if you have nested tuples
    // in tuples? What if you have dictionaries?) but it seems to hit
    // the cases that are triggered in practice in onnx-pytorch.  Revisit
    // this code if this is not the case.
    //
    // By the way, one non-solution for this problem is to monkeypatch
    // tuple.__str__; this doesn't work because Python doesn't allow
    // monkeypatching methods of built-in types.
    auto pytuple = pyobj.cast<py::tuple>();
    out << "(";
    size_t i = 0;
    for (auto& o : pytuple) {
      if (i > 0) {
        out << ", ";
      }
      THPObjectPtr str(py::str(o).release().ptr());
      out << THPUtils_unpackString(str.get());
      i++;
    }
    if (i == 1) {
      out << ",";
    }
    out << ")";
    return out;
  } else {
    return out << THPUtils_unpackString(py::str(pyobj).ptr());
  }
}

at::optional<THPObjectPtr> PythonOp::autogradFunction() const {
  AutoGIL gil;
  py::handle obj = const_cast<PyObject*>(pyobj.get());

  auto r = py::getattr(obj, "__self__", py::none());
  if(r.is_none())
    return at::nullopt;

  auto apply = py::getattr(r, "apply", py::none());
  if(apply.is_none())
    return at::nullopt;

  auto c = PyObject_RichCompareBool(apply.ptr(), obj.ptr(), Py_NE);
  if(PyErr_Occurred())
    throw py::error_already_set();
  if(c)
    return at::nullopt;

  return THPObjectPtr(r.release().ptr());
}

std::string pythonOpName_python(const PythonOp* t){
  AutoGIL gil;
  if(auto autograd = t->autogradFunction()) {
    return getPythonName(autograd->get());
  } else {
    return getPythonName(t->pyobj.get());
  }
}

void PythonOp::cloneFrom(Node * other_) {
  Node::cloneFrom(other_);
  auto other = other_->cast<PythonOp>();
  this->cconv = other->cconv;
  Py_INCREF(other->pyobj.get());
  this->pyobj = THPObjectPtr(other->pyobj.get());
  this->var_flags = other->var_flags;
  for(auto & sa : other->scalar_args) {
    Py_INCREF(sa.get());
    this->scalar_args.emplace_back(sa.get());
  }
  this->tracing_autograd_python_function =
      other->tracing_autograd_python_function;
}

}} // namespace torch::jit
