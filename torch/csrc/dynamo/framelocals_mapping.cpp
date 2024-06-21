#include <torch/csrc/dynamo/cpython_defs.h>
#include <torch/csrc/dynamo/cpython_includes.h>
#include <torch/csrc/dynamo/debug_macros.h>
#include <torch/csrc/dynamo/framelocals_mapping.h>

#ifdef IS_PYTHON_3_11_PLUS
#include <internal/pycore_code.h>
#endif

FrameLocalsMapping::FrameLocalsMapping(THP_EVAL_API_FRAME_OBJECT* frame) {
  PyCodeObject* co = F_CODE(frame);

  auto getname = [&](int i) {
    return std::string(
        PyUnicode_AsUTF8(PyTuple_GET_ITEM(co->co_localsplusnames, i)));
  };

  auto ishidden = [&](int i) {
#if IS_PYTHON_3_12_PLUS
    _PyLocals_Kind kind = _PyLocals_GetKind(co->co_localspluskinds, i);
    return bool(kind & CO_FAST_HIDDEN);
#else
    return false;
#endif
  };

  auto maybe_unwrap_cell = [&](int i, PyObject* value) {
    _PyLocals_Kind kind = _PyLocals_GetKind(co->co_localspluskinds, i);
    if (kind == CO_FAST_FREE || kind & CO_FAST_CELL) {
      CHECK(PyCell_Check(value));
      return PyCell_GET(value);
    }
    return value;
  };

  int offset = co->co_nlocalsplus - co->co_nfreevars;
  for (int i = 0; i < offset; i++) {
    if (ishidden(i)) {
      continue;
    }
    std::string name = getname(i);
    this->mapping[name] =
        py::cast<py::object>(maybe_unwrap_cell(i, frame->localsplus[i]));
  }
  // Get references to closure variables
  PyObject* closure = ((PyFunctionObject*)FUNC(frame))->func_closure;
  for (int i = 0; i < co->co_nfreevars; ++i) {
    if (ishidden(offset + i)) {
      continue;
    }
    std::string name = getname(offset + i);
    PyObject* o = PyTuple_GET_ITEM(closure, i);
    this->mapping[name] =
        py::cast<py::object>(maybe_unwrap_cell(offset + i, o));
  }
}

py::handle FrameLocalsMapping::getitem(const std::string& key) const {
  printf("get item %s\n", key.c_str());
  return this->mapping.at(key);
}

PyObject* FrameLocalsMapping_new(THP_EVAL_API_FRAME_OBJECT* frame) {
  FrameLocalsMapping* mapping = new FrameLocalsMapping(frame);
  NULL_CHECK(mapping);
  return py::cast(mapping, py::return_value_policy::reference).release().ptr();
}

void FrameLocalsMapping_delete(PyObject** obj) {
  FrameLocalsMapping* mapping = py::cast<FrameLocalsMapping*>(py::handle(*obj));
  delete mapping;
  *obj = nullptr;
}

PyObject* FrameLocalsMapping_todict(PyObject* obj) {
  FrameLocalsMapping* mapping = py::cast<FrameLocalsMapping*>(py::handle(obj));
  return py::cast(mapping->mapping, py::return_value_policy::reference)
      .release()
      .ptr();
}
