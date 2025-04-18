#include <torch/csrc/dynamo/framelocals_mapping.h>

#include <torch/csrc/dynamo/cpython_defs.h>
#include <torch/csrc/dynamo/cpython_includes.h>
#include <torch/csrc/dynamo/debug_macros.h>

#include <internal/pycore_code.h>

#if IS_PYTHON_3_11_PLUS

// Our own version of PyFrame_GetLocals.
// Also combines functionality from frame_init_get_vars and frame_get_var.
// PyFrame_GetLocals:
// https://github.com/python/cpython/blob/0325a8a8cdba6c091bcbbb3c995f3bf1d1217012/Objects/frameobject.c#L1213
// frame_init_get_vars:
// https://github.com/python/cpython/blob/0325a8a8cdba6c091bcbbb3c995f3bf1d1217012/Objects/frameobject.c#L1136
// frame_get_var:
// https://github.com/python/cpython/blob/0325a8a8cdba6c091bcbbb3c995f3bf1d1217012/Objects/frameobject.c#L1162
// PyFrame_GetLocals returns the frame locals dict.
// frame_init_get_vars initializes free variables from the closure.
// frame_get_var fetches the variable value from the frame given the index
// NOTE: hidden variables are not included.
// Returns a new reference.
FrameLocalsMapping::FrameLocalsMapping(FrameLocalsFrameType* frame)
    : _code_obj(py::cast<py::object>((PyObject*)F_CODE(frame))) {
  PyCodeObject* co = F_CODE(frame);
  _framelocals.resize(co->co_nlocalsplus, nullptr);

  if (!frame->stacktop) {
    return;
  }

  auto update_framelocals = [&](int i, PyObject* value) {
    _PyLocals_Kind kind = _PyLocals_GetKind(co->co_localspluskinds, i);

    if (kind & CO_FAST_FREE && !(co->co_flags & CO_OPTIMIZED)) {
      return;
    }

#if IS_PYTHON_3_12_PLUS
    if (kind & CO_FAST_HIDDEN) {
      return;
    }
#endif

    if (kind & CO_FAST_FREE) {
      CHECK(value != nullptr && PyCell_Check(value));
      value = PyCell_GET(value);
    }

    DEBUG_CHECK(0 <= i && i < _framelocals.size());
    _framelocals[i] = value;
  };

  auto offset = co->co_nlocalsplus - co->co_nfreevars;
  for (int i = 0; i < offset; i++) {
    update_framelocals(i, frame->localsplus[i]);
  }
  // Get references to closure variables
  PyObject* closure = ((PyFunctionObject*)FUNC(frame))->func_closure;
  for (int i = 0; i < co->co_nfreevars; i++) {
    update_framelocals(offset + i, PyTuple_GET_ITEM(closure, i));
  }

  // NOTE no need to move the instruction pointer to after COPY_FREE_VARS
  // since we don't actually copy free vars from the closure to the frame
  // localsplus.
}

void FrameLocalsMapping::_realize_dict() {
  _dict = py::dict();
  py::tuple framelocals_names = code_framelocals_names(_code_obj);

  auto nlocalsplus = ((PyCodeObject*)_code_obj.ptr())->co_nlocalsplus;
  DEBUG_CHECK(nlocalsplus == _framelocals.size());
  for (int i = 0; i < nlocalsplus; i++) {
    if (_framelocals[i]) {
      _dict[framelocals_names[i]] = _framelocals[i];
    }
  }
}

py::tuple code_framelocals_names(py::handle code) {
  CHECK(PyCode_Check(code.ptr()));
  return py::cast<py::tuple>(((PyCodeObject*)code.ptr())->co_localsplusnames);
}

#else

// Based on
// https://github.com/python/cpython/blob/5f24da9d75bb0150781b17ee4706e93e6bb364ea/Objects/frameobject.c#L1016
FrameLocalsMapping::FrameLocalsMapping(FrameLocalsFrameType* frame)
    : _code_obj(py::cast<py::object>((PyObject*)F_CODE(frame))) {
  PyCodeObject* co = (PyCodeObject*)_code_obj.ptr();
  auto nlocals =
      std::min<int>(co->co_nlocals, (int)PyTuple_GET_SIZE(co->co_varnames));
  auto ncells = PyCode_GetNCellvars(co);
  auto nfree = PyCode_GetNFreevars(co);

  _framelocals.resize(co->co_nlocals + ncells + nfree, nullptr);

  auto update_framelocals = [&](int i, bool deref) {
    DEBUG_CHECK(0 <= i && i < _framelocals.size());
    PyObject* value = frame->f_localsplus[i];
    if (deref) {
      CHECK(value != nullptr && PyCell_Check(value));
      value = PyCell_GET(value);
    }
    _framelocals[i] = value;
  };

  // locals
  for (int i = 0; i < nlocals; i++) {
    update_framelocals(i, false);
  }

  // cellvars
  for (int i = 0; i < ncells; i++) {
    update_framelocals(co->co_nlocals + i, true);
  }

  // freevars
  if (co->co_flags & CO_OPTIMIZED) {
    for (int i = 0; i < nfree; i++) {
      update_framelocals(co->co_nlocals + ncells + i, true);
    }
  }
}

void FrameLocalsMapping::_realize_dict() {
  _dict = py::dict();
  py::tuple framelocals_names = code_framelocals_names(_code_obj);
  PyCodeObject* co = (PyCodeObject*)_code_obj.ptr();

  auto update_mapping = [&](int i) {
    DEBUG_CHECK(0 <= i && i < _framelocals.size());
    PyObject* value = _framelocals[i].ptr();
    if (value == nullptr) {
      _dict.attr("pop")(framelocals_names[i], py::none());
    } else {
      _dict[framelocals_names[i]] = value;
    }
  };

  // locals
  py::tuple varnames = _code_obj.attr("co_varnames");
  auto nlocals = std::min(co->co_nlocals, (int)varnames.size());
  for (int i = 0; i < nlocals; i++) {
    update_mapping(i);
  }

  // cellvars
  auto ncells = PyCode_GetNCellvars(co);
  for (int i = 0; i < ncells; i++) {
    update_mapping(co->co_nlocals + i);
  }

  // freevars
  if (co->co_flags & CO_OPTIMIZED) {
    auto nfree = PyCode_GetNFreevars(co);
    for (int i = 0; i < nfree; i++) {
      update_mapping(co->co_nlocals + ncells + i);
    }
  }
}

py::tuple code_framelocals_names(py::handle code) {
  CHECK(PyCode_Check(code.ptr()));
  py::tuple names = code.attr("co_varnames") + code.attr("co_cellvars");
  if (((PyCodeObject*)code.ptr())->co_flags & CO_OPTIMIZED) {
    names += code.attr("co_freevars");
  }
  return names;
}

#endif

PyObject* FrameLocalsMapping::get(int idx) {
  DEBUG_CHECK(0 <= idx && idx < _framelocals.size());
  return _framelocals[idx].ptr();
}

PyDictObject* framelocals_mapping_to_dict(FrameLocalsMapping* map) {
  return map->to_dict();
}
