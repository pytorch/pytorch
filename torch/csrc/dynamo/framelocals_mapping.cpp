#include <torch/csrc/dynamo/framelocals_mapping.h>

#if IS_PYTHON_3_12_PLUS
#include <torch/csrc/dynamo/cpython_defs.h>
#include <torch/csrc/dynamo/cpython_includes.h>
#include <torch/csrc/dynamo/debug_macros.h>
#include <torch/csrc/utils/pybind.h>

#include <internal/pycore_code.h>

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
PyObject* get_framelocals_mapping(_PyInterpreterFrame* frame) {
  if (!frame->stacktop) {
    return py::dict().release().ptr();
  }

  PyCodeObject* co = F_CODE(frame);
  py::dict mapping;

  auto update_mapping = [&](int i, PyObject* value) {
    _PyLocals_Kind kind = _PyLocals_GetKind(co->co_localspluskinds, i);

    if (kind & CO_FAST_FREE && !(co->co_flags & CO_OPTIMIZED)) {
      return;
    }
    if (kind & CO_FAST_HIDDEN) {
      return;
    }

    if (kind & CO_FAST_FREE) {
      CHECK(value != nullptr && PyCell_Check(value));
      value = PyCell_GET(value);
    }

    if (value != nullptr) {
      py::str name =
          py::cast<py::str>(PyTuple_GET_ITEM(co->co_localsplusnames, i));
      mapping[name] = py::cast<py::object>(value);
    }
  };

  int offset = co->co_nlocalsplus - co->co_nfreevars;
  for (int i = 0; i < offset; i++) {
    update_mapping(i, frame->localsplus[i]);
  }
  // Get references to closure variables
  PyObject* closure = ((PyFunctionObject*)FUNC(frame))->func_closure;
  for (int i = 0; i < co->co_nfreevars; ++i) {
    update_mapping(offset + i, PyTuple_GET_ITEM(closure, i));
  }

  // NOTE no need to move the instruction pointer to after COPY_FREE_VARS
  // since we don't actually copy free vars from the closure to the frame
  // localsplus.

  return mapping.release().ptr();
}

#endif
