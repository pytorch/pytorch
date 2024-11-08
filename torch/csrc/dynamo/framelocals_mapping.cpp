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
FrameLocalsMapping* get_framelocals_mapping(_PyInterpreterFrame* frame) {
  if (!frame->stacktop) {
    return new FrameLocalsMapping();
  }

  PyCodeObject* co = F_CODE(frame);
  FrameLocalsMapping* map = new FrameLocalsMapping();

  auto update_mapping = [&](int i, PyObject* value) {
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

    if (value != nullptr) {
      auto name = PyUnicode_AsUTF8(PyTuple_GET_ITEM(co->co_localsplusnames, i));
      map->set(name, value);
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

  return map;
}

#else

// Based on
// https://github.com/python/cpython/blob/5f24da9d75bb0150781b17ee4706e93e6bb364ea/Objects/frameobject.c#L1016
FrameLocalsMapping* get_framelocals_mapping(PyFrameObject* frame) {
  PyCodeObject* co = F_CODE(frame);
  FrameLocalsMapping* map = new FrameLocalsMapping();

  auto update_mapping =
      [&](PyObject* names, int i, PyObject* value, bool deref) {
        auto name = PyUnicode_AsUTF8(PyTuple_GET_ITEM(names, i));
        if (deref) {
          CHECK(value != nullptr && PyCell_Check(value));
          value = PyCell_GET(value);
        }
        if (value == nullptr) {
          map->erase(name);
        } else {
          map->set(name, value);
        }
      };

  // locals
  int nlocals = PyTuple_GET_SIZE(co->co_varnames);
  if (nlocals > co->co_nlocals) {
    nlocals = co->co_nlocals;
  }
  for (int i = 0; i < nlocals; i++) {
    update_mapping(co->co_varnames, i, frame->f_localsplus[i], false);
  }

  // cellvars
  int ncells = PyTuple_GET_SIZE(co->co_cellvars);
  for (int i = 0; i < ncells; i++) {
    update_mapping(
        co->co_cellvars, i, frame->f_localsplus[co->co_nlocals + i], true);
  }

  // freevars
  if (co->co_flags & CO_OPTIMIZED) {
    int nfree = PyTuple_GET_SIZE(co->co_freevars);
    for (int i = 0; i < nfree; i++) {
      update_mapping(
          co->co_freevars,
          i,
          frame->f_localsplus[co->co_nlocals + ncells + i],
          true);
    }
  }

  return map;
}

#endif

void framelocals_mapping_free(FrameLocalsMapping* map) {
  delete map;
}

PyDictObject* framelocals_mapping_to_dict(FrameLocalsMapping* map) {
  return map->to_dict();
}
