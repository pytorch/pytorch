#ifndef PYTHON_COMPAT
#define PYTHON_COMPAT

#include <torch/csrc/utils/pythoncapi_compat.h>

#ifdef __cplusplus
extern "C" {
#endif

// PyTorch-only compat functions

#define IS_PYTHON_3_11_PLUS (PY_VERSION_HEX >= 0x030B00C1)
#define IS_PYTHON_3_12_PLUS (PY_VERSION_HEX >= 0x030C0000)
#define IS_PYTHON_3_13_PLUS (PY_VERSION_HEX >= 0x030D0000)
#define IS_PYTHON_3_14_PLUS (PY_VERSION_HEX >= 0x030E0000)
#define IS_PYTHON_3_15_PLUS (PY_VERSION_HEX >= 0x030F0000)

static inline int PyCode_GetNCellvars(PyCodeObject* code) {
// gh-26364 added co_ncellvars to Python 3.11.0rc1
#if IS_PYTHON_3_11_PLUS
  return code->co_ncellvars;
#else
  return PyTuple_GET_SIZE(code->co_cellvars);
#endif
}

static inline int PyCode_GetNFreevars(PyCodeObject* code) {
// gh-26364 added co_nfreevars to Python 3.11.0rc1
#if IS_PYTHON_3_11_PLUS
  return code->co_nfreevars;
#else
  return PyTuple_GET_SIZE(code->co_freevars);
#endif
}

#if !IS_PYTHON_3_14_PLUS
static inline int PyUnstable_TryIncRef(PyObject* obj) {
#if defined(Py_GIL_DISABLED)
  uint32_t local = _Py_atomic_load_uint32_relaxed(&obj->ob_ref_local);
  local += 1;
  if (local == 0) {
    // immortal
    return true;
  }
  if (_Py_IsOwnedByCurrentThread(obj)) {
    _Py_atomic_store_uint32_relaxed(&obj->ob_ref_local, local);
#ifdef Py_REF_DEBUG
    _Py_INCREF_IncRefTotal();
#endif
    return true;
  }
  Py_ssize_t shared = _Py_atomic_load_ssize_relaxed(&obj->ob_ref_shared);
  for (;;) {
    // If the shared refcount is zero and the object is either merged
    // or may not have weak references, then we cannot incref it.
    if (shared == 0 || shared == _Py_REF_MERGED) {
      return false;
    }

    if (_Py_atomic_compare_exchange_ssize(
            &obj->ob_ref_shared,
            &shared,
            shared + (1 << _Py_REF_SHARED_SHIFT))) {
#ifdef Py_REF_DEBUG
      _Py_INCREF_IncRefTotal();
#endif
      return true;
    }
  }
#else
  if (Py_REFCNT(obj) > 0) {
    Py_INCREF(obj);
    return true;
  }
  return false;
#endif
}

static inline void PyUnstable_EnableTryIncRef(PyObject* obj) {
#ifdef Py_GIL_DISABLED
  if (_Py_IsImmortal(obj)) {
    return;
  }
  for (;;) {
    Py_ssize_t shared = _Py_atomic_load_ssize_relaxed(&obj->ob_ref_shared);
    if ((shared & _Py_REF_SHARED_FLAG_MASK) != 0) {
      // Nothing to do if it's in WEAKREFS, QUEUED, or MERGED states.
      return;
    }
    if (_Py_atomic_compare_exchange_ssize(
            &obj->ob_ref_shared, &shared, shared | _Py_REF_MAYBE_WEAKREF)) {
      return;
    }
  }
#endif
}
#endif

#ifdef __cplusplus
}
#endif
#endif // PYTHON_COMPAT
