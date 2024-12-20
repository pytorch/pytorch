#pragma once

#include <torch/csrc/utils/python_compat.h>

#ifdef __cplusplus

#include <string>
#include <unordered_map>

#include <torch/csrc/dynamo/utils.h>
#include <torch/csrc/utils/pybind.h>

extern "C" {

#if IS_PYTHON_3_11_PLUS
using FrameLocalsFrameType = _PyInterpreterFrame;
#else
using FrameLocalsFrameType = PyFrameObject;
#endif // IS_PYTHON_3_11_PLUS

/**
 * Utility to view a frame's localsplus (locals + cells + freevars)
 * in C/C++ and Python, without changing the state of the frame.
 *
 * Notes on usage:
 *  - C/C++ can directly read the frame's localsplus using an index.
 *  - Cell/free variables are unboxed.
 *  - Can be converted into a dict for use in Python.
 *    The dict is constructed once per FrameLocalsMapping, lazily.
 *  - Lifetime should not exceed the lifetime of the frame
 *
 * How do guards use FrameLocalsMapping?
 * - When a guard accesses a frame's localsplus, we find the index of the
 *   variable name in the frame's code object and create a
 *   FrameLocalsGuardAccessor.
 * - We create a FrameLocalsMapping for the frame that we pass on to guard eval.
 * - LeafGuards/GuardManagers/GuardAccessors now need to define how they
 *   handle FrameLocalsMapping. By default, the FrameLocalsMapping is converted
 *   to a Python dict and the guard check is performed on the resulting dict.
 * - Some guard checks don't actually depend on the input arguments, e.g. they
 *   only check global state. In this case, no dict conversion of
 *   FrameLocalsMapping is done.
 * - FrameLocalsGuardAccessor is like DictGetItemGuardAccessor, except it knows
 *   how to handle FrameLocalsMapping - by using the framelocals variable name
 *   index that it was given when it was built.
 */
typedef struct VISIBILITY_HIDDEN FrameLocalsMapping {
 private:
  py::object _code_obj;
  // can't use localsplus directly due to closure variables:
  // - in 3.11+, the closure vars in the frame's closure object and
  //   the corresponding localsplus entry is nullptr
  // - regardless of Python version, we need to unbox the cell variable
  std::vector<py::handle> _framelocals;

  py::object _dict{py::none()};

  void _realize_dict();

 public:
  explicit FrameLocalsMapping(FrameLocalsFrameType* frame);

  PyObject* get(int idx);

  bool dict_realized() const {
    return _dict.is_none();
  }

  // Borrowed reference
  PyDictObject* to_dict() {
    if (this->dict_realized()) {
      _realize_dict();
    }
    return (PyDictObject*)_dict.ptr();
  }
} FrameLocalsMapping;

#else

// opaque type for C
typedef struct FrameLocalsMapping FrameLocalsMapping;

#endif

#if IS_PYTHON_3_11_PLUS
typedef struct _PyInterpreterFrame _PyInterpreterFrame;
FrameLocalsMapping* get_framelocals_mapping(_PyInterpreterFrame* frame);
#else
FrameLocalsMapping* get_framelocals_mapping(PyFrameObject* frame);
#endif

void framelocals_mapping_free(FrameLocalsMapping* map);

// Borrowed reference
PyDictObject* framelocals_mapping_to_dict(FrameLocalsMapping* map);

#ifdef __cplusplus
} // extern "C"

py::tuple code_framelocals_names(py::handle code);
#endif
