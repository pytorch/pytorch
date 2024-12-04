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

typedef struct VISIBILITY_HIDDEN FrameLocalsMapping {
 private:
  FrameLocalsFrameType* _frame; // non-owning
  py::object _dict{py::none()};

  void _realize_dict();

 public:
  explicit FrameLocalsMapping(FrameLocalsFrameType* frame) : _frame(frame) {}

  PyObject* get(uint64_t idx);

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
#endif
