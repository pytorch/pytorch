#pragma once

#include <torch/csrc/utils/python_compat.h>

#ifdef __cplusplus

#include <string>
#include <unordered_map>

#include <torch/csrc/dynamo/utils.h>
#include <torch/csrc/utils/pybind.h>

extern "C" {

typedef struct VISIBILITY_HIDDEN FrameLocalsMapping {
 private:
  std::unordered_map<std::string, PyObject*> _map;
  py::object _dict{py::none()};

 public:
  PyObject* get(const std::string& key) {
    return _map[key];
  }

  PyObject* get(py::handle key) {
    return _map[key.cast<std::string>()];
  }

  void set(const std::string& key, PyObject* value) {
    _map[key] = value;
  }

  void erase(const std::string& key) {
    _map.erase(key);
  }

  bool dict_realized() const {
    return _dict.is_none();
  }

  // Borrowed reference
  PyDictObject* to_dict() {
    if (this->dict_realized()) {
      _dict = py::dict();
      for (auto& [name, value] : _map) {
        _dict[py::str(name)] = py::cast<py::object>(value);
      }
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
