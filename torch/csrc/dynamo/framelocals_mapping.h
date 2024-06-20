#pragma once

#include <torch/csrc/utils/python_compat.h>

#if IS_PYTHON_3_11_PLUS
#define THP_EVAL_API_FRAME_OBJECT _PyInterpreterFrame
#else
#define THP_EVAL_API_FRAME_OBJECT PyFrameObject
#endif
typedef struct THP_EVAL_API_FRAME_OBJECT THP_EVAL_API_FRAME_OBJECT;

#ifdef __cplusplus

#include <torch/csrc/dynamo/utils.h>
#include <torch/csrc/utils/pybind.h>
#include <string>
#include <unordered_map>

namespace py = pybind11;

extern "C" {

#endif

typedef struct FrameLocalsMapping FrameLocalsMapping;

#ifdef __cplusplus

typedef struct VISIBILITY_HIDDEN FrameLocalsMapping {
  FrameLocalsMapping(THP_EVAL_API_FRAME_OBJECT* frame);
  py::handle getitem(const std::string& key) const;

  // private:
  std::unordered_map<std::string, py::object> mapping;
} FrameLocalsMapping;

#endif

PyObject* FrameLocalsMapping_new(THP_EVAL_API_FRAME_OBJECT* frame);
void FrameLocalsMapping_delete(PyObject** obj);
PyObject* FrameLocalsMapping_todict(PyObject* obj);

#ifdef __cplusplus
} // extern "C"
#endif
