#pragma once

#include <torch/csrc/python_headers.h>
#include <torch/csrc/Types.h>

namespace torch {

struct THPVoidTensor {
  PyObject_HEAD
  c10::TensorImpl *cdata;
  char device_type;
  char data_type;
};

struct THPVoidStorage {
  PyObject_HEAD
  c10::StorageImpl *cdata;
};

} // namespace torch
