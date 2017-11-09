#pragma once

#include "torch/csrc/utils/python_stub.h"
#include "cudnn-wrapper.h"

namespace torch { namespace cudnn {

// Provides conversions between Python objects and CuDNN objects

PyObject * getTensorClass(PyObject *args);
cudnnDataType_t getCudnnDataType(PyObject *tensorClass);

}}  // namespace torch::cudnn
