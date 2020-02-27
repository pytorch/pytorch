
copy: fbcode/caffe2/torch/csrc/jit/python/init.h
copyrev: a663fa49c0cad21e518824b64a69300f70b9773e

#pragma once

#include <torch/csrc/utils/pybind.h>

namespace torch {
namespace jit {

void initJITBindings(PyObject* module);

}
} // namespace torch
