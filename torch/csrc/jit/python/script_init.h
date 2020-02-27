
copy: fbcode/caffe2/torch/csrc/jit/python/script_init.h
copyrev: 209ed6d7188315614ba49dbf016cfbe28174004f

#pragma once

#include <torch/csrc/jit/python/pybind.h>

namespace torch {
namespace jit {
namespace script {
void initJitScriptBindings(PyObject* module);
} // namespace script
} // namespace jit
} // namespace torch
