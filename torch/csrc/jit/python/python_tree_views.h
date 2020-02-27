
copy: fbcode/caffe2/torch/csrc/jit/python/python_tree_views.h
copyrev: e9b63a8cdfbf1751a851dd9d53a44bb03b684d16

#pragma once

#include <torch/csrc/python_headers.h>

namespace torch {
namespace jit {
namespace script {

void initTreeViewBindings(PyObject* module);

}
} // namespace jit
} // namespace torch
