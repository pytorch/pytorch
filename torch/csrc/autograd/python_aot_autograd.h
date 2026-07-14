#pragma once

#include <torch/csrc/python_headers.h>

#include <cstdint>
#include <vector>

namespace torch::autograd {

struct Range {
  int64_t start;
  int64_t count;
};

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct AOTAutogradSavePlan {
  PyObject_HEAD
  Range tensors_saved_with_vc_check_range;
  Range tensors_saved_no_vc_check_range;
  Range opaque_object_outs_range;
  Range symint_outs_range;
  std::vector<uint8_t> saved_tensor_is_graph_input;
  std::vector<PyObject*> dynamic_dims;
};

PyTypeObject* getAOTAutogradSavePlanType();
PyObject* THPModule_aot_autograd_save_from_forward(
    PyObject* _unused,
    PyObject* const* args,
    Py_ssize_t nargs);

} // namespace torch::autograd
