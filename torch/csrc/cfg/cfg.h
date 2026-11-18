#pragma once

// C++ side of torch.cfg_compile. For now this is a Python binding that shows
// what the dispatcher does, built on c10::DispatchCaptureSink.

#include <torch/csrc/python_headers.h>

namespace torch::cfg {

void initCfgBindings(PyObject* module);

} // namespace torch::cfg
