#pragma once
#include <torch/csrc/QScheme.h>

namespace torch::utils {

PyObject* getTHPQScheme(at::QScheme qscheme);
void initializeQSchemes();

} // namespace torch::utils
