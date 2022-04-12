#pragma once
#include <torch/csrc/QScheme.h>

namespace torch { namespace utils {

PyObject* getTHPQScheme(at::QScheme qscheme);
void initializeQSchemes();

}} // namespace torch::utils
