#pragma once
#include <torch/csrc/QBackend.h>

namespace torch {
namespace utils {

PyObject* getTHPQBackend(at::QBackend qbackend);
void initializeQBackends();

} // namespace utils
} // namespace torch
