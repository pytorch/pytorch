#pragma once

#include <Python.h>
#include <ATen/ATen.h>

namespace torch { namespace utils {

std::string type_to_string(const at::Type& type);
at::Type& type_from_string(const std::string& str);


}} // namespace torch::utils
