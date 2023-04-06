#pragma once

#include <ATen/ATen.h>
#include <string>
#include <tuple>

namespace torch {
namespace utils {

std::pair<std::string, std::string> getDtypeNames(at::ScalarType scalarType);

void initializeDtypes();

} // namespace utils
} // namespace torch
