#pragma once

#include <tuple>
#include <string>
#include <ATen/ATen.h>

namespace torch { namespace utils {

std::pair<std::string, std::string> getDtypeNames(at::ScalarType scalarType);

void initializeDtypes();

}} // namespace torch::utils
