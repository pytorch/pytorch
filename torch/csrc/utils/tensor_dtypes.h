#pragma once

#include <c10/core/ScalarType.h>
#include <string>
#include <tuple>

namespace torch::utils {

std::pair<std::string, std::string> getDtypeNames(at::ScalarType scalarType);

void initializeDtypes();

} // namespace torch::utils
