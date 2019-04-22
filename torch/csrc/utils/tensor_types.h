#pragma once

#include <ATen/ATen.h>
#include <utility>
#include <vector>

namespace torch { namespace utils {

std::string type_to_string(const at::Type& type, const at::ScalarType scalar_type);
std::pair<at::Type*, at::ScalarType> type_from_string(const std::string& str);

// return a vector of all "declared" types, even those that weren't compiled
std::vector<std::pair<at::Backend, at::ScalarType>> all_declared_types();

}} // namespace torch::utils
