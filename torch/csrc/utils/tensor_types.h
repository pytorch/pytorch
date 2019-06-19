#pragma once

#include <ATen/ATen.h>
#include <utility>
#include <vector>

namespace torch { namespace utils {

std::string type_to_string(const at::DeprecatedTypeProperties& type);
std::string options_to_string(const at::TensorOptions& options);
at::DeprecatedTypeProperties* type_from_string(const std::string& str);

// return a vector of all "declared" types, even those that weren't compiled
std::vector<std::pair<at::Backend, at::ScalarType>> all_declared_types();

}} // namespace torch::utils
