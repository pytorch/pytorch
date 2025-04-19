#pragma once

#include <string_view>
#include <vector>

namespace torch::nativert {

std::vector<std::string_view> split(std::string_view target, char delimiter);

std::string join(
    std::string_view delimiter,
    const std::vector<std::string>& keys);

// These helpers should be replaced by string_view.starts_with and
// string_view.ends_with in C++20, when they are available.
bool starts_with(std::string_view target, std::string_view prefix);
bool ends_with(std::string_view target, std::string_view prefix);

} // namespace torch::nativert
