#pragma once
#include <optional>
#include <string>

namespace torch::runtime {
std::optional<std::string> maybeGetEnv(std::string_view envVar);
}
