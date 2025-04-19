#pragma once
#include <optional>
#include <string>

namespace torch::nativert {
std::optional<std::string> maybeGetEnv(std::string_view envVar);
}
