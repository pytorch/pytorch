#pragma once

// ${generated_comment}

#include <torch/headeronly/macros/Macros.h>

namespace at {
// Enum of valid tags obtained from the entries in tags.yaml
enum class Tag { ${enum_of_valid_tags} };
} // namespace at

HIDDEN_NAMESPACE_BEGIN(torch, headeronly)
using at::Tag;
HIDDEN_NAMESPACE_END(torch, headeronly)
