#pragma once

// ${generated_comment}

#include <torch/headeronly/macros/Macros.h>

HIDDEN_NAMESPACE_BEGIN(torch, headeronly)

// Enum of valid tags obtained from the entries in tags.yaml
enum class Tag {
    ${enum_of_valid_tags}
};

HIDDEN_NAMESPACE_END(torch, headeronly)

// Re-expose in the at:: namespace for backward compatibility
namespace at {
    using torch::headeronly::Tag;
}
