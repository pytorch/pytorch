#pragma once

#include <c10/macros/Export.h>

namespace c10 {
struct OperatorName;
}

namespace at {

// check if an op is a custom op (i.e. did not come from native_functions.yaml)
TORCH_API bool is_custom_op(const c10::OperatorName& opName);
} // namespace at
