#include <ATen/core/jit_type.h>

namespace c10 {
TORCH_API TypePtr parseType(const std::string& pythonStr);
} // namespace c10
