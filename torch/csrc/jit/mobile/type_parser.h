#include <ATen/core/jit_type.h>

namespace c10 {

TORCH_API TypePtr parseType(const std::string& pythonStr);

TORCH_API std::vector<TypePtr> parseType(std::vector<std::string>& pythonStr);
} // namespace c10
