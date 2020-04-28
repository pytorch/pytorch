#include <ATen/core/jit_type.h>

namespace c10 {
TORCH_API TypePtr parseType(const std::string& pythonStr);
TORCH_API TypePtr parseType(
    const std::string& pythonStr,
    // Function to resolve non-primitive types to a TypePtr, typically by
    // looking them up in some sort of environment. Similar to
    // `torch::jit::Resolver`, which we don't use here because it pulls in some
    // JIT headers.
    std::function<TypePtr(const std::string&)> resolver);
} // namespace c10
