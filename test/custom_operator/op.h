#include <torch/script.h>

#include <cstddef>
#include <vector>
#include <string>

// clang-format off
#  if defined(_WIN32)
#    if defined(custom_ops_EXPORTS)
#      define CUSTOM_OP_API __declspec(dllexport)
#    else
#      define CUSTOM_OP_API __declspec(dllimport)
#    endif
#  else
#    define CUSTOM_OP_API
#  endif
// clang-format on

CUSTOM_OP_API std::vector<torch::Tensor> custom_op(
    torch::Tensor tensor,
    double scalar,
    int64_t repeat);

CUSTOM_OP_API int64_t custom_op2(std::string s1, std::string s2);
