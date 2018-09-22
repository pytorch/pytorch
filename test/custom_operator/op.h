#include <torch/script.h>

#include <cstddef>
#include <vector>

TORCH_API std::vector<at::Tensor> custom_op(
    at::Tensor tensor,
    double scalar,
    int64_t repeat);
