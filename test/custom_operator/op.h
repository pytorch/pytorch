#include <torch/op.h>

#include <cstddef>
#include <vector>

std::vector<at::Tensor> custom_op(
    at::Tensor tensor,
    double scalar,
    int64_t repeat);
