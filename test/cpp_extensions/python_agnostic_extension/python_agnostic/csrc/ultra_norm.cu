#include <ATen/ops/_foreach_norm_native.h>
#include <ATen/ops/cat_cuda_dispatch.h>
#include <ATen/ops/norm_cuda_dispatch.h>
#include <ATen/ops/unsqueeze.h>
#include <torch/extension.h>

at::Tensor ultra_norm(at::TensorList inputs) {
    auto res = at::native::foreach_tensor_norm_cuda(inputs);
    std::vector<at::Tensor> unsqueezed;
    for (const auto& scalar_tensor : res) {
        unsqueezed.push_back(at::unsqueeze(scalar_tensor, 0));
    }
    auto stacked = at::cuda::cat(unsqueezed);
    return at::cuda::norm(stacked, 2, at::IntArrayRef{}, false);
}

TORCH_LIBRARY_IMPL(python_agnostic, CUDA, m) {
  m.impl("python_agnostic::ultra_norm", &ultra_norm);
}
