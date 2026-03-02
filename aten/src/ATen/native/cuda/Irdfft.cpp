// Copyright (c) 2025 PyTorch Contributors.
// All rights reserved.
#include "Irdfft.h"
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/SpectralOpsUtils.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/core/SymInt.h>
#include <c10/util/Optional.h>
#include <string>

// Forward declaration of CUDA kernel
namespace at { namespace native {
void irdfft_cuda_inplace(at::Tensor& self,
    std::optional<c10::SymInt> n, int64_t dim,
    std::optional<std::string> norm);
}}

namespace at { namespace native {

// Inplace CUDA dispatch function for fft_irdfft_
Tensor& fft_irdfft_cuda_(Tensor& self,
    std::optional<c10::SymInt> n, int64_t dim,
    std::optional<c10::string_view> norm) {
    TORCH_CHECK(self.is_cuda(),
        "Input tensor must be on CUDA device");
    TORCH_CHECK(self.is_floating_point(),
        "irdfft expects a floating point input tensor, but got ",
        self.scalar_type());

    const auto input_dim = self.dim();
    const auto d = maybe_wrap_dim(dim, input_dim, /*wrap_scalar=*/false);

    const auto n_val = n.value_or(self.sym_sizes()[d]);
    TORCH_CHECK(n_val >= 1,
    "Invalid number of data points (", n_val, ") specified");

    // Convert string_view to string for kernel
    std::optional<std::string> norm_str;
    if (norm.has_value()) {
        norm_str = std::string(norm.value());
    }

    irdfft_cuda_inplace(self, n_val, d, norm_str);
    return self;
}


Tensor& fft_irdfft_cuda_(Tensor& self,
    std::optional<int64_t> n, int64_t dim,
    std::optional<std::string_view> norm) {
    std::optional<c10::SymInt> n_symint;
    if (n.has_value()) {
        n_symint = c10::SymInt(*n);
    }
    return fft_irdfft_cuda_(self, n_symint, dim, norm);
}

}  // namespace native
}  // namespace at
