// Irdfft.h - Inverse Real-to-Real FFT CUDA kernel declarations
#pragma once
#include <ATen/ATen.h>
#include <c10/core/SymInt.h>
#include <c10/util/Optional.h>
#include <string>

namespace at { namespace native {
void irdfft_cuda_inplace(at::Tensor& self, std::optional<c10::SymInt> n, int64_t dim, std::optional<std::string> norm);
}}
