#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/TensorUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/CPUFunctions.h>
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/bmm.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_strided.h>
#include <ATen/ops/mm.h>
#endif

namespace at::native {

inline bool check_valid_strides_and_return_transposed(const Tensor& mat) {
  IntArrayRef tensor_strides = mat.strides();
  IntArrayRef tensor_sizes = mat.sizes();
  int end_dim = mat.dim() - 1;
  int alignment = 16 / mat.element_size();
  TORCH_CHECK(uint64_t(mat.data_ptr()) % 16 ==0, "expected data_ptr to be aligned to 16 bytes\n");
  if ((tensor_strides[end_dim - 1] == 1) && (tensor_strides[end_dim] >= std::max<int64_t>(1, tensor_sizes[end_dim - 1]))) {
    TORCH_CHECK(tensor_strides[end_dim] % alignment == 0, "strides should be multiple of 16 bytes");
    return true;
  } else if ((tensor_strides[end_dim] == 1) && (tensor_strides[end_dim - 1] >= std::max<int64_t>(1, tensor_sizes[end_dim]))) {
    TORCH_CHECK(tensor_strides[end_dim - 1] % alignment == 0, "strides should be multiple of 16 bytes");
    return false;
  } else {
    TORCH_CHECK(false, "Invalid strides/sizes, got ", mat.strides(), " for strides and ", mat.sizes(), " for sizes");
  }
}

inline at::Tensor create_grouped_gemm_output_tensor(const Tensor& mat_a,
const Tensor& mat_b,
const std::optional<at::Tensor>& offs,
c10::ScalarType out_dtype
) {
  c10::SmallVector<int64_t, 3> out_size;
  const bool a_is_2d = mat_a.dim() == 2;
  const bool b_is_2d = mat_b.dim() == 2;
  if (a_is_2d) {
    if (b_is_2d) {
      out_size = {offs->size(0), mat_a.size(0), mat_b.size(1)};
    } else {
      TORCH_CHECK(offs->size(0) == mat_b.size(0), "matrix batch sizes have to match");
      out_size = {mat_a.size(0), mat_b.size(-1)};
    }
  } else {
    if (b_is_2d) {
      // this case is not actually encountered for MoE gemms
      TORCH_CHECK(offs->size(0) == mat_a.size(0), "matrix batch sizes have to match");
      out_size = {mat_a.size(1), mat_b.size(1)};
    } else { // regular bmm
      TORCH_CHECK(mat_a.size(0) == mat_b.size(0), "batched dimension has to match");
      out_size = {mat_a.size(0), mat_a.size(1), mat_b.size(-1)};
    }
  }

  #ifndef USE_ROCM
  // For TMA transfers, strides of output tensor have to be either
  // 1, or aligned to 16 bytes.
  const auto last_dim = out_size.size() - 1;
  const auto alignment = 16 / c10::elementSize(out_dtype);
  const int64_t size_padded = (out_size[last_dim] + alignment - 1) / alignment * alignment;
  std::vector<int64_t> out_stride;
  if (a_is_2d != b_is_2d) {
    out_stride = {size_padded, 1};
  } else {
    out_stride = {out_size[1] * size_padded, size_padded, 1};
  }
  return at::empty_strided(out_size, out_stride, mat_a.options().dtype(out_dtype));
  #else
  return at::empty(out_size, mat_a.options().dtype(out_dtype));
  #endif
}

inline void _grouped_mm_validate_inputs(const Tensor& mat_a, const Tensor& mat_b,
const std::optional<at::Tensor>& offs,
const std::optional<at::Tensor>& bias,
std::optional<c10::ScalarType> out_dtype) {
  TORCH_CHECK((mat_a.dtype() == at::kBFloat16) || (mat_a.dtype() == at::kFloat) || (mat_a.dtype() == at::kHalf), "Expected mat_a to be Float32, BFloat16 or Float16 matrix, got ", mat_a.scalar_type());
  TORCH_CHECK((mat_b.dtype() == at::kBFloat16) || (mat_b.dtype() == at::kFloat) || (mat_b.dtype() == at::kHalf), "Expected mat_b to be Float32, BFloat16 or Float16 matrix, got ", mat_b.scalar_type());
  TORCH_CHECK(mat_a.dim() == 2 || mat_a.dim() == 3, "mat_a has to be 2 or 3d");
  TORCH_CHECK(mat_b.dim() == 2 || mat_b.dim() == 3, "mat_b has to be 2 or 3d");
  const bool a_is_2d = mat_a.dim() == 2;
  const bool b_is_2d = mat_b.dim() == 2;
  if (!a_is_2d || !b_is_2d) {
    TORCH_CHECK(mat_a.size(-1) == mat_b.size(-2), "contraction dimension of mat_a and mat_b must match");
  }

  // check that the strides are valid, the fn will throw an error if not
  check_valid_strides_and_return_transposed(mat_a);
  check_valid_strides_and_return_transposed(mat_b);
  TORCH_CHECK(offs.has_value() ==  (a_is_2d || b_is_2d), "Have to provide offsets if there is a 2d matrix, or no offset if both matrices are 3d");

  if (offs.has_value()) {
    TORCH_CHECK(offs->dim() == 1, "offs has to be 1D");
    TORCH_CHECK(offs->dtype() == at::kInt, "Offsets have to be int32");
  }
  TORCH_CHECK(!bias.has_value(), "Bias not supported yet");
}

inline c10::ScalarType _resolve_grouped_mm_out_dtype(const Tensor& mat_a, [[maybe_unused]] const Tensor& mat_b,
std::optional<c10::ScalarType> out_dtype) {
  const auto out_dtype_ = out_dtype.value_or(mat_a.scalar_type());
  // TODO(future PR): enable float32 output dtype for bfloat16 and float16 inputs
  TORCH_CHECK(out_dtype_ == mat_a.dtype(), "Grouped gemm output dtype must match `mat_a` dtype");
  return out_dtype_;
}


inline void _grouped_mm_fallback(const Tensor& mat_a, const Tensor& mat_b,
const std::optional<at::Tensor>& offs,
const std::optional<at::Tensor>& bias,
std::optional<c10::ScalarType> out_dtype,
Tensor out) {
  LOG(INFO) << "fallback path for `torch._grouped_mm`, performance may not be optimal";
  const bool a_is_2d = mat_a.dim() == 2;
  const bool b_is_2d = mat_b.dim() == 2;
  if (a_is_2d && !b_is_2d) {
    // 2d x 3d with offsets
    int group_start_idx = 0;
    auto offs_cpu = offs.value().cpu();
    for (int group_idx = 0; group_idx < offs_cpu.size(0); group_idx++) {
      int group_end_idx = offs_cpu[group_idx].item<int>();
      auto mat_a_slice = mat_a.slice(0, group_start_idx, group_end_idx);
      auto out_slice = out.slice(0, group_start_idx, group_end_idx);
      at::mm_out(out_slice, mat_a_slice, mat_b[group_idx]);
      group_start_idx = group_end_idx;
    }

  } else if (!a_is_2d && b_is_2d) {
    // 3d x 2d with offsets
    int group_start_idx = 0;
    auto offs_cpu = offs.value().cpu();
    for (int group_idx = 0; group_idx < offs_cpu.size(0); group_idx++) {
      int group_end_idx = offs_cpu[group_idx].item<int>();
      auto mat_b_slice = mat_b.slice(1, group_start_idx, group_end_idx);
      auto out_slice = out.slice(1, group_start_idx, group_end_idx);
      at::mm_out(out_slice, mat_a[group_idx], mat_b_slice);
      group_start_idx = group_end_idx;
    }

  } else if (a_is_2d && b_is_2d) {
    // 2d x 2d with offsets
    int group_start_idx = 0;
    auto offs_cpu = offs.value().cpu();
    for (int group_idx = 0; group_idx < offs_cpu.size(0); group_idx++) {
      int group_end_idx = offs_cpu[group_idx].item<int>();
      auto mat_a_slice = mat_a.slice(1, group_start_idx, group_end_idx);
      auto mat_b_slice = mat_b.slice(0, group_start_idx, group_end_idx);
      auto out_slice = out[group_idx];
      at::mm_out(out_slice, mat_a_slice, mat_b_slice);
      group_start_idx = group_end_idx;
    }

  } else {
    // 3d x 3d without offsets - regular bmm
    at::bmm_out(out, mat_a, mat_b);
  }
}


} // namespace at::native
