//  Copyright © 2022 Apple Inc.
#include <optional>
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/mps/EmptyTensor.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/native/mps/OperationUtils.h>
#include <fmt/format.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/tril_indices_native.h>
#include <ATen/ops/tril_native.h>
#include <ATen/ops/triu_indices_native.h>
#include <ATen/ops/triu_native.h>
#endif

#include <MetalPerformanceShaders/MetalPerformanceShaders.h>

namespace at::native {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/TriangularOps_metallib.h>
#endif

template <typename T>
static std::vector<T> reverse_array(const IntArrayRef& arr) {
  std::vector<T> rc(arr.size());
  for (const auto& i : c10::irange(arr.size())) {
    rc[i] = arr[arr.size() - 1 - i];
  }
  return rc;
}

static void triu_tril_impl(const Tensor& self, int64_t k, const Tensor& out, const std::string& name) {
  using namespace mps;
  if (self.numel() == 0) {
    return;
  }
  auto sizes = reverse_array<uint32_t>(self.sizes());
  auto inp_strides = reverse_array<int32_t>(self.strides());
  auto out_strides = reverse_array<int32_t>(out.strides());
  std::array<int, 2> k_ndim = {int(k), int(self.ndimension())};
  const bool inplace = self.is_same(out);
  const auto kernel_name =
      fmt::format("{}{}_{}_{}", name, inplace ? "_inplace" : "", "int", scalarToMetalTypeString(self));
  auto triuPSO = lib.getPipelineStateForFunc(kernel_name);
  uint32_t max_threads_per_group = [triuPSO maxTotalThreadsPerThreadgroup];
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto computeEncoder = stream->commandEncoder();
      [computeEncoder setComputePipelineState:triuPSO];
      if (inplace) {
        mtl_setArgs(computeEncoder, self, inp_strides, sizes, k_ndim);
      } else {
        mtl_setArgs(computeEncoder, out, self, out_strides, inp_strides, sizes, k_ndim);
      }
      [computeEncoder dispatchThreads:MTLSizeMake(sizes[0], sizes[1], self.numel() / (sizes[0] * sizes[1]))
                threadsPerThreadgroup:MTLSizeMake(std::min(max_threads_per_group, sizes[0]), 1, 1)];
    }
  });
}

TORCH_IMPL_FUNC(triu_mps_out)
(const Tensor& self, int64_t k, const Tensor& output) {
  triu_tril_impl(self, k, output, "triu");
}

TORCH_IMPL_FUNC(tril_mps_out)
(const Tensor& self, int64_t k, const Tensor& output) {
  triu_tril_impl(self, k, output, "tril");
}

Tensor tril_indices_mps(int64_t row,
                        int64_t col,
                        int64_t offset,
                        std::optional<ScalarType> dtype_opt,
                        std::optional<Layout> layout_opt,
                        std::optional<Device> device_opt,
                        std::optional<bool> pin_memory_opt) {
  check_args(row, col, layout_opt);

  auto tril_size = get_tril_size(row, col, offset);
  auto tensor = at::detail::empty_mps({2, tril_size}, dtype_opt, layout_opt, device_opt, pin_memory_opt, std::nullopt);
  if (tril_size <= 0) {
    return tensor;
  }
  auto m_first_row = offset > 0 ? std::min<int64_t>(col, 1 + offset) : // upper bounded by col
      row + offset > 0; // either 0 or 1
  auto trapezoid_row_offset = std::max<int64_t>(0, -offset);
  auto rectangle_row_offset = trapezoid_row_offset + col - m_first_row + 1;
  int64_t rectangle_size = 0;
  if (rectangle_row_offset < row) {
    rectangle_size = (row - rectangle_row_offset) * col;
  }
  using namespace mps;
  auto trilPSO = lib.getPipelineStateForFunc("tril_indices_" + scalarToMetalTypeString(tensor));
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto computeEncoder = stream->commandEncoder();
      [computeEncoder setComputePipelineState:trilPSO];
      mtl_setArgs(
          computeEncoder, tensor, trapezoid_row_offset, m_first_row, col, tril_size - rectangle_size, tril_size);
      mtl_dispatch1DJob(computeEncoder, trilPSO, tril_size);
    }
  });

  return tensor;
}

Tensor triu_indices_mps(int64_t row,
                        int64_t col,
                        int64_t offset,
                        std::optional<ScalarType> dtype_opt,
                        std::optional<Layout> layout_opt,
                        std::optional<Device> device_opt,
                        std::optional<bool> pin_memory_opt) {
  check_args(row, col, layout_opt);

  auto triu_size = row * col - get_tril_size(row, col, offset - 1);
  auto tensor = at::detail::empty_mps({2, triu_size}, dtype_opt, layout_opt, device_opt, pin_memory_opt, std::nullopt);
  if (triu_size <= 0) {
    return tensor;
  }
  // # of triu elements in the first row
  auto m_first_row = offset > 0 ? std::max<int64_t>(col - offset, 0) : // upper bounded by col
      col;

  // size of the top rectangle
  int64_t rectangle_size = 0;
  if (offset < 0) {
    rectangle_size = std::min<int64_t>(row, -offset) * col;
  }
  using namespace mps;
  auto triuPSO = lib.getPipelineStateForFunc("triu_indices_" + scalarToMetalTypeString(tensor));
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto computeEncoder = stream->commandEncoder();
      [computeEncoder setComputePipelineState:triuPSO];
      mtl_setArgs(computeEncoder, tensor, std::max<int64_t>(0, offset), m_first_row, col, rectangle_size, triu_size);
      mtl_dispatch1DJob(computeEncoder, triuPSO, triu_size);
    }
  });

  return tensor;
}
} // namespace at::native
