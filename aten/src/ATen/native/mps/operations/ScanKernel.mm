#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_cummax_helper_native.h>
#include <ATen/ops/_cummin_helper_native.h>
#endif
#include <fmt/format.h>

namespace at::native {
namespace mps {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/ScanKernel_metallib.h>
#endif

// Generic scan implementation that handles both simple scans and scans with indices
static void scan_mps_impl(const Tensor& self,
                          const std::vector<Tensor>& outputs,
                          int64_t dim,
                          const std::string& op_name) {
  if (outputs[0].numel() == 0) {
    return;
  }

  const int64_t ndim = self.dim();
  const int64_t wrapped_dim = maybe_wrap_dim(dim, ndim);

  // Calculate dimensions for scan operation
  int64_t row_size = self.size(wrapped_dim);
  auto sizes = self.sizes();

  bool is_innermost = (wrapped_dim == ndim - 1);

  // Check if all tensors are contiguous
  bool is_contiguous = self.is_contiguous();
  for (const auto& output : outputs) {
    is_contiguous = is_contiguous && output.is_contiguous();
  }

  uint32_t num_rows, num_orows, num_irows, num_threads;

  if (is_innermost) {
    // Treat all outer dimensions as a single dimension
    num_rows = self.numel() / row_size;
    num_threads = num_rows;
  } else {
    // Treat all outer dimensions (i.e. dim_ < dim) as one
    num_orows = c10::multiply_integers(sizes.begin(), sizes.begin() + wrapped_dim);
    // Treat all inner dimensions (i.e. dim > dimension) as one
    num_irows = c10::multiply_integers(sizes.begin() + wrapped_dim + 1, sizes.end());
    num_threads = num_orows * num_irows;
  }

  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();

      // Choose kernel based on contiguity and dimension
      std::string kernel_name;
      if (is_contiguous) {
        kernel_name =
            op_name + "_contiguous_" + (is_innermost ? "innermost_" : "outer_") + scalarToMetalTypeString(self);
      } else {
        kernel_name = op_name + "_strided_" + scalarToMetalTypeString(self);
      }

      id<MTLComputePipelineState> scanPSO = lib.getPipelineStateForFunc(kernel_name);

      // this function call is a no-op if MPS Profiler is not enabled
      getMPSProfiler().beginProfileKernel(scanPSO, op_name, [&]() {
        std::vector<Tensor> all_tensors = {self};
        all_tensors.insert(all_tensors.end(), outputs.begin(), outputs.end());
        return all_tensors;
      }());

      [computeEncoder setComputePipelineState:scanPSO];

      // Set input tensor
      mtl_setBuffer(computeEncoder, self, 0);

      // Set output tensors
      for (size_t i = 0; i < outputs.size(); ++i) {
        mtl_setBuffer(computeEncoder, outputs[i], i + 1);
      }

      if (is_contiguous) {
        // Contiguous kernels
        if (is_innermost) {
          if (outputs.size() == 1) {
            // Simple scan
            mtl_setArgs<2>(computeEncoder, num_rows, static_cast<uint32_t>(row_size));
          } else {
            // Scan with indices
            mtl_setArgs<3>(computeEncoder, num_rows, static_cast<uint32_t>(row_size));
          }
        } else {
          if (outputs.size() == 1) {
            // Simple scan
            mtl_setArgs<2>(computeEncoder, num_orows, num_irows, static_cast<uint32_t>(row_size));
          } else {
            // Scan with indices
            mtl_setArgs<3>(computeEncoder, num_orows, num_irows, static_cast<uint32_t>(row_size));
          }
        }
      } else {
        // Strided kernels - pass full tensor information
        if (outputs.size() == 1) {
          // Simple scan
          mtl_setArgs<2>(computeEncoder,
                         self.sizes(),
                         self.strides(),
                         outputs[0].strides(),
                         static_cast<uint32_t>(self.ndimension()),
                         static_cast<uint32_t>(wrapped_dim));
        } else {
          // Scan with indices
          mtl_setArgs<3>(computeEncoder,
                         self.sizes(),
                         self.strides(),
                         outputs[0].strides(),
                         outputs[1].strides(),
                         static_cast<uint32_t>(self.ndimension()),
                         static_cast<uint32_t>(wrapped_dim));
        }
      }

      mtl_dispatch1DJob(computeEncoder, scanPSO, num_threads);

      getMPSProfiler().endProfileKernel(scanPSO);
    }
  });
}

// Utility function to get 2D grid dimensions for dispatch
static std::pair<uint32_t, uint32_t> get_2d_grid_dims(const IntArrayRef& shape, const int64_t dim) {
  size_t grid_x = 1;
  size_t grid_y = 1;

  for (const auto i : c10::irange(dim)) {
    if (grid_x * shape[i] < UINT32_MAX) {
      grid_x *= shape[i];
    } else {
      grid_y *= shape[i];
    }
  }

  TORCH_CHECK(grid_y <= UINT32_MAX && grid_x <= UINT32_MAX, "Unable to safely factor shape for grid dimensions.");

  if (grid_y > grid_x) {
    std::swap(grid_x, grid_y);
  }

  return {static_cast<uint32_t>(grid_x), static_cast<uint32_t>(grid_y)};
}

// Specialized implementation for cummin/cummax that returns both values and indices
static void scan_with_indices_mps_impl(const Tensor& self,
                                       const Tensor& values_output,
                                       const Tensor& indices_output,
                                       int64_t dim,
                                       const std::string& op_name) {
  if (values_output.numel() == 0) {
    return;
  }

  const int64_t ndim = self.dim();
  const int64_t wrapped_dim = maybe_wrap_dim(dim, ndim);
  const int64_t axis_size = self.size(wrapped_dim);

  // Preprocess input tensor - ensure it's contiguous for Metal shaders
  auto input_tensor = self.contiguous();

  // Preprocess output tensors - ensure they're contiguous for Metal shaders
  auto values_tensor = values_output.contiguous();
  auto indices_tensor = indices_output.contiguous();
  const bool values_needs_copy = !values_output.is_contiguous();
  const bool indices_needs_copy = !indices_output.is_contiguous();

  // Determine which kernel to use based on scan dimension position
  bool is_innermost_scan = (wrapped_dim == ndim - 1);

  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();

      // Build kernel name based on scan type
      const auto type_str = scalarToMetalTypeString(input_tensor);
      const auto kernel_name = fmt::format("{}_{}_{}", op_name, is_innermost_scan ? "innermost" : "outer", type_str);

      id<MTLComputePipelineState> scanPSO = lib.getPipelineStateForFunc(kernel_name);

      // this function call is a no-op if MPS Profiler is not enabled
      getMPSProfiler().beginProfileKernel(scanPSO, op_name, {input_tensor, values_tensor, indices_tensor});

      [computeEncoder setComputePipelineState:scanPSO];

      // Set input and output buffers (all guaranteed contiguous)
      mtl_setArgs(computeEncoder, input_tensor, values_tensor, indices_tensor);

      constexpr int simd_size = 32;

      if (is_innermost_scan) {
        // Contiguous scan dispatch (scanning innermost dimension)
        mtl_setArgs<3>(computeEncoder, axis_size);

        int n_reads = (input_tensor.element_size() <= 4) ? 4 : 2;

        int elements_per_simd = n_reads * simd_size;
        int thread_group_size = static_cast<int>(scanPSO.maxTotalThreadsPerThreadgroup);

        if (axis_size <= n_reads * 1024) {
          thread_group_size = ((axis_size + elements_per_simd - 1) / elements_per_simd) * simd_size;
        } else if (axis_size <= n_reads * 2048) {
          thread_group_size = ((axis_size / 2 + elements_per_simd - 1) / elements_per_simd) * simd_size;
        }
        thread_group_size = std::min(thread_group_size, static_cast<int>(scanPSO.maxTotalThreadsPerThreadgroup));

        auto tmp_grid_dims = get_2d_grid_dims(input_tensor.sizes(), wrapped_dim);

        [computeEncoder dispatchThreads:MTLSizeMake(thread_group_size, tmp_grid_dims.first, tmp_grid_dims.second)
                  threadsPerThreadgroup:MTLSizeMake(thread_group_size, 1, 1)];
      } else {
        // Strided scan dispatch (scanning non-innermost dimension)
        size_t stride = input_tensor.strides()[wrapped_dim];
        constexpr int bn = 32;
        size_t stride_blocks = (stride + bn - 1) / bn;

        mtl_setArgs<3>(computeEncoder, axis_size, stride, stride_blocks);

        int n_reads = (input_tensor.element_size() <= 4) ? 4 : 2;
        int n_simdgroups = bn / n_reads;
        int thread_group_size = n_simdgroups * simd_size;

        auto tmp_grid_dims = get_2d_grid_dims(input_tensor.sizes(), wrapped_dim);
        if (tmp_grid_dims.first * stride_blocks <= UINT_MAX) {
          tmp_grid_dims.first *= stride_blocks;
        } else {
          tmp_grid_dims.second *= stride_blocks;
        }

        [computeEncoder dispatchThreads:MTLSizeMake(thread_group_size, tmp_grid_dims.first, tmp_grid_dims.second)
                  threadsPerThreadgroup:MTLSizeMake(thread_group_size, 1, 1)];
      }

      getMPSProfiler().endProfileKernel(scanPSO);
    }
  });

  // Post-process: copy results back to original output tensors if needed
  if (values_needs_copy) {
    values_output.copy_(values_tensor);
  }
  if (indices_needs_copy) {
    indices_output.copy_(indices_tensor);
  }
}

} // namespace mps

void cummax_helper_mps(const Tensor& self, Tensor& values, Tensor& indices, int64_t dim) {
  if (is_macos_13_or_newer(MacOSVersion::MACOS_VER_14_0_PLUS)) {
    mps::scan_with_indices_mps_impl(self, values, indices, dim, "cummax");
  } else {
    mps::scan_mps_impl(self, {values, indices}, dim, "cummax");
  }
}

void cummin_helper_mps(const Tensor& self, Tensor& values, Tensor& indices, int64_t dim) {
  if (is_macos_13_or_newer(MacOSVersion::MACOS_VER_14_0_PLUS)) {
    mps::scan_with_indices_mps_impl(self, values, indices, dim, "cummin");
  } else {
    mps::scan_mps_impl(self, {values, indices}, dim, "cummin");
  }
}

} // namespace at::native
