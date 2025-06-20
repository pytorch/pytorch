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

namespace at::native {
namespace mps {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/ScanKernel_metallib.h>
#endif

// Utility function to get 2D grid dimensions for dispatch
static std::pair<uint32_t, uint32_t> get_2d_grid_dims(const IntArrayRef& shape,
                                                      const IntArrayRef& strides,
                                                      size_t divisor) {
  size_t total_elements = 1;
  for (auto s : shape) {
    total_elements *= s;
  }

  size_t grid_size = total_elements / divisor;

  // Simple 2D grid layout
  uint32_t width = std::min(grid_size, static_cast<size_t>(65535));
  uint32_t height = (grid_size + width - 1) / width;
  return std::make_pair(width, height);
}

static void scan_mps_impl(const Tensor& self, const Tensor& output, int64_t dim, const std::string& op_name) {
  if (output.numel() == 0) {
    return;
  }

  const int64_t ndim = self.dim();
  const int64_t wrapped_dim = maybe_wrap_dim(dim, ndim);
  const int64_t axis_size = self.size(wrapped_dim);

  // Preprocess input tensor - ensure it's contiguous for Metal shaders
  Tensor input_tensor = self;
  bool input_needs_copy = !self.is_contiguous();

  if (input_needs_copy) {
    input_tensor = self.contiguous();
  }

  // Preprocess output tensor - ensure it's contiguous for Metal shaders
  Tensor output_tensor = output;
  bool output_needs_copy = !output.is_contiguous();
  Tensor temp_output;

  if (output_needs_copy) {
    // Create a temporary contiguous tensor with the same shape and type
    temp_output = at::empty_like(output, output.options()).contiguous();
    output_tensor = temp_output;
  }

  // Determine which kernel to use based on scan dimension position
  bool is_innermost_scan = (wrapped_dim == ndim - 1);

  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();

      // Build kernel name based on scan dimension position
      std::string kernel_name;
      std::string type_str = scalarToMetalTypeString(input_tensor);

      if (is_innermost_scan) {
        kernel_name = op_name + "_innermost_" + type_str;
      } else {
        kernel_name = op_name + "_outer_" + type_str;
      }

      id<MTLComputePipelineState> scanPSO = lib.getPipelineStateForFunc(kernel_name);

      // this function call is a no-op if MPS Profiler is not enabled
      getMPSProfiler().beginProfileKernel(scanPSO, op_name, [&]() {
        std::vector<Tensor> all_tensors = {input_tensor, output_tensor};
        return all_tensors;
      }());

      [computeEncoder setComputePipelineState:scanPSO];

      // Set input and output buffers (both guaranteed contiguous)
      mtl_setBuffer(computeEncoder, input_tensor, 0);
      mtl_setBuffer(computeEncoder, output_tensor, 1);

      if (is_innermost_scan) {
        // Contiguous scan dispatch (scanning innermost dimension)
        mtl_setBytes(computeEncoder, axis_size, 2);

        int n_reads = (input_tensor.element_size() <= 4) ? 4 : 2;
        constexpr int simd_size = 32;
        int elements_per_simd = n_reads * simd_size;
        int thread_group_size = static_cast<int>(scanPSO.maxTotalThreadsPerThreadgroup);

        if (axis_size <= n_reads * 1024) {
          thread_group_size = ((axis_size + elements_per_simd - 1) / elements_per_simd) * simd_size;
        } else if (axis_size <= n_reads * 2048) {
          thread_group_size = ((axis_size / 2 + elements_per_simd - 1) / elements_per_simd) * simd_size;
        }
        thread_group_size = std::min(thread_group_size, static_cast<int>(scanPSO.maxTotalThreadsPerThreadgroup));

        auto tmp_grid_dims = get_2d_grid_dims(input_tensor.sizes(), input_tensor.strides(), axis_size);

        [computeEncoder dispatchThreads:MTLSizeMake(thread_group_size, tmp_grid_dims.first, tmp_grid_dims.second)
                  threadsPerThreadgroup:MTLSizeMake(thread_group_size, 1, 1)];
      } else {
        // Strided scan dispatch (scanning non-innermost dimension)
        size_t stride = input_tensor.strides()[wrapped_dim];
        int bn = 32;
        size_t stride_blocks = (stride + bn - 1) / bn;

        mtl_setBytes(computeEncoder, axis_size, 2);
        mtl_setBytes(computeEncoder, stride, 3);
        mtl_setBytes(computeEncoder, stride_blocks, 4);

        int n_reads = (input_tensor.element_size() <= 4) ? 4 : 2;
        int n_simdgroups = bn / n_reads;
        int thread_group_size = n_simdgroups * 32;

        auto tmp_grid_dims = get_2d_grid_dims(input_tensor.sizes(), input_tensor.strides(), axis_size * stride);
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

  // Post-process: copy result back to original output tensor if needed
  if (output_needs_copy) {
    output.copy_(output_tensor);
  }
}

// Generic scan implementation that handles both simple scans and scans with indices
static void scan_mps_impl_generic(const Tensor& self,
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
            // Simple scan (cumsum, cumprod)
            mtl_setArgs<2>(computeEncoder, num_rows, static_cast<uint32_t>(row_size));
          } else {
            // Scan with indices (cummin, cummax)
            mtl_setArgs<3>(computeEncoder, num_rows, static_cast<uint32_t>(row_size));
          }
        } else {
          if (outputs.size() == 1) {
            // Simple scan (cumsum, cumprod)
            mtl_setArgs<2>(computeEncoder, num_orows, num_irows, static_cast<uint32_t>(row_size));
          } else {
            // Scan with indices (cummin, cummax)
            mtl_setArgs<3>(computeEncoder, num_orows, num_irows, static_cast<uint32_t>(row_size));
          }
        }
      } else {
        // Strided kernels - pass full tensor information
        if (outputs.size() == 1) {
          // Simple scan (cumsum, cumprod)
          mtl_setArgs<2>(computeEncoder,
                         self.sizes(),
                         self.strides(),
                         outputs[0].strides(),
                         static_cast<uint32_t>(self.ndimension()),
                         static_cast<uint32_t>(wrapped_dim));
        } else {
          // Scan with indices (cummin, cummax)
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

} // namespace mps

static void cumsum_mps_kernel(const Tensor& result, const Tensor& self, int64_t dim) {
  mps::scan_mps_impl(self, result, dim, "cumsum");
}

static void cumprod_mps_kernel(const Tensor& result, const Tensor& self, int64_t dim) {
  mps::scan_mps_impl(self, result, dim, "cumprod");
}

void cummax_helper_mps(const Tensor& self, Tensor& values, Tensor& indices, int64_t dim) {
  mps::scan_mps_impl_generic(self, {values, indices}, dim, "cummax");
}

void cummin_helper_mps(const Tensor& self, Tensor& values, Tensor& indices, int64_t dim) {
  mps::scan_mps_impl_generic(self, {values, indices}, dim, "cummin");
}

// Register dispatch functions
REGISTER_DISPATCH(cumsum_stub, &cumsum_mps_kernel)
REGISTER_DISPATCH(cumprod_stub, &cumprod_mps_kernel)

} // namespace at::native
