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
  mps::scan_mps_impl(self, {result}, dim, "cumsum");
}

static void cumprod_mps_kernel(const Tensor& result, const Tensor& self, int64_t dim) {
  mps::scan_mps_impl(self, {result}, dim, "cumprod");
}

void cummax_helper_mps(const Tensor& self, Tensor& values, Tensor& indices, int64_t dim) {
  mps::scan_mps_impl(self, {values, indices}, dim, "cummax");
}

void cummin_helper_mps(const Tensor& self, Tensor& values, Tensor& indices, int64_t dim) {
  mps::scan_mps_impl(self, {values, indices}, dim, "cummin");
}

// Register dispatch functions
REGISTER_DISPATCH(cumsum_stub, &cumsum_mps_kernel)
REGISTER_DISPATCH(cumprod_stub, &cumprod_mps_kernel)

} // namespace at::native
