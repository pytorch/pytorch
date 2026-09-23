#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/ceil_div.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/Histogram.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/aminmax.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/sum.h>
#endif
#include <c10/util/irange.h>

namespace at::native {
namespace mps {

enum BIN_SELECTION_ALGORITHM {
  LINEAR_INTERPOLATION,
  LINEAR_INTERPOLATION_WITH_LOCAL_SEARCH,
  BINARY_SEARCH,
};

constexpr NSUInteger kHistcThreadsPerThreadgroup = 256;
// Keep threadgroup memory below 8 KiB so at least four local histograms can
// reside within the 32 KiB available on supported Apple GPUs.
constexpr NSUInteger kHistcMaxThreadgroupMemoryLength = 8 * 1024;
// M1 sweeps over 8, 32, 64, 128, and 256 threadgroups showed a broad
// performance plateau from 32 through 256. Use the midpoint to bound the
// number of local histograms merged into global memory.
constexpr NSUInteger kHistcMaxThreadgroups = 128;
constexpr NSUInteger kMetalThreadgroupMemoryAlignment = 16;

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/HistogramKernel_metallib.h>
#endif

template <BIN_SELECTION_ALGORITHM algorithm>
void histogramdd_kernel_impl(Tensor& hist_output,
                             const TensorList& bin_edges,
                             const Tensor& input,
                             const std::optional<Tensor>& weight) {
  TORCH_INTERNAL_ASSERT(input.dim() == 2);

  constexpr uint8_t bin_selection_algorithm = algorithm;
  const int64_t N = input.size(0);
  const bool has_weight = weight.has_value();

  if (has_weight) {
    TORCH_INTERNAL_ASSERT(weight.value().dim() == 1 && weight.value().numel() == N);
    TORCH_INTERNAL_ASSERT(weight.value().scalar_type() == input.scalar_type());
  }

  // Unweighted counts are integers, so they can be accumulated into a single
  // histogram with 32-bit atomics. Weighted sums stay on the per-thread path:
  // float atomics would make the result depend on completion order.
  const bool use_atomic = !has_weight;

  const int64_t weight_stride = has_weight ? weight.value().stride(0) : -1;
  const int64_t D = input.size(1);
  TORCH_INTERNAL_ASSERT(int64_t(bin_edges.size()) == D);
  for (const auto dim : c10::irange(D)) {
    TORCH_INTERNAL_ASSERT(hist_output.size(dim) + 1 == bin_edges[dim].numel());
  }

  if (D == 0) {
    // hist is an empty tensor in this case; nothing to do here
    return;
  }

  std::vector<int64_t> num_bin_edges(D);

  std::vector<Tensor> bin_edges_dev;
  bin_edges_dev.reserve(D);
  for (const auto dim : c10::irange(D)) {
    const Tensor& dim_edges = bin_edges[dim];
    bin_edges_dev.push_back(dim_edges.to(input.device()));
    num_bin_edges[dim] = dim_edges.numel();
  }
  // The kernel indexes the edges linearly, so they must be contiguous. cat already
  // returns a contiguous tensor; the single dimension case can be a view of the
  // caller's out= tensor.
  const Tensor bin_seq_t = (D == 1 ? bin_edges_dev[0] : at::cat(bin_edges_dev)).contiguous();

  // for MPSProfiler
  auto allTensorsList = bin_edges.vec();
  allTensorsList.push_back(input);
  if (has_weight) {
    allTensorsList.push_back(weight.value());
  }

  const auto numThreads = c10::checked_convert<uint32_t>(N, "uint32_t");
  const auto hist_sizes = hist_output.sizes();

  Tensor counts, thread_histograms;
  if (use_atomic) {
    counts = at::zeros(hist_sizes, input.options().dtype(kUInt32));
  } else {
    DimVector thread_hist_sizes(hist_sizes.size() + 1); // [n_threads, output_sizes...]
    thread_hist_sizes[0] = numThreads;
    std::copy(hist_sizes.begin(), hist_sizes.end(), thread_hist_sizes.begin() + 1);
    thread_histograms = at::zeros(
        thread_hist_sizes, hist_output.scalar_type(), std::nullopt /* layout */, kMPS, std::nullopt /* pin_memory */
    );
    TORCH_INTERNAL_ASSERT(thread_histograms.is_contiguous());
  }

  MPSStream* mpsStream = getCurrentMPSStream();

  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();

      const std::string kernel = (use_atomic ? "histogramdd_atomic_" : "histogramdd_") + scalarToMetalTypeString(input);
      id<MTLComputePipelineState> histogramPSO = lib.getPipelineStateForFunc(kernel);

      // this function call is a no-op if MPS Profiler is not enabled
      getMPSProfiler().beginProfileKernel(histogramPSO, "histogram", allTensorsList, mpsStream);

      [computeEncoder setComputePipelineState:histogramPSO];
      if (use_atomic) {
        mtl_setArgs(computeEncoder,
                    input,
                    counts,
                    input.strides(),
                    D,
                    bin_seq_t,
                    num_bin_edges,
                    counts.strides(),
                    bin_selection_algorithm);
      } else {
        mtl_setArgs(computeEncoder,
                    input,
                    weight,
                    thread_histograms,
                    input.strides(),
                    D,
                    bin_seq_t,
                    num_bin_edges,
                    thread_histograms.strides(),
                    bin_selection_algorithm,
                    weight_stride);
      }

      mtl_dispatch1DJob(computeEncoder, histogramPSO, numThreads);

      getMPSProfiler().endProfileKernel(histogramPSO, mpsStream);
    }
  });
  if (use_atomic) {
    hist_output.copy_(counts);
  } else {
    at::sum_out(hist_output, thread_histograms, /*dim=*/{0});
  }
}

static void histc_atomic_kernel_impl(Tensor& hist_output, const TensorList& bin_edges, const Tensor& input) {
  const auto input_dtype = input.scalar_type();
  TORCH_CHECK_NOT_IMPLEMENTED(supportedFloatingType(input) || isIntegralType(input_dtype, /*includeBool=*/false),
                              "\"histc_mps\" not implemented for '",
                              input_dtype,
                              "'");
  TORCH_INTERNAL_ASSERT(input.dim() == 2 && input.size(1) == 1);
  TORCH_INTERNAL_ASSERT(bin_edges.size() == 1);
  TORCH_INTERNAL_ASSERT(hist_output.numel() + 1 == bin_edges[0].numel());

  const int64_t num_bins = hist_output.numel();
  const auto num_elements = c10::checked_convert<uint32_t>(input.numel(), "uint32_t");
  Tensor counts = at::zeros({num_bins}, input.options().dtype(kUInt32));
  if (num_elements == 0) {
    hist_output.copy_(counts);
    return;
  }

  id<MTLDevice> device = MPSDevice::getInstance()->device();
  MPSStream* mpsStream = getCurrentMPSStream();

  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();

      const NSUInteger max_threadgroup_memory_length =
          std::min<NSUInteger>(kHistcMaxThreadgroupMemoryLength, [device maxThreadgroupMemoryLength]);
      const bool use_threadgroup =
          num_bins <= c10::checked_convert<int64_t>(max_threadgroup_memory_length / sizeof(uint), "int64_t");
      const std::string kernel =
          fmt::format("histc_atomic_{}_{}", use_threadgroup ? "threadgroup" : "global", scalarToMetalTypeString(input));
      auto histogramPSO = lib.getPipelineStateForFunc(kernel);
      getMPSProfiler().beginProfileKernel(histogramPSO, "histc", {input, counts}, mpsStream);
      [computeEncoder setComputePipelineState:histogramPSO];
      mtl_setArgs(computeEncoder, input, counts, input.stride(0), num_elements, num_bins, bin_edges[0]);

      if (use_threadgroup) {
        const NSUInteger threadgroup_memory_length = at::round_up(
            c10::checked_convert<NSUInteger>(num_bins, "NSUInteger") * sizeof(uint), kMetalThreadgroupMemoryAlignment);
        const NSUInteger threadgroup_size =
            std::min<NSUInteger>(kHistcThreadsPerThreadgroup, [histogramPSO maxTotalThreadsPerThreadgroup]);
        const NSUInteger threadgroups =
            std::min<NSUInteger>(at::ceil_div(NSUInteger(num_elements), threadgroup_size), kHistcMaxThreadgroups);
        const uint32_t total_threads = threadgroups * threadgroup_size;
        mtl_setArgs<6>(computeEncoder, total_threads);
        [computeEncoder setThreadgroupMemoryLength:threadgroup_memory_length atIndex:0];
        [computeEncoder dispatchThreadgroups:MTLSizeMake(threadgroups, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(threadgroup_size, 1, 1)];
      } else {
        mtl_dispatch1DJob(computeEncoder, histogramPSO, num_elements);
      }
      getMPSProfiler().endProfileKernel(histogramPSO, mpsStream);
    }
  });
  hist_output.copy_(counts);
}

template <BIN_SELECTION_ALGORITHM bin_algorithm>
static void histogramdd_out_mps_template(const Tensor& self,
                                         const std::optional<Tensor>& weight,
                                         bool density,
                                         Tensor& hist,
                                         const TensorList& bin_edges) {
  hist.fill_(0);

  const int64_t N = self.size(-1);
  const int64_t M =
      std::accumulate(self.sizes().begin(), self.sizes().end() - 1, (int64_t)1, std::multiplies<int64_t>());

  const Tensor reshaped_input = self.reshape({M, N});

  const auto reshaped_weight =
      weight.has_value() ? std::optional<Tensor>(weight.value().reshape({M})) : std::optional<Tensor>();

  // CPU dispatches histogram over floating types only; integral inputs were
  // accepted here but overflow the output, which carries the input's dtype.
  TORCH_CHECK_NOT_IMPLEMENTED(
      supportedFloatingType(self), "\"histogram_mps\" not implemented for '", self.scalar_type(), "'");
  mps::histogramdd_kernel_impl<bin_algorithm>(hist, bin_edges, reshaped_input, reshaped_weight);

  /* Divides each bin's value by the total count/weight in all bins,
   * and by the bin's volume.
   */
  if (density) {
    const auto hist_sum = hist.sum().item();
    hist.div_(hist_sum);

    /* For each dimension, divides each bin's value
     * by the bin's length in that dimension.
     */
    for (const auto dim : c10::irange(N)) {
      const auto bin_lengths = bin_edges[dim].diff();

      // Used to reshape bin_lengths to align with the corresponding dimension of hist.
      std::vector<int64_t> shape(N, 1);
      shape[dim] = bin_lengths.numel();

      hist.div_(bin_lengths.reshape(shape));
    }
  }
}
} // namespace mps

// TODO: Remove once codegen emits device checks for MPS; it only does so for backends with device guards (CUDA, XPU)
static void check_same_device(const Tensor& input, const Tensor& hist) {
  TORCH_CHECK(hist.device() == input.device(),
              "Expected out tensor to have device ",
              input.device(),
              ", but got ",
              hist.device(),
              " instead");
}

static void histogramdd_kernel(const Tensor& self,
                               const std::optional<Tensor>& weight,
                               bool density,
                               Tensor& hist,
                               const TensorList& bin_edges) {
  check_same_device(self, hist);
  mps::histogramdd_out_mps_template<mps::BINARY_SEARCH>(self, weight, density, hist, bin_edges);
}

static void histogramdd_linear_kernel(const Tensor& self,
                                      const std::optional<Tensor>& weight,
                                      bool density,
                                      Tensor& hist,
                                      const TensorList& bin_edges,
                                      bool local_search) {
  check_same_device(self, hist);
  if (local_search) {
    // histogramdd codepath: both hist and bin_edges are eventually returned as output,
    // so we'll keep them consistent. CPU accepts half and bfloat16 here, but MPS
    // linspace disagrees with CPU's in those dtypes (float32 is exact), and the
    // edges are part of the result, so widening this would return edges and counts
    // that do not match CPU.
    TORCH_CHECK(self.scalar_type() == kFloat, "histogram is only supported for float32");
    mps::histogramdd_out_mps_template<mps::LINEAR_INTERPOLATION_WITH_LOCAL_SEARCH>(
        self, weight, density, hist, bin_edges);
  } else {
    // histc codepath: bin_edges are not returned to the caller
    TORCH_INTERNAL_ASSERT(!weight.has_value() && !density);
    mps::histc_atomic_kernel_impl(hist, bin_edges, self);
  }
}

static void histogram_select_outer_bin_edges_kernel(const Tensor& input,
                                                    const int64_t N,
                                                    std::vector<double>& leftmost_edges,
                                                    std::vector<double>& rightmost_edges) {
  auto [min, max] = at::aminmax(input, 0);

  for (const auto i : c10::irange(N)) {
    leftmost_edges[i] = min[i].item().to<double>();
    rightmost_edges[i] = max[i].item().to<double>();
  }
}

REGISTER_DISPATCH(histogramdd_stub, &histogramdd_kernel)
REGISTER_DISPATCH(histogramdd_linear_stub, &histogramdd_linear_kernel)
REGISTER_DISPATCH(histogram_select_outer_bin_edges_stub, &histogram_select_outer_bin_edges_kernel)
} // namespace at::native
