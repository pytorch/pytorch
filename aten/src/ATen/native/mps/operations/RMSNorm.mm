#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_fused_rms_norm_backward_native.h>
#include <ATen/ops/_fused_rms_norm_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#endif
#include <ATen/native/layer_norm.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/RMSNorm.h>
#include <fmt/format.h>

namespace at::native {
using namespace mps;

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/RMSNorm_metallib.h>
#endif

namespace {

// The grad_weight partials cost n_row_blocks * N floats to write and then reduce,
// so budget elements rather than rows: a flat row count makes that temporary grow
// with N. MIN_ROW_BLOCKS overrides the budget for very wide rows, where too few
// blocks would leave the GPU idle. That keeps the partials bounded relative to the
// input rather than absolutely -- n_row_blocks <= M, so they are at most 1x the
// input for float32 and 2x for half, never unbounded.
//
// Tuned over a 14-shape sweep on an M4 Pro, within 1.11x of the per-shape optimum;
// a flat 512 was 1.55x. Perturbing each by 4x in both directions on that hardware:
// MAX_ROW_BLOCKS barely matters (1.20-1.28x worst case across a 16x range) and the
// budget tolerates being raised but not lowered (1.28x against 1.59x), but
// MIN_ROW_BLOCKS is sensitive in both (1.45x at 8, 1.73x at 128). It is the floor a
// GPU with a materially different core count would want changed, and it is a
// constant rather than a device query because Metal exposes no core count -- only
// name, family and memory size, none of which scale with it.
constexpr size_t PARTIAL_BUDGET = 1 << 16;
constexpr size_t MIN_ROW_BLOCKS = 32;
constexpr size_t MAX_ROW_BLOCKS = 1024;

struct RmsNormKernel {
  id<MTLComputePipelineState> pso;
  size_t threadgroup_size;
  bool looped;
};

// The single-row kernel covers a row in one pass, so it needs one thread per N_READS
// elements. Whether its pipeline can field that many is a property of the kernel's
// register pressure, not of N alone, so a wide row is not on its own enough to know.
// The looped kernel strides over the row with whatever threadgroup size it is given,
// which makes it the fallback whenever the single-row one will not fit.
// maxTotalThreadsPerThreadgroup is a per-pipeline property, so it can differ
// between the dtype instantiations of one kernel. Taking the smallest keeps the
// variant and the threadgroup size a function of the shape alone: otherwise two
// dtypes at the same shape could reduce in a different order, and comparing them
// for exact equality -- which test_fused_rms_norm_weight_multiply_in_fp32 does --
// would depend on the device.
size_t rms_norm_thread_limit(const std::string& base_name) {
  const auto smallest_across_dtypes = [](const char* base) {
    size_t limit = std::numeric_limits<size_t>::max();
    for (const auto* dtype_name : {"float", "half", "bfloat"}) {
      auto pso = lib.getPipelineStateForFunc(fmt::format("{}_{}", base, dtype_name));
      limit = std::min(limit, static_cast<size_t>([pso maxTotalThreadsPerThreadgroup]));
    }
    return limit;
  };
  if (base_name == "rms_norm_backward") {
    static const size_t backward_limit = smallest_across_dtypes("rms_norm_backward");
    return backward_limit;
  }
  static const size_t forward_limit = smallest_across_dtypes("rms_norm");
  return forward_limit;
}

RmsNormKernel rms_norm_pick_kernel(const std::string& base_name, const std::string& dtype_name, size_t N) {
  const size_t thread_limit = rms_norm_thread_limit(base_name);
  if (N <= LOOPED_LIMIT) {
    const auto threads_needed = c10::metal::ceil_div(N, size_t(N_READS));
    const size_t threadgroup_size = c10::metal::round_up(threads_needed, size_t(c10::metal::simdgroup_size));
    if (threadgroup_size <= thread_limit) {
      return {lib.getPipelineStateForFunc(fmt::format("{}_{}", base_name, dtype_name)), threadgroup_size, false};
    }
  }
  auto pso = lib.getPipelineStateForFunc(fmt::format("{}_looped_{}", base_name, dtype_name));
  // threadgroup_sum stages through simdgroup_size floats, so a threadgroup wider
  // than simdgroup_size^2 would index past that scratch.
  constexpr size_t max_reducible = size_t(c10::metal::simdgroup_size) * c10::metal::simdgroup_size;
  return {pso, std::min(thread_limit, max_reducible), true};
}

std::pair<size_t, size_t> rms_norm_split_M_N(const Tensor& input, IntArrayRef normalized_shape, const Tensor& weight) {
  const auto M_N = _check_layer_norm_inputs(input, normalized_shape, weight, weight);
  // Both reach the kernels as 32-bit counts, and a threadgroup id is a uint, so a
  // larger tensor would wrap to a smaller row count and quietly leave most of the
  // output unwritten. Element offsets are computed in 64 bits and are unaffected.
  // The backward strides rows by num_row_blocks in 32-bit, so the final increment
  // must not wrap; leaving a block's worth of headroom keeps the loop terminating.
  constexpr int64_t kMaxRows = std::numeric_limits<uint32_t>::max() - MAX_ROW_BLOCKS;
  constexpr int64_t kMaxCount = std::numeric_limits<uint32_t>::max();
  TORCH_CHECK(M_N.first <= kMaxRows && M_N.second <= kMaxCount,
              "rms_norm on MPS is limited to ",
              kMaxRows,
              " rows and ",
              kMaxCount,
              " elements per row, but got ",
              M_N.first,
              " and ",
              M_N.second);
  return {static_cast<size_t>(M_N.first), static_cast<size_t>(M_N.second)};
}

// Shape the forward's rstd so it broadcasts against the input, matching CUDA.
std::vector<int64_t> rms_norm_stat_shape(const Tensor& input, IntArrayRef normalized_shape) {
  const auto axis = input.dim() - static_cast<int64_t>(normalized_shape.size());
  std::vector<int64_t> stat_shape(input.sizes().begin(), input.sizes().begin() + axis);
  stat_shape.resize(input.dim(), 1);
  return stat_shape;
}

} // anonymous namespace

std::tuple<Tensor, Tensor> _fused_rms_norm_mps(const Tensor& input,
                                               IntArrayRef normalized_shape,
                                               const std::optional<Tensor>& weight_opt,
                                               const std::optional<double> eps) {
  TORCH_CHECK(weight_opt.has_value() && weight_opt->defined(), "_fused_rms_norm_mps requires a weight tensor");
  const Tensor weight = weight_opt.value().contiguous();
  const Tensor input_contig = input.contiguous();
  // One kernel is instantiated per dtype and every buffer is bound as that type,
  // so a mismatch would reinterpret the weight's bytes rather than convert them.
  TORCH_CHECK(weight.scalar_type() == input.scalar_type(),
              "Expected weight to have dtype ",
              input.scalar_type(),
              " but got ",
              weight.scalar_type());
  // The kernel accumulates in float, so the default matches what CUDA uses for a
  // float accumulate type. Defaulting to double epsilon here would leave eps far
  // below what float32 can represent against the mean of squares.
  const auto eps_val = eps.value_or(std::numeric_limits<float>::epsilon());

  const auto MN = rms_norm_split_M_N(input_contig, normalized_shape, weight);
  const size_t M = MN.first;
  const size_t N = MN.second;
  auto output = at::empty_like(input_contig);
  auto rstd = at::empty({static_cast<int64_t>(M)}, input_contig.options().dtype(kFloat));
  if (M == 0 || N == 0) {
    // No kernel runs, so rstd would otherwise be handed back uninitialized. An
    // empty normalized dim means a mean over no elements, which is what the
    // composite produces for the same input.
    rstd.fill_(std::numeric_limits<float>::quiet_NaN());
    return std::make_tuple(std::move(output), rstd.view(rms_norm_stat_shape(input_contig, normalized_shape)));
  }

  const auto kernel = rms_norm_pick_kernel("rms_norm", scalarToMetalTypeString(input_contig), N);
  id<MTLComputePipelineState> pso = kernel.pso;
  const size_t threadgroup_size = kernel.threadgroup_size;

  MPSStream* mpsStream = getCurrentMPSStream();

  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      getMPSProfiler().beginProfileKernel(pso, "rms_norm", {input_contig, weight});
      [computeEncoder setComputePipelineState:pso];
      mtl_setArgs(computeEncoder, input_contig, weight, output, rstd, eps_val, static_cast<uint32_t>(N));

      [computeEncoder dispatchThreads:MTLSizeMake(M * threadgroup_size, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threadgroup_size, 1, 1)];
      getMPSProfiler().endProfileKernel(pso);
    }
  });

  return std::make_tuple(std::move(output), rstd.view(rms_norm_stat_shape(input_contig, normalized_shape)));
}

std::tuple<Tensor, Tensor> _fused_rms_norm_backward_mps(const Tensor& grad_out,
                                                        const Tensor& input,
                                                        IntArrayRef normalized_shape,
                                                        const Tensor& rstd,
                                                        const std::optional<Tensor>& weight_opt,
                                                        std::array<bool, 2> grad_input_mask) {
  TORCH_CHECK(weight_opt.has_value() && weight_opt->defined(), "_fused_rms_norm_backward_mps requires a weight tensor");
  const Tensor weight = weight_opt.value().contiguous();
  const Tensor input_contig = input.contiguous();
  const Tensor grad_out_contig = grad_out.contiguous();
  TORCH_CHECK(rstd.scalar_type() == kFloat, "Expected rstd to have dtype Float but got ", rstd.scalar_type());
  const Tensor rstd_contig = rstd.contiguous();
  TORCH_CHECK(weight.scalar_type() == input.scalar_type() && grad_out.scalar_type() == input.scalar_type(),
              "Expected weight and grad_out to have dtype ",
              input.scalar_type(),
              " but got ",
              weight.scalar_type(),
              " and ",
              grad_out.scalar_type());

  const auto MN = rms_norm_split_M_N(input_contig, normalized_shape, weight);
  const size_t M = MN.first;
  const size_t N = MN.second;
  TORCH_CHECK(grad_out_contig.sizes() == input_contig.sizes(),
              "Expected grad_out to have shape ",
              input_contig.sizes(),
              " but got ",
              grad_out_contig.sizes());
  TORCH_CHECK(static_cast<size_t>(rstd_contig.numel()) == M,
              "Expected rstd to have ",
              M,
              " elements but got ",
              rstd_contig.numel());
  const bool compute_dx = grad_input_mask[0];
  const bool compute_dw = grad_input_mask[1];
  auto grad_input = compute_dx ? at::empty_like(input_contig) : Tensor();
  if (M == 0 || N == 0 || !(compute_dx || compute_dw)) {
    return std::make_tuple(std::move(grad_input), compute_dw ? at::zeros_like(weight) : Tensor());
  }

  const auto kernel = rms_norm_pick_kernel("rms_norm_backward", scalarToMetalTypeString(input_contig), N);
  id<MTLComputePipelineState> pso = kernel.pso;
  const size_t threadgroup_size = kernel.threadgroup_size;
  const bool looped = kernel.looped;

  // The cap exists only to bound the grad_weight partials. With no partials to
  // write there is nothing to bound, and one block per row is strictly more
  // parallel -- which is what the forward does.
  const size_t max_row_blocks = compute_dw ? std::min(std::max(PARTIAL_BUDGET / N, MIN_ROW_BLOCKS), MAX_ROW_BLOCKS) : M;
  const size_t rows_per_block = c10::metal::ceil_div(M, max_row_blocks);
  const size_t n_row_blocks = c10::metal::ceil_div(M, rows_per_block);
  const auto partial_sizes = {static_cast<int64_t>(n_row_blocks), static_cast<int64_t>(N)};
  const auto partial_options = input_contig.options().dtype(kFloat);
  // The looped kernel accumulates into these in device memory; the single-row one
  // keeps its partials in registers and writes each element exactly once.
  Tensor grad_weight_partial;
  if (compute_dw) {
    // at::empty is only safe because the single-row kernel writes every partial
    // element exactly once, which needs its threadgroup to span the whole row.
    TORCH_INTERNAL_ASSERT(looped || threadgroup_size * N_READS >= N);
    grad_weight_partial =
        looped ? at::zeros(partial_sizes, partial_options) : at::empty(partial_sizes, partial_options);
  }
  // A gradient the caller did not ask for is bound as a null buffer and its
  // stores are switched off in the kernel, so it costs no allocation or traffic.
  const std::optional<Tensor> dx_arg = compute_dx ? std::optional<Tensor>(grad_input) : std::nullopt;
  const std::optional<Tensor> dw_arg = compute_dw ? std::optional<Tensor>(grad_weight_partial) : std::nullopt;

  Tensor grad_weight;
  id<MTLComputePipelineState> reduce_pso = nil;
  if (compute_dw) {
    grad_weight = at::empty(weight.sizes(), weight.options());
    reduce_pso =
        lib.getPipelineStateForFunc(fmt::format("rms_norm_reduce_partials_{}", scalarToMetalTypeString(weight)));
  }

  MPSStream* mpsStream = getCurrentMPSStream();

  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      getMPSProfiler().beginProfileKernel(pso, "rms_norm_backward", {input_contig, grad_out_contig});
      [computeEncoder setComputePipelineState:pso];
      mtl_setArgs(computeEncoder,
                  input_contig,
                  weight,
                  grad_out_contig,
                  rstd_contig,
                  dx_arg,
                  dw_arg,
                  static_cast<uint32_t>(N),
                  static_cast<uint32_t>(M),
                  static_cast<uint32_t>(compute_dx),
                  static_cast<uint32_t>(compute_dw));

      [computeEncoder dispatchThreads:MTLSizeMake(n_row_blocks * threadgroup_size, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threadgroup_size, 1, 1)];
      getMPSProfiler().endProfileKernel(pso);

      if (compute_dw) {
        getMPSProfiler().beginProfileKernel(reduce_pso, "rms_norm_reduce_partials", {grad_weight_partial});
        [computeEncoder setComputePipelineState:reduce_pso];
        mtl_setArgs(computeEncoder,
                    grad_weight_partial,
                    grad_weight,
                    static_cast<uint32_t>(N),
                    static_cast<uint32_t>(n_row_blocks));
        [computeEncoder dispatchThreads:MTLSizeMake(N, REDUCE_SLICES, 1)
                  threadsPerThreadgroup:MTLSizeMake(REDUCE_COLS, REDUCE_SLICES, 1)];
        getMPSProfiler().endProfileKernel(reduce_pso);
      }
    }
  });

  return std::make_tuple(std::move(grad_input), std::move(grad_weight));
}

} // namespace at::native
