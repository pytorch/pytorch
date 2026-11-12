#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/OpMathType.h>
#include <ATen/ceil_div.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/BatchLinearAlgebra.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/LinearAlgebra.h>
#include <fmt/format.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/eye.h>
#include <ATen/ops/matmul.h>
#endif

#include <algorithm>
#include <bit>

namespace at::native {
namespace mps {
#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/QR_metallib.h>
#endif

constexpr int64_t simd_size = c10::metal::simdgroup_size;
constexpr int64_t tall_aspect_ratio = 4;

static GeqrfParams<> get_geqrf_params(const Tensor& A, const Tensor& tau) {
  TORCH_CHECK_NOT_IMPLEMENTED(A.dim() <= c10::metal::max_ndim, "MPS QR: at most ", c10::metal::max_ndim, " dims");
  TORCH_CHECK_NOT_IMPLEMENTED(canUse32BitIndexMath(A) && canUse32BitIndexMath(tau),
                              "MPS QR requires tensors addressable with 32-bit indices");
  GeqrfParams params{.num_batch_dims = c10::checked_convert<int32_t>(A.dim() - 2, "int32_t")};
  for (const auto dim : c10::irange(A.dim())) {
    params.A_sizes[dim] = c10::checked_convert<uint32_t>(A.size(dim), "uint32_t");
    params.A_strides[dim] = c10::checked_convert<uint32_t>(A.stride(dim), "uint32_t");
    if (dim < tau.dim()) {
      params.tau_strides[dim] = c10::checked_convert<uint32_t>(tau.stride(dim), "uint32_t");
    }
  }
  return params;
}

// Combines the reflectors stored below the diagonal of A into V and W, so that the whole block can be
// applied with two matmuls.
static std::pair<Tensor, Tensor> householder_block(const Tensor& A, const Tensor& tau) {
  // One threadgroup per reflector, sized to spread about target_threads threads over all of them
  // with one to max_rows_per_thread rows per thread.
  constexpr int64_t target_threads = 32768;
  constexpr int64_t max_rows_per_thread = 8;
  auto rows = A.size(-2);
  auto num_groups = A.size(-1) * batchCount(A);
  auto group_size = std::clamp(target_threads / num_groups, rows / max_rows_per_thread, rows);
  auto V = at::empty(A.mT().sizes(), A.options()).mT();
  auto W = at::empty_like(V);
  auto params = get_geqrf_params(A, tau);
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto encoder = stream->commandEncoder();
      auto pso = lib.getPipelineStateForFunc(fmt::format("householder_block_{}", scalarToMetalTypeString(A)));
      getMPSProfiler().beginProfileKernel(pso, "householder_block", {A, tau}, stream);
      [encoder setComputePipelineState:pso];
      mtl_setArgs(encoder, A, tau, V, W, params);
      auto max_threads = pso.maxTotalThreadsPerThreadgroup / simd_size * simd_size;
      auto threads = std::min<NSUInteger>(std::bit_floor<uint64_t>(std::max(group_size, simd_size)), max_threads);
      [encoder dispatchThreadgroups:MTLSizeMake(num_groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
      getMPSProfiler().endProfileKernel(pso, stream);
    }
  });
  return {std::move(V), std::move(W)};
}

static void apply_householder_block(const Tensor& V, const Tensor& W, const Tensor& target) {
  // Apply I - V W^H; swap the factors when constructing Q.
  auto projection = at::matmul(W.mH(), target);
  if (target.dim() == 2) {
    target.addmm_(V, projection, 1, -1);
  } else {
    target.sub_(at::matmul(V, projection));
  }
}

static Tensor& orgqr_stub_impl(Tensor& self, const Tensor& tau) {
  if (self.numel() == 0) {
    return self;
  }

  auto rows = self.size(-2);
  auto rows_squared = rows * rows;
  auto cols = self.size(-1);
  auto num_reflectors = tau.size(-1);

  if (tau.numel() == 0) {
    return self.copy_(at::eye(rows, cols, self.options()));
  }

  if (rows > simd_size) {
    auto opmath_dtype = at::toOpMathType(self.scalar_type());
    if (self.scalar_type() != opmath_dtype) {
      auto result = self.to(opmath_dtype);
      orgqr_stub_impl(result, tau.to(result.scalar_type()));
      return self.copy_(result);
    }
    // Tall-skinny inputs spend most of their time building blocks, so they use narrower ones.
    constexpr int64_t tall_min_rows = 2048;
    constexpr int64_t tall_block_size = 32;
    constexpr int64_t default_block_size = 128;
    auto is_tall = rows > tall_min_rows && rows > tall_aspect_ratio * cols;
    auto block_size = is_tall ? tall_block_size : default_block_size;
    auto reflectors = cloneBatchedColumnMajor(self.narrow(-1, 0, num_reflectors));
    auto tau_work = self.is_alias_of(tau) ? tau.clone() : tau;
    // Apply the blocks of reflectors to the identity starting from the last one: rows and columns
    // before a block's start then still hold the identity, so the block only updates the rest of Q.
    self.copy_(at::eye(rows, cols, self.options()));
    for (int64_t start = (num_reflectors - 1) / block_size * block_size; start >= 0; start -= block_size) {
      auto width = std::min<int64_t>(block_size, num_reflectors - start);
      auto panel = reflectors.narrow(-2, start, rows - start).narrow(-1, start, width);
      auto [V, W] = householder_block(panel, tau_work.narrow(-1, start, width));
      apply_householder_block(W, V, self.narrow(-2, start, rows - start).narrow(-1, start, cols - start));
    }
    return self;
  }

  auto num_batch_dims = self.dim() - 2;
  auto batch_sizes = self.sizes().slice(0, num_batch_dims);
  int64_t num_batches = c10::multiply_integers(batch_sizes);

  std::vector<int64_t> H_sizes(num_batch_dims + 2);
  for (auto dim : c10::irange(num_batch_dims)) {
    H_sizes[dim] = self.size(dim);
  }
  H_sizes[num_batch_dims] = rows;
  H_sizes[num_batch_dims + 1] = rows;

  auto H = at::empty(H_sizes, self.options().memory_format(MemoryFormat::Contiguous));
  auto H_prod = at::empty_like(H);
  auto H_prod_work = at::empty_like(H);

  OrgqrParams params;

  params.num_batch_dims = num_batch_dims;
  params.m = rows;
  params.m2 = rows_squared;
  params.n = cols;
  params.k = num_reflectors;

  for (const auto dim : c10::irange(self.dim())) {
    params.A_strides[dim] = self.stride(dim);

    if (dim < tau.dim()) {
      params.tau_strides[dim] = tau.stride(dim);
    }

    params.H_strides[dim] = H.stride(dim);
    params.H_sizes[dim] = H.size(dim);
  }

  MPSStream* stream = getCurrentMPSStream();

  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> compute_encoder = stream->commandEncoder();
      auto pipeline_state = lib.getPipelineStateForFunc(fmt::format("orgqr_{}", scalarToMetalTypeString(self)));
      getMPSProfiler().beginProfileKernel(pipeline_state, "orgqr", {self, tau}, stream);
      [compute_encoder setComputePipelineState:pipeline_state];
      mtl_setArgs(compute_encoder, self, tau, H, H_prod, H_prod_work, params);
      static_assert(sizeof(NSUInteger) == sizeof(uint64_t));
      auto max_threadgroup_size = pipeline_state.maxTotalThreadsPerThreadgroup;
      auto threads_per_group = std::min(max_threadgroup_size, NSUInteger(rows_squared));
      NSUInteger num_threads = threads_per_group * num_batches;
      [compute_encoder dispatchThreads:MTLSizeMake(num_threads, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
      getMPSProfiler().endProfileKernel(pipeline_state, stream);
    }
  });

  return self;
}

// Blocked QR over panels of up to simd_size columns. Panels with 33-1024 rows run geqrf_panel from
// registers, others run the generic geqrf kernel.
static void geqrf_kernel_mps(const Tensor& A, const Tensor& tau) {
  using namespace mps;

  TORCH_CHECK(A.scalar_type() == kFloat, "geqrf: MPS currently supports float32 only");

  if (A.numel() == 0) {
    return;
  }

  auto rows = A.size(-2);
  auto cols = A.size(-1);
  if (rows > simd_size && cols > simd_size) {
    // Tall-skinny inputs spend most of their time factoring panels, so they use narrower ones.
    constexpr int64_t tall_min_rows = 4096;
    constexpr int64_t tall_block_size = 8;
    auto is_tall = rows > tall_min_rows && rows > tall_aspect_ratio * cols;
    auto block_size = is_tall ? tall_block_size : simd_size;
    auto num_reflectors = std::min(rows, cols);
    // Right-looking blocked QR: factor a panel of block_size columns with the unblocked kernels below,
    // then apply its reflectors to the trailing columns through matmul.
    for (int64_t start = 0; start < num_reflectors; start += block_size) {
      auto width = std::min<int64_t>(block_size, num_reflectors - start);
      auto panel = A.narrow(-2, start, rows - start).narrow(-1, start, width);
      auto panel_tau = tau.narrow(-1, start, width);
      geqrf_kernel_mps(panel, panel_tau);
      if (start + width < cols) {
        auto [V, W] = householder_block(panel, panel_tau);
        auto trailing = A.narrow(-2, start, rows - start).narrow(-1, start + width, cols - start - width);
        apply_householder_block(V, W, trailing);
      }
    }
    return;
  }

  auto batch_size = batchCount(A);
  // geqrf_panel keeps the whole panel in registers: one SIMD group per column (cols <= simd_size here)
  // and a power of two rows per lane, instantiated for 8, 16 and 32.
  constexpr int64_t min_rows_per_thread = 8;
  constexpr int64_t max_rows_per_thread = 32;
  auto use_panel = rows > simd_size && rows <= max_rows_per_thread * simd_size;
  auto rows_per_thread = std::max<int64_t>(min_rows_per_thread, std::bit_ceil<uint64_t>(at::ceil_div(rows, simd_size)));
  auto pso = lib.getPipelineStateForFunc(use_panel ? fmt::format("geqrf_panel_{}", rows_per_thread)
                                                   : fmt::format("geqrf_{}", scalarToMetalTypeString(A)));
  auto v_work = use_panel ? std::nullopt : std::make_optional(at::empty({batch_size, rows}, A.options()));
  auto params = get_geqrf_params(A, tau);

  MPSStream* stream = getCurrentMPSStream();

  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto compute_encoder = stream->commandEncoder();
      getMPSProfiler().beginProfileKernel(pso, use_panel ? "geqrf_panel" : "geqrf", {A}, stream);
      [compute_encoder setComputePipelineState:pso];

      // Matrices up to simd_size x simd_size (i.e. 32x32 matrix for example)
      // get a 32-thread threadgroup instead of a full one(1024 in this case),
      // so more matrices can run at once (batched case).
      auto fits_in_simd_group = rows <= simd_size && cols <= simd_size;
      NSUInteger threads = pso.maxTotalThreadsPerThreadgroup;
      if (use_panel) {
        threads = cols * simd_size;
      } else if (fits_in_simd_group) {
        threads = simd_size;
      }
      MTLSize threadGroupSize = MTLSizeMake(threads, 1, 1);
      MTLSize gridSize = MTLSizeMake(batch_size, 1, 1);

      mtl_setArgs(compute_encoder, A, tau, params, v_work);
      [compute_encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadGroupSize];

      getMPSProfiler().endProfileKernel(pso, stream);
    }
  });
}

} // namespace mps

REGISTER_DISPATCH(orgqr_stub, mps::orgqr_stub_impl);
REGISTER_DISPATCH(geqrf_stub, mps::geqrf_kernel_mps)

} // namespace at::native
