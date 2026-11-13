//  Copyright © 2022 Apple Inc.

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/ceil_div.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/LinearAlgebra.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/mps/MPSGraphSequoiaOps.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/LinearAlgebra.h>
#include <ATen/native/mps/operations/GemmHeuristics.h>
#include <ATen/native/mps/operations/MatmulUtils.h>
#include <utility>

#include <fmt/format.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ScalarOps.h>
#include <ATen/ops/_int_mm_native.h>
#include <ATen/ops/addbmm_native.h>
#include <ATen/ops/addmm_native.h>
#include <ATen/ops/baddbmm_native.h>
#include <ATen/ops/bmm_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/mm_native.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/view_as_real.h>
#include <ATen/ops/zeros.h>
#endif

#include <c10/util/safe_conv.h>
#include <algorithm>
#include <limits>
#include <string>
#include <string_view>
#include <unordered_set>

namespace at::native {
namespace mps {
namespace {
#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/Matmul_metallib.h>
#endif

// Union to hold alpha and beta scalar values in the appropriate type for Metal kernels
union AlphaBeta {
  std::array<int64_t, 2> i64;
  std::array<int32_t, 2> i32;
  std::array<float, 2> f32;
  std::array<c10::complex<float>, 2> c64;
};

AlphaBeta make_alpha_beta(const Scalar& alpha, const Scalar& beta, ScalarType scalar_type) {
  AlphaBeta alpha_beta{};
  if (scalar_type == kLong) {
    alpha_beta.i64 = {alpha.toLong(), beta.toLong()};
  } else if (c10::isIntegralType(scalar_type, true)) {
    alpha_beta.i32 = {alpha.toInt(), beta.toInt()};
  } else if (c10::isComplexType(scalar_type)) {
    alpha_beta.c64 = {alpha.toComplexFloat(), beta.toComplexFloat()};
  } else {
    alpha_beta.f32 = {alpha.toFloat(), beta.toFloat()};
  }
  return alpha_beta;
}

// Describes how a kernel reads one logical matrix.
struct ResolvedMatrix {
  Tensor tensor;
  int64_t ld; // leading stride in the selected orientation
  int64_t stride; // stride along the reduction/output dimension
  bool transposed; // true when stored column-major
};

ResolvedMatrix resolve_matrix(const Tensor& matrix) {
  const auto rows = matrix.size(0);
  const auto cols = matrix.size(1);
  const auto row_stride = matrix.stride(0);
  const auto col_stride = matrix.stride(1);
  if (col_stride == 1 && row_stride >= cols) {
    return {matrix, row_stride, col_stride, false};
  }
  if (row_stride == 1 && col_stride >= rows) {
    return {matrix, col_stride, row_stride, true};
  }
  return {matrix, row_stride, col_stride, false};
}

struct GemvLaunch {
  std::string kernel;
  MTLComputePipelineState_t pso;
  NSUInteger threads_per_tg;
  int64_t num_groups;
};

std::optional<GemvConfig> normalize_gemv_config(GemvConfig config,
                                                c10::ScalarType dt,
                                                bool use_t,
                                                bool matrix_contiguous,
                                                int64_t align) {
  if (config.kernel == GemvKernel::T2D) {
    const int t2d_vec = static_cast<int>(16 / c10::elementSize(dt));
    if (!use_t || !matrix_contiguous || (align & (t2d_vec - 1))) {
      return std::nullopt;
    }
    return config;
  }
  return GemvPolicy::clamp_vec(config, align);
}

std::string gemv_kernel_name(c10::ScalarType dt,
                             GemvConfig config,
                             GemmEpilogue epi,
                             bool use_t,
                             bool matrix_contiguous,
                             bool idx64,
                             int64_t vec_xs,
                             int64_t vec_offset) {
  using namespace std::string_view_literals;

  const auto dt_str = scalarToMetalTypeString(dt);
  const bool use_t2d = use_t && config.kernel == GemvKernel::T2D;
  // Vectorized x loads need x unit-stride and VEC-aligned (nt only).
  const bool xc = !use_t && config.vec > 1 && vec_xs == 1 && (vec_offset % config.vec) == 0;
  const auto epi_str = epi == GemmEpilogue::Bias ? "ab" : "none";
  const auto matrix_str = matrix_contiguous ? "" : "_strided";
  const auto idx_str = idx64 ? "_i64" : "";
  const auto x_str = xc ? "xc"sv : "xs"sv;
  if (use_t2d) {
    return fmt::format("gemv_t2d_{}_{}_{}_{}", dt_str, config.nsimd, config.kq, epi_str);
  }
  if (use_t) {
    return fmt::format("gemv_t_{}_{}_{}_{}{}{}", dt_str, config.nsimd, config.vec, epi_str, matrix_str, idx_str);
  }
  return fmt::format(
      "gemv_nt_{}_{}_{}_{}_{}{}{}", dt_str, config.nsimd, config.vec, epi_str, x_str, matrix_str, idx_str);
}

GemvLaunch prepare_gemv_launch(c10::ScalarType dt,
                               GemvConfig config,
                               GemmEpilogue epi,
                               bool use_t,
                               bool matrix_contiguous,
                               bool idx64,
                               int64_t outlen,
                               int64_t vec_xs,
                               int64_t vec_offset) {
  const bool use_t2d = use_t && config.kernel == GemvKernel::T2D;
  auto kernel = gemv_kernel_name(dt, config, epi, use_t, matrix_contiguous, idx64, vec_xs, vec_offset);
  const int t2d_vec = static_cast<int>(16 / c10::elementSize(dt));
  const int64_t rows_per_tg = use_t2d ? (c10::metal::simdgroup_size / config.kq) * t2d_vec
      : use_t                         ? c10::metal::simdgroup_size * config.vec
                                      : config.nsimd;
  auto pso = lib.getPipelineStateForFunc(kernel);
  return {std::move(kernel),
          pso,
          static_cast<NSUInteger>(config.nsimd * c10::metal::simdgroup_size),
          at::ceil_div(outlen, rows_per_tg)};
}

void encode_gemv_launch(at::mps::MPSStream* stream,
                        const GemvLaunch& launch,
                        const Tensor& mat,
                        const Tensor& vec,
                        const Tensor& out,
                        const GemvDims& dims,
                        const Tensor& bias,
                        const std::array<float, 2>& alpha_beta) {
  getMPSProfiler().beginProfileKernel(launch.pso, "gemm_gemv/" + launch.kernel, {mat, vec}, stream);
  auto enc = stream->commandEncoder();
  [enc setComputePipelineState:launch.pso];
  mtl_setArgs(enc, mat, vec, out, dims, bias, alpha_beta);
  [enc dispatchThreadgroups:MTLSizeMake(launch.num_groups, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(launch.threads_per_tg, 1, 1)];
  getMPSProfiler().endProfileKernel(launch.pso, stream);
}

// Rank-1 GEMV launch. Matrix orientation selects gemv_t vs gemv_nt; one
// GemvDims packing handles all four mat/vec layouts.
void dispatch_gemv(const Tensor& A,
                   const Tensor& B,
                   const Tensor& out,
                   const std::optional<Tensor>& bias,
                   const ResolvedMatrix& matrix,
                   const Scalar& alpha,
                   const Scalar& beta,
                   GemmEpilogue epi,
                   const GemvPolicy& policy,
                   bool m_is_one,
                   int64_t outlen,
                   bool idx64) {
  const auto dt = out.scalar_type();
  constexpr int64_t r = 0, c = 1;
  const auto K = A.size(1);

  // gemv_t when the output runs along the matrix's columns; else gemv_nt.
  const bool gemv_use_t = m_is_one ? !matrix.transposed : matrix.transposed;
  const bool matrix_contiguous = matrix.stride == 1;
  const int64_t align = matrix_contiguous ? matrix.ld | matrix.tensor.storage_offset() : 0;
  const auto vvec = m_is_one ? A : B;
  const auto vec_xs = m_is_one ? A.stride(c) : B.stride(r);
  const auto vec_offset = vvec.storage_offset();

  Tensor expanded_bias;
  int64_t out_stride = 0;
  if (epi == GemmEpilogue::Bias) {
    TORCH_INTERNAL_ASSERT(bias.has_value());
    expanded_bias = bias->expand_as(out);
    out_stride = m_is_one ? expanded_bias.stride(c) : expanded_bias.stride(r);
  } else {
    expanded_bias = matrix.tensor; // dummy binding for buffer(4); never dereferenced
  }

  // gemv_t indexes bias at (0,n); gemv_nt indexes it at (row,0).
  GemvDims dims;
  dims.n = outlen;
  dims.K = K;
  dims.ld = matrix.ld;
  dims.ms = matrix.stride;
  dims.xs = vec_xs;
  dims.bias_r = gemv_use_t ? 0 : out_stride;
  dims.bias_c = gemv_use_t ? out_stride : 0;
  const std::array<float, 2> alpha_beta = {static_cast<float>(alpha.toDouble()), static_cast<float>(beta.toDouble())};

  GemvConfig config;
  if (idx64) {
    // Offsets overflow int32: such operands are DRAM-bound, so skip the
    // policy and use the fixed configs the _i64 variants are built at.
    config = gemv_use_t ? GemvConfig{16, 2} : GemvConfig{8, dt == at::kFloat ? 4 : 8};
  } else {
    config = gemv_use_t ? policy.pick_t(dt, outlen, K, align) : policy.pick_nt(dt, outlen, K, align);
  }
  // T2D loads a full 16 bytes per lane; misaligned matrices fall back to the
  // scalar-column standard kernel.
  auto normalized = normalize_gemv_config(config, dt, gemv_use_t, matrix_contiguous, align);
  if (normalized.has_value()) {
    config = *normalized;
  } else {
    config.kernel = GemvKernel::Standard;
    config.vec = matrix_contiguous ? 1 : 2;
  }

  auto stream = getCurrentMPSStream();
  const auto launch =
      prepare_gemv_launch(dt, config, epi, gemv_use_t, matrix_contiguous, idx64, outlen, vec_xs, vec_offset);
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      encode_gemv_launch(stream, launch, matrix.tensor, vvec, out, dims, expanded_bias, alpha_beta);
    }
  });
}

// Rank-1 GEMV gate for mm/addmm: dispatch when out is rank-1 (M==1 xor N==1)
// and unit-stride along its length, otherwise return false to fall back to MPSGraph
// A=(M,K), B=(K,N).
bool try_mps_gemv(const Tensor& A,
                  const Tensor& B,
                  const Tensor& out,
                  const std::optional<Tensor>& bias,
                  const Scalar& alpha,
                  const Scalar& beta,
                  GemmEpilogue epi) {
  const auto dtype = out.scalar_type();
  if (dtype != at::kFloat && dtype != at::kHalf && dtype != at::kBFloat16) {
    return false;
  }
  const auto M = A.size(0);
  const auto N = B.size(1);
  const auto K = A.size(1);
  if (M != 1 && N != 1) {
    return false;
  }
  const auto m_is_one = (M == 1);
  const auto outlen = m_is_one ? N : M;
  // Output must be unit-stride along its length for the flat store.
  const auto out_unit = m_is_one ? (out.stride(1) == 1) : (out.stride(0) == 1);
  if (!out_unit) {
    return false;
  }
  const auto matrix = resolve_matrix(m_is_one ? B : A);
  // Dispatch 64-bit-index variants when numel or storage offsets exceed int32.
  const bool idx64 = !offsetsFitIn<int32_t>(A, B, out) || (bias.has_value() && !offsetsFitIn<int32_t>(*bias));
  const GemvPolicy policy = GemvPolicy::current();
  dispatch_gemv(A, B, out, bias, matrix, alpha, beta, epi, policy, m_is_one, outlen, idx64);
  return true;
}

Tensor& do_metal_mm(const Tensor& self, const Tensor& other, Tensor& output) {
  // Handle conjugated inputs by creating resolved copies
  auto self_ = self.is_conj() ? self.resolve_conj() : self;
  auto other_ = other.is_conj() ? other.resolve_conj() : other;

  auto stream = getCurrentMPSStream();
  auto device = MPSDevice::getInstance()->device();
  auto matmulPSO = lib.getPipelineStateForFunc("matmul_" + mps::scalarToMetalTypeString(output));
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      getMPSProfiler().beginProfileKernel(matmulPSO, "matmul", {self_, other_}, stream);
      auto computeEncoder = stream->commandEncoder();
      [computeEncoder setComputePipelineState:matmulPSO];
      c10::metal::vec3<uint32_t> sizes = {static_cast<uint32_t>(self_.size(0)),
                                          static_cast<uint32_t>(self_.size(1)),
                                          static_cast<uint32_t>(output.size(1))};
      std::array<int64_t, 6> strides = {
          self_.stride(0), self_.stride(1), other_.stride(0), other_.stride(1), output.stride(0), output.stride(1)};
      constexpr uint32_t TILE_DIM = 16; // fastest performance from tests on multiple macs
      uint32_t gridSizeX = (output.size(1) + TILE_DIM - 1) / TILE_DIM;
      uint32_t gridSizeY = (self_.size(0) + TILE_DIM - 1) / TILE_DIM;

      MTLSize threadsPerThreadgroup = MTLSizeMake(TILE_DIM, TILE_DIM, 1);
      MTLSize threadgroupsPerGrid = MTLSizeMake(gridSizeX, gridSizeY, 1);
      mtl_setArgs(computeEncoder, self_, other_, output, strides, sizes);
      [computeEncoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
      getMPSProfiler().endProfileKernel(matmulPSO, stream);
    }
  });
  return output;
}

Tensor& int_mm_out_mps_impl(const Tensor& self, const Tensor& mat2, Tensor& result) {
  static constexpr std::string_view func_name = "int_mm_out_mps";
  check_mm_shapes(self, mat2, "_int_mm");
  TORCH_CHECK(self.dtype() == at::kChar || self.dtype() == at::kByte,
              func_name,
              ": Expected self dtype to be int8 or uint8 but got ",
              self.dtype());
  TORCH_CHECK(mat2.dtype() == at::kChar, func_name, ": Expected mat2 dtype to be of type int8 but got ", mat2.dtype());
  TORCH_CHECK(
      result.dtype() == at::kInt, func_name, ": Expected result dtype to be of type kInt but got ", result.dtype());
  TORCH_CHECK(result.size(0) == self.size(0),
              func_name,
              ": Expected result.size(0) to be ",
              self.size(0),
              " but got ",
              result.size(0));
  TORCH_CHECK(result.size(1) == mat2.size(1),
              func_name,
              ": Expected result.size(1) to be ",
              mat2.size(1),
              " but got ",
              result.size(1));
  TORCH_CHECK(result.dim() == 2, func_name, ": Expected result to be of dimension 2 but got ", result.dim());
  TORCH_CHECK(result.is_contiguous(), func_name, ": Expected result to be contiguous.");

  if (result.numel() == 0 || self.size(1) == 0) {
    return result.zero_();
  }

  // The kernels tile the output, so its threads cover m and n rounded up to the tile size.
  TORCH_CHECK(std::in_range<int32_t>(self.size(0)) && std::in_range<int32_t>(mat2.size(1)),
              func_name,
              ": MPS does not support output of size ",
              result.sizes());
  const auto m = c10::safe_conv<uint32_t>(self.size(0), "uint32_t");
  const auto k = c10::safe_conv<uint32_t>(self.size(1), "uint32_t");
  const auto n = c10::safe_conv<uint32_t>(mat2.size(1), "uint32_t");
  constexpr const char* mpp_kernel = "int_mm_mpp_64_64_4";
  // MPP integer matmul requires operands with matching signedness, while _int_mm permits uint8 only for self.
  const bool use_mpp = self.scalar_type() == at::kChar && self.stride(1) == 1 && mat2.stride(1) == 1 &&
      std::in_range<int32_t>(m) && std::in_range<int32_t>(k) && std::in_range<int32_t>(n) &&
      std::in_range<int32_t>(self.stride(0)) && std::in_range<int32_t>(mat2.stride(0)) && has_mpp() &&
      lib.hasFunction(mpp_kernel);
  const auto kernel_name = use_mpp ? mpp_kernel : self.scalar_type() == at::kByte ? "int_mm_uchar" : "int_mm_char";
  auto stream = getCurrentMPSStream();
  auto pso = lib.getPipelineStateForFunc(kernel_name);

  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      getMPSProfiler().beginProfileKernel(pso, "int_mm", {self, mat2}, stream);
      auto computeEncoder = stream->commandEncoder();
      [computeEncoder setComputePipelineState:pso];
      const c10::metal::vec3<uint32_t> sizes = {m, k, n};
      const std::array<int64_t, 6> strides = {
          self.stride(0), self.stride(1), mat2.stride(0), mat2.stride(1), result.stride(0), result.stride(1)};
      const uint32_t tile_dim = use_mpp ? 64 : 16;
      const auto grid_size_x = at::ceil_div(n, tile_dim);
      const auto grid_size_y = at::ceil_div(m, tile_dim);
      const auto threads_per_threadgroup = use_mpp ? MTLSizeMake(128, 1, 1) : MTLSizeMake(16, 16, 1);

      mtl_setArgs(computeEncoder, self, mat2, result, strides, sizes);
      [computeEncoder dispatchThreadgroups:MTLSizeMake(grid_size_x, grid_size_y, 1)
                     threadsPerThreadgroup:threads_per_threadgroup];
      getMPSProfiler().endProfileKernel(pso, stream);
    }
  });
  return result;
}

Tensor& do_metal_bmm(const Tensor& batch1, const Tensor& batch2, Tensor& output) {
  // Handle conjugated inputs by creating resolved copies
  auto batch1_ = batch1.is_conj() ? batch1.resolve_conj() : batch1;
  auto batch2_ = batch2.is_conj() ? batch2.resolve_conj() : batch2;

  auto stream = getCurrentMPSStream();
  auto device = MPSDevice::getInstance()->device();
  auto matmulPSO = lib.getPipelineStateForFunc("naive_bmm_" + mps::scalarToMetalTypeString(output));
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      getMPSProfiler().beginProfileKernel(matmulPSO, "naive_batch_matmul", {batch1_, batch2_}, stream);
      auto computeEncoder = stream->commandEncoder();
      [computeEncoder setComputePipelineState:matmulPSO];
      std::array<uint32_t, 4> sizes = {static_cast<uint32_t>(batch1_.size(1)),
                                       static_cast<uint32_t>(batch1_.size(2)),
                                       static_cast<uint32_t>(output.size(2)),
                                       static_cast<uint32_t>(output.size(0))};
      std::array<int64_t, 9> strides = {batch1_.stride(2),
                                        batch1_.stride(1),
                                        batch1_.stride(0),
                                        batch2_.stride(2),
                                        batch2_.stride(1),
                                        batch2_.stride(0),
                                        output.stride(2),
                                        output.stride(1),
                                        output.stride(0)};
      constexpr uint32_t TILE_DIM = 16;
      uint32_t gridSizeX = (output.size(2) + TILE_DIM - 1) / TILE_DIM;
      uint32_t gridSizeY = (batch1_.size(1) + TILE_DIM - 1) / TILE_DIM;
      uint32_t gridSizeZ = output.size(0);

      MTLSize threadsPerThreadgroup = MTLSizeMake(TILE_DIM, TILE_DIM, 1);
      MTLSize threadgroupsPerGrid = MTLSizeMake(gridSizeX, gridSizeY, gridSizeZ);

      mtl_setArgs(computeEncoder, batch1_, batch2_, output, strides, sizes);
      [computeEncoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
      getMPSProfiler().endProfileKernel(matmulPSO, stream);
    }
  });
  return output;
}

Tensor& do_metal_addmm(const Tensor& self,
                       const Tensor& other,
                       Tensor& output,
                       const Scalar& alpha,
                       const Scalar& beta,
                       const Tensor& bias) {
  // beta == 0 means bias is ignored, so nan/inf in it must not propagate. Lower to a plain
  // mm when alpha == 1; otherwise hand the fused kernel a zero bias (it applies alpha before rounding).
  if (beta.toComplexDouble() == 0.0 && alpha.toComplexDouble() == 1.0) {
    do_metal_mm(self, other, output);
    return output;
  }
  // Handle conjugated inputs by creating resolved copies
  auto self_ = self.is_conj() ? self.resolve_conj() : self;
  auto other_ = other.is_conj() ? other.resolve_conj() : other;
  auto bias_ = beta.toComplexDouble() == 0.0 ? at::zeros({}, bias.options()).expand_as(output) : bias.resolve_conj();

  auto stream = getCurrentMPSStream();
  auto device = MPSDevice::getInstance()->device();
  auto matmulPSO = lib.getPipelineStateForFunc("addmm_" + mps::scalarToMetalTypeString(output));
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      getMPSProfiler().beginProfileKernel(matmulPSO, "addmm", {self_, other_}, stream);
      auto computeEncoder = stream->commandEncoder();
      [computeEncoder setComputePipelineState:matmulPSO];
      c10::metal::vec3<uint32_t> sizes = {static_cast<uint32_t>(self_.size(0)),
                                          static_cast<uint32_t>(self_.size(1)),
                                          static_cast<uint32_t>(output.size(1))};
      std::array<int64_t, 8> strides = {self_.stride(0),
                                        self_.stride(1),
                                        other_.stride(0),
                                        other_.stride(1),
                                        output.stride(0),
                                        output.stride(1),
                                        bias_.stride(0),
                                        bias_.stride(1)};
      auto alpha_beta = make_alpha_beta(alpha, beta, output.scalar_type());
      constexpr uint32_t TILE_DIM = 16; // fastest performance from tests on multiple macs
      uint32_t gridSizeX = (output.size(1) + TILE_DIM - 1) / TILE_DIM;
      uint32_t gridSizeY = (self_.size(0) + TILE_DIM - 1) / TILE_DIM;

      MTLSize threadsPerThreadgroup = MTLSizeMake(TILE_DIM, TILE_DIM, 1);
      MTLSize threadgroupsPerGrid = MTLSizeMake(gridSizeX, gridSizeY, 1);
      mtl_setArgs(computeEncoder, self_, other_, output, bias_, alpha_beta.i64, strides, sizes);
      [computeEncoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
      getMPSProfiler().endProfileKernel(matmulPSO, stream);
    }
  });
  return output;
}

Tensor& do_metal_addbmm_or_baddbmm(const Tensor& bias,
                                   const Tensor& batch1,
                                   const Tensor& batch2,
                                   const Scalar& alpha,
                                   const Scalar& beta,
                                   Tensor& output,
                                   bool is_baddbmm) {
  // beta == 0 means bias is ignored, so nan/inf in it must not propagate. baddbmm then
  // lowers to a plain bmm (skipping the bias) with an in-place alpha scale for alpha != 1.
  if (is_baddbmm && beta.toComplexDouble() == 0.0) {
    do_metal_bmm(batch1, batch2, output);
    if (alpha.toComplexDouble() != 1.0) {
      output.mul_(alpha);
    }
    return output;
  }

  // Handle conjugated inputs by creating resolved copies. addbmm still needs its batch
  // reduction, so when beta == 0 feed the kernel a broadcast scalar zero for the bias.
  auto batch1_ = batch1.is_conj() ? batch1.resolve_conj() : batch1;
  auto batch2_ = batch2.is_conj() ? batch2.resolve_conj() : batch2;
  auto bias_src = beta.toComplexDouble() == 0.0 ? at::zeros({}, bias.options()) : bias;
  auto bias_ = bias_src.is_conj() ? bias_src.resolve_conj() : bias_src;

  auto stream = getCurrentMPSStream();
  auto device = MPSDevice::getInstance()->device();
  const char* op_name = is_baddbmm ? "baddbmm" : "addbmm";
  auto matmulPSO =
      lib.getPipelineStateForFunc(std::string("naive_") + op_name + "_" + mps::scalarToMetalTypeString(output));

  // Expand bias to match output shape for broadcasting
  auto bias_expanded = bias_.expand_as(output);

  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      getMPSProfiler().beginProfileKernel(
          matmulPSO, std::string("naive_") + op_name, {batch1_, batch2_, bias_expanded}, stream);
      auto computeEncoder = stream->commandEncoder();
      [computeEncoder setComputePipelineState:matmulPSO];

      std::array<uint32_t, 4> sizes;
      if (is_baddbmm) {
        sizes = {static_cast<uint32_t>(batch1_.size(1)),
                 static_cast<uint32_t>(batch1_.size(2)),
                 static_cast<uint32_t>(output.size(2)),
                 static_cast<uint32_t>(output.size(0))};
      } else {
        sizes = {static_cast<uint32_t>(batch1_.size(1)),
                 static_cast<uint32_t>(batch1_.size(2)),
                 static_cast<uint32_t>(output.size(1)),
                 static_cast<uint32_t>(batch1_.size(0))};
      }

      auto alpha_beta = make_alpha_beta(alpha, beta, output.scalar_type());

      constexpr uint32_t TILE_DIM = 16;
      uint32_t gridSizeX = (output.size(-1) + TILE_DIM - 1) / TILE_DIM;
      uint32_t gridSizeY = (batch1_.size(1) + TILE_DIM - 1) / TILE_DIM;
      uint32_t gridSizeZ = is_baddbmm ? output.size(0) : 1;

      // Unified stride layout for both baddbmm and addbmm:
      // [0-2]: batch1 (col, row, batch)
      // [3-5]: batch2 (col, row, batch)
      // [6-8]: output (col, row, batch)
      // [9-11]: bias (col, row, batch)
      std::array<int64_t, 12> strides = {batch1_.stride(2),
                                         batch1_.stride(1),
                                         batch1_.stride(0),
                                         batch2_.stride(2),
                                         batch2_.stride(1),
                                         batch2_.stride(0),
                                         output.stride(-1),
                                         output.stride(-2),
                                         output.stride(0), // Output batch is unused for addbmm
                                         bias_expanded.stride(-1),
                                         bias_expanded.stride(-2),
                                         bias_expanded.stride(0)}; // Output bias is unused for addbmm

      MTLSize threadsPerThreadgroup = MTLSizeMake(TILE_DIM, TILE_DIM, 1);
      MTLSize threadgroupsPerGrid = MTLSizeMake(gridSizeX, gridSizeY, gridSizeZ);

      mtl_setArgs(computeEncoder, batch1_, batch2_, output, bias_expanded, alpha_beta.i64, strides, sizes);
      [computeEncoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];

      getMPSProfiler().endProfileKernel(matmulPSO, stream);
    }
  });

  return output;
}

// out = alpha * a @ b + beta * bias for 2-D or batched float32, float16 and bfloat16 operands: matmul_mpp where
// matmul2d takes the layout, else matmul_simd.
void metal_gemm(const Tensor& a,
                const Tensor& b,
                const Tensor& out,
                const Tensor& bias = {},
                const Scalar& alpha = 1,
                const Scalar& beta = 0) {
  // out stands in for a bias that is not read.
  const bool add = bias.defined() && beta.toDouble() != 0;
  const auto bias_ = add ? bias.expand_as(out) : out;
  const auto params = matmul_params<uint64_t>(a, b, out, bias_, alpha.toFloat(), add ? beta.toFloat() : 0.0f);
  const int64_t batch = out.dim() == 3 ? out.size(0) : 1;
  static const int64_t full_grid = kMatmulFullGridPerCore * MPSDevice::getInstance()->getCoreCount();
  constexpr uint32_t mpp_bn = kMatmulBlock * kMatmulBlockSimdgroups;
  const int64_t mpp_tiles = at::ceil_div(params.m, kMatmulBlock) * at::ceil_div(params.n, mpp_bn) * batch;
  const uint32_t mpp_bm = mpp_tiles < full_grid ? kMatmulBlock / 2 : kMatmulBlock;
  const bool use_mpp = has_mpp() && params.out_stride_n == 1 && params.bias_stride_n == 1 &&
      (params.a_stride_m == 1 || params.a_stride_k == 1) && (params.b_stride_k == 1 || params.b_stride_n == 1) &&
      matmul_mpp_indices_fit(params, mpp_bm, mpp_bn);
  const auto bm = use_mpp ? mpp_bm : matmul_tile_rows(params.m);
  const auto bn = use_mpp ? mpp_bn : kMatmulTileN;
  const auto simdgroups = use_mpp ? kMatmulBlockSimdgroups : matmul_simdgroups(bm, bn);
  const bool use_u32 = !use_mpp && offsetsFitIn<int32_t>(a, b, out, bias_);
  const auto dtype = scalarToMetalTypeString(out);
  const auto name = use_mpp ? fmt::format("matmul_mpp_{}{}_{}_bm{}",
                                          params.a_stride_k == 1 ? 'n' : 't',
                                          params.b_stride_k == 1 ? 't' : 'n',
                                          dtype,
                                          bm)
                            : fmt::format("matmul_simd_{}_bm{}{}", dtype, bm, mtlIdxSuffix(use_u32));
  const auto pso = lib.getPipelineStateForFunc(name);
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      getMPSProfiler().beginProfileKernel(pso, "metal_mm", {a, b}, stream);
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pso];
      if (use_u32) {
        mtl_setArgs(encoder, a, b, out, bias_, matmul_params<uint32_t>(a, b, out, bias_, params.alpha, params.beta));
      } else {
        mtl_setArgs(encoder, a, b, out, bias_, params);
      }
      [encoder dispatchThreadgroups:MTLSizeMake(at::ceil_div(params.n, bn), at::ceil_div(params.m, bm), batch)
              threadsPerThreadgroup:MTLSizeMake(simdgroups * c10::metal::simdgroup_size, 1, 1)];
      getMPSProfiler().endProfileKernel(pso, stream);
    }
  });
}

// metal_gemm, or the GEMV kernels for a 2-D product with one row or column.
void real_gemm(const Tensor& a, const Tensor& b, const Tensor& out, const Tensor& bias = {}, const Scalar& beta = 0) {
  const auto gemv_bias = bias.defined() ? std::optional<Tensor>(bias) : std::nullopt;
  const auto epilogue = bias.defined() ? GemmEpilogue::Bias : GemmEpilogue::None;
  if (a.dim() != 2 || !try_mps_gemv(a, b, out, gemv_bias, 1, beta, epilogue)) {
    metal_gemm(a, b, out, bias, 1, beta);
  }
}

// The real view of a complex matrix or batch: each row its re/im pairs. The conjugation flag is dropped; the callers
// fold it into the signs.
Tensor real_view(const Tensor& t) {
  auto sizes = t.sizes().vec();
  sizes.back() *= 2;
  return at::view_as_real(t.is_conj() ? t._conj() : t).view(sizes);
}

// X expanded by matmul_complex_embed (see MatmulComplexEmbedParams). A batch sharing one plane expands it once.
Tensor complex_expanded(const Tensor& X, bool blocks, c10::complex<float> alpha, float sign_in, float sign_out) {
  const bool batched = X.dim() == 3;
  if (batched && X.stride(0) == 0 && X.size(0) > 1) {
    return complex_expanded(X[0], blocks, alpha, sign_in, sign_out).expand({X.size(0), -1, -1});
  }
  const auto dt = c10::toRealValueType(X.scalar_type());
  const int64_t R = X.size(-2), C = X.size(-1), batch = batched ? X.size(0) : 1;
  const int64_t rows = 2 * R, cols = blocks ? 2 * C : C;
  const int64_t line = kMatmulComplexRowAlignBytes / c10::elementSize(dt), ld = at::ceil_div(cols, line) * line;
  auto out = at::empty(batched ? std::vector<int64_t>{batch, rows, ld} : std::vector<int64_t>{rows, ld},
                       X.options().dtype(dt));
  const MatmulComplexEmbedParams p{.rows = R,
                                   .cols = C,
                                   .rs = X.stride(-2),
                                   .cs = X.stride(-1),
                                   .plane = batched ? X.stride(0) : 0,
                                   .ld = ld,
                                   .out_plane = rows * ld,
                                   .alpha_re = alpha.real(),
                                   .alpha_im = alpha.imag(),
                                   .sign_in = sign_in,
                                   .sign_out = sign_out,
                                   .blocks = blocks};
  const auto pso = lib.getPipelineStateForFunc(fmt::format("matmul_complex_embed_{}", scalarToMetalTypeString(dt)));
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      getMPSProfiler().beginProfileKernel(pso, "matmul_complex_embed", {X}, stream);
      auto enc = stream->commandEncoder();
      [enc setComputePipelineState:pso];
      mtl_setArgs(enc, X, out, p);
      [enc dispatchThreadgroups:MTLSizeMake(at::ceil_div(C, int64_t{c10::metal::simdgroup_size}),
                                            at::ceil_div(R, int64_t{kMatmulComplexRowsPerThreadgroup}),
                                            batch)
          threadsPerThreadgroup:MTLSizeMake(c10::metal::simdgroup_size, kMatmulComplexRowsPerThreadgroup, 1)];
      getMPSProfiler().endProfileKernel(pso, stream);
    }
  });
  return out.narrow(-1, 0, cols);
}

// out = F's row pairs (and column pairs, signed by B's conjugation) gathered + beta * bias.
void complex_interleave(const Tensor& F,
                        const Tensor& out,
                        bool column_pairs,
                        float sign,
                        const Tensor& bias,
                        c10::complex<float> beta) {
  const bool batched = out.dim() == 3;
  const int64_t M = out.size(-2), N = out.size(-1), batch = batched ? out.size(0) : 1;
  const auto bias_ = bias.defined() ? bias : out; // out stands in for a bias that is not read
  const MatmulComplexInterleaveParams p{.M = M,
                                        .N = N,
                                        .ldf = F.stride(-2),
                                        .ldc = out.stride(-2),
                                        .f_plane = batched ? F.stride(0) : 0,
                                        .c_plane = batched ? out.stride(0) : 0,
                                        .bias_plane = batched ? bias_.stride(0) : 0,
                                        .bias_rs = bias_.stride(-2),
                                        .bias_cs = bias_.stride(-1),
                                        .beta_re = bias.defined() ? beta.real() : 0.0f,
                                        .beta_im = bias.defined() ? beta.imag() : 0.0f,
                                        .sign = sign,
                                        .column_pairs = column_pairs};
  const auto pso = lib.getPipelineStateForFunc(fmt::format("matmul_complex_interleave_{}", scalarToMetalTypeString(F)));
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      getMPSProfiler().beginProfileKernel(pso, "matmul_complex_interleave", {F}, stream);
      auto enc = stream->commandEncoder();
      [enc setComputePipelineState:pso];
      mtl_setArgs(enc, F, out, bias_, p);
      [enc dispatchThreadgroups:MTLSizeMake(at::ceil_div(N, int64_t{c10::metal::simdgroup_size}),
                                            at::ceil_div(M, int64_t{kMatmulComplexRowsPerThreadgroup}),
                                            batch)
          threadsPerThreadgroup:MTLSizeMake(c10::metal::simdgroup_size, kMatmulComplexRowsPerThreadgroup, 1)];
      getMPSProfiler().endProfileKernel(pso, stream);
    }
  });
}

// Bytes of an operand's own storage (a plane shared by a batch counts once).
int64_t operand_bytes(const Tensor& t) {
  const int64_t planes = t.dim() == 3 && t.stride(0) != 0 ? t.size(0) : 1;
  return planes * t.size(-2) * t.size(-1) * t.element_size();
}

// out = alpha * A @ B + beta * bias for complex 2-D or batched operands, as a real product of twice the size. With B
// expanded to blocks, out's real view is A's real view times it; with A expanded, the product is interleaved into out.
// alpha and the conjugations fold into the expansion. The expanded operand is the one that moves fewer bytes.
void metal_complex_gemm(const Tensor& A,
                        const Tensor& B,
                        const Tensor& out,
                        const Tensor& bias = {},
                        const Scalar& alpha = 1,
                        const Scalar& beta = 0) {
  const auto bias_ = bias.defined() ? bias.expand_as(out) : bias;
  if (out.is_conj()) {
    // The storage of a conjugated out holds the conjugate of the result.
    return metal_complex_gemm(
        A.conj(), B.conj(), out.conj(), bias_.defined() ? bias_.conj() : bias_, alpha.conj(), beta.conj());
  }
  const bool batched = A.dim() == 3;
  const int64_t M = out.size(-2), N = out.size(-1), batch = batched ? A.size(0) : 1;
  // out's pairs run along its rows: a column-major out, and a contiguous single column of it (so that the GEMV
  // kernels stream A), take the product of the transposes; any other layout a copy.
  const bool column = N == 1 && M > 1 && out.stride(-2) == 1;
  if ((out.stride(-1) != 1 && N > 1) || column) {
    if (column || M == 1 || out.stride(-2) == 1) {
      return metal_complex_gemm(B.mT(), A.mT(), out.mT(), bias_.defined() ? bias_.mT() : bias_, alpha, beta);
    }
    const auto c = at::empty_like(out, MemoryFormat::Contiguous);
    metal_complex_gemm(A, B, c, bias_, alpha, beta);
    out.copy_(c);
    return;
  }
  const auto alpha_c = alpha.toComplexFloat(), beta_c = beta.toComplexFloat();
  const bool add = bias_.defined() && beta_c != c10::complex<float>(0.0f);
  const auto bias_c = add ? bias_.resolve_conj() : Tensor();
  // B is read raw along K (stored column-major) or along N, else copied.
  const bool b_along_k = B.stride(-2) == 1 && B.stride(-1) != 1;
  const Tensor B_ = b_along_k || B.stride(-1) == 1 ? B : B.contiguous();
  const bool a_raw = A.stride(-1) == 1;
  // Bytes moved beyond the product's own reads: an expansion reads its operand and writes twice its size (into rows,
  // its size), a copy of A reads and writes it; with A expanded the interleave reads the product and writes out, and
  // with B read along N the product is twice out.
  const int64_t a_bytes = operand_bytes(A), b_bytes = operand_bytes(B_), c_bytes = operand_bytes(out);
  const int64_t expand_b = 3 * b_bytes + (a_raw ? 0 : 2 * a_bytes);
  const int64_t expand_a = b_along_k ? 3 * a_bytes + 2 * c_bytes : 2 * a_bytes + 4 * c_bytes;
  const auto conj_sign = [](const Tensor& t) { return t.is_conj() ? -1.0f : 1.0f; };
  const float sign_b = conj_sign(B_);
  if (expand_b <= expand_a) {
    const Tensor A_ = a_raw ? A : A.contiguous();
    const auto Bp = complex_expanded(B_, true, alpha_c, sign_b, conj_sign(A_));
    // The real epilogue scales the bias by a real beta: a complex one is applied first.
    const bool real_beta = beta_c.imag() == 0.0f;
    Tensor bias_r;
    if (add) {
      const auto b = bias_c.stride(-1) == 1 ? bias_c : bias_c.contiguous();
      bias_r = real_view(real_beta ? b : at::mul(b, beta));
    }
    real_gemm(real_view(A_), Bp, real_view(out), bias_r, add && real_beta ? beta_c.real() : 1.0f);
    return;
  }
  // A expanded to blocks against B's real view along K (its conjugation moving to A's side), or to rows against B's
  // real view along N.
  const float sign_a = conj_sign(A);
  const auto Ap = b_along_k
      ? complex_expanded(A, true, {alpha_c.real(), -sign_b * alpha_c.imag()}, -sign_a * sign_b, sign_b)
      : complex_expanded(A, false, alpha_c, sign_a, 1.0f);
  const auto B_r = b_along_k ? real_view(B_.mT()).mT() : real_view(B_);
  const int64_t cols = b_along_k ? N : 2 * N;
  const auto F =
      at::empty(batched ? std::vector<int64_t>{batch, 2 * M, cols} : std::vector<int64_t>{2 * M, cols}, Ap.options());
  real_gemm(Ap, B_r, F);
  complex_interleave(F, out, !b_along_k, sign_b, bias_c, beta_c);
}

} // anonymous namespace

void metal_mm_out(const Tensor& a, const Tensor& b, const Tensor& out) {
  check_mm_shapes(a, b, "metal_mm_out");
  checkSize("metal_mm_out", TensorArg(out, "out", 3), {a.size(0), b.size(1)});
  const auto scalar_type = out.scalar_type();
  TORCH_CHECK_TYPE(a.scalar_type() == scalar_type && b.scalar_type() == scalar_type,
                   "metal_mm_out: expected matching dtypes, got ",
                   a.scalar_type(),
                   ", ",
                   b.scalar_type(),
                   " and ",
                   scalar_type);
  TORCH_CHECK_TYPE(supportedFloatingType(scalar_type), "metal_mm_out: unsupported dtype ", scalar_type);
  if (out.numel() == 0) {
    return;
  }
  if (a.size(1) == 0) {
    out.zero_();
    return;
  }
  metal_gemm(a, b, out);
}

static bool naive_complex(const Tensor& a, const Tensor& out) {
  return std::max({a.size(-1), out.size(-2), out.size(-1)}) <= kMatmulComplexNaiveMaxDim;
}

static Tensor& mm_out_mps_impl(const Tensor& self, const Tensor& other, Tensor& output) {
  using namespace mps;
  TORCH_CHECK(self.dim() == 2 && other.dim() == 2, "tensors must be 2-D");
  TORCH_CHECK(self.dtype() == other.dtype(),
              "expected mat1 and mat2 to have the same dtype, but got: ",
              self.dtype(),
              " != ",
              other.dtype())
  TensorArg args[]{{output, "out", 0}, {self, "mat1", 1}, {other, "mat2", 2}};
  checkAllSameGPU("mm", args);

  TORCH_CHECK(output.is_mps());

  // Edge case behaviors must match _int_mm_out_cpu CPU implementation
  // Transpose inputs if needed
  // Outer or inner dimension is 0
  if (output.numel() == 0 || self.size(1) == 0) {
    return output.zero_();
  }

  // Rank-1 (M==1 xor N==1): route mm/mv to the metal GEMV kernels.
  if (try_mps_gemv(self, other, output, std::nullopt, 1, 0, GemmEpilogue::None)) {
    return output;
  }

  // MPS matmul returns silently incorrect results if one of the matrix dimensions is greater than 2**15
  // And crashes if its a view of matrix with dimensions larger than 2**15
  // See https://github.com/pytorch/pytorch/issues/116769#issuecomment-1888302095
  // In such cases, fallback to naive but accurate metal shader
  if (supportedFloatingType(self.scalar_type())) {
    metal_gemm(self, other, output);
    return output;
  }
  if (c10::isComplexType(self.scalar_type()) && !naive_complex(self, output)) {
    metal_complex_gemm(self, other, output);
    return output;
  }
  return do_metal_mm(self, other, output);
}

enum LinearAlgebraOpType { ADDBMM_OP_TYPE, BADDBMM_OP_TYPE };

static Tensor& addbmm_or_baddbmm_out_mps_impl(const Tensor& input,
                                              const Tensor& batch1,
                                              const Tensor& batch2,
                                              const Scalar& beta,
                                              const Scalar& alpha,
                                              Tensor& result,
                                              LinearAlgebraOpType opType) {
  using namespace mps;

  TORCH_CHECK(input.is_mps());
  TORCH_CHECK(batch1.is_mps());
  TORCH_CHECK(batch2.is_mps());
  TORCH_CHECK(result.is_mps());

  TORCH_CHECK(supportedFloatingOrComplexType(batch1) || c10::isIntegralType(batch1.scalar_type(), true),
              "MPS device does not support addbmm or baddbmm for this input type");

  // Reject what the reference rejects; MPSGraph would abort rather than raise. baddbmm is already
  // covered by its meta function.
  TORCH_CHECK(batch1.scalar_type() == batch2.scalar_type(),
              "self and mat2 must have the same dtype, but got ",
              batch1.scalar_type(),
              " and ",
              batch2.scalar_type());
  TORCH_CHECK(input.scalar_type() == batch1.scalar_type(),
              "Input dtypes must be the same, got: input ",
              input.scalar_type(),
              ", batch1: ",
              batch1.scalar_type(),
              ", batch2: ",
              batch2.scalar_type());
  TORCH_CHECK(result.scalar_type() == input.scalar_type(),
              "Expected out tensor to have dtype ",
              input.scalar_type(),
              ", but got ",
              result.scalar_type(),
              " instead");

  TORCH_CHECK(batch1.dim() == 3, "batch1 must be a 3D tensor");
  TORCH_CHECK(batch2.dim() == 3, "batch2 must be a 3D tensor");
  TORCH_CHECK(batch1.size(0) == batch2.size(0),
              "batch1 and batch2 must have same number of batches, got ",
              batch1.size(0),
              " and ",
              batch2.size(0));
  TORCH_CHECK(batch1.size(2) == batch2.size(1),
              "Incompatible matrix sizes for bmm (",
              batch1.size(1),
              "x",
              batch1.size(2),
              " and ",
              batch2.size(1),
              "x",
              batch2.size(2),
              ")");

  if (opType == ADDBMM_OP_TYPE) {
    result.resize_as_(input);
  }

  // Empty tensors would hit the Placeholder [srcBuf length] > 0 assertion.
  if (result.numel() == 0) {
    return result;
  }
  if ((opType == ADDBMM_OP_TYPE && batch1.size(0) == 0) || batch1.size(2) == 0) {
    if (beta.toComplexDouble() == 0.0) {
      result.zero_();
    } else {
      at::mul_out(result, input, wrapped_scalar_tensor(beta));
    }
    return result;
  }

  const bool complex = c10::isComplexType(batch1.scalar_type());
  if (!c10::isIntegralType(batch1.scalar_type(), true) && !(complex && naive_complex(batch1, result))) {
    // addbmm is one product over the batches' K laid end to end.
    const bool sum = opType == ADDBMM_OP_TYPE;
    const auto a = sum ? batch1.transpose(0, 1).reshape({batch1.size(1), -1}) : batch1;
    const auto b = sum ? batch2.reshape({-1, batch2.size(2)}) : batch2;
    if (complex) {
      metal_complex_gemm(a, b, result, input, alpha, beta);
    } else {
      metal_gemm(a, b, result, input, alpha, beta);
    }
    return result;
  }
  return do_metal_addbmm_or_baddbmm(input, batch1, batch2, alpha, beta, result, opType == BADDBMM_OP_TYPE);
}

static Tensor& addmm_out_mps_impl(const Tensor& bias,
                                  const Tensor& self, // input
                                  const Tensor& other, // weight
                                  const Scalar& beta,
                                  const Scalar& alpha,
                                  Tensor& output) {
  using namespace mps;

  TORCH_CHECK(output.is_mps());
  TORCH_CHECK(self.dim() == 2 && other.dim() == 2, "tensors must be 2-D");

  TensorArg args[]{{output, "out", 0}, {bias, "self", 1}, {self, "mat1", 2}, {other, "mat2", 3}};
  checkAllSameGPU(__func__, args);

  IntArrayRef mat1_sizes = self.sizes();
  IntArrayRef mat2_sizes = other.sizes();
  IntArrayRef bias_sizes;
  c10::MaybeOwned<Tensor> bias_;
  if (&output != &bias) {
    bias_ = expand_size(bias, {mat1_sizes[0], mat2_sizes[1]}, "addmm");
    bias_sizes = bias_->sizes();
  } else {
    bias_ = c10::MaybeOwned<Tensor>::borrowed(bias);
    bias_sizes = bias_->sizes();
    TORCH_CHECK(output.dim() == 2, "tensors must be 2-D");
    TORCH_CHECK(bias_sizes[0] == mat1_sizes[0], "self_ dim 0 must match mat1 dim 0");
    TORCH_CHECK(bias_sizes[1] == mat2_sizes[1], "self_ dim 1 must match mat2 dim 1");
  }

  if (&output != &self) {
    output.resize_(bias_sizes);
  }
  if (output.numel() == 0) {
    return output;
  }
  // Inner dimension is 0
  // Early out as some paths in the code below do not handle this case correctly
  if (self.size(1) == 0) {
    if (beta.toComplexDouble() == 0.0) {
      output.zero_();
    } else {
      output.copy_(*bias_);
      output.mul_(beta);
    }
    return output;
  }

  // Rank-1 (M==1 xor N==1): route addmm/addmv to the metal GEMV kernels.
  if (try_mps_gemv(self, other, output, *bias_, alpha, beta, GemmEpilogue::Bias)) {
    return output;
  }

  if (supportedFloatingType(self.scalar_type())) {
    metal_gemm(self, other, output, *bias_, alpha, beta);
    return output;
  }
  if (c10::isComplexType(self.scalar_type()) && !naive_complex(self, output)) {
    metal_complex_gemm(self, other, output, *bias_, alpha, beta);
    return output;
  }
  return do_metal_addmm(self, other, output, alpha, beta, *bias_);
}

static Tensor& bmm_out_mps_impl(const Tensor& batch1, const Tensor& batch2, Tensor& result) {
  TORCH_CHECK(batch1.scalar_type() == batch2.scalar_type(),
              "Expected arguments of same type but got ",
              batch1.scalar_type(),
              " and ",
              batch2.scalar_type());
  using namespace mps;

  // Matmul not supported if any output dimension size is larger than 2**32
  for (auto elem : result.sizes()) {
    TORCH_CHECK_NOT_IMPLEMENTED(elem <= pow(2, 32),
                                "Output dim sizes larger than 2**32 elements for matmul not supported on MPS device.");
  }

  if (batch1.numel() == 0 || batch2.numel() == 0) {
    result.zero_();
    return result;
  }

  if (supportedFloatingType(batch1.scalar_type())) {
    metal_gemm(batch1, batch2, result);
    return result;
  }
  if (c10::isComplexType(batch1.scalar_type()) && !naive_complex(batch1, result)) {
    metal_complex_gemm(batch1, batch2, result);
    return result;
  }
  return do_metal_bmm(batch1, batch2, result);
}

} // namespace mps

TORCH_IMPL_FUNC(mm_out_mps)(const Tensor& self, const Tensor& mat2, const Tensor& result) {
  mps::mm_out_mps_impl(self, mat2, const_cast<Tensor&>(result));
}

Tensor& _int_mm_out_mps(const Tensor& self, const Tensor& mat2, Tensor& result) {
  return mps::int_mm_out_mps_impl(self, mat2, result);
}

Tensor _int_mm_mps(const Tensor& self, const Tensor& mat2) {
  auto result = at::empty({self.size(0), mat2.size(1)}, self.options().dtype(at::kInt));
  return _int_mm_out_mps(self, mat2, result);
}

TORCH_IMPL_FUNC(addmm_out_mps)
(const Tensor& self,
 const Tensor& mat1,
 const Tensor& mat2,
 const Scalar& beta,
 const Scalar& alpha,
 const Tensor& result) {
  mps::addmm_out_mps_impl(self, mat1, mat2, beta, alpha, const_cast<Tensor&>(result));
}

TORCH_IMPL_FUNC(bmm_out_mps)(const Tensor& batch1, const Tensor& batch2, const Tensor& result) {
  mps::bmm_out_mps_impl(batch1, batch2, const_cast<Tensor&>(result));
}

TORCH_IMPL_FUNC(baddbmm_out_mps)
(const Tensor& self,
 const Tensor& batch1,
 const Tensor& batch2,
 const Scalar& beta,
 const Scalar& alpha,
 const Tensor& result) {
  mps::addbmm_or_baddbmm_out_mps_impl(
      self, batch1, batch2, beta, alpha, const_cast<Tensor&>(result), mps::BADDBMM_OP_TYPE);
}

Tensor& addbmm_out_mps(const Tensor& self,
                       const Tensor& batch1,
                       const Tensor& batch2,
                       const Scalar& beta,
                       const Scalar& alpha,
                       Tensor& result) {
  auto b_self = expand_size(self, {batch1.size(1), batch2.size(2)}, "addbmm_out");

  mps::addbmm_or_baddbmm_out_mps_impl(*b_self, batch1, batch2, beta, alpha, result, mps::ADDBMM_OP_TYPE);
  return result;
}

Tensor addbmm_mps(const Tensor& self,
                  const Tensor& batch1,
                  const Tensor& batch2,
                  const Scalar& beta,
                  const Scalar& alpha) {
  Tensor result = at::empty({0}, self.options());
  return addbmm_out_mps(self, batch1, batch2, beta, alpha, result);
}

Tensor& addbmm_mps_(Tensor& self, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha) {
  return addbmm_out_mps(self, batch1, batch2, beta, alpha, self);
}

} // namespace at::native
