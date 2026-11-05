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

#include <fmt/format.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ScalarOps.h>
#include <ATen/ops/_int_mm_native.h>
#include <ATen/ops/addbmm_native.h>
#include <ATen/ops/addmm_native.h>
#include <ATen/ops/addr_native.h>
#include <ATen/ops/baddbmm_native.h>
#include <ATen/ops/bmm_native.h>
#include <ATen/ops/mm_native.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/zeros.h>
#endif

#include <c10/util/TypeCast.h>
#include <c10/util/env.h>
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

  const auto m = c10::checked_convert<uint32_t>(self.size(0), "self.size(0)");
  const auto k = c10::checked_convert<uint32_t>(self.size(1), "self.size(1)");
  const auto n = c10::checked_convert<uint32_t>(mat2.size(1), "mat2.size(1)");
  constexpr auto int_max = static_cast<uint32_t>(std::numeric_limits<int32_t>::max());
  constexpr const char* mpp_kernel = "int_mm_mpp_64_64_4";
  // MPP integer matmul requires operands with matching signedness, while _int_mm permits uint8 only for self.
  const bool use_mpp = self.scalar_type() == at::kChar && self.stride(1) == 1 && mat2.stride(1) == 1 && m <= int_max &&
      k <= int_max && n <= int_max && self.stride(0) <= int_max && mat2.stride(0) <= int_max && has_mpp() &&
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

std::tuple<MPSGraphTensor*, MPSGraphTensor*, MPSGraphTensor*> do_mm(MPSGraph* graph,
                                                                    const Tensor& self,
                                                                    const Tensor& other) {
  if (self.numel() == 0 || other.numel() == 0) {
    auto output = [graph constantWithScalar:0.0
                                      shape:getMPSShape({self.size(0), other.size(1)})
                                   dataType:getMPSDataType(self)];
    return {nil, nil, output};
  }
  auto selfTensor_ = mpsGraphRankedPlaceHolder(graph, self);
  auto otherTensor_ = mpsGraphRankedPlaceHolder(graph, other);
  auto selfTensor = self.is_conj() ? [graph conjugateWithTensor:selfTensor_ name:nil] : selfTensor_;
  auto otherTensor = other.is_conj() ? [graph conjugateWithTensor:otherTensor_ name:nil] : otherTensor_;
  auto output = [graph matrixMultiplicationWithPrimaryTensor:selfTensor secondaryTensor:otherTensor name:nil];
  return {selfTensor_, otherTensor_, output};
}

bool use_metal_mm(const Tensor& self, const Tensor& other, const Tensor& output) {
  static bool always_use_metal = c10::utils::has_env("PYTORCH_MPS_PREFER_METAL");
  constexpr auto max_stride_size = 32768;
  constexpr auto max_complex_inner_size = 2048;
  if (always_use_metal || c10::isIntegralType(self.scalar_type(), true)) {
    return true;
  }
  // MPSGraph mis-writes a non-contiguous output before macOS 26; the metal
  // kernels honor the output strides.
  static const bool is_macos_26_0_or_newer = is_macos_at_least(MacOSVersion::MACOS_26_0);
  if (!output.is_contiguous() && !is_macos_26_0_or_newer) {
    return true;
  }
  // multiplicationWithPrimaryTensor: returns incorrect results if inner size exceeds 2048
  // See https://github.com/pytorch/pytorch/issues/167727#issuecomment-3529308548
  if (c10::isComplexType(self.scalar_type()) && self.size(1) > max_complex_inner_size) {
    return true;
  }
  // Detect conditions that would trigger LORADOWN GEMV kernel with potential padding overflow
  // See https://github.com/pytorch/pytorch/issues/178056
  if (self.scalar_type() == at::ScalarType::Half && (self.size(0) <= 16 || other.size(1) <= 16) &&
      self.stride(1) == 1 && other.stride(0) == 1) {
    int64_t self_padding = self.stride(0) - self.size(1);
    int64_t other_padding = other.stride(1) - other.size(0);

    if (self_padding > 15 || other_padding > 15 || self_padding % 4 != 0 || other_padding % 4 != 0) {
      TORCH_WARN_ONCE(
          "MPS mm implementation has a known issue with this shape, dtype and slice. Dispatching to metal implementation instead. This may impact performance.");
      return true;
    }
  }

  // On Apple7/8, MPSGraph intermittently corrupts matmuls with a reduction
  // dimension over 2^15 when both output dimensions use the matrix kernels;
  // whether a given call misbehaves depends on allocator/session state, and
  // fully contiguous operands are affected too. Apple9+ handles this
  // correctly.
  static const bool is_affected_gpu = !is_apple_family_or_newer(AppleGPUFamily::APPLE_9_PLUS);
  constexpr int64_t min_matrix_dim = 16;
  return is_affected_gpu && self.size(1) > max_stride_size && self.size(0) >= min_matrix_dim &&
      other.size(1) >= min_matrix_dim;
}

} // anonymous namespace

static Tensor& mm_out_mps_impl(const Tensor& self, const Tensor& other, Tensor& output) {
  using namespace mps;
  using CachedGraph = MPSBinaryCachedGraph;
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
  if (use_metal_mm(self, other, output)) {
    return do_metal_mm(self, other, output);
  }

  @autoreleasepool {
    std::string key = "mm_out_mps_impl" + getTensorsStringKey({self, other});

    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      std::tie(newCachedGraph->inputTensor_, newCachedGraph->otherTensor_, newCachedGraph->outputTensor_) =
          do_mm(mpsGraph, self, other);
    });
    // MPS TODO:
    // Strided API doesn't play nice with complex data types (at least not in case of matmul).
    // MPSGraph's matrixMultiplication produces incorrect results with stride-0 NDArray
    // inputs (only every 16th row is computed) on macOS < 26.4, and again on macOS 27.0.0;
    // 26.4+ and 27.0.1+ are fine. On affected versions, contiguify such tensors by disabling
    // the strided API so they go through the gather/clone path first.
    // See https://github.com/pytorch/pytorch/issues/180201
    static const bool has_stride_zero_mm_bug = !is_macos_at_least(MacOSVersion::MACOS_26_4) ||
        (is_macos_at_least(MacOSVersion::MACOS_27_0) && !is_macos_at_least(MacOSVersion::MACOS_27_0_1));
    auto hasZeroStride = [](const Tensor& t) {
      return std::ranges::any_of(t.strides(), [](auto s) { return s == 0; });
    };
    auto useStridedSelf = !isComplexType(self.scalar_type()) && (!has_stride_zero_mm_bug || !hasZeroStride(self));
    auto useStridedOther = !isComplexType(other.scalar_type()) && (!has_stride_zero_mm_bug || !hasZeroStride(other));
    auto selfPlaceholder = self.numel() != 0
        ? Placeholder(cachedGraph->inputTensor_, self, nil, true, MPSDataTypeInvalid, useStridedSelf)
        : Placeholder();
    auto otherPlaceholder = other.numel() != 0
        ? Placeholder(cachedGraph->otherTensor_, other, nil, true, MPSDataTypeInvalid, useStridedOther)
        : Placeholder();
    auto outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);

    auto feeds = self.numel() != 0 ? dictionaryFromPlaceholders(selfPlaceholder, otherPlaceholder) : nil;
    runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), feeds, outputPlaceholder);
  }

  return output;
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

  // Use Metal kernels for integer and complex types
  if (c10::isIntegralType(batch1.scalar_type(), true) || c10::isComplexType(batch1.scalar_type())) {
    return do_metal_addbmm_or_baddbmm(input, batch1, batch2, alpha, beta, result, opType == BADDBMM_OP_TYPE);
  }

  auto stream = getCurrentMPSStream();

  struct CachedGraph : public mps::MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* batch1Tensor_ = nil;
    MPSGraphTensor* batch2Tensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
  };

  @autoreleasepool {
    std::string key = (opType == ADDBMM_OP_TYPE) ? ("addbmm_out_mps_impl") : ("baddbmm_out_mps_impl");
    key += getTensorsStringKey({batch1, batch2, input}) + ":" + std::to_string(beta.toDouble()) + ":" +
        std::to_string(alpha.toDouble());

    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mps::mpsGraphRankedPlaceHolder(mpsGraph, input);
      MPSGraphTensor* batch1Tensor = mps::mpsGraphRankedPlaceHolder(mpsGraph, batch1);
      MPSGraphTensor* batch2Tensor = mps::mpsGraphRankedPlaceHolder(mpsGraph, batch2);

      // Intermediate for alpha
      MPSGraphTensor* alphaTensor = [mpsGraph constantWithScalar:alpha.toDouble()
                                                        dataType:getMPSScalarType(batch1.scalar_type())];

      MPSGraphTensor* productTensor = [mpsGraph matrixMultiplicationWithPrimaryTensor:batch1Tensor
                                                                      secondaryTensor:batch2Tensor
                                                                                 name:@"(batch1@batch2)"];

      MPSGraphTensor* reductionSumTensor = productTensor;
      if (opType == ADDBMM_OP_TYPE) {
        reductionSumTensor = [mpsGraph reductionSumWithTensor:productTensor axis:0 name:@"reductionSum(batch1@batch2)"];
      }

      // Intermediate for multiplying by alpha
      MPSGraphTensor* reductionSumTimesAlphaTensor =
          [mpsGraph multiplicationWithPrimaryTensor:reductionSumTensor
                                    secondaryTensor:alphaTensor
                                               name:@"alpha*(batch1@batch2)"];

      // When beta == 0, input is ignored so nan/inf in it are not propagated (matches CPU/CUDA and addmm).
      const double betaVal = beta.toDouble();
      MPSGraphTensor* outputTensor = reductionSumTimesAlphaTensor;
      if (betaVal != 0.0) {
        MPSGraphTensor* betaTensor = [mpsGraph constantWithScalar:betaVal
                                                         dataType:getMPSScalarType(input.scalar_type())];
        MPSGraphTensor* biasTimesBetaTensor = [mpsGraph multiplicationWithPrimaryTensor:inputTensor
                                                                        secondaryTensor:betaTensor
                                                                                   name:@"beta*input"];
        outputTensor = [mpsGraph additionWithPrimaryTensor:reductionSumTimesAlphaTensor
                                           secondaryTensor:biasTimesBetaTensor
                                                      name:@"beta*input + alpha*(batch1@batch2)"];
      }

      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->batch1Tensor_ = batch1Tensor;
      newCachedGraph->batch2Tensor_ = batch2Tensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

    Placeholder inputPlaceholder = Placeholder(cachedGraph->inputTensor_, input);
    Placeholder batch1Placeholder = Placeholder(cachedGraph->batch1Tensor_, batch1);
    Placeholder batch2Placeholder = Placeholder(cachedGraph->batch2Tensor_, batch2);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, result);

    auto feeds = dictionaryFromPlaceholders(inputPlaceholder, batch1Placeholder, batch2Placeholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }

  return result;
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

  // Complex addmm must use the Metal kernel (like bmm/baddbmm): the MPSGraph-based fallback
  // for real types below encodes alpha/beta as real (double) constants via toDouble(), which
  // throws for a complex scalar and drops its imaginary part.
  if (use_metal_mm(self, other, output) || c10::isComplexType(self.scalar_type())) {
    return do_metal_addmm(self, other, output, alpha, beta, *bias_);
  }

  bool is_beta_non_zero = beta.toDouble() != 0.0;

  struct CachedGraph : public mps::MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* selfTensor_ = nil;
    MPSGraphTensor* otherTensor_ = nil;
    MPSGraphTensor* biasTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
  };

  @autoreleasepool {
    std::string key = "addmm_out_mps_impl" + getTensorsStringKey({self, other, *bias_}) + ":" +
        std::to_string(beta.toDouble()) + ":" + std::to_string(alpha.toDouble());
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      auto biasTensor = mpsGraphRankedPlaceHolder(mpsGraph, *bias_);
      auto biasTensor_ = bias_->is_conj() ? [mpsGraph conjugateWithTensor:biasTensor name:nil] : biasTensor;

      // TODO: Use alpha and beta here with fill_.Scalar and mul
      auto [selfTensor, otherTensor, productTensor] = do_mm(mpsGraph, self, other);

      auto productTimesAlphaTensor = productTensor;
      if (alpha.toDouble() != 1.0) {
        auto alphaTensor = [mpsGraph constantWithScalar:alpha.toDouble() dataType:getMPSScalarType(self.scalar_type())];

        productTimesAlphaTensor = [mpsGraph multiplicationWithPrimaryTensor:productTensor
                                                            secondaryTensor:alphaTensor
                                                                       name:@"MM/alpha*(mat1@mat2)"];
      }
      auto biasTimesBetaTensor = biasTensor_;
      if (is_beta_non_zero && beta.toDouble() != 1.0) {
        auto betaTensor = [mpsGraph constantWithScalar:beta.toDouble()
                                              dataType:getMPSScalarType((*bias_).scalar_type())];
        biasTimesBetaTensor = [mpsGraph multiplicationWithPrimaryTensor:biasTensor_
                                                        secondaryTensor:betaTensor
                                                                   name:@"MM/beta*input"];
      }

      MPSGraphTensor* outputTensor = productTimesAlphaTensor;
      if (is_beta_non_zero) {
        outputTensor = [mpsGraph additionWithPrimaryTensor:productTimesAlphaTensor
                                           secondaryTensor:biasTimesBetaTensor
                                                      name:@"MM/beta*input + alpha*(mat1@mat2)"];
      }

      newCachedGraph->selfTensor_ = selfTensor;
      newCachedGraph->otherTensor_ = otherTensor;
      newCachedGraph->biasTensor_ = biasTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

    Placeholder selfPlaceholder = self.numel() != 0 ? Placeholder(cachedGraph->selfTensor_, self) : Placeholder();
    Placeholder otherPlaceholder = other.numel() != 0 ? Placeholder(cachedGraph->otherTensor_, other) : Placeholder();
    Placeholder biasPlaceholder = Placeholder(cachedGraph->biasTensor_, *bias_);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);

    auto feeds = self.numel() != 0 ? dictionaryFromPlaceholders(selfPlaceholder, otherPlaceholder, biasPlaceholder)
                                   : dictionaryFromPlaceholders(biasPlaceholder);
    runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), feeds, outputPlaceholder);
  }

  return output;
}

static Tensor& tiled_bmm_out_mps_impl(const Tensor& batch1, const Tensor& batch2, Tensor& result) {
  using namespace mps;

  id<MTLBuffer> aBuffer = getMTLBufferStorage(batch1);
  id<MTLBuffer> bBuffer = getMTLBufferStorage(batch2);
  id<MTLBuffer> resBuffer = getMTLBufferStorage(result);

  MPSStream* mpsStream = getCurrentMPSStream();
  id<MTLDevice> device = MPSDevice::getInstance()->device();

  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      mpsStream->endKernelCoalescing();
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();

      uint64_t originalBatchSize = batch1.sizes().size() > 2 ? batch1.size(0) : 1;
      uint64_t aRows = batch1.size(-2);
      uint64_t bRows = batch2.size(-2);
      uint64_t resRows = result.size(-2);
      uint64_t aCols = batch1.size(-1);
      uint64_t bCols = batch2.size(-1);
      uint64_t resCols = result.size(-1);
      uint64_t aElemSize = batch1.element_size();
      uint64_t bElemSize = batch2.element_size();
      uint64_t resElemSize = result.element_size();
      MPSDataType dtype = getMPSDataType(batch1);

      uint64_t elemInMatrix = resRows * resCols;
      // if largest supported batch size is zero, we need to split up the computation more
      uint64_t largestSupportedBatchSize = floor(pow(2, 32) / elemInMatrix);
      bool tileEachMatmul = largestSupportedBatchSize == 0;
      uint64_t batchSize = largestSupportedBatchSize > 0 ? std::min(largestSupportedBatchSize, originalBatchSize) : 1;
      uint64_t lastBatchSize = originalBatchSize % batchSize;

      uint64_t aRowsTiled = aRows;
      uint64_t resRowsTiled = resRows;
      if (tileEachMatmul) {
        uint64_t maxNumRows = floor(pow(2, 32) / resCols);
        aRowsTiled = std::min(uint64_t(512), maxNumRows);
        resRowsTiled = aRowsTiled;
      }
      uint64_t lastTileSize = aRows % aRowsTiled;

      id<MTLCommandBuffer> commandBuffer = mpsStream->commandBuffer();

      auto matmul = [[MPSNDArrayMatrixMultiplication alloc] initWithDevice:device sourceCount:2];

      MPSShape* aShape = @[ @(batchSize), @(aRowsTiled), @(aCols) ];
      MPSShape* bShape = @[ @(batchSize), @(bRows), @(bCols) ];
      MPSShape* resShape = @[ @(batchSize), @(resRowsTiled), @(resCols) ];
      auto aDesc_ = [MPSNDArrayDescriptor descriptorWithDataType:dtype shape:aShape];
      aDesc_.preferPackedRows = true;
      auto bDesc_ = [MPSNDArrayDescriptor descriptorWithDataType:dtype shape:bShape];
      bDesc_.preferPackedRows = true;

      auto resDesc_ = [MPSNDArrayDescriptor descriptorWithDataType:dtype shape:resShape];
      resDesc_.preferPackedRows = true;

      getMPSProfiler().beginProfileKernel(matmul, " tiled_bmm_mps", {batch1, batch2}, mpsStream);

      // Descriptors to use for last batch if it exists
      //.matrices is a readonly property so we need a separate descriptor.
      MPSNDArrayDescriptor *aDescLastBatch_, *bDescLastBatch_, *resDescLastBatch_;
      if (lastBatchSize != 0) {
        aDescLastBatch_ = [MPSNDArrayDescriptor descriptorWithDataType:dtype
                                                                 shape:@[ @(lastBatchSize), @(aRowsTiled), @(aCols) ]];
        aDescLastBatch_.preferPackedRows = true;
        bDescLastBatch_ = [MPSNDArrayDescriptor descriptorWithDataType:dtype
                                                                 shape:@[ @(lastBatchSize), @(bRows), @(bCols) ]];
        bDescLastBatch_.preferPackedRows = true;
        resDescLastBatch_ =
            [MPSNDArrayDescriptor descriptorWithDataType:dtype
                                                   shape:@[ @(lastBatchSize), @(resRowsTiled), @(resCols) ]];
        resDescLastBatch_.preferPackedRows = true;
      }

      MPSNDArrayDescriptor *aDescLastTile_, *resDescLastTile_;
      if (lastTileSize != 0) {
        aDescLastTile_ = [MPSNDArrayDescriptor descriptorWithDataType:dtype
                                                                shape:@[ @(batchSize), @(lastTileSize), @(aCols) ]];
        aDescLastTile_.preferPackedRows = true;
        resDescLastTile_ = [MPSNDArrayDescriptor descriptorWithDataType:dtype
                                                                  shape:@[ @(batchSize), @(lastTileSize), @(resCols) ]];
        resDescLastTile_.preferPackedRows = true;
      }

      uint64_t requiredIterations = ceil(float(originalBatchSize) / batchSize);
      uint64_t requiredTileIterations = ceil(float(aRows) / aRowsTiled);
      auto aDesc = aDesc_;
      auto bDesc = bDesc_;
      auto resDesc = resDesc_;
      for (const auto i : c10::irange(requiredIterations)) {
        if (i == requiredIterations - 1 && lastBatchSize != 0) {
          aDesc = aDescLastBatch_;
          bDesc = bDescLastBatch_;
          resDesc = resDescLastBatch_;
        }
        for (const auto j : c10::irange(requiredTileIterations)) {
          if (j == requiredTileIterations - 1 && lastTileSize != 0) {
            aDesc = aDescLastTile_;
            resDesc = resDescLastTile_;
          }
          const uint64_t aArrayOffset = i * batchSize * aCols * aRows + j * aRowsTiled * aCols;
          const uint64_t bArrayOffset = i * batchSize * bCols * bRows;
          const uint64_t resArrayOffset = i * batchSize * resCols * resRows + j * resRowsTiled * resCols;

          auto aMatrix = [[[MPSNDArray alloc] initWithBuffer:aBuffer
                                                      offset:(batch1.storage_offset() + aArrayOffset) * aElemSize
                                                  descriptor:aDesc] autorelease];
          auto bMatrix = [[[MPSNDArray alloc] initWithBuffer:bBuffer
                                                      offset:(batch2.storage_offset() + bArrayOffset) * bElemSize
                                                  descriptor:bDesc] autorelease];
          auto resMatrix = [[[MPSNDArray alloc] initWithBuffer:resBuffer
                                                        offset:(result.storage_offset() + resArrayOffset) * resElemSize
                                                    descriptor:resDesc] autorelease];
          [matmul encodeToCommandEncoder:computeEncoder
                           commandBuffer:commandBuffer
                            sourceArrays:@[ aMatrix, bMatrix ]
                        destinationArray:resMatrix];
        }
      }
    }
  });
  return result;
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

  if (c10::isIntegralType(batch1.scalar_type(), true)) {
    return do_metal_bmm(batch1, batch2, result);
  }

  // MPSGraph mis-writes a non-contiguous output before macOS 26; the metal
  // kernel honors the output strides.
  static const bool is_macos_26_0_or_newer = is_macos_at_least(MacOSVersion::MACOS_26_0);
  if (!result.is_contiguous() && !is_macos_26_0_or_newer) {
    return do_metal_bmm(batch1, batch2, result);
  }

  // Call tiled implementation if the number of elements exceeds 2^32
  uint64_t resultSize = batch1.size(0) * batch1.size(1) * batch2.size(2);
  if (resultSize > pow(2, 32)) {
    // Tiled path uses MPSNDArray directly, so resolve conjugate views upfront
    result = tiled_bmm_out_mps_impl(batch1.resolve_conj(), batch2.resolve_conj(), result);
    return result;
  }

  MPSStream* stream = getCurrentMPSStream();

  struct CachedGraph : public mps::MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* batch1Tensor_ = nil;
    MPSGraphTensor* batch2Tensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
  };

  @autoreleasepool {
    std::string key = "bmm_out_mps_impl" + getTensorsStringKey({batch1, batch2}, true, /*exclude_shape*/ true);

    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      auto batch1Tensor = mps::mpsGraphUnrankedPlaceHolder(mpsGraph, getMPSDataType(batch1.scalar_type()));
      auto batch2Tensor = mps::mpsGraphUnrankedPlaceHolder(mpsGraph, getMPSDataType(batch2.scalar_type()));

      auto batch1TensorOp = batch1.is_conj() ? [mpsGraph conjugateWithTensor:batch1Tensor name:nil] : batch1Tensor;
      auto batch2TensorOp = batch2.is_conj() ? [mpsGraph conjugateWithTensor:batch2Tensor name:nil] : batch2Tensor;

      MPSGraphTensor* productTensor = [mpsGraph matrixMultiplicationWithPrimaryTensor:batch1TensorOp
                                                                      secondaryTensor:batch2TensorOp
                                                                                 name:@"MM/(batch1@batch2)"];

      newCachedGraph->batch1Tensor_ = batch1Tensor;
      newCachedGraph->batch2Tensor_ = batch2Tensor;
      newCachedGraph->outputTensor_ = productTensor;
    });
    Placeholder batch1Placeholder = Placeholder(cachedGraph->batch1Tensor_, batch1);
    Placeholder batch2Placeholder = Placeholder(cachedGraph->batch2Tensor_, batch2);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, result);

    auto feeds = dictionaryFromPlaceholders(batch1Placeholder, batch2Placeholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }

  return result;
}

} // namespace mps

Tensor addr_mps(const Tensor& self, const Tensor& vec1, const Tensor& vec2, const Scalar& beta, const Scalar& alpha) {
  // The reference builds a TensorIterator, so the result takes the promoted dtype, not self's.
  const auto dtype = c10::promoteTypes(c10::promoteTypes(self.scalar_type(), vec1.scalar_type()), vec2.scalar_type());
  Tensor result = at::empty({0}, self.options().dtype(dtype));
  addr_out_mps(self, vec1, vec2, beta, alpha, result);
  return result;
}

Tensor& addr_out_mps(const Tensor& self,
                     const Tensor& vec1,
                     const Tensor& vec2,
                     const Scalar& beta,
                     const Scalar& alpha,
                     Tensor& result) {
  using namespace mps;

  TORCH_CHECK(result.is_mps());
  TORCH_CHECK(vec1.dim() == 1 && vec2.dim() == 1, "tensors must be 1-D");
  TORCH_CHECK(supportedFloatingOrComplexType(vec1), "MPS device does not support addr for non-float input");

  TensorArg args[]{{result, "out", 0}, {self, "self", 1}, {vec1, "vec1", 2}, {vec2, "vec2", 3}};
  checkAllSameGPU(__func__, args);

  // The reference computes at the promoted input dtype and only then casts into `result`.
  const auto opmathType =
      c10::promoteTypes(c10::promoteTypes(self.scalar_type(), vec1.scalar_type()), vec2.scalar_type());
  TORCH_CHECK(c10::canCast(opmathType, result.scalar_type()),
              "result type ",
              opmathType,
              " can't be cast to the desired output type ",
              result.scalar_type());

  IntArrayRef vec1_sizes = vec1.sizes();
  IntArrayRef vec2_sizes = vec2.sizes();
  IntArrayRef self_sizes;

  c10::MaybeOwned<Tensor> self_;
  if (&result != &self) {
    self_ = expand_size(self, {vec1_sizes[0], vec2_sizes[0]}, "addr");
    self_sizes = self_->sizes();
  } else {
    self_ = c10::MaybeOwned<Tensor>::borrowed(self);
    self_sizes = self_->sizes();
    TORCH_CHECK(result.dim() == 2, "tensors must be 2-D");
    TORCH_CHECK(self_sizes[0] == vec1_sizes[0], "vec1_ dim 0 must match vec1 dim 0");
    TORCH_CHECK(self_sizes[1] == vec2_sizes[0], "vec1_ dim 1 must match vec2 dim 0");
  }

  if (&result != &vec1) {
    result.resize_(self_sizes);
    if (beta.toComplexDouble() != 0.0) {
      result.copy_(*self_);
    }
  }

  IntArrayRef result_sizes = result.sizes();
  if ((result_sizes[0] == 0) || (result_sizes[1] == 0)) {
    return result;
  }

  MPSStream* stream = getCurrentMPSStream();
  bool is_beta_non_zero = beta.toDouble() != 0.0;
  MPSShape* inputShape = @[ @(vec1.numel()), @(1) ];
  MPSShape* otherShape = @[ @(1), @(vec2.numel()) ];

  struct CachedGraph : public mps::MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* vec1Tensor_ = nil;
    MPSGraphTensor* vec2Tensor_ = nil;
    MPSGraphTensor* selfTensor_ = nil;
    MPSGraphTensor* resultTensor_ = nil;
  };

  @autoreleasepool {
    std::string key = "addr_out_mps_impl" + getTensorsStringKey({vec1, vec2, *self_, result}) + ":" +
        std::to_string(beta.toDouble()) + ":" + std::to_string(alpha.toDouble());
    const auto computeType = getMPSDataType(opmathType);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      auto vec1Placeholder = mps::mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(vec1), inputShape);
      auto vec2Placeholder = mps::mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(vec2), otherShape);
      auto selfPlaceholder = mps::mpsGraphRankedPlaceHolder(mpsGraph, *self_);
      auto t1 = castMPSTensor(mpsGraph, vec1Placeholder, computeType);
      auto t2 = castMPSTensor(mpsGraph, vec2Placeholder, computeType);
      auto selfTensor = castMPSTensor(mpsGraph, selfPlaceholder, computeType);

      // Intermediate as placeholder
      MPSGraphTensor* productTensor = [mpsGraph matrixMultiplicationWithPrimaryTensor:t1
                                                                      secondaryTensor:t2
                                                                                 name:@"MM/(vec1Xvec2)"];

      // Intermediates for beta and alpha
      MPSGraphTensor* betaTensor = [mpsGraph constantWithScalar:beta.toDouble() dataType:computeType];
      MPSGraphTensor* alphaTensor = [mpsGraph constantWithScalar:alpha.toDouble() dataType:computeType];

      // Intermediates for multiplying by beta and alpha
      MPSGraphTensor* productTimesAlphaTensor = [mpsGraph multiplicationWithPrimaryTensor:productTensor
                                                                          secondaryTensor:alphaTensor
                                                                                     name:@"MM/alpha*(vec1Xvec2)"];
      MPSGraphTensor* selfTimesBetaTensor = selfTensor;
      if (is_beta_non_zero) {
        selfTimesBetaTensor = [mpsGraph multiplicationWithPrimaryTensor:selfTensor
                                                        secondaryTensor:betaTensor
                                                                   name:@"MM/beta*input"];
      }

      MPSGraphTensor* resultTensor = productTimesAlphaTensor;
      if (is_beta_non_zero) {
        resultTensor = [mpsGraph additionWithPrimaryTensor:productTimesAlphaTensor
                                           secondaryTensor:selfTimesBetaTensor
                                                      name:@"MM/beta*input+alpha*(vec1@vec2)"];
      }

      newCachedGraph->vec1Tensor_ = vec1Placeholder;
      newCachedGraph->vec2Tensor_ = vec2Placeholder;
      newCachedGraph->selfTensor_ = selfPlaceholder;
      newCachedGraph->resultTensor_ = castMPSTensor(mpsGraph, resultTensor, getMPSDataType(result));
    });

    Placeholder vec1Placeholder = Placeholder(cachedGraph->vec1Tensor_, vec1, inputShape);
    Placeholder vec2Placeholder = Placeholder(cachedGraph->vec2Tensor_, vec2, otherShape);
    Placeholder selfPlaceholder = Placeholder(cachedGraph->selfTensor_, *self_);
    Placeholder resultPlaceholder = Placeholder(cachedGraph->resultTensor_, result);

    auto feeds = dictionaryFromPlaceholders(vec1Placeholder, vec2Placeholder, selfPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, resultPlaceholder);
  }

  return result;
}

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
