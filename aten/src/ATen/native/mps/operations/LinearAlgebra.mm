//  Copyright © 2022 Apple Inc.

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/ceil_div.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/BatchLinearAlgebra.h>
#include <ATen/native/LinearAlgebra.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/Pool.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TransposeType.h>
#include <ATen/native/mps/MPSGraphSequoiaOps.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/LinearAlgebra.h>

#include <fmt/format.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ScalarOps.h>
#include <ATen/ops/_cholesky_solve_helper_native.h>
#include <ATen/ops/_linalg_check_errors.h>
#include <ATen/ops/_linalg_eigh.h>
#include <ATen/ops/_linalg_solve_ex_native.h>
#include <ATen/ops/addbmm_native.h>
#include <ATen/ops/addmm_native.h>
#include <ATen/ops/addr_native.h>
#include <ATen/ops/all.h>
#include <ATen/ops/baddbmm_native.h>
#include <ATen/ops/bmm_native.h>
#include <ATen/ops/eye.h>
#include <ATen/ops/eye_native.h>
#include <ATen/ops/linalg_cholesky_ex_native.h>
#include <ATen/ops/linalg_inv_ex_native.h>
#include <ATen/ops/linalg_lstsq_native.h>
#include <ATen/ops/linalg_lu_factor_ex_native.h>
#include <ATen/ops/linalg_lu_factor_native.h>
#include <ATen/ops/linalg_lu_native.h>
#include <ATen/ops/linalg_lu_solve.h>
#include <ATen/ops/linalg_qr.h>
#include <ATen/ops/linalg_qr_native.h>
#include <ATen/ops/linalg_solve_triangular_native.h>
#include <ATen/ops/linalg_svd.h>
#include <ATen/ops/linalg_vector_norm.h>
#include <ATen/ops/lu_unpack.h>
#include <ATen/ops/lu_unpack_native.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/mm_native.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/orgqr_native.h>
#include <ATen/ops/real.h>
#include <ATen/ops/slice.h>
#include <ATen/ops/stack.h>
#include <ATen/ops/triangular_solve_native.h>
#include <ATen/ops/where.h>
#include <ATen/ops/zeros.h>
#endif

#include <c10/util/env.h>
#include <algorithm>
#include <string>
#include <unordered_set>

namespace at::native {
namespace mps {
namespace {
#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/LinearAlgebra_metallib.h>
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

Tensor& do_metal_mm(const Tensor& self, const Tensor& other, Tensor& output) {
  // Handle conjugated inputs by creating resolved copies
  auto self_ = self.is_conj() ? self.resolve_conj() : self;
  auto other_ = other.is_conj() ? other.resolve_conj() : other;

  auto stream = getCurrentMPSStream();
  auto device = MPSDevice::getInstance()->device();
  auto matmulPSO = lib.getPipelineStateForFunc("matmul_" + mps::scalarToMetalTypeString(output));
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      getMPSProfiler().beginProfileKernel(matmulPSO, "matmul", {self_, other_});
      auto computeEncoder = stream->commandEncoder();
      [computeEncoder setComputePipelineState:matmulPSO];
      std::array<uint32_t, 3> sizes = {static_cast<uint32_t>(self_.size(0)),
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
      getMPSProfiler().endProfileKernel(matmulPSO);
    }
  });
  return output;
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
      getMPSProfiler().beginProfileKernel(matmulPSO, "naive_batch_matmul", {batch1_, batch2_});
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
      getMPSProfiler().endProfileKernel(matmulPSO);
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
  if (beta.isFloatingPoint() && alpha.isFloatingPoint() && beta.toDouble() == 0 && alpha.toDouble() == 1) {
    return do_metal_mm(self, other, output);
  }
  // Handle conjugated inputs by creating resolved copies
  auto self_ = self.is_conj() ? self.resolve_conj() : self;
  auto other_ = other.is_conj() ? other.resolve_conj() : other;
  auto bias_ = bias.is_conj() ? bias.resolve_conj() : bias;

  auto stream = getCurrentMPSStream();
  auto device = MPSDevice::getInstance()->device();
  auto matmulPSO = lib.getPipelineStateForFunc("addmm_" + mps::scalarToMetalTypeString(output));
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      getMPSProfiler().beginProfileKernel(matmulPSO, "addmm", {self_, other_});
      auto computeEncoder = stream->commandEncoder();
      [computeEncoder setComputePipelineState:matmulPSO];
      std::array<uint32_t, 3> sizes = {static_cast<uint32_t>(self_.size(0)),
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
      getMPSProfiler().endProfileKernel(matmulPSO);
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
  // Handle conjugated inputs by creating resolved copies
  auto batch1_ = batch1.is_conj() ? batch1.resolve_conj() : batch1;
  auto batch2_ = batch2.is_conj() ? batch2.resolve_conj() : batch2;
  auto bias_ = bias.is_conj() ? bias.resolve_conj() : bias;

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
          matmulPSO, std::string("naive_") + op_name, {batch1_, batch2_, bias_expanded});
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

      getMPSProfiler().endProfileKernel(matmulPSO);
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
  static bool is_macos_14_4_or_newer = is_macos_at_least(MacOSVersion::MACOS_14_4);
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

  return !is_macos_14_4_or_newer &&
      (self.stride(0) > max_stride_size || self.stride(1) > max_stride_size || self.size(0) > max_stride_size ||
       self.size(1) > max_stride_size || other.stride(0) > max_stride_size || other.stride(1) > max_stride_size ||
       other.size(0) > max_stride_size || other.size(1) > max_stride_size);
}

} // anonymous namespace

// Blocked right-looking LU with partial pivoting, factored in place on a
// row-major fp32 (B, M, N) buffer; pivots are 1-based, info follows LAPACK.
static void lu_factor_panel_encode(const Tensor& LU,
                                   const Tensor& pivots,
                                   const Tensor& info,
                                   int64_t M,
                                   int64_t N,
                                   int64_t B,
                                   bool transposeResult) {
  auto stream = getCurrentMPSStream();
  const bool useMpp = has_mpp();

  auto factorW32PSO = lib.getPipelineStateForFunc("factorPanelLU_1_32");
  auto factorW16PSO = lib.getPipelineStateForFunc("factorPanelLU_2_16");
  auto factorW8PSO = lib.getPipelineStateForFunc("factorPanelLU_4_8");
  auto streamUpdatePSO = lib.getPipelineStateForFunc("luStreamUpdate");
  auto streamPivotPSO = lib.getPipelineStateForFunc("luStreamPivot");
  auto laswpPSO = lib.getPipelineStateForFunc("laswpGatherLU");
  auto trsm8PSO = lib.getPipelineStateForFunc("trsmPanelLU_8");
  auto trsm16PSO = lib.getPipelineStateForFunc("trsmPanelLU_16");
  auto trsm32PSO = lib.getPipelineStateForFunc("trsmPanelLU_32");
  uint32_t maxG = static_cast<uint32_t>(std::min({factorW32PSO.maxTotalThreadsPerThreadgroup,
                                                  factorW16PSO.maxTotalThreadsPerThreadgroup,
                                                  factorW8PSO.maxTotalThreadsPerThreadgroup}));
  maxG = std::max(32u, maxG / 32 * 32);
  auto gemmBigPSO = useMpp ? lib.getPipelineStateForFunc("gemmLU_64_64_4") : nil;
  auto gemmSmallPSO = useMpp ? lib.getPipelineStateForFunc("gemmLU_32_64_2") : nil;
  auto gemmSimdPSO = useMpp ? nil : lib.getPipelineStateForFunc("gemmSimdLU");
  auto transposePSO = transposeResult ? lib.getPipelineStateForFunc("transposeInPlaceLU") : nil;

  const auto uM = static_cast<uint32_t>(M);
  const auto uN = static_cast<uint32_t>(N);
  const auto uB = static_cast<uint32_t>(B);
  const uint32_t mn = std::min(uM, uN);
  const uint32_t NBo = mn <= 1024 ? 32 : mn <= 2048 ? 64 : 128;
  // panels taller than this use the streaming kernels (kLUStreamNT argmax
  // partials and the 32-float U row per batch in scratch)
  const uint32_t kStreamMinRows = 4 * maxG;
  Tensor scratch;
  if (uM > kStreamMinRows) {
    scratch = at::empty({B, 2 * kLUStreamNT + 32}, LU.options());
  }

  @autoreleasepool {
    dispatch_sync_with_rethrow(stream->queue(), ^() {
      auto enc = stream->commandEncoder();
      mtl_setArgs(enc, LU, pivots, info);
      const uint32_t dims[2] = {uM, uN};
      [enc setBytes:dims length:8 atIndex:3];
      if (scratch.defined()) {
        mtl_setArgs<6>(enc, scratch);
      }

      auto setP4 = [&](uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
        const uint32_t p[4] = {a, b, c, d};
        [enc setBytes:p length:16 atIndex:4];
      };
      auto setP5 = [&](uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
        const uint32_t p[4] = {a, b, c, d};
        [enc setBytes:p length:16 atIndex:5];
      };
      auto laswp = [&](uint32_t d0, uint32_t nb, uint32_t cs0, uint32_t ce0, uint32_t cs1, uint32_t ce1) {
        const uint32_t W = (ce0 - cs0) + (ce1 - cs1);
        if (W == 0) {
          return;
        }
        setP5(cs0, ce0, cs1, ce1);
        [enc setComputePipelineState:laswpPSO];
        for (uint32_t s0 = 0; s0 < nb; s0 += 32) {
          setP4(d0 + s0, std::min(32u, nb - s0), 0, 0);
          [enc dispatchThreadgroups:MTLSizeMake((W + 63) / 64, uB, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
      };
      auto trsm = [&](uint32_t d0, uint32_t cs, uint32_t ce, id<MTLComputePipelineState> pso, uint32_t nr) {
        if (cs >= ce) {
          return;
        }
        setP4(d0, cs, ce, nr);
        [enc setComputePipelineState:pso];
        [enc dispatchThreadgroups:MTLSizeMake(uB, (ce - cs + 127) / 128, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      };
      auto gemm = [&](uint32_t rs, uint32_t re, uint32_t cs, uint32_t ce, uint32_t kc, uint32_t kw) {
        if (rs >= re || cs >= ce) {
          return;
        }
        setP4(rs, re, cs, ce);
        setP5(kc, kw, 0, 0);
        const uint32_t Tm = re - rs;
        const uint32_t Tn = ce - cs;
        if (!useMpp) {
          [enc setComputePipelineState:gemmSimdPSO];
          [enc dispatchThreadgroups:MTLSizeMake((Tn + 63) / 64, (Tm + 31) / 32, uB)
              threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
          return;
        }
        const bool big = Tm >= 1024 && Tn >= 64;
        const uint32_t BM = big ? 64 : 32;
        const uint32_t BN = 64;
        const uint32_t NSG = big ? 4 : 2;
        [enc setComputePipelineState:(big ? gemmBigPSO : gemmSmallPSO)];
        [enc dispatchThreadgroups:MTLSizeMake((Tn + BN - 1) / BN, (Tm + BM - 1) / BM, uB)
            threadsPerThreadgroup:MTLSizeMake(NSG * 32, 1, 1)];
      };
      auto factorSub = [&](uint32_t d0, id<MTLComputePipelineState> pso, uint32_t G) {
        setP4(d0, 0, 0, 0);
        [enc setComputePipelineState:pso];
        [enc dispatchThreadgroups:MTLSizeMake(uB, 1, 1) threadsPerThreadgroup:MTLSizeMake(G, 1, 1)];
      };
      auto factorStream = [&](uint32_t c0) {
        const uint32_t H = uM - c0;
        const uint32_t nb = std::min(32u, mn - c0);
        auto update = [&](uint32_t j, bool searchOnly) -> uint32_t {
          const uint32_t rowStart = searchOnly ? j : j + 1;
          if (rowStart >= H) {
            return 0;
          }
          const uint32_t n = H - rowStart;
          // Each threadgroup is kLUStreamWarpsPerTG simdgroups, each factoring
          // RPT rows. Pick RPT so the threadgroup count nTG stays within the
          // kLUStreamNT argmax-partial slots scratch holds per batch.
          const uint32_t RPT = std::max(1u, at::ceil_div(n, kLUStreamWarpsPerTG * kLUStreamNT));
          const uint32_t nTG = at::ceil_div(n, kLUStreamWarpsPerTG * RPT);
          setP4(c0, j, RPT, searchOnly ? 1 : 0);
          [enc setComputePipelineState:streamUpdatePSO];
          [enc dispatchThreadgroups:MTLSizeMake(nTG, uB, 1) threadsPerThreadgroup:MTLSizeMake(kLUStreamNT, 1, 1)];
          return nTG;
        };
        uint32_t npart = update(0, true);
        for (uint32_t j = 0; j < nb; j++) {
          setP4(c0, j, npart, 0);
          [enc setComputePipelineState:streamPivotPSO];
          [enc dispatchThreadgroups:MTLSizeMake(uB, 1, 1) threadsPerThreadgroup:MTLSizeMake(kLUStreamNT, 1, 1)];
          npart = update(j, false);
        }
      };
      // factor the 32-wide block at c0 via W-wide register panels
      auto factor = [&](uint32_t c0, uint32_t sw) {
        const uint32_t H = uM - c0;
        if (H > kStreamMinRows) {
          factorStream(c0);
          return;
        }
        // smallest R (widest panel) whose per-thread row count fits in maxG
        uint32_t R = 1;
        while (R < 4 && (H + R - 1) / R > maxG) {
          R *= 2;
        }
        const uint32_t W = 32 / R;
        const uint32_t G = std::min(maxG, (((H + R - 1) / R) + 31) / 32 * 32);
        const auto pso = R == 1 ? factorW32PSO : R == 2 ? factorW16PSO : factorW8PSO;
        if (W >= sw) {
          factorSub(c0, pso, G);
          return;
        }
        const auto tpso = W == 16 ? trsm16PSO : trsm8PSO;
        for (uint32_t q0 = c0; q0 < c0 + sw; q0 += W) {
          const uint32_t qw = std::min(W, c0 + sw - q0);
          factorSub(q0, pso, G);
          laswp(q0, qw, c0, q0, q0 + qw, c0 + sw);
          if (q0 + qw < c0 + sw) {
            trsm(q0, q0 + qw, c0 + sw, tpso, W);
            gemm(q0 + qw, uM, q0 + qw, c0 + sw, q0, qw);
          }
        }
      };

      for (uint32_t p0 = 0; p0 < mn; p0 += NBo) {
        const uint32_t pw = std::min(NBo, mn - p0);
        for (uint32_t c0 = p0; c0 < p0 + pw; c0 += 32) {
          const uint32_t sw = std::min(32u, mn - c0);
          factor(c0, sw);
          laswp(c0, sw, p0, c0, c0 + sw, p0 + pw);
          if (c0 + sw < p0 + pw) {
            trsm(c0, c0 + sw, p0 + pw, trsm32PSO, 32);
            gemm(c0 + sw, uM, c0 + sw, p0 + pw, c0, sw);
          }
        }
        laswp(p0, pw, 0, p0, p0 + pw, uN);
        if (p0 + pw < uN) {
          for (uint32_t b0 = p0; b0 < p0 + pw; b0 += 32) {
            trsm(b0, p0 + pw, uN, trsm32PSO, std::min(32u, mn - b0));
            if (b0 + 32 < p0 + pw) {
              gemm(b0 + 32, std::min(p0 + pw, uM), p0 + pw, uN, b0, 32);
            }
          }
        }
        if (p0 + pw < mn) {
          gemm(p0 + pw, uM, p0 + pw, uN, p0, pw);
        }
      }
      if (transposeResult) {
        [enc setComputePipelineState:transposePSO];
        const uint32_t nt = (uN + 31) / 32;
        [enc dispatchThreadgroups:MTLSizeMake(nt, nt, uB) threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
      }
    });
  }
}

static void linalg_lu_factor_ex_out_mps_impl(const Tensor& A,
                                             bool pivot,
                                             const Tensor& LU,
                                             const Tensor& pivots,
                                             const Tensor& info,
                                             bool check_errors) {
  using namespace mps;

  TORCH_CHECK(A.scalar_type() == kFloat && LU.scalar_type() == kFloat,
              "linalg.lu_factor(): MPS doesn't support complex types.");
  TORCH_CHECK(pivot, "linalg.lu_factor(): MPS doesn't allow pivot == False.");

  int64_t aRows = A.size(-2);
  int64_t aCols = A.size(-1);
  int64_t numPivots = std::min(aRows, aCols);
  std::vector<int64_t> pivot_sizes(A.sizes().begin(), A.sizes().end() - 2);
  resize_output(info, pivot_sizes);
  pivot_sizes.push_back(numPivots);
  resize_output(pivots, pivot_sizes);

  if (A.numel() == 0) {
    info.zero_();
    return;
  }

  // kernels factor row-major, the LU output is column-major: square factors
  // in the LU.mT() view and transposes in place, the rest go via a scratch
  resize_output(LU, A.sizes());
  const bool inPlace = aRows == aCols && !LU.is_same(A) && LU.mT().is_contiguous();
  Tensor work_full;
  if (inPlace) {
    work_full = LU.mT();
  } else {
    work_full = at::empty(A.sizes(), A.options());
  }
  work_full.copy_(A);
  Tensor work = work_full.dim() > 3 ? work_full.flatten(0, -3) : work_full;
  TORCH_INTERNAL_ASSERT(work.is_contiguous())
  int64_t batchSize = work.dim() > 2 ? work.size(0) : 1;

  Tensor pivots_ = pivots.is_contiguous() ? pivots : at::empty(pivots.sizes(), pivots.options());
  Tensor info_ = info.is_contiguous() ? info : at::empty(info.sizes(), info.options());
  lu_factor_panel_encode(work, pivots_, info_, aRows, aCols, batchSize, inPlace);
  if (!inPlace) {
    LU.copy_(work_full);
  }
  if (!pivots_.is_same(pivots)) {
    pivots.copy_(pivots_);
  }
  if (!info_.is_same(info)) {
    info.copy_(info_);
  }
  if (check_errors) {
    at::_linalg_check_errors(info, "linalg.lu_factor_ex", A.dim() == 2);
  }
}

static void lu_solve_encode(const Tensor& W, const Tensor& pivots, int64_t n, int64_t k, int64_t Bnum, bool adjoint) {
  auto stream = getCurrentMPSStream();
  const auto useMpp = has_mpp();
  const auto un = static_cast<uint32_t>(n);
  const auto uk = static_cast<uint32_t>(k);
  const auto uB = static_cast<uint32_t>(Bnum);
  const auto N = un + uk;

  auto pivotPSO = lib.getPipelineStateForFunc("luApplyPivotsRHS");
  auto fwdPSO = lib.getPipelineStateForFunc(adjoint ? "trsmDiagSolveLU_lower_nonunit" : "trsmDiagSolveLU_lower_unit");
  auto backPSO = lib.getPipelineStateForFunc(adjoint ? "trsmDiagSolveLU_upper_unit" : "trsmDiagSolveLU_upper_nonunit");
  auto gemmBigPSO = useMpp ? lib.getPipelineStateForFunc("gemmLU_64_64_4") : nil;
  auto gemmSmallPSO = useMpp ? lib.getPipelineStateForFunc("gemmLU_32_64_2") : nil;
  auto gemmSimdPSO = useMpp ? nil : lib.getPipelineStateForFunc("gemmSimdLU");

  @autoreleasepool {
    dispatch_sync_with_rethrow(stream->queue(), ^() {
      auto enc = stream->commandEncoder();
      mtl_setArgs(enc, W, pivots);
      mtl_setArgs<3>(enc, std::array<uint32_t, 2>{un, N});

      auto setP4 = [&](uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
        mtl_setArgs<4>(enc, std::array<uint32_t, 4>{a, b, c, d});
      };
      auto setP5 = [&](uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
        mtl_setArgs<5>(enc, std::array<uint32_t, 4>{a, b, c, d});
      };
      auto pivotApply = [&](bool inverse) {
        setP4(un, uk, un, inverse ? 1u : 0u);
        [enc setComputePipelineState:pivotPSO];
        [enc dispatchThreadgroups:MTLSizeMake(uB, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(std::clamp(uk, 1u, 256u), 1, 1)];
      };
      auto trsm = [&](id<MTLComputePipelineState> pso, uint32_t d0, uint32_t nr) {
        setP4(d0, un, N, nr);
        [enc setComputePipelineState:pso];
        [enc dispatchThreadgroups:MTLSizeMake(uB, at::ceil_div(uk, 128u), 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      };
      auto gemm = [&](uint32_t rs, uint32_t re, uint32_t kc, uint32_t kw) {
        if (rs >= re || uk == 0) {
          return;
        }
        const auto Tm = re - rs;
        setP4(rs, re, un, N);
        setP5(kc, kw, 0, 0);
        if (!useMpp) {
          [enc setComputePipelineState:gemmSimdPSO];
          [enc dispatchThreadgroups:MTLSizeMake(at::ceil_div(uk, 64u), at::ceil_div(Tm, 32u), uB)
              threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
          return;
        }
        const auto big = Tm >= 1024 && uk >= 64;
        const auto BM = big ? 64u : 32u;
        const auto BN = 64u;
        const auto NSG = big ? 4u : 2u;
        [enc setComputePipelineState:(big ? gemmBigPSO : gemmSmallPSO)];
        [enc dispatchThreadgroups:MTLSizeMake(at::ceil_div(uk, BN), at::ceil_div(Tm, BM), uB)
            threadsPerThreadgroup:MTLSizeMake(NSG * 32, 1, 1)];
      };

      if (!adjoint) {
        pivotApply(false);
      }
      for (auto p0 = 0u; p0 < un; p0 += 32) {
        const auto pw = std::min(32u, un - p0);
        trsm(fwdPSO, p0, pw);
        gemm(p0 + pw, un, p0, pw);
      }
      for (auto pb = un ? (static_cast<int64_t>(un) - 1) / 32 * 32 : int64_t{0}; pb >= 0; pb -= 32) {
        const auto p0 = static_cast<uint32_t>(pb);
        const auto pw = std::min(32u, un - p0);
        trsm(backPSO, p0, pw);
        gemm(0, p0, p0, pw);
      }
      if (adjoint) {
        pivotApply(true);
      }
    });
  }
}

static void mps_lu_solve_kernel(const Tensor& LU, const Tensor& pivots, const Tensor& B, TransposeType trans) {
  using namespace mps;
  TORCH_CHECK(LU.scalar_type() == kFloat, "linalg.lu_solve(): MPS only supports float32.");
  if (B.numel() == 0) {
    return;
  }
  const auto adjoint = trans != TransposeType::NoTranspose;
  const auto n = LU.size(-1);
  const auto k = B.size(-1);

  std::vector<int64_t> batch(B.sizes().begin(), B.sizes().end() - 2);
  const auto Bnum = c10::multiply_integers(batch);
  auto with_mat = [&](int64_t r, int64_t c) {
    auto s = batch;
    s.push_back(r);
    s.push_back(c);
    return s;
  };
  auto piv_shape = batch;
  piv_shape.push_back(n);
  auto piv_b = pivots.expand(piv_shape).contiguous().reshape({Bnum, n});

  auto W = at::empty({Bnum, n, n + k}, LU.options());
  auto factor = adjoint ? LU.expand(with_mat(n, n)).mH() : LU.expand(with_mat(n, n));
  W.narrow(-1, 0, n).copy_(factor.reshape({Bnum, n, n}));
  W.narrow(-1, n, k).copy_(B.reshape({Bnum, n, k}));

  lu_solve_encode(W, piv_b, n, k, Bnum, adjoint);

  B.copy_(W.narrow(-1, n, k).reshape(with_mat(n, k)));
}

static void linalg_solve_out_mps_impl(const Tensor& A,
                                      const Tensor& B,
                                      bool left,
                                      bool check_errors,
                                      const Tensor& result,
                                      const Tensor& LU,
                                      const Tensor& pivots,
                                      const Tensor& info) {
  using namespace mps;
  linalg_lu_factor_ex_out_mps_impl(A, true, LU, pivots, info, false);
  if (check_errors) {
    at::_linalg_check_errors(info, "torch.linalg.solve_ex", A.dim() == 2);
  }
  const auto vector_case = at::native::linalg_solve_is_vector_rhs(LU, B);
  auto result_ = vector_case ? result.unsqueeze(-1) : result;
  const auto B_ = vector_case ? B.unsqueeze(-1) : B;
  at::linalg_lu_solve_out(result_, LU, pivots, B_, left, false);
}

static void linalg_inv_ex_out_mps_impl(const Tensor& A, bool check_errors, const Tensor& result, const Tensor& info) {
  using namespace mps;
  TORCH_CHECK(result.is_mps(), "Output tensor is not MPS");
  TORCH_CHECK(!A.is_complex(), "linalg_inv: not supported for complex types yet!");

  info.zero_();
  if (A.numel() == 0) {
    return;
  }

  auto A_sizes = A.sizes();
  int ndim = A.dim();

  Tensor LU = empty_like(A, MemoryFormat::Contiguous);
  Tensor identity = eye(A.size(-2), A.size(-1), A.scalar_type(), A.options().layout(), A.device()).expand_as(A);
  Tensor pivots = empty({A_sizes.begin(), A_sizes.end() - 1}, A.options().dtype(kInt));
  // need to do this to keep the strides of the result tensor
  // mps's solve expects row major layout, while inductor
  // expects result to be column major
  Tensor tmp = empty_like(A, MemoryFormat::Contiguous);
  linalg_solve_out_mps_impl(A, identity, true, check_errors, tmp, LU, pivots, info);
  result.copy_(tmp);
}

static Tensor& mm_out_mps_impl(const Tensor& self, const Tensor& other, Tensor& output) {
  using namespace mps;
  static const bool is_macOS_15_0_or_newer = is_macos_at_least(MacOSVersion::MACOS_15_0);

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
    // inputs on macOS < 26.4 (only every 16th row is computed). Contiguify such tensors
    // by disabling the strided API so they go through the gather/clone path first.
    // See https://github.com/pytorch/pytorch/issues/180201
    static const bool is_macOS_26_4_or_newer = is_macos_at_least(MacOSVersion::MACOS_26_4);
    auto hasZeroStride = [](const Tensor& t) {
      return std::ranges::any_of(t.strides(), [](auto s) { return s == 0; });
    };
    auto useStridedSelf = !isComplexType(self.scalar_type()) && (is_macOS_26_4_or_newer || !hasZeroStride(self));
    auto useStridedOther = !isComplexType(other.scalar_type()) && (is_macOS_26_4_or_newer || !hasZeroStride(other));
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
    at::mul_out(result, input, wrapped_scalar_tensor(beta));
    return result;
  }

  // Use Metal kernels for integer and complex types
  if (c10::isIntegralType(batch1.scalar_type(), true) || c10::isComplexType(batch1.scalar_type())) {
    return do_metal_addbmm_or_baddbmm(input, batch1, batch2, alpha, beta, result, opType == BADDBMM_OP_TYPE);
  }

  MPSStream* stream = getCurrentMPSStream();

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
    if (beta.toDouble() == 0.0) {
      output.zero_();
    } else {
      output.copy_(*bias_);
      output.mul_(beta);
    }
    return output;
  }

  if (use_metal_mm(self, other, output)) {
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
  if (is_macos_at_least(MacOSVersion::MACOS_15_0)) {
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

        getMPSProfiler().beginProfileKernel(matmul, " tiled_bmm_mps", {batch1, batch2});

        // Descriptors to use for last batch if it exists
        //.matrices is a readonly property so we need a separate descriptor.
        MPSNDArrayDescriptor *aDescLastBatch_, *bDescLastBatch_, *resDescLastBatch_;
        if (lastBatchSize != 0) {
          aDescLastBatch_ =
              [MPSNDArrayDescriptor descriptorWithDataType:dtype shape:@[ @(lastBatchSize), @(aRowsTiled), @(aCols) ]];
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
          resDescLastTile_ =
              [MPSNDArrayDescriptor descriptorWithDataType:dtype shape:@[ @(batchSize), @(lastTileSize), @(resCols) ]];
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
            auto resMatrix =
                [[[MPSNDArray alloc] initWithBuffer:resBuffer
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
  } else {
    TORCH_CHECK(false, "Tiling of batch matmul for larger than 2**32 entries only available from MacOS15 onwards");
  }
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

  static const bool is_macOS_15_0_or_newer = is_macos_at_least(MacOSVersion::MACOS_15_0);
  MPSShape* shape = nil;
  bool doTranspose = false;

  // Handle transposes for the second batch of matrices.
  // In macOS 15 this is detected automatically (for all shapes/ranks)
  // through the strided MPS support.
  if (!is_macOS_15_0_or_newer) {
    if (batch2.is_view() && !batch2.is_contiguous()) {
      if (batch2.numel() == batch2._base().numel()) {
        const IntArrayRef& viewSizes = batch2.sizes();

        // Handle 3D and 4D tensors.
        // For 4D tensors, first it must have been reshaped from 4D to 3D and then transposed.
        int32_t baseTransposeStrideDim = batch2._base().dim() == 4 ? -3 : -2;
        if (batch2._base().stride(0) == batch2.stride(0) &&
            batch2._base().stride(baseTransposeStrideDim) == batch2.stride(-1)) {
          shape = @[ @(viewSizes[0]), @(viewSizes[2]), @(viewSizes[1]) ];
          doTranspose = true;
        }
      }
    }
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
    std::string key = "bmm_out_mps_impl" + getTensorsStringKey({batch1, batch2}, true, /*exclude_shape*/ true) +
        std::to_string(doTranspose);

    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      auto batch1Tensor = mps::mpsGraphUnrankedPlaceHolder(mpsGraph, getMPSDataType(batch1.scalar_type()));
      auto batch2Tensor = mps::mpsGraphUnrankedPlaceHolder(mpsGraph, getMPSDataType(batch2.scalar_type()));

      auto batch1TensorOp = batch1.is_conj() ? [mpsGraph conjugateWithTensor:batch1Tensor name:nil] : batch1Tensor;
      auto batch2TensorOp = batch2.is_conj() ? [mpsGraph conjugateWithTensor:batch2Tensor name:nil] : batch2Tensor;

      if (doTranspose) {
        batch2TensorOp = [mpsGraph transposeTensor:batch2TensorOp dimension:-1 withDimension:-2 name:nil];
      }

      MPSGraphTensor* productTensor = [mpsGraph matrixMultiplicationWithPrimaryTensor:batch1TensorOp
                                                                      secondaryTensor:batch2TensorOp
                                                                                 name:@"MM/(batch1@batch2)"];

      newCachedGraph->batch1Tensor_ = batch1Tensor;
      newCachedGraph->batch2Tensor_ = batch2Tensor;
      newCachedGraph->outputTensor_ = productTensor;
    });
    Placeholder batch1Placeholder = Placeholder(cachedGraph->batch1Tensor_, batch1);
    Placeholder batch2Placeholder = Placeholder(cachedGraph->batch2Tensor_, batch2, shape, !doTranspose);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, result);

    auto feeds = dictionaryFromPlaceholders(batch1Placeholder, batch2Placeholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }

  return result;
}

static Tensor& linalg_solve_triangular_mps_impl(const Tensor& A,
                                                const Tensor& B,
                                                bool upper,
                                                bool transpose,
                                                bool left,
                                                bool unitriangular,
                                                Tensor& out) {
  using namespace mps;

  checkInputsSolver(A, B, left, "linalg.solve_triangular");
  TORCH_CHECK(A.scalar_type() == kFloat && B.scalar_type() == kFloat,
              "linalg.solve.triangular(); Only float is supported!");
  Tensor A_t, B_t;
  std::tie(B_t, A_t) = _linalg_broadcast_batch_dims(B, A, /*don't check errors*/ nullptr);
  at::native::resize_output(out, B_t.sizes());

  if (A.numel() == 0 || B.numel() == 0 || out.numel() == 0) {
    out.zero_();
    return out;
  }

  Tensor A_ = A_t;
  Tensor B_ = B_t;
  if (!A_t.is_contiguous()) {
    A_ = A_t.clone(at::MemoryFormat::Contiguous);
  }
  if (!B_t.is_contiguous()) {
    B_ = B_t.clone(at::MemoryFormat::Contiguous);
  }
  id<MTLBuffer> aBuffer = getMTLBufferStorage(A_);
  id<MTLBuffer> bBuffer = getMTLBufferStorage(B_);
  id<MTLBuffer> outBuffer = getMTLBufferStorage(out);
  MPSStream* mpsStream = getCurrentMPSStream();
  id<MTLDevice> device = MPSDevice::getInstance()->device();

  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      mpsStream->endKernelCoalescing();
      id<MTLCommandBuffer> commandBuffer = mpsStream->commandBuffer();
      uint64_t batchSize = std::accumulate(A.sizes().begin(), A.sizes().end() - 2, 1ULL, std::multiplies<uint64_t>());
      uint64_t aRows = A_.size(-2);
      uint64_t bRows = B_.size(-2);
      uint64_t aCols = A_.size(-1);
      uint64_t bCols = B_.size(-1);
      uint64_t aElemSize = A_.element_size();
      uint64_t bElemSize = B_.element_size();

      MPSMatrixSolveTriangular* filter = [[[MPSMatrixSolveTriangular alloc] initWithDevice:device
                                                                                     right:!left
                                                                                     upper:upper
                                                                                 transpose:transpose
                                                                                      unit:unitriangular
                                                                                     order:left ? bRows : bCols
                                                                    numberOfRightHandSides:left ? bCols : bRows
                                                                                     alpha:1.0f] autorelease];
      // this function call is a no-op if MPS Profiler is not enabled
      getMPSProfiler().beginProfileKernel(filter, " solve_triangular_mps", {A_, B_});

      MPSMatrixDescriptor* sourceMatrixDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:aRows
                                                                                    columns:aCols
                                                                                   matrices:batchSize
                                                                                   rowBytes:aCols * aElemSize
                                                                                matrixBytes:aRows * aCols * aElemSize
                                                                                   dataType:getMPSDataType(A_)];
      MPSMatrixDescriptor* rightHandSideMatrixDesc =
          [MPSMatrixDescriptor matrixDescriptorWithRows:bRows
                                                columns:bCols
                                               matrices:batchSize
                                               rowBytes:bCols * bElemSize
                                            matrixBytes:bRows * bCols * bElemSize
                                               dataType:getMPSDataType(B_)];
      for (const auto i : c10::irange(batchSize)) {
        const uint64_t aBatchOffset = i * aRows * aCols;
        const uint64_t bBatchOffset = i * bRows * bCols;
        MPSMatrix* sourceMatrix = [[[MPSMatrix alloc] initWithBuffer:aBuffer
                                                              offset:(A_.storage_offset() + aBatchOffset) * aElemSize
                                                          descriptor:sourceMatrixDesc] autorelease];
        MPSMatrix* rightHandSideMatrix =
            [[[MPSMatrix alloc] initWithBuffer:bBuffer
                                        offset:(B_.storage_offset() + bBatchOffset) * bElemSize
                                    descriptor:rightHandSideMatrixDesc] autorelease];
        MPSMatrix* solutionMatrix = [[[MPSMatrix alloc] initWithBuffer:outBuffer
                                                                offset:(out.storage_offset() + bBatchOffset) * bElemSize
                                                            descriptor:rightHandSideMatrixDesc] autorelease];

        [filter encodeToCommandBuffer:commandBuffer
                         sourceMatrix:sourceMatrix
                  rightHandSideMatrix:rightHandSideMatrix
                       solutionMatrix:solutionMatrix];
      }
      getMPSProfiler().endProfileKernel(filter);
    }
  });
  return out;
}

static void unpack_pivots_stub_impl(TensorIterator& iter, const int64_t dim_size, const int64_t max_pivot) {
  if (iter.numel() == 0 || dim_size == 0) {
    return;
  }

  auto perm = iter.tensor(0);
  auto pivots = iter.tensor(1);

  // TODO: Perhaps this should be disabled since it requires a sync?
  TORCH_CHECK_TENSOR_ALL(pivots.le(max_pivot).logical_and(pivots.ge(1)),
                         "pivots passed to lu_unpack must be between 1 and LU.size(-2) inclusive."
                         "Did you properly pass the result of lu_factor?");

  auto num_threads = iter.numel();
  MPSStream* stream = getCurrentMPSStream();

  UnpackPivotsParams params;
  params.perm_batch_stride = safe_downcast<uint32_t, int64_t>((perm.dim() > 1) ? perm.stride(-2) : 0);
  params.pivots_batch_stride = safe_downcast<uint32_t, int64_t>((pivots.dim() > 1) ? pivots.stride(-2) : 0);
  params.dim_size = safe_downcast<uint32_t, int64_t>(dim_size);

  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> compute_encoder = stream->commandEncoder();
      auto pipeline_state = lib.getPipelineStateForFunc(
          fmt::format("unpack_pivots_{}_{}", scalarToMetalTypeString(perm), scalarToMetalTypeString(pivots)));
      getMPSProfiler().beginProfileKernel(pipeline_state, "unpack_pivots", {pivots});
      [compute_encoder setComputePipelineState:pipeline_state];
      mtl_setArgs(compute_encoder, perm, pivots, params);
      mtl_dispatch1DJob(compute_encoder, pipeline_state, num_threads);
      getMPSProfiler().endProfileKernel(pipeline_state);
    }
  });
}

static void cholesky_panel_impl(const Tensor& out, const Tensor& info_, int64_t N, int64_t B, bool upper) {
  auto stream = getCurrentMPSStream();

  constexpr auto NB = 96;
  auto diagPanelPSO = lib.getPipelineStateForFunc(upper ? "factorDiagonalPanelU" : "factorDiagonalPanelL");
  auto trsmPSO = lib.getPipelineStateForFunc(upper ? "applyPanelTRSMU" : "applyPanelTRSML");
  const auto big = N - NB >= 1024;
  const auto BM = big ? 64 : 32;
  constexpr auto BN = 64;
  const auto NSG = big ? 4 : 2;
  auto syrkPSO =
      lib.getPipelineStateForFunc(fmt::format("applySYRKTrailing{}_{}_{}_{}", upper ? "U" : "L", BM, BN, NSG));

  auto numPanels = (N + NB - 1) / NB;

  @autoreleasepool {
    dispatch_sync_with_rethrow(stream->queue(), ^() {
      auto computeEncoder = stream->commandEncoder();
      mtl_setArgs(computeEncoder, out, info_, N, NB);
      for (auto k = 0; k < numPanels; k++) {
        mtl_setArgs<4>(computeEncoder, k);
        [computeEncoder setComputePipelineState:diagPanelPSO];
        [computeEncoder dispatchThreadgroups:MTLSizeMake(B, 1, 1) threadsPerThreadgroup:MTLSizeMake(96, 1, 1)];

        auto T = N - (k + 1) * NB;
        if (T > 0) {
          [computeEncoder setComputePipelineState:trsmPSO];
          [computeEncoder dispatchThreadgroups:MTLSizeMake(B, (T + 31) / 32, 1)
                         threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

          [computeEncoder setComputePipelineState:syrkPSO];
          [computeEncoder dispatchThreadgroups:MTLSizeMake((T + BN - 1) / BN, (T + BM - 1) / BM, B)
                         threadsPerThreadgroup:MTLSizeMake(NSG * 32, 1, 1)];
        }
      }
    });
  }
}

static void cholesky_stub_impl(const Tensor& out, const Tensor& info, bool upper) {
  auto input_sizes = out.sizes();

  int64_t ndim = out.dim();
  int64_t N = out.size(-1);
  int64_t B = c10::multiply_integers(input_sizes.begin(), input_sizes.end() - 2);

  auto stream = getCurrentMPSStream();
  auto device = MPSDevice::getInstance()->device();
  auto info_ = info.dim() >= 2 ? info.view({B}) : info;
  auto info_sizes = info.sizes();
  if (has_mpp()) {
    return cholesky_panel_impl(out, info_, N, B, upper);
  }
  info_.fill_(0);

  auto factorDiagonalPSO = lib.getPipelineStateForFunc(upper ? "factorDiagonalBlockU" : "factorDiagonalBlockL");
  auto applyTRSMPSO = lib.getPipelineStateForFunc(upper ? "applyTRSMU" : "applyTRSML");
  auto applySYRKPSO = lib.getPipelineStateForFunc(upper ? "applySYRKU" : "applySYRKL");

  int64_t NB = std::min<int64_t>(32, N);
  int64_t numBlocks = (N + NB - 1) / NB;

  MTLSize threadGroupSize = MTLSizeMake(32, 8, 1);

  @autoreleasepool {
    dispatch_sync_with_rethrow(stream->queue(), ^() {
      auto computeEncoder = stream->commandEncoder();
      mtl_setArgs(computeEncoder, out, info_, N, NB);
      for (int64_t k = 0; k < numBlocks; k++) {
        [computeEncoder setComputePipelineState:factorDiagonalPSO];
        mtl_setBytes(computeEncoder, k, 4);
        MTLSize gridSize = MTLSizeMake(B, 1, 1);
        [computeEncoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadGroupSize];

        // process all remaining blocks in this row/column in parallel
        if (k < numBlocks - 1) {
          int64_t startJ = k + 1;
          int64_t nBlocksJ = (numBlocks - startJ);

          if (nBlocksJ > 0) {
            // TRSM for all blocks in parallel
            MTLSize trsmGridSize = MTLSizeMake(B, nBlocksJ, 1);
            [computeEncoder setComputePipelineState:applyTRSMPSO];
            [computeEncoder dispatchThreadgroups:trsmGridSize threadsPerThreadgroup:threadGroupSize];

            // SYRK for all independent block pairs in parallel
            uint32_t nPairs = nBlocksJ * (nBlocksJ + 1) / 2;
            MTLSize syrkGridSize = MTLSizeMake(B, nPairs, 1);
            [computeEncoder setComputePipelineState:applySYRKPSO];
            [computeEncoder dispatchThreadgroups:syrkGridSize threadsPerThreadgroup:threadGroupSize];
          }
        }
      }
    });
  }
}

static Tensor& orgqr_stub_impl(Tensor& self, const Tensor& tau) {
  if (self.numel() == 0) {
    return self;
  }

  auto m = self.size(-2);
  auto m2 = m * m;
  auto n = self.size(-1);
  auto k = tau.size(-1);

  if (tau.numel() == 0) {
    auto I = eye(m, self.scalar_type(), std::nullopt, self.device());
    return self.copy_(I.slice(-1, 0, n));
  }

  auto num_batch_dims = self.dim() - 2;
  auto batch_sizes = self.sizes().slice(0, num_batch_dims);
  int64_t num_batches = c10::multiply_integers(batch_sizes);

  std::vector<int64_t> H_sizes(num_batch_dims + 2);
  for (auto dim : c10::irange(num_batch_dims)) {
    H_sizes[dim] = self.size(dim);
  }
  H_sizes[num_batch_dims] = m;
  H_sizes[num_batch_dims + 1] = m;

  auto H = at::empty(H_sizes, self.options().memory_format(MemoryFormat::Contiguous));
  auto H_prod = at::empty_like(H);
  auto H_prod_work = at::empty_like(H);

  OrgqrParams params;

  params.num_batch_dims = num_batch_dims;
  params.m = m;
  params.m2 = m2;
  params.n = n;
  params.k = k;

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
      getMPSProfiler().beginProfileKernel(pipeline_state, "orgqr", {self, tau});
      [compute_encoder setComputePipelineState:pipeline_state];
      mtl_setArgs(compute_encoder, self, tau, H, H_prod, H_prod_work, params);
      static_assert(sizeof(NSUInteger) == sizeof(uint64_t));
      auto max_threadgroup_size = pipeline_state.maxTotalThreadsPerThreadgroup;
      auto threads_per_group = std::min(max_threadgroup_size, NSUInteger(m2));
      NSUInteger num_threads = threads_per_group * num_batches;
      [compute_encoder dispatchThreads:MTLSizeMake(num_threads, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
      getMPSProfiler().endProfileKernel(pipeline_state);
    }
  });

  return self;
}

static Tensor mps_orthonormal_complement(const Tensor& proj, int64_t want) {
  const int64_t dim = proj.size(-1);
  auto bsh = proj.sizes().slice(0, proj.dim() - 2).vec();
  std::vector<int64_t> out_sh = bsh;
  out_sh.push_back(dim);
  out_sh.push_back(want);
  Tensor out = at::zeros(out_sh, proj.options());
  int64_t got = 0;
  for (int64_t c = 0; c < dim && got < want; ++c) {
    Tensor v = proj.narrow(-1, c, 1);
    if (got > 0) {
      Tensor Qc = out.narrow(-1, 0, got);
      Tensor coeff = Qc.mH().matmul(v);
      v = v.sub(Qc.matmul(coeff));
    }
    Tensor nrm = at::linalg_vector_norm(v, 2, {-2}, true);
    Tensor safe = nrm.clamp_min(1e-12);
    Tensor vn = v.div(safe.to(v.dtype()));
    out.narrow(-1, got, 1).copy_(vn);
    got += 1;
  }
  return out;
}

static void svd_kernel_mps(const Tensor& A,
                           const bool full_matrices,
                           const bool compute_uv,
                           const std::optional<std::string_view>& driver,
                           const Tensor& U,
                           const Tensor& S,
                           const Tensor& Vh,
                           const Tensor& info) {
  using namespace mps;
  // Metal has no float64; only float32 / complex64 run on GPU.
  TORCH_CHECK(A.scalar_type() == kFloat || A.scalar_type() == kComplexFloat,
              "linalg.svd: the MPS backend supports only float32 and complex64. Got ",
              A.scalar_type(),
              ". Move the tensor to CPU for other dtypes.");
  TORCH_CHECK(!driver.has_value(), "linalg.svd: driver= is not supported on MPS.");

  if (A.numel() == 0) {
    return;
  }

  const int64_t m = A.size(-2);
  const int64_t n = A.size(-1);
  const int64_t k = std::min(m, n);
  const int64_t batch = c10::multiply_integers(A.sizes().slice(0, A.dim() - 2));

  const int64_t elem_size = A.element_size();
  const int64_t wmax = std::max(m, n);
  const int64_t staging_bytes = wmax * k * elem_size;
  const int64_t tg_limit = static_cast<int64_t>([MPSDevice::getInstance()->device() maxThreadgroupMemoryLength]);
  const int64_t v_staging_bytes = k * k * elem_size;
  const bool stage_v = compute_uv && (staging_bytes + v_staging_bytes <= tg_limit);
  const bool too_large = (staging_bytes > tg_limit);
  const bool too_small = (batch * m * n < 8192);

  if (too_large || too_small) {
    if (too_large) {
      TORCH_WARN_ONCE("linalg.svd: matrix too large to stage in MPS threadgroup memory (",
                      staging_bytes,
                      " > ",
                      tg_limit,
                      " bytes); falling back to CPU.");
    }
    auto [U_cpu, S_cpu, Vh_cpu] = at::linalg_svd(A.to(at::kCPU), full_matrices, driver);
    if (compute_uv) {
      const_cast<Tensor&>(U).copy_(U_cpu.to(at::kMPS));
      const_cast<Tensor&>(Vh).copy_(Vh_cpu.to(at::kMPS));
    }
    const_cast<Tensor&>(S).copy_(S_cpu.to(at::kMPS));
    return;
  }

  const bool transposed = m < n;
  // Kernel needs rows >= cols. For m<n run it on A^H: SVD(A^H)=(V,S,U^H), and
  // params.transposed tells the kernel to swap left/right into the right outputs.
  const int64_t wm = transposed ? n : m;
  Tensor in = (transposed ? A.mH() : A).contiguous().reshape({batch, wm, k});

  auto opts = A.options();
  const bool S_direct = S.is_contiguous() && S.scalar_type() == c10::toRealValueType(A.scalar_type());
  const bool info_direct = info.is_contiguous() && info.scalar_type() == kInt;
  Tensor S_k =
      S_direct ? S.reshape({batch, k}) : at::empty({batch, k}, opts.dtype(c10::toRealValueType(A.scalar_type())));
  // Device-memory accumulator only when V is not staged; tiny placeholder otherwise.
  Tensor Vacc_k = stage_v ? at::empty({1}, opts) : at::empty({batch, k, k}, opts);
  Tensor info_b = info_direct ? info.reshape({batch}) : at::empty({batch}, opts.dtype(kInt));
  // svdvals: U/Vh empty, so bind scratch for the kernel's (still-run) U writeback.
  Tensor U_scratch, Vh_scratch;
  if (!compute_uv) {
    U_scratch = at::empty({batch, wm, wm}, opts);
    Vh_scratch = at::empty({batch, wm, wm}, opts);
  }

  // Kernel writes straight into the column-major U/Vh outputs (first k cols/rows;
  // full_matrices fills the rest below).
  const int64_t u_ld = compute_uv ? U.size(-2) : wm;
  const int64_t u_bs = compute_uv ? U.size(-2) * U.size(-1) : wm * wm;
  const int64_t v_ld = compute_uv ? Vh.size(-2) : wm;
  const int64_t v_bs = compute_uv ? Vh.size(-2) * Vh.size(-1) : wm * wm;

  SvdParams params{static_cast<uint32_t>(wm),
                   static_cast<uint32_t>(k),
                   /*max_sweeps=*/30u,
                   static_cast<uint32_t>(compute_uv ? 1 : 0),
                   /*tol=*/1e-6f,
                   static_cast<uint32_t>(u_ld),
                   static_cast<uint32_t>(u_bs),
                   static_cast<uint32_t>(v_ld),
                   static_cast<uint32_t>(v_bs),
                   static_cast<uint32_t>(transposed ? 1 : 0),
                   static_cast<uint32_t>(stage_v ? 1 : 0)};

  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
      auto pso = lib.getPipelineStateForFunc(fmt::format("svd_jacobi_{}", scalarToMetalTypeString(A)));
      getMPSProfiler().beginProfileKernel(pso, "svd_jacobi", {A});
      [enc setComputePipelineState:pso];
      Tensor Ubind = compute_uv ? U : U_scratch;
      Tensor Vbind = compute_uv ? Vh : Vh_scratch;
      mtl_setArgs(enc, in, Ubind, S_k, Vbind, Vacc_k, info_b, params);
      [enc setThreadgroupMemoryLength:wm * k * elem_size atIndex:0];
      [enc setThreadgroupMemoryLength:(stage_v ? k * k * elem_size : elem_size) atIndex:1];
      // One threadgroup per batch matrix; cap at 32 SIMD-groups (1024 threads).
      const NSUInteger simd = 32;
      const NSUInteger kMaxSimdGroups = 32;
      const NSUInteger maxThreads = pso.maxTotalThreadsPerThreadgroup;
      const NSUInteger nPairs = (k + 1) / 2;
      const NSUInteger wantSG = std::min<NSUInteger>(std::max<NSUInteger>(nPairs, 1), kMaxSimdGroups);
      NSUInteger tgs = std::min<NSUInteger>(maxThreads, wantSG * simd);
      [enc dispatchThreads:MTLSizeMake(tgs * batch, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgs, 1, 1)];
      getMPSProfiler().endProfileKernel(pso);
    }
  });

  if (!S_direct) {
    const_cast<Tensor&>(S).copy_(S_k.reshape(S.sizes()));
  }
  if (!info_direct) {
    const_cast<Tensor&>(info).copy_(info_b.reshape(info.sizes()));
  }

  if (compute_uv && full_matrices && (m != k || n != k)) {
    // full_matrices: complete cols k.. via an orthonormal basis of (I - Q Q^H).
    auto complete = [](Tensor M, int64_t kk) {
      const int64_t dim = M.size(-1);
      if (kk == dim)
        return;
      Tensor Q = M.narrow(-1, 0, kk);
      std::vector<int64_t> ish = M.sizes().slice(0, M.dim() - 2).vec();
      ish.push_back(dim);
      ish.push_back(dim);
      Tensor I = at::eye(dim, M.options()).expand(ish);
      Tensor proj = I.sub(Q.matmul(Q.mH()));
      // linalg_qr is float-only on MPS, so complex goes through Gram-Schmidt.
      Tensor comp;
      if (c10::isComplexType(proj.scalar_type())) {
        comp = mps_orthonormal_complement(proj, dim - kk);
      } else {
        comp = std::get<0>(at::linalg_qr(proj, "reduced")).narrow(-1, 0, dim - kk);
      }
      M.narrow(-1, kk, dim - kk).copy_(comp);
    };
    if (m > k)
      complete(const_cast<Tensor&>(U), k);
    if (n > k)
      complete(const_cast<Tensor&>(Vh).mH(), k); // V = Vh^H
  }
}

static void eigh_kernel_mps(const Tensor& eigenvalues,
                            const Tensor& eigenvectors,
                            const Tensor& infos,
                            bool upper,
                            bool compute_eigenvectors) {
  using namespace mps;

  if (eigenvectors.numel() == 0) {
    return;
  }

  const auto dtype = eigenvectors.scalar_type();
  const int64_t n = eigenvectors.size(-1);
  const int64_t batch = c10::multiply_integers(eigenvectors.sizes().slice(0, eigenvectors.dim() - 2));
  const int64_t elem_size = eigenvectors.element_size();
  const int64_t staging_bytes = n * n * elem_size;
  const int64_t tg_limit = static_cast<int64_t>([MPSDevice::getInstance()->device() maxThreadgroupMemoryLength]);
  const bool fits = (2 * staging_bytes) <= tg_limit;
  const bool unsupported_dtype = (dtype != kFloat && dtype != kComplexFloat);
  const bool too_small = (batch * n * n < 12288);

  if (unsupported_dtype || !fits || too_small) {
    if (!fits) {
      TORCH_WARN_ONCE("linalg.eigh: matrix too large to stage in MPS threadgroup memory (",
                      2 * staging_bytes,
                      " > ",
                      tg_limit,
                      " bytes); falling back to CPU.");
    }
    auto [L_cpu, V_cpu] = at::_linalg_eigh(eigenvectors.to(at::kCPU), upper ? "U" : "L", compute_eigenvectors);
    const_cast<Tensor&>(eigenvalues).copy_(L_cpu);
    if (compute_eigenvectors) {
      const_cast<Tensor&>(eigenvectors).copy_(V_cpu);
    }
    const_cast<Tensor&>(infos).zero_();
    return;
  }

  Tensor V = eigenvectors.reshape({batch, n, n});
  const bool W_direct = eigenvalues.is_contiguous() && eigenvalues.scalar_type() == c10::toRealValueType(dtype);
  const bool info_direct = infos.is_contiguous() && infos.scalar_type() == kInt;
  Tensor W = W_direct ? eigenvalues.reshape({batch, n})
                      : at::empty({batch, n}, eigenvectors.options().dtype(c10::toRealValueType(dtype)));
  Tensor info_b = info_direct ? infos.reshape({batch}) : at::empty({batch}, eigenvectors.options().dtype(kInt));

  EighParams params{static_cast<uint32_t>(n),
                    /*max_sweeps=*/80u,
                    static_cast<uint32_t>(compute_eigenvectors ? 1 : 0),
                    static_cast<uint32_t>(upper ? 1 : 0),
                    /*tol=*/1e-6f};

  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
      auto pso = lib.getPipelineStateForFunc(fmt::format("eigh_jacobi_{}", scalarToMetalTypeString(eigenvectors)));
      getMPSProfiler().beginProfileKernel(pso, "eigh_jacobi", {eigenvectors});
      [enc setComputePipelineState:pso];
      // V binds to both A (in) and Q (out): safe since all reads stage into
      // threadgroup memory before any device writeback.
      mtl_setArgs(enc, V, W, V, info_b, params);
      [enc setThreadgroupMemoryLength:n * n * elem_size atIndex:0];
      [enc setThreadgroupMemoryLength:(compute_eigenvectors ? n * n * elem_size : elem_size) atIndex:1];
      const NSUInteger simd = 32;
      const NSUInteger kMaxSimdGroups = 16;
      const NSUInteger maxThreads = pso.maxTotalThreadsPerThreadgroup;
      const NSUInteger nPairs = (n + 1) / 2;
      const NSUInteger wantSG = std::min<NSUInteger>(std::max<NSUInteger>(nPairs, 1), kMaxSimdGroups);
      NSUInteger tgs = std::min<NSUInteger>(maxThreads, wantSG * simd);
      [enc dispatchThreads:MTLSizeMake(tgs * batch, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgs, 1, 1)];
      getMPSProfiler().endProfileKernel(pso);
    }
  });

  if (!W_direct) {
    const_cast<Tensor&>(eigenvalues).copy_(W.reshape(eigenvalues.sizes()));
  }
  if (!info_direct) {
    const_cast<Tensor&>(infos).copy_(info_b.reshape(infos.sizes()));
  }
}

static Tensor& cholesky_inverse_kernel_impl_mps(Tensor& result, Tensor& infos, bool upper) {
  using namespace mps;
  TORCH_CHECK(result.is_mps(), "Output tensor is not MPS");
  TORCH_CHECK(result.scalar_type() == kFloat, "cholesky_inverse: MPS only supports float type!");

  infos.zero_();
  if (result.numel() == 0) {
    return result;
  }
  auto cholesky =
      upper ? result.triu().clone(at::MemoryFormat::Contiguous) : result.tril().clone(at::MemoryFormat::Contiguous);

  auto n = result.size(-1);
  auto identity = at::eye(n, result.options()).expand_as(result).contiguous();
  auto temp = at::empty(result.sizes(), result.options());
  linalg_solve_triangular_mps_impl(cholesky,
                                   identity,
                                   upper,
                                   /*transpose=*/false,
                                   /*left=*/true,
                                   /*unitriangular=*/false,
                                   temp);
  if (upper) {
    result.copy_(at::matmul(temp, temp.mT()));
  } else {
    result.copy_(at::matmul(temp.mT(), temp));
  }
  return result;
}

static void metal_qr_kernel_impl(const Tensor& A, const Tensor& Q, const Tensor& R, bool reduced_mode) {
  using namespace mps;

  auto m = A.size(-2);
  auto n = A.size(-1);

  int64_t batch_size = 1;
  for (int64_t i = 0; i < A.dim() - 2; i++) {
    batch_size *= A.size(i);
  }

  auto A_work = A.reshape({batch_size, m, n}).contiguous();

  QrParams params;
  params.m = m;
  params.n = n;

  auto info = at::zeros({1}, A.options().dtype(kInt));
  MPSStream* stream = getCurrentMPSStream();

  Tensor Q_work = at::empty({batch_size, m, m}, A.options());
  Tensor R_work = at::empty({batch_size, m, n}, A.options());
  Tensor v_work = at::empty({batch_size, m}, A.options());

  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto compute_encoder = stream->commandEncoder();
      auto pso = lib.getPipelineStateForFunc(fmt::format("linalg_qr_householder_{}", scalarToMetalTypeString(A)));

      getMPSProfiler().beginProfileKernel(pso, "linalg_qr", {A});
      [compute_encoder setComputePipelineState:pso];

      MTLSize threadGroupSize = MTLSizeMake(1024, 1, 1);
      // one threadgroup per matrix in batch
      MTLSize gridSize = MTLSizeMake(batch_size, 1, 1);

      mtl_setArgs(compute_encoder, A_work, Q_work, R_work, info, params, v_work);
      [compute_encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadGroupSize];

      getMPSProfiler().endProfileKernel(pso);
    }
  });

  bool is_batched = A.dim() > 2;

  if (reduced_mode) {
    auto k = std::min(m, n);
    auto Q_reduced = Q_work.narrow(-1, 0, k); // [batch, m, k]
    auto R_reduced = R_work.narrow(-2, 0, k); // [batch, k, n]

    if (is_batched) {
      Q.copy_(Q_reduced.reshape(Q.sizes()));
      R.copy_(R_reduced.reshape(R.sizes()));
    } else {
      Q.copy_(Q_reduced.squeeze(0));
      R.copy_(R_reduced.squeeze(0));
    }
  } else {
    // Q=mxm, R=mxn
    if (is_batched) {
      Q.copy_(Q_work.reshape(Q.sizes()));
      R.copy_(R_work.reshape(R.sizes()));
    } else {
      Q.copy_(Q_work.squeeze(0));
      R.copy_(R_work.squeeze(0));
    }
  }

  if (info.item<int>() != 0) {
    TORCH_CHECK(false, "linalg_qr: MPS kernel failed with error code ", info.item<int>());
  }
}

static void linalg_qr_out_impl_mps(const Tensor& A, const Tensor& Q, const Tensor& R, const c10::string_view mode) {
  using namespace mps;

  TORCH_CHECK(A.scalar_type() == kFloat, "linalg_qr: MPS currently supports float32 only");

  if (A.numel() == 0) {
    return;
  }

  auto m = A.size(-2);
  auto n = A.size(-1);

  if (std::min(m, n) > 512) {
    TORCH_WARN_ONCE(
        "linalg_qr: MPS implementation is currently limited to min(m,n) <= 512, "
        "falling back to CPU.");
    auto A_cpu = A.to(at::kCPU);
    auto [Q_cpu, R_cpu] = at::linalg_qr(A_cpu, mode);
    const_cast<Tensor&>(Q).copy_(Q_cpu.to(at::kMPS));
    const_cast<Tensor&>(R).copy_(R_cpu.to(at::kMPS));
    return;
  }

  bool reduced_mode = (mode != "complete");

  metal_qr_kernel_impl(A, Q, R, reduced_mode);
}

static void lstsq_kernel_mps(const Tensor& a,
                             Tensor& b,
                             Tensor& rank,
                             Tensor& singular_values,
                             Tensor& infos,
                             double rcond,
                             std::string driver_name) {
  const auto scalar_type = a.scalar_type();
  const auto real_dtype = c10::toRealValueType(scalar_type);
  const auto m = a.size(-2);
  const auto n = a.size(-1);

  const bool sets_rank = (driver_name != "gels");
  const bool sets_singular_values = (driver_name == "gelsd" || driver_name == "gelss");

  const double rcond_value =
      rcond > 0 ? rcond : _get_epsilon(real_dtype) * static_cast<double>(std::max<int64_t>(m, n));

  // RHS occupies the first m rows of the (.., max(m, n), nrhs) buffer.
  Tensor rhs = b.narrow(-2, 0, m);

  Tensor U, S, Vh;
  std::tie(U, S, Vh) = at::linalg_svd(a, /*full_matrices=*/false);

  Tensor s_max = std::get<0>(S.max(/*dim=*/-1, /*keepdim=*/true));
  Tensor above = S.gt(s_max.mul(rcond_value));
  Tensor s_inv = at::where(above, S.reciprocal(), at::zeros({}, S.options()));

  Tensor tmp = at::matmul(U.mH(), rhs).mul(s_inv.unsqueeze(-1));
  Tensor solution = at::matmul(Vh.mH(), tmp); // (.., n, nrhs)

  if (m > n) {
    Tensor rss = at::matmul(a, solution).sub(rhs).abs().square().sum(/*dim=*/-2, /*keepdim=*/true);
    Tensor tail = b.narrow(-2, n, m - n);
    tail.zero_();
    tail.narrow(-2, 0, 1).copy_(rss.sqrt());
  }
  b.narrow(-2, 0, n).copy_(solution);

  if (sets_rank) {
    rank.copy_(above.sum(/*dim=*/-1).to(at::kLong));
  }
  if (sets_singular_values) {
    singular_values.copy_(S.to(real_dtype));
  }
  infos.zero_();
}

} // namespace mps

Tensor addr_mps(const Tensor& self, const Tensor& vec1, const Tensor& vec2, const Scalar& beta, const Scalar& alpha) {
  Tensor result = at::empty({0}, self.options());
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
    std::string key = "addr_out_mps_impl" + getTensorsStringKey({vec1, vec2, *self_}) + ":" +
        std::to_string(beta.toDouble()) + ":" + std::to_string(alpha.toDouble());
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* t1 = mps::mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(vec1), inputShape);
      MPSGraphTensor* t2 = mps::mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(vec2), otherShape);
      MPSGraphTensor* selfTensor = mps::mpsGraphRankedPlaceHolder(mpsGraph, *self_);

      // Intermediate as placeholder
      MPSGraphTensor* productTensor = [mpsGraph matrixMultiplicationWithPrimaryTensor:t1
                                                                      secondaryTensor:t2
                                                                                 name:@"MM/(vec1Xvec2)"];

      // Intermediates for beta and alpha
      MPSGraphTensor* betaTensor = [mpsGraph constantWithScalar:beta.toDouble()
                                                       dataType:getMPSScalarType((*self_).scalar_type())];
      MPSGraphTensor* alphaTensor = [mpsGraph constantWithScalar:alpha.toDouble()
                                                        dataType:getMPSScalarType(vec1.scalar_type())];

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

      newCachedGraph->vec1Tensor_ = t1;
      newCachedGraph->vec2Tensor_ = t2;
      newCachedGraph->selfTensor_ = selfTensor;
      newCachedGraph->resultTensor_ = resultTensor;
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

Tensor& linalg_solve_triangular_mps_out(const Tensor& A,
                                        const Tensor& B,
                                        bool upper,
                                        bool left,
                                        bool unitriangular,
                                        Tensor& out) {
  return mps::linalg_solve_triangular_mps_impl(A, B, upper, /*transpose=*/false, left, unitriangular, out);
}

Tensor linalg_solve_triangular_mps(const Tensor& A, const Tensor& B, bool upper, bool left, bool unitriangular) {
  Tensor out = at::empty({0}, A.scalar_type(), std::nullopt, kMPS, std::nullopt, MemoryFormat::Contiguous);
  mps::linalg_solve_triangular_mps_impl(A, B, upper, /*transpose=*/false, left, unitriangular, out);
  return out;
}

Tensor _cholesky_solve_helper_mps(const Tensor& self, const Tensor& A, bool upper) {
  auto out = at::empty({0}, self.options().memory_format(MemoryFormat::Contiguous));
  const bool first_transpose = upper;
  const bool second_transpose = !upper;

  mps::linalg_solve_triangular_mps_impl(A,
                                        self,
                                        upper,
                                        first_transpose,
                                        /*left=*/true,
                                        /*unitriangular=*/false,
                                        out);
  mps::linalg_solve_triangular_mps_impl(A,
                                        out,
                                        upper,
                                        second_transpose,
                                        /*left=*/true,
                                        /*unitriangular=*/false,
                                        out);
  return out;
}

TORCH_IMPL_FUNC(triangular_solve_mps_out)
(const Tensor& self,
 const Tensor& A,
 bool upper,
 bool transpose,
 bool unitriangular,
 const Tensor& result,
 const Tensor& clone_A) {
  clone_A.copy_(A);
  Tensor out = at::empty({0}, A.scalar_type(), std::nullopt, kMPS, std::nullopt, MemoryFormat::Contiguous);
  mps::linalg_solve_triangular_mps_impl(A, self, upper, transpose, /*left=*/true, unitriangular, out);
  result.resize_(out.sizes());
  result.copy_(out);
}

TORCH_IMPL_FUNC(_linalg_solve_ex_out_mps)
(const Tensor& A,
 const Tensor& B,
 bool left,
 bool check_errors,
 const Tensor& result,
 const Tensor& LU,
 const Tensor& pivots,
 const Tensor& info) {
  mps::linalg_solve_out_mps_impl(A, B, left, check_errors, result, LU, pivots, info);
}

REGISTER_MPS_DISPATCH(lu_solve_stub, &mps::mps_lu_solve_kernel)

TORCH_IMPL_FUNC(linalg_lu_factor_ex_out_mps)
(const Tensor& A, bool pivot, bool check_errors, const Tensor& LU, const Tensor& pivots, const Tensor& info) {
  mps::linalg_lu_factor_ex_out_mps_impl(A, pivot, LU, pivots, info, check_errors);
}

TORCH_IMPL_FUNC(linalg_inv_ex_out_mps)(const Tensor& A, bool check_errors, const Tensor& result, const Tensor& info) {
  mps::linalg_inv_ex_out_mps_impl(A, check_errors, result, info);
}

TORCH_IMPL_FUNC(linalg_qr_out_mps)(const Tensor& A, c10::string_view mode, const Tensor& Q, const Tensor& R) {
  mps::linalg_qr_out_impl_mps(A, Q, R, mode);
}

REGISTER_DISPATCH(cholesky_stub, mps::cholesky_stub_impl)
REGISTER_DISPATCH(unpack_pivots_stub, mps::unpack_pivots_stub_impl)
REGISTER_DISPATCH(orgqr_stub, mps::orgqr_stub_impl);
REGISTER_DISPATCH(cholesky_inverse_stub, mps::cholesky_inverse_kernel_impl_mps);
REGISTER_DISPATCH(svd_stub, mps::svd_kernel_mps);
REGISTER_DISPATCH(linalg_eigh_stub, mps::eigh_kernel_mps);
REGISTER_DISPATCH(lstsq_stub, mps::lstsq_kernel_mps);

} // namespace at::native
