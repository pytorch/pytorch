#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/ceil_div.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/BatchLinearAlgebra.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/Pool.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TransposeType.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/LinearAlgebra.h>
#include <fmt/format.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_linalg_check_errors.h>
#include <ATen/ops/_linalg_solve_ex_native.h>
#include <ATen/ops/eye_native.h>
#include <ATen/ops/linalg_inv_ex_native.h>
#include <ATen/ops/linalg_lu_factor_ex_native.h>
#include <ATen/ops/linalg_lu_solve.h>
#endif

#include <algorithm>

namespace at::native {
namespace mps {
#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/LU_metallib.h>
#endif

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
  // complex64 runs the same blocked schedule through the float2 kernel
  // instantiations; the float pipeline behavior is unchanged.
  const bool isComplex = LU.scalar_type() == kComplexFloat;
  const auto tname = mps::scalarToMetalTypeString(LU);
  const bool useMpp = has_mpp() && !isComplex;

  auto factorW32PSO = lib.getPipelineStateForFunc(fmt::format("factorPanelLU_{}_1_32", tname));
  auto factorW16PSO = lib.getPipelineStateForFunc(fmt::format("factorPanelLU_{}_2_16", tname));
  auto factorW8PSO = lib.getPipelineStateForFunc(fmt::format("factorPanelLU_{}_4_8", tname));
  auto streamUpdatePSO = lib.getPipelineStateForFunc(fmt::format("luStreamUpdate_{}", tname));
  auto streamPivotPSO = lib.getPipelineStateForFunc(fmt::format("luStreamPivot_{}", tname));
  auto laswpPSO = lib.getPipelineStateForFunc(fmt::format("laswpGatherLU_{}", tname));
  auto trsm8PSO = lib.getPipelineStateForFunc(fmt::format("trsmPanelLU_{}_8", tname));
  auto trsm16PSO = lib.getPipelineStateForFunc(fmt::format("trsmPanelLU_{}_16", tname));
  auto trsm32PSO = lib.getPipelineStateForFunc(fmt::format("trsmPanelLU_{}_32", tname));
  uint32_t maxG = static_cast<uint32_t>(std::min({factorW32PSO.maxTotalThreadsPerThreadgroup,
                                                  factorW16PSO.maxTotalThreadsPerThreadgroup,
                                                  factorW8PSO.maxTotalThreadsPerThreadgroup}));
  maxG = std::max(32u, maxG / 32 * 32);
  // simdgroup_matrix and MPP matmul2d are float-only hardware paths, so the
  // complex Schur update routes to the threadgroup-tiled scalar kernel.
  auto gemmBigPSO = useMpp ? lib.getPipelineStateForFunc("gemmLU_64_64_4") : nil;
  auto gemmSmallPSO = useMpp ? lib.getPipelineStateForFunc("gemmLU_32_64_2") : nil;
  auto gemmSimdPSO = (!useMpp && !isComplex) ? lib.getPipelineStateForFunc("gemmSimdLU") : nil;
  auto gemmTiledPSO = isComplex ? lib.getPipelineStateForFunc(fmt::format("gemmTiledLU_{}", tname)) : nil;
  auto transposePSO = transposeResult ? lib.getPipelineStateForFunc(fmt::format("transposeInPlaceLU_{}", tname)) : nil;

  const auto uM = static_cast<uint32_t>(M);
  const auto uN = static_cast<uint32_t>(N);
  const auto uB = static_cast<uint32_t>(B);
  const uint32_t mn = std::min(uM, uN);
  const uint32_t NBo = mn <= 1024 ? 32 : mn <= 2048 ? 64 : 128;
  // panels taller than this use the streaming kernels; scratch is one
  // LUStreamScratch<T> slice per batch, bound untyped as a raw byte buffer.
  const uint32_t kStreamMinRows = 4 * maxG;
  Tensor scratch;
  if (uM > kStreamMinRows) {
    const int64_t perBatch = isComplex ? sizeof(LUStreamScratch<c10::complex<float>>) : sizeof(LUStreamScratch<float>);
    scratch = at::empty({B * perBatch}, LU.options().dtype(kByte));
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
        // column-chunk width staged per threadgroup: 64 floats, 32 float2
        // (matches CW in the laswpGatherLU instantiations)
        const uint32_t lcw = isComplex ? 32 : 64;
        setP5(cs0, ce0, cs1, ce1);
        [enc setComputePipelineState:laswpPSO];
        for (uint32_t s0 = 0; s0 < nb; s0 += 32) {
          setP4(d0 + s0, std::min(32u, nb - s0), 0, 0);
          [enc dispatchThreadgroups:MTLSizeMake((W + lcw - 1) / lcw, uB, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
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
        if (isComplex) {
          [enc setComputePipelineState:gemmTiledPSO];
          [enc dispatchThreadgroups:MTLSizeMake((Tn + 15) / 16, (Tm + 15) / 16, uB)
              threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
          return;
        }
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

  TORCH_CHECK((A.scalar_type() == kFloat || A.scalar_type() == kComplexFloat) && LU.scalar_type() == A.scalar_type(),
              "linalg.lu_factor(): MPS supports float32 and complex64 inputs, got ",
              A.scalar_type());
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
  // simdgroup_matrix and MPP matmul2d are float-only hardware paths, so the
  // complex Schur update routes to the threadgroup-tiled scalar kernel.
  const bool isComplex = c10::isComplexType(W.scalar_type());
  const auto tname = mps::scalarToMetalTypeString(W);
  const auto useMpp = has_mpp() && !isComplex;
  const auto un = static_cast<uint32_t>(n);
  const auto uk = static_cast<uint32_t>(k);
  const auto uB = static_cast<uint32_t>(Bnum);
  const auto N = un + uk;

  auto pivotPSO = lib.getPipelineStateForFunc(fmt::format("luApplyPivotsRHS_{}", tname));
  auto fwdPSO = lib.getPipelineStateForFunc(
      fmt::format("trsmDiagSolveLU_{}_{}", tname, adjoint ? "lower_nonunit" : "lower_unit"));
  auto backPSO = lib.getPipelineStateForFunc(
      fmt::format("trsmDiagSolveLU_{}_{}", tname, adjoint ? "upper_unit" : "upper_nonunit"));
  auto gemmBigPSO = useMpp ? lib.getPipelineStateForFunc("gemmLU_64_64_4") : nil;
  auto gemmSmallPSO = useMpp ? lib.getPipelineStateForFunc("gemmLU_32_64_2") : nil;
  auto gemmSimdPSO = (!useMpp && !isComplex) ? lib.getPipelineStateForFunc("gemmSimdLU") : nil;
  auto gemmTiledPSO = isComplex ? lib.getPipelineStateForFunc(fmt::format("gemmTiledLU_{}", tname)) : nil;

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
        if (isComplex) {
          [enc setComputePipelineState:gemmTiledPSO];
          [enc dispatchThreadgroups:MTLSizeMake(at::ceil_div(uk, 16u), at::ceil_div(Tm, 16u), uB)
              threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
          return;
        }
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
  TORCH_CHECK(LU.scalar_type() == kFloat || LU.scalar_type() == kComplexFloat,
              "linalg.lu_solve(): MPS only supports float32 and complex64, got ",
              LU.scalar_type());
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

static void lu_inv_small_encode(const Tensor& A, const Tensor& result, const Tensor& info) {
  const auto n = A.size(-1);
  const auto batch = c10::multiply_integers(A.sizes().begin(), A.sizes().end() - 2);
  const auto A_ = A.reshape({batch, n, n});
  const auto X = result.view({batch, n, n});
  const auto threads = c10::checked_convert<uint32_t>(batch, "uint32_t");
  const LUSmallInvParams<> params{.A_bstride = A_.stride(0),
                                  .A_rstride = A_.stride(1),
                                  .A_cstride = A_.stride(2),
                                  .X_bstride = X.stride(0),
                                  .X_rstride = X.stride(1),
                                  .X_cstride = X.stride(2)};
  auto pso = lib.getPipelineStateForFunc(fmt::format("luInvSmall_{}", n));
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto enc = stream->commandEncoder();
      [enc setComputePipelineState:pso];
      mtl_setArgs(enc, A_, X, info, params);
      // using mtl_dispatch1DJob would launch a single threadgroup of up to 1024 threads
      // all of which would be pinned to one GPU core, which is not ideal for such work
      [enc dispatchThreads:MTLSizeMake(threads, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(c10::metal::simdgroup_size, 1, 1)];
    }
  });
}

static void linalg_inv_ex_out_mps_impl(const Tensor& A, bool check_errors, const Tensor& result, const Tensor& info) {
  using namespace mps;
  TORCH_CHECK(result.is_mps(), "Output tensor is not MPS");
  TORCH_CHECK(!A.is_complex(), "linalg_inv: not supported for complex types yet!");

  if (A.numel() == 0) {
    info.zero_();
    return;
  }
  if (A.size(-1) <= kLUSmallInvMax) {
    lu_inv_small_encode(A, result, info);
    if (check_errors) {
      at::_linalg_check_errors(info, "linalg.inv_ex", A.dim() == 2);
    }
    return;
  }

  auto A_sizes = A.sizes();
  int ndim = A.dim();

  // F-contiguous LU lets lu_factor factor in place
  Tensor LU = empty(A.mT().sizes(), A.options()).mT();
  Tensor identity = eye(A.size(-2), A.size(-1), A.scalar_type(), A.options().layout(), A.device()).expand_as(A);
  Tensor pivots = empty({A_sizes.begin(), A_sizes.end() - 1}, A.options().dtype(kInt));
  linalg_solve_out_mps_impl(A, identity, true, check_errors, result, LU, pivots, info);
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
      getMPSProfiler().beginProfileKernel(pipeline_state, "unpack_pivots", {pivots}, stream);
      [compute_encoder setComputePipelineState:pipeline_state];
      mtl_setArgs(compute_encoder, perm, pivots, params);
      mtl_dispatch1DJob(compute_encoder, pipeline_state, num_threads);
      getMPSProfiler().endProfileKernel(pipeline_state, stream);
    }
  });
}

} // namespace mps

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

REGISTER_DISPATCH(unpack_pivots_stub, mps::unpack_pivots_stub_impl)

} // namespace at::native
