#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/BatchLinearAlgebra.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/operations/TriangularSolve.h>
#include <fmt/format.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_cholesky_solve_helper_native.h>
#include <ATen/ops/eye.h>
#include <ATen/ops/matmul.h>
#endif

#include <algorithm>

namespace at::native {
namespace mps {
#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/Cholesky_metallib.h>
#endif

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
  TORCH_CHECK(out.scalar_type() == kFloat || out.scalar_type() == kComplexFloat,
              "linalg.cholesky: MPS supports float32 and complex64, but got ",
              out.scalar_type());
  auto input_sizes = out.sizes();

  int64_t ndim = out.dim();
  int64_t N = out.size(-1);
  int64_t B = c10::multiply_integers(input_sizes.begin(), input_sizes.end() - 2);

  auto stream = getCurrentMPSStream();
  auto device = MPSDevice::getInstance()->device();
  auto info_ = info.dim() >= 2 ? info.view({B}) : info;
  auto info_sizes = info.sizes();
  if (has_mpp() && !isComplexType(out.scalar_type())) {
    return cholesky_panel_impl(out, info_, N, B, upper);
  }
  info_.fill_(0);

  const auto dtypeStr = scalarToMetalTypeString(out);
  auto factorDiagonalPSO =
      lib.getPipelineStateForFunc(fmt::format("factorDiagonalBlock{}_{}", upper ? 'U' : 'L', dtypeStr));
  auto applyTRSMPSO = lib.getPipelineStateForFunc(fmt::format("applyTRSM{}_{}", upper ? 'U' : 'L', dtypeStr));
  auto applySYRKPSO = lib.getPipelineStateForFunc(fmt::format("applySYRK{}_{}", upper ? 'U' : 'L', dtypeStr));

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

} // namespace mps

Tensor _cholesky_solve_helper_mps(const Tensor& self, const Tensor& A, bool upper) {
  // A row-major intermediate, which the blocked solve writes without a copy, and
  // a separate out rather than an in-place second solve, which the small-matrix
  // kernel would have to clone.
  auto tmp = at::empty(self.sizes(), self.options());
  auto out = at::empty({0}, self.options());
  const bool first_transpose = upper;
  const bool second_transpose = !upper;

  // A = L L^H (lower) or U^H U (upper); the transposed solve is against the
  // conjugate-transpose factor, so it must be adjoint (no-op for real inputs).
  mps::linalg_solve_triangular_mps_impl(A,
                                        self,
                                        upper,
                                        first_transpose,
                                        /*left=*/true,
                                        /*unitriangular=*/false,
                                        tmp,
                                        /*conjugate=*/first_transpose);
  mps::linalg_solve_triangular_mps_impl(A,
                                        tmp,
                                        upper,
                                        second_transpose,
                                        /*left=*/true,
                                        /*unitriangular=*/false,
                                        out,
                                        /*conjugate=*/second_transpose);
  return out;
}

REGISTER_DISPATCH(cholesky_stub, mps::cholesky_stub_impl)
REGISTER_DISPATCH(cholesky_inverse_stub, mps::cholesky_inverse_kernel_impl_mps);

} // namespace at::native
