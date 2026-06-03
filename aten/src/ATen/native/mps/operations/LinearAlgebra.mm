//  Copyright © 2022 Apple Inc.

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/BatchLinearAlgebra.h>
#include <ATen/native/LinearAlgebra.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/Pool.h>
#include <ATen/native/Resize.h>
#include <ATen/native/mps/MPSGraphSequoiaOps.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/LinearAlgebra.h>
#include <ATen/native/mps/operations/GemmMetal.h>

#include <fmt/format.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_cholesky_solve_helper_native.h>
#include <ATen/ops/_linalg_solve_ex_native.h>
#include <ATen/ops/addbmm_native.h>
#include <ATen/ops/addmm_native.h>
#include <ATen/ops/addr_native.h>
#include <ATen/ops/baddbmm_native.h>
#include <ATen/ops/bmm_native.h>
#include <ATen/ops/cholesky_native.h>
#include <ATen/ops/eye.h>
#include <ATen/ops/eye_native.h>
#include <ATen/ops/linalg_cholesky_ex_native.h>
#include <ATen/ops/linalg_inv_ex_native.h>
#include <ATen/ops/linalg_lu_factor_ex_native.h>
#include <ATen/ops/linalg_lu_factor_native.h>
#include <ATen/ops/linalg_lu_native.h>
#include <ATen/ops/linalg_qr.h>
#include <ATen/ops/linalg_qr_native.h>
#include <ATen/ops/linalg_solve_triangular_native.h>
#include <ATen/ops/lu_unpack.h>
#include <ATen/ops/lu_unpack_native.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/mm_native.h>
#include <ATen/ops/orgqr_native.h>
#include <ATen/ops/slice.h>
#include <ATen/ops/stack.h>
#include <ATen/ops/triangular_solve_native.h>
#endif

#include <c10/util/env.h>
#include <algorithm>

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

bool use_metal_mm(const Tensor& self, const Tensor& other, const Tensor& output) {
  static bool always_use_metal = c10::utils::has_env("PYTORCH_MPS_PREFER_METAL");
  constexpr auto max_stride_size = 32768;
  constexpr auto max_complex_inner_size = 2048;
  static bool is_macos_14_4_or_newer = is_macos_13_or_newer(MacOSVersion::MACOS_VER_14_4_PLUS);
  if (always_use_metal || c10::isIntegralType(self.scalar_type(), true)) {
    return true;
  }
  // multiplicationWithPrimaryTensor: returns incorrect results if inner size exceeds 2048
  // See https://github.com/pytorch/pytorch/issues/167727#issuecomment-3529308548
  if (c10::isComplexType(self.scalar_type()) && self.size(1) > max_complex_inner_size) {
    return true;
  }
  // (The half "LORADOWN" GEMV padding workaround for #178056 was dropped: it
  // guarded the MPSGraph/MPSNDArray GEMV path, which float mm no longer uses - the
  // hand-written gemv/simd/m5 kernels handle these shapes correctly.)
  return !is_macos_14_4_or_newer &&
      (self.stride(0) > max_stride_size || self.stride(1) > max_stride_size || self.size(0) > max_stride_size ||
       self.size(1) > max_stride_size || other.stride(0) > max_stride_size || other.stride(1) > max_stride_size ||
       other.size(0) > max_stride_size || other.size(1) > max_stride_size);
}

void map_mps_decomposition_error_code_to_blas(const Tensor& status) {
  const auto& status_flat = status.view(-1);

  for (const auto i : c10::irange(status_flat.size(0))) {
    int code = status_flat[i].item<int>();
    switch (code) {
      case MPSMatrixDecompositionStatusSuccess:
        status_flat[i] = 0;
        break;
      case MPSMatrixDecompositionStatusNonPositiveDefinite:
      case MPSMatrixDecompositionStatusSingular:
        status_flat[i] = 2;
        break;
      case MPSMatrixDecompositionStatusFailure:
        status_flat[i] = -1;
        break;
      default:
        TORCH_INTERNAL_ASSERT(false, "Unknown MPSMatrixDecompositionStatus enum value: ", code);
    }
  }
}

} // anonymous namespace

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

  Tensor A_t = A.contiguous();
  uint64_t aRows = A_t.size(-2);
  uint64_t aCols = A_t.size(-1);
  uint64_t aElemSize = A_t.element_size();
  uint64_t numPivots = std::min(aRows, aCols);
  std::vector<int64_t> pivot_sizes(A_t.sizes().begin(), A_t.sizes().end() - 2);
  resize_output(info, pivot_sizes);
  pivot_sizes.push_back(numPivots);
  resize_output(pivots, pivot_sizes);

  if (A_t.numel() == 0) {
    return;
  }

  Tensor A_ = A_t.dim() > 3 ? A_t.flatten(0, -3) : A_t;

  uint64_t batchSize = A_.dim() > 2 ? A_.size(0) : 1;
  std::vector<Tensor> status_tensors;
  std::vector<Tensor> pivots_list;

  status_tensors.reserve(batchSize);
  pivots_list.reserve(batchSize);
  for ([[maybe_unused]] const auto i : c10::irange(batchSize)) {
    status_tensors.push_back(at::zeros(1, kInt, std::nullopt, kMPS, std::nullopt));
    pivots_list.push_back(at::zeros(numPivots, kInt, std::nullopt, kMPS, std::nullopt));
  }

  // Since the MPSMatrixDecompositionLU functions in-place if the result matrix completely aliases the source matrix,
  // We copy LU from A as the new A.
  resize_output(LU, A_.sizes());
  if (!LU.is_same(A_)) {
    A_ = LU.copy_(A_);
  } else {
    A_ = LU;
  }

  TORCH_INTERNAL_ASSERT(A_.is_contiguous())

  id<MTLBuffer> aBuffer = getMTLBufferStorage(A_);

  MPSStream* mpsStream = getCurrentMPSStream();
  id<MTLDevice> device = MPSDevice::getInstance()->device();

  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLCommandBuffer> commandBuffer = mpsStream->commandBuffer();
      auto filter = [[[MPSMatrixDecompositionLU alloc] initWithDevice:device rows:aRows columns:aCols] autorelease];

      auto sourceMatrixDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:aRows
                                                                    columns:aCols
                                                                   matrices:1
                                                                   rowBytes:aCols * aElemSize
                                                                matrixBytes:aRows * aCols * aElemSize
                                                                   dataType:getMPSDataType(A_)];
      auto pivotsMatrixDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:1
                                                                    columns:numPivots
                                                                   matrices:1
                                                                   rowBytes:numPivots * sizeof(uint32_t)
                                                                matrixBytes:numPivots * sizeof(uint32_t)
                                                                   dataType:MPSDataTypeUInt32];

      for (const auto i : c10::irange(batchSize)) {
        const uint64_t aBatchOffset = i * aRows * aCols;
        auto sourceMatrix = [[[MPSMatrix alloc] initWithBuffer:aBuffer
                                                        offset:(A_.storage_offset() + aBatchOffset) * aElemSize
                                                    descriptor:sourceMatrixDesc] autorelease];
        auto pivotIndices = [[[MPSMatrix alloc] initWithBuffer:getMTLBufferStorage(pivots_list[i])
                                                        offset:0
                                                    descriptor:pivotsMatrixDesc] autorelease];
        auto solutionMatrix = [[[MPSMatrix alloc] initWithBuffer:aBuffer
                                                          offset:(A_.storage_offset() + aBatchOffset) * aElemSize
                                                      descriptor:sourceMatrixDesc] autorelease];
        id<MTLBuffer> statusBuffer = getMTLBufferStorage(status_tensors[i]);
        [filter encodeToCommandBuffer:commandBuffer
                         sourceMatrix:sourceMatrix
                         resultMatrix:solutionMatrix
                         pivotIndices:pivotIndices
                               status:statusBuffer];
      }
    }
  });
  auto stacked_pivots = A_.dim() > 2 ? at::stack(pivots_list) : pivots_list[0];
  auto stacked_status = A_.dim() > 2 ? at::stack(status_tensors) : status_tensors[0];
  if (A_t.dim() > 3) {
    resize_output(LU, A_t.sizes());
    pivots.copy_(stacked_pivots.view(pivot_sizes));
  } else {
    pivots.copy_(stacked_pivots);
  }
  pivot_sizes.pop_back();
  info.copy_(stacked_status.view(pivot_sizes));
  pivots.add_(1); // PyTorch's `pivots` is 1-index.
  if (check_errors) {
    for (const auto i : c10::irange(status_tensors.size())) {
      int status = status_tensors[i].item<int>();
      TORCH_CHECK(
          status == 0,
          "lu_factor(): LU factorization failure at the ",
          i + 1,
          " sample with status: ",
          status,
          ". See https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixdecompositionstatus for details.");
    }
  }

  map_mps_decomposition_error_code_to_blas(info);
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

  TORCH_CHECK(A.scalar_type() == kFloat && LU.scalar_type() == kFloat, "linalg.lu_factor(): MPS only supports floats.");
  Tensor A_t, B_t;
  // If 'left' is false, reinterpret the problem so that Ax = B becomes A^T ⋅ (x^T) = B^T
  // Then we solve the normal "left" case on the transposed matrices and transpose x finally to get the output
  if (left) {
    A_t = A.contiguous();
    B_t = B.contiguous();
  } else {
    A_t = A.transpose(-2, -1).contiguous();
    B_t = B.transpose(-2, -1).contiguous();
  }

  uint64_t aRows = A_t.size(-2);
  uint64_t aCols = A_t.size(-1);
  uint64_t aElemSize = A_t.element_size();
  int a_ndim = A_t.dim();
  int b_ndim = B_t.dim();
  int numberOfRightHandSides = (b_ndim == a_ndim - 1) ? 1 : (b_ndim >= 2 ? B_t.size(-1) : 1);

  uint64_t numPivots = std::min(aRows, aCols);
  std::vector<int64_t> pivot_sizes(A_t.sizes().begin(), A_t.sizes().end() - 2);
  resize_output(info, pivot_sizes);
  info.fill_(0); // will be set by kernel on failure
  pivot_sizes.push_back(numPivots);
  resize_output(pivots, pivot_sizes);

  if (A_t.numel() == 0) {
    return;
  }

  // Save original shape before flattening for the LU output
  auto A_original_sizes = A_t.sizes().vec();

  if (A_t.dim() > 3) {
    A_t = A_t.flatten(0, -3);
  }

  uint64_t batchSize = (A_t.dim() > 2) ? A_t.size(0) : 1;
  std::vector<Tensor> status_tensors;
  status_tensors.reserve(batchSize);
  for ([[maybe_unused]] const auto i : c10::irange(batchSize)) {
    status_tensors.push_back(at::zeros(1, kInt, std::nullopt, kMPS, std::nullopt));
  }

  // LU must keep the original (unflattened) shape for the backward pass
  resize_output(LU, A_original_sizes);
  Tensor LU_ = (LU.dim() > 3) ? LU.flatten(0, -3) : LU;
  if (!LU_.is_same(A_t)) {
    A_t = LU_.copy_(A_t);
  } else {
    A_t = LU_;
  }

  TORCH_INTERNAL_ASSERT(A_t.is_contiguous());

  Tensor result_t;
  if (!left) {
    // For right solve, we'll need to transpose the result back later
    result_t = at::empty_like(B_t, B_t.options());
  } else {
    result_t = result;
  }
  id<MTLBuffer> luBuffer = getMTLBufferStorage(LU_);
  id<MTLBuffer> bBuffer = getMTLBufferStorage(B_t);
  id<MTLBuffer> resultBuffer = getMTLBufferStorage(result_t);
  id<MTLBuffer> pivotsBuffer = getMTLBufferStorage(pivots);

  MPSStream* mpsStream = getCurrentMPSStream();
  id<MTLDevice> device = MPSDevice::getInstance()->device();

  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLCommandBuffer> commandBuffer = mpsStream->commandBuffer();

      auto lu_decomp = [[[MPSMatrixDecompositionLU alloc] initWithDevice:device rows:aRows columns:aCols] autorelease];

      auto solver = [[[MPSMatrixSolveLU alloc] initWithDevice:device
                                                    transpose:false
                                                        order:aRows
                                       numberOfRightHandSides:numberOfRightHandSides] autorelease];

      auto luMatrixDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:aRows
                                                                columns:aCols
                                                               matrices:1
                                                               rowBytes:aCols * aElemSize
                                                            matrixBytes:aRows * aCols * aElemSize
                                                               dataType:getMPSDataType(LU_)];
      auto rhsMatrixDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:aRows
                                                                 columns:numberOfRightHandSides
                                                                matrices:1
                                                                rowBytes:numberOfRightHandSides * aElemSize
                                                             matrixBytes:aRows * numberOfRightHandSides * aElemSize
                                                                dataType:getMPSDataType(B_t)];
      auto resultMatrixDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:aRows
                                                                    columns:numberOfRightHandSides
                                                                   matrices:1
                                                                   rowBytes:numberOfRightHandSides * aElemSize
                                                                matrixBytes:aRows * numberOfRightHandSides * aElemSize
                                                                   dataType:getMPSDataType(result_t)];
      auto pivotsMatrixDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:1
                                                                    columns:numPivots
                                                                   matrices:1
                                                                   rowBytes:numPivots * sizeof(uint32_t)
                                                                matrixBytes:numPivots * sizeof(uint32_t)
                                                                   dataType:MPSDataTypeUInt32];

      for (const auto i : c10::irange(batchSize)) {
        const uint64_t batchOffsetA = i * aRows * aCols;
        const uint64_t batchOffsetB = i * aRows * numberOfRightHandSides;
        MPSMatrix* mpsLU = [[[MPSMatrix alloc] initWithBuffer:luBuffer
                                                       offset:(LU_.storage_offset() + batchOffsetA) * aElemSize
                                                   descriptor:luMatrixDesc] autorelease];

        MPSMatrix* mpsRHS = [[[MPSMatrix alloc] initWithBuffer:bBuffer
                                                        offset:(B_t.storage_offset() + batchOffsetB) * aElemSize
                                                    descriptor:rhsMatrixDesc] autorelease];

        MPSMatrix* mpsResult = [[[MPSMatrix alloc] initWithBuffer:resultBuffer
                                                           offset:(result_t.storage_offset() + batchOffsetB) * aElemSize
                                                       descriptor:resultMatrixDesc] autorelease];

        auto mpsPivots = [[[MPSMatrix alloc] initWithBuffer:pivotsBuffer
                                                     offset:(pivots.storage_offset() + i * numPivots) * sizeof(uint32_t)
                                                 descriptor:pivotsMatrixDesc] autorelease];
        [lu_decomp encodeToCommandBuffer:commandBuffer
                            sourceMatrix:mpsLU
                            resultMatrix:mpsLU
                            pivotIndices:mpsPivots
                                  status:getMTLBufferStorage(status_tensors[i])];
        [solver encodeToCommandBuffer:commandBuffer
                         sourceMatrix:mpsLU
                  rightHandSideMatrix:mpsRHS
                         pivotIndices:mpsPivots
                       solutionMatrix:mpsResult];
      }
    }
  });

  pivots.add_(1); // MPS is 0-based, PyTorch/LAPACK is 1-based

  auto stacked_status = batchSize > 1 ? at::stack(status_tensors) : status_tensors[0];
  info.copy_(stacked_status.view(info.sizes()));

  if (check_errors) {
    for (const auto i : c10::irange(batchSize)) {
      int status = status_tensors[i].item<int>();
      TORCH_CHECK(status == 0,
                  "solve(): Linear solve failed at the ",
                  i + 1,
                  " sample with status: ",
                  status,
                  ". See https://developer.apple.com/documentation/metalperformanceshaders/"
                  "mpsmatrixdecompositionstatus for details.");
    }
  }

  map_mps_decomposition_error_code_to_blas(info);

  if (!left) {
    // If this was a right solve, transpose the result back
    result.copy_(result_t.transpose(-2, -1).contiguous());
  }
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
  // Outer or inner dimension is 0
  if (output.numel() == 0 || self.size(1) == 0) {
    return output.zero_();
  }

  // Complex decomposes into four real GEMMs. Integer + float/half/bfloat run the
  // hand-written GEMM kernels (int_gemm / simd / m5_tensor). bool and a few legacy
  // float edge cases caught by use_metal_mm use the naive-metal kernel. The
  // use_metal_mm float workarounds don't apply to integers.
  if (self.is_complex()) {
    mps_gemm_complex(self, other, output, std::nullopt, /*alpha=*/1, /*beta=*/0, at_gemm::GemmEpilogue::None);
    return output;
  }
  const bool is_int = c10::isIntegralType(self.scalar_type(), /*includeBool=*/false);
  if (!gemm_supported_dtype(self.scalar_type()) ||
      (!is_int && use_metal_mm(self, other, output))) {
    return do_metal_mm(self, other, output);
  }
  mps_gemm(self, other, output, std::nullopt, /*alpha=*/1, /*beta=*/0, at_gemm::GemmEpilogue::None);
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

    const int64_t num_batches = batch1.size(0);

    if (num_batches == 0) {
      result.zero_();
      return result;
    }
  }

  // baddbmm applies a per-batch epilogue and maps directly onto the batched GEMM
  // kernel (integer via int_gemm, float/half/bfloat via simd/m5_tensor, complex
  // via the decomposed real GEMMs).
  if (opType == BADDBMM_OP_TYPE && gemm_supported_dtype(batch1.scalar_type())) {
    mps_gemm(batch1, batch2, result, input, alpha, beta, at_gemm::GemmEpilogue::AlphaBeta);
    return result;
  }
  if (opType == BADDBMM_OP_TYPE && c10::isComplexType(batch1.scalar_type())) {
    mps_gemm_complex(batch1, batch2, result, input, alpha, beta, at_gemm::GemmEpilogue::AlphaBeta);
    return result;
  }
  // Remaining: addbmm (reduces over the batch dim; float/half/bfloat go through the
  // MPSGraph-free bmm + reduction below), plus int/complex addbmm and bool baddbmm
  // -> naive Metal (do_metal_*).
  if (c10::isIntegralType(batch1.scalar_type(), true) || c10::isComplexType(batch1.scalar_type())) {
    return do_metal_addbmm_or_baddbmm(input, batch1, batch2, alpha, beta, result, opType == BADDBMM_OP_TYPE);
  }

  // addbmm (float/half/bfloat): reduce the per-batch products over the batch dim,
  // then apply the alpha/beta epilogue. Reuses the (MPSGraph-free) bmm path plus
  // elementwise ops; baddbmm and all other dtypes were handled above.
  TORCH_INTERNAL_ASSERT(opType == ADDBMM_OP_TYPE);
  Tensor summed = batch1.bmm(batch2).sum(0);
  summed.mul_(alpha);
  if (beta.toDouble() != 0.0) {
    summed.add_(input, beta);
  }
  result.copy_(summed);
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

  // Complex decomposes into four real GEMMs with a host-side alpha/beta epilogue.
  // Integer + float/half/bfloat run the GEMM kernels with a fused epilogue;
  // integers skip the use_metal_mm float workarounds. bool falls to do_metal.
  if (self.is_complex()) {
    mps_gemm_complex(self, other, output, *bias_, alpha, beta, at_gemm::GemmEpilogue::AlphaBeta);
    return output;
  }
  const bool is_int = c10::isIntegralType(self.scalar_type(), /*includeBool=*/false);
  if (gemm_supported_dtype(self.scalar_type()) && (is_int || !use_metal_mm(self, other, output))) {
    mps_gemm(self, other, output, *bias_, alpha, beta, at_gemm::GemmEpilogue::AlphaBeta);
    return output;
  }
  // Every dtype is handled above (complex -> mps_gemm_complex; int/float ->
  // mps_gemm; bool + use_metal_mm float edge cases -> do_metal_addmm). MPSGraph is
  // no longer reachable for addmm.
  TORCH_INTERNAL_ASSERT(false, "addmm_out_mps_impl: unhandled dtype ", self.scalar_type());
  return output;
}

static Tensor& tiled_bmm_out_mps_impl(const Tensor& batch1, const Tensor& batch2, Tensor& result) {
  if (is_macos_13_or_newer(MacOSVersion::MACOS_VER_15_0_PLUS)) {
    using namespace mps;

    id<MTLBuffer> aBuffer = getMTLBufferStorage(batch1);
    id<MTLBuffer> bBuffer = getMTLBufferStorage(batch2);
    id<MTLBuffer> resBuffer = getMTLBufferStorage(result);

    MPSStream* mpsStream = getCurrentMPSStream();
    id<MTLDevice> device = MPSDevice::getInstance()->device();
    id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();

    dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
      @autoreleasepool {
        mpsStream->endKernelCoalescing();

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

  // bool has no int_gemm instantiation; huge integer results overflow the
  // kernel's int32 indexing -> both fall back to the naive int64-strided kernel.
  if (batch1.scalar_type() == kBool ||
      (c10::isIntegralType(batch1.scalar_type(), false) &&
       static_cast<uint64_t>(batch1.size(0)) * batch1.size(1) * batch2.size(2) > (1ull << 32))) {
    return do_metal_bmm(batch1, batch2, result);
  }

  // Results above 2^32 elements overflow the kernels' int32 indexing: fall back to
  // the tiled MPSNDArray path (resolve conj views first; it reads raw buffers).
  uint64_t resultSize = batch1.size(0) * batch1.size(1) * batch2.size(2);
  if (resultSize > pow(2, 32)) {
    result = tiled_bmm_out_mps_impl(batch1.resolve_conj(), batch2.resolve_conj(), result);
    return result;
  }
  if (batch1.is_complex()) {
    mps_gemm_complex(batch1, batch2, result, std::nullopt, 1, 0, at_gemm::GemmEpilogue::None);
    return result;
  }
  if (gemm_supported_dtype(batch1.scalar_type())) {
    mps_gemm(batch1, batch2, result, std::nullopt, 1, 0, at_gemm::GemmEpilogue::None);
    return result;
  }
  TORCH_INTERNAL_ASSERT(false, "bmm_out_mps_impl: unhandled dtype ", batch1.scalar_type());
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

static void cholesky_stub_impl(const Tensor& out, const Tensor& info, bool upper) {
  auto input_sizes = out.sizes();

  int64_t ndim = out.dim();
  int64_t N = out.size(-1);
  int64_t B = c10::multiply_integers(input_sizes.begin(), input_sizes.end() - 2);

  auto stream = getCurrentMPSStream();
  auto device = MPSDevice::getInstance()->device();

  auto factorDiagonalPSO = lib.getPipelineStateForFunc(upper ? "factorDiagonalBlockU" : "factorDiagonalBlockL");
  auto applyTRSMPSO = lib.getPipelineStateForFunc(upper ? "applyTRSMU" : "applyTRSML");
  auto applySYRKPSO = lib.getPipelineStateForFunc(upper ? "applySYRKU" : "applySYRKL");

  int64_t NB = std::min<int64_t>(32, N);
  int64_t numBlocks = (N + NB - 1) / NB;

  auto info_ = info.dim() >= 2 ? info.view({B}) : info;
  auto info_sizes = info.sizes();
  info_.fill_(0);

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

  // Outer product: result = beta*self + alpha*(vec1 (x) vec2). This is a rank-1
  // (K=1) GEMM (M,1) x (1,N) with the alpha/beta epilogue; self_ broadcasts to
  // (M, N) via the kernel's epilogue strides.
  auto A = vec1.unsqueeze(1);
  auto B = vec2.unsqueeze(0);
  if (vec1.is_complex()) {
    mps_gemm_complex(A, B, result, *self_, alpha, beta, at_gemm::GemmEpilogue::AlphaBeta);
  } else {
    mps_gemm(A, B, result, *self_, alpha, beta, at_gemm::GemmEpilogue::AlphaBeta);
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

} // namespace at::native
