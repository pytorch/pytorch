//  Copyright © 2022 Apple Inc.

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/Resize.h>
// For MTLLanguageVersion_3_1
#include <ATen/native/mps/MPSGraphSequoiaOps.h>
#include <ATen/native/mps/MPSGraphSonomaOps.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/addbmm_native.h>
#include <ATen/ops/addmm_native.h>
#include <ATen/ops/addr_native.h>
#include <ATen/ops/baddbmm_native.h>
#include <ATen/ops/bmm_native.h>
#include <ATen/ops/linalg_lu_factor_native.h>
#include <ATen/ops/linalg_solve_triangular_native.h>
#include <ATen/ops/mm_native.h>
#include <ATen/ops/stack.h>
#include <ATen/ops/triangular_solve_native.h>
#endif

#include <algorithm>

namespace at::native {
namespace mps {
namespace {
#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/LinearAlgebra_metallib.h>
#endif

Tensor& do_metal_mm(const Tensor& self, const Tensor& other, Tensor& output) {
  auto stream = getCurrentMPSStream();
  auto device = MPSDevice::getInstance()->device();
  auto matmulPSO = lib.getPipelineStateForFunc("naive_matmul_" + mps::scalarToMetalTypeString(output));
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      getMPSProfiler().beginProfileKernel(matmulPSO, "naive_matmul", {self, other});
      auto computeEncoder = stream->commandEncoder();
      [computeEncoder setComputePipelineState:matmulPSO];
      std::array<uint32_t, 3> sizes = {static_cast<uint32_t>(self.size(0)),
                                       static_cast<uint32_t>(self.size(1)),
                                       static_cast<uint32_t>(output.size(1))};
      std::array<int64_t, 6> strides = {
          self.stride(0), self.stride(1), other.stride(0), other.stride(1), output.stride(0), output.stride(1)};
      mtl_setArgs(computeEncoder, self, other, output, strides, sizes);
      mtl_dispatch1DJob(computeEncoder, matmulPSO, output.numel());
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
  auto selfTensor = mpsGraphRankedPlaceHolder(graph, self);
  auto otherTensor = mpsGraphRankedPlaceHolder(graph, other);
  auto output = [graph matrixMultiplicationWithPrimaryTensor:selfTensor secondaryTensor:otherTensor name:nil];
  return {selfTensor, otherTensor, output};
}

bool use_metal_mm(const Tensor& self, const Tensor& other, const Tensor& output) {
  static bool always_use_metal = std::getenv("PYTORCH_MPS_PREFER_METAL") != nullptr;
  constexpr auto max_stride_size = 32768;
  static bool is_macos_14_4_or_newer = is_macos_13_or_newer(MacOSVersion::MACOS_VER_14_4_PLUS);
  return always_use_metal ||
      (!is_macos_14_4_or_newer &&
       (self.stride(0) > max_stride_size || self.stride(1) > max_stride_size || self.size(0) > max_stride_size ||
        self.size(1) > max_stride_size || other.stride(0) > max_stride_size || other.stride(1) > max_stride_size ||
        other.size(0) > max_stride_size || other.size(1) > max_stride_size));
}

} // anonymous namespace

static void linalg_lu_factor_out_mps_impl(const Tensor& A, bool pivot, Tensor& LU, Tensor& pivots) {
  using namespace mps;

  TORCH_CHECK(!c10::isComplexType(A.scalar_type()) && !c10::isComplexType(LU.scalar_type()),
              "linalg.lu_factor(): MPS doesn't support complex types.");
  TORCH_CHECK(pivot, "linalg.lu_factor(): MPS doesn't allow pivot == False.");

  Tensor A_t = A;
  uint64_t aRows = A_t.size(-2);
  uint64_t aCols = A_t.size(-1);
  uint64_t aElemSize = A_t.element_size();
  uint64_t numPivots = std::min(aRows, aCols);
  std::vector<int64_t> pivot_sizes(A_t.sizes().begin(), A_t.sizes().end() - 2);
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
      MPSMatrixDecompositionLU* filter = [[[MPSMatrixDecompositionLU alloc] initWithDevice:device
                                                                                      rows:aRows
                                                                                   columns:aCols] autorelease];

      MPSMatrixDescriptor* sourceMatrixDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:aRows
                                                                                    columns:aCols
                                                                                   matrices:batchSize
                                                                                   rowBytes:aCols * aElemSize
                                                                                matrixBytes:aRows * aCols * aElemSize
                                                                                   dataType:getMPSDataType(A_)];
      MPSMatrixDescriptor* pivotsMatrixDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:1
                                                                                    columns:numPivots
                                                                                   matrices:1
                                                                                   rowBytes:numPivots * sizeof(uint32_t)
                                                                                matrixBytes:numPivots * sizeof(uint32_t)
                                                                                   dataType:MPSDataTypeUInt32];

      for (const auto i : c10::irange(batchSize)) {
        const uint64_t aBatchOffset = i * aRows * aCols;
        MPSMatrix* sourceMatrix = [[[MPSMatrix alloc] initWithBuffer:aBuffer
                                                              offset:(A_.storage_offset() + aBatchOffset) * aElemSize
                                                          descriptor:sourceMatrixDesc] autorelease];
        MPSMatrix* pivotIndices = [[[MPSMatrix alloc] initWithBuffer:getMTLBufferStorage(pivots_list[i])
                                                              offset:0
                                                          descriptor:pivotsMatrixDesc] autorelease];
        MPSMatrix* solutionMatrix = [[[MPSMatrix alloc] initWithBuffer:aBuffer
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
  if (A_t.dim() > 3) {
    resize_output(LU, A_t.sizes());
    pivots.copy_(stacked_pivots.view(pivot_sizes));
  } else {
    pivots.copy_(stacked_pivots);
  }
  pivots += 1; // PyTorch's `pivots` is 1-index.

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

static Tensor& mm_out_mps_impl(const Tensor& self, const Tensor& other, Tensor& output) {
  using namespace mps;
  static const bool is_macOS_15_0_or_newer = is_macos_13_or_newer(MacOSVersion::MACOS_VER_15_0_PLUS);

  using CachedGraph = MPSBinaryCachedGraph;
  TORCH_CHECK(self.dim() == 2 && other.dim() == 2, "tensors must be 2-D");
  TORCH_CHECK(supportedFloatingOrComplexType(self), "MPS device does not support mm for non-float inputs");

  TensorArg args[]{{output, "out", 0}, {self, "mat1", 1}, {other, "mat2", 2}};
  checkAllSameGPU("mm", args);

  TORCH_CHECK(output.is_mps());

  // Transpose inputs if needed
  if (output.numel() == 0) {
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
    string key = "mm_out_mps_impl" + getTensorsStringKey({self, other});

    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      std::tie(newCachedGraph->inputTensor_, newCachedGraph->otherTensor_, newCachedGraph->outputTensor_) =
          do_mm(mpsGraph, self, other);
    });
    // MPS TODO:
    // Strided API doesn't play nice with complex data types (at least not in case of matmul).
    auto selfPlaceholder = self.numel() != 0
        ? Placeholder(
              cachedGraph->inputTensor_, self, nil, true, MPSDataTypeInvalid, !isComplexType(self.scalar_type()))
        : Placeholder();
    auto otherPlaceholder = other.numel() != 0
        ? Placeholder(
              cachedGraph->otherTensor_, other, nil, true, MPSDataTypeInvalid, !isComplexType(other.scalar_type()))
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

  TORCH_CHECK(supportedFloatingOrComplexType(batch1),
              "MPS device does not support addbmm or baddbmm for non-float inputs");

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

  MPSStream* stream = getCurrentMPSStream();

  struct CachedGraph : public mps::MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* batch1Tensor_ = nil;
    MPSGraphTensor* batch2Tensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
  };

  @autoreleasepool {
    string key = (opType == ADDBMM_OP_TYPE) ? ("addbmm_out_mps_impl") : ("baddbmm_out_mps_impl");
    key += getTensorsStringKey({batch1, batch2, input}) + ":" + std::to_string(beta.toDouble()) + ":" +
        std::to_string(alpha.toDouble());

    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mps::mpsGraphRankedPlaceHolder(mpsGraph, input);
      MPSGraphTensor* batch1Tensor = mps::mpsGraphRankedPlaceHolder(mpsGraph, batch1);
      MPSGraphTensor* batch2Tensor = mps::mpsGraphRankedPlaceHolder(mpsGraph, batch2);

      // Intermediates for beta and alpha
      MPSGraphTensor* betaTensor = [mpsGraph constantWithScalar:beta.toDouble()
                                                       dataType:getMPSScalarType(input.scalar_type())];
      MPSGraphTensor* alphaTensor = [mpsGraph constantWithScalar:alpha.toDouble()
                                                        dataType:getMPSScalarType(batch1.scalar_type())];

      MPSGraphTensor* productTensor = [mpsGraph matrixMultiplicationWithPrimaryTensor:batch1Tensor
                                                                      secondaryTensor:batch2Tensor
                                                                                 name:@"(batch1@batch2)"];

      MPSGraphTensor* reductionSumTensor = productTensor;
      if (opType == ADDBMM_OP_TYPE) {
        reductionSumTensor = [mpsGraph reductionSumWithTensor:productTensor axis:0 name:@"reductionSum(batch1@batch2)"];
      }

      // Intermediates for multiplying by beta and alpha
      MPSGraphTensor* reductionSumTimesAlphaTensor =
          [mpsGraph multiplicationWithPrimaryTensor:reductionSumTensor
                                    secondaryTensor:alphaTensor
                                               name:@"alpha*(batch1@batch2)"];
      MPSGraphTensor* biasTimesBetaTensor = [mpsGraph multiplicationWithPrimaryTensor:inputTensor
                                                                      secondaryTensor:betaTensor
                                                                                 name:@"beta*input"];

      MPSGraphTensor* outputTensor = [mpsGraph additionWithPrimaryTensor:reductionSumTimesAlphaTensor
                                                         secondaryTensor:biasTimesBetaTensor
                                                                    name:@"beta*input + alpha*(batch1@batch2)"];

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
  TORCH_CHECK(supportedFloatingOrComplexType(self), "MPS device does not support addmm for non-float input");

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

  bool is_beta_non_zero = beta.toDouble() != 0.0;

  struct CachedGraph : public mps::MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* selfTensor_ = nil;
    MPSGraphTensor* otherTensor_ = nil;
    MPSGraphTensor* biasTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
  };

  @autoreleasepool {
    string key = "addmm_out_mps_impl" + getTensorsStringKey({self, other, *bias_}) + ":" +
        std::to_string(beta.toDouble()) + ":" + std::to_string(alpha.toDouble());
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* biasTensor = mpsGraphRankedPlaceHolder(mpsGraph, *bias_);

      // TODO: Use alpha and beta here with fill_.Scalar and mul
      auto [selfTensor, otherTensor, productTensor] = do_mm(mpsGraph, self, other);

      auto productTimesAlphaTensor = productTensor;
      if (alpha.toDouble() != 1.0) {
        auto alphaTensor = [mpsGraph constantWithScalar:alpha.toDouble() dataType:getMPSScalarType(self.scalar_type())];

        productTimesAlphaTensor = [mpsGraph multiplicationWithPrimaryTensor:productTensor
                                                            secondaryTensor:alphaTensor
                                                                       name:@"MM/alpha*(mat1@mat2)"];
      }
      auto biasTimesBetaTensor = biasTensor;
      if (is_beta_non_zero && beta.toDouble() != 1.0) {
        auto betaTensor = [mpsGraph constantWithScalar:beta.toDouble()
                                              dataType:getMPSScalarType((*bias_).scalar_type())];
        biasTimesBetaTensor = [mpsGraph multiplicationWithPrimaryTensor:biasTensor
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
  using namespace mps;

  TORCH_CHECK(supportedFloatingOrComplexType(batch1), "MPS device does not support bmm for non-float inputs");

  // Matmul not supported if any output dimension size is larger than 2**32
  for (auto elem : result.sizes()) {
    TORCH_CHECK_NOT_IMPLEMENTED(elem <= pow(2, 32),
                                "Output dim sizes larger than 2**32 elements for matmul not supported on MPS device.");
  }

  if (batch1.numel() == 0 || batch2.numel() == 0) {
    result.zero_();
    return result;
  }

  static const bool is_macOS_15_0_or_newer = is_macos_13_or_newer(MacOSVersion::MACOS_VER_15_0_PLUS);
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
    result = tiled_bmm_out_mps_impl(batch1, batch2, result);
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
    string key = "bmm_out_mps_impl" + getTensorsStringKey({batch1, batch2}, true, /*exclude_shape*/ true) +
        std::to_string(doTranspose);

    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* batch1Tensor = mps::mpsGraphUnrankedPlaceHolder(mpsGraph, getMPSDataType(batch1.scalar_type()));
      MPSGraphTensor* batch2Tensor = mps::mpsGraphUnrankedPlaceHolder(mpsGraph, getMPSDataType(batch2.scalar_type()));
      MPSGraphTensor* batch2TensorTranspose = batch2Tensor;

      if (doTranspose) {
        batch2TensorTranspose = [mpsGraph transposeTensor:batch2Tensor dimension:-1 withDimension:-2 name:nil];
      }

      MPSGraphTensor* productTensor = [mpsGraph matrixMultiplicationWithPrimaryTensor:batch1Tensor
                                                                      secondaryTensor:batch2TensorTranspose
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
  TORCH_CHECK(!A.is_complex() && !B.is_complex(), "linalg.solve.triangular(); Not supported for complex yet!");
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
      uint64_t batchSize = A_.sizes().size() > 2 ? A_.size(0) : 1;
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
                                                              offset:(A_t.storage_offset() + aBatchOffset) * aElemSize
                                                          descriptor:sourceMatrixDesc] autorelease];
        MPSMatrix* rightHandSideMatrix =
            [[[MPSMatrix alloc] initWithBuffer:bBuffer
                                        offset:(B_t.storage_offset() + bBatchOffset) * bElemSize
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
    string key = "addr_out_mps_impl" + getTensorsStringKey({vec1, vec2, *self_}) + ":" +
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

std::tuple<Tensor&, Tensor&> linalg_lu_factor_out_mps(const Tensor& A, bool pivot, Tensor& LU, Tensor& pivots) {
  mps::linalg_lu_factor_out_mps_impl(A, pivot, LU, pivots);
  return std::tie(LU, pivots);
}

std::tuple<Tensor, Tensor> linalg_lu_factor_mps(const Tensor& A, bool pivot) {
  Tensor LU = at::empty({0}, A.options());
  Tensor pivots = at::empty({0}, A.options().dtype(kInt));
  mps::linalg_lu_factor_out_mps_impl(A, pivot, LU, pivots);
  return std::make_tuple(std::move(LU), std::move(pivots));
}

} // namespace at::native
