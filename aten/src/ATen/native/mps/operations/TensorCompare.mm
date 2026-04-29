//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/ScalarOps.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorCompare.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/TensorCompare.h>
#include <fmt/format.h>
#include <algorithm>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/isin_native.h>
#include <ATen/ops/nan_to_num_native.h>
#include <ATen/ops/result_type.h>
#include <ATen/ops/sort.h>
#include <ATen/ops/where_native.h>
#endif

namespace at::native {
#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/TensorCompare_metallib.h>
#endif

namespace mps {

static void isin_default_kernel_mps(const Tensor& elements,
                                    const Tensor& test_elements,
                                    bool invert,
                                    const Tensor& out) {
  TORCH_CHECK(elements.is_mps() && test_elements.is_mps(),
              "Expected elements.is_mps() && test_elements.is_mps(), got ",
              elements.device(),
              " and ",
              test_elements.device());

  if (test_elements.numel() == 0) {
    return;
  }

  const auto common_type = at::result_type(elements, test_elements);
  const Tensor elements_contig = elements.to(common_type).contiguous();
  const Tensor test_elements_contig = test_elements.to(common_type).contiguous();
  Tensor output_contig = out.is_contiguous() ? out : at::empty_like(out, at::MemoryFormat::Contiguous);

  const int64_t numel_elements = elements_contig.numel();
  const int64_t numel_test = test_elements_contig.numel();
  TORCH_CHECK(
      numel_elements <= std::numeric_limits<uint32_t>::max() && numel_test <= std::numeric_limits<uint32_t>::max(),
      "isin_mps: tensor too large, numel must fit in uint32_t");

  int64_t num_chunks = std::max<int64_t>(1, (ISIN_TARGET_THREADGROUPS + numel_elements - 1) / numel_elements);
  num_chunks = std::min<int64_t>(num_chunks, std::max<int64_t>(1, numel_test / ISIN_THREADS_PER_THREADGROUP));

  IsinParams params{
      static_cast<uint32_t>(numel_elements), static_cast<uint32_t>(numel_test), static_cast<uint32_t>(num_chunks)};

  Tensor counts = at::zeros({numel_elements}, elements_contig.options().dtype(at::kInt));

  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = stream->commandEncoder();

      const std::string kernel_name = fmt::format("isin_{}", scalarToMetalTypeString(common_type));
      id<MTLComputePipelineState> isinPSO = lib.getPipelineStateForFunc(kernel_name);
      getMPSProfiler().beginProfileKernel(isinPSO, kernel_name, {elements_contig, test_elements_contig, counts});
      [computeEncoder setComputePipelineState:isinPSO];
      mtl_setArgs(computeEncoder, elements_contig, test_elements_contig, counts, params);
      [computeEncoder dispatchThreadgroups:MTLSizeMake(numel_elements * num_chunks, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(ISIN_THREADS_PER_THREADGROUP, 1, 1)];
      getMPSProfiler().endProfileKernel(isinPSO);

      const std::string invert_kernel_name = "isin_apply_invert";
      id<MTLComputePipelineState> invertPSO = lib.getPipelineStateForFunc(invert_kernel_name);
      getMPSProfiler().beginProfileKernel(invertPSO, invert_kernel_name, {counts, output_contig});
      [computeEncoder setComputePipelineState:invertPSO];
      mtl_setArgs(computeEncoder, counts, output_contig, invert);
      mtl_dispatch1DJob(computeEncoder, invertPSO, numel_elements);
      getMPSProfiler().endProfileKernel(invertPSO);
    }
  });

  if (!out.is_contiguous()) {
    out.copy_(output_contig);
  }
}

static void isin_sorting_kernel_mps(const Tensor& elements,
                                    const Tensor& test_elements,
                                    bool invert,
                                    const Tensor& out) {
  TORCH_CHECK(elements.is_mps() && test_elements.is_mps(),
              "Expected elements.is_mps() && test_elements.is_mps(), got ",
              elements.device(),
              " and ",
              test_elements.device());

  if (test_elements.numel() == 0) {
    return;
  }

  const auto common_type = at::result_type(elements, test_elements);
  const Tensor elements_contig = elements.to(common_type).contiguous();
  const Tensor test_elements_contig = test_elements.to(common_type).contiguous();
  Tensor output_contig = out.is_contiguous() ? out : at::empty_like(out, at::MemoryFormat::Contiguous);

  const Tensor sorted_test = std::get<0>(at::sort(test_elements_contig.flatten(), 0, /*descending=*/false));

  const int64_t numel_elements = elements_contig.numel();
  const int64_t numel_test = test_elements_contig.numel();
  TORCH_CHECK(
      numel_elements <= std::numeric_limits<uint32_t>::max() && numel_test <= std::numeric_limits<uint32_t>::max(),
      "isin_mps: tensor too large, numel must fit in uint32_t");

  IsinSortedParams params{static_cast<uint32_t>(numel_test), invert};

  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = stream->commandEncoder();
      const std::string kernel_name = fmt::format("isin_sorted_{}", scalarToMetalTypeString(common_type));
      id<MTLComputePipelineState> pso = lib.getPipelineStateForFunc(kernel_name);
      getMPSProfiler().beginProfileKernel(pso, kernel_name, {elements_contig, sorted_test, output_contig});
      [computeEncoder setComputePipelineState:pso];
      mtl_setArgs(computeEncoder, elements_contig, sorted_test, output_contig, params);
      mtl_dispatch1DJob(computeEncoder, pso, numel_elements);
      getMPSProfiler().endProfileKernel(pso);
    }
  });

  if (!out.is_contiguous()) {
    out.copy_(output_contig);
  }
}

static void is_posneginf_helper(TensorIteratorBase& iter, bool is_neg) {
  if (iter.numel() == 0) {
    return;
  }
  const auto& self = iter.input(0);
  auto& out = iter.output(0);
  @autoreleasepool {
    auto cachedGraph = LookUpOrCreateCachedGraph<MPSUnaryCachedGraph>(
        __func__ + std::to_string(is_neg) + getTensorsStringKey(self), [&](auto mpsGraph, auto newCachedGraph) {
          auto infTensor = [mpsGraph constantWithScalar:is_neg ? -std::numeric_limits<float>::infinity()
                                                               : std::numeric_limits<float>::infinity()
                                               dataType:getMPSScalarType(self)];
          newCachedGraph->inputTensor_ = mpsGraphRankedPlaceHolder(mpsGraph, self);
          newCachedGraph->outputTensor_ = [mpsGraph equalWithPrimaryTensor:newCachedGraph->inputTensor_
                                                           secondaryTensor:infTensor
                                                                      name:nil];
        });
    auto selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    auto outputPlaceholder = Placeholder(cachedGraph->outputTensor_, out);
    runMPSGraph(
        getCurrentMPSStream(), cachedGraph->graph(), dictionaryFromPlaceholders(selfPlaceholder), outputPlaceholder);
  }
}
} // namespace mps

static void where_kernel_mps(TensorIterator& iter) {
  const auto& condition = iter.input(0);
  const auto& self = iter.input(1);
  const auto& other = iter.input(2);
  auto& out = iter.output(0);
  TORCH_CHECK(condition.device() == self.device() && self.device() == other.device(),
              "Expected all tensors to be on the same device, but found at least two devices.");
  TORCH_CHECK(self.dtype() == other.dtype(), "expected scalar type ", self.dtype(), " but found ", other.dtype());

  if (condition.scalar_type() == ScalarType::Byte) {
    TORCH_WARN_ONCE(
        "where received a uint8 condition tensor. This behavior is deprecated and will be removed in a future version of PyTorch. Use a boolean condition instead.");
  } else {
    TORCH_CHECK(condition.scalar_type() == ScalarType::Bool,
                "where expected condition to be a boolean tensor, but got a tensor with dtype ",
                condition.scalar_type());
  }
  Tensor cond_bool = condition.scalar_type() == ScalarType::Byte ? condition.to(ScalarType::Bool) : condition;

  using namespace mps;
  MPSStream* stream = getCurrentMPSStream();

  // Empty output
  if (out.numel() == 0) {
    return;
  }

  Tensor out_;
  if (needsGather(out)) {
    out_ = out.contiguous();
  }

  // Derive from MPSCachedGraph
  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* conditionTensor_ = nil;
    MPSGraphTensor* selfTensor_ = nil;
    MPSGraphTensor* otherTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
  };

  MPSDataType conditionDataType = getMPSScalarType(condition.scalar_type());
  MPSDataType selfDataType = getMPSScalarType(self.scalar_type());
  MPSDataType otherDataType = getMPSScalarType(other.scalar_type());

  @autoreleasepool {
    std::string key = "where_self_out_mps:" + getTensorsStringKey({cond_bool, self, other});

    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* conditionTensor = mpsGraphRankedPlaceHolder(mpsGraph, conditionDataType, getMPSShape(cond_bool));
      MPSGraphTensor* selfTensor = mpsGraphRankedPlaceHolder(mpsGraph, selfDataType, getMPSShape(self));
      MPSGraphTensor* otherTensor = mpsGraphRankedPlaceHolder(mpsGraph, otherDataType, getMPSShape(other));

      MPSGraphTensor* outputTensor = [mpsGraph selectWithPredicateTensor:conditionTensor
                                                     truePredicateTensor:selfTensor
                                                    falsePredicateTensor:otherTensor
                                                                    name:nil];

      newCachedGraph->conditionTensor_ = conditionTensor;
      newCachedGraph->selfTensor_ = selfTensor;
      newCachedGraph->otherTensor_ = otherTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

    Placeholder conditionPlaceholder = Placeholder(
        cachedGraph->conditionTensor_, cond_bool, /*mpsShape=*/nullptr, /*gatherTensorData=*/true, conditionDataType);
    Placeholder selfPlaceholder =
        Placeholder(cachedGraph->selfTensor_, self, /*mpsShape=*/nullptr, /*gatherTensorData=*/true, selfDataType);
    Placeholder otherPlaceholder =
        Placeholder(cachedGraph->otherTensor_, other, /*mpsShape=*/nullptr, /*gatherTensorData=*/true, otherDataType);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_,
                                                needsGather(out) ? out_ : out,
                                                /*mpsShape=*/nullptr,
                                                /*gatherTensorData=*/needsGather(out),
                                                getMPSScalarType(out.scalar_type()));

    auto feeds = dictionaryFromPlaceholders(conditionPlaceholder, selfPlaceholder, otherPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }

  if (needsGather(out)) {
    out.copy_(out_);
  }
}

Tensor& nan_to_num_out_mps(const Tensor& self,
                           std::optional<double> nan,
                           std::optional<double> pos_inf,
                           std::optional<double> neg_inf,
                           Tensor& result) {
  TORCH_CHECK(self.scalar_type() == result.scalar_type(),
              "nan_to_num: dtype of out: ",
              result.scalar_type(),
              " should be same as input: ",
              self.scalar_type());
  if (result.numel() == 0) {
    return result;
  }
  if (c10::isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
    at::native::resize_output(result, self.sizes());
    result.copy_(self);
    return result;
  }
  using namespace mps;
  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* selfTensor = nil;
    MPSGraphTensor* outputTensor = nil;
    MPSGraphTensor* nanReplacementTensor = nil;
    MPSGraphTensor* posInfReplacementTensor = nil;
    MPSGraphTensor* negInfReplacementTensor = nil;
  };

  @autoreleasepool {
    std::string key = "nan_to_num" + getTensorsStringKey({self});
    MPSDataType self_dtype = getMPSScalarType(self.scalar_type());

    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      newCachedGraph->selfTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
      newCachedGraph->nanReplacementTensor = mpsGraphRankedPlaceHolder(mpsGraph, self_dtype, @[ @1 ]);
      newCachedGraph->posInfReplacementTensor = mpsGraphRankedPlaceHolder(mpsGraph, self_dtype, @[ @1 ]);
      newCachedGraph->negInfReplacementTensor = mpsGraphRankedPlaceHolder(mpsGraph, self_dtype, @[ @1 ]);

      MPSGraphTensor* nanFreeTensor =
          [mpsGraph selectWithPredicateTensor:[mpsGraph isNaNWithTensor:newCachedGraph->selfTensor name:nil]
                          truePredicateTensor:newCachedGraph->nanReplacementTensor
                         falsePredicateTensor:newCachedGraph->selfTensor
                                         name:nil];
      MPSGraphTensor* subZeroTensor = [mpsGraph lessThanWithPrimaryTensor:nanFreeTensor
                                                          secondaryTensor:[mpsGraph constantWithScalar:0.0
                                                                                              dataType:self_dtype]
                                                                     name:nil];
      MPSGraphTensor* isInfTensor = [mpsGraph isInfiniteWithTensor:nanFreeTensor name:nil];
      // workaround for Monterey; On Ventura the output of lessThan() is always Boolean
      if (subZeroTensor.dataType != MPSDataTypeBool) {
        subZeroTensor = castMPSTensor(mpsGraph, subZeroTensor, kBool);
      }
      if (isInfTensor.dataType != MPSDataTypeBool) {
        isInfTensor = castMPSTensor(mpsGraph, isInfTensor, kBool);
      }
      MPSGraphTensor* isNegInfTensor = [mpsGraph logicalANDWithPrimaryTensor:subZeroTensor
                                                             secondaryTensor:isInfTensor
                                                                        name:nil];
      MPSGraphTensor* negInfFreeTensor = [mpsGraph selectWithPredicateTensor:isNegInfTensor
                                                         truePredicateTensor:newCachedGraph->negInfReplacementTensor
                                                        falsePredicateTensor:nanFreeTensor
                                                                        name:nil];
      newCachedGraph->outputTensor = [mpsGraph selectWithPredicateTensor:[mpsGraph isInfiniteWithTensor:negInfFreeTensor
                                                                                                   name:nil]
                                                     truePredicateTensor:newCachedGraph->posInfReplacementTensor
                                                    falsePredicateTensor:negInfFreeTensor
                                                                    name:nil];
    });
    MPSScalar nanReplacementScalar, posInfReplacementScalar, negInfReplacementScalar;
    AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, self.scalar_type(), "nan_to_num_mps", [&]() {
      scalar_t nan_replacement = static_cast<scalar_t>(nan.value_or(0.));
      scalar_t pos_inf_replacement =
          pos_inf.has_value() ? static_cast<scalar_t>(pos_inf.value()) : std::numeric_limits<scalar_t>::max();
      scalar_t neg_inf_replacement =
          neg_inf.has_value() ? static_cast<scalar_t>(neg_inf.value()) : std::numeric_limits<scalar_t>::lowest();

      nanReplacementScalar = getMPSScalar(nan_replacement, self.scalar_type());
      posInfReplacementScalar = getMPSScalar(pos_inf_replacement, self.scalar_type());
      negInfReplacementScalar = getMPSScalar(neg_inf_replacement, self.scalar_type());
    });

    MPSStream* stream = getCurrentMPSStream();
    Placeholder selfPlaceholder = Placeholder(cachedGraph->selfTensor, self);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor, result);

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      selfPlaceholder.getMPSGraphTensor() : selfPlaceholder.getMPSGraphTensorData(),
      cachedGraph->nanReplacementTensor : getMPSGraphTensorFromScalar(stream, nanReplacementScalar),
      cachedGraph->posInfReplacementTensor : getMPSGraphTensorFromScalar(stream, posInfReplacementScalar),
      cachedGraph->negInfReplacementTensor : getMPSGraphTensorFromScalar(stream, negInfReplacementScalar),
    };
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }
  return result;
}

static void isneginf_kernel_mps(TensorIteratorBase& iter) {
  mps::is_posneginf_helper(iter, true);
}

static void isposinf_kernel_mps(TensorIteratorBase& iter) {
  mps::is_posneginf_helper(iter, false);
}

static void clamp_kernel_mps(TensorIteratorBase& iter) {
  lib.exec_ternary_kernel(iter, "clamp");
}

static void clamp_scalar_kernel_mps(TensorIteratorBase& iter, const Scalar& min_, const Scalar& max_) {
  auto scalar_type = iter.common_dtype();
  AT_DISPATCH_ALL_TYPES_AND3(c10::kBool, c10::kBFloat16, c10::kHalf, scalar_type, "clamp_scalar_mps", [&]() {
    ClampScalarParams<scalar_t> params{min_.to<scalar_t>(), max_.to<scalar_t>()};
    lib.exec_unary_kernel_with_params(
        iter, "clamp_scalar", params, fmt::format("ClampScalarParams_{}", mps::scalarToMetalTypeString(scalar_type)));
  });
}

static void clamp_min_scalar_kernel_mps(TensorIteratorBase& iter, Scalar min_) {
  lib.exec_unary_kernel(iter, "clamp_min_scalar", min_, iter.common_dtype());
}

static void clamp_max_scalar_kernel_mps(TensorIteratorBase& iter, Scalar max_) {
  lib.exec_unary_kernel(iter, "clamp_max_scalar", max_, iter.common_dtype());
}

REGISTER_DISPATCH(where_kernel, &where_kernel_mps)
REGISTER_DISPATCH(isneginf_stub, &isneginf_kernel_mps)
REGISTER_DISPATCH(isposinf_stub, &isposinf_kernel_mps)
REGISTER_DISPATCH(clamp_stub, &clamp_kernel_mps)
REGISTER_DISPATCH(clamp_scalar_stub, &clamp_scalar_kernel_mps)
REGISTER_DISPATCH(clamp_min_scalar_stub, &clamp_min_scalar_kernel_mps)
REGISTER_DISPATCH(clamp_max_scalar_stub, &clamp_max_scalar_kernel_mps)
REGISTER_DISPATCH(isin_default_stub, &mps::isin_default_kernel_mps)
REGISTER_DISPATCH(isin_sorting_stub, &mps::isin_sorting_kernel_mps)

} // namespace at::native
