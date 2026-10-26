//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/Resize.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/mps/Copy.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/operations/ScanKernel.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/MPSFunctions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_copy_from_and_resize.h>
#include <ATen/ops/acos_native.h>
#include <ATen/ops/asin_native.h>
#include <ATen/ops/atan_native.h>
#include <ATen/ops/cos_native.h>
#include <ATen/ops/cosh_native.h>
#include <ATen/ops/cumprod_native.h>
#include <ATen/ops/cumsum_native.h>
#include <ATen/ops/erf_native.h>
#include <ATen/ops/exp2_native.h>
#include <ATen/ops/imag.h>
#include <ATen/ops/logical_not_native.h>
#include <ATen/ops/logit_native.h>
#include <ATen/ops/neg.h>
#include <ATen/ops/neg_native.h>
#include <ATen/ops/real.h>
#include <ATen/ops/reshape.h>
#include <ATen/ops/rsqrt_native.h>
#include <ATen/ops/sin_native.h>
#include <ATen/ops/sinh_native.h>
#include <ATen/ops/sqrt_native.h>
#include <ATen/ops/tan_native.h>
#include <ATen/ops/view_as_real.h>
#endif

namespace at::native {

enum class MPSCumulativeOpType : uint8_t {
  CUMSUM = 0,
  CUMPROD = 1,
};

namespace mps {

typedef MPSGraphTensor* (^UnaryOpBlock)(MPSGraph*, MPSGraphTensor*);
using is_noop_p = std::function<bool(const Tensor&)>;

static bool is_empty_tensor(const Tensor& self) {
  return self.numel() == 0;
}

static void unary_op_noresize(const Tensor& self, const Tensor& output, std::string op_name, UnaryOpBlock unaryBlock) {
  @autoreleasepool {
    std::string key = op_name + getTensorsStringKey({self, output});
    auto cachedGraph = LookUpOrCreateCachedGraph<MPSUnaryCachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      newCachedGraph->inputTensor_ = mpsGraphRankedPlaceHolder(mpsGraph, self);
      MPSGraphTensor* castTensor = newCachedGraph->inputTensor_;
      // Integer input must be cast to float if output is float
      if (isIntegralType(self.scalar_type(), true) && isFloatingType(output.scalar_type())) {
        castTensor = castMPSTensor(mpsGraph, newCachedGraph->inputTensor_, output.scalar_type());
      }
      newCachedGraph->outputTensor_ = unaryBlock(mpsGraph, castTensor);
    });

    // See https://github.com/pytorch/pytorch/issues/100764
    auto selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    auto outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);
    auto feeds = dictionaryFromPlaceholders(selfPlaceholder);
    runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), feeds, outputPlaceholder);
  }
}

static void unary_op(const Tensor& self,
                     const Tensor& output_,
                     std::string op_name,
                     UnaryOpBlock unaryBlock,
                     is_noop_p is_noop = is_empty_tensor) {
  at::native::resize_output(output_, self.sizes());

  if (is_noop(self)) {
    output_.copy_(self);
    return;
  }

  unary_op_noresize(self, output_, op_name, unaryBlock);
}

} // namespace mps

static void logit_mps_impl(const Tensor& self, std::optional<double> eps, Tensor& output, const std::string& op_name) {
  std::string key = op_name + ":[" + (eps.has_value() ? std::to_string(eps.value()) : "NULL") + "]";

  mps::unary_op(self, output, key, ^MPSGraphTensor*(MPSGraph* mpsGraph, MPSGraphTensor* inputTensor) {
    MPSGraphTensor* oneTensor = [mpsGraph constantWithScalar:1.0 shape:@[ @1 ] dataType:inputTensor.dataType];
    MPSGraphTensor* logitInputTensor;

    if (eps.has_value()) {
      MPSGraphTensor* lowTensor = [mpsGraph constantWithScalar:eps.value() shape:@[ @1 ] dataType:inputTensor.dataType];
      MPSGraphTensor* highTensor = [mpsGraph subtractionWithPrimaryTensor:oneTensor secondaryTensor:lowTensor name:nil];
      // Apply lo last so it wins when eps > 1 - eps, matching the
      // scalar `x < lo ? lo : (x > hi ? hi : x)` priority.
      MPSGraphTensor* inputGreaterThanHighTensor = [mpsGraph greaterThanWithPrimaryTensor:inputTensor
                                                                          secondaryTensor:highTensor
                                                                                     name:nil];
      MPSGraphTensor* clampedHighTensor = [mpsGraph selectWithPredicateTensor:inputGreaterThanHighTensor
                                                          truePredicateTensor:highTensor
                                                         falsePredicateTensor:inputTensor
                                                                         name:nil];
      MPSGraphTensor* inputLessThanLowTensor = [mpsGraph lessThanWithPrimaryTensor:inputTensor
                                                                   secondaryTensor:lowTensor
                                                                              name:nil];
      logitInputTensor = [mpsGraph selectWithPredicateTensor:inputLessThanLowTensor
                                         truePredicateTensor:lowTensor
                                        falsePredicateTensor:clampedHighTensor
                                                        name:nil];
    } else {
      logitInputTensor = inputTensor;
    }

    MPSGraphTensor* oneMinusInputTensor = [mpsGraph subtractionWithPrimaryTensor:oneTensor
                                                                 secondaryTensor:logitInputTensor
                                                                            name:nil];
    MPSGraphTensor* outputTensor = [mpsGraph divisionWithPrimaryTensor:logitInputTensor
                                                       secondaryTensor:oneMinusInputTensor
                                                                  name:nil];
    return [mpsGraph logarithmWithTensor:outputTensor name:nil];
  });
}

Tensor& logit_out_mps(const Tensor& self, std::optional<double> eps, Tensor& result) {
  logit_mps_impl(self, eps, result, "logit_out_mps");
  return result;
}

Tensor logit_mps(const Tensor& self, std::optional<double> eps) {
  auto out_dtype = self.scalar_type();
  if (c10::isIntegralType(out_dtype, /*includeBool*/ true)) {
    out_dtype = kFloat;
  }
  Tensor result = at::empty(self.sizes(), out_dtype, std::nullopt, kMPS, std::nullopt, std::nullopt);
  logit_mps_impl(self, eps, result, "logit_mps");
  return result;
}

static void cumulative_op_impl(const Tensor& self,
                               int64_t dim,
                               std::optional<ScalarType> dtype,
                               const Tensor& result,
                               MPSCumulativeOpType cumulativeOpType) {
  auto nDims = self.dim();
  auto wrapped_dim = maybe_wrap_dim(dim, nDims);
  TORCH_CHECK(wrapped_dim >= 0 && wrapped_dim < std::max(1LL, self.ndimension()),
              "Expected wrapped dim to be between 0 and ",
              self.ndimension(),
              " but got ",
              wrapped_dim,
              "(original dim is ",
              dim,
              ")");
  auto input = dtype.has_value() ? self.to(dtype.value()) : self;
  const std::string scan_op_name = cumulativeOpType == MPSCumulativeOpType::CUMSUM ? "cumsum" : "cumprod";

  if (input.is_complex()) {
    if (cumulativeOpType == MPSCumulativeOpType::CUMSUM) {
      auto input_real = at::view_as_real(input.dim() == 0 ? input.view({1}) : input);
      auto result_real = at::view_as_real(result.dim() == 0 ? result.view({1}) : result);
      return cumulative_op_impl(input_real, wrapped_dim, std::nullopt, result_real, MPSCumulativeOpType::CUMSUM);
    }
    auto input_view = input.dim() == 0 ? input.view({1}) : input;
    auto result_view = result.dim() == 0 ? result.view({1}) : result;
    return mps::scan_simple_mps_impl(input_view, result_view, wrapped_dim, scan_op_name);
  }

  // cumsum/cumprod of a 0-dim tensor is the identity.
  if (input.dim() == 0) {
    result.copy_(input);
    return;
  }

  const bool castInputData = isIntegralType(input.scalar_type(), /*includeBool=*/true) &&
      input.scalar_type() != ScalarType::Int && input.scalar_type() != ScalarType::Long;
  if (castInputData) {
    auto input_i32 = input.to(ScalarType::Int);
    if (result.scalar_type() == ScalarType::Long) {
      mps::scan_simple_mps_impl(input_i32, result, wrapped_dim, scan_op_name);
    } else {
      auto result_i32 = at::empty(result.sizes(), result.options().dtype(ScalarType::Int));
      mps::scan_simple_mps_impl(input_i32, result_i32, wrapped_dim, scan_op_name);
      result.copy_(result_i32);
    }
    return;
  }
  // int32 -> int64 result: scan int32 directly; the kernel widens (no upcast).
  if (input.scalar_type() == ScalarType::Int && result.scalar_type() == ScalarType::Long) {
    mps::scan_simple_mps_impl(input, result, wrapped_dim, scan_op_name);
    return;
  }

  // int64 input, or a dtype= override: match the scan input dtype to the result.
  auto scan_input = input.scalar_type() == result.scalar_type() ? input : input.to(result.scalar_type());

  mps::scan_simple_mps_impl(scan_input, result, wrapped_dim, scan_op_name);
}

TORCH_IMPL_FUNC(cumsum_out_mps)
(const Tensor& self, int64_t dim, std::optional<ScalarType> dtype, const Tensor& result) {
  return cumulative_op_impl(self, dim, dtype, result, MPSCumulativeOpType::CUMSUM);
}

TORCH_IMPL_FUNC(cumprod_out_mps)
(const Tensor& self, int64_t dim, std::optional<ScalarType> dtype, const Tensor& result) {
  return cumulative_op_impl(self, dim, dtype, result, MPSCumulativeOpType::CUMPROD);
}

} // namespace at::native
