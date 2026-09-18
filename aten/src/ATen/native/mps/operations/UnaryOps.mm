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
#include <ATen/ops/frac_native.h>
#include <ATen/ops/imag.h>
#include <ATen/ops/logical_not_native.h>
#include <ATen/ops/logit_native.h>
#include <ATen/ops/neg.h>
#include <ATen/ops/neg_native.h>
#include <ATen/ops/real.h>
#include <ATen/ops/reshape.h>
#include <ATen/ops/rsqrt_native.h>
#include <ATen/ops/sgn_native.h>
#include <ATen/ops/sign_mps_dispatch.h>
#include <ATen/ops/sign_native.h>
#include <ATen/ops/signbit_native.h>
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

static void unary_op_noresize(const Tensor& self, const Tensor& output_, std::string op_name, UnaryOpBlock unaryBlock) {
  static const bool is_macOS_15_0_or_newer = is_macos_at_least(MacOSVersion::MACOS_15_0);

  auto output = output_;
  bool needsCopyToOutput = false;
  if (needsGather(output)) {
    output = at::empty(output.sizes(), output.scalar_type(), std::nullopt, kMPS, std::nullopt, std::nullopt);
    needsCopyToOutput = true;
  }

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

    // If self is non-densely mapped in storage, create a dense output-like representation
    at::Tensor self_;
    if (!is_dense_in_storage(self) && !is_macOS_15_0_or_newer) {
      self_ = at::empty_like(output, self.scalar_type());
      mps::mps_copy_(self_, self, false);
    } else {
      self_ = self;
    }

    bool gatherTensorData = true;
    // NS: This check is wrong and needs to be fixed, as it would produce wrong results for transposed outputs
    // See https://github.com/pytorch/pytorch/issues/100764

    if (!output.is_contiguous() || output.is_view()) {
      gatherTensorData = false;
    }

    auto selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self_, /*mpsShape=*/nullptr, gatherTensorData);
    auto outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output, /*mpsShape=*/nullptr, false);
    auto feeds = dictionaryFromPlaceholders(selfPlaceholder);
    runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), feeds, outputPlaceholder);

    if (needsCopyToOutput) {
      output_.copy_(output);
    }
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

MPSGraphTensor* log1p(MPSGraph* mpsGraph, MPSGraphTensor* inputTensor) {
  MPSGraphTensor* oneTensor = [mpsGraph constantWithScalar:1.0 dataType:inputTensor.dataType];
  MPSGraphTensor* addedTensor = [mpsGraph additionWithPrimaryTensor:inputTensor secondaryTensor:oneTensor name:nil];
  return [mpsGraph logarithmWithTensor:addedTensor name:nil];
}

static MPSGraphTensor* lengthOfComplexAsReal(MPSGraph* mpsGraph, MPSGraphTensor* inputTensor) {
  auto squares = [mpsGraph squareWithTensor:inputTensor name:nil];
  auto sumSquares = [mpsGraph reductionSumWithTensor:squares axis:-1 name:nil];
  return [mpsGraph squareRootWithTensor:sumSquares name:nil];
}

} // namespace mps

TORCH_IMPL_FUNC(signbit_out_mps)(const Tensor& self, const Tensor& output) {
  mps::unary_op(self, output, "signbit_out_mps", ^MPSGraphTensor*(MPSGraph* mpsGraph, MPSGraphTensor* inputTensor) {
    MPSGraphTensor* output;
    // signbit is not implemented for int64 type.
    // workaround for `Function signbitOp_i64 was not found in the library`
    if ([inputTensor dataType] == MPSDataTypeInt64) {
      MPSGraphTensor* zeroTensor = [mpsGraph constantWithScalar:0.0 dataType:inputTensor.dataType];
      output = [mpsGraph lessThanWithPrimaryTensor:inputTensor secondaryTensor:zeroTensor name:nil];
    } else {
      output = [mpsGraph signbitWithTensor:inputTensor name:nil];
    }
    return mps::castMPSTensor(mpsGraph, output, ScalarType::Bool);
  });
}

TORCH_IMPL_FUNC(sign_out_mps)(const Tensor& self, const Tensor& output) {
  mps::unary_op(self, output, "sign_out_mps", ^MPSGraphTensor*(MPSGraph* mpsGraph, MPSGraphTensor* inputTensor) {
    // Sign op is not implemented in MPS as of MacOS13.0 beta, so simulate it using clamp
    if ([inputTensor dataType] == MPSDataTypeInt64) {
      return [mpsGraph clampWithTensor:inputTensor
                        minValueTensor:[mpsGraph constantWithScalar:-1 dataType:MPSDataTypeInt64]
                        maxValueTensor:[mpsGraph constantWithScalar:1 dataType:MPSDataTypeInt64]
                                  name:nil];
    }
    return [mpsGraph signWithTensor:inputTensor name:nil];
  });
}

#define REGISTER_MPS_UNARY_STUB(func, mps_func)                                                                        \
  static void mps_##func##_kernel(TensorIteratorBase& iter) {                                                          \
    mps::unary_op(                                                                                                     \
        iter.input(0), iter.output(0), __func__, ^MPSGraphTensor*(MPSGraph * mpsGraph, MPSGraphTensor * inputTensor) { \
          return [mpsGraph mps_func##WithTensor:inputTensor name:nil];                                                 \
        });                                                                                                            \
  }                                                                                                                    \
  REGISTER_DISPATCH(func##_stub, mps_##func##_kernel)

REGISTER_MPS_UNARY_STUB(acosh, acosh);
REGISTER_MPS_UNARY_STUB(asinh, asinh);
REGISTER_MPS_UNARY_STUB(atanh, atanh);
REGISTER_MPS_UNARY_STUB(ceil, ceil);
REGISTER_MPS_UNARY_STUB(floor, floor);
REGISTER_MPS_UNARY_STUB(trunc, truncate);

TORCH_IMPL_FUNC(frac_out_mps)(const Tensor& self, const Tensor& output) {
  TORCH_CHECK(isFloatingType(self.scalar_type()), "frac_out_mps is only implemented for floating types");
  mps::unary_op(self, output, "frac_out_mps", ^MPSGraphTensor*(MPSGraph* mpsGraph, MPSGraphTensor* inputTensor) {
    auto zeroTensor = [mpsGraph constantWithScalar:0.0 dataType:inputTensor.dataType];
    auto predicateTensor = [mpsGraph lessThanWithPrimaryTensor:inputTensor secondaryTensor:zeroTensor name:nil];
    auto truncTensor = [mpsGraph selectWithPredicateTensor:predicateTensor
                                       truePredicateTensor:[mpsGraph ceilWithTensor:inputTensor name:nil]
                                      falsePredicateTensor:[mpsGraph floorWithTensor:inputTensor name:nil]
                                                      name:nil];
    return [mpsGraph subtractionWithPrimaryTensor:inputTensor secondaryTensor:truncTensor name:nil];
  });
}

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

TORCH_IMPL_FUNC(sgn_out_mps)(const Tensor& self, const Tensor& output) {
  if (!self.is_complex()) {
    at::mps::sign_outf(self, const_cast<Tensor&>(output));
    return;
  }

  if (!output.is_same_size(self)) {
    output.resize_(self.sizes());
  }

  Tensor realInput = at::view_as_real(self);
  Tensor realOutput = at::view_as_real(output);

  auto complex_sgn_op = [&](MPSGraph* mpsGraph, MPSGraphTensor* inputTensor) -> MPSGraphTensor* {
    MPSGraphTensor* norm = mps::lengthOfComplexAsReal(mpsGraph, inputTensor);
    MPSGraphTensor* zero = [mpsGraph constantWithScalar:0.0 dataType:norm.dataType];
    MPSGraphTensor* isZero = [mpsGraph equalWithPrimaryTensor:norm secondaryTensor:zero name:nil];
    MPSGraphTensor* sgnTensor = [mpsGraph divisionWithPrimaryTensor:inputTensor secondaryTensor:norm name:nil];
    return [mpsGraph selectWithPredicateTensor:isZero truePredicateTensor:zero falsePredicateTensor:sgnTensor name:nil];
  };

  mps::unary_op(realInput, realOutput, "sgn_out_mps", complex_sgn_op);
}

} // namespace at::native
