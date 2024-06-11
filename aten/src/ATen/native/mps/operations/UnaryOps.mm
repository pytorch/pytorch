//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/mps/Copy.h>
#include <ATen/native/mps/MPSGraphSonomaOps.h>
#include <ATen/native/mps/MPSGraphVenturaOps.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/MPSFunctions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_copy_from_and_resize.h>
#include <ATen/ops/abs_native.h>
#include <ATen/ops/acos_native.h>
#include <ATen/ops/acosh_native.h>
#include <ATen/ops/asin_native.h>
#include <ATen/ops/asinh_native.h>
#include <ATen/ops/atan_native.h>
#include <ATen/ops/atanh_native.h>
#include <ATen/ops/ceil_native.h>
#include <ATen/ops/conj_physical_native.h>
#include <ATen/ops/cos_native.h>
#include <ATen/ops/cosh_native.h>
#include <ATen/ops/cumprod_native.h>
#include <ATen/ops/cumsum_native.h>
#include <ATen/ops/erf_native.h>
#include <ATen/ops/exp2_native.h>
#include <ATen/ops/expm1_native.h>
#include <ATen/ops/floor_native.h>
#include <ATen/ops/frac_native.h>
#include <ATen/ops/imag.h>
#include <ATen/ops/log10_native.h>
#include <ATen/ops/log1p_native.h>
#include <ATen/ops/log2_native.h>
#include <ATen/ops/log_native.h>
#include <ATen/ops/logical_not_native.h>
#include <ATen/ops/logit_backward_native.h>
#include <ATen/ops/logit_native.h>
#include <ATen/ops/neg.h>
#include <ATen/ops/neg_native.h>
#include <ATen/ops/real.h>
#include <ATen/ops/reciprocal_native.h>
#include <ATen/ops/reshape.h>
#include <ATen/ops/round_native.h>
#include <ATen/ops/rsqrt_native.h>
#include <ATen/ops/sgn_native.h>
#include <ATen/ops/sigmoid_native.h>
#include <ATen/ops/sign_mps_dispatch.h>
#include <ATen/ops/sign_native.h>
#include <ATen/ops/signbit_native.h>
#include <ATen/ops/sin_native.h>
#include <ATen/ops/sinh_native.h>
#include <ATen/ops/sqrt_native.h>
#include <ATen/ops/tan_native.h>
#include <ATen/ops/trunc_native.h>
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
  TORCH_CHECK(!(!is_macos_13_or_newer() && self.scalar_type() == ScalarType::Byte),
              "MPS support unary op with uint8 natively starting from macOS 13.0");

  auto output = output_;
  bool needsCopyToOutput = false;
  if (needsGather(output)) {
    output = at::empty(output.sizes(), output.scalar_type(), c10::nullopt, kMPS, c10::nullopt, c10::nullopt);
    needsCopyToOutput = true;
  }

  @autoreleasepool {
    string key = op_name + getTensorsStringKey({self, output});
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
    if (!is_dense_in_storage(self)) {
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
  if (!output_.is_same_size(self)) {
    output_.resize_(self.sizes());
  }

  if (is_noop(self)) {
    output_.copy_(self);
    return;
  }

  unary_op_noresize(self, output_, op_name, unaryBlock);
}

MPSGraphTensor* trunc_tensor(MPSGraph* mpsGraph, MPSGraphTensor* inputTensor) {
  // Rounding is a no-op for integral types, and also a reasonable workaround
  // For MPSGraph bug on Apple Silicon, that throws `Function floorOp_i64 was not found in the library`
  // See https://github.com/pytorch/pytorch/issues/84995
  bool isFloatInput = ([inputTensor dataType] & MPSDataTypeFloatBit) != 0;
  if (!isFloatInput) {
    return inputTensor;
  }

  if (!is_macos_13_or_newer()) {
    MPSGraphTensor* zeroTensor = [mpsGraph constantWithScalar:0.0 dataType:inputTensor.dataType];
    MPSGraphTensor* predicateTensor = [mpsGraph lessThanWithPrimaryTensor:inputTensor
                                                          secondaryTensor:zeroTensor
                                                                     name:nil];
    return [mpsGraph selectWithPredicateTensor:predicateTensor
                           truePredicateTensor:[mpsGraph ceilWithTensor:inputTensor name:nil]
                          falsePredicateTensor:[mpsGraph floorWithTensor:inputTensor name:nil]
                                          name:nil];
  } else {
    return [mpsGraph truncateWithTensor:inputTensor name:nil];
  }
};

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

TORCH_IMPL_FUNC(trunc_out_mps)(const Tensor& self, const Tensor& output) {
  mps::unary_op(self, output, "trunc_out_mps", ^MPSGraphTensor*(MPSGraph* mpsGraph, MPSGraphTensor* inputTensor) {
    return mps::trunc_tensor(mpsGraph, inputTensor);
  });
}

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

#define CREATE_MPS_STRUCTURED_UNARY_ROUNDING_TORCH_IMPL_FUNC(func_out, func_stub)                         \
  TORCH_IMPL_FUNC(func_out)(const Tensor& self, const Tensor& output) {                                   \
    mps::unary_op(                                                                                        \
        self,                                                                                             \
        output,                                                                                           \
        #func_out,                                                                                        \
        ^MPSGraphTensor*(MPSGraph * mpsGraph, MPSGraphTensor * inputTensor) {                             \
          return [mpsGraph func_stub##WithTensor:inputTensor name:nil];                                   \
        },                                                                                                \
        [](const Tensor& t) -> bool { return t.numel() == 0 || isIntegralType(t.scalar_type(), true); }); \
  }
CREATE_MPS_STRUCTURED_UNARY_ROUNDING_TORCH_IMPL_FUNC(ceil_out_mps, ceil)
CREATE_MPS_STRUCTURED_UNARY_ROUNDING_TORCH_IMPL_FUNC(floor_out_mps, floor)
CREATE_MPS_STRUCTURED_UNARY_ROUNDING_TORCH_IMPL_FUNC(round_out_mps, round)

#define CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(func_out, func_stub)                                         \
  TORCH_IMPL_FUNC(func_out)(const Tensor& self, const Tensor& output) {                                          \
    mps::unary_op(self, output, #func_out, ^MPSGraphTensor*(MPSGraph * mpsGraph, MPSGraphTensor * inputTensor) { \
      return [mpsGraph func_stub##WithTensor:inputTensor name:nil];                                              \
    });                                                                                                          \
  }

CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(exp2_out_mps, exponentBase2)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(reciprocal_out_mps, reciprocal)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(sqrt_out_mps, squareRoot)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(rsqrt_out_mps, reverseSquareRoot)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(neg_out_mps, negative)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(log_out_mps, logarithm)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(log10_out_mps, logarithmBase10)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(log2_out_mps, logarithmBase2)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(erf_out_mps, erf)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(sin_out_mps, sin)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(cos_out_mps, cos)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(tan_out_mps, tan)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(asin_out_mps, asin)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(acos_out_mps, acos)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(atan_out_mps, atan)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(sinh_out_mps, sinh)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(cosh_out_mps, cosh)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(asinh_out_mps, asinh)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(acosh_out_mps, acosh)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(atanh_out_mps, atanh)

Tensor& abs_out_mps(const Tensor& self, Tensor& output) {
  using namespace mps;

  if (!output.is_same_size(self)) {
    output.resize_(self.sizes());
  }

  if (self.numel() == 0) {
    return output;
  }

  if (supportsComplex() || !self.is_complex()) {
    unary_op_noresize(self, output, "abs_out_mps", ^MPSGraphTensor*(MPSGraph* mpsGraph, MPSGraphTensor* inputTensor) {
      auto rc = [mpsGraph absoluteWithTensor:inputTensor name:nil];
      if (self.is_complex()) {
        rc = [mpsGraph realPartOfTensor:rc name:nil];
      }
      return rc;
    });
  } else {
    Tensor realInput = at::view_as_real(self);
    unary_op_noresize(
        realInput, output, "abs_out_mps", ^MPSGraphTensor*(MPSGraph* mpsGraph, MPSGraphTensor* inputTensor) {
          auto rc = lengthOfComplexAsReal(mpsGraph, inputTensor);
          return [mpsGraph reshapeTensor:rc withShape:getMPSShape(output) name:nil];
        });
  }
  return output;
}

Tensor& logical_not_out_mps(const Tensor& self, Tensor& output) {
  auto bool_self = self.to(ScalarType::Bool);
  mps::unary_op(bool_self, output, "logical_not_out_mps", [](MPSGraph* mpsGraph, MPSGraphTensor* inputTensor) {
    return [mpsGraph notWithTensor:inputTensor name:nil];
  });
  return output;
}

TORCH_IMPL_FUNC(sigmoid_out_mps)(const Tensor& self, const Tensor& output) {
  TORCH_CHECK(self.scalar_type() != ScalarType::Long, "MPS does not support sigmoid op with int64 input");
  mps::unary_op(self, output, "sigmoid_out_mps", ^MPSGraphTensor*(MPSGraph* mpsGraph, MPSGraphTensor* inputTensor) {
    return [mpsGraph sigmoidWithTensor:inputTensor name:nil];
  });
}

TORCH_IMPL_FUNC(log1p_out_mps)(const Tensor& self, const Tensor& output) {
  TORCH_CHECK(self.scalar_type() != ScalarType::Long, "MPS does not support log1p op with int64 input");
  mps::unary_op(self, output, "log1p_out_mps", ^MPSGraphTensor*(MPSGraph* mpsGraph, MPSGraphTensor* inputTensor) {
    return mps::log1p(mpsGraph, inputTensor);
  });
}

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

TORCH_IMPL_FUNC(expm1_out_mps)(const Tensor& self, const Tensor& output) {
  mps::unary_op(self, output, "expm1_out_mps", ^MPSGraphTensor*(MPSGraph* mpsGraph, MPSGraphTensor* inputTensor) {
    MPSGraphTensor* oneTensor = [mpsGraph constantWithScalar:1.0 shape:@[ @1 ] dataType:inputTensor.dataType];
    MPSGraphTensor* ePowTensor = [mpsGraph exponentWithTensor:inputTensor name:nil];
    return [mpsGraph subtractionWithPrimaryTensor:ePowTensor secondaryTensor:oneTensor name:nil];
  });
}

static void logit_mps_impl(const Tensor& self, c10::optional<double> eps, Tensor& output, const std::string op_name) {
  std::string key = op_name + ":[" + (eps.has_value() ? std::to_string(eps.value()) : "NULL") + "]";

  mps::unary_op(self, output, key, ^MPSGraphTensor*(MPSGraph* mpsGraph, MPSGraphTensor* inputTensor) {
    MPSGraphTensor* oneTensor = [mpsGraph constantWithScalar:1.0 shape:@[ @1 ] dataType:inputTensor.dataType];
    MPSGraphTensor* logitInputTensor;

    if (eps.has_value()) {
      MPSGraphTensor* lowTensor = [mpsGraph constantWithScalar:eps.value() shape:@[ @1 ] dataType:inputTensor.dataType];
      MPSGraphTensor* highTensor = [mpsGraph subtractionWithPrimaryTensor:oneTensor secondaryTensor:lowTensor name:nil];
      logitInputTensor = [mpsGraph clampWithTensor:inputTensor
                                    minValueTensor:lowTensor
                                    maxValueTensor:highTensor
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

Tensor& logit_out_mps(const Tensor& self, c10::optional<double> eps, Tensor& result) {
  logit_mps_impl(self, eps, result, "logit_out_mps");
  return result;
}

Tensor logit_mps(const Tensor& self, c10::optional<double> eps) {
  Tensor result = at::empty(self.sizes(), ScalarType::Float, c10::nullopt, kMPS, c10::nullopt, c10::nullopt);
  logit_mps_impl(self, eps, result, "logit_mps");
  return result;
}

TORCH_IMPL_FUNC(logit_backward_out_mps)
(const Tensor& grad_output, const Tensor& input, c10::optional<double> eps, const Tensor& grad_input) {
  using namespace mps;
  using CachedGraph = MPSUnaryGradCachedGraph;

  // Empty output
  if (grad_input.numel() == 0)
    return;

  double eps_ = eps ? eps.value() : -1.0;

  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    std::string key = "logit_backward_out_mps:" + getTensorsStringKey({grad_output, input}) + ":" + "[" +
        (eps.has_value() ? std::to_string(eps.value()) : "-1") + "]";

    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, input);
      MPSGraphTensor* gradOutputTensor = mpsGraphRankedPlaceHolder(mpsGraph, grad_output);
      MPSGraphTensor* outputTensor = mpsGraphRankedPlaceHolder(mpsGraph, grad_input);
      MPSGraphTensor* zeroTensor = [mpsGraph constantWithScalar:0.0 shape:@[ @1 ] dataType:inputTensor.dataType];
      MPSGraphTensor* oneTensor = [mpsGraph constantWithScalar:1.0 shape:@[ @1 ] dataType:inputTensor.dataType];
      MPSGraphTensor* lowTensor = [mpsGraph constantWithScalar:eps_ shape:@[ @1 ] dataType:inputTensor.dataType];
      MPSGraphTensor* inputLessThanLowPredicateTensor = [mpsGraph lessThanWithPrimaryTensor:inputTensor
                                                                            secondaryTensor:lowTensor
                                                                                       name:nil];
      MPSGraphTensor* highTensor = [mpsGraph subtractionWithPrimaryTensor:oneTensor secondaryTensor:lowTensor name:nil];
      MPSGraphTensor* inputGreaterThanHighPredicateTensor = [mpsGraph greaterThanWithPrimaryTensor:inputTensor
                                                                                   secondaryTensor:highTensor
                                                                                              name:nil];
      MPSGraphTensor* outOfIntervalTensor = [mpsGraph logicalORWithPrimaryTensor:inputLessThanLowPredicateTensor
                                                                 secondaryTensor:inputGreaterThanHighPredicateTensor
                                                                            name:nil];
      MPSGraphTensor* oneMinusInputTensor = [mpsGraph subtractionWithPrimaryTensor:oneTensor
                                                                   secondaryTensor:inputTensor
                                                                              name:nil];
      outputTensor = [mpsGraph multiplicationWithPrimaryTensor:inputTensor
                                               secondaryTensor:oneMinusInputTensor
                                                          name:nil];
      outputTensor = [mpsGraph divisionWithPrimaryTensor:gradOutputTensor secondaryTensor:outputTensor name:nil];
      outputTensor = [mpsGraph selectWithPredicateTensor:outOfIntervalTensor
                                     truePredicateTensor:zeroTensor
                                    falsePredicateTensor:outputTensor
                                                    name:nil];

      newCachedGraph->gradOutputTensor_ = gradOutputTensor;
      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->gradInputTensor_ = outputTensor;
    });
    Placeholder gradOutputPlaceholder = Placeholder(cachedGraph->gradOutputTensor_, grad_output);
    Placeholder inputPlaceholder = Placeholder(cachedGraph->inputTensor_, input);
    Placeholder gradInputPlaceholder = Placeholder(cachedGraph->gradInputTensor_, grad_input);

    // Create dictionary of inputs and outputs
    auto feeds = dictionaryFromPlaceholders(gradOutputPlaceholder, inputPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, gradInputPlaceholder);
  }
}

static void cumulative_op_impl(const Tensor& self,
                               int64_t dim,
                               c10::optional<ScalarType> dtype,
                               const Tensor& result,
                               MPSCumulativeOpType cumulativeOpType,
                               const std::string& op_name) {
  bool macOS13_3_plus = is_macos_13_or_newer(MacOSVersion::MACOS_VER_13_3_PLUS);
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
  if (!is_macos_13_or_newer()) {
    TORCH_WARN_ONCE(op_name, " supported by MPS on MacOS 13+, please upgrade");
    Tensor cpu_result;
    if (cumulativeOpType == MPSCumulativeOpType::CUMSUM) {
      cpu_result = self.to(at::Device(kCPU)).cumsum(dim, dtype);
    } else if (cumulativeOpType == MPSCumulativeOpType::CUMPROD) {
      cpu_result = self.to(at::Device(kCPU)).cumprod(dim, dtype);
    }
    at::_copy_from_and_resize(cpu_result, result);
    return;
  }
  TORCH_CHECK(!self.is_complex(), "cumulative ops are not yet supported for complex");
  auto input = dtype.has_value() ? self.to(dtype.value()) : self;

  // issue #103810551: cumsum / cumprod are broken for int8, int16 and as chances for overflow are pretty high, cast to
  // int32 fixed in macOS 13.3
  bool castInputData = (isIntegralType(input.scalar_type(), true) && input.scalar_type() != ScalarType::Int &&
                        input.scalar_type() != ScalarType::Long);

  TORCH_CHECK(macOS13_3_plus || input.scalar_type() != ScalarType::Long,
              "MPS does not support ",
              op_name,
              " op with int64 input. Support has been added in macOS 13.3");

  mps::unary_op(
      input, result, op_name + std::to_string(dim), ^MPSGraphTensor*(MPSGraph* mpsGraph, MPSGraphTensor* inputTensor) {
        if (castInputData) {
          inputTensor = mps::castMPSTensor(mpsGraph, inputTensor, ScalarType::Int);
        }
        MPSGraphTensor* rc;
        if (cumulativeOpType == MPSCumulativeOpType::CUMSUM) {
          rc = [mpsGraph cumulativeSumWithTensor:inputTensor axis:dim name:nil];
        } else if (cumulativeOpType == MPSCumulativeOpType::CUMPROD) {
          rc = [mpsGraph cumulativeProductWithTensor:inputTensor axis:dim name:nil];
        }
        if ((mps::getMPSDataType(result) != [rc dataType]) || castInputData) {
          return mps::castMPSTensor(mpsGraph, rc, result.scalar_type());
        }
        return rc;
      });
}

TORCH_IMPL_FUNC(cumsum_out_mps)
(const Tensor& self, int64_t dim, c10::optional<ScalarType> dtype, const Tensor& result) {
  return cumulative_op_impl(self, dim, dtype, result, MPSCumulativeOpType::CUMSUM, "cumsum_out_mps");
}

TORCH_IMPL_FUNC(cumprod_out_mps)
(const Tensor& self, int64_t dim, c10::optional<ScalarType> dtype, const Tensor& result) {
  return cumulative_op_impl(self, dim, dtype, result, MPSCumulativeOpType::CUMPROD, "cumprod_out_mps");
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

Tensor& conj_physical_out_mps(const Tensor& self, Tensor& result) {
  TORCH_CHECK(self.is_complex());
  if (!mps::supportsComplex()) {
    if (!result.is_same_size(self)) {
      result.resize_(self.sizes());
    }
    at::real(result).copy_(at::real(self));
    at::imag(result).copy_(at::neg(at::imag(self)));
  } else {
    mps::unary_op(self, result, "conj", ^MPSGraphTensor*(MPSGraph* mpsGraph, MPSGraphTensor* inputTensor) {
      return [mpsGraph conjugateWithTensor:inputTensor name:nil];
    });
  }
  return result;
}

} // namespace at::native
