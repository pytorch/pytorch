//  Copyright © 2022 Apple Inc.

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <ATen/Utils.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/MPSGraphVenturaOps.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace mps {

typedef MPSGraphTensor* (^UnaryOpBlock)(MPSGraph*, MPSGraphTensor*);
using is_noop_p = std::function<bool(const Tensor&)>;


bool is_empty_tensor(const Tensor& self) {
  return self.numel() == 0;
}

void unary_op(const Tensor& self, const Tensor& output, std::string op_name, UnaryOpBlock unaryBlock, is_noop_p is_noop = is_empty_tensor)
{
  if (!output.is_same_size(self)) {
    output.resize_(self.sizes());
  }
  if (is_noop(self)) {
    output.copy_(self);
    return;
  }
  MPSGraphCache* cache_ = MPSGraphCache::getInstance();
  @autoreleasepool {
    string key = op_name + getTensorsStringKey({self, output}, /*use_scalar_value*/ false);
    auto cachedGraph = cache_->LookUpAs<MPSUnaryCachedGraph>(key);

    if(!cachedGraph) {
      cachedGraph = cache_->CreateCachedGraphAs<MPSUnaryCachedGraph>(key, ^ MPSCachedGraph* () {
        MPSUnaryCachedGraph *newCachedGraph = nil;
        @autoreleasepool {
          MPSGraph* mpsGraph = make_mps_graph();
          newCachedGraph = new MPSUnaryCachedGraph(mpsGraph);
          newCachedGraph->inputTensor_ = mpsGraphRankedPlaceHolder(mpsGraph, self);
          MPSGraphTensor* castTensor = newCachedGraph->inputTensor_;
          // Integer input must be cast to float if output is float
          if (isIntegralType(self.scalar_type()) && isFloatingType(output.scalar_type())) {
            castTensor = castMPSTensor(mpsGraph, newCachedGraph->inputTensor_, output.scalar_type());
          }
          newCachedGraph->outputTensor_ = unaryBlock(mpsGraph, castTensor);
        }
        return newCachedGraph;
      });
    }

    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      selfPlaceholder.getMPSGraphTensor() : selfPlaceholder.getMPSGraphTensorData()
    };
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };
    runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), feeds, results);
  }
}

MPSGraphTensor* trunc_tensor(MPSGraph* mpsGraph, MPSGraphTensor* inputTensor)
{
  // Rounding is a no-op for integral types, and also a reasonable workaround
  // For MPSGraph bug on Apple Silicon, that throws `Function floorOp_i64 was not found in the library`
  // See https://github.com/pytorch/pytorch/issues/84995
  bool isFloatInput = ([inputTensor dataType] & MPSDataTypeFloatBit) != 0;
  if (!isFloatInput) {
    return inputTensor;
  }

  MPSGraphTensor* zeroTensor = [mpsGraph constantWithScalar:0.0
                                                   dataType:inputTensor.dataType];
  MPSGraphTensor* predicateTensor = [mpsGraph lessThanWithPrimaryTensor:inputTensor
                                                        secondaryTensor:zeroTensor
                                                                    name:nil];
  return [mpsGraph selectWithPredicateTensor:predicateTensor
                         truePredicateTensor:[mpsGraph ceilWithTensor :inputTensor name:nil]
                        falsePredicateTensor:[mpsGraph floorWithTensor:inputTensor name:nil]
                                        name:nil];
};

} // namespace mps

TORCH_IMPL_FUNC(trunc_out_mps) (const Tensor& self, const Tensor& output) {
  mps::unary_op(self, output, "trunc_out_mps",
                ^ MPSGraphTensor* (MPSGraph* mpsGraph, MPSGraphTensor* inputTensor)
                  { return mps::trunc_tensor(mpsGraph, inputTensor); });
}

TORCH_IMPL_FUNC(signbit_out_mps) (const Tensor& self, const Tensor& output) {
  mps::unary_op(self, output, "signbit_out_mps",
                ^ MPSGraphTensor* (MPSGraph* mpsGraph, MPSGraphTensor* inputTensor) {
                    MPSGraphTensor* output;
                    // signbit is not implemented for int64 type.
                    // workaround for `Function signbitOp_i64 was not found in the library`
                    if ([inputTensor dataType] == MPSDataTypeInt64) {
                      MPSGraphTensor* zeroTensor = [mpsGraph constantWithScalar:0.0 dataType:inputTensor.dataType];
                      output = [mpsGraph lessThanWithPrimaryTensor:inputTensor
                                                   secondaryTensor:zeroTensor
                                                              name:nil];
                    } else {
                      output = [mpsGraph signbitWithTensor: inputTensor name: nil];
                    }
                    return mps::castMPSTensor(mpsGraph, output, ScalarType::Bool);
                 });
}

TORCH_IMPL_FUNC(sign_out_mps) (const Tensor& self, const Tensor& output) {
  mps::unary_op(self, output, "sign_out_mps",
                ^ MPSGraphTensor* (MPSGraph* mpsGraph, MPSGraphTensor* inputTensor) {
                    // Sign op is not implemented in MPS as of MacOS13.0 beta, so simulate it using clamp
                    if ([inputTensor dataType] == MPSDataTypeInt64) {
                      return [mpsGraph clampWithTensor:inputTensor
                                        minValueTensor:[mpsGraph constantWithScalar:-1 dataType:MPSDataTypeInt64]
                                        maxValueTensor:[mpsGraph constantWithScalar:1 dataType:MPSDataTypeInt64]
                                                  name: nil];
                    }
                    return [mpsGraph signWithTensor: inputTensor name: nil];
                 });
}

#define CREATE_MPS_STRUCTURED_UNARY_ROUNDING_TORCH_IMPL_FUNC(func_out, func_stub)     \
TORCH_IMPL_FUNC(func_out) (const Tensor& self, const Tensor& output) {                \
  mps::unary_op(self, output, #func_out,                                              \
                ^ MPSGraphTensor* (MPSGraph* mpsGraph, MPSGraphTensor* inputTensor)   \
                  { return [mpsGraph func_stub##WithTensor:inputTensor name:nil]; },  \
                  [](const Tensor& t) -> bool {                                       \
                  return t.numel() == 0 || isIntegralType(t.scalar_type());           \
                });                                                                   \
}
CREATE_MPS_STRUCTURED_UNARY_ROUNDING_TORCH_IMPL_FUNC(ceil_out_mps, ceil)
CREATE_MPS_STRUCTURED_UNARY_ROUNDING_TORCH_IMPL_FUNC(floor_out_mps, floor)
CREATE_MPS_STRUCTURED_UNARY_ROUNDING_TORCH_IMPL_FUNC(round_out_mps, round)

#define CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(func_out, func_stub)              \
TORCH_IMPL_FUNC(func_out) (const Tensor& self, const Tensor& output) {                \
  mps::unary_op(self, output, #func_out,                                              \
                ^ MPSGraphTensor* (MPSGraph* mpsGraph, MPSGraphTensor* inputTensor)   \
                  { return [mpsGraph func_stub##WithTensor:inputTensor name:nil]; }); \
}

#define CREATE_MPS_UNARY_TORCH_IMPL_FUNC(func_out, func_stub)                         \
Tensor& func_out(const Tensor& self, Tensor& output) {                                \
  mps::unary_op(self, output, #func_out,                                              \
                ^ MPSGraphTensor* (MPSGraph* mpsGraph, MPSGraphTensor* inputTensor)   \
                  { return [mpsGraph func_stub##WithTensor:inputTensor name:nil]; }); \
  return output;                                                                      \
}


CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(exp_out_mps, exponent)
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
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(tanh_out_mps, tanh)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(asinh_out_mps, asinh)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(acosh_out_mps, acosh)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(atanh_out_mps, atanh)

CREATE_MPS_UNARY_TORCH_IMPL_FUNC(abs_out_mps, absolute)

Tensor& logical_not_out_mps(const Tensor& self, Tensor& output)
{
  auto bool_self = self.to(ScalarType::Bool);
  mps::unary_op(bool_self, output, "logical_not_out_mps", [](MPSGraph* mpsGraph, MPSGraphTensor* inputTensor){ return [mpsGraph notWithTensor:inputTensor name:nil];});
  return output;
}

TORCH_IMPL_FUNC(log1p_out_mps) (const Tensor& self, const Tensor& output)
{
    using namespace mps;
    if (!output.is_same_size(self)) {
      output.resize_(self.sizes());
    }
    MPSGraphCache* cache_ = MPSGraphCache::getInstance();
    @autoreleasepool {
      string key = string("log1p_out_mps") + getTensorsStringKey({self});
      auto cachedGraph = cache_->LookUpAs<MPSUnaryCachedGraph>(key);

      if(!cachedGraph) {
        cachedGraph = cache_->CreateCachedGraphAs<MPSUnaryCachedGraph>(key, ^ MPSCachedGraph* () {
          MPSUnaryCachedGraph *newCachedGraph = nil;
          @autoreleasepool {
            MPSGraph* mpsGraph = make_mps_graph();
            newCachedGraph = new MPSUnaryCachedGraph(mpsGraph);
            newCachedGraph->inputTensor_ = mpsGraphRankedPlaceHolder(mpsGraph, self);
              MPSGraphTensor* oneTensor = [mpsGraph constantWithScalar:1.0
                                                          shape:getMPSShape(self)
                                                       dataType:mps::getMPSDataType(self.scalar_type())];
              MPSGraphTensor* addedTensor = [mpsGraph additionWithPrimaryTensor:newCachedGraph->inputTensor_
                                                         secondaryTensor:oneTensor
                                                                    name:nil];
            newCachedGraph->outputTensor_ = [mpsGraph logarithmWithTensor:addedTensor
                                                                    name:nil];
          }
          return newCachedGraph;
        });
      }

      Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
      Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);
      NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
        selfPlaceholder.getMPSGraphTensor() : selfPlaceholder.getMPSGraphTensorData()
      };
      NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
        outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
      };
      runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), feeds, results);
    }
}

TORCH_IMPL_FUNC(frac_out_mps) (const Tensor& self, const Tensor& output) {
  TORCH_CHECK(isFloatingType(self.scalar_type()), "frac_out_mps is only implemented for floating types");
  mps::unary_op(self, output, "frac_out_mps",
                ^ MPSGraphTensor* (MPSGraph* mpsGraph, MPSGraphTensor* inputTensor) {
                  auto zeroTensor = [mpsGraph constantWithScalar:0.0
                                                       dataType:inputTensor.dataType];
                  auto predicateTensor = [mpsGraph lessThanWithPrimaryTensor:inputTensor
                                                             secondaryTensor:zeroTensor
                                                                        name:nil];
                  auto truncTensor = [mpsGraph selectWithPredicateTensor:predicateTensor
                                                     truePredicateTensor:[mpsGraph ceilWithTensor :inputTensor name:nil]
                                                    falsePredicateTensor:[mpsGraph floorWithTensor:inputTensor name:nil]
                                                                    name:nil];
                  return [mpsGraph subtractionWithPrimaryTensor:inputTensor
                                               secondaryTensor:truncTensor
                                                   name: nil];
                });
}

TORCH_IMPL_FUNC(expm1_out_mps) (const Tensor& self, const Tensor& output) {
  mps::unary_op(self, output, "expm1_out_mps",
                ^ MPSGraphTensor* (MPSGraph* mpsGraph, MPSGraphTensor* inputTensor) {
                  MPSGraphTensor* oneTensor = [mpsGraph constantWithScalar:1.0
                                                       shape:@[@1]
                                                       dataType:inputTensor.dataType];
                  MPSGraphTensor* ePowTensor = [mpsGraph exponentWithTensor:inputTensor
                                                                         name:nil];
                  return [mpsGraph subtractionWithPrimaryTensor:ePowTensor
                                               secondaryTensor:oneTensor
                                                   name: nil];
                });
}



TORCH_IMPL_FUNC(cumsum_out_mps)
(const Tensor& self,
 int64_t dim,
 c10::optional<ScalarType> dtype,
 const Tensor& result) {
  TORCH_CHECK(dim >=0 && dim < std::max(1LL, self.ndimension()), "Expected dim to be between 0 and ", self.ndimension(), " but got ", dim);
  if (!mps::supportsVenturaOps()) {
    TORCH_WARN_ONCE("torch.cumsum supported by MPS on MacOS 13+, please upgrade");
    auto cpu_result = self.to(at::Device(kCPU)).cumsum(dim, dtype);
    at::_copy_from_and_resize(cpu_result, result);
    return;
  }
  auto input = dtype.has_value() ? self.to(dtype.value()) : self;
  mps::unary_op(input, result, "cumsum_out_mp" + std::to_string(dim),
                ^ MPSGraphTensor* (MPSGraph* mpsGraph, MPSGraphTensor* inputTensor) {
       // cumsum is horribly broken for int8, int16 and as chances for overflow is pretty high, cast to int32
       if (isIntegralType(input.scalar_type()) && input.scalar_type() !=ScalarType::Int) {
           inputTensor = mps::castMPSTensor(mpsGraph, inputTensor, result.scalar_type());
       }
       auto rc = [mpsGraph cumulativeSumWithTensor: inputTensor
                                              axis: dim
                                              name: nil];
       if (result.scalar_type()!= input.scalar_type() ||
           (isIntegralType(input.scalar_type()) && input.scalar_type() !=ScalarType::Int)) {
         return mps::castMPSTensor(mpsGraph, rc, result.scalar_type());
       }
       return rc;
    });
}

} // namespace native
} // namespace at
