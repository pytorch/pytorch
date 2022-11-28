//  Copyright © 2022 Apple Inc.

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <ATen/Utils.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>
#include <torch/library.h>
#include <c10/util/Optional.h>
#include <ATen/native/BinaryOps.h>

namespace at {
namespace native {
namespace mps {

struct BinaryOpCachedGraph : public MPSCachedGraph
{
  BinaryOpCachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
  MPSGraphTensor *primaryTensor = nil, *secondaryTensor = nil;
  MPSGraphTensor *alphaTensor = nil, *outputTensor = nil;
};

typedef MPSGraphTensor* (^BinaryOpBlock)(BinaryOpCachedGraph*, MPSGraphTensor*, MPSGraphTensor*);
#define BinaryOpFn(graph, primary, secondary) MPSGraphTensor* (mps::BinaryOpCachedGraph* graph, MPSGraphTensor* primary, MPSGraphTensor* secondary)

// alpha is always 1.0 except when this function is called from add_sub_template()
void binaryOpTensor(const Tensor& self, const Tensor& other, const Scalar& alpha,
                    const Tensor& output_, std::string op_name, BinaryOpBlock binaryBlock)
{
  MPSStream* mpsStream = getCurrentMPSStream();

  const bool is_self_scalar = self.dim() == 0;
  const bool is_other_scalar = other.dim() == 0;

  auto new_size = at::infer_size(self.sizes(), other.sizes());
  if (!output_.sizes().equals(new_size)) {
      output_.resize_(new_size);
  }

  // it's possible to receive empty tensors here
  if (self.numel() == 0 || other.numel() == 0) {
    return;
  }

  Tensor output = output_;
  bool needsCopyToOutput = false;

  if (!output_.is_contiguous()) {
    output = output_.contiguous();
    needsCopyToOutput = true;
  // else, determine if this is an in-place operation on a view output
  } else if (output_.is_view() && (self.is_alias_of(output_) || other.is_alias_of(output_))) {
    output = at::native::empty_mps(output_.sizes(), output_.scalar_type(), c10::nullopt, kMPS);
    needsCopyToOutput = true;
  }

  MPSGraphCache* cache_ = MPSGraphCache::getInstance();
  @autoreleasepool {
    string key = op_name + getTensorsStringKey({self, other, output_}, /*use_scalar_value*/ false);
    BinaryOpCachedGraph* cachedGraph = static_cast<BinaryOpCachedGraph *>(cache_->LookUp(key));

    if(!cachedGraph) {
      MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ MPSCachedGraph* () {
        BinaryOpCachedGraph *newCachedGraph = nil;
        @autoreleasepool {
          MPSGraph* mpsGraph = make_mps_graph();
          newCachedGraph = new BinaryOpCachedGraph(mpsGraph);
          newCachedGraph->primaryTensor   = mpsGraphRankedPlaceHolder(mpsGraph, self);
          newCachedGraph->secondaryTensor = mpsGraphRankedPlaceHolder(mpsGraph, other);

          MPSGraphTensor* primaryCastTensor   = newCachedGraph->primaryTensor;
          MPSGraphTensor* secondaryCastTensor = newCachedGraph->secondaryTensor;

          // this type inference is only required at the time of graph creation
          const ScalarType common_dtype = c10::promoteTypes(self.scalar_type(), other.scalar_type());

          // Condition -
          // 1. Division operation
          // 2. Inputs are not float
          bool div_condition = op_name.rfind("div", 0) == 0
                                  && (!(common_dtype == ScalarType::Float || common_dtype == ScalarType::Half));

          auto compute_type = ScalarType::Float;

          if(div_condition) {

            if(output_.scalar_type() == ScalarType::Float || output_.scalar_type() == ScalarType::Half)
              compute_type = output_.scalar_type();

            primaryCastTensor = castMPSTensor(mpsGraph, newCachedGraph->primaryTensor, compute_type);
            secondaryCastTensor = castMPSTensor(mpsGraph, newCachedGraph->secondaryTensor, compute_type);
          }
          else  {
            if (self.scalar_type() != common_dtype) {
              primaryCastTensor = castMPSTensor(mpsGraph, newCachedGraph->primaryTensor, common_dtype);
            }
            if (other.scalar_type() != common_dtype) {
              secondaryCastTensor = castMPSTensor(mpsGraph, newCachedGraph->secondaryTensor, common_dtype);
            }
          }
          newCachedGraph->outputTensor = binaryBlock(newCachedGraph, primaryCastTensor, secondaryCastTensor);
          // Cast output tensor to an expected type if needed, which addresses discrepancy when int64 scalar is added to int32 tensor
          // Output tensor should have been promoted but it remains an int32 tensor

          if ((div_condition && compute_type != output_.scalar_type()) ||
              output_.scalar_type() != common_dtype) {
            newCachedGraph->outputTensor = castMPSTensor(mpsGraph, newCachedGraph->outputTensor, output_.scalar_type());
          }
        }
        return newCachedGraph;
      });
      cachedGraph = static_cast<BinaryOpCachedGraph *>(tmpCachedGraph);
    }

    NSMutableDictionary *feeds = [[NSMutableDictionary new] autorelease];
    Placeholder selfPlaceholder;
    Placeholder otherPlaceholder;
    MPSScalar self_scalar;
    MPSScalar other_scalar;
    MPSScalar alpha_scalar;

    if (is_self_scalar && !self.is_mps()) {
      self_scalar = getMPSScalar(self.item(), self.scalar_type());
      feeds[cachedGraph->primaryTensor] = getMPSGraphTensorFromScalar(mpsStream, self_scalar);
    } else {
      selfPlaceholder = Placeholder(cachedGraph->primaryTensor, self);
      feeds[selfPlaceholder.getMPSGraphTensor()] = selfPlaceholder.getMPSGraphTensorData();
    }
    if (is_other_scalar && !other.is_mps()) {
      other_scalar = getMPSScalar(other.item(), other.scalar_type());
      feeds[cachedGraph->secondaryTensor] = getMPSGraphTensorFromScalar(mpsStream, other_scalar);
    } else {
      otherPlaceholder = Placeholder(cachedGraph->secondaryTensor, other);
      feeds[otherPlaceholder.getMPSGraphTensor()] = otherPlaceholder.getMPSGraphTensorData();
    }

    // 'cachedGraph->alphaTensor' is not nil only if add_sub_template() was called with an alpha value != 1.0
    if (cachedGraph->alphaTensor) {
      alpha_scalar = getMPSScalar(alpha, other.scalar_type());
      feeds[cachedGraph->alphaTensor] = getMPSGraphTensorFromScalar(mpsStream, alpha_scalar);
    }

    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor, needsCopyToOutput ? output : output_);
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };
    runMPSGraph(mpsStream, cachedGraph->graph(), feeds, results);

    if (needsCopyToOutput) {
      output_.copy_(output);
    }
  }
}

void binaryOpScalar(const Tensor& self, const Scalar& other, const Scalar& alpha,
                    const Tensor& output, std::string op_name, BinaryOpBlock binaryBlock)
{
  binaryOpTensor(self, wrapped_scalar_tensor(other), alpha, output, op_name, binaryBlock);
}

void div_mode_template(const Tensor& self, const Tensor& other,
                       c10::optional<c10::string_view> rounding_mode,
                       const Tensor& output, const string op_name)
{
  BinaryOpBlock div_mode_op_block = ^BinaryOpFn(cachedGraph, primaryCastTensor, secondaryCastTensor) {
    MPSGraph* mpsGraph = cachedGraph->graph();
    MPSGraphTensor* divTensor =  [mpsGraph divisionWithPrimaryTensor:primaryCastTensor
                                                     secondaryTensor:secondaryCastTensor
                                                                name:nil];
    // Rounding is a no-op for integral types, and also a reasonable workaround
    // For MPSGraph bug on Apple Silicon, that throws `Function floorOp_i64 was not found in the library`
    // See https://github.com/pytorch/pytorch/issues/84995
    bool isFloatOutput = ([divTensor dataType] & MPSDataTypeFloatBit) != 0;
    if (!rounding_mode.has_value() || !isFloatOutput) {
      return divTensor;
    } else if (*rounding_mode == "trunc") {
      return trunc_tensor(mpsGraph, divTensor);
    } else if (*rounding_mode == "floor") {
      return [mpsGraph floorWithTensor:divTensor name:nil];
    }
    assert(0 && "Invalid rounding mode\n");
    return nullptr;
  };
  binaryOpTensor(self, other, Scalar(1.0), output, op_name + "_out_mps:" + (rounding_mode.has_value() ? c10::str(*rounding_mode) : ""), div_mode_op_block);
}

void add_sub_template(const Tensor& self, const Tensor& other, const Scalar& alpha, const Tensor& output, std::string op_name)
{
  if (alpha.toDouble() == 0.0) {
    const_cast<Tensor&>(output) = self.clone();
    return;
  }

  const bool alpha_has_value = alpha.toDouble() != 1.0;
  if (alpha_has_value) {
    auto commonDtype = at::result_type(self, other);
    at::native::alpha_check(commonDtype, alpha);
  }

  BinaryOpBlock add_sub_op_block = ^BinaryOpFn(cachedGraph, primaryCastTensor, secondaryCastTensor) {
    MPSGraph* mpsGraph = cachedGraph->graph();
    MPSGraphTensor* secondaryTensor = secondaryCastTensor;

    // if alpha is 1.0, then we don't bother adding another multiply to graph
    if (alpha_has_value) {
      cachedGraph->alphaTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSScalarType(other.scalar_type()), @[@1]);
      secondaryTensor = [mpsGraph multiplicationWithPrimaryTensor:secondaryCastTensor
                                                  secondaryTensor:cachedGraph->alphaTensor
                                                             name:nil];
    }
    if (op_name == "add")
      return [mpsGraph additionWithPrimaryTensor:primaryCastTensor
                                 secondaryTensor:secondaryTensor
                                            name:nil];
    else
      return [mpsGraph subtractionWithPrimaryTensor:primaryCastTensor
                                    secondaryTensor:secondaryTensor
                                               name:nil];
  };
  // add alpha's type to the key only if multiply was added to graph
  binaryOpTensor(self, other, alpha, output, op_name + "_out_mps:" + (alpha_has_value ? getMPSTypeString(alpha.type()) : ""), add_sub_op_block);
}

} // namespace mps

#define CREATE_MPS_BINARY_COMPARISON_OP_FUNC(func_out, func_stub, other_type)                                             \
Tensor& func_out (const Tensor& self, const other_type& other, Tensor& output) {                                          \
  mps::binaryOp##other_type(self, other, Scalar(1.0), output, #func_stub,                                                 \
    ^BinaryOpFn(cachedGraph, primaryCastTensor, secondaryCastTensor) {                                                    \
      MPSGraph* mpsGraph = cachedGraph->graph();                                                                          \
      return [mpsGraph func_stub##WithPrimaryTensor:mps::castMPSTensor(mpsGraph, primaryCastTensor, ScalarType::Bool)     \
                                    secondaryTensor:mps::castMPSTensor(mpsGraph, secondaryCastTensor, ScalarType::Bool)   \
                                               name:nil]; });                                                             \
  return output;                                                                                                          \
}

#define CREATE_MPS_STRUCTURED_BINARY_OP_FUNC(func_out, func_stub, other_type)                   \
TORCH_IMPL_FUNC(func_out) (const Tensor& self, const other_type& other, const Tensor& output) { \
  TORCH_CHECK(!(self.scalar_type() == ScalarType::Long &&                                       \
               (std::string(#func_stub) == "power" || std::string(#func_stub) == "atan2")),     \
               "MPS does not support ", #func_stub, " op with int64 input")                     \
  mps::binaryOp##other_type(self, other, Scalar(1.0), output, #func_stub,                       \
    ^BinaryOpFn(cachedGraph, primaryCastTensor, secondaryCastTensor) {                          \
      MPSGraph* mpsGraph = cachedGraph->graph();                                                \
      return [mpsGraph func_stub##WithPrimaryTensor:primaryCastTensor                           \
                                    secondaryTensor:secondaryCastTensor                         \
                                               name:nil]; });                                   \
}

// Boolean Ops require casting output to "MPSDataTypeBool"
#define CREATE_MPS_STRUCTURED_BOOLEAN_OP_FUNC(func_out, func_stub, other_type)                  \
TORCH_IMPL_FUNC(func_out) (const Tensor& self, const other_type& other, const Tensor& output) { \
  mps::binaryOp##other_type(self, other, Scalar(1.0), output, #func_stub,                       \
    ^BinaryOpFn(cachedGraph, primaryCastTensor, secondaryCastTensor) {                          \
      MPSGraph* mpsGraph = cachedGraph->graph();                                                \
      MPSGraphTensor* outputTensor = [mpsGraph func_stub##WithPrimaryTensor:primaryCastTensor   \
                                                            secondaryTensor:secondaryCastTensor \
                                                                       name:nil];               \
      return mps::castMPSTensor(mpsGraph, outputTensor, ScalarType::Bool); });                  \
}

// Boolean Binary Ops
CREATE_MPS_STRUCTURED_BOOLEAN_OP_FUNC(eq_scalar_out_mps, equal, Scalar);
CREATE_MPS_STRUCTURED_BOOLEAN_OP_FUNC(eq_tensor_out_mps, equal, Tensor);
CREATE_MPS_STRUCTURED_BOOLEAN_OP_FUNC(ne_scalar_out_mps, notEqual, Scalar);
CREATE_MPS_STRUCTURED_BOOLEAN_OP_FUNC(ne_tensor_out_mps, notEqual, Tensor);
CREATE_MPS_STRUCTURED_BOOLEAN_OP_FUNC(le_scalar_out_mps, lessThanOrEqualTo, Scalar);
CREATE_MPS_STRUCTURED_BOOLEAN_OP_FUNC(le_tensor_out_mps, lessThanOrEqualTo, Tensor);
CREATE_MPS_STRUCTURED_BOOLEAN_OP_FUNC(lt_scalar_out_mps, lessThan, Scalar);
CREATE_MPS_STRUCTURED_BOOLEAN_OP_FUNC(lt_tensor_out_mps, lessThan, Tensor);
CREATE_MPS_STRUCTURED_BOOLEAN_OP_FUNC(ge_scalar_out_mps, greaterThanOrEqualTo, Scalar);
CREATE_MPS_STRUCTURED_BOOLEAN_OP_FUNC(ge_tensor_out_mps, greaterThanOrEqualTo, Tensor);
CREATE_MPS_STRUCTURED_BOOLEAN_OP_FUNC(gt_scalar_out_mps, greaterThan, Scalar);
CREATE_MPS_STRUCTURED_BOOLEAN_OP_FUNC(gt_tensor_out_mps, greaterThan, Tensor);

// Arithmetic Binary Ops
CREATE_MPS_STRUCTURED_BINARY_OP_FUNC(minimum_out_mps, minimum, Tensor);
CREATE_MPS_STRUCTURED_BINARY_OP_FUNC(maximum_out_mps, maximum, Tensor);
CREATE_MPS_STRUCTURED_BINARY_OP_FUNC(mul_out_mps, multiplication, Tensor);
CREATE_MPS_STRUCTURED_BINARY_OP_FUNC(pow_tensor_scalar_out_mps, power, Scalar);
CREATE_MPS_STRUCTURED_BINARY_OP_FUNC(pow_tensor_tensor_out_mps, power, Tensor);
CREATE_MPS_STRUCTURED_BINARY_OP_FUNC(atan2_mps_out, atan2, Tensor);

CREATE_MPS_BINARY_COMPARISON_OP_FUNC(logical_and_out_mps, logicalAND, Tensor);
CREATE_MPS_BINARY_COMPARISON_OP_FUNC(logical_or_out_mps, logicalOR, Tensor);
CREATE_MPS_BINARY_COMPARISON_OP_FUNC(logical_xor_out_mps, logicalXOR, Tensor);


TORCH_IMPL_FUNC(div_out_mode_mps) (const Tensor& self, const Tensor& other, c10::optional<c10::string_view> rounding_mode, const Tensor& output) {
  mps::div_mode_template(self, other, rounding_mode, output, "div_mode");
}

TORCH_IMPL_FUNC(div_out_mps) (const Tensor& self, const Tensor& other, const Tensor& output) {
  mps::div_mode_template(self, other, c10::nullopt, output, "div");
}

TORCH_IMPL_FUNC(add_out_mps) (const Tensor& self, const Tensor& other, const Scalar& alpha, const Tensor& output) {
  mps::add_sub_template(self, other, alpha, output, "add");
}

TORCH_IMPL_FUNC(sub_out_mps) (const Tensor& self, const Tensor& other, const Scalar& alpha, const Tensor& output) {
  mps::add_sub_template(self, other, alpha, output, "sub");
}


TORCH_IMPL_FUNC(logaddexp_out_mps) (const Tensor& self, const Tensor& other, const Tensor& output)
{
      using namespace mps;
      MPSStream* stream = getCurrentMPSStream();

      if (&output != &self) {
          output.resize_(self.sizes());;
      }

      // Derive from MPSCachedGraph
      struct CachedGraph : public MPSCachedGraph
      {
        CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
        MPSGraphTensor *inputTensor_ = nil;
        MPSGraphTensor *otherTensor_ = nil;
        MPSGraphTensor *outputTensor_ = nil;
      };

      MPSGraphCache* cache_ = MPSGraphCache::getInstance();

      @autoreleasepool {
        string key = "log_base_e_out_mps:" + getTensorsStringKey({self, other});
        CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));

        if(!cachedGraph) {
          MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ MPSCachedGraph * () {
            CachedGraph *newCachedGraph = nil;

            @autoreleasepool {
              MPSGraph* mpsGraph = make_mps_graph();
              newCachedGraph = new CachedGraph(mpsGraph);
              MPSGraphTensor* xTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
              MPSGraphTensor* yTensor = mpsGraphRankedPlaceHolder(mpsGraph, other);
              MPSGraphTensor* ePowXTensor = [mpsGraph exponentWithTensor:xTensor
                                                                         name:nil];
              MPSGraphTensor* ePowYTensor = [mpsGraph exponentWithTensor:yTensor
                                                                         name:nil];
              MPSGraphTensor* sumTensor = [mpsGraph additionWithPrimaryTensor:ePowXTensor
                                                                secondaryTensor:ePowYTensor
                                                                     name:nil];
              MPSGraphTensor* outputTensor = [mpsGraph logarithmWithTensor:sumTensor
                                                                     name:nil];

              newCachedGraph->inputTensor_ = xTensor;
              newCachedGraph->otherTensor_ = yTensor;
              newCachedGraph->outputTensor_ = outputTensor;
            }
            return newCachedGraph;
          });
          cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
        }

        Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
        Placeholder otherPlaceholder = Placeholder(cachedGraph->otherTensor_, other);
        Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);

        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
          selfPlaceholder.getMPSGraphTensor() : selfPlaceholder.getMPSGraphTensorData(),
          otherPlaceholder.getMPSGraphTensor() : otherPlaceholder.getMPSGraphTensorData()
        };
        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
          outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
        };

        runMPSGraph(stream, cachedGraph->graph(), feeds, results);
      }

    }

TORCH_IMPL_FUNC(logaddexp2_out_mps) (const Tensor& self, const Tensor& other, const Tensor& output)
{
      using namespace mps;
      MPSStream* stream = getCurrentMPSStream();

      if (&output != &self) {
          output.resize_(self.sizes());;
      }

      // Derive from MPSCachedGraph
      struct CachedGraph : public MPSCachedGraph
      {
        CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
        MPSGraphTensor *inputTensor_ = nil;
        MPSGraphTensor *otherTensor_ = nil;
        MPSGraphTensor *outputTensor_ = nil;
      };

      MPSGraphCache* cache_ = MPSGraphCache::getInstance();

      @autoreleasepool {
        string key = "log_base_two_out_mps:" + getTensorsStringKey({self, other});
        CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));

        if(!cachedGraph) {
          MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ MPSCachedGraph * () {
            CachedGraph *newCachedGraph = nil;

            @autoreleasepool {
              MPSGraph* mpsGraph = make_mps_graph();
              newCachedGraph = new CachedGraph(mpsGraph);
              MPSGraphTensor* xTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
              MPSGraphTensor* yTensor = mpsGraphRankedPlaceHolder(mpsGraph, other);
              MPSGraphTensor* twoPowXTensor = [mpsGraph exponentBase2WithTensor:xTensor
                                                                         name:nil];
              MPSGraphTensor* twoPowYTensor = [mpsGraph exponentBase2WithTensor:yTensor
                                                                         name:nil];
              MPSGraphTensor* sumTensor = [mpsGraph additionWithPrimaryTensor:twoPowXTensor
                                                                secondaryTensor:twoPowYTensor
                                                                     name:nil];
              MPSGraphTensor* outputTensor = [mpsGraph logarithmBase2WithTensor:sumTensor
                                                                     name:nil];

              newCachedGraph->inputTensor_ = xTensor;
              newCachedGraph->otherTensor_ = yTensor;
              newCachedGraph->outputTensor_ = outputTensor;
            }
            return newCachedGraph;
          });
          cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
        }

        Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
        Placeholder otherPlaceholder = Placeholder(cachedGraph->otherTensor_, other);
        Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);

        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
          selfPlaceholder.getMPSGraphTensor() : selfPlaceholder.getMPSGraphTensorData(),
          otherPlaceholder.getMPSGraphTensor() : otherPlaceholder.getMPSGraphTensorData()
        };
        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
          outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
        };

        runMPSGraph(stream, cachedGraph->graph(), feeds, results);
      }
}

} // namespace native
} // namespace at
