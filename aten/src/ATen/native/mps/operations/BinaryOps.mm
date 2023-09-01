//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/ExpandUtils.h>
#include <ATen/ScalarOps.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/add_native.h>
#include <ATen/ops/atan2_native.h>
#include <ATen/ops/div_native.h>
#include <ATen/ops/eq_native.h>
#include <ATen/ops/floor_divide_native.h>
#include <ATen/ops/fmod_native.h>
#include <ATen/ops/ge_native.h>
#include <ATen/ops/gt_native.h>
#include <ATen/ops/hypot_native.h>
#include <ATen/ops/le_native.h>
#include <ATen/ops/lerp_native.h>
#include <ATen/ops/logaddexp2_native.h>
#include <ATen/ops/logaddexp_native.h>
#include <ATen/ops/logical_and_native.h>
#include <ATen/ops/logical_or_native.h>
#include <ATen/ops/logical_xor_native.h>
#include <ATen/ops/lt_native.h>
#include <ATen/ops/maximum_native.h>
#include <ATen/ops/minimum_native.h>
#include <ATen/ops/mul_native.h>
#include <ATen/ops/ne_native.h>
#include <ATen/ops/pow.h>
#include <ATen/ops/pow_native.h>
#include <ATen/ops/remainder_native.h>
#include <ATen/ops/result_type.h>
#include <ATen/ops/sub_native.h>
#include <ATen/ops/view_as_real.h>
#include <ATen/ops/xlogy_native.h>
#endif

namespace at::native {
namespace mps {

struct BinaryOpCachedGraph : public MPSCachedGraph {
  BinaryOpCachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
  MPSGraphTensor *primaryTensor = nil, *secondaryTensor = nil;
  MPSGraphTensor *alphaTensor = nil, *outputTensor = nil;
};

typedef MPSGraphTensor* (^BinaryOpBlock)(BinaryOpCachedGraph*, MPSGraphTensor*, MPSGraphTensor*);
#define BinaryOpFn(graph, primary, secondary) \
  MPSGraphTensor*(mps::BinaryOpCachedGraph * graph, MPSGraphTensor * primary, MPSGraphTensor * secondary)

// alpha is always 1.0 except when this function is called from add_sub_lerp_template()
static void binaryOpTensor(const Tensor& self,
                           const Tensor& other,
                           const Scalar& alpha,
                           const Tensor& output_,
                           std::string op_name,
                           BinaryOpBlock binaryBlock) {
  TORCH_CHECK(!(!is_macos_13_or_newer() && self.scalar_type() == ScalarType::Byte),
              "MPS support binary op with uint8 natively starting from macOS 13.0");
  TORCH_CHECK(!(op_name == "power" && !is_macos_13_or_newer(MacOSVersion::MACOS_VER_13_2_PLUS) &&
                (self.scalar_type() == ScalarType::Long ||
                 (other.scalar_type() == ScalarType::Long &&
                  (self.scalar_type() != ScalarType::Half && self.scalar_type() != ScalarType::Float)))),
              "MPS: ",
              op_name,
              " op with int64 input is supported natively starting from macOS 13.2");
  TORCH_CHECK_TYPE(!isComplexType(self.scalar_type()), "Complex types are unsupported on MPS");
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
    output = at::empty(output_.sizes(), output_.scalar_type(), c10::nullopt, kMPS, c10::nullopt, c10::nullopt);
    needsCopyToOutput = true;
  }

  auto inputDataType = self.scalar_type();
  auto otherDataType = other.scalar_type();
  auto outputDataType = output_.scalar_type();
  if (!is_macos_13_or_newer()) {
    // workaround for signed vs. unsigned comparison issue in MacOS 12
    if (outputDataType == kBool && (inputDataType == kByte || otherDataType == kByte)) {
      inputDataType = otherDataType = kByte;
    } else {
      if (inputDataType == kBool || inputDataType == kByte) {
        inputDataType = kChar;
      }
      if (otherDataType == kBool || otherDataType == kByte) {
        otherDataType = kChar;
      }
    }
  }

  @autoreleasepool {
    string key = op_name + getTensorsStringKey({self, other, output_});
    auto cachedGraph = LookUpOrCreateCachedGraph<BinaryOpCachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      newCachedGraph->primaryTensor =
          mpsGraphRankedPlaceHolder(mpsGraph, getMPSScalarType(inputDataType), getMPSShape(self));
      newCachedGraph->secondaryTensor =
          mpsGraphRankedPlaceHolder(mpsGraph, getMPSScalarType(otherDataType), getMPSShape(other));

      MPSGraphTensor* primaryCastTensor = newCachedGraph->primaryTensor;
      MPSGraphTensor* secondaryCastTensor = newCachedGraph->secondaryTensor;

      // this type inference is only required at the time of graph creation
      ScalarType common_dtype = c10::promoteTypes(inputDataType, otherDataType);
      if (isIntegralType(common_dtype, true)) {
        // integer inputs must be cast to float, if output is float
        if (isFloatingType(outputDataType)) {
          common_dtype = outputDataType;
          // in boolean comparison ops with signed vs. unsigned integers, we always cast to the unsigned type
        } else if (outputDataType == ScalarType::Bool &&
                   (inputDataType == ScalarType::Byte || otherDataType == ScalarType::Byte)) {
          common_dtype = ScalarType::Byte;
        }
      }
      if (inputDataType != common_dtype) {
        primaryCastTensor = castMPSTensor(mpsGraph, newCachedGraph->primaryTensor, common_dtype);
      }
      if (otherDataType != common_dtype) {
        secondaryCastTensor = castMPSTensor(mpsGraph, newCachedGraph->secondaryTensor, common_dtype);
      }
      newCachedGraph->outputTensor = binaryBlock(newCachedGraph, primaryCastTensor, secondaryCastTensor);
      // Cast output tensor to an expected type if needed, which addresses discrepancy when int64 scalar is added to
      // int32 tensor Output tensor should have been promoted but it remains an int32 tensor
      if (outputDataType != common_dtype || [newCachedGraph->outputTensor dataType] != getMPSDataType(outputDataType)) {
        newCachedGraph->outputTensor = castMPSTensor(mpsGraph, newCachedGraph->outputTensor, outputDataType);
      }
    });

    NSMutableDictionary* feeds = [[NSMutableDictionary new] autorelease];
    Placeholder selfPlaceholder;
    Placeholder otherPlaceholder;
    MPSScalar self_scalar;
    MPSScalar other_scalar;
    MPSScalar alpha_scalar;

    if (is_self_scalar && !self.is_mps()) {
      self_scalar = getMPSScalar(self.item(), inputDataType);
      feeds[cachedGraph->primaryTensor] = getMPSGraphTensorFromScalar(mpsStream, self_scalar);
    } else {
      selfPlaceholder = Placeholder(cachedGraph->primaryTensor,
                                    self,
                                    /*mpsShape*/ nil,
                                    /*gatherTensorData=*/true,
                                    getMPSScalarType(inputDataType));
      feeds[selfPlaceholder.getMPSGraphTensor()] = selfPlaceholder.getMPSGraphTensorData();
    }
    if (is_other_scalar && !other.is_mps()) {
      other_scalar = getMPSScalar(other.item(), otherDataType);
      feeds[cachedGraph->secondaryTensor] = getMPSGraphTensorFromScalar(mpsStream, other_scalar);
    } else {
      otherPlaceholder = Placeholder(cachedGraph->secondaryTensor,
                                     other,
                                     /*mpsShape*/ nil,
                                     /*gatherTensorData=*/true,
                                     getMPSScalarType(otherDataType));
      feeds[otherPlaceholder.getMPSGraphTensor()] = otherPlaceholder.getMPSGraphTensorData();
    }

    // 'cachedGraph->alphaTensor' is not nil only if add_sub_lerp_template() was called with an alpha value != 1.0
    if (cachedGraph->alphaTensor) {
      alpha_scalar = getMPSScalar(alpha, other.scalar_type());
      feeds[cachedGraph->alphaTensor] = getMPSGraphTensorFromScalar(mpsStream, alpha_scalar);
    }

    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor, needsCopyToOutput ? output : output_);
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results =
        @{outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()};
    runMPSGraph(mpsStream, cachedGraph->graph(), feeds, results);

    if (needsCopyToOutput) {
      output_.copy_(output);
    }
  }
}

static void binaryOpScalar(const Tensor& self,
                           const Scalar& other,
                           const Scalar& alpha,
                           const Tensor& output,
                           std::string op_name,
                           BinaryOpBlock binaryBlock) {
  binaryOpTensor(self, wrapped_scalar_tensor(other), alpha, output, op_name, binaryBlock);
}

static void div_mode_template(const Tensor& self,
                              const Tensor& other,
                              c10::optional<c10::string_view> rounding_mode,
                              const Tensor& output,
                              const string op_name) {
  if (rounding_mode.has_value() && *rounding_mode == "trunc") {
    TORCH_CHECK(self.scalar_type() != ScalarType::Half, "MPS: does not support trunc_divide op with float16 input");
  }
  BinaryOpBlock div_mode_op_block = ^BinaryOpFn(cachedGraph, primaryCastTensor, secondaryCastTensor) {
    MPSGraph* mpsGraph = cachedGraph->graph();
    bool isFloatInput = ([primaryCastTensor dataType] & MPSDataTypeFloatBit) != 0;
    if (!isFloatInput && rounding_mode.has_value() && (*rounding_mode == "floor" || *rounding_mode == "trunc")) {
      primaryCastTensor = [mpsGraph castTensor:primaryCastTensor toType:MPSDataTypeFloat32 name:@"primaryCastTensor"];
      secondaryCastTensor = [mpsGraph castTensor:secondaryCastTensor
                                          toType:MPSDataTypeFloat32
                                            name:@"secondaryCastTensor"];
    }
    MPSGraphTensor* divTensor = [mpsGraph divisionWithPrimaryTensor:primaryCastTensor
                                                    secondaryTensor:secondaryCastTensor
                                                               name:nil];
    // Rounding is a no-op for integral types, and also a reasonable workaround
    // For MPSGraph bug on Apple Silicon, that throws `Function floorOp_i64 was not found in the library`
    // See https://github.com/pytorch/pytorch/issues/84995
    bool isFloatOutput = ([divTensor dataType] & MPSDataTypeFloatBit) != 0;
    if (!rounding_mode.has_value() || !isFloatOutput) {
      return divTensor;
    } else if (*rounding_mode == "trunc") {
      auto truncTensor = trunc_tensor(mpsGraph, divTensor);
      if (op_name == "fmod_mps_out") {
        auto mulTensor = [mpsGraph multiplicationWithPrimaryTensor:truncTensor
                                                   secondaryTensor:secondaryCastTensor
                                                              name:nil];
        return [mpsGraph subtractionWithPrimaryTensor:primaryCastTensor secondaryTensor:mulTensor name:nil];
      }
      return truncTensor;
    } else if (*rounding_mode == "floor") {
      MPSGraphTensor* floorTensor = [mpsGraph floorWithTensor:divTensor name:nil];
      if (op_name == "remainder_out_mps") {
        auto mulTensor = [mpsGraph multiplicationWithPrimaryTensor:floorTensor
                                                   secondaryTensor:secondaryCastTensor
                                                              name:nil];
        return [mpsGraph subtractionWithPrimaryTensor:primaryCastTensor secondaryTensor:mulTensor name:nil];
      }
      return floorTensor;
    }
    assert(0 && "Invalid rounding mode\n");
    return nullptr;
  };
  binaryOpTensor(self,
                 other,
                 Scalar(1.0),
                 output,
                 op_name + "_mps:" + (rounding_mode.has_value() ? c10::str(*rounding_mode) : ""),
                 div_mode_op_block);
}

static void add_sub_lerp_template(const Tensor& self,
                                  const Tensor& other,
                                  const Scalar& alpha,
                                  const Tensor& output,
                                  std::string op_name) {
  if (alpha.toDouble() == 0.0) {
    if (!self.is_alias_of(output)) { // if inplace, no-op
      output.copy_(self);
    }
    return;
  }

  const bool alpha_has_value = alpha.toDouble() != 1.0;
  if (alpha_has_value) {
    auto commonDtype = at::result_type(self, other);
    at::native::alpha_check(commonDtype, alpha);
  }

  if (!alpha_has_value && op_name == "lerp") {
    if (!self.is_alias_of(other)) { // if inplace, no-op
      output.copy_(other);
    }
    return;
  }

  BinaryOpBlock add_sub_lerp_op_block = ^BinaryOpFn(cachedGraph, primaryCastTensor, secondaryCastTensor) {
    MPSGraph* mpsGraph = cachedGraph->graph();
    MPSGraphTensor* secondaryTensor = secondaryCastTensor;

    if (op_name == "lerp") {
      secondaryCastTensor = [mpsGraph subtractionWithPrimaryTensor:secondaryCastTensor
                                                   secondaryTensor:primaryCastTensor
                                                              name:nil];
    }

    // if alpha is 1.0, then we don't bother adding another multiply to graph
    if (alpha_has_value) {
      cachedGraph->alphaTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSScalarType(other.scalar_type()), @[ @1 ]);
      secondaryTensor = [mpsGraph multiplicationWithPrimaryTensor:secondaryCastTensor
                                                  secondaryTensor:cachedGraph->alphaTensor
                                                             name:nil];
    }
    if (op_name == "add" || op_name == "lerp")
      return [mpsGraph additionWithPrimaryTensor:primaryCastTensor secondaryTensor:secondaryTensor name:nil];
    else
      return [mpsGraph subtractionWithPrimaryTensor:primaryCastTensor secondaryTensor:secondaryTensor name:nil];
  };
  // add alpha's type to the key only if multiply was added to graph
  binaryOpTensor(self,
                 other,
                 alpha,
                 output,
                 op_name + "_out_mps:" + (alpha_has_value ? getMPSTypeString(alpha.type()) : ""),
                 add_sub_lerp_op_block);
}

} // namespace mps

#define CREATE_MPS_BINARY_COMPARISON_OP_FUNC(func_out, func_stub, other_type)                       \
  Tensor& func_out(const Tensor& self, const other_type& other, Tensor& output) {                   \
    mps::binaryOp##other_type(                                                                      \
        self,                                                                                       \
        other,                                                                                      \
        Scalar(1.0),                                                                                \
        output,                                                                                     \
        #func_stub,                                                                                 \
        ^BinaryOpFn(cachedGraph, primaryCastTensor, secondaryCastTensor) {                          \
          MPSGraph* mpsGraph = cachedGraph->graph();                                                \
          return [mpsGraph func_stub##                                                              \
              WithPrimaryTensor:mps::castMPSTensor(mpsGraph, primaryCastTensor, ScalarType::Bool)   \
                secondaryTensor:mps::castMPSTensor(mpsGraph, secondaryCastTensor, ScalarType::Bool) \
                           name:nil];                                                               \
        });                                                                                         \
    return output;                                                                                  \
  }

#define CREATE_MPS_STRUCTURED_BINARY_OP_FUNC(func_out, func_stub, other_type)                     \
  TORCH_IMPL_FUNC(func_out)(const Tensor& self, const other_type& other, const Tensor& output) {  \
    mps::binaryOp##other_type(self,                                                               \
                              other,                                                              \
                              Scalar(1.0),                                                        \
                              output,                                                             \
                              #func_stub,                                                         \
                              ^BinaryOpFn(cachedGraph, primaryCastTensor, secondaryCastTensor) {  \
                                MPSGraph* mpsGraph = cachedGraph->graph();                        \
                                return [mpsGraph func_stub##WithPrimaryTensor:primaryCastTensor   \
                                                              secondaryTensor:secondaryCastTensor \
                                                                         name:nil];               \
                              });                                                                 \
  }

// output of Boolean Ops will be cast to "MPSDataTypeBool" at the end of binaryOpTensor()
#define CREATE_MPS_STRUCTURED_BOOLEAN_OP_FUNC(func_out, func_stub, other_type)                    \
  TORCH_IMPL_FUNC(func_out)(const Tensor& self, const other_type& other, const Tensor& output) {  \
    mps::binaryOp##other_type(self,                                                               \
                              other,                                                              \
                              Scalar(1.0),                                                        \
                              output,                                                             \
                              #func_stub,                                                         \
                              ^BinaryOpFn(cachedGraph, primaryCastTensor, secondaryCastTensor) {  \
                                MPSGraph* mpsGraph = cachedGraph->graph();                        \
                                return [mpsGraph func_stub##WithPrimaryTensor:primaryCastTensor   \
                                                              secondaryTensor:secondaryCastTensor \
                                                                         name:nil];               \
                              });                                                                 \
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
CREATE_MPS_BINARY_COMPARISON_OP_FUNC(logical_and_out_mps, logicalAND, Tensor);
CREATE_MPS_BINARY_COMPARISON_OP_FUNC(logical_or_out_mps, logicalOR, Tensor);
CREATE_MPS_BINARY_COMPARISON_OP_FUNC(logical_xor_out_mps, logicalXOR, Tensor);

TORCH_IMPL_FUNC(atan2_out_mps)(const Tensor& self, const Tensor& other, const Tensor& output) {
  TORCH_CHECK(self.scalar_type() != ScalarType::Long, "MPS does not support atan2 op with int64 input");
  mps::binaryOpTensor(
      self, other, Scalar(1.0), output, "atan2", ^BinaryOpFn(cachedGraph, primaryCastTensor, secondaryCastTensor) {
        MPSGraph* mpsGraph = cachedGraph->graph();
        return [mpsGraph atan2WithPrimaryTensor:primaryCastTensor secondaryTensor:secondaryCastTensor name:nil];
      });
}

TORCH_IMPL_FUNC(div_out_mode_mps)
(const Tensor& self, const Tensor& other, c10::optional<c10::string_view> rounding_mode, const Tensor& output) {
  mps::div_mode_template(self, other, rounding_mode, output, "div_mode_out");
}

TORCH_IMPL_FUNC(div_out_mps)(const Tensor& self, const Tensor& other, const Tensor& output) {
  mps::div_mode_template(self, other, c10::nullopt, output, "div_out");
}

TORCH_IMPL_FUNC(add_out_mps)(const Tensor& self, const Tensor& other, const Scalar& alpha, const Tensor& output) {
  if (isComplexType(self.scalar_type()) && isComplexType(other.scalar_type()) && !alpha.isComplex()) {
    // Complex add with non-complex alpha is just add over views
    return mps::add_sub_lerp_template(
        at::view_as_real(self), at::view_as_real(other), alpha, at::view_as_real(output), "add");
  }
  mps::add_sub_lerp_template(self, other, alpha, output, "add");
}

TORCH_IMPL_FUNC(sub_out_mps)(const Tensor& self, const Tensor& other, const Scalar& alpha, const Tensor& output) {
  if (isComplexType(self.scalar_type()) && isComplexType(other.scalar_type()) && !alpha.isComplex()) {
    // Complex sub with non-complex alpha is just add over views
    return mps::add_sub_lerp_template(
        at::view_as_real(self), at::view_as_real(other), alpha, at::view_as_real(output), "sub");
  }
  mps::add_sub_lerp_template(self, other, alpha, output, "sub");
}

TORCH_IMPL_FUNC(pow_Scalar_out_mps)(const Scalar& base, const Tensor& exp, const Tensor& out) {
  if (base.equal(1.0)) {
    out.fill_(1);
  } else {
    at::pow_out(const_cast<Tensor&>(out), mps::wrapped_scalar_tensor_mps(base, exp.device()), exp); // redispatch!
  }
}

Tensor& floor_divide_out_mps(const Tensor& self, const Tensor& other, Tensor& result) {
  mps::div_mode_template(self, other, "floor", result, "floor_divide_out");
  return result;
}

Tensor floor_divide_mps(const Tensor& self, const Tensor& other) {
  Tensor output = at::empty_like(self);
  mps::div_mode_template(self, other, "floor", output, "floor_divide");
  return output;
}

Tensor& floor_divide_mps_(Tensor& self, const Tensor& other) {
  return floor_divide_out_mps(self, other, self);
}

TORCH_IMPL_FUNC(remainder_out_mps)(const Tensor& self, const Tensor& other, const Tensor& output) {
  mps::div_mode_template(self, other, "floor", output, "remainder_out_mps");
}

TORCH_IMPL_FUNC(fmod_mps_out)(const Tensor& self, const Tensor& other, const Tensor& output) {
  mps::div_mode_template(self, other, "trunc", output, "fmod_mps_out");
}

TORCH_IMPL_FUNC(hypot_out_mps)(const Tensor& self, const Tensor& other, const Tensor& output) {
  mps::BinaryOpBlock hypot_op_block = ^BinaryOpFn(cachedGraph, primaryCastTensor, secondaryCastTensor) {
    MPSGraph* mpsGraph = cachedGraph->graph();
    MPSGraphTensor* twoTensor = [mpsGraph constantWithScalar:2.0 shape:@[ @1 ] dataType:primaryCastTensor.dataType];
    MPSGraphTensor* sumTensor = [mpsGraph additionWithPrimaryTensor:[mpsGraph powerWithPrimaryTensor:primaryCastTensor
                                                                                     secondaryTensor:twoTensor
                                                                                                name:nil]
                                                    secondaryTensor:[mpsGraph powerWithPrimaryTensor:secondaryCastTensor
                                                                                     secondaryTensor:twoTensor
                                                                                                name:nil]
                                                               name:nil];
    return [mpsGraph squareRootWithTensor:sumTensor name:nil];
  };
  mps::binaryOpTensor(self, other, Scalar(1.0), output, "hypot_out_mps", hypot_op_block);
}

TORCH_IMPL_FUNC(logaddexp_out_mps)(const Tensor& self, const Tensor& other, const Tensor& output) {
  mps::BinaryOpBlock logaddexp_op_block = ^BinaryOpFn(cachedGraph, primaryCastTensor, secondaryCastTensor) {
    MPSGraph* mpsGraph = cachedGraph->graph();
    MPSGraphTensor* sumTensor =
        [mpsGraph additionWithPrimaryTensor:[mpsGraph exponentWithTensor:primaryCastTensor name:nil]
                            secondaryTensor:[mpsGraph exponentWithTensor:secondaryCastTensor name:nil]
                                       name:nil];
    return [mpsGraph logarithmWithTensor:sumTensor name:nil];
  };
  mps::binaryOpTensor(self, other, Scalar(1.0), output, "logaddexp_out_mps", logaddexp_op_block);
}

TORCH_IMPL_FUNC(logaddexp2_out_mps)(const Tensor& self, const Tensor& other, const Tensor& output) {
  mps::BinaryOpBlock logaddexp2_op_block = ^BinaryOpFn(cachedGraph, primaryCastTensor, secondaryCastTensor) {
    MPSGraph* mpsGraph = cachedGraph->graph();
    MPSGraphTensor* sumTensor =
        [mpsGraph additionWithPrimaryTensor:[mpsGraph exponentBase2WithTensor:primaryCastTensor name:nil]
                            secondaryTensor:[mpsGraph exponentBase2WithTensor:secondaryCastTensor name:nil]
                                       name:nil];
    return [mpsGraph logarithmBase2WithTensor:sumTensor name:nil];
  };
  mps::binaryOpTensor(self, other, Scalar(1.0), output, "logaddexp2_out_mps", logaddexp2_op_block);
}

TORCH_IMPL_FUNC(xlogy_out_mps)(const Tensor& self, const Tensor& other, const Tensor& output) {
  mps::BinaryOpBlock xlogy_op_block = ^BinaryOpFn(cachedGraph, primaryCastTensor, secondaryCastTensor) {
    MPSGraph* mpsGraph = cachedGraph->graph();
    MPSGraphTensor* zeroTensor = [mpsGraph constantWithScalar:0.0 shape:@[ @1 ] dataType:primaryCastTensor.dataType];
    MPSGraphTensor* yIsNaNPredicateTensor = [mpsGraph isNaNWithTensor:secondaryCastTensor name:nil];
    MPSGraphTensor* logyTensor = [mpsGraph logarithmWithTensor:secondaryCastTensor name:nil];
    MPSGraphTensor* xlogyTensor = [mpsGraph multiplicationWithPrimaryTensor:primaryCastTensor
                                                            secondaryTensor:logyTensor
                                                                       name:nil];
    MPSGraphTensor* xEqualZeroPredicateTensor = [mpsGraph equalWithPrimaryTensor:primaryCastTensor
                                                                 secondaryTensor:zeroTensor
                                                                            name:nil];
    MPSGraphTensor* outputTensor = [mpsGraph selectWithPredicateTensor:xEqualZeroPredicateTensor
                                                   truePredicateTensor:zeroTensor
                                                  falsePredicateTensor:xlogyTensor
                                                                  name:nil];
    outputTensor = [mpsGraph selectWithPredicateTensor:yIsNaNPredicateTensor
                                   truePredicateTensor:secondaryCastTensor
                                  falsePredicateTensor:outputTensor
                                                  name:nil];
    return outputTensor;
  };
  mps::binaryOpTensor(self, other, Scalar(1.0), output, "xlogy_out_mps", xlogy_op_block);
}

TORCH_IMPL_FUNC(lerp_Scalar_mps)(const Tensor& self, const Tensor& end, const Scalar& weight, const Tensor& out) {
  mps::add_sub_lerp_template(self, end, weight, out, "lerp");
}
} // namespace at::native
