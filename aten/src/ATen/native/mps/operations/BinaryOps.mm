//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/ExpandUtils.h>
#include <ATen/ScalarOps.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/operations/BinaryKernel.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/add_native.h>
#include <ATen/ops/atan2_native.h>
#include <ATen/ops/div_native.h>
#include <ATen/ops/eq_native.h>
#include <ATen/ops/ge_native.h>
#include <ATen/ops/gt_native.h>
#include <ATen/ops/le_native.h>
#include <ATen/ops/logical_and_native.h>
#include <ATen/ops/logical_or_native.h>
#include <ATen/ops/logical_xor_native.h>
#include <ATen/ops/lt_native.h>
#include <ATen/ops/maximum_native.h>
#include <ATen/ops/minimum_native.h>
#include <ATen/ops/ne_native.h>
#include <ATen/ops/pow.h>
#include <ATen/ops/pow_native.h>
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
  MPSGraphTensor* outputTensor = nil;
};

typedef MPSGraphTensor* (^BinaryOpBlock)(BinaryOpCachedGraph*, MPSGraphTensor*, MPSGraphTensor*);
#define BinaryOpFn(graph, primary, secondary) \
  MPSGraphTensor*(mps::BinaryOpCachedGraph * graph, MPSGraphTensor * primary, MPSGraphTensor * secondary)

static void binaryOpTensor(const Tensor& self,
                           const Tensor& other,
                           const Tensor& output_,
                           std::string op_name,
                           BinaryOpBlock binaryBlock) {
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

  static const bool is_macOS_15_0_or_newer = is_macos_13_or_newer(MacOSVersion::MACOS_VER_15_0_PLUS);
  if (!is_macOS_15_0_or_newer &&
      (needsGather(output_) || (output_.is_view() && (self.is_alias_of(output_) || other.is_alias_of(output_))))) {
    output = at::empty(output_.sizes(), output_.scalar_type(), std::nullopt, kMPS, std::nullopt, std::nullopt);
    needsCopyToOutput = true;
  }

  auto inputDataType = self.scalar_type();
  auto otherDataType = other.scalar_type();
  auto outputDataType = output_.scalar_type();
  auto common_dtype = c10::promoteTypes(inputDataType, otherDataType);
  // this type inference is only required at the time of graph creation
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

  @autoreleasepool {
    std::string key = op_name + getTensorsStringKey({self, other, output_});
    auto cachedGraph = LookUpOrCreateCachedGraph<BinaryOpCachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      newCachedGraph->primaryTensor =
          mpsGraphRankedPlaceHolder(mpsGraph, getMPSScalarType(inputDataType), getMPSShape(self));
      newCachedGraph->secondaryTensor =
          mpsGraphRankedPlaceHolder(mpsGraph, getMPSScalarType(otherDataType), getMPSShape(other));

      MPSGraphTensor* primaryCastTensor = newCachedGraph->primaryTensor;
      MPSGraphTensor* secondaryCastTensor = newCachedGraph->secondaryTensor;

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

    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor, needsCopyToOutput ? output : output_);
    runMPSGraph(mpsStream, cachedGraph->graph(), feeds, outputPlaceholder);

    if (needsCopyToOutput) {
      output_.copy_(output);
    }
  }
}

static void binaryOpScalar(const Tensor& self,
                           const Scalar& other,
                           const Tensor& output,
                           std::string op_name,
                           BinaryOpBlock binaryBlock) {
  binaryOpTensor(self, wrapped_scalar_tensor(other), output, op_name, binaryBlock);
}

static void add_sub_lerp_template(const Tensor& self,
                                  const Tensor& other,
                                  const Scalar& alpha,
                                  const Tensor& output,
                                  std::string op_name) {
  if (!alpha.isComplex() && alpha.toDouble() == 0.0) {
    if (!self.is_alias_of(output)) { // if inplace, no-op
      output.copy_(self);
    }
    return;
  }

  const bool alpha_has_value = alpha.isComplex() || alpha.toDouble() != 1.0;
  if (!alpha_has_value && op_name == "lerp") {
    if (!self.is_alias_of(other)) { // if inplace, no-op
      output.copy_(other);
    }
    return;
  }

  if (alpha_has_value) {
    auto commonDtype = at::result_type(self, other);
    at::native::alpha_check(commonDtype, alpha);
    mps::binary_op_kernel(op_name + "_alpha", self, other, output, alpha);
  } else {
    mps::binary_op_kernel(op_name, self, other, output);
  }
}

} // namespace mps

#define CREATE_MPS_BINARY_COMPARISON_OP_FUNC(func_out, func_stub, other_type)                               \
  Tensor& func_out(const Tensor& self, const other_type& other, Tensor& output) {                           \
    mps::binaryOp##other_type(                                                                              \
        self, other, output, #func_stub, ^BinaryOpFn(cachedGraph, primaryCastTensor, secondaryCastTensor) { \
          MPSGraph* mpsGraph = cachedGraph->graph();                                                        \
          return [mpsGraph func_stub##                                                                      \
              WithPrimaryTensor:mps::castMPSTensor(mpsGraph, primaryCastTensor, ScalarType::Bool)           \
                secondaryTensor:mps::castMPSTensor(mpsGraph, secondaryCastTensor, ScalarType::Bool)         \
                           name:nil];                                                                       \
        });                                                                                                 \
    return output;                                                                                          \
  }

#define CREATE_MPS_STRUCTURED_BINARY_OP_FUNC(func_out, func_stub, other_type)                               \
  TORCH_IMPL_FUNC(func_out)(const Tensor& self, const other_type& other, const Tensor& output) {            \
    mps::binaryOp##other_type(                                                                              \
        self, other, output, #func_stub, ^BinaryOpFn(cachedGraph, primaryCastTensor, secondaryCastTensor) { \
          MPSGraph* mpsGraph = cachedGraph->graph();                                                        \
          return [mpsGraph func_stub##WithPrimaryTensor:primaryCastTensor                                   \
                                        secondaryTensor:secondaryCastTensor                                 \
                                                   name:nil];                                               \
        });                                                                                                 \
  }

// output of Boolean Ops will be cast to "MPSDataTypeBool" at the end of binaryOpTensor()
#define CREATE_MPS_STRUCTURED_BOOLEAN_OP_FUNC(func_out, func_stub, other_type)                              \
  TORCH_IMPL_FUNC(func_out)(const Tensor& self, const other_type& other, const Tensor& output) {            \
    mps::binaryOp##other_type(                                                                              \
        self, other, output, #func_stub, ^BinaryOpFn(cachedGraph, primaryCastTensor, secondaryCastTensor) { \
          MPSGraph* mpsGraph = cachedGraph->graph();                                                        \
          return [mpsGraph func_stub##WithPrimaryTensor:primaryCastTensor                                   \
                                        secondaryTensor:secondaryCastTensor                                 \
                                                   name:nil];                                               \
        });                                                                                                 \
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
CREATE_MPS_STRUCTURED_BINARY_OP_FUNC(minimum_out_mps, minimumWithNaNPropagationAndIntFallback, Tensor);
CREATE_MPS_STRUCTURED_BINARY_OP_FUNC(maximum_out_mps, maximumWithNaNPropagationAndIntFallback, Tensor);
CREATE_MPS_STRUCTURED_BINARY_OP_FUNC(pow_tensor_tensor_out_mps, power, Tensor);
CREATE_MPS_BINARY_COMPARISON_OP_FUNC(logical_and_out_mps, logicalAND, Tensor);
CREATE_MPS_BINARY_COMPARISON_OP_FUNC(logical_or_out_mps, logicalOR, Tensor);
CREATE_MPS_BINARY_COMPARISON_OP_FUNC(logical_xor_out_mps, logicalXOR, Tensor);

TORCH_IMPL_FUNC(atan2_out_mps)(const Tensor& self, const Tensor& other, const Tensor& output) {
  mps::binaryOpTensor(self, other, output, "atan2", ^BinaryOpFn(cachedGraph, primaryCastTensor, secondaryCastTensor) {
    MPSGraph* mpsGraph = cachedGraph->graph();
    return [mpsGraph atan2WithPrimaryTensor:primaryCastTensor secondaryTensor:secondaryCastTensor name:nil];
  });
}

TORCH_IMPL_FUNC(add_out_mps)(const Tensor& self, const Tensor& other, const Scalar& alpha, const Tensor& output) {
  mps::add_sub_lerp_template(self, other, alpha, output, "add");
}

TORCH_IMPL_FUNC(sub_out_mps)(const Tensor& self, const Tensor& other, const Scalar& alpha, const Tensor& output) {
  mps::add_sub_lerp_template(self, other, alpha, output, "sub");
}

TORCH_IMPL_FUNC(pow_Scalar_out_mps)(const Scalar& base, const Tensor& exp, const Tensor& out) {
  if (base.equal(1.0)) {
    out.fill_(1);
  } else {
    at::pow_out(const_cast<Tensor&>(out), mps::wrapped_scalar_tensor_mps(base, exp.device()), exp); // redispatch!
  }
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
  mps::binaryOpTensor(self, other, output, "xlogy_out_mps", xlogy_op_block);
}

} // namespace at::native
