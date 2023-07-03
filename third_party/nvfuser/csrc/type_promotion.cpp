#include <type_promotion.h>

#include <arith.h>
#include <ir_interface_nodes.h>

#include <ATen/native/TypeProperties.h>
#include <c10/core/ScalarType.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

enum ValueType { Tensor, Scalar, None };

struct OperandType {
  ValueType value_type = ValueType::Tensor;
  c10::ScalarType scalar_type = c10::ScalarType::Undefined;
  size_t dim = 0;
};

c10::ScalarType promoteTypesSkipUndefined(
    c10::ScalarType a,
    c10::ScalarType b) {
  if (a == c10::ScalarType::Undefined) {
    return b;
  }
  if (b == c10::ScalarType::Undefined) {
    return a;
  }
  return c10::promoteTypes(a, b);
}

at::native::ResultTypeState updateResultTypeState(
    OperandType tensor,
    const at::native::ResultTypeState& in_state) {
  at::native::ResultTypeState new_state = in_state;
  c10::ScalarType current = tensor.scalar_type;

  if (tensor.dim > 0) {
    new_state.dimResult =
        promoteTypesSkipUndefined(in_state.dimResult, current);
  } else {
    new_state.zeroResult =
        promoteTypesSkipUndefined(in_state.zeroResult, current);
  }
  return new_state;
}

at::native::ResultTypeState updateResultTypeState(
    const c10::ScalarType scalar,
    const at::native::ResultTypeState& in_state) {
  at::native::ResultTypeState new_state = in_state;
  c10::ScalarType current = scalar;
  if (c10::isFloatingType(scalar)) {
    current = c10::typeMetaToScalarType(at::get_default_dtype());
  }
  new_state.wrappedResult =
      promoteTypesSkipUndefined(in_state.wrappedResult, current);
  return new_state;
}

// Computes a common dtype using type promotion
c10::ScalarType computeCommonDtype(const std::vector<OperandType>& operands) {
  at::native::ResultTypeState state = {};
  for (const auto& op : operands) {
    if (op.value_type == ValueType::Tensor) {
      state = updateResultTypeState(op, state);
    } else {
      state = updateResultTypeState(op.scalar_type, state);
    }
  }
  auto common_dtype = at::native::result_type(state);
  TORCH_INTERNAL_ASSERT(common_dtype != c10::ScalarType::Undefined);
  return common_dtype;
}

c10::ScalarType computeTypes(
    const TypePromotionConfig& config,
    const std::vector<OperandType>& operands) {
  auto common_dtype = c10::ScalarType::Undefined;

  bool has_different_input_dtypes = false;
  for (auto& op : operands) {
    if (op.scalar_type != common_dtype) {
      if (common_dtype == c10::ScalarType::Undefined) {
        common_dtype = op.scalar_type;
      } else {
        has_different_input_dtypes = true;
      }
    }
  }

  // Computes a common dtype, if needed
  if (has_different_input_dtypes) {
    common_dtype = computeCommonDtype(operands);
  }

  // Promotes common dtype to the default float scalar type, if needed
  if (config.promote_integer_inputs_to_float &&
      c10::isIntegralType(common_dtype, /*includeBool=*/true)) {
    common_dtype = c10::get_default_dtype_as_scalartype();
  }
  return common_dtype;
}

OperandType getValueType(TypePtr type) {
  if (auto tensor_type = type->cast<TensorType>()) {
    TORCH_INTERNAL_ASSERT(
        tensor_type->scalarType().has_value(),
        "Missing Scalar Type information");
    // TODO: Type Inference does not propagate Shape Information
    return {
        ValueType::Tensor,
        tensor_type->scalarType().value(),
        tensor_type->dim().has_value() ? tensor_type->dim().value() : 1};
  } else if (auto scalar_type = tryScalarTypeFromJitType(*type)) {
    return {ValueType::Scalar, scalar_type.value()};
  } else {
    return {ValueType::None, c10::ScalarType::Undefined};
  }
}

OperandType getValueType(Val* type) {
  TORCH_INTERNAL_ASSERT(type->getDataType().has_value());

  if (type->isA<TensorView>()) {
    auto tensor_view = type->as<TensorView>();
    return {
        ValueType::Tensor,
        data_type_to_aten(tensor_view->getDataType().value()),
        tensor_view->getMaybeRFactorDomain().size()};
  } else if (type->getDataType().has_value()) {
    return {ValueType::Scalar, data_type_to_aten(type->getDataType().value())};
  } else {
    return {ValueType::None, c10::ScalarType::Undefined};
  }
}

} // namespace

c10::ScalarType computeTypes(
    const TypePromotionConfig& config,
    const std::vector<TypePtr>& operands) {
  std::vector<OperandType> vt_operands;
  vt_operands.reserve(operands.size());
  for (const auto& op : operands) {
    vt_operands.emplace_back(getValueType(op));
  }
  return computeTypes(config, vt_operands);
}

DataType computeTypes(
    const TypePromotionConfig& config,
    const std::vector<Val*>& operands) {
  std::vector<OperandType> vt_operands;
  vt_operands.reserve(operands.size());
  for (const auto& op : operands) {
    vt_operands.push_back(getValueType(op));
  }

  auto common_type = aten_to_data_type(computeTypes(config, vt_operands));

  // Cast FP16 / BFloat16 to Float
  if (common_type == DataType::Half || common_type == DataType::BFloat16) {
    common_type = DataType::Float;
  }

  return common_type;
}

std::vector<Val*> promoteValues(
    const std::vector<Val*>& operands,
    DataType common_type) {
  std::vector<Val*> promoted_operands;
  promoted_operands.reserve(operands.size());
  for (auto op : operands) {
    promoted_operands.push_back(optionalCast(common_type, op));
  }

  TORCH_INTERNAL_ASSERT(operands.size() == promoted_operands.size());
  return promoted_operands;
}

std::vector<Val*> promoteValues(
    const TypePromotionConfig& config,
    const std::vector<Val*>& operands) {
  return promoteValues(operands, computeTypes(config, operands));
}

Val* optionalCast(DataType dtype, Val* v) {
  TORCH_INTERNAL_ASSERT(v->getDataType().has_value());
  // Avoid casting Float/Int/ComplexDouble scalar to any corresponding
  // FloatingPoint/Integral/Double type in fusion. Instead, we cast them
  // directly. The exception is Bool, which is always cast to the desired
  // type.
  const bool kSameDtype = v->getDataType().value() == dtype;
  const bool kIsScalarFloat =
      !v->isA<TensorView>() && isFloatingPointType(dtype);
  const bool kIsScalarInt = !v->isA<TensorView>() && isIntegralType(dtype);
  const bool kIsScalarComplex = !v->isA<TensorView>() && isComplexType(dtype);
  if (kSameDtype ||
      (kIsScalarFloat && isFloatingPointType(v->getDataType().value())) ||
      (kIsScalarInt && isIntegralType(v->getDataType().value())) ||
      (kIsScalarComplex && isComplexType(v->getDataType().value()))) {
    return v;
  } else {
    return castOp(dtype, v);
  }
}

Val* optionalCastStrict(DataType dtype, Val* v) {
  TORCH_INTERNAL_ASSERT(v->getDataType().has_value());
  const bool kSameDtype = v->getDataType().value() == dtype;
  return (kSameDtype) ? v : castOp(dtype, v);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
