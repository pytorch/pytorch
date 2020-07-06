#include <torch/csrc/jit/codegen/cuda/utils.h>

#include <c10/util/Exception.h>

#include <torch/csrc/jit/codegen/cuda/tensor_meta.h>

#include <algorithm>
#include <ostream>

namespace torch {
namespace jit {
namespace fuser {

/*
 * Functions for printing ATen IR
 */

void printScalar(std::ostream& stream, const Value* const value) {
  if (value->node()->kind() == prim::Constant) {
    stream << "Const Scalar: ";
  } else {
    stream << "Scalar: ";
  }

  if (value->type() == FloatType::get()) {
    stream << "float ";
    const float val = value->node()->f(attr::value);
    stream << val;
  } else if (value->type() == IntType::get()) {
    stream << "int ";
    const int val = value->node()->i(attr::value);
    stream << val;
  } else {
    stream << "unknown";
  }
  stream << std::endl;
}

// Note: innermost dimension is at nDims - 1 (when nDims > 0)
void printStrides(
    std::ostream& stream,
    const c10::VaryingShape<int64_t>& strides) {
  stream << "Strides=(";
  for (size_t i = 0; i < *(strides.size()); ++i) {
    stream << *(strides[i]);
    if (i != *(strides.size()) - 1) {
      stream << ", ";
    } else {
      stream << ")";
    }
  }
}

void printSizes(std::ostream& stream, const c10::VaryingShape<int64_t>& sizes) {
  stream << "Sizes=(";
  for (size_t i = 0; i < *(sizes.size()); ++i) {
    stream << *(sizes[i]);
    if (i != *(sizes.size()) - 1) {
      stream << ", ";
    } else {
      stream << ")";
    }
  }
}

void printCompleteTensor(
    std::ostream& stream,
    const std::shared_ptr<c10::TensorType>& tensor) {
  stream << "Complete Tensor: ";
  stream << *(tensor->device()) << " ";
  stream << *(tensor->scalarType()) << " ";
  stream << "nDims: " << *(tensor->dim()) << " ";
  stream << std::endl;
  printSizes(stream, tensor->sizes());
  stream << ", ";
  printStrides(stream, tensor->strides());
  stream << std::endl;
}

void printValue(std::ostream& stream, const Value* const value) {
  if (value->isCompleteTensor()) {
    printCompleteTensor(stream, value->type()->expect<TensorType>());
  } else if (value->type()->isSubtypeOf(NumberType::get())) {
    printScalar(stream, value);
  } else {
    stream << "Request to print unknown value" << std::endl;
  }
}

/*
 * Functions for acquiring devices and device types from ATen IR nodes
 */

c10::Device getFusionDevice(const Node* const fusion) {
  const std::shared_ptr<c10::TensorType> out_tensor =
      fusion->outputs()[0]->type()->expect<TensorType>();
  return *(out_tensor->device());
}

c10::DeviceType getFusionDeviceType(const Node* const node) {
  return getFusionDevice(node).type();
}

/*
 * Functions for obtaining parts of complete tensors
 */

std::vector<int64_t> extractStrides(
    const std::shared_ptr<c10::TensorType>& tensor) {
  const c10::VaryingShape<int64_t>& strides = tensor->strides();
  const auto size = *(strides.size());
  std::vector<int64_t> extracted_strides;

  for (auto i = decltype(size){0}; i < size; ++i) {
    extracted_strides.push_back(*(strides[i]));
  }

  return extracted_strides;
}

std::vector<int64_t> extractSizes(
    const std::shared_ptr<c10::TensorType>& tensor) {
  const c10::VaryingShape<int64_t>& sizes = tensor->sizes();
  const auto size = *(sizes.size());
  std::vector<int64_t> extracted_sizes;

  for (auto i = decltype(size){0}; i < size; ++i) {
    extracted_sizes.push_back(*(sizes[i]));
  }

  return extracted_sizes;
}

c10::DeviceType getDeviceType(const std::shared_ptr<c10::TensorType>& tensor) {
  return (*(tensor->device())).type();
}

size_t getRank(const std::shared_ptr<c10::TensorType>& tensor) {
  return *(tensor->dim());
}

size_t getNumel(const std::shared_ptr<c10::TensorType>& tensor) {
  return *(tensor->numel());
}

/*
 * Functions for working with scalar Values
 */

bool isScalar(const Value* const value) {
  return value->type()->isSubtypeOf(NumberType::get());
}

c10::optional<float> getFloat(const Value* const value) {
  if (value->type() == FloatType::get()) {
    return value->node()->f(attr::value);
  }

  return c10::nullopt;
}

c10::optional<int> getInt(const Value* const value) {
  if (value->type() == IntType::get()) {
    return value->node()->i(attr::value);
  }

  return c10::nullopt;
}

float getAsFloat(const Value* const value) {
  if (value->type() == FloatType::get()) {
    return value->node()->f(attr::value);
  }
  if (value->type() == IntType::get()) {
    return static_cast<float>(value->node()->i(attr::value));
  }

  TORCH_CHECK(false, "getAsFloat() found unknown scalar type!");
}

/*
 * Functions for comparing complete tensors
 */

bool haveSameDevice(
    const std::shared_ptr<c10::TensorType>& lhs,
    const std::shared_ptr<c10::TensorType>& rhs) {
  const auto lhs_device = *(lhs->device());
  const auto rhs_device = *(rhs->device());
  return (lhs_device == rhs_device);
}

bool haveSameScalarType(
    const std::shared_ptr<c10::TensorType>& lhs,
    const std::shared_ptr<c10::TensorType>& rhs) {
  const auto lhs_scalar_type = *(lhs->scalarType());
  const auto rhs_scalar_type = *(rhs->scalarType());
  return (lhs_scalar_type == rhs_scalar_type);
}

bool haveSameSizes(
    const std::shared_ptr<c10::TensorType>& lhs,
    const std::shared_ptr<c10::TensorType>& rhs) {
  const auto& lhs_sizes = lhs->sizes();
  const auto& rhs_sizes = rhs->sizes();

  if (*(lhs_sizes.size()) != *(rhs_sizes.size())) {
    return false;
  }

  for (size_t i = 0; i < *(lhs_sizes.size()); ++i) {
    if (*(lhs_sizes[i]) != *(rhs_sizes[i])) {
      return false;
    }
  }

  return true;
}

bool haveSameStrides(
    const std::shared_ptr<c10::TensorType>& lhs,
    const std::shared_ptr<c10::TensorType>& rhs) {
  const auto& lhs_strides = lhs->strides();
  const auto& strides = rhs->strides();

  if (*(lhs_strides.size()) != *(strides.size())) {
    return false;
  }

  for (size_t i = 0; i < *(lhs_strides.size()); ++i) {
    if (*(lhs_strides[i]) != *(strides[i])) {
      return false;
    }
  }

  return true;
}

bool haveSameShape(
    const std::shared_ptr<c10::TensorType>& lhs,
    const std::shared_ptr<c10::TensorType>& rhs) {
  return (
      haveSameDevice(lhs, rhs) && haveSameScalarType(lhs, rhs) &&
      haveSameSizes(lhs, rhs) && haveSameStrides(lhs, rhs));
}

} // namespace fuser
} // namespace jit
} // namespace torch
