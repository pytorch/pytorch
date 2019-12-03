#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h> // TORCH_API
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/util/Optional.h>

#include <aten/src/ATen/core/jit_type.h>
#include <torch/csrc/jit/ir.h>

namespace torch {
namespace jit {
namespace fuser {

struct TensorMeta;
using RankType = std::vector<int64_t>::size_type;

/*
 * Functions for printing ATen IR
*/

TORCH_API void printScalar(std::ostream& stream, const Value* const value);

TORCH_API void printStrides(
  std::ostream& stream
, const c10::VaryingStrides& strides);

TORCH_API void printSizes(
  std::ostream& stream
, const c10::VaryingShape& sizes);

TORCH_API void printCompleteTensor(
  std::ostream& stream
, const std::shared_ptr<c10::TensorType>& tensor);

TORCH_API void printValue(std::ostream& stream, const Value* const value);

/*
 * Functions for acquiring devices and device types from ATen IR nodes
*/

// Warning: assumes all fusion outputs are complete tensors
TORCH_API c10::Device getFusionDevice(const Node* const node);

TORCH_API c10::DeviceType getFusionDeviceType(const Node* const node);

/*
 * Functions for obtaining parts of complete tensors
*/

TORCH_API c10::DeviceType getDeviceType(
  const std::shared_ptr<c10::TensorType>& tensor);

TORCH_API std::vector<int64_t> extractStrides(
  const std::shared_ptr<c10::TensorType>& tensor);

TORCH_API std::vector<int64_t> extractSizes(
  const std::shared_ptr<c10::TensorType>& tensor);

TORCH_API size_t getRank(
  const std::shared_ptr<c10::TensorType>& tensor);

TORCH_API size_t getNumel(
  const std::shared_ptr<c10::TensorType>& tensor);

/*
 * Functions for working with scalar Values
*/

TORCH_API bool isScalar(const Value* const value);

TORCH_API c10::optional<float> getFloat(const Value* const value);

TORCH_API c10::optional<int> getInt(const Value* const value);

// Returns the scalar as a float, regardless of its scalar type
// TODO: remove me
TORCH_API float getAsFloat(const ::torch::jit::Value* const value);


/*
 * Functions for comparing complete tensors
*/

TORCH_API bool haveSameDevice(
  const std::shared_ptr<c10::TensorType>& lhs
, const std::shared_ptr<c10::TensorType>& rhs
);

TORCH_API bool haveSameScalarType(
  const std::shared_ptr<c10::TensorType>& lhs
, const std::shared_ptr<c10::TensorType>& rhs
);

TORCH_API bool haveSameSizes(
  const std::shared_ptr<c10::TensorType>& lhs
, const std::shared_ptr<c10::TensorType>& rhs
);

TORCH_API bool haveSameStrides(
  const std::shared_ptr<c10::TensorType>& lhs
, const std::shared_ptr<c10::TensorType>& rhs
);

TORCH_API bool haveSameShape(
  const std::shared_ptr<c10::TensorType>& lhs
, const std::shared_ptr<c10::TensorType>& rhs
);

/*
 * Functions for acquiring and working with TensorMetas
*/

// TODO: assumes only one output
TORCH_API std::vector<TensorMeta> getLoopMetas(
  const at::ArrayRef<const Value*> outputs
, const at::ArrayRef<const Value*> inputs
);

TORCH_API void printMeta(
  std::ostream& stream
, const TensorMeta& meta
);

}}} // namespace torch::jit::fuser
