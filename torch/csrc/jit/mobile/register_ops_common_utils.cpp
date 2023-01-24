#include <ATen/core/dynamic_type.h>
#include <ATen/core/type_factory.h>
#include <torch/csrc/jit/mobile/register_ops_common_utils.h>

namespace torch {
namespace jit {

int64_t normalizeIndex(int64_t idx, int64_t list_size) {
  if (idx < 0) {
    // Handle negative indexing
    idx = list_size + idx;
  }
  return idx;
}

IValue tensorToListRecursive(
    char* data,
    int64_t cur_dim,
    int64_t num_tensor_dims,
    at::TypePtr ty,
    at::ScalarType scalar_ty,
    at::IntArrayRef sizes,
    at::IntArrayRef strides,
    size_t element_size) {
  // If ty is a ListType, get the element type.
  if (auto list_type = ty->cast<at::ListType>()) {
    ty = list_type->getElementType();
  } else {
    // If the output type is a scalar, read and push one scalar of
    // the right type onto the stack.
    if (ty == at::IntType::get()) {
      int64_t scalar = *(int64_t*)data;
      return IValue(scalar);
    } else if (ty == at::FloatType::get()) {
      TORCH_INTERNAL_ASSERT(
          scalar_ty == at::ScalarType::Float ||
              scalar_ty == at::ScalarType::Double,
          "Unexpected scalar type for Tensor");
      double scalar =
          scalar_ty == at::ScalarType::Float ? *(float*)data : *(double*)data;
      return IValue(scalar);
    } else if (ty == at::ComplexType::get()) {
      TORCH_INTERNAL_ASSERT(
          scalar_ty == at::ScalarType::ComplexFloat ||
              scalar_ty == at::ScalarType::ComplexDouble,
          "Unexpected scalar type for Tensor");
      c10::complex<double> scalar = scalar_ty == at::ScalarType::ComplexFloat
          ? *(c10::complex<float>*)data
          : *(c10::complex<double>*)data;
      return IValue(scalar);
    } else if (ty == at::BoolType::get()) {
      bool scalar = *(bool*)data;
      return IValue(scalar);
    } else {
      TORCH_CHECK(
          false,
          ty->repr_str(),
          " is not one of the supported types for tolist: int, float, bool");
    }
  }

  // Make the result list consisting of elements of type ty. Since this
  // invocation is processing dimension cur_dim, there will be sizes[cur_dim]
  // output elements.
  auto result = c10::impl::GenericList(ty);
  result.reserve(sizes[cur_dim]);

  // Since ty was a list type, tensorToListRecursive needs to be called
  // recursively on each slice of the tensor in the current dimension.
  for (int64_t i = 0, e = sizes[cur_dim]; i < e; ++i) {
    auto inner_result = tensorToListRecursive(
        data,
        cur_dim + 1,
        num_tensor_dims,
        ty,
        scalar_ty,
        sizes,
        strides,
        element_size);

    if (inner_result.isList()) {
      result.emplace_back(inner_result.toList());
    } else if (inner_result.isComplexDouble()) {
      result.emplace_back(inner_result.toComplexDouble());
    } else if (inner_result.isDouble()) {
      result.emplace_back(inner_result.toDouble());
    } else if (inner_result.isInt()) {
      result.emplace_back(inner_result.toInt());
    } else if (inner_result.isBool()) {
      result.emplace_back(inner_result.toBool());
    } else {
      TORCH_INTERNAL_ASSERT(
          false && "Unknown return type for tensorToListRecursive");
    }

    data += strides[cur_dim] * element_size;
  }

  return result;
}

} // namespace jit
} // namespace torch
