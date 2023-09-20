#include <ATen/core/TensorBase.h>
#include <algorithm>
#include <vector>
#include <ATen/core/TensorBody.h>
#include <c10/util/OptionalArrayRef.h>

#ifdef AT_PER_OPERATOR_HEADERS
#include <ATen/ops/_assert_tensor_metadata_native.h>
#include <ATen/ops/_functional_assert_tensor_metadata_native.h>
#endif

namespace at {

class Tensor;

namespace native {

static void _assert_equal(const std::string& name, const bool& condition) {
    if (!condition) {
        std::stringstream msg;
        msg << "Tensor " << name << " mismatch!";
        AT_ASSERT(condition, msg.str());
    }
}

static void _assert_match(const c10::SymIntArrayRef& original, const c10::OptionalArrayRef<long int>& compared, const std::string& name) {
    if (compared) {
        auto sym_compared = c10::SymIntArrayRef(
            reinterpret_cast<const c10::SymInt*>(compared.value().data()), compared.value().size());
        _assert_equal(name, original == sym_compared);
    }
}

static void _assert_match(const c10::ScalarType& original, const c10::optional<c10::ScalarType>& compared, const std::string& name) {
    if (compared) {
        _assert_equal(name, original == compared.value());
    }
}

static void _assert_match(const caffe2::TypeMeta& original, const c10::optional<c10::ScalarType>& compared, const std::string& name) {
    if (compared) {
        _assert_equal(name, original == compared);
    }
}

void _assert_tensor_metadata(at::Tensor const& tensor, at::OptionalIntArrayRef sizes, at::OptionalIntArrayRef strides, c10::optional<c10::ScalarType> dtype) {
  _assert_match(tensor.sym_sizes(), sizes, "sizes");
  _assert_match(tensor.sym_strides(), strides, "strides");
  _assert_match(tensor.dtype(), dtype, "dtype");
}

Tensor _functional_assert_tensor_metadata(at::Tensor const& tensor, at::OptionalIntArrayRef sizes, at::OptionalIntArrayRef strides, c10::optional<c10::ScalarType> dtype) {
  _assert_tensor_metadata(tensor, sizes, strides, dtype);
  return tensor.clone();
}


}
}  // namespace at::native
