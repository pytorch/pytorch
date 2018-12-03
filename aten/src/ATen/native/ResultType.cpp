#include <ATen/native/ResultType.h>

#include <ATen/ATen.h>

namespace at {

namespace {

template <typename F>
static std::tuple<ScalarType, Backend>
compute_result_type(TensorList tensors, const F& predicate) {
  auto result_type = ScalarType::Undefined;
  auto backend = Backend::Undefined;
  for (auto& tensor : tensors) {
    if (!tensor.defined()) continue;
    if (!predicate(tensor)) continue;
    auto dtype = tensor.type().scalarType();;
    result_type = (result_type == ScalarType::Undefined
        ? dtype
        : promoteTypes(result_type, dtype));
    if (backend == Backend::Undefined) {
      backend = tensor.type().backend();
    } else if (backend != tensor.type().backend()) {
      AT_ERROR(
          "Cannot run operations between backends ",
          backend,
          " and ",
          tensor.type().backend());
    }
  }
  return std::make_tuple(result_type, backend);
}

} // anonymous namespace

Type& resultType(TensorList tensors) {
  auto result_type = ScalarType::Undefined;
  auto backend = Backend::Undefined;
  std::tie(result_type, backend) = compute_result_type(tensors, [](const Tensor& t) {
    return t.dim() > 0;
  });
  if (result_type == ScalarType::Undefined) {
    std::tie(result_type, backend) = compute_result_type(tensors, [](const Tensor& t) {
      return !t.unsafeGetTensorImpl()->is_wrapped_number();
    });
  }
  if (result_type == ScalarType::Undefined) {
    std::tie(result_type, backend) = compute_result_type(tensors, [](const Tensor& t) {
      return true;
    });
  }

  AT_ASSERT(result_type != ScalarType::Undefined);
  AT_ASSERT(backend != Backend::Undefined);

  return at::globalContext().getNonVariableType(backend, result_type);
}

Type& resultTypeForOutput(Tensor output, TensorList inputs) {
  SmallVector<Tensor, 4> tensors(inputs.begin(), inputs.end());
  tensors.emplace_back(output);
  Type& result_type = resultType(tensors);
  if (output.defined() && result_type != output.type()) {
    AT_ERROR(
        "Cannot store result of type ",
        result_type.toString(),
        " to an output of type ",
        output.type().toString());
  }
  return result_type;
}

}  // namespace at
