#include <ATen/native/ResultType.h>

#include <ATen/ATen.h>

namespace at {

namespace {

template <typename F>
static std::tuple<ScalarType, Backend> compute_result_type(TensorList tensors, const F& predicate) {
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

ScalarType resultType(ArrayRef<ScalarTypeSource> inputs) {
    // Check first that this is not a mixed backend operation.
    auto backend = Backend::Undefined;
    for (auto& input : inputs) {
      if (input.isTensor()) {
        if (backend == Backend::Undefined) {
          backend = input.backend();
        } else if (backend != input.backend()) {
          AT_ERROR(
              "Cannot run operations between backends ",
              backend, " and ", input.backend());
        }
      }
    }

    typedef std::function<bool(const ScalarTypeSource)> ParticipatesFunction;

    // Operands of the highest kind determine result type.
    static const std::vector<ParticipatesFunction> kind_participation_order = {
      [](const ScalarTypeSource& s) { return isComplexType(s.scalarType()); },
      [](const ScalarTypeSource& s) { return isFloatingType(s.scalarType()); },
      [](const ScalarTypeSource& s) { return isIntegralType(s.scalarType()); },
    };

    // Priority amongst the operands of the same kind.
    static const std::vector<ParticipatesFunction> priority_participation_order = {
      &ScalarTypeSource::isScalarType,
      &ScalarTypeSource::isNonZeroDimTensor,
      &ScalarTypeSource::isZeroDimTensor,
      &ScalarTypeSource::isScalar,
    };

    for (auto& kind_participation : kind_participation_order) {
      for (auto& priority_participation : priority_participation_order) {
        auto result_type = ScalarType::Undefined;
        for (auto& input : inputs) {
          if (kind_participation(input) && priority_participation(input)) {
            if (result_type == ScalarType::Undefined) {
              result_type = input.scalarType();
            } else {
              result_type = promoteTypes(result_type, input.scalarType());
            }
          }
        }
        if (result_type != ScalarType::Undefined) {
          return result_type;
        }
      }
    }
    AT_ERROR("Cannot determine result type.");
    return ScalarType::Undefined;
}

}  // namespace at
