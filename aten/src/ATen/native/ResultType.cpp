#include <ATen/native/ResultType.h>

#include <ATen/ATen.h>

namespace at {

ScalarType resultType(ArrayRef<ScalarTypeSource> inputs) {
  typedef std::function<bool(const ScalarTypeSource)> ParticipatesFunction;

  // Operands of the highest kind determine result type.
  static const std::vector<ParticipatesFunction> kind_participation_order = {
         [](const ScalarTypeSource& s) { return true; }
  // TODO: add actual kind rules when it works for all binary ops
  //     [](const ScalarTypeSource& s) { return isComplexType(s.scalarType()); },
  //     [](const ScalarTypeSource& s) { return isFloatingType(s.scalarType()); },
  //     [](const ScalarTypeSource& s) { return isIntegralType(s.scalarType()); },
  };

  // Priority amongst the operands of the same kind.
  static const std::vector<ParticipatesFunction> priority_participation_order =
      {
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
}

} // namespace at
