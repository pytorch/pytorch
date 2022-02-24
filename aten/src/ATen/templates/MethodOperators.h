#pragma once

// ${generated_comment}

#ifdef TORCH_ASSERT_NO_OPERATORS
#error This change adds a dependency on native_functions.yaml,             \
  meaning the file will need to be re-compiled every time an operator      \
  is changed or added. Consider if your change would be better placed in   \
  another file, or if a more specific header might achieve the same goal.  \
  See NOTE: [Tensor vs. TensorBase]
#endif

// Forward declarations of any types needed in the operator signatures.
// We can't directly include these classes because it will cause circular include dependencies.
// This file is included by TensorBody.h, which defines the Tensor class.
namespace c10 {

template<typename T>
class optional;
template<typename T>
class List;
template<typename T>
class IList;
class Stream;
class Scalar;
struct Storage;
struct TensorOptions;

}

namespace at {

class OptionalTensorRef;
class Tensor;
struct Dimname;
struct Generator;
using TensorList = c10::ArrayRef<Tensor>;
using ITensorList = c10::IList<Tensor>;
using IOptTensorRefList = IList<OptionalTensorRef>;
using DimnameList = c10::ArrayRef<Dimname>;
using c10::Stream;
using c10::Storage;
using c10::QScheme;
using c10::Scalar;
using c10::TensorOptions;

}

${MethodOperators_includes}

namespace at {
namespace _ops {
${MethodOperators_declarations}
} // namespace _ops
} // namespace at
