#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/mobile/module.h>

namespace torch {
namespace jit {

// Compares 2 mobile::Module. Comparison is done as follows:
// 1. _ivalue() returned by both should be equal according to ivalueEquals below
// 2. all functions with same name shall have same instructions and constants
// 3. all functions in lhs exists in rhs.
TORCH_API bool moduleEquals(
    const mobile::Module& lhs,
    const mobile::Module& rhs);

// This is a function used in unittests to see if 2 IValue are the same.
// If print is true; then it will print out where the ivalue differs.
// Behavior of this function is different from IValue::operator== in the
// following parts:
// 1. Tensors are compared with allclose and returns bool (instead of bool
// tensor)
// 2. Therefore, comparing List[Tensor] or deeply nested tensor works
// 3. 2 Capsules compares to true: this is because we intent to use this to
// compare 2 IValue's after
//    saving and loading.
TORCH_API bool ivalueEquals(
    const IValue& lhs,
    const IValue& rhs,
    bool print,
    int print_indent = 0);

} // namespace jit
} // namespace torch
