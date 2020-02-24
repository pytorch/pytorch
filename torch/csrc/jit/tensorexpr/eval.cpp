#include "torch/csrc/jit/tensorexpr/eval.h"

namespace torch {
namespace jit {
namespace tensorexpr {

RegisterCodeGen<SimpleIREvaluator> reg("simple_ir_eval");

} // namespace tensorexpr
} // namespace jit
} // namespace torch
