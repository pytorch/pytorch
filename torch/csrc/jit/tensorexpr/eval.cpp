#include "torch/csrc/jit/tensorexpr/eval.h"

namespace torch {
namespace jit {
namespace tensorexpr {

DEFINE_TRIGGER(simple_ir_eval_executed);

RegisterCodeGen<SimpleIREvaluator> reg("simple_ir_eval");

} // namespace tensorexpr
} // namespace jit
} // namespace torch
