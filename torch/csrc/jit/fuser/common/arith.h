#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/fuser/common/ir.h>

#include <torch/csrc/jit/fuser/common/type.h>
#include <c10/util/Exception.h>

/*
 * Creating an Expr node returns the node that was created. This is useful
 * if you already have things connected and are directly modifying Exprs.
 * However, creating Expr nodes directly requires you to instantiate all
 * intermediate values. Arith is intended to cover all Exprs, however,
 * its goal is to return a value that is the output of the Expr.
 */ 

namespace torch{
namespace jit{
namespace fuser{

//Return new value of type that v1 and v2 promotes to
TORCH_API const Val* new_val(ValType type);

TORCH_API const Val* unary_op(UnaryOpType type, const Val* v1);
TORCH_API const Val* binary_op(BinaryOpType type, const Val* v1, const Val* v2);

}}}
