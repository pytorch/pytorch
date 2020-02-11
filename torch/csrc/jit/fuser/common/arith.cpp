#include <torch/csrc/jit/fuser/common/arith.h>
#include <torch/csrc/jit/fuser/common/type.h>
#include <c10/util/Exception.h>

namespace torch{
namespace jit{
namespace fuser{
TORCH_API Val* new_val(ValType type){
    switch(type){
        case(ValType::Tensor):
            return new Tensor();
        case(ValType::Float):
            return new Float();
        case(ValType::Int):
            return new Int();
    }
    std::runtime_error("Did not recognize out type.");
    return new Int(-1);
}

TORCH_API Val* unary_op(UnaryOpType type, Val* v1){
    Val* out = new_val(v1->getValType().value());
    Statement* expr = new UnaryOp(type, out, v1);
    return out;
}

TORCH_API Val* binary_op(BinaryOpType type, Val* v1, Val* v2){
    ValType out_type = promote_scalar(v1->getValType().value(), v2->getValType().value());
    Val* out = new_val(out_type);
    Statement* expr = new BinaryOp(type, out, v1, v2);
    return out;
}

}}}
