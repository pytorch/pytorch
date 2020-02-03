#include <torch/csrc/jit/fuser/common/arith.h>
#include <torch/csrc/jit/fuser/common/type.h>
#include <c10/util/Exception.h>

namespace torch{
namespace jit{
namespace fuser{
//Return new value of type that v1 and v2 promotes to
TORCH_API Val* promote_new(Val *v1, Val* v2){
    TORCH_CHECK(v1->isVal() && v2->isVal());
    ValType out_type = promote_scalar(v1->getValType().value(), v2->getValType().value());
    switch(out_type){
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

TORCH_API Val* add(Val* v1, Val* v2){
    Val* out = promote_new(v1, v2);
    Statement* expr = new Add(out, v1, v2);
    return out;
}

}}}
