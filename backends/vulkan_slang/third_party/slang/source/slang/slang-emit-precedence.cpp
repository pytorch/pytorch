// slang-emit-precedence.cpp
#include "slang-emit-precedence.h"

namespace Slang
{

#define SLANG_OP_INFO_EXPAND(op, name, precedence) \
    {                                              \
        name,                                      \
        kEPrecedence_##precedence##_Left,          \
        kEPrecedence_##precedence##_Right,         \
    },

/* static */ const EmitOpInfo EmitOpInfo::s_infos[int(EmitOp::CountOf)] = {
    SLANG_OP_INFO(SLANG_OP_INFO_EXPAND)};


EmitOp getEmitOpForOp(IROp op)
{
    switch (op)
    {
    case kIROp_Add:
        return EmitOp::Add;
    case kIROp_Sub:
        return EmitOp::Sub;
    case kIROp_Mul:
        return EmitOp::Mul;
    case kIROp_Div:
        return EmitOp::Div;
    case kIROp_IRem:
        return EmitOp::Rem;
    case kIROp_FRem:
        return EmitOp::Rem;

    case kIROp_Lsh:
        return EmitOp::Lsh;
    case kIROp_Rsh:
        return EmitOp::Rsh;

    case kIROp_Eql:
        return EmitOp::Eql;
    case kIROp_Neq:
        return EmitOp::Neq;
    case kIROp_Greater:
        return EmitOp::Greater;
    case kIROp_Less:
        return EmitOp::Less;
    case kIROp_Geq:
        return EmitOp::Geq;
    case kIROp_Leq:
        return EmitOp::Leq;

    case kIROp_BitXor:
        return EmitOp::BitXor;
    case kIROp_BitOr:
        return EmitOp::BitOr;
    case kIROp_BitAnd:
        return EmitOp::BitAnd;

    case kIROp_And:
        return EmitOp::And;
    case kIROp_Or:
        return EmitOp::Or;

    case kIROp_Not:
        return EmitOp::Not;
    case kIROp_Neg:
        return EmitOp::Neg;
    case kIROp_BitNot:
        return EmitOp::BitNot;
    }

    return EmitOp::None;
}

} // namespace Slang
