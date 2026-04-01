// slang-emit-precedence.h
#ifndef SLANG_EMIT_PRECEDENCE_H_INCLUDED
#define SLANG_EMIT_PRECEDENCE_H_INCLUDED

#include "../core/slang-basic.h"
#include "slang-ir.h"

namespace Slang
{

// Macros for setting up precedence
#define SLANG_PRECEDENCE_LEFT(NAME) kEPrecedence_##NAME##_Left, kEPrecedence_##NAME##_Right,

#define SLANG_PRECEDENCE_RIGHT(NAME) kEPrecedence_##NAME##_Right, kEPrecedence_##NAME##_Left,

#define SLANG_PRECEDENCE_NON_ASSOC(NAME) \
    kEPrecedence_##NAME##_Left, kEPrecedence_##NAME##_Right = kEPrecedence_##NAME##_Left,

#define SLANG_PRECEDENCE_EXPAND(NAME, ASSOC) SLANG_PRECEDENCE_##ASSOC(NAME)

// x macro of precedence of types in order.
// Used because in header, need to prefix macros to avoid clashes, and this style allows for
// prefixing without additional clutter
// clang-format off
#define SLANG_PRECEDENCE(x) \
    x(None,         NON_ASSOC) \
    x(Comma,        LEFT) \
    \
    x(General,      NON_ASSOC) \
    \
    x(Assign,       RIGHT) \
    \
    x(Conditional,  RIGHT) \
    \
    x(Or,           LEFT) \
    x(And,          LEFT) \
    x(BitOr,        LEFT) \
    x(BitXor,       LEFT) \
    x(BitAnd,       LEFT) \
    \
    x(Equality,     LEFT) \
    x(Relational,   LEFT) \
    x(Shift,        LEFT) \
    x(Additive,     LEFT) \
    x(Multiplicative, LEFT) \
    x(Prefix,       RIGHT) \
    x(Postfix,      LEFT) \
    x(Atomic,       NON_ASSOC)
// clang-format on

// Precedence enum produced from the SLANG_PRECEDENCE macro
enum EPrecedence
{
    SLANG_PRECEDENCE(SLANG_PRECEDENCE_EXPAND)
};

// Macro for define OpInfo and an associated enum type. Order or macro parameters is
// Op, OpName, Precedence
// clang-format off
#define SLANG_OP_INFO(x) \
    x(None, "", None) \
    \
    x(Comma, ",", Comma) \
    \
    x(General, "", General) \
    \
    x(Assign, "=", Assign) \
    x(AddAssign, "+=", Assign) \
    x(SubAssign, "-=", Assign) \
    x(MulAssign, "*=", Assign) \
    x(DivAssign, "/=", Assign) \
    x(ModAssign, "%=", Assign) \
    x(LshAssign, "<<=", Assign) \
    x(RshAssign, ">>=", Assign) \
    x(OrAssign, "|=", Assign) \
    x(AndAssign, "&=", Assign) \
    x(XorAssign, "^=", Assign) \
    \
    x(Conditional, "?:", Conditional) \
    \
    x(Or, "||", Or) \
    x(And, "&&", And) \
    x(BitOr, "|", BitOr) \
    x(BitXor, "^", BitXor) \
    x(BitAnd, "&", BitAnd) \
    \
    x(Eql, "==", Equality) \
    x(Neq, "!=", Equality) \
    \
    x(Less, "<", Relational) \
    x(Greater, ">", Relational) \
    x(Leq, "<=", Relational) \
    x(Geq, ">=", Relational) \
    \
    x(Lsh, "<<", Shift) \
    x(Rsh, ">>", Shift) \
    \
    x(Add, "+", Additive) \
    x(Sub, "-", Additive) \
    \
    x(Mul, "*", Multiplicative) \
    x(Div, "/", Multiplicative) \
    x(Rem, "%", Multiplicative) \
    \
    x(Prefix, "", Prefix) \
    x(Postfix, "", Postfix) \
    x(Atomic, "", Atomic) \
    \
    x(Not, "!", Prefix) \
    x(Neg, "-", Prefix) \
    x(BitNot, "~", Prefix)
// clang-format on

#define SLANG_OP_INFO_ENUM(op, name, precedence) op,

enum class EmitOp
{
    SLANG_OP_INFO(SLANG_OP_INFO_ENUM) CountOf,
};

// Info on an op for emit purposes
struct EmitOpInfo
{
    SLANG_FORCE_INLINE static const EmitOpInfo& get(EmitOp inOp) { return s_infos[int(inOp)]; }

    char const* op;
    EPrecedence leftPrecedence;
    EPrecedence rightPrecedence;

    static const EmitOpInfo s_infos[int(EmitOp::CountOf)];
};

SLANG_FORCE_INLINE const EmitOpInfo& getInfo(EmitOp op)
{
    return EmitOpInfo::s_infos[int(op)];
}

SLANG_INLINE EmitOpInfo leftSide(EmitOpInfo const& outerPrec, EmitOpInfo const& prec)
{
    EmitOpInfo result;
    result.op = nullptr;
    result.leftPrecedence = outerPrec.leftPrecedence;
    result.rightPrecedence = prec.leftPrecedence;
    return result;
}

SLANG_INLINE EmitOpInfo rightSide(EmitOpInfo const& outerPrec, EmitOpInfo const& prec)
{
    EmitOpInfo result;
    result.op = nullptr;
    result.leftPrecedence = prec.rightPrecedence;
    result.rightPrecedence = outerPrec.rightPrecedence;
    return result;
}

EmitOp getEmitOpForOp(IROp op);

// Precedence macros no longer needed
#undef SLANG_PRECEDENCE_EXPAND
#undef SLANG_PRECEDENCE_NON_ASSOC
#undef SLANG_PRECEDENCE_RIGHT
#undef SLANG_PRECEDENCE_LEFT

} // namespace Slang
#endif
