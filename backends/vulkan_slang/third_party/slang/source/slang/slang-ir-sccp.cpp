// slang-ir-sccp.cpp
#include "slang-ir-sccp.h"

#include "slang-ir-insts.h"
#include "slang-ir.h"

namespace Slang
{


// This file implements the Spare Conditional Constant Propagation (SCCP) optimization.
//
// We will apply the optimization over individual functions, so we will start with
// a context struct for the state that we will share across functions:
//
struct SharedSCCPContext
{
    IRModule* module;
    DiagnosticSink* sink;
};
//
// Next we have a context struct that will be applied for each function (or other
// code-bearing value) that we optimize:
//
struct SCCPContext
{
    SharedSCCPContext* shared;   // shared state across functions
    IRGlobalValueWithCode* code; // the function/code we are optimizing

    // The SCCP algorithm applies abstract interpretation to the code of the
    // function using a "lattice" of values. We can think of a node on the
    // lattice as representing a set of values that a given instruction
    // might take on.
    //
    struct LatticeVal
    {
        // We will use three "flavors" of values on our lattice.
        //
        enum class Flavor
        {
            // The `None` flavor represent an empty set of values, meaning
            // that we've never seen any indication that the instruction
            // produces a (well-defined) value. This could indicate an
            // instruction that does not appear to execute, but it could
            // also indicate an instruction that we know invokes undefined
            // behavior, so we can freely pick a value for it on a whim.
            None,

            // The `Constant` flavor represents an instuction that we
            // have only ever seen produce a single, fixed value. It's
            // `value` field will hold that constant value.
            Constant,

            // The `Any` flavor represents an instruction that might produce
            // different values at runtime, so we go ahead and approximate
            // this as it potentially yielding any value whatsoever. A
            // more precise analysis could use sets or intervals of values,
            // but for SCCP anything that could take on more than 1 value
            // at runtime is assumed to be able to take on *any* value.
            Any,
        };

        // The flavor of this value (`None`, `Constant`, or `Any`)
        Flavor flavor;

        // If this is a `Constant` lattice value, then this field
        // points to the IR instruction that defines the actual constant value.
        // For all other flavors it should be null.
        IRInst* value = nullptr;

        // For convenience, we define `static` factory functions to
        // produce values of each of the flavors.

        static LatticeVal getNone()
        {
            LatticeVal result;
            result.flavor = Flavor::None;
            return result;
        }

        static LatticeVal getAny()
        {
            LatticeVal result;
            result.flavor = Flavor::Any;
            return result;
        }

        static LatticeVal getConstant(IRInst* value)
        {
            LatticeVal result;
            result.flavor = Flavor::Constant;
            result.value = value;
            return result;
        }

        // We also need to be able to test if two lattice
        // values are equal, so that we can avoid updating
        // downstream dependencies if our knowledge about
        // an instruction hasn't actually changed.
        //
        bool operator==(LatticeVal const& that)
        {
            return this->flavor == that.flavor && this->value == that.value;
        }

        bool operator!=(LatticeVal const& that) { return !(*this == that); }
    };

    static bool isEvaluableOpCode(IROp op)
    {
        switch (op)
        {
        case kIROp_IntLit:
        case kIROp_BoolLit:
        case kIROp_FloatLit:
        case kIROp_StringLit:
        case kIROp_Add:
        case kIROp_Sub:
        case kIROp_Mul:
        case kIROp_Div:
        case kIROp_Neg:
        case kIROp_Not:
        case kIROp_Eql:
        case kIROp_Neq:
        case kIROp_Leq:
        case kIROp_Geq:
        case kIROp_Less:
        case kIROp_IRem:
        case kIROp_FRem:
        case kIROp_Greater:
        case kIROp_Lsh:
        case kIROp_Rsh:
        case kIROp_BitAnd:
        case kIROp_BitOr:
        case kIROp_BitXor:
        case kIROp_BitNot:
        case kIROp_BitCast:
        case kIROp_CastIntToFloat:
        case kIROp_CastFloatToInt:
        case kIROp_IntCast:
        case kIROp_FloatCast:
        case kIROp_Select:
            return true;
        default:
            return false;
        }
    }

    // If we imagine a variable (actually an SSA phi node...) that
    // might be assigned lattice value A at one point in the code,
    // and lattice value B at another point, we need a way to
    // combine these to form our knowledge of the possible value(s)
    // for the variable.
    //
    // In terms of computation on a lattice, we want the "meet"
    // operation, which computes the lower bound on what we know.
    // If we interpret our lattice values as sets, then we are
    // trying to compute the union.
    //
    LatticeVal meet(LatticeVal const& left, LatticeVal const& right)
    {
        // If either value is `None` (the empty set), then the union
        // will be the other value.
        //
        if (left.flavor == LatticeVal::Flavor::None)
            return right;
        if (right.flavor == LatticeVal::Flavor::None)
            return left;

        // If either value is `Any` (the universal set), then
        // the union is also the universal set.
        //
        if (left.flavor == LatticeVal::Flavor::Any)
            return LatticeVal::getAny();
        if (right.flavor == LatticeVal::Flavor::Any)
            return LatticeVal::getAny();

        // At this point we've ruled out the case where either value
        // is `None` *or* `Any`, so we can assume both values are
        // `Constant`s.
        SLANG_ASSERT(left.flavor == LatticeVal::Flavor::Constant);
        //
        SLANG_ASSERT(right.flavor == LatticeVal::Flavor::Constant);

        // If the two lattice values represent the *same* constant value
        // (they are the same singleton set) then the union is that
        // singleton set as well.
        //
        // TODO: This comparison assumes that constants with
        // the same value with be represented with the
        // same instruction, which is not *always*
        // guaranteed in the IR today.
        //
        if (left.value == right.value)
            return left;

        // Otherwise, we have two distinct singleton sets, and their
        // union should be a set with two elements. We can't represent
        // that on the lattice for SCCP, so the proper lower bound
        // is the universal set (`Any`)
        //
        return LatticeVal::getAny();
    }

    // During the execution of the SCCP algorithm, we will track our best
    // "estimate" so far of the set of values each instruction could take
    // on. This amounts to a mapping from IR instructions to lattice values,
    // where any instruction not present in the map is assumed to default
    // to the `None` case (the empty set)
    //
    Dictionary<IRInst*, LatticeVal> mapInstToLatticeVal;

    // Updating the lattice value for an instruction is easy, but we'll
    // use a simple function to make our intention clear.
    //
    void setLatticeVal(IRInst* inst, LatticeVal const& val) { mapInstToLatticeVal[inst] = val; }

    // Querying the lattice value for an instruction isn't *just* a matter
    // of looking it up in the dictionary, because we need to account for
    // cases of lattice values that might come from outside the current
    // function.
    //
    LatticeVal getLatticeVal(IRInst* inst)
    {
        // Instructions that represent constant values should always
        // have a lattice value that reflects this.
        //
        switch (inst->getOp())
        {
        case kIROp_IntLit:
        case kIROp_FloatLit:
        case kIROp_StringLit:
        case kIROp_BoolLit:
            return LatticeVal::getConstant(inst);
            break;

            // TODO: We might want to start having support for constant
            // values of aggregate types (e.g., a `makeArray` or `makeStruct`
            // where all the operands are constant is itself a constant).

        default:
            break;
        }

        // Look up in the dictionary and just return the value we get from it.
        LatticeVal latticeVal;
        if (mapInstToLatticeVal.tryGetValue(inst, latticeVal))
            return latticeVal;

        // If we can't find the value from dictionary, we want to return None if this is a value
        // in the same function as the one we are working with right now. If it is defined
        // elsewhere, we return Any.
        auto parentBlock = as<IRBlock>(inst->getParent());
        bool isProcessingGlobalScope = (code == nullptr);
        if (!parentBlock && isProcessingGlobalScope)
        {
            // We are folding constant in the global scope, continue registering the inst as Any.
        }
        else
        {
            // If we are processing a function and asked for the lattice value of an instruction
            // not contained in the current function, we will treat it as having potentially any
            // value, rather than the default of none.
            //
            if (!parentBlock || parentBlock->getParent() != code)
                return LatticeVal::getAny();
        }

        return LatticeVal::getNone();
    }

    // Along the way we might need to create new IR instructions
    // to represnet new constant values we find, or new control
    // flow instructiosn when we start simplifying things.
    //
    IRBuilder builderStorage;
    IRBuilder* getBuilder() { return &builderStorage; }

    // LatticeVal constant evaluation methods.
#define SLANG_SCCP_RETURN_IF_NONE_OR_ANY(v) \
    switch (v.flavor)                       \
    {                                       \
    case LatticeVal::Flavor::None:          \
        return LatticeVal::getNone();       \
    case LatticeVal::Flavor::Any:           \
        return LatticeVal::getAny();        \
    default:                                \
        break;                              \
    }

    LatticeVal evalCast(IRType* type, LatticeVal v0)
    {
        SLANG_SCCP_RETURN_IF_NONE_OR_ANY(v0)
        auto irConstant = as<IRConstant>(v0.value);

        IRInst* resultVal = nullptr;
        if (type->getOp() == irConstant->getOp())
            return LatticeVal::getConstant(irConstant);

        switch (type->getOp())
        {
        case kIROp_Int8Type:
        case kIROp_Int16Type:
        case kIROp_IntType:
        case kIROp_Int64Type:
        case kIROp_UInt8Type:
        case kIROp_UInt16Type:
        case kIROp_UIntType:
        case kIROp_UInt64Type:
        case kIROp_IntPtrType:
        case kIROp_UIntPtrType:
            switch (irConstant->getOp())
            {
            case kIROp_FloatLit:
                resultVal =
                    getBuilder()->getIntValue(type, (IRIntegerValue)irConstant->value.floatVal);
                break;
            case kIROp_IntLit:
            case kIROp_BoolLit:
                {
                    IRIntegerValue intVal = irConstant->value.intVal;
                    resultVal = getBuilder()->getIntValue(type, (IRIntegerValue)intVal);
                }
                break;
            default:
                return LatticeVal::getAny();
            }
            break;
        case kIROp_FloatType:
        case kIROp_DoubleType:
        case kIROp_HalfType:
            switch (irConstant->getOp())
            {
            case kIROp_FloatLit:
                resultVal = getBuilder()->getFloatValue(
                    type,
                    (IRFloatingPointValue)irConstant->value.floatVal);
                break;
            case kIROp_IntLit:
            case kIROp_BoolLit:
                resultVal = getBuilder()->getFloatValue(
                    type,
                    (IRFloatingPointValue)irConstant->value.intVal);
                break;
            default:
                return LatticeVal::getAny();
            }
            break;
        case kIROp_BoolType:
            switch (irConstant->getOp())
            {
            case kIROp_FloatLit:
                resultVal = getBuilder()->getBoolValue(irConstant->value.floatVal != 0);
                break;
            case kIROp_IntLit:
            case kIROp_BoolLit:
                {
                    resultVal = getBuilder()->getBoolValue(irConstant->value.intVal != 0);
                }
                break;
            default:
                return LatticeVal::getAny();
            }
        }
        if (!resultVal)
            return LatticeVal::getAny();
        return LatticeVal::getConstant(resultVal);
    }

    LatticeVal evalDefaultConstruct(IRType* type)
    {
        IRInst* resultVal = nullptr;
        switch (type->getOp())
        {
        case kIROp_Int8Type:
        case kIROp_Int16Type:
        case kIROp_IntType:
        case kIROp_Int64Type:
        case kIROp_IntPtrType:
        case kIROp_UInt8Type:
        case kIROp_UInt16Type:
        case kIROp_UIntType:
        case kIROp_UInt64Type:
        case kIROp_UIntPtrType:
            resultVal = getBuilder()->getIntValue(type, (IRIntegerValue)0);
            break;

        case kIROp_FloatType:
        case kIROp_DoubleType:
        case kIROp_HalfType:
            resultVal = getBuilder()->getFloatValue(type, (IRFloatingPointValue)0.0);
            break;

        case kIROp_BoolType:
            resultVal = getBuilder()->getBoolValue(false);
            break;
        }
        if (!resultVal)
            return LatticeVal::getAny();
        return LatticeVal::getConstant(resultVal);
    }

    template<typename TIntFunc, typename TFloatFunc>
    LatticeVal evalBinaryImpl(
        IRType* type,
        LatticeVal v0,
        LatticeVal v1,
        const TIntFunc& intFunc,
        const TFloatFunc& floatFunc)
    {
        SLANG_SCCP_RETURN_IF_NONE_OR_ANY(v0)
        auto c0 = as<IRConstant>(v0.value);
        SLANG_SCCP_RETURN_IF_NONE_OR_ANY(v1)
        auto c1 = as<IRConstant>(v1.value);
        IRInst* resultVal = nullptr;
        switch (type->getOp())
        {
        case kIROp_Int8Type:
        case kIROp_Int16Type:
        case kIROp_IntType:
        case kIROp_Int64Type:
        case kIROp_UInt8Type:
        case kIROp_UInt16Type:
        case kIROp_UIntType:
        case kIROp_UInt64Type:
        case kIROp_IntPtrType:
        case kIROp_UIntPtrType:
        case kIROp_BoolType:
            resultVal =
                getBuilder()->getIntValue(type, intFunc(c0->value.intVal, c1->value.intVal));
            break;
        case kIROp_FloatType:
        case kIROp_DoubleType:
        case kIROp_HalfType:
            resultVal = getBuilder()->getFloatValue(
                type,
                floatFunc(c0->value.floatVal, c1->value.floatVal));
            break;
        default:
            break;
        }
        if (!resultVal)
            return LatticeVal::getAny();
        return LatticeVal::getConstant(resultVal);
    }

    template<typename TIntFunc>
    LatticeVal evalBinaryIntImpl(
        IRType* type,
        LatticeVal v0,
        LatticeVal v1,
        const TIntFunc& intFunc)
    {
        SLANG_SCCP_RETURN_IF_NONE_OR_ANY(v0)
        auto c0 = as<IRConstant>(v0.value);
        SLANG_SCCP_RETURN_IF_NONE_OR_ANY(v1)
        auto c1 = as<IRConstant>(v1.value);
        IRInst* resultVal = nullptr;
        switch (type->getOp())
        {
        case kIROp_Int8Type:
        case kIROp_Int16Type:
        case kIROp_IntType:
        case kIROp_Int64Type:
        case kIROp_IntPtrType:
        case kIROp_UInt8Type:
        case kIROp_UInt16Type:
        case kIROp_UIntType:
        case kIROp_UInt64Type:
        case kIROp_UIntPtrType:
        case kIROp_BoolType:
            resultVal =
                getBuilder()->getIntValue(type, intFunc(c0->value.intVal, c1->value.intVal));
            break;
        default:
            break;
        }
        if (!resultVal)
            return LatticeVal::getAny();
        return LatticeVal::getConstant(resultVal);
    }

    template<typename TIntFunc>
    LatticeVal evalUnaryIntImpl(IRType* type, LatticeVal v0, const TIntFunc& intFunc)
    {
        SLANG_SCCP_RETURN_IF_NONE_OR_ANY(v0)
        auto c0 = as<IRConstant>(v0.value);
        IRInst* resultVal = nullptr;
        switch (type->getOp())
        {
        case kIROp_Int8Type:
        case kIROp_Int16Type:
        case kIROp_IntType:
        case kIROp_Int64Type:
        case kIROp_IntPtrType:
        case kIROp_UInt8Type:
        case kIROp_UInt16Type:
        case kIROp_UIntType:
        case kIROp_UInt64Type:
        case kIROp_UIntPtrType:
        case kIROp_BoolType:
            resultVal = getBuilder()->getIntValue(type, intFunc(c0->value.intVal));
            break;
        default:
            break;
        }
        if (!resultVal)
            return LatticeVal::getAny();
        return LatticeVal::getConstant(resultVal);
    }

    template<typename TIntFunc, typename TFloatFunc>
    LatticeVal evalComparisonImpl(
        IRType* type,
        LatticeVal v0,
        LatticeVal v1,
        const TIntFunc& intFunc,
        const TFloatFunc& floatFunc)
    {
        SLANG_SCCP_RETURN_IF_NONE_OR_ANY(v0)
        auto c0 = as<IRConstant>(v0.value);
        SLANG_SCCP_RETURN_IF_NONE_OR_ANY(v1)
        auto c1 = as<IRConstant>(v1.value);
        IRInst* resultVal = nullptr;
        switch (type->getOp())
        {
        case kIROp_Int8Type:
        case kIROp_Int16Type:
        case kIROp_IntType:
        case kIROp_Int64Type:
        case kIROp_IntPtrType:
        case kIROp_UInt8Type:
        case kIROp_UInt16Type:
        case kIROp_UIntType:
        case kIROp_UInt64Type:
        case kIROp_UIntPtrType:
        case kIROp_BoolType:
            resultVal = getBuilder()->getBoolValue(intFunc(c0->value.intVal, c1->value.intVal));
            break;
        case kIROp_FloatType:
        case kIROp_DoubleType:
        case kIROp_HalfType:
            resultVal =
                getBuilder()->getBoolValue(floatFunc(c0->value.floatVal, c1->value.floatVal));
            break;
        default:
            break;
        }
        if (!resultVal)
            return LatticeVal::getAny();
        return LatticeVal::getConstant(resultVal);
    }

    LatticeVal evalAdd(IRType* type, LatticeVal v0, LatticeVal v1)
    {
        return evalBinaryImpl(
            type,
            v0,
            v1,
            [](IRIntegerValue c0, IRIntegerValue c1) { return c0 + c1; },
            [](IRFloatingPointValue c0, IRFloatingPointValue c1) { return c0 + c1; });
    }
    LatticeVal evalSub(IRType* type, LatticeVal v0, LatticeVal v1)
    {
        return evalBinaryImpl(
            type,
            v0,
            v1,
            [](IRIntegerValue c0, IRIntegerValue c1) { return c0 - c1; },
            [](IRFloatingPointValue c0, IRFloatingPointValue c1) { return c0 - c1; });
    }
    LatticeVal evalMul(IRType* type, LatticeVal v0, LatticeVal v1)
    {
        return evalBinaryImpl(
            type,
            v0,
            v1,
            [](IRIntegerValue c0, IRIntegerValue c1) { return c0 * c1; },
            [](IRFloatingPointValue c0, IRFloatingPointValue c1) { return c0 * c1; });
    }
    LatticeVal evalDiv(IRType* type, LatticeVal v0, LatticeVal v1)
    {
        return evalBinaryImpl(
            type,
            v0,
            v1,
            [](IRIntegerValue c0, IRIntegerValue c1) { return c0 / c1; },
            [](IRFloatingPointValue c0, IRFloatingPointValue c1) { return c0 / c1; });
    }
    LatticeVal evalRem(IRType* type, LatticeVal v0, LatticeVal v1)
    {
        return evalBinaryImpl(
            type,
            v0,
            v1,
            [](IRIntegerValue c0, IRIntegerValue c1) { return c0 % c1; },
            [](IRFloatingPointValue c0, IRFloatingPointValue c1) { return fmod(c0, c1); });
    }
    LatticeVal evalEql(IRType* type, LatticeVal v0, LatticeVal v1)
    {
        return evalComparisonImpl(
            type,
            v0,
            v1,
            [](IRIntegerValue c0, IRIntegerValue c1) { return c0 == c1; },
            [](IRFloatingPointValue c0, IRFloatingPointValue c1) { return c0 == c1; });
    }
    LatticeVal evalNeq(IRType* type, LatticeVal v0, LatticeVal v1)
    {
        return evalComparisonImpl(
            type,
            v0,
            v1,
            [](IRIntegerValue c0, IRIntegerValue c1) { return c0 != c1; },
            [](IRFloatingPointValue c0, IRFloatingPointValue c1) { return c0 != c1; });
    }
    LatticeVal evalGeq(IRType* type, LatticeVal v0, LatticeVal v1)
    {
        return evalComparisonImpl(
            type,
            v0,
            v1,
            [](IRIntegerValue c0, IRIntegerValue c1) { return c0 >= c1; },
            [](IRFloatingPointValue c0, IRFloatingPointValue c1) { return c0 >= c1; });
    }
    LatticeVal evalLeq(IRType* type, LatticeVal v0, LatticeVal v1)
    {
        return evalComparisonImpl(
            type,
            v0,
            v1,
            [](IRIntegerValue c0, IRIntegerValue c1) { return c0 <= c1; },
            [](IRFloatingPointValue c0, IRFloatingPointValue c1) { return c0 <= c1; });
    }
    LatticeVal evalGreater(IRType* type, LatticeVal v0, LatticeVal v1)
    {
        return evalComparisonImpl(
            type,
            v0,
            v1,
            [](IRIntegerValue c0, IRIntegerValue c1) { return c0 > c1; },
            [](IRFloatingPointValue c0, IRFloatingPointValue c1) { return c0 > c1; });
    }
    LatticeVal evalLess(IRType* type, LatticeVal v0, LatticeVal v1)
    {
        return evalComparisonImpl(
            type,
            v0,
            v1,
            [](IRIntegerValue c0, IRIntegerValue c1) { return c0 < c1; },
            [](IRFloatingPointValue c0, IRFloatingPointValue c1) { return c0 < c1; });
    }
    LatticeVal evalAnd(IRType* type, LatticeVal v0, LatticeVal v1)
    {
        return evalBinaryIntImpl(
            type,
            v0,
            v1,
            [](IRIntegerValue c0, IRIntegerValue c1) { return c0 != 0 && c1 != 0; });
    }
    LatticeVal evalOr(IRType* type, LatticeVal v0, LatticeVal v1)
    {
        return evalBinaryIntImpl(
            type,
            v0,
            v1,
            [](IRIntegerValue c0, IRIntegerValue c1) { return c0 != 0 || c1 != 0; });
    }
    LatticeVal evalNot(IRType* type, LatticeVal v0)
    {
        return evalUnaryIntImpl(type, v0, [](IRIntegerValue c0) { return c0 == 0; });
    }
    LatticeVal evalBitAnd(IRType* type, LatticeVal v0, LatticeVal v1)
    {
        return evalBinaryIntImpl(
            type,
            v0,
            v1,
            [](IRIntegerValue c0, IRIntegerValue c1) { return c0 & c1; });
    }
    LatticeVal evalBitOr(IRType* type, LatticeVal v0, LatticeVal v1)
    {
        return evalBinaryIntImpl(
            type,
            v0,
            v1,
            [](IRIntegerValue c0, IRIntegerValue c1) { return c0 | c1; });
    }
    LatticeVal evalBitNot(IRType* type, LatticeVal v0)
    {
        return evalUnaryIntImpl(type, v0, [](IRIntegerValue c0) { return ~c0; });
    }
    LatticeVal evalBitXor(IRType* type, LatticeVal v0, LatticeVal v1)
    {
        return evalBinaryIntImpl(
            type,
            v0,
            v1,
            [](IRIntegerValue c0, IRIntegerValue c1) { return c0 ^ c1; });
    }
    LatticeVal evalLsh(IRType* type, LatticeVal v0, LatticeVal v1)
    {
        IntInfo info = getIntTypeInfo(type);
        if (info.isSigned == false)
        {
            return evalBinaryIntImpl(
                type,
                v0,
                v1,
                [](IRUnsignedIntegerValue c0, IRUnsignedIntegerValue c1) { return c0 << c1; });
        }
        return evalBinaryIntImpl(
            type,
            v0,
            v1,
            [](IRIntegerValue c0, IRIntegerValue c1) { return c0 << c1; });
    }
    LatticeVal evalRsh(IRType* type, LatticeVal v0, LatticeVal v1)
    {
        IntInfo info = getIntTypeInfo(type);
        if (info.isSigned == false)
        {
            return evalBinaryIntImpl(
                type,
                v0,
                v1,
                [](IRUnsignedIntegerValue c0, IRUnsignedIntegerValue c1) { return c0 >> c1; });
        }
        return evalBinaryIntImpl(
            type,
            v0,
            v1,
            [](IRIntegerValue c0, IRIntegerValue c1) { return c0 >> c1; });
    }
    LatticeVal evalNeg(IRType* type, LatticeVal v0)
    {
        SLANG_SCCP_RETURN_IF_NONE_OR_ANY(v0)
        auto c0 = as<IRConstant>(v0.value);
        IRInst* resultVal = nullptr;
        switch (type->getOp())
        {
        case kIROp_Int8Type:
        case kIROp_Int16Type:
        case kIROp_IntType:
        case kIROp_Int64Type:
        case kIROp_IntPtrType:
        case kIROp_UInt8Type:
        case kIROp_UInt16Type:
        case kIROp_UIntType:
        case kIROp_UInt64Type:
        case kIROp_UIntPtrType:
            resultVal = getBuilder()->getIntValue(type, -c0->value.intVal);
            break;
        case kIROp_FloatType:
        case kIROp_DoubleType:
        case kIROp_HalfType:
            resultVal = getBuilder()->getFloatValue(type, -c0->value.floatVal);
            break;
        default:
            break;
        }
        if (!resultVal)
            return LatticeVal::getAny();
        return LatticeVal::getConstant(resultVal);
    }

    LatticeVal evalBitCast(IRType* type, LatticeVal v0)
    {
        SLANG_SCCP_RETURN_IF_NONE_OR_ANY(v0)
        auto c0 = as<IRConstant>(v0.value);

        uint64_t sourceValueBits = 0;
        switch (c0->getDataType()->getOp())
        {
        case kIROp_FloatType:
            {
                float fval = (float)c0->value.floatVal;
                memcpy(&sourceValueBits, &fval, sizeof(fval));
                break;
            }
        case kIROp_DoubleType:
            {
                double dval = c0->value.floatVal;
                memcpy(&sourceValueBits, &dval, sizeof(dval));
                break;
            }
        case kIROp_BoolType:
            {
                sourceValueBits = c0->value.intVal;
                break;
            }
        default:
            if (isIntegralType(c0->getDataType()))
            {
                sourceValueBits = c0->value.intVal;
            }
            else
            {
                return LatticeVal::getAny();
            }
            break;
        }

        IRInst* resultVal = nullptr;
        switch (type->getOp())
        {
        case kIROp_Int64Type:
        case kIROp_UInt64Type:
#if SLANG_PTR_IS_64
        case kIROp_IntPtrType:
        case kIROp_UIntPtrType:
#endif
            resultVal = getBuilder()->getIntValue(type, sourceValueBits);
            break;
        case kIROp_IntType:
        case kIROp_UIntType:
#if SLANG_PTR_IS_32
        case kIROp_IntPtrType:
        case kIROp_UIntPtrType:
#endif
            resultVal = getBuilder()->getIntValue(type, (uint32_t)sourceValueBits);
            break;
        case kIROp_FloatType:
            {
                uint32_t val = (uint32_t)sourceValueBits;
                float floatVal = IntAsFloat((int)val);
                resultVal = getBuilder()->getFloatValue(type, floatVal);
            }
            break;
        case kIROp_DoubleType:
            resultVal = getBuilder()->getFloatValue(type, Int64AsDouble(sourceValueBits));
            break;
        default:
            break;
        }
        if (!resultVal)
            return LatticeVal::getAny();
        return LatticeVal::getConstant(resultVal);
    }

    LatticeVal evalSelect(LatticeVal v0, LatticeVal v1, LatticeVal v2)
    {
        SLANG_SCCP_RETURN_IF_NONE_OR_ANY(v0)
        auto c0 = as<IRConstant>(v0.value);
        return c0->value.intVal != 0 ? v1 : v2;
    }

    // In order to perform constant folding, we need to be able to
    // interpret an instruction over the lattice values.
    //
    LatticeVal interpretOverLattice(IRInst* inst)
    {
        SLANG_UNUSED(inst);

        // Certain instruction always produce constants, and we
        // want to special-case them here.
        switch (inst->getOp())
        {
        case kIROp_IntLit:
        case kIROp_FloatLit:
        case kIROp_StringLit:
        case kIROp_BoolLit:
            return LatticeVal::getConstant(inst);

        // We might also want to special-case certain
        // instructions where we shouldn't bother trying to
        // constant-fold them and should just default to the
        // `Any` value right away.
        case kIROp_Call:
        case kIROp_ByteAddressBufferLoad:
        case kIROp_ByteAddressBufferStore:
        case kIROp_Alloca:
        case kIROp_Store:
        case kIROp_Load:
            return LatticeVal::getAny();
        default:
            break;
        }

        // TODO: We should now look up the lattice values for
        // the operands of the instruction.
        //
        // If all of the operands have `Constant` lattice values,
        // then we can potential execute the operation directly
        // on those constant values, create a fresh `IRConstant`,
        // and return a `Constant` lattice value for it. This
        // would allow us to achieve true constant folding here.
        //
        // Textbook discussions of SCCP often point out that it
        // is also possible to perform certain algebraic simplifications
        // here, such as evaluating a multiply by a `Constant` zero
        // to zero.
        //
        // As a default, if any operand has the `Any` value
        // then the result of the operation should be treated as
        // `Any`. There are exceptions to this, however, with the
        // multiply-by-zero example being an important example.
        // If we had previously decided that (Any * None) -> Any
        // but then we refine our estimates and have (Any * Constant(0)) -> Constant(0)
        // then we have violated the monotonicity rules for how
        // our values move through the lattice, and we may break
        // the convergence guarantees of the analysis.
        //
        // When we have a mix of `None` and `Constant` operands,
        // then the `None` values imply that our operation is using
        // uninitialized data or the results of undefined behavior.
        // We could try to propagate the `None` through, and allow
        // the compiler to speculatively assume that the operation
        // produces whatever value we find convenient. Alternatively,
        // we can be less aggressive and treat an operation with
        // `None` inputs as producing `Any` to make sure we don't
        // optimize the code based on non-obvious assumptions.
        //
        // For now we implement only basic folding operations for
        // scalar values.
        if (!as<IRBasicType>(inst->getDataType()))
            return LatticeVal::getAny();

        switch (inst->getOp())
        {
        case kIROp_IntCast:
        case kIROp_FloatCast:
        case kIROp_CastIntToFloat:
        case kIROp_CastFloatToInt:
            switch (inst->getOperandCount())
            {
            case 1:
                return evalCast(inst->getDataType(), getLatticeVal(inst->getOperand(0)));

            default:
                return LatticeVal::getAny();
            }
        case kIROp_DefaultConstruct:
            return evalDefaultConstruct(inst->getDataType());
        case kIROp_Add:
            return evalAdd(
                inst->getDataType(),
                getLatticeVal(inst->getOperand(0)),
                getLatticeVal(inst->getOperand(1)));
        case kIROp_Sub:
            return evalSub(
                inst->getDataType(),
                getLatticeVal(inst->getOperand(0)),
                getLatticeVal(inst->getOperand(1)));
        case kIROp_Mul:
            return evalMul(
                inst->getDataType(),
                getLatticeVal(inst->getOperand(0)),
                getLatticeVal(inst->getOperand(1)));
        case kIROp_Div:
            {
                // Detect divide by zero error.
                auto divisor = getLatticeVal(inst->getOperand(1));
                if (divisor.flavor == LatticeVal::Flavor::Constant)
                {
                    if (isIntegralType(divisor.value->getDataType()))
                    {
                        auto c = as<IRConstant>(divisor.value);
                        if (c->value.intVal == 0)
                        {
                            if (shared->sink)
                                shared->sink->diagnose(inst->sourceLoc, Diagnostics::divideByZero);
                            return LatticeVal::getAny();
                        }
                    }
                }
                return evalDiv(inst->getDataType(), getLatticeVal(inst->getOperand(0)), divisor);
            }
        case kIROp_FRem:
        case kIROp_IRem:
            {
                // Detect divide by zero error.
                auto divisor = getLatticeVal(inst->getOperand(1));
                if (divisor.flavor == LatticeVal::Flavor::Constant)
                {
                    if (isIntegralType(divisor.value->getDataType()))
                    {
                        auto c = as<IRConstant>(divisor.value);
                        if (c->value.intVal == 0)
                        {
                            if (shared->sink)
                                shared->sink->diagnose(inst->sourceLoc, Diagnostics::divideByZero);
                            return LatticeVal::getAny();
                        }
                    }
                }
                return evalRem(inst->getDataType(), getLatticeVal(inst->getOperand(0)), divisor);
            }
        case kIROp_Eql:
            return evalEql(
                inst->getDataType(),
                getLatticeVal(inst->getOperand(0)),
                getLatticeVal(inst->getOperand(1)));
        case kIROp_Neq:
            return evalNeq(
                inst->getDataType(),
                getLatticeVal(inst->getOperand(0)),
                getLatticeVal(inst->getOperand(1)));
        case kIROp_Greater:
            return evalGreater(
                inst->getDataType(),
                getLatticeVal(inst->getOperand(0)),
                getLatticeVal(inst->getOperand(1)));
        case kIROp_Less:
            return evalLess(
                inst->getDataType(),
                getLatticeVal(inst->getOperand(0)),
                getLatticeVal(inst->getOperand(1)));
        case kIROp_Leq:
            return evalLeq(
                inst->getDataType(),
                getLatticeVal(inst->getOperand(0)),
                getLatticeVal(inst->getOperand(1)));
        case kIROp_Geq:
            return evalGeq(
                inst->getDataType(),
                getLatticeVal(inst->getOperand(0)),
                getLatticeVal(inst->getOperand(1)));
        case kIROp_And:
            return evalAnd(
                inst->getDataType(),
                getLatticeVal(inst->getOperand(0)),
                getLatticeVal(inst->getOperand(1)));
        case kIROp_Or:
            return evalOr(
                inst->getDataType(),
                getLatticeVal(inst->getOperand(0)),
                getLatticeVal(inst->getOperand(1)));
        case kIROp_Not:
            return evalNot(inst->getDataType(), getLatticeVal(inst->getOperand(0)));
        case kIROp_BitAnd:
            return evalBitAnd(
                inst->getDataType(),
                getLatticeVal(inst->getOperand(0)),
                getLatticeVal(inst->getOperand(1)));
        case kIROp_BitOr:
            return evalBitOr(
                inst->getDataType(),
                getLatticeVal(inst->getOperand(0)),
                getLatticeVal(inst->getOperand(1)));
        case kIROp_BitNot:
            return evalBitNot(inst->getDataType(), getLatticeVal(inst->getOperand(0)));
        case kIROp_BitXor:
            return evalBitXor(
                inst->getDataType(),
                getLatticeVal(inst->getOperand(0)),
                getLatticeVal(inst->getOperand(1)));
        case kIROp_BitCast:
            return evalBitCast(inst->getDataType(), getLatticeVal(inst->getOperand(0)));
        case kIROp_Neg:
            return evalNeg(inst->getDataType(), getLatticeVal(inst->getOperand(0)));
        case kIROp_Lsh:
            return evalLsh(
                inst->getDataType(),
                getLatticeVal(inst->getOperand(0)),
                getLatticeVal(inst->getOperand(1)));
        case kIROp_Rsh:
            return evalRsh(
                inst->getDataType(),
                getLatticeVal(inst->getOperand(0)),
                getLatticeVal(inst->getOperand(1)));
        case kIROp_Select:
            return evalSelect(
                getLatticeVal(inst->getOperand(0)),
                getLatticeVal(inst->getOperand(1)),
                getLatticeVal(inst->getOperand(2)));
        default:
            break;
        }

        // A safe default is to assume that every instruction not
        // handled by one of the cases above could produce *any*
        // value whatsoever.
        return LatticeVal::getAny();
    }


    // For basic blocks, we will do tracking very similar to what we do for
    // ordinary instructions, just with a simpler lattice: every block
    // will either be marked as "never executed" or in a "possibly executed"
    // state. We track this as a set of the blocks that have been
    // marked as possibly executed, plus a getter and setter function.

    HashSet<IRBlock*> executedBlocks;

    bool isMarkedAsExecuted(IRBlock* block) { return executedBlocks.contains(block); }

    void markAsExecuted(IRBlock* block) { executedBlocks.add(block); }

    // The core of the algorithm is based on two work lists.
    // One list holds CFG nodes (basic blocks) that we have
    // discovered might execute, and thus need to be processed,
    // and the other holds SSA nodes (instructions) that need
    // their "estimated" value to be updated.

    List<IRBlock*> cfgWorkList;
    List<IRInst*> ssaWorkList;

    // A key operation is to take an IR instruction and update
    // its "estimated" value on the lattice. This might happen when
    // we first discover the instruction could be executed, or
    // when we discover that one or more of its operands has
    // changed its lattice value so that we need to update our estimate.
    //
    void updateValueForInst(IRInst* inst)
    {
        // Block parameters are conceptually SSA "phi nodes", and it
        // doesn't make sense to update their values here, because the
        // actual candidate values for them comes from the predecessor blocks
        // that provide arguments. We will see that logic shortly, when
        // handling `IRUnconditionalBranch`.
        //
        if (as<IRParam, IRDynamicCastBehavior::NoUnwrap>(inst))
            return;

        // We want to special-case terminator instructions here,
        // since abstract interpretation of them should cause blocks to
        // be marked as executed, etc.
        //
        if (const auto terminator = as<IRTerminatorInst>(inst))
        {
            if (auto unconditionalBranch = as<IRUnconditionalBranch>(inst))
            {
                // When our abstract interpreter "executes" an unconditional
                // branch, it needs to mark the target block as potentially
                // executed. We do this by adding the target to our CFG work list.
                //
                auto target = unconditionalBranch->getTargetBlock();
                cfgWorkList.add(target);

                // Besides transferring control to another block, the other
                // thing our unconditional branch instructions do is provide
                // the arguments for phi nodes in the target block.
                // We thus need to interpret each argument on the branch
                // instruction like an "assignment" to the corresponding
                // parameter of the target block.
                //
                UInt argCount = unconditionalBranch->getArgCount();
                IRParam* pp = target->getFirstParam();
                for (UInt aa = 0; aa < argCount; ++aa, pp = pp->getNextParam())
                {
                    IRInst* arg = unconditionalBranch->getArg(aa);
                    IRInst* param = pp;

                    // We expect the number of arguments and parameters to match,
                    // or else the IR is violating its own invariants.
                    //
                    SLANG_ASSERT(param);

                    // We will update the value for the target block's parameter
                    // using our "meet" operation (union of sets of possible values)
                    //
                    LatticeVal oldVal = getLatticeVal(param);

                    // If we've already determined that the block parameter could
                    // have any value whatsoever, there is no reason to bother
                    // updating it.
                    //
                    if (oldVal.flavor == LatticeVal::Flavor::Any)
                        continue;

                    // We can look up the lattice value for the argument,
                    // because we should have interpreted it already
                    //
                    LatticeVal argVal = getLatticeVal(arg);

                    // Now we apply the meet operation and see if the value changed.
                    //
                    LatticeVal newVal = meet(oldVal, argVal);
                    if (newVal != oldVal)
                    {
                        // If the "estimated" value for the parameter has changed,
                        // then we need to update it in our dictionary, and then
                        // make sure that all of the users of the parameter get
                        // their estimates updated as well.
                        //
                        setLatticeVal(param, newVal);
                        for (auto use = param->firstUse; use; use = use->nextUse)
                        {
                            ssaWorkList.add(use->getUser());
                        }
                    }
                }
            }
            else if (auto conditionalBranch = as<IRConditionalBranch>(inst))
            {
                // An `IRConditionalBranch` is used for two-way branches.
                // We will look at the lattice value for the condition,
                // to see if we can narrow down which of the two ways
                // might actually be taken.
                //
                auto condVal = getLatticeVal(conditionalBranch->getCondition());

                // We do not expect to see a `None` value here, because that
                // would mean the user is branching based on an undefined
                // value.
                //
                // TODO: We should make sure there is no way for the user
                // to trigger this assert with bad code that involves
                // uninitialized variables. Right now we don't special
                // case the `undefined` instruction when computing lattice
                // values, so it shouldn't be a problem.
                //
                SLANG_ASSERT(condVal.flavor != LatticeVal::Flavor::None);

                // If the branch condition is a constant, we expect it to
                // be a Boolean constant. We won't assert that it is the
                // case here, just to be defensive.
                //
                if (condVal.flavor == LatticeVal::Flavor::Constant)
                {
                    if (auto boolConst = as<IRBoolLit>(condVal.value))
                    {
                        // Only one of the two targe blocks is possible to
                        // execute, based on what we know of the condition,
                        // so we will add that target to our work list and
                        // bail out now.
                        //
                        auto target = boolConst->getValue() ? conditionalBranch->getTrueBlock()
                                                            : conditionalBranch->getFalseBlock();
                        cfgWorkList.add(target);
                        return;
                    }
                }

                // As a fallback, if the condition isn't constant
                // (or somehow wasn't a Boolean constnat), we will
                // assume that either side of the branch could be
                // taken, so that both of the target blocks are
                // potentially executed.
                //
                cfgWorkList.add(conditionalBranch->getTrueBlock());
                cfgWorkList.add(conditionalBranch->getFalseBlock());
            }
            else if (auto switchInst = as<IRSwitch>(inst))
            {
                // The handling of a `switch` instruction is similar to the
                // case for a two-way branch, with the main difference that
                // we have to deal with an integer condition value.

                auto condVal = getLatticeVal(switchInst->getCondition());
                SLANG_ASSERT(condVal.flavor != LatticeVal::Flavor::None);

                UInt caseCount = switchInst->getCaseCount();
                if (condVal.flavor == LatticeVal::Flavor::Constant)
                {
                    if (auto condConst = as<IRIntLit>(condVal.value))
                    {
                        // At this point we have a constant integer condition
                        // value, and we just need to find the case (if any)
                        // that matches it. We will default to considering
                        // the `default` label as the target.
                        //
                        auto target = switchInst->getDefaultLabel();
                        for (UInt cc = 0; cc < caseCount; ++cc)
                        {
                            if (auto caseConst = as<IRIntLit>(switchInst->getCaseValue(cc)))
                            {
                                if (caseConst->getValue() == condConst->getValue())
                                {
                                    target = switchInst->getCaseLabel(cc);
                                    break;
                                }
                            }
                        }

                        // Whatever single block we decided will get executed,
                        // we need to make sure it gets processed and then bail.
                        //
                        cfgWorkList.add(target);
                        return;
                    }
                }

                // The fallback is to assume that the `switch` instruction might
                // branch to any of its cases, or the `default` label.
                //
                for (UInt cc = 0; cc < caseCount; ++cc)
                {
                    cfgWorkList.add(switchInst->getCaseLabel(cc));
                }
                cfgWorkList.add(switchInst->getDefaultLabel());
            }
            else if (auto targetSwitch = as<IRTargetSwitch>(inst))
            {
                for (UInt cc = 0; cc < targetSwitch->getCaseCount(); ++cc)
                {
                    cfgWorkList.add(targetSwitch->getCaseBlock(cc));
                }
            }
            // There are other cases of terminator instructions not handled
            // above (e.g., `return` instructions), but these can't cause
            // additional basic blocks in the CFG to execute, so we don't
            // need to consider them here.
            //
            // No matter what, we are done with a terminator instruction
            // after inspecting it, and there is no reason we have to
            // try and compute its "value."
            return;
        }

        // For an "ordinary" instruction, we will first check what value
        // has been registered for it already.
        //
        LatticeVal oldVal = getLatticeVal(inst);

        // If we have previous decided that the instruction could take
        // on any value whatsoever, then any further update to our
        // guess can't expand things more, and so there is nothing to do.
        //
        if (oldVal.flavor == LatticeVal::Flavor::Any)
        {
            return;
        }

        // Otherwise, we compute a new guess at the value of
        // the instruction based on the lattice values of the
        // stuff it depends on.
        //
        LatticeVal newVal = interpretOverLattice(inst);

        // If nothing changed about our guess, then there is nothing
        // further to do, because users of this instruction have
        // already computed their guess based on its current value.
        //
        if (newVal == oldVal)
        {
            return;
        }

        // If the guess did change, then we want to register our
        // new guess as the lattice value for this instruction.
        //
        setLatticeVal(inst, newVal);

        // Next we iterate over all the users of this instruction
        // and add them to our work list so that we can update
        // their values based on the new information.
        //
        for (auto use = inst->firstUse; use; use = use->nextUse)
        {
            ssaWorkList.add(use->getUser());
        }
    }

    // Run the constant folding on global scope and specialized types only.
    bool applyOnGlobalScope(IRModule* module)
    {
        bool changed = applyOnScope(module->getModuleInst());
        for (auto child : module->getModuleInst()->getChildren())
        {
            switch (child->getOp())
            {
            case kIROp_StructType:
            case kIROp_ClassType:
            case kIROp_InterfaceType:
            case kIROp_WitnessTable:
                changed |= applyOnScope(child);
                break;
            }
        }
        return changed;
    }

    bool applyOnScope(IRInst* scopeInst)
    {
        builderStorage = IRBuilder(scopeInst);
        for (auto child : scopeInst->getChildren())
        {
            // Only consider evaluable opcodes.
            if (!isEvaluableOpCode(child->getOp()))
                continue;

            updateValueForInst(child);
        }
        while (ssaWorkList.getCount())
        {
            auto inst = ssaWorkList[0];
            ssaWorkList.fastRemoveAt(0);
            // Only consider evaluable opcodes and insts at global scope.
            if (!isEvaluableOpCode(inst->getOp()) || inst->getParent() != scopeInst)
                continue;
            updateValueForInst(inst);
        }

        bool changed = false;
        // Replace the insts with their values.
        List<IRInst*> instsToRemove;
        for (auto child : scopeInst->getChildren())
        {
            if (!isEvaluableOpCode(child->getOp()))
                continue;

            auto latticeVal = getLatticeVal(child);
            if (latticeVal.flavor == LatticeVal::Flavor::Constant && latticeVal.value != child)
            {
                child->replaceUsesWith(latticeVal.value);
                instsToRemove.add(child);
            }
        }

        if (instsToRemove.getCount())
        {
            changed = true;
            for (auto inst : instsToRemove)
                inst->removeAndDeallocate();
        }
        return changed;
    }

    // The `apply()` function will run the full algorithm.
    //
    bool apply()
    {
        bool changed = false;
        // We start with the busy-work of setting up our IR builder.
        //
        builderStorage = IRBuilder(shared->module);

        // We expect the caller to have filtered out functions with
        // no bodies, so there should always be at least one basic block.
        //
        auto firstBlock = code->getFirstBlock();
        SLANG_ASSERT(firstBlock);

        // The entry block is always going to be executed when the
        // function gets called, so we will process it right away.
        //
        cfgWorkList.add(firstBlock);

        // The parameters of the first block are our function parameters,
        // and we want to operate on the assumption that they could have
        // any value possible, so we will record that in our dictionary.
        //
        for (auto pp : firstBlock->getParams())
        {
            setLatticeVal(pp, LatticeVal::getAny());
        }

        // Now we will iterate until both of our work lists go dry.
        //
        while (cfgWorkList.getCount() || ssaWorkList.getCount())
        {
            // Note: there is a design choice to be had here
            // around whether we do `if if` or `while while`
            // for these nested checks. The choice can affect
            // how long things take to converge.

            // We will start by processing any blocks that we
            // have determined are potentially reachable.
            //
            while (cfgWorkList.getCount())
            {
                // We pop one block off of the work list.
                //
                auto block = cfgWorkList[0];
                cfgWorkList.fastRemoveAt(0);

                // We only want to process blocks that haven't
                // already been marked as executed, so that we
                // don't do redundant work.
                //
                if (!isMarkedAsExecuted(block))
                {
                    // We should mark this new block as executed,
                    // so we can ignore it if it ever ends up on
                    // the work list again.
                    //
                    markAsExecuted(block);

                    // If the block is potentially executed, then
                    // that means the instructions in the block are too.
                    // We will walk through the block and update our
                    // guess at the value of each instruction, which
                    // may in turn add other blocks/instructions to
                    // the work lists.
                    //
                    for (auto inst : block->getDecorationsAndChildren())
                    {
                        updateValueForInst(inst);
                    }
                }
            }

            // Once we've cleared the work list of blocks, we
            // will start looking at individual instructions that
            // need to be updated.
            //
            while (ssaWorkList.getCount())
            {
                // We pop one instruction that needs an update.
                //
                auto inst = ssaWorkList[0];
                ssaWorkList.fastRemoveAt(0);

                // Before updating the instruction, we will check if
                // the parent block of the instructin is marked as
                // being executed. If it isn't, there is no reason
                // to update the value for the instruction, since
                // it might never be used anyway.
                //
                IRBlock* block = as<IRBlock>(inst->getParent());

                // It is possible that an instruction ended up on
                // our SSA work list because it is a user of an
                // instruction in a block of `code`, but it is not
                // itself an instruction a block of `code`.
                //
                // For example, if `code` is an `IRGeneric` that
                // yields a function, then `inst` might be an
                // instruction of that nested function, and not
                // an instruction of the generic itself.
                // Note that in such a case, the `inst` cannot
                // possible affect the values computed in the outer
                // generic, or the control-flow paths it might take,
                // so there is no reason to consider it.
                //
                // We guard against this case by only processing `inst`
                // if it is a child of a block in the current `code`.
                //
                if (!block || block->getParent() != code)
                    continue;

                if (isMarkedAsExecuted(block))
                {
                    // If the instruction is potentially executed, we update
                    // its lattice value based on our abstraction interpretation.
                    //
                    updateValueForInst(inst);
                }
            }
        }

        // Once the work lists are empty, our "guesses" at the value
        // of different instructions and the potentially-executed-ness
        // of blocks should have converged to a conservative steady state.
        //
        // We are now equiped to start using the information we've gathered
        // to modify the code.

        // First, we will walk through all the code and replace instructions
        // with constants where it is possible.
        //
        List<IRInst*> instsToRemove;
        for (auto block : code->getBlocks())
        {
            for (auto inst : block->getDecorationsAndChildren())
            {
                // We look for instructions that have a constnat value on
                // the lattice.
                //
                LatticeVal latticeVal = getLatticeVal(inst);
                if (latticeVal.flavor != LatticeVal::Flavor::Constant)
                    continue;

                // As a small sanity check, we won't go replacing an
                // instruction with itself (this shouldn't really come
                // up, since constants are supposed to be at the global
                // scope right now)
                //
                IRInst* constantVal = latticeVal.value;
                if (constantVal == inst)
                    continue;

                // We replace any uses of the instruction with its
                // constant expected value, and add it to a list of
                // instructions to be removed *iff* the instruction
                // is known to have no obersvable side effects.
                //
                inst->replaceUsesWith(constantVal);
                if (!inst->mightHaveSideEffects())
                {
                    // Don't delete phi parameters, they will be cleaned up in CFG simplification.
                    if (inst->getOp() != kIROp_Param)
                        instsToRemove.add(inst);
                }
            }
        }

        if (instsToRemove.getCount() != 0)
            changed = true;

        // Once we've replaced the uses of instructions that evaluate
        // to constants, we make a second pass to remove the instructions
        // themselves (or at least those without side effects).
        //
        for (auto inst : instsToRemove)
        {
            inst->removeAndDeallocate();
        }

        // Next we are going to walk through all of the terminator
        // instructions on blocks and look for ones that branch
        // based on a constant condition. These will be rewritten
        // to use direct branching instructions, which will of course
        // need to be emitted using a builder.
        //
        auto builder = getBuilder();
        for (auto block : code->getBlocks())
        {
            auto terminator = block->getTerminator();

            // We check if we have a `switch` instruction with a constant
            // integer as its condition.
            //
            if (auto switchInst = as<IRSwitch>(terminator))
            {
                if (auto constVal = as<IRIntLit>(switchInst->getCondition()))
                {
                    // We will select the one branch that gets taken, based
                    // on the constant condition value. The `default` label
                    // will of course be taken if no `case` label matches.
                    //
                    IRBlock* target = switchInst->getDefaultLabel();
                    UInt caseCount = switchInst->getCaseCount();
                    for (UInt cc = 0; cc < caseCount; ++cc)
                    {
                        auto caseVal = switchInst->getCaseValue(cc);
                        if (auto caseConst = as<IRIntLit>(caseVal))
                        {
                            if (caseConst->getValue() == constVal->getValue())
                            {
                                target = switchInst->getCaseLabel(cc);
                                break;
                            }
                        }
                    }

                    // Once we've found the target, we will emit a direct
                    // branch to it before the old terminator, and then remove
                    // the old terminator instruction.
                    //
                    builder->setInsertBefore(terminator);
                    builder->emitBranch(target);
                    terminator->removeAndDeallocate();
                    changed = true;
                }
            }
            else if (auto condBranchInst = as<IRConditionalBranch>(terminator))
            {
                if (auto constVal = as<IRBoolLit>(condBranchInst->getCondition()))
                {
                    // The case for a two-sided conditional branch is similar
                    // to the `switch` case, but simpler.

                    IRBlock* target = constVal->getValue() ? condBranchInst->getTrueBlock()
                                                           : condBranchInst->getFalseBlock();

                    builder->setInsertBefore(terminator);
                    builder->emitBranch(target);
                    terminator->removeAndDeallocate();
                    changed = true;
                }
            }
        }

        // At this point we've replaced some conditional branches
        // that would always go the same way (e.g., a `while(true)`),
        // which should render some of our blocks unreachable.
        // We will collect all those unreachable blocks into a list
        // of blocks to be removed, and then go about trying to
        // remove them.
        //
        List<IRBlock*> unreachableBlocks;
        for (auto block : code->getBlocks())
        {
            if (!isMarkedAsExecuted(block))
            {
                unreachableBlocks.add(block);
            }
        }
        //
        // It might seem like we could just do:
        //
        //      block->removeAndDeallocate();
        //
        // for each of the blocks in `unreachableBlocks`, but there
        // is a subtle point that has to be considered:
        //
        // We have a structured control-flow representation where
        // certain branching instructions name "join points" where
        // control flow logically re-converges. It is possible that
        // one of our unreachable blocks is still being used as
        // a join point.
        //
        // For example:
        //
        //      if(A)
        //          return B;
        //      else
        //          return C;
        //      D;
        //
        // In the above example, the block that computes `D` is
        // unreachable, but it is still the join point for the `if(A)`
        // branch.
        //
        // Rather than complicate the encoding of join points to
        // try to special-case an unreachable join point, we will
        // instead retain the join point as a block with only a single
        // `unreachable` instruction.
        //
        // To detect which blocks are unreachable and unreferenced,
        // we will check which blocks have any uses. Of course, it
        // might be that some of our unreachable blocks still reference
        // one another (e.g., an unreachable loop) so we will start
        // by removing the instructions from the bodies of our unreachable
        // blocks to eliminate any cross-references between them.
        //
        for (auto block : unreachableBlocks)
        {
            // TODO: In principle we could produce a diagnostic here
            // if any of these unreachable blocks appears to have
            // "non-trivial" code in it (that is, any code explicitly
            // written by the user, and not just code synthesized by
            // the compiler to satisfy language rules). Making that
            // determination could be tricky, so for now we will
            // err on the side of allowing unreachable code without
            // a warning.
            //
            block->removeAndDeallocateAllDecorationsAndChildren();
        }
        //
        // At this point every one of our unreachable blocks is empty,
        // and there should be no branches from reachable blocks
        // to unreachable ones.
        //
        // We will iterate over our unreachable blocks, and process
        // them differently based on whether they have any remaining uses.
        //
        for (auto block : unreachableBlocks)
        {
            // At this point there had better be no edges branching to
            // our block. We determined it was unreachable, so there had
            // better not be branches from reachable blocks to this one,
            // and all the unreachable blocks had their instructions
            // removed, so there should be no branches to it from other
            // unreachable blocks (or itself).
            //
            SLANG_ASSERT(block->getPredecessors().isEmpty());

            // If the block is completely unreferenced, we can safely
            // remove and deallocate it now.
            //
            if (!block->hasUses())
            {
                block->removeAndDeallocate();
            }
            else
            {
                // Otherwise, the block has at least one use (but
                // no predecessors), which should indicate that it
                // is an unreachable join point.
                //
                // We will keep the block around, but its entire
                // body will consist of a single `unreachable`
                // instruction.
                //
                builder->setInsertInto(block);
                builder->emitUnreachable();
            }
        }
        return changed;
    }
};

static bool applySparseConditionalConstantPropagationRec(
    const SCCPContext& globalContext,
    IRInst* inst)
{
    bool changed = false;
    if (auto code = as<IRGlobalValueWithCode>(inst))
    {
        if (code->getFirstBlock())
        {
            SCCPContext context;
            context.shared = globalContext.shared;
            context.code = code;
            context.mapInstToLatticeVal = globalContext.mapInstToLatticeVal;
            changed |= context.apply();
        }
    }

    for (auto childInst : inst->getDecorationsAndChildren())
    {
        switch (childInst->getOp())
        {
        case kIROp_Func:
        case kIROp_Block:
        case kIROp_Generic:
            break;
        default:
            // Skip other op codes.
            continue;
        }
        changed |= applySparseConditionalConstantPropagationRec(globalContext, childInst);
    }
    return changed;
}

bool applySparseConditionalConstantPropagation(IRModule* module, DiagnosticSink* sink)
{
    if (sink && sink->getErrorCount())
        return false;

    SharedSCCPContext shared;
    shared.module = module;
    shared.sink = sink;

    // First we fold constants at global scope.
    SCCPContext globalContext;
    globalContext.shared = &shared;
    globalContext.code = nullptr;
    bool changed = globalContext.applyOnGlobalScope(module);

    // Now run recursive SCCP passes on each child code block.
    changed |= applySparseConditionalConstantPropagationRec(globalContext, module->getModuleInst());

    return changed;
}

bool applySparseConditionalConstantPropagationForGlobalScope(IRModule* module, DiagnosticSink* sink)
{
    if (sink && sink->getErrorCount())
        return false;

    SharedSCCPContext shared;
    shared.module = module;
    shared.sink = sink;
    SCCPContext globalContext;
    globalContext.shared = &shared;
    globalContext.code = nullptr;
    bool changed = globalContext.applyOnGlobalScope(module);
    return changed;
}

bool applySparseConditionalConstantPropagation(IRInst* func, DiagnosticSink* sink)
{
    if (sink && sink->getErrorCount())
        return false;

    SharedSCCPContext shared;
    shared.module = func->getModule();
    shared.sink = sink;

    SCCPContext globalContext;
    globalContext.shared = &shared;
    globalContext.code = nullptr;

    // Run recursive SCCP passes on each child code block.
    return applySparseConditionalConstantPropagationRec(globalContext, func);
}

IRInst* tryConstantFoldInst(IRModule* module, IRInst* inst)
{
    SharedSCCPContext shared;
    shared.module = module;
    SCCPContext instContext;
    instContext.shared = &shared;
    instContext.code = nullptr;
    instContext.builderStorage = IRBuilder(module);
    auto foldResult = instContext.interpretOverLattice(inst);
    if (!foldResult.value)
    {
        return inst;
    }
    inst->replaceUsesWith(foldResult.value);
    return foldResult.value;
}

} // namespace Slang
