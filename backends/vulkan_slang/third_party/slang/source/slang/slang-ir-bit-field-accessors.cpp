#include "slang-ir-bit-field-accessors.h"

#include "slang-ir-insts.h"
#include "slang-ir.h"

namespace Slang
{
static IRInst* shl(IRBuilder& builder, IRInst* inst, const IRIntegerValue value)
{
    if (value == 0)
        return inst;
    const auto [width, isSigned] = getIntTypeInfo(inst->getDataType());
    if (value >= width)
        return builder.getIntValue(inst->getDataType(), 0);
    if (value == 0)
        return inst;
    return builder.emitShl(
        inst->getDataType(),
        inst,
        builder.getIntValue(builder.getIntType(), value));
}

static IRInst* shr(IRBuilder& builder, IRInst* inst, const IRIntegerValue value)
{
    if (value == 0)
        return inst;
    const auto [width, isSigned] = getIntTypeInfo(inst->getDataType());
    // If it's not signed, then we just shift all the set bits away
    if (value >= width && !isSigned)
        return builder.getIntValue(inst->getDataType(), 0);
    // Since on many platforms bit shifting by the number of bits in the number
    // is undefined, correct this here assuming that the Slang IR has the same
    // restriction
    if (value >= width && isSigned)
        return builder.emitShr(
            inst->getDataType(),
            inst,
            builder.getIntValue(builder.getIntType(), width - 1));
    if (value == 0)
        return inst;
    return builder.emitShr(
        inst->getDataType(),
        inst,
        builder.getIntValue(builder.getIntType(), value));
}

static void synthesizeBitFieldGetter(IRFunc* func, IRBitFieldAccessorDecoration* dec)
{
    const auto bitFieldType = func->getResultType();
    SLANG_ASSERT(isIntegralType(bitFieldType));
    SLANG_ASSERT(func->getParamCount() == 1);
    const auto structParamType = func->getParamType(0);
    const auto structType = as<IRStructType>(getResolvedInstForDecorations(structParamType));
    SLANG_ASSERT(structType);

    const auto backingMember = findStructField(structType, dec->getBackingMemberKey());
    const auto backingType = backingMember->getFieldType();
    SLANG_ASSERT(isIntegralType(backingType));

    IRBuilder builder{func};

    const auto isSigned = getIntTypeInfo(func->getResultType()).isSigned;
    builder.setInsertInto(func);
    builder.emitBlock();
    const auto s = builder.emitParam(structParamType);

    // Construct the equivalent of this:
    // Note the cast of the backing value in order to get the correct sign
    // extension behaviour on the right shift
    // return (int(_backing) << (backingWidth-topOfFoo)) >> (backingWidth-fooWidth);

    const auto backingWidth = getIntTypeInfo(backingType).width;
    const auto fieldWidth = dec->getFieldWidth();
    const auto topOfField = dec->getFieldOffset() + fieldWidth;
    const auto leftShiftAmount = backingWidth - topOfField;
    const auto rightShiftAmount = backingWidth - fieldWidth;
    const auto backingValue = builder.emitFieldExtract(backingType, s, dec->getBackingMemberKey());
    const auto castBackingType = builder.getType(getIntTypeOpFromInfo({backingWidth, isSigned}));
    const auto castedBacking = builder.emitCast(castBackingType, backingValue);
    const auto leftShifted = shl(builder, castedBacking, leftShiftAmount);
    const auto rightShifted = shr(builder, leftShifted, rightShiftAmount);
    const auto castedToBitFieldType = builder.emitCast(bitFieldType, rightShifted);
    builder.emitReturn(castedToBitFieldType);

    builder.addSimpleDecoration<IRForceInlineDecoration>(func);
}

static IRIntegerValue setLowBits(IRIntegerValue bits)
{
    SLANG_ASSERT(bits >= 0 && bits <= 64);
    return ~(bits >= 64 ? 0 : (~0ULL << bits));
}

static void synthesizeBitFieldSetter(IRFunc* func, IRBitFieldAccessorDecoration* dec)
{
    SLANG_ASSERT(func->getParamCount() == 2);
    const auto ptrType = as<IRPtrTypeBase>(func->getParamType(0));
    SLANG_ASSERT(ptrType);
    const auto structParamType = ptrType->getValueType();
    const auto structType = as<IRStructType>(getResolvedInstForDecorations(structParamType));
    SLANG_ASSERT(structType);
    const auto bitFieldType = func->getParamType(1);
    SLANG_ASSERT(isIntegralType(bitFieldType));

    const auto backingMember = findStructField(structType, dec->getBackingMemberKey());
    const auto backingType = backingMember->getFieldType();
    SLANG_ASSERT(isIntegralType(backingType));

    IRBuilder builder{func};

    builder.setInsertInto(func);
    builder.emitBlock();
    const auto s = builder.emitParam(ptrType);
    const auto v = builder.emitParam(bitFieldType);

    // Construct the equivalent of this:
    // let fooMask = 0x00000FF0;
    // let bottomOfFoo = 4;
    // _backing = int((_backing & ~fooMask) | ((int(x) << bottomOfFoo) & fooMask));

    const auto fieldWidth = dec->getFieldWidth();
    const auto bottomOfField = dec->getFieldOffset();
    const auto maskBits = setLowBits(fieldWidth) << bottomOfField;
    const auto mask = builder.getIntValue(backingType, maskBits);
    const auto notMask = builder.getIntValue(backingType, ~maskBits);
    const auto memberAddr =
        builder.emitFieldAddress(builder.getPtrType(backingType), s, dec->getBackingMemberKey());
    const auto backingValue = builder.emitLoad(memberAddr);
    const auto maskedOut = builder.emitBitAnd(backingType, backingValue, notMask);
    const auto castValue = builder.emitCast(backingType, v);
    const auto shiftedLeft = shl(builder, castValue, bottomOfField);
    const auto maskedValue = builder.emitBitAnd(backingType, shiftedLeft, mask);
    const auto combined = builder.emitBitOr(backingType, maskedOut, maskedValue);
    builder.emitStore(memberAddr, combined);
    builder.emitReturn();

    builder.addSimpleDecoration<IRForceInlineDecoration>(func);
}

void synthesizeBitFieldAccessors(IRModule* module)
{
    for (const auto inst : module->getModuleInst()->getGlobalInsts())
    {
        const auto func = as<IRFunc>(getResolvedInstForDecorations(inst));
        if (!func)
            continue;
        const auto bfd = func->findDecoration<IRBitFieldAccessorDecoration>();
        if (!bfd)
            continue;
        if (func->getParamCount() == 1)
            synthesizeBitFieldGetter(func, bfd);
        else
            synthesizeBitFieldSetter(func, bfd);
    }
}
} // namespace Slang
