#include "slang-emit-base.h"

#include "slang-ir-util.h"

namespace Slang
{

IRInst* SourceEmitterBase::getSpecializedValue(IRSpecialize* specInst)
{
    auto base = specInst->getBase();

    // It is possible to have a `specialize(...)` where the first
    // operand is also a `specialize(...)`, so that we need to
    // look at what declaration is being specialized at the inner
    // step to find the one being specialized at the outer step.
    //
    while (auto baseSpecialize = as<IRSpecialize>(base))
    {
        base = getSpecializedValue(baseSpecialize);
    }

    auto baseGeneric = as<IRGeneric>(base);
    if (!baseGeneric)
        return base;

    auto lastBlock = baseGeneric->getLastBlock();
    if (!lastBlock)
        return base;

    auto returnInst = as<IRReturn>(lastBlock->getTerminator());
    if (!returnInst)
        return base;

    return returnInst->getVal();
}

void SourceEmitterBase::handleRequiredCapabilities(IRInst* inst)
{
    auto decoratedValue = inst;
    while (auto specInst = as<IRSpecialize>(decoratedValue))
    {
        decoratedValue = getSpecializedValue(specInst);
    }

    handleRequiredCapabilitiesImpl(decoratedValue);
}

IRVarLayout* SourceEmitterBase::getVarLayout(IRInst* var)
{
    return findVarLayout(var);
}

BaseType SourceEmitterBase::extractBaseType(IRType* inType)
{
    auto type = inType;
    for (;;)
    {
        if (auto irBaseType = as<IRBasicType>(type))
        {
            return irBaseType->getBaseType();
        }
        else if (auto vecType = as<IRVectorType>(type))
        {
            type = vecType->getElementType();
            continue;
        }
        else
        {
            return BaseType::Void;
        }
    }
}

} // namespace Slang
