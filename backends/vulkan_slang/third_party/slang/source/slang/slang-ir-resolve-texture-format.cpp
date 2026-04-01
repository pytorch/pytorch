#include "slang-ir-resolve-texture-format.h"

#include "slang-ir-clone.h"
#include "slang-ir-insts.h"

namespace Slang
{
static IRType* replaceImageElementType(IRInst* originalType, IRInst* newElementType)
{
    switch (originalType->getOp())
    {
    case kIROp_ArrayType:
    case kIROp_UnsizedArrayType:
    case kIROp_PtrType:
    case kIROp_OutType:
    case kIROp_RefType:
    case kIROp_ConstRefType:
    case kIROp_InOutType:
        {
            auto newInnerType =
                replaceImageElementType(originalType->getOperand(0), newElementType);
            if (newInnerType != originalType->getOperand(0))
            {
                IRBuilder builder(originalType);
                builder.setInsertBefore(originalType);
                IRCloneEnv cloneEnv;
                cloneEnv.mapOldValToNew.add(originalType->getOperand(0), newInnerType);
                return (IRType*)cloneInst(&cloneEnv, &builder, originalType);
            }
            return (IRType*)originalType;
        }

    default:
        if (as<IRResourceTypeBase>(originalType))
            return (IRType*)newElementType;
        return (IRType*)originalType;
    }
}

static void resolveTextureFormatForParameter(IRInst* textureInst, IRTextureTypeBase* textureType)
{
    ImageFormat format = (ImageFormat)(textureType->getFormat());
    auto decor = textureInst->findDecoration<IRFormatDecoration>();
    if (!decor)
        return;
    if (decor->getFormat() == (ImageFormat)textureType->getFormat())
        return;

    format = decor->getFormat();
    if (format != ImageFormat::unknown)
    {
        IRBuilder builder(textureInst->getModule());
        builder.setInsertBefore(textureInst);
        auto formatArg = builder.getIntValue(builder.getUIntType(), IRIntegerValue(format));

        auto newType = builder.getTextureType(
            textureType->getElementType(),
            textureType->getShapeInst(),
            textureType->getIsArrayInst(),
            textureType->getIsMultisampleInst(),
            textureType->getSampleCountInst(),
            textureType->getAccessInst(),
            textureType->getIsShadowInst(),
            textureType->getIsCombinedInst(),
            formatArg);

        List<IRUse*> typeReplacementWorkList;
        HashSet<IRUse*> typeReplacementWorkListSet;

        auto newInstType = (IRType*)replaceImageElementType(textureInst->getFullType(), newType);
        textureInst->setFullType(newInstType);

        for (auto use = textureInst->firstUse; use; use = use->nextUse)
        {
            if (typeReplacementWorkListSet.add(use))
                typeReplacementWorkList.add(use);
        }

        // Update the types on dependent insts.
        for (Index i = 0; i < typeReplacementWorkList.getCount(); i++)
        {
            auto use = typeReplacementWorkList[i];
            auto user = use->getUser();
            switch (user->getOp())
            {
            case kIROp_GetElementPtr:
            case kIROp_GetElement:
            case kIROp_Load:
            case kIROp_Var:
                {
                    auto newUserType =
                        (IRType*)replaceImageElementType(user->getFullType(), newType);
                    if (newUserType != user->getFullType())
                    {
                        user->setFullType(newUserType);
                        for (auto u = user->firstUse; u; u = u->nextUse)
                        {
                            if (typeReplacementWorkListSet.add(u))
                                typeReplacementWorkList.add(u);
                        }
                    }
                    break;
                }
            case kIROp_Store:
                {
                    auto store = as<IRStore>(user);
                    if (use == store->getValUse())
                    {
                        auto ptr = store->getPtr();
                        auto newPtrType =
                            (IRType*)replaceImageElementType(ptr->getFullType(), newType);
                        if (newPtrType != ptr->getFullType())
                        {
                            ptr->setFullType(newPtrType);
                            for (auto u = ptr->firstUse; u; u = u->nextUse)
                            {
                                if (typeReplacementWorkListSet.add(u))
                                    typeReplacementWorkList.add(u);
                            }
                        }
                    }
                    break;
                }
            }
        }
    }
}

void resolveTextureFormat(IRModule* module)
{
    for (auto globalInst : module->getGlobalInsts())
    {
        if (as<IRTextureTypeBase>(globalInst->getDataType()))
        {
            resolveTextureFormatForParameter(
                globalInst,
                (IRTextureTypeBase*)globalInst->getDataType());
        }
        else if (auto arrayType = as<IRArrayTypeBase>(globalInst->getDataType()))
        {
            if (as<IRTextureTypeBase>(arrayType->getElementType()))
            {
                resolveTextureFormatForParameter(
                    globalInst,
                    (IRTextureTypeBase*)arrayType->getElementType());
            }
        }
    }
}
} // namespace Slang
