#include "slang-ir-insert-debug-value-store.h"

#include "slang-ir-insts.h"
#include "slang-ir-util.h"
#include "slang-ir.h"

namespace Slang
{
struct DebugValueStoreContext
{
    Dictionary<IRType*, bool> m_mapTypeToDebugability;
    bool isTypeKind(IRInst* inst)
    {
        if (!inst)
            return true;
        switch (inst->getOp())
        {
        case kIROp_TypeKind:
        case kIROp_TypeType:
            return true;
        default:
            return false;
        }
    }
    bool isDebuggableType(IRType* type)
    {
        if (bool* result = m_mapTypeToDebugability.tryGetValue(type))
            return *result;

        bool debuggable = false;
        switch (type->getOp())
        {
        case kIROp_VoidType:
            break;
        case kIROp_StructType:
            {
                auto structType = static_cast<IRStructType*>(type);
                bool structDebuggable = true;
                for (auto field : structType->getFields())
                {
                    if (!isDebuggableType(field->getFieldType()))
                    {
                        structDebuggable = false;
                        break;
                    }
                }
                debuggable = structDebuggable;
                break;
            }
        case kIROp_ArrayType:
        case kIROp_UnsizedArrayType:
            {
                auto arrayType = static_cast<IRArrayTypeBase*>(type);
                debuggable = isDebuggableType(arrayType->getElementType());
                break;
            }
        case kIROp_VectorType:
        case kIROp_MatrixType:
        case kIROp_PtrType:
            debuggable = true;
            break;
        case kIROp_Param:
            // Assume generic parameters are debuggable.
            debuggable = true;
            break;
        case kIROp_Specialize:
            {
                auto specType = as<IRSpecialize>(type);
                auto specTypeDebuggable =
                    isDebuggableType((IRType*)getResolvedInstForDecorations(specType));
                if (!specTypeDebuggable)
                    break;
                for (UInt i = 0; i < specType->getArgCount(); i++)
                {
                    auto arg = specType->getArg(i);
                    if (isTypeKind(arg->getDataType()) &&
                        !isDebuggableType((IRType*)specType->getArg(i)))
                    {
                        specTypeDebuggable = false;
                        break;
                    }
                }
                debuggable = false; // specTypeDebuggable;
                break;
            }
        default:
            if (as<IRBasicType>(type))
                debuggable = true;
            break;
        }
        m_mapTypeToDebugability[type] = debuggable;
        return debuggable;
    }

    void insertDebugValueStore(IRFunc* func)
    {
        IRBuilder builder(func);
        Dictionary<IRInst*, IRInst*> mapVarToDebugVar;
        auto firstBlock = func->getFirstBlock();
        if (!firstBlock)
            return;
        auto funcDebugLoc = func->findDecoration<IRDebugLocationDecoration>();
        if (!funcDebugLoc)
            return;
        List<IRInst*> params;
        for (auto param : firstBlock->getParams())
        {
            params.add(param);
        }
        Index paramIndex = 0;
        for (auto param : params)
        {
            builder.setInsertBefore(firstBlock->getFirstOrdinaryInst());
            auto paramType = param->getDataType();
            bool isRefParam = false;
            if (auto outType = as<IROutTypeBase>(paramType))
            {
                isRefParam = true;
                paramType = outType->getValueType();
            }
            if (!isDebuggableType(paramType))
                continue;
            auto debugVar = builder.emitDebugVar(
                paramType,
                funcDebugLoc->getSource(),
                funcDebugLoc->getLine(),
                funcDebugLoc->getCol(),
                builder.getIntValue(builder.getUIntType(), paramIndex));
            copyNameHintAndDebugDecorations(debugVar, param);

            mapVarToDebugVar[param] = debugVar;

            // Store the initial value of the parameter into the debug var.
            IRInst* paramVal = nullptr;
            if (!isRefParam)
                paramVal = param;
            else if (as<IRInOutType>(param->getDataType()))
                paramVal = builder.emitLoad(param);
            if (paramVal)
            {
                builder.emitDebugValue(debugVar, paramVal);
            }
            paramIndex++;
        }

        for (auto block : func->getBlocks())
        {
            IRInst* nextInst = nullptr;
            for (auto inst = block->getFirstInst(); inst; inst = nextInst)
            {
                nextInst = inst->getNextInst();
                if (auto varInst = as<IRVar>(inst))
                {
                    if (auto debugLoc = varInst->findDecoration<IRDebugLocationDecoration>())
                    {
                        auto varType = tryGetPointedToType(&builder, varInst->getDataType());
                        builder.setInsertBefore(varInst);
                        if (!isDebuggableType(varType))
                            continue;
                        auto debugVar = builder.emitDebugVar(
                            varType,
                            debugLoc->getSource(),
                            debugLoc->getLine(),
                            debugLoc->getCol());
                        copyNameHintAndDebugDecorations(debugVar, varInst);
                        mapVarToDebugVar[varInst] = debugVar;
                    }
                }
            }
        }

        // Collect all stores and insert debug value insts to update debug vars.

        // Helper func to insert debugValue updates.
        auto setDebugValue = [&](IRInst* debugVar, IRInst* newValue, ArrayView<IRInst*> accessChain)
        {
            auto ptr = builder.emitElementAddress(debugVar, accessChain);
            builder.emitDebugValue(ptr, newValue);
        };
        for (auto block : func->getBlocks())
        {
            IRInst* nextInst = nullptr;
            for (auto inst = block->getFirstInst(); inst; inst = nextInst)
            {
                nextInst = inst->getNextInst();

                if (auto storeInst = as<IRStore>(inst))
                {
                    List<IRInst*> accessChain;
                    auto varInst = getRootAddr(storeInst->getPtr(), accessChain);
                    IRInst* debugVar = nullptr;
                    if (mapVarToDebugVar.tryGetValue(varInst, debugVar))
                    {
                        builder.setInsertAfter(storeInst);
                        setDebugValue(debugVar, storeInst->getVal(), accessChain.getArrayView());
                    }
                }
                else if (auto swizzledStore = as<IRSwizzledStore>(inst))
                {
                    List<IRInst*> accessChain;
                    auto varInst = getRootAddr(swizzledStore->getDest(), accessChain);
                    IRInst* debugVar = nullptr;
                    if (mapVarToDebugVar.tryGetValue(varInst, debugVar))
                    {
                        builder.setInsertAfter(swizzledStore);
                        auto loadVal = builder.emitLoad(swizzledStore->getDest());
                        setDebugValue(debugVar, loadVal, accessChain.getArrayView());
                    }
                }
                else if (auto callInst = as<IRCall>(inst))
                {
                    auto funcValue = getResolvedInstForDecorations(callInst->getCallee());
                    if (!funcValue)
                        continue;
                    for (UInt i = 0; i < callInst->getArgCount(); i++)
                    {
                        auto arg = callInst->getArg(i);
                        if (!as<IRPtrTypeBase>(arg->getDataType()))
                            continue;
                        List<IRInst*> accessChain;
                        auto varInst = getRootAddr(arg, accessChain);
                        IRInst* debugVar = nullptr;
                        if (mapVarToDebugVar.tryGetValue(varInst, debugVar))
                        {
                            builder.setInsertAfter(callInst);
                            auto loadVal = builder.emitLoad(arg);
                            setDebugValue(debugVar, loadVal, accessChain.getArrayView());
                        }
                    }
                }
            }
        }
    }
};

void insertDebugValueStore(IRModule* module)
{
    DebugValueStoreContext context;
    for (auto globalInst : module->getGlobalInsts())
    {
        if (auto genericInst = as<IRGeneric>(globalInst))
        {
            if (auto func = as<IRFunc>(findGenericReturnVal(genericInst)))
            {
                context.insertDebugValueStore(func);
            }
        }
        else if (auto func = as<IRFunc>(globalInst))
        {
            context.insertDebugValueStore(func);
        }
    }
}
} // namespace Slang
