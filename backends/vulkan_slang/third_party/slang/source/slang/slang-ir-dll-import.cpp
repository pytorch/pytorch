// slang-ir-dll-import.cpp
#include "slang-ir-dll-import.h"

#include "slang-ir-insts.h"
#include "slang-ir-layout.h"
#include "slang-ir-marshal-native-call.h"
#include "slang-ir.h"

namespace Slang
{

struct DllImportContext
{
    IRModule* module;
    DiagnosticSink* diagnosticSink;
    TargetProgram* targetProgram;

    IRFunc* loadDllFunc = nullptr;
    IRFunc* loadFuncPtrFunc = nullptr;
    IRFunc* stringGetBufferFunc = nullptr;

    IRFunc* createBuiltinIntrinsicFunc(
        UInt paramCount,
        IRType** paramTypes,
        IRType* resultType,
        UnownedStringSlice targetIntrinsic)
    {
        IRBuilder builder(module);
        builder.setInsertInto(module->getModuleInst());
        IRFunc* result = builder.createFunc();
        builder.setInsertInto(result);
        auto funcType = builder.getFuncType(paramCount, paramTypes, resultType);
        builder.setDataType(result, funcType);
        builder.addTargetIntrinsicDecoration(
            result,
            CapabilitySet(CapabilityName::cpp),
            targetIntrinsic);
        return result;
    }

    IRFunc* getLoadDllFunc()
    {
        if (!loadDllFunc)
        {
            IRBuilder builder(module);
            builder.setInsertInto(module->getModuleInst());
            IRType* stringType = builder.getStringType();
            loadDllFunc = createBuiltinIntrinsicFunc(
                1,
                &stringType,
                builder.getPtrType(builder.getVoidType()),
                UnownedStringSlice("_slang_rt_load_dll($0)"));
        }
        return loadDllFunc;
    }

    IRFunc* getLoadFuncPtrFunc()
    {
        if (!loadFuncPtrFunc)
        {
            IRBuilder builder(module);
            builder.setInsertInto(module->getModuleInst());

            IRType* stringType = builder.getStringType();
            IRType* uintType = builder.getUIntType();
            IRType* paramTypes[] = {
                builder.getPtrType(builder.getVoidType()),
                stringType,
                uintType};
            loadFuncPtrFunc = createBuiltinIntrinsicFunc(
                3,
                paramTypes,
                builder.getPtrType(builder.getVoidType()),
                UnownedStringSlice("_slang_rt_load_dll_func($0, $1, $2)"));
        }
        return loadFuncPtrFunc;
    }

    IRFunc* getStringGetBufferFunc()
    {
        if (!stringGetBufferFunc)
        {
            IRBuilder builder(module);
            builder.setInsertInto(module->getModuleInst());

            IRType* stringType = builder.getStringType();
            IRType* paramTypes[] = {stringType};
            stringGetBufferFunc = createBuiltinIntrinsicFunc(
                1,
                paramTypes,
                builder.getPtrType(builder.getCharType()),
                UnownedStringSlice("const_cast<char*>($0.getBuffer())"));
        }
        return stringGetBufferFunc;
    }

    uint32_t getStdCallArgumentSize(IRFunc* func)
    {
        uint32_t result = 0;
        for (auto param : func->getParams())
        {
            IRSizeAndAlignment sizeAndAlignment;
            getNaturalSizeAndAlignment(
                targetProgram->getOptionSet(),
                param->getDataType(),
                &sizeAndAlignment);
            result += (uint32_t)align(sizeAndAlignment.size, 4);
        }
        return result;
    }

    void processFunc(IRFunc* func, IRDllImportDecoration* dllImportDecoration)
    {
        assert(func->getFirstBlock() == nullptr);

        IRBuilder builder(module);
        NativeCallMarshallingContext marshalContext;

        auto nativeType = marshalContext.getNativeFuncType(builder, func->getDataType());
        builder.setInsertInto(module->getModuleInst());
        auto funcPtr = builder.createGlobalVar(nativeType);
        builder.setInsertInto(funcPtr);
        builder.emitBlock();
        builder.emitReturn(builder.getNullVoidPtrValue());

        builder.setInsertInto(func);
        auto block = builder.emitBlock();
        builder.setInsertInto(block);

        // Emit parameters.
        auto declaredFuncType = func->getDataType();
        List<IRParam*> params;
        for (UInt i = 0; i < declaredFuncType->getParamCount(); ++i)
        {
            auto paramType = declaredFuncType->getParamType(i);
            params.add(builder.emitParam((IRType*)paramType));
        }

        IRInst* cmpArgs[] = {builder.emitLoad(nativeType, funcPtr), builder.getNullVoidPtrValue()};
        auto isUninitialized =
            builder.emitIntrinsicInst(builder.getBoolType(), kIROp_Eql, 2, cmpArgs);

        auto trueBlock = builder.emitBlock();
        auto afterBlock = builder.emitBlock();

        builder.setInsertInto(block);
        builder.emitIf(isUninitialized, trueBlock, afterBlock);

        builder.setInsertInto(trueBlock);
        IRInst* modulePtr;

        if (dllImportDecoration->getLibraryName() == "")
        {
            modulePtr = builder.getNullVoidPtrValue();
        }
        else
        {
            modulePtr = builder.emitCallInst(
                builder.getPtrType(builder.getVoidType()),
                getLoadDllFunc(),
                builder.getStringValue(dllImportDecoration->getLibraryName()));
        }
        auto argumentSize =
            builder.getIntValue(builder.getUIntType(), getStdCallArgumentSize(func));
        IRInst* loadDllFuncArgs[] = {
            modulePtr,
            builder.getStringValue(dllImportDecoration->getFunctionName()),
            argumentSize};
        auto loadedNativeFuncPtr = builder.emitCallInst(
            builder.getPtrType(builder.getVoidType()),
            getLoadFuncPtrFunc(),
            3,
            loadDllFuncArgs);
        builder.emitStore(funcPtr, builder.emitBitCast(nativeType, loadedNativeFuncPtr));
        builder.emitBranch(afterBlock);
        builder.setInsertInto(afterBlock);

        marshalContext.diagnosticSink = diagnosticSink;
        auto callResult = marshalContext.marshalNativeCall(
            builder,
            declaredFuncType,
            nativeType,
            builder.emitLoad(funcPtr),
            params.getCount(),
            (IRInst**)params.getBuffer());
        builder.emitReturn(callResult);
    }

    void processModule()
    {
        for (auto childFunc : module->getGlobalInsts())
        {
            switch (childFunc->getOp())
            {
            case kIROp_Func:
                if (auto dllImportDecoration = childFunc->findDecoration<IRDllImportDecoration>())
                {
                    processFunc(as<IRFunc>(childFunc), dllImportDecoration);
                }
                break;
            default:
                break;
            }
        }
    }
};

void generateDllImportFuncs(TargetProgram* targetProgram, IRModule* module, DiagnosticSink* sink)
{
    DllImportContext context;
    context.module = module;
    context.targetProgram = targetProgram;
    context.diagnosticSink = sink;
    return context.processModule();
}

} // namespace Slang
