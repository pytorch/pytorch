#include "slang-vm-inst-impl.h"

#include "slang-vm.h"

using namespace slang;

namespace Slang
{
ByteCodeInterpreter* convert(IByteCodeRunner* runner)
{
    return static_cast<ByteCodeInterpreter*>(runner);
}

#define SIMPLE_BINARY_SCALAR_FUNC(name, op)                      \
    struct name##ScalarFunc                                      \
    {                                                            \
        template<typename TR, typename T1, typename T2>          \
        static void run(TR* dst, const T1* src1, const T2* src2) \
        {                                                        \
            *dst = (*src1)op(*src2);                             \
        }                                                        \
    }

SIMPLE_BINARY_SCALAR_FUNC(Add, +);
SIMPLE_BINARY_SCALAR_FUNC(Sub, -);
SIMPLE_BINARY_SCALAR_FUNC(Mul, *);
SIMPLE_BINARY_SCALAR_FUNC(Div, /);
SIMPLE_BINARY_SCALAR_FUNC(And, &&);
SIMPLE_BINARY_SCALAR_FUNC(Or, ||);
SIMPLE_BINARY_SCALAR_FUNC(BitAnd, &);
SIMPLE_BINARY_SCALAR_FUNC(BitOr, |);
SIMPLE_BINARY_SCALAR_FUNC(BitXor, ^);
SIMPLE_BINARY_SCALAR_FUNC(Shl, <<);
SIMPLE_BINARY_SCALAR_FUNC(Shr, >>);
SIMPLE_BINARY_SCALAR_FUNC(Less, <);
SIMPLE_BINARY_SCALAR_FUNC(Leq, <=);
SIMPLE_BINARY_SCALAR_FUNC(Greater, >);
SIMPLE_BINARY_SCALAR_FUNC(Geq, >=);
SIMPLE_BINARY_SCALAR_FUNC(Equal, ==);
SIMPLE_BINARY_SCALAR_FUNC(Neq, !=);

template<typename TR, typename T1, typename T2>
void scalarMod(TR* dst, const T1* src1, const T2* src2)
{
    *dst = *src1 % *src2;
}

template<>
void scalarMod<float, float, float>(float* dst, const float* src1, const float* src2)
{
    *dst = fmodf(*src1, *src2);
}

template<>
void scalarMod<double, double, double>(double* dst, const double* src1, const double* src2)
{
    *dst = fmod(*src1, *src2);
}

struct ModScalarFunc
{
    template<typename TR, typename T1, typename T2>
    static void run(TR* dst, const T1* src1, const T2* src2)
    {
        scalarMod<TR, T1, T2>(dst, src1, src2);
    }
};

#define SIMPLE_UNARY_SCALAR_FUNC(name, op)       \
    struct name##ScalarFunc                      \
    {                                            \
        template<typename TR, typename T1>       \
        static void run(TR* dst, const T1* src1) \
        {                                        \
            *dst = op(*src1);                    \
        }                                        \
    }
SIMPLE_UNARY_SCALAR_FUNC(Neg, -);
SIMPLE_UNARY_SCALAR_FUNC(Not, !);
SIMPLE_UNARY_SCALAR_FUNC(BitNot, ~);

template<typename ScalarFunc, typename TR, typename T1, typename T2, int elementCount>
struct BinaryVectorFunc
{
    static void run(IByteCodeRunner* context, VMExecInstHeader* inst, void* userData)
    {
        SLANG_UNUSED(context);
        SLANG_UNUSED(userData);
        TR* dst = (TR*)inst->getOperand(0).getPtr();
        T1* src1 = (T1*)inst->getOperand(1).getPtr();
        T2* src2 = (T2*)inst->getOperand(2).getPtr();
        for (int i = 0; i < elementCount; ++i)
        {
            ScalarFunc::template run<TR, T1, T2>(&dst[i], &src1[i], &src2[i]);
        }
    }
};

template<typename ScalarFunc, typename TR, typename T1, typename T2>
struct GeneralBinaryVectorFunc
{
    static void run(IByteCodeRunner* context, VMExecInstHeader* inst, void* userData)
    {
        SLANG_UNUSED(context);
        SLANG_UNUSED(userData);
        TR* dst = (TR*)inst->getOperand(0).getPtr();
        T1* src1 = (T1*)inst->getOperand(1).getPtr();
        T2* src2 = (T2*)inst->getOperand(2).getPtr();
        ArithmeticExtCode arithExtCode;
        memcpy(&arithExtCode, &inst->opcodeExtension, sizeof(arithExtCode));
        for (uint32_t i = 0; i < arithExtCode.vectorSize; ++i)
        {
            ScalarFunc::template run<TR, T1, T2>(&dst[i], &src1[i], &src2[i]);
        }
    }
};

template<typename Func, typename TR, typename T1 = TR, typename T2 = TR>
VMExtFunction binaryArithmeticInstHandler(int elementCount)
{
    switch (elementCount)
    {
    case 0:
    case 1:
        return BinaryVectorFunc<Func, TR, T1, T2, 1>::run;
    case 2:
        return BinaryVectorFunc<Func, TR, T1, T2, 2>::run;
    case 3:
        return BinaryVectorFunc<Func, TR, T1, T2, 3>::run;
    case 4:
        return BinaryVectorFunc<Func, TR, T1, T2, 4>::run;
    case 6:
        return BinaryVectorFunc<Func, TR, T1, T2, 6>::run;
    case 8:
        return BinaryVectorFunc<Func, TR, T1, T2, 8>::run;
    case 9:
        return BinaryVectorFunc<Func, TR, T1, T2, 9>::run;
    case 10:
        return BinaryVectorFunc<Func, TR, T1, T2, 10>::run;
    case 12:
        return BinaryVectorFunc<Func, TR, T1, T2, 12>::run;
    case 16:
        return BinaryVectorFunc<Func, TR, T1, T2, 16>::run;
    default:
        return GeneralBinaryVectorFunc<Func, TR, T1, T2>::run;
    }
}

template<typename Func>
VMExtFunction binaryArithmeticInstHandler(uint32_t extCode)
{
    ArithmeticExtCode arithExtCode;
    memcpy(&arithExtCode, &extCode, sizeof(arithExtCode));
    switch (arithExtCode.scalarType)
    {
    case kSlangByteCodeScalarTypeSignedInt:
        switch (arithExtCode.scalarBitWidth)
        {
        case 0:
            return binaryArithmeticInstHandler<Func, int8_t>(arithExtCode.vectorSize);
        case 1:
            return binaryArithmeticInstHandler<Func, int16_t>(arithExtCode.vectorSize);
        case 2:
            return binaryArithmeticInstHandler<Func, int32_t>(arithExtCode.vectorSize);
        case 3:
            return binaryArithmeticInstHandler<Func, int64_t>(arithExtCode.vectorSize);
        }
    case kSlangByteCodeScalarTypeUnsignedInt:
        switch (arithExtCode.scalarBitWidth)
        {
        case 0:
            return binaryArithmeticInstHandler<Func, uint8_t>(arithExtCode.vectorSize);
        case 1:
            return binaryArithmeticInstHandler<Func, uint16_t>(arithExtCode.vectorSize);
        case 2:
            return binaryArithmeticInstHandler<Func, uint32_t>(arithExtCode.vectorSize);
        case 3:
            return binaryArithmeticInstHandler<Func, uint64_t>(arithExtCode.vectorSize);
        }
    case kSlangByteCodeScalarTypeFloat:
        switch (arithExtCode.scalarBitWidth)
        {
        case 2:
            return binaryArithmeticInstHandler<Func, float>(arithExtCode.vectorSize);
        case 3:
            return binaryArithmeticInstHandler<Func, double>(arithExtCode.vectorSize);
        default:
            return nullptr; // Unsupported scalar bit width
        }
    }
    return nullptr;
}

template<typename Func>
VMExtFunction binaryArithmeticLogicalInstHandler(uint32_t extCode)
{
    ArithmeticExtCode arithExtCode;
    memcpy(&arithExtCode, &extCode, sizeof(arithExtCode));
    switch (arithExtCode.scalarBitWidth)
    {
    case 0:
        return binaryArithmeticInstHandler<Func, uint8_t>(arithExtCode.vectorSize);
    case 1:
        return binaryArithmeticInstHandler<Func, uint16_t>(arithExtCode.vectorSize);
    case 2:
        return binaryArithmeticInstHandler<Func, uint32_t>(arithExtCode.vectorSize);
    case 3:
        return binaryArithmeticInstHandler<Func, uint64_t>(arithExtCode.vectorSize);
    }
    return nullptr;
}

template<typename Func>
VMExtFunction binaryArithmeticIntInstHandler(uint32_t extCode)
{
    ArithmeticExtCode arithExtCode;
    memcpy(&arithExtCode, &extCode, sizeof(arithExtCode));
    switch (arithExtCode.scalarType)
    {
    case kSlangByteCodeScalarTypeSignedInt:
        switch (arithExtCode.scalarBitWidth)
        {
        case 0:
            return binaryArithmeticInstHandler<Func, int8_t>(arithExtCode.vectorSize);
        case 1:
            return binaryArithmeticInstHandler<Func, int16_t>(arithExtCode.vectorSize);
        case 2:
            return binaryArithmeticInstHandler<Func, int32_t>(arithExtCode.vectorSize);
        case 3:
            return binaryArithmeticInstHandler<Func, int64_t>(arithExtCode.vectorSize);
        }
    case kSlangByteCodeScalarTypeUnsignedInt:
        switch (arithExtCode.scalarBitWidth)
        {
        case 0:
            return binaryArithmeticInstHandler<Func, uint8_t>(arithExtCode.vectorSize);
        case 1:
            return binaryArithmeticInstHandler<Func, uint16_t>(arithExtCode.vectorSize);
        case 2:
            return binaryArithmeticInstHandler<Func, uint32_t>(arithExtCode.vectorSize);
        case 3:
            return binaryArithmeticInstHandler<Func, uint64_t>(arithExtCode.vectorSize);
        }
    }
    return nullptr;
}

template<typename Func>
VMExtFunction binaryArithmeticCompareInstHandler(uint32_t extCode)
{
    ArithmeticExtCode arithExtCode;
    memcpy(&arithExtCode, &extCode, sizeof(arithExtCode));
    switch (arithExtCode.scalarType)
    {
    case kSlangByteCodeScalarTypeSignedInt:
        switch (arithExtCode.scalarBitWidth)
        {
        case 0:
            return binaryArithmeticInstHandler<Func, uint32_t, int8_t, int8_t>(
                arithExtCode.vectorSize);
        case 1:
            return binaryArithmeticInstHandler<Func, uint32_t, int16_t, int16_t>(
                arithExtCode.vectorSize);
        case 2:
            return binaryArithmeticInstHandler<Func, uint32_t, int32_t, int32_t>(
                arithExtCode.vectorSize);
        case 3:
            return binaryArithmeticInstHandler<Func, uint32_t, int64_t, int64_t>(
                arithExtCode.vectorSize);
        }
    case kSlangByteCodeScalarTypeUnsignedInt:
        switch (arithExtCode.scalarBitWidth)
        {
        case 0:
            return binaryArithmeticInstHandler<Func, uint32_t, uint8_t, uint8_t>(
                arithExtCode.vectorSize);
        case 1:
            return binaryArithmeticInstHandler<Func, uint32_t, uint16_t, uint16_t>(
                arithExtCode.vectorSize);
        case 2:
            return binaryArithmeticInstHandler<Func, uint32_t, uint32_t, uint32_t>(
                arithExtCode.vectorSize);
        case 3:
            return binaryArithmeticInstHandler<Func, uint32_t, uint64_t, uint64_t>(
                arithExtCode.vectorSize);
        }
    case kSlangByteCodeScalarTypeFloat:
        switch (arithExtCode.scalarBitWidth)
        {
        case 2:
            return binaryArithmeticInstHandler<Func, uint32_t, float, float>(
                arithExtCode.vectorSize);
        case 3:
            return binaryArithmeticInstHandler<Func, uint32_t, double, double>(
                arithExtCode.vectorSize);
        default:
            return nullptr; // Unsupported scalar bit width
        }
    }
    return nullptr;
}

////////
template<typename ScalarFunc, typename TR, typename T1, int elementCount>
struct UnaryVectorFunc
{
    static void run(IByteCodeRunner* context, VMExecInstHeader* inst, void* userData)
    {
        SLANG_UNUSED(context);
        SLANG_UNUSED(userData);
        TR* dst = (TR*)inst->getOperand(0).getPtr();
        T1* src1 = (T1*)inst->getOperand(1).getPtr();
        for (int i = 0; i < elementCount; ++i)
        {
            ScalarFunc::template run<TR, T1>(&dst[i], &src1[i]);
        }
    }
};

template<typename ScalarFunc, typename TR, typename T1>
struct GeneralUnaryVectorFunc
{
    static void run(IByteCodeRunner* context, VMExecInstHeader* inst, void* userData)
    {
        SLANG_UNUSED(context);
        SLANG_UNUSED(userData);
        TR* dst = (TR*)inst->getOperand(0).getPtr();
        T1* src1 = (T1*)inst->getOperand(1).getPtr();
        ArithmeticExtCode arithExtCode;
        memcpy(&arithExtCode, &inst->opcodeExtension, sizeof(arithExtCode));
        for (uint32_t i = 0; i < arithExtCode.vectorSize; ++i)
        {
            ScalarFunc::template run<TR, T1>(&dst[i], &src1[i]);
        }
    }
};

template<typename Func, typename TR, typename T1 = TR>
VMExtFunction unaryArithmeticInstHandler(int elementCount)
{
    switch (elementCount)
    {
    case 0:
    case 1:
        return UnaryVectorFunc<Func, TR, T1, 1>::run;
    case 2:
        return UnaryVectorFunc<Func, TR, T1, 2>::run;
    case 3:
        return UnaryVectorFunc<Func, TR, T1, 3>::run;
    case 4:
        return UnaryVectorFunc<Func, TR, T1, 4>::run;
    case 6:
        return UnaryVectorFunc<Func, TR, T1, 6>::run;
    case 8:
        return UnaryVectorFunc<Func, TR, T1, 8>::run;
    case 9:
        return UnaryVectorFunc<Func, TR, T1, 9>::run;
    case 10:
        return UnaryVectorFunc<Func, TR, T1, 10>::run;
    case 12:
        return UnaryVectorFunc<Func, TR, T1, 12>::run;
    case 16:
        return UnaryVectorFunc<Func, TR, T1, 16>::run;
    default:
        return GeneralUnaryVectorFunc<Func, TR, T1>::run;
    }
}

template<typename Func>
VMExtFunction unaryArithmeticLogicalInstHandler(uint32_t extCode)
{
    ArithmeticExtCode arithExtCode;
    memcpy(&arithExtCode, &extCode, sizeof(arithExtCode));
    switch (arithExtCode.scalarBitWidth)
    {
    case 0:
        return unaryArithmeticInstHandler<Func, uint8_t>(arithExtCode.vectorSize);
    case 1:
        return unaryArithmeticInstHandler<Func, uint16_t>(arithExtCode.vectorSize);
    case 2:
        return unaryArithmeticInstHandler<Func, uint32_t>(arithExtCode.vectorSize);
    case 3:
        return unaryArithmeticInstHandler<Func, uint64_t>(arithExtCode.vectorSize);
    }
    return nullptr;
}

template<typename Func>
VMExtFunction unaryArithmeticIntInstHandler(uint32_t extCode)
{
    ArithmeticExtCode arithExtCode;
    memcpy(&arithExtCode, &extCode, sizeof(arithExtCode));
    switch (arithExtCode.scalarType)
    {
    case kSlangByteCodeScalarTypeSignedInt:
        switch (arithExtCode.scalarBitWidth)
        {
        case 0:
            return unaryArithmeticInstHandler<Func, int8_t>(arithExtCode.vectorSize);
        case 1:
            return unaryArithmeticInstHandler<Func, int16_t>(arithExtCode.vectorSize);
        case 2:
            return unaryArithmeticInstHandler<Func, int32_t>(arithExtCode.vectorSize);
        case 3:
            return unaryArithmeticInstHandler<Func, int64_t>(arithExtCode.vectorSize);
        }
    case kSlangByteCodeScalarTypeUnsignedInt:
        switch (arithExtCode.scalarBitWidth)
        {
        case 0:
            return unaryArithmeticInstHandler<Func, uint8_t>(arithExtCode.vectorSize);
        case 1:
            return unaryArithmeticInstHandler<Func, uint16_t>(arithExtCode.vectorSize);
        case 2:
            return unaryArithmeticInstHandler<Func, uint32_t>(arithExtCode.vectorSize);
        case 3:
            return unaryArithmeticInstHandler<Func, uint64_t>(arithExtCode.vectorSize);
        }
    }
    return nullptr;
}

template<typename Func>
VMExtFunction negInstHandler(uint32_t extCode)
{
    ArithmeticExtCode arithExtCode;
    memcpy(&arithExtCode, &extCode, sizeof(arithExtCode));
    switch (arithExtCode.scalarType)
    {
    case kSlangByteCodeScalarTypeSignedInt:
    case kSlangByteCodeScalarTypeUnsignedInt:
        switch (arithExtCode.scalarBitWidth)
        {
        case 0:
            return unaryArithmeticInstHandler<Func, int8_t>(arithExtCode.vectorSize);
        case 1:
            return unaryArithmeticInstHandler<Func, int16_t>(arithExtCode.vectorSize);
        case 2:
            return unaryArithmeticInstHandler<Func, int32_t>(arithExtCode.vectorSize);
        case 3:
            return unaryArithmeticInstHandler<Func, int64_t>(arithExtCode.vectorSize);
        }
    case kSlangByteCodeScalarTypeFloat:
        switch (arithExtCode.scalarBitWidth)
        {
        case 2:
            return unaryArithmeticInstHandler<Func, float>(arithExtCode.vectorSize);
        case 3:
            return unaryArithmeticInstHandler<Func, double>(arithExtCode.vectorSize);
        default:
            return nullptr; // Unsupported scalar bit width
        }
    }
    return nullptr;
}

static void nopHandler(IByteCodeRunner*, VMExecInstHeader*, void*) {}

void callHandler(IByteCodeRunner* inCtx, VMExecInstHeader* inst, void*)
{
    auto ctx = convert(inCtx);
    auto funcId = inst->getOperand(1).offset;
    auto& func = ctx->m_functions[funcId];
    auto funcHeader = func.m_header;

    // Alloc working set.
    ctx->pushFrame(funcHeader->workingSetSizeInBytes);

    // Save current instruction pointer.
    auto& stackFrame = ctx->m_stack.getLast();
    stackFrame.m_currentInst = inst;
    stackFrame.m_currentFuncCode = ctx->m_currentFuncCode;
    auto newWorkingSetPtr = (uint8_t*)ctx->m_currentWorkingSet;
    auto callerWorkingSetPtr =
        (uint8_t*)(ctx->m_workingSetBuffer.getBuffer() + stackFrame.m_workingSetOffset);

    // Set working set pointer to the caller's working set.
    ctx->m_currentWorkingSet = callerWorkingSetPtr;

    // Copy arguments to the callee's working set.
    for (uint32_t i = 0; i < funcHeader->parameterCount; ++i)
    {
        auto dst = newWorkingSetPtr + func.m_parameterOffsets[i];
        auto src = (uint8_t*)inst->getOperand(i + 2).getPtr();

        // func.m_parameterOffsets should be initialized to contain parameterCount+1 elements,
        // where the last element is the total size of the parameters.
        auto nextParamOffset = func.m_parameterOffsets[i + 1];
        memcpy(dst, src, nextParamOffset - func.m_parameterOffsets[i]);
    }
    ctx->m_currentWorkingSet = newWorkingSetPtr;
    ctx->m_currentFuncCode = func.m_codeBuffer.getBuffer();
    ctx->m_currentInst = (VMExecInstHeader*)func.m_codeBuffer.getBuffer();
}

static void retHandler(IByteCodeRunner* inCtx, VMExecInstHeader* inst, void*)
{
    auto ctx = convert(inCtx);
    if (inst->opcodeExtension != 0)
    {
        void* resultPtr = nullptr;
        if (ctx->m_stack.getCount())
        {
            auto callInst = ctx->m_stack.getLast().m_currentInst;
            auto callerWorkingSetPtr = (uint8_t*)(ctx->m_workingSetBuffer.getBuffer() +
                                                  ctx->m_stack.getLast().m_workingSetOffset);
            resultPtr = callerWorkingSetPtr + callInst->getOperand(0).offset;
        }
        else
        {
            // If there is no stack frame, we assume the result is stored in the return register.
            ctx->m_returnRegister.setCount(inst->opcodeExtension);
            resultPtr = ctx->m_returnRegister.getBuffer();
            ctx->m_returnValSize = inst->opcodeExtension;
        }
        memcpy(resultPtr, inst->getOperand(0).getPtr(), inst->opcodeExtension);
    }

    // If we are returning from a main function, there is nothing to pop from the stack frame,
    // and we should stop execution.
    if (ctx->m_stack.getCount() == 0)
    {
        ctx->m_currentInst = nullptr;
        return;
    }

    // Pop the working set.
    ctx->popFrame();
}

static void jumpHandler(IByteCodeRunner* inCtx, VMExecInstHeader* inst, void*)
{
    auto ctx = convert(inCtx);
    ctx->m_currentInst = (VMExecInstHeader*)inst->getOperand(0).getPtr();
}

static void jumpIfHandler(IByteCodeRunner* inCtx, VMExecInstHeader* inst, void*)
{
    auto ctx = convert(inCtx);

    auto cond = *(uint32_t*)inst->getOperand(0).getPtr();
    if (cond)
    {
        ctx->m_currentInst = (VMExecInstHeader*)inst->getOperand(1).getPtr();
    }
    else
    {
        ctx->m_currentInst = (VMExecInstHeader*)inst->getOperand(2).getPtr();
    }
}

static void getWorkingSetPtrHandler(IByteCodeRunner* inCtx, VMExecInstHeader* inst, void*)
{
    auto ctx = convert(inCtx);
    auto dst = (void**)inst->getOperand(0).getPtr();
    auto ptr = (uint8_t*)ctx->m_currentWorkingSet + inst->opcodeExtension;
    *dst = ptr;
}

static void getElementPtrHandler(IByteCodeRunner* ctx, VMExecInstHeader* inst, void*)
{
    SLANG_UNUSED(ctx);
    auto dst = (void**)inst->getOperand(0).getPtr();
    auto basePtr = *(uint8_t**)inst->getOperand(1).getPtr();
    auto elementIndex = *(uint32_t*)inst->getOperand(2).getPtr();
    *dst = (uint8_t*)basePtr + elementIndex * inst->opcodeExtension;
}

static void getElementHandler(IByteCodeRunner* ctx, VMExecInstHeader* inst, void*)
{
    SLANG_UNUSED(ctx);
    auto dst = (void*)inst->getOperand(0).getPtr();
    auto basePtr = (uint8_t*)inst->getOperand(1).getPtr();
    auto elementIndex = *(uint32_t*)inst->getOperand(2).getPtr();
    memcpy(dst, basePtr + elementIndex * inst->opcodeExtension, inst->opcodeExtension);
}

static void offsetPtrHandler(IByteCodeRunner* ctx, VMExecInstHeader* inst, void*)
{
    SLANG_UNUSED(ctx);
    auto dst = (void**)inst->getOperand(0).getPtr();
    auto basePtr = *(uint8_t**)inst->getOperand(1).getPtr();
    auto offset = *(int32_t*)inst->getOperand(2).getPtr();
    *dst = basePtr + offset * inst->opcodeExtension;
}

void loadHandler8(IByteCodeRunner* ctx, VMExecInstHeader* inst, void*)
{
    SLANG_UNUSED(ctx);
    auto dst = (uint8_t*)inst->getOperand(0).getPtr();
    auto src = *(uint8_t**)inst->getOperand(1).getPtr();
    *dst = *src;
}
void loadHandler16(IByteCodeRunner* ctx, VMExecInstHeader* inst, void*)
{
    SLANG_UNUSED(ctx);
    auto dst = (uint16_t*)inst->getOperand(0).getPtr();
    auto src = *(uint16_t**)inst->getOperand(1).getPtr();
    *dst = *src;
}
void loadHandler32(IByteCodeRunner* ctx, VMExecInstHeader* inst, void*)
{
    SLANG_UNUSED(ctx);
    auto dst = (uint32_t*)inst->getOperand(0).getPtr();
    auto src = *(uint32_t**)inst->getOperand(1).getPtr();
    *dst = *src;
}
void loadHandler64(IByteCodeRunner* ctx, VMExecInstHeader* inst, void*)
{
    SLANG_UNUSED(ctx);
    auto dst = (uint64_t*)inst->getOperand(0).getPtr();
    auto src = *(uint64_t**)inst->getOperand(1).getPtr();
    *dst = *src;
}

void generalLoadHandler(IByteCodeRunner* ctx, VMExecInstHeader* inst, void*)
{
    SLANG_UNUSED(ctx);
    auto dst = (uint8_t*)inst->getOperand(0).getPtr();
    auto src = *(uint8_t**)inst->getOperand(1).getPtr();
    memcpy(dst, src, inst->opcodeExtension);
}

VMExtFunction getLoadHandler(uint32_t extCode)
{
    switch (extCode)
    {
    case 1:
        return loadHandler8;
    case 2:
        return loadHandler16;
    case 4:
        return loadHandler32;
    case 8:
        return loadHandler64;
    default:
        return generalLoadHandler;
    }
}

void storeHandler8(IByteCodeRunner* ctx, VMExecInstHeader* inst, void*)
{
    SLANG_UNUSED(ctx);
    auto dst = *(uint8_t**)inst->getOperand(0).getPtr();
    auto src = (uint8_t*)inst->getOperand(1).getPtr();
    *dst = *src;
}

void storeHandler16(IByteCodeRunner* ctx, VMExecInstHeader* inst, void*)
{
    SLANG_UNUSED(ctx);
    auto dst = *(uint16_t**)inst->getOperand(0).getPtr();
    auto src = (uint16_t*)inst->getOperand(1).getPtr();
    *dst = *src;
}

void storeHandler32(IByteCodeRunner* ctx, VMExecInstHeader* inst, void*)
{
    SLANG_UNUSED(ctx);
    auto dst = *(uint32_t**)inst->getOperand(0).getPtr();
    auto src = (uint32_t*)inst->getOperand(1).getPtr();
    *dst = *src;
}

void storeHandler64(IByteCodeRunner* ctx, VMExecInstHeader* inst, void*)
{
    SLANG_UNUSED(ctx);
    auto dst = *(uint64_t**)inst->getOperand(0).getPtr();
    auto src = (uint64_t*)inst->getOperand(1).getPtr();
    *dst = *src;
}

void generalStoreHandler(IByteCodeRunner* ctx, VMExecInstHeader* inst, void*)
{
    SLANG_UNUSED(ctx);
    auto dst = *(uint8_t**)inst->getOperand(0).getPtr();
    auto src = (uint8_t*)inst->getOperand(1).getPtr();
    memcpy(dst, src, inst->opcodeExtension);
}

VMExtFunction getStoreHandler(uint32_t extCode)
{
    switch (extCode)
    {
    case 1:
        return storeHandler8;
    case 2:
        return storeHandler16;
    case 4:
        return storeHandler32;
    case 8:
        return storeHandler64;
    default:
        return generalStoreHandler;
    }
}

void copyHandler8(IByteCodeRunner* ctx, VMExecInstHeader* inst, void*)
{
    SLANG_UNUSED(ctx);
    auto dst = (uint8_t*)inst->getOperand(0).getPtr();
    auto src = (uint8_t*)inst->getOperand(1).getPtr();
    *dst = *src;
}

void copyHandler16(IByteCodeRunner* ctx, VMExecInstHeader* inst, void*)
{
    SLANG_UNUSED(ctx);
    auto dst = (uint16_t*)inst->getOperand(0).getPtr();
    auto src = (uint16_t*)inst->getOperand(1).getPtr();
    *dst = *src;
}

void copyHandler32(IByteCodeRunner* ctx, VMExecInstHeader* inst, void*)
{
    SLANG_UNUSED(ctx);
    auto dst = (uint32_t*)inst->getOperand(0).getPtr();
    auto src = (uint32_t*)inst->getOperand(1).getPtr();
    *dst = *src;
}

void copyHandler64(IByteCodeRunner* ctx, VMExecInstHeader* inst, void*)
{
    SLANG_UNUSED(ctx);
    auto dst = (uint64_t*)inst->getOperand(0).getPtr();
    auto src = (uint64_t*)inst->getOperand(1).getPtr();
    *dst = *src;
}

void generalCopyHandler(IByteCodeRunner* ctx, VMExecInstHeader* inst, void*)
{
    SLANG_UNUSED(ctx);
    auto dst = (uint8_t*)inst->getOperand(0).getPtr();
    auto src = (uint8_t*)inst->getOperand(1).getPtr();
    memcpy(dst, src, inst->opcodeExtension);
}

VMExtFunction getCopyHandler(uint32_t extCode)
{
    switch (extCode)
    {
    case 1:
        return copyHandler8;
    case 2:
        return copyHandler16;
    case 4:
        return copyHandler32;
    case 8:
        return copyHandler64;
    default:
        return generalCopyHandler;
    }
}

template<typename T>
void swizzleHandler(IByteCodeRunner* ctx, VMExecInstHeader* inst, void* userData)
{
    SLANG_UNUSED(ctx);
    SLANG_UNUSED(userData);
    auto dst = (T*)inst->getOperand(0).getPtr();
    auto src = (T*)inst->getOperand(1).getPtr();
    for (uint32_t i = 2; i < inst->operandCount; ++i)
    {
        dst[i - 2] = src[inst->getOperand(i).offset];
    }
}

VMExtFunction getSwizzleHandler(uint32_t extCode)
{
    ArithmeticExtCode arithExtCode;
    memcpy(&arithExtCode, &extCode, sizeof(arithExtCode));
    switch (arithExtCode.scalarType)
    {
    case kSlangByteCodeScalarTypeSignedInt:
        switch (arithExtCode.scalarBitWidth)
        {
        case 0:
            return swizzleHandler<int8_t>;
        case 1:
            return swizzleHandler<int16_t>;
        case 2:
            return swizzleHandler<int32_t>;
        case 3:
            return swizzleHandler<int64_t>;
        }
    case kSlangByteCodeScalarTypeUnsignedInt:
        switch (arithExtCode.scalarBitWidth)
        {
        case 0:
            return swizzleHandler<uint8_t>;
        case 1:
            return swizzleHandler<uint16_t>;
        case 2:
            return swizzleHandler<uint32_t>;
        case 3:
            return swizzleHandler<uint64_t>;
        }
    case kSlangByteCodeScalarTypeFloat:
        switch (arithExtCode.scalarBitWidth)
        {
        case 2:
            return swizzleHandler<float>;
        case 3:
            return swizzleHandler<double>;
        default:
            return nullptr; // Unsupported scalar bit width
        }
    }
    return nullptr;
}

template<typename To, typename From, int vectorSize>
void castHandler(IByteCodeRunner* ctx, VMExecInstHeader* inst, void*)
{
    SLANG_UNUSED(ctx);
    To* dst = (To*)inst->getOperand(0).getPtr();
    From* src = (From*)inst->getOperand(1).getPtr();
    for (int i = 0; i < vectorSize; ++i)
    {
        dst[i] = static_cast<To>(src[i]);
    }
}

template<typename From, int vectorSize>
VMExtFunction getCastHandler(uint32_t extCode)
{
    ArithmeticExtCode arithExtCode;
    memcpy(&arithExtCode, &extCode, sizeof(arithExtCode));
    switch (arithExtCode.scalarType)
    {
    case kSlangByteCodeScalarTypeSignedInt:
        switch (arithExtCode.scalarBitWidth)
        {
        case 0:
            return castHandler<uint8_t, From, vectorSize>;
        case 1:
            return castHandler<uint16_t, From, vectorSize>;
        case 2:
            return castHandler<uint32_t, From, vectorSize>;
        case 3:
            return castHandler<uint64_t, From, vectorSize>;
        }
    case kSlangByteCodeScalarTypeUnsignedInt:
        switch (arithExtCode.scalarBitWidth)
        {
        case 0:
            return castHandler<uint8_t, From, vectorSize>;
        case 1:
            return castHandler<uint16_t, From, vectorSize>;
        case 2:
            return castHandler<uint32_t, From, vectorSize>;
        case 3:
            return castHandler<uint64_t, From, vectorSize>;
        }
    case kSlangByteCodeScalarTypeFloat:
        switch (arithExtCode.scalarBitWidth)
        {
        case 2:
            return castHandler<float, From, vectorSize>;
        case 3:
            return castHandler<double, From, vectorSize>;
        default:
            return nullptr; // Unsupported scalar bit width
        }
    }
    return nullptr;
}

template<int vectorSize>
VMExtFunction getCastHandler(uint32_t extCode)
{
    uint32_t fromExtCode = extCode >> 16;
    ArithmeticExtCode arithExtCode;
    memcpy(&arithExtCode, &fromExtCode, sizeof(arithExtCode));
    switch (arithExtCode.scalarType)
    {
    case kSlangByteCodeScalarTypeSignedInt:
        switch (arithExtCode.scalarBitWidth)
        {
        case 0:
            return getCastHandler<uint8_t, vectorSize>(extCode);
        case 1:
            return getCastHandler<uint16_t, vectorSize>(extCode);
        case 2:
            return getCastHandler<uint32_t, vectorSize>(extCode);
        case 3:
            return getCastHandler<uint64_t, vectorSize>(extCode);
        }
    case kSlangByteCodeScalarTypeUnsignedInt:
        switch (arithExtCode.scalarBitWidth)
        {
        case 0:
            return getCastHandler<uint8_t, vectorSize>(extCode);
        case 1:
            return getCastHandler<uint16_t, vectorSize>(extCode);
        case 2:
            return getCastHandler<uint32_t, vectorSize>(extCode);
        case 3:
            return getCastHandler<uint64_t, vectorSize>(extCode);
        }
    case kSlangByteCodeScalarTypeFloat:
        switch (arithExtCode.scalarBitWidth)
        {
        case 2:
            return getCastHandler<float, vectorSize>(extCode);
        case 3:
            return getCastHandler<double, vectorSize>(extCode);
        default:
            return nullptr; // Unsupported scalar bit width
        }
    }
    return nullptr;
}

VMExtFunction getCastHandler(uint32_t extCode)
{
    uint32_t fromExtCode = extCode >> 16;
    ArithmeticExtCode arithExtCode;
    memcpy(&arithExtCode, &fromExtCode, sizeof(arithExtCode));
    switch (arithExtCode.vectorSize)
    {
    case 0:
    case 1:
        return getCastHandler<1>(extCode);
    case 2:
        return getCastHandler<2>(extCode);
    case 3:
        return getCastHandler<3>(extCode);
    case 4:
        return getCastHandler<4>(extCode);
    case 6:
        return getCastHandler<6>(extCode);
    case 8:
        return getCastHandler<8>(extCode);
    case 9:
        return getCastHandler<9>(extCode);
    case 12:
        return getCastHandler<12>(extCode);
    case 16:
        return getCastHandler<16>(extCode);
    }
    return nullptr;
}

void printHandler(IByteCodeRunner* inCtx, VMExecInstHeader* inst, void* userData)
{
    auto ctx = convert(inCtx);
    SLANG_UNUSED(userData);
    const char* formatString = nullptr;
    formatString = *(const char**)inst->getOperand(0).getPtr();

    List<List<uint8_t>> args;
    List<const void*> argPtrs;
    for (uint32_t i = 1; i < inst->operandCount; ++i)
    {
        auto& arg = inst->getOperand(i);
        List<uint8_t> data;
        data.setCount(arg.size);
        memcpy(data.getBuffer(), arg.getPtr(), arg.size);
        args.add(data);
    }
    for (auto& arg : args)
    {
        argPtrs.add(arg.getBuffer());
    }
    auto result =
        StringUtil::makeStringWithFormatFromArgArray(formatString, argPtrs.getArrayView());
    ctx->m_printCallback(result.getBuffer(), ctx->m_printCallbackUserData);
}


VMExtFunction mapInstToFunction(
    VMInstHeader* instHeader,
    VMModuleView* module,
    Dictionary<String, slang::VMExtFunction>& extInstHandlers)
{
    switch (instHeader->opcode)
    {
    case VMOp::Nop:
        return nopHandler;
    case VMOp::Add:
        return binaryArithmeticInstHandler<AddScalarFunc>(instHeader->opcodeExtension);
    case VMOp::Sub:
        return binaryArithmeticInstHandler<SubScalarFunc>(instHeader->opcodeExtension);
    case VMOp::Mul:
        return binaryArithmeticInstHandler<MulScalarFunc>(instHeader->opcodeExtension);
    case VMOp::Div:
        return binaryArithmeticInstHandler<DivScalarFunc>(instHeader->opcodeExtension);
    case VMOp::Rem:
        return binaryArithmeticInstHandler<ModScalarFunc>(instHeader->opcodeExtension);
    case VMOp::And:
        return binaryArithmeticLogicalInstHandler<AndScalarFunc>(instHeader->opcodeExtension);
    case VMOp::Or:
        return binaryArithmeticLogicalInstHandler<OrScalarFunc>(instHeader->opcodeExtension);
    case VMOp::BitAnd:
        return binaryArithmeticLogicalInstHandler<BitAndScalarFunc>(instHeader->opcodeExtension);
    case VMOp::BitOr:
        return binaryArithmeticLogicalInstHandler<BitOrScalarFunc>(instHeader->opcodeExtension);
    case VMOp::BitXor:
        return binaryArithmeticLogicalInstHandler<BitXorScalarFunc>(instHeader->opcodeExtension);
    case VMOp::Shl:
        return binaryArithmeticIntInstHandler<ShlScalarFunc>(instHeader->opcodeExtension);
    case VMOp::Shr:
        return binaryArithmeticIntInstHandler<ShrScalarFunc>(instHeader->opcodeExtension);
    case VMOp::Less:
        return binaryArithmeticCompareInstHandler<LessScalarFunc>(instHeader->opcodeExtension);
    case VMOp::Leq:
        return binaryArithmeticCompareInstHandler<LeqScalarFunc>(instHeader->opcodeExtension);
    case VMOp::Greater:
        return binaryArithmeticCompareInstHandler<GreaterScalarFunc>(instHeader->opcodeExtension);
    case VMOp::Geq:
        return binaryArithmeticCompareInstHandler<GeqScalarFunc>(instHeader->opcodeExtension);
    case VMOp::Equal:
        return binaryArithmeticCompareInstHandler<EqualScalarFunc>(instHeader->opcodeExtension);
    case VMOp::Neq:
        return binaryArithmeticCompareInstHandler<NeqScalarFunc>(instHeader->opcodeExtension);
    case VMOp::Neg:
        return negInstHandler<NegScalarFunc>(instHeader->opcodeExtension);
    case VMOp::Not:
        return unaryArithmeticLogicalInstHandler<NotScalarFunc>(instHeader->opcodeExtension);
    case VMOp::BitNot:
        return unaryArithmeticIntInstHandler<BitNotScalarFunc>(instHeader->opcodeExtension);
    case VMOp::Ret:
        return retHandler;
    case VMOp::Call:
        return callHandler;
    case VMOp::Jump:
        return jumpHandler;
    case VMOp::JumpIf:
        return jumpIfHandler;
    case VMOp::Load:
        return getLoadHandler(instHeader->opcodeExtension);
    case VMOp::Store:
        return getStoreHandler(instHeader->opcodeExtension);
    case VMOp::Copy:
        return getCopyHandler(instHeader->opcodeExtension);
    case VMOp::GetWorkingSetPtr:
        return getWorkingSetPtrHandler;
    case VMOp::GetElementPtr:
        return getElementPtrHandler;
    case VMOp::OffsetPtr:
        return offsetPtrHandler;
    case VMOp::GetElement:
        return getElementHandler;
    case VMOp::Swizzle:
        return getSwizzleHandler(instHeader->opcodeExtension);
    case VMOp::Cast:
        return getCastHandler(instHeader->opcodeExtension);
    case VMOp::CallExt:
        {
            if (instHeader->getOperand(0).offset >= module->stringCount)
                return nullptr;
            auto funcName = (const char*)module->constants +
                            module->stringOffsets[instHeader->getOperand(0).offset];
            VMExtFunction handler = nullptr;
            if (!extInstHandlers.tryGetValue(funcName, handler))
                return nullptr;
            return handler;
        }
    case VMOp::Print:
        return printHandler;
    }
    return VMExtFunction();
}

} // namespace Slang
