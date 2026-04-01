// slang-emit-cuda.cpp
#include "slang-emit-cuda.h"

#include "../core/slang-writer.h"
#include "slang-emit-source-writer.h"
#include "slang-mangled-lexer.h"

#include <assert.h>

namespace Slang
{

static CUDAExtensionTracker::BaseTypeFlags _findBaseTypesUsed(IRModule* module)
{
    typedef CUDAExtensionTracker::BaseTypeFlags Flags;

    // All basic types are hoistable so must be in global scope.
    Flags baseTypesUsed = 0;

    auto moduleInst = module->getModuleInst();

    // Search all the insts in global scope, for BasicTypes
    for (auto inst : moduleInst->getChildren())
    {
        if (auto basicType = as<IRBasicType>(inst))
        {
            // Get the base type, and set the bit
            const auto baseTypeEnum = basicType->getBaseType();

            baseTypesUsed |= Flags(1) << int(baseTypeEnum);
        }
    }

    return baseTypesUsed;
}

void CUDAExtensionTracker::finalize()
{
    if (isBaseTypeRequired(BaseType::Half))
    {
        // The cuda_fp16.hpp header indicates the need is for version 5.3, but when this is tried
        // NVRTC says it cannot load builtins.
        // The lowest version that this does work for is 6.0, so that's what we use here.

        // https://docs.nvidia.com/cuda/nvrtc/index.html#group__options
        requireSMVersion(SemanticVersion(6, 0));
    }
}

UnownedStringSlice CUDASourceEmitter::getBuiltinTypeName(IROp op)
{
    switch (op)
    {
    case kIROp_VoidType:
        return UnownedStringSlice("void");
    case kIROp_BoolType:
        return UnownedStringSlice("bool");

    case kIROp_Int8Type:
        return UnownedStringSlice("char");
    case kIROp_Int16Type:
        return UnownedStringSlice("short");
    case kIROp_IntType:
        return UnownedStringSlice("int");
    case kIROp_Int64Type:
        return UnownedStringSlice("longlong");

    case kIROp_UInt8Type:
        return UnownedStringSlice("uchar");
    case kIROp_UInt16Type:
        return UnownedStringSlice("ushort");
    case kIROp_UIntType:
        return UnownedStringSlice("uint");
    case kIROp_UInt64Type:
        return UnownedStringSlice("ulonglong");
#if SLANG_PTR_IS_64
    case kIROp_IntPtrType:
        return UnownedStringSlice("int64_t");
    case kIROp_UIntPtrType:
        return UnownedStringSlice("uint64_t");
#else
    case kIROp_IntPtrType:
        return UnownedStringSlice("int");
    case kIROp_UIntPtrType:
        return UnownedStringSlice("uint");
#endif

    case kIROp_HalfType:
        return UnownedStringSlice("__half");

    case kIROp_FloatType:
        return UnownedStringSlice("float");
    case kIROp_DoubleType:
        return UnownedStringSlice("double");
    default:
        return UnownedStringSlice();
    }
}


UnownedStringSlice CUDASourceEmitter::getVectorPrefix(IROp op)
{
    switch (op)
    {
    case kIROp_BoolType:
        return UnownedStringSlice("bool");

    case kIROp_Int8Type:
        return UnownedStringSlice("char");
    case kIROp_Int16Type:
        return UnownedStringSlice("short");
    case kIROp_IntType:
        return UnownedStringSlice("int");
    case kIROp_Int64Type:
        return UnownedStringSlice("longlong");

    case kIROp_UInt8Type:
        return UnownedStringSlice("uchar");
    case kIROp_UInt16Type:
        return UnownedStringSlice("ushort");
    case kIROp_UIntType:
        return UnownedStringSlice("uint");
    case kIROp_UInt64Type:
        return UnownedStringSlice("ulonglong");

    case kIROp_HalfType:
        return UnownedStringSlice("__half");

    case kIROp_FloatType:
        return UnownedStringSlice("float");
    case kIROp_DoubleType:
        return UnownedStringSlice("double");
    default:
        return UnownedStringSlice();
    }
}

void CUDASourceEmitter::emitTempModifiers(IRInst* temp)
{
    CPPSourceEmitter::emitTempModifiers(temp);
    if (as<IRModuleInst>(temp->getParent()))
    {
        m_writer->emit("__device__ ");
    }
}

SlangResult CUDASourceEmitter::_calcCUDATextureTypeName(
    IRTextureTypeBase* texType,
    StringBuilder& outName)
{
    // Not clear how to do this yet
    if (texType->isMultisample())
    {
        return SLANG_FAIL;
    }

    switch (texType->getAccess())
    {
    case SLANG_RESOURCE_ACCESS_READ:
        {
            outName << "CUtexObject";
            return SLANG_OK;
        }
    case SLANG_RESOURCE_ACCESS_READ_WRITE:
    case SLANG_RESOURCE_ACCESS_RASTER_ORDERED:
    case SLANG_RESOURCE_ACCESS_WRITE:
        {
            outName << "CUsurfObject";
            return SLANG_OK;
        }
    default:
        break;
    }
    return SLANG_FAIL;
}

SlangResult CUDASourceEmitter::calcTypeName(IRType* type, CodeGenTarget target, StringBuilder& out)
{
    SLANG_UNUSED(target);

    // The names CUDA produces are all compatible with 'C' (ie they aren't templated types)
    SLANG_ASSERT(target == CodeGenTarget::CUDASource || target == CodeGenTarget::CSource);

    switch (type->getOp())
    {
    case kIROp_VectorType:
        {
            auto vecType = static_cast<IRVectorType*>(type);
            auto vecCount = int(getIntVal(vecType->getElementCount()));
            const IROp elemType = vecType->getElementType()->getOp();

            UnownedStringSlice prefix = getVectorPrefix(elemType);
            if (prefix.getLength() <= 0)
            {
                return SLANG_FAIL;
            }
            out << prefix << vecCount;
            return SLANG_OK;
        }
    case kIROp_TensorViewType:
        {
            out << "TensorView";
            return SLANG_OK;
        }
    default:
        {
            if (isNominalOp(type->getOp()))
            {
                out << getName(type);
                return SLANG_OK;
            }

            if (IRBasicType::isaImpl(type->getOp()))
            {
                out << getBuiltinTypeName(type->getOp());
                return SLANG_OK;
            }

            if (auto texType = as<IRTextureTypeBase>(type))
            {
                return _calcCUDATextureTypeName(texType, out);
            }

            switch (type->getOp())
            {
            case kIROp_SamplerStateType:
                out << "SamplerState";
                return SLANG_OK;
            case kIROp_SamplerComparisonStateType:
                out << "SamplerComparisonState";
                return SLANG_OK;
            default:
                break;
            }

            break;
        }
    }

    if (auto untypedBufferType = as<IRUntypedBufferResourceType>(type))
    {
        switch (untypedBufferType->getOp())
        {
        case kIROp_RaytracingAccelerationStructureType:
            {
                m_writer->emit("OptixTraversableHandle");
                return SLANG_OK;
                break;
            }

        default:
            break;
        }
    }

    return Super::calcTypeName(type, target, out);
}

void CUDASourceEmitter::emitLayoutSemanticsImpl(
    IRInst* inst,
    char const* uniformSemanticSpelling,
    EmitLayoutSemanticOption layoutSemanticOption)
{
    Super::emitLayoutSemanticsImpl(inst, uniformSemanticSpelling, layoutSemanticOption);
}

void CUDASourceEmitter::emitParameterGroupImpl(
    IRGlobalParam* varDecl,
    IRUniformParameterGroupType* type)
{
    auto elementType = type->getElementType();

    m_writer->emit("extern \"C\" __constant__ ");
    emitType(elementType, "SLANG_globalParams");
    m_writer->emit(";\n");

    m_writer->emit("#define ");
    m_writer->emit(getName(varDecl));
    m_writer->emit(" (&SLANG_globalParams)\n");
}

void CUDASourceEmitter::emitEntryPointAttributesImpl(
    IRFunc* irFunc,
    IREntryPointDecoration* entryPointDecor)
{
    SLANG_UNUSED(irFunc);
    SLANG_UNUSED(entryPointDecor);
}

void CUDASourceEmitter::emitFunctionPreambleImpl(IRInst* inst)
{
    if (!inst)
        return;
    if (inst->findDecoration<IREntryPointDecoration>())
    {
        m_writer->emit("extern \"C\" __global__ ");
        return;
    }

    if (inst->findDecoration<IRCudaKernelDecoration>())
    {
        m_writer->emit("__global__ ");
    }
    else if (inst->findDecoration<IRCudaHostDecoration>())
    {
        m_writer->emit("__host__ ");
    }
    else
    {
        m_writer->emit("__device__ ");
    }
}

String CUDASourceEmitter::generateEntryPointNameImpl(IREntryPointDecoration* entryPointDecor)
{
    // We have an entry-point function in the IR module, which we
    // will want to emit as a `__global__` function in the generated
    // CUDA C++.
    //
    // The most common case will be a compute kernel, in which case
    // we will emit the function more or less as-is, including
    // usingits original name as the name of the global symbol.
    //
    String funcName = Super::generateEntryPointNameImpl(entryPointDecor);
    String globalSymbolName = funcName;

    // We also suport emitting ray tracing kernels for use with
    // OptiX, and in that case the name of the global symbol
    // must be prefixed to indicate to the OptiX runtime what
    // stage it is to be compiled for.
    //
    auto stage = entryPointDecor->getProfile().getStage();
    switch (stage)
    {
    default:
        break;

#define CASE(STAGE, PREFIX)                    \
    case Stage::STAGE:                         \
        globalSymbolName = #PREFIX + funcName; \
        break

        // Optix 7 Guide, Section 6.1 (Program input)
        //
        // > The input PTX should include one or more NVIDIA OptiX programs.
        // > The type of program affects how the program can be used during
        // > the execution of the pipeline. These program types are specified
        // by prefixing the program name with the following:
        //
        // >    Program type        Function name prefix
        CASE(RayGeneration, __raygen__);
        CASE(Intersection, __intersection__);
        CASE(AnyHit, __anyhit__);
        CASE(ClosestHit, __closesthit__);
        CASE(Miss, __miss__);
        CASE(Callable, __direct_callable__);
        //
        // There are two stages (or "program types") supported by OptiX
        // that Slang currently cannot target:
        //
        // CASE(ContinuationCallable,   __continuation_callable__);
        // CASE(Exception,              __exception__);
        //
#undef CASE
    }

    return globalSymbolName;
}

void CUDASourceEmitter::emitGlobalRTTISymbolPrefix()
{
    m_writer->emit("__constant__ ");
}

void CUDASourceEmitter::emitLoopControlDecorationImpl(IRLoopControlDecoration* decl)
{
    if (decl->getMode() == kIRLoopControl_Unroll)
    {
        m_writer->emit("#pragma unroll\n");
    }
}

void CUDASourceEmitter::_emitInitializerListValue(IRType* dstType, IRInst* value)
{
    // When constructing a matrix or vector from a single value this is handled by the default path

    switch (value->getOp())
    {
    case kIROp_MakeVector:
    case kIROp_MakeMatrix:
        {
            IRType* type = value->getDataType();

            // If the types are the same, we can can just break down and use
            if (dstType == type)
            {
                if (auto vecType = as<IRVectorType>(type))
                {
                    if (UInt(getIntVal(vecType->getElementCount())) == value->getOperandCount())
                    {
                        emitType(type);
                        _emitInitializerList(
                            vecType->getElementType(),
                            value->getOperands(),
                            value->getOperandCount());
                        return;
                    }
                }
                else if (auto matType = as<IRMatrixType>(type))
                {
                    const Index colCount = Index(getIntVal(matType->getColumnCount()));
                    const Index rowCount = Index(getIntVal(matType->getRowCount()));

                    // TODO(JS): If num cols = 1, then it *doesn't* actually return a vector.
                    // That could be argued is an error because we want swizzling or [] to work.
                    IRBuilder builder(matType->getModule());
                    builder.setInsertBefore(matType);
                    const Index operandCount = Index(value->getOperandCount());

                    // Can init, with vectors.
                    // For now special case if the rowVectorType is not actually a vector (when
                    // elementSize == 1)
                    if (operandCount == rowCount)
                    {
                        // Emit the braces for the Matrix struct, and then each row vector in its
                        // own line.
                        emitType(matType);
                        m_writer->emit("{\n");
                        m_writer->indent();
                        for (Index i = 0; i < rowCount; ++i)
                        {
                            if (i != 0)
                                m_writer->emit(",\n");
                            emitType(matType->getElementType());
                            m_writer->emit(colCount);
                            _emitInitializerList(
                                matType->getElementType(),
                                value->getOperand(i)->getOperands(),
                                colCount);
                        }
                        m_writer->dedent();
                        m_writer->emit("\n}");
                        return;
                    }
                    else if (operandCount == rowCount * colCount)
                    {
                        // Handle if all are explicitly defined
                        IRType* elementType = matType->getElementType();
                        IRUse* operands = value->getOperands();

                        // Emit the braces for the Matrix struct, and the elements of each row in
                        // its own line.
                        emitType(matType);
                        m_writer->emit("{\n");
                        m_writer->indent();
                        for (Index i = 0; i < rowCount; ++i)
                        {
                            if (i != 0)
                                m_writer->emit(",\n");
                            _emitInitializerListContent(elementType, operands, colCount);
                            operands += colCount;
                        }
                        m_writer->dedent();
                        m_writer->emit("\n}");
                        return;
                    }
                }
            }

            break;
        }
    }

    // All other cases we just use the default emitting - might not work on arrays defined in global
    // scope on CUDA though
    emitOperand(value, getInfo(EmitOp::General));
}

void CUDASourceEmitter::_emitInitializerListContent(
    IRType* elementType,
    IRUse* operands,
    Index operandCount)
{
    for (Index i = 0; i < operandCount; ++i)
    {
        if (i != 0)
            m_writer->emit(", ");
        _emitInitializerListValue(elementType, operands[i].get());
    }
}


void CUDASourceEmitter::_emitInitializerList(
    IRType* elementType,
    IRUse* operands,
    Index operandCount)
{
    m_writer->emit("{\n");
    m_writer->indent();

    _emitInitializerListContent(elementType, operands, operandCount);

    m_writer->dedent();
    m_writer->emit("\n}");
}

void CUDASourceEmitter::emitIntrinsicCallExprImpl(
    IRCall* inst,
    UnownedStringSlice intrinsicDefinition,
    IRInst* intrinsicInst,
    EmitOpInfo const& inOuterPrec)
{
    // This works around the problem, where some intrinsics that require the "half" type enabled
    // don't use the half/float16_t type. For example `f16tof32` can operate on float16_t *and*
    // uint. If the input is uint, although we are using the half feature (as far as CUDA is
    // concerned), the half/float16_t type is not visible/directly used.
    if (intrinsicDefinition.startsWith(toSlice("__half")))
    {
        m_extensionTracker->requireBaseType(BaseType::Half);
    }

    Super::emitIntrinsicCallExprImpl(inst, intrinsicDefinition, intrinsicInst, inOuterPrec);
}

bool CUDASourceEmitter::tryEmitInstStmtImpl(IRInst* inst)
{
    switch (inst->getOp())
    {
    case kIROp_StructuredBufferGetDimensions:
        {
            auto count = _generateUniqueName(UnownedStringSlice("_elementCount"));
            auto stride = _generateUniqueName(UnownedStringSlice("_stride"));

            m_writer->emit("uint ");
            m_writer->emit(count);
            m_writer->emit(";\n");
            m_writer->emit("uint ");
            m_writer->emit(stride);
            m_writer->emit(";\n");
            emitOperand(
                inst->getOperand(0),
                leftSide(getInfo(EmitOp::General), getInfo(EmitOp::Postfix)));
            m_writer->emit(".GetDimensions(&");
            m_writer->emit(count);
            m_writer->emit(", &");
            m_writer->emit(stride);
            m_writer->emit(");\n");
            emitInstResultDecl(inst);
            m_writer->emit("make_uint2(");
            m_writer->emit(count);
            m_writer->emit(", ");
            m_writer->emit(stride);
            m_writer->emit(");\n");
            return true;
        }
    case kIROp_AtomicLoad:
        {
            emitInstResultDecl(inst);
            emitDereferenceOperand(inst->getOperand(0), getInfo(EmitOp::General));
            m_writer->emit(";\n");
            return true;
        }
    case kIROp_AtomicStore:
        {
            emitDereferenceOperand(inst->getOperand(0), getInfo(EmitOp::General));
            m_writer->emit(" = ");
            emitOperand(inst->getOperand(1), getInfo(EmitOp::General));
            m_writer->emit(";\n");
            return true;
        }
    case kIROp_AtomicExchange:
        {
            emitInstResultDecl(inst);
            m_writer->emit("atomicExch(");
            emitOperand(inst->getOperand(0), getInfo(EmitOp::General));
            m_writer->emit(", ");
            emitOperand(inst->getOperand(1), getInfo(EmitOp::General));
            m_writer->emit(");\n");
            return true;
        }
    case kIROp_AtomicCompareExchange:
        {
            emitInstResultDecl(inst);
            m_writer->emit("atomicCAS(");
            emitOperand(inst->getOperand(0), getInfo(EmitOp::General));
            m_writer->emit(", ");
            emitOperand(inst->getOperand(1), getInfo(EmitOp::General));
            m_writer->emit(", ");
            emitOperand(inst->getOperand(2), getInfo(EmitOp::General));
            m_writer->emit(");\n");
            return true;
        }
    case kIROp_AtomicAdd:
        {
            emitInstResultDecl(inst);
            m_writer->emit("atomicAdd(");
            bool needCloseTypeCast = false;
            if (inst->getDataType()->getOp() == kIROp_Int64Type)
            {
                m_writer->emit("(unsigned long long*)(");
                needCloseTypeCast = true;
            }
            emitOperand(inst->getOperand(0), getInfo(EmitOp::General));
            if (needCloseTypeCast)
            {
                m_writer->emit(")");
            }
            m_writer->emit(", ");
            emitOperand(inst->getOperand(1), getInfo(EmitOp::General));
            m_writer->emit(");\n");
            return true;
        }
    case kIROp_AtomicSub:
        {
            emitInstResultDecl(inst);
            m_writer->emit("atomicAdd(");
            bool needCloseTypeCast = false;
            if (inst->getDataType()->getOp() == kIROp_Int64Type)
            {
                m_writer->emit("(unsigned long long*)(");
                needCloseTypeCast = true;
            }
            emitOperand(inst->getOperand(0), getInfo(EmitOp::General));
            if (needCloseTypeCast)
            {
                m_writer->emit(")");
            }
            m_writer->emit(", -(");
            emitOperand(inst->getOperand(1), getInfo(EmitOp::General));
            m_writer->emit("));\n");
            return true;
        }
    case kIROp_AtomicAnd:
        {
            emitInstResultDecl(inst);
            m_writer->emit("atomicAnd(");
            emitOperand(inst->getOperand(0), getInfo(EmitOp::General));
            m_writer->emit(", ");
            emitOperand(inst->getOperand(1), getInfo(EmitOp::General));
            m_writer->emit(");\n");
            return true;
        }
    case kIROp_AtomicOr:
        {
            emitInstResultDecl(inst);
            m_writer->emit("atomicOr(");
            emitOperand(inst->getOperand(0), getInfo(EmitOp::General));
            m_writer->emit(", ");
            emitOperand(inst->getOperand(1), getInfo(EmitOp::General));
            m_writer->emit(");\n");
            return true;
        }
    case kIROp_AtomicXor:
        {
            emitInstResultDecl(inst);
            m_writer->emit("atomicXor(");
            emitOperand(inst->getOperand(0), getInfo(EmitOp::General));
            m_writer->emit(", ");
            emitOperand(inst->getOperand(1), getInfo(EmitOp::General));
            m_writer->emit(");\n");
            return true;
        }
    case kIROp_AtomicMin:
        {
            emitInstResultDecl(inst);
            m_writer->emit("atomicMin(");
            emitOperand(inst->getOperand(0), getInfo(EmitOp::General));
            m_writer->emit(", ");
            emitOperand(inst->getOperand(1), getInfo(EmitOp::General));
            m_writer->emit(");\n");
            return true;
        }
    case kIROp_AtomicMax:
        {
            emitInstResultDecl(inst);
            m_writer->emit("atomicMax(");
            emitOperand(inst->getOperand(0), getInfo(EmitOp::General));
            m_writer->emit(", ");
            emitOperand(inst->getOperand(1), getInfo(EmitOp::General));
            m_writer->emit(");\n");
            return true;
        }
    case kIROp_AtomicInc:
        {
            emitInstResultDecl(inst);
            m_writer->emit("atomicAdd(");
            emitOperand(inst->getOperand(0), getInfo(EmitOp::General));
            m_writer->emit(", 1);\n");
            return true;
        }
    case kIROp_AtomicDec:
        {
            emitInstResultDecl(inst);
            m_writer->emit("atomicAdd(");
            emitOperand(inst->getOperand(0), getInfo(EmitOp::General));
            m_writer->emit(", -1);\n");
            return true;
        }
    default:
        return false;
    }
}

bool CUDASourceEmitter::tryEmitInstExprImpl(IRInst* inst, const EmitOpInfo& inOuterPrec)
{
    switch (inst->getOp())
    {
    case kIROp_MakeVector:
    case kIROp_MakeVectorFromScalar:
        {
            m_writer->emit("make_");
            emitType(inst->getDataType());
            m_writer->emit("(");
            bool isFirst = true;
            char xyzwNames[] = "xyzw";
            for (UInt i = 0; i < inst->getOperandCount(); i++)
            {
                auto arg = inst->getOperand(i);
                if (auto vectorType = as<IRVectorType>(arg->getDataType()))
                {
                    for (int j = 0; j < cast<IRIntLit>(vectorType->getElementCount())->getValue();
                         j++)
                    {
                        if (isFirst)
                            isFirst = false;
                        else
                            m_writer->emit(", ");
                        auto outerPrec = getInfo(EmitOp::General);
                        auto prec = getInfo(EmitOp::Postfix);
                        emitOperand(arg, leftSide(outerPrec, prec));
                        m_writer->emit(".");
                        m_writer->emitChar(xyzwNames[j]);
                    }
                }
                else
                {
                    if (isFirst)
                        isFirst = false;
                    else
                        m_writer->emit(", ");
                    emitOperand(arg, getInfo(EmitOp::General));
                }
            }
            m_writer->emit(")");
            return true;
        }
    case kIROp_FloatCast:
    case kIROp_CastIntToFloat:
    case kIROp_IntCast:
    case kIROp_CastFloatToInt:
        {
            if (auto dstVectorType = as<IRVectorType>(inst->getDataType()))
            {
                m_writer->emit("make_");
                emitType(inst->getDataType());
                m_writer->emit("(");
                bool isFirst = true;
                char xyzwNames[] = "xyzw";
                for (UInt i = 0; i < inst->getOperandCount(); i++)
                {
                    auto arg = inst->getOperand(i);
                    if (auto vectorType = as<IRVectorType>(arg->getDataType()))
                    {
                        for (int j = 0;
                             j < cast<IRIntLit>(vectorType->getElementCount())->getValue();
                             j++)
                        {
                            if (isFirst)
                                isFirst = false;
                            else
                                m_writer->emit(", ");
                            m_writer->emit("(");
                            emitType(dstVectorType->getElementType());
                            m_writer->emit(")");
                            auto outerPrec = getInfo(EmitOp::General);
                            auto prec = getInfo(EmitOp::Postfix);
                            emitOperand(arg, leftSide(outerPrec, prec));
                            m_writer->emit(".");
                            m_writer->emitChar(xyzwNames[j]);
                        }
                    }
                    else
                    {
                        if (isFirst)
                            isFirst = false;
                        else
                            m_writer->emit(", ");
                        m_writer->emit("(");
                        emitType(dstVectorType->getElementType());
                        m_writer->emit(")");
                        emitOperand(arg, getInfo(EmitOp::General));
                    }
                }
                m_writer->emit(")");
                return true;
            }
            else if (const auto matrixType = as<IRMatrixType>(inst->getDataType()))
            {
                m_writer->emit("make");
                emitType(inst->getDataType());
                m_writer->emit("(");
                for (UInt i = 0; i < inst->getOperandCount(); i++)
                {
                    auto arg = inst->getOperand(i);
                    if (i > 0)
                        m_writer->emit(", ");
                    emitOperand(arg, getInfo(EmitOp::General));
                }
                m_writer->emit(")");
                return true;
            }
            return false;
        }
    case kIROp_MakeMatrix:
    case kIROp_MakeMatrixFromScalar:
    case kIROp_MatrixReshape:
        {
            m_writer->emit("make");
            emitType(inst->getDataType());
            m_writer->emit("(");
            for (UInt i = 0; i < inst->getOperandCount(); i++)
            {
                auto arg = inst->getOperand(i);
                if (i > 0)
                    m_writer->emit(", ");
                emitOperand(arg, getInfo(EmitOp::General));
            }
            m_writer->emit(")");
            return true;
        }
    case kIROp_MakeArray:
        {
            IRType* dataType = inst->getDataType();
            IRArrayType* arrayType = as<IRArrayType>(dataType);

            IRType* elementType = arrayType->getElementType();

            // Emit braces for the FixedArray struct.

            _emitInitializerList(elementType, inst->getOperands(), Index(inst->getOperandCount()));

            return true;
        }
    case kIROp_WaveMaskBallot:
        {
            m_extensionTracker->requireSMVersion(SemanticVersion(7, 0));

            m_writer->emit("__ballot_sync(");
            emitOperand(inst->getOperand(0), getInfo(EmitOp::General));
            m_writer->emit(", ");
            emitOperand(inst->getOperand(1), getInfo(EmitOp::General));
            m_writer->emit(")");
            return true;
        }
    case kIROp_WaveMaskMatch:
        {
            m_extensionTracker->requireSMVersion(SemanticVersion(7, 0));

            m_writer->emit("__match_any_sync(");
            emitOperand(inst->getOperand(0), getInfo(EmitOp::General));
            m_writer->emit(", ");
            emitOperand(inst->getOperand(1), getInfo(EmitOp::General));
            m_writer->emit(")");
            return true;
        }
    case kIROp_GetOptiXRayPayloadPtr:
        {
            m_writer->emit("(");
            emitType(inst->getDataType());
            m_writer->emit(")getOptiXRayPayloadPtr()");
            return true;
        }
    case kIROp_GetOptiXHitAttribute:
        {
            auto typeToFetch = inst->getOperand(0);
            auto idxInst = as<IRIntLit>(inst->getOperand(1));
            IRIntegerValue idx = idxInst->getValue();
            if (typeToFetch->getOp() == kIROp_FloatType)
            {
                m_writer->emit("__int_as_float(optixGetAttribute_");
            }
            else
            {
                m_writer->emit("optixGetAttribute_");
            }
            m_writer->emit(idx);
            if (typeToFetch->getOp() == kIROp_FloatType)
            {
                m_writer->emit("())");
            }
            else
            {
                m_writer->emit("()");
            }
            return true;
        }
    case kIROp_GetOptiXSbtDataPtr:
        {
            m_writer->emit("((");
            emitType(inst->getDataType());
            m_writer->emit(")optixGetSbtDataPointer())");
            return true;
        }
    case kIROp_DispatchKernel:
        {
            auto dispatchInst = as<IRDispatchKernel>(inst);
            emitOperand(dispatchInst->getBaseFn(), getInfo(EmitOp::Atomic));
            m_writer->emit("<<<");
            emitOperand(dispatchInst->getThreadGroupSize(), getInfo(EmitOp::General));
            m_writer->emit(", ");
            emitOperand(dispatchInst->getDispatchSize(), getInfo(EmitOp::General));
            m_writer->emit(">>>(");
            for (UInt i = 0; i < dispatchInst->getArgCount(); i++)
            {
                if (i > 0)
                    m_writer->emit(", ");
                emitOperand(dispatchInst->getArg(i), getInfo(EmitOp::General));
            }
            m_writer->emit(")");
            return true;
        }
    default:
        break;
    }

    return Super::tryEmitInstExprImpl(inst, inOuterPrec);
}

void CUDASourceEmitter::handleRequiredCapabilitiesImpl(IRInst* inst)
{
    // Does this function declare any requirements on CUDA capabilities
    // that should affect output?

    for (auto decoration : inst->getDecorations())
    {
        if (auto smDecoration = as<IRRequireCUDASMVersionDecoration>(decoration))
        {
            SemanticVersion version;
            version.setFromInteger(SemanticVersion::IntegerType(smDecoration->getCUDASMVersion()));
            m_extensionTracker->requireSMVersion(version);
        }
    }
}

void CUDASourceEmitter::emitVectorTypeNameImpl(IRType* elementType, IRIntegerValue elementCount)
{
    m_writer->emit(getVectorPrefix(elementType->getOp()));
    m_writer->emit(elementCount);
}

void CUDASourceEmitter::emitSimpleTypeImpl(IRType* type)
{
    switch (type->getOp())
    {
    case kIROp_VectorType:
        {
            auto vectorType = as<IRVectorType>(type);
            m_writer->emit(getVectorPrefix(vectorType->getElementType()->getOp()));
            m_writer->emit(as<IRIntLit>(vectorType->getElementCount())->getValue());
            break;
        }
    default:
        m_writer->emit(_getTypeName(type));
        break;
    }
}

void CUDASourceEmitter::emitRateQualifiersAndAddressSpaceImpl(
    IRRate* rate,
    [[maybe_unused]] AddressSpace addressSpace)
{
    if (as<IRGroupSharedRate>(rate))
    {
        m_writer->emit("__shared__ ");
    }
}

void CUDASourceEmitter::emitSimpleFuncParamsImpl(IRFunc* func)
{
    m_writer->emit("(");

    bool hasEmittedParam = false;
    auto firstParam = func->getFirstParam();
    for (auto pp = firstParam; pp; pp = pp->getNextParam())
    {
        auto varLayout = getVarLayout(pp);
        if (varLayout && varLayout->findSystemValueSemanticAttr())
        {
            // If it has a semantic don't output, it will be accessed via a global
            continue;
        }

        if (hasEmittedParam)
            m_writer->emit(", ");

        emitSimpleFuncParamImpl(pp);
        hasEmittedParam = true;
    }

    m_writer->emit(")");
}

void CUDASourceEmitter::emitSimpleFuncImpl(IRFunc* func)
{
    // Skip the CPP impl - as it does some processing we don't need here for entry points.
    CLikeSourceEmitter::emitSimpleFuncImpl(func);
}

void CUDASourceEmitter::emitSimpleValueImpl(IRInst* inst)
{
    // Make sure we convert float to half when emitting a half literal to avoid
    // overload ambiguity errors from CUDA.
    if (inst->getOp() == kIROp_FloatLit)
    {
        if (inst->getDataType()->getOp() == kIROp_HalfType)
        {
            m_writer->emit("__half(");
            CLikeSourceEmitter::emitSimpleValueImpl(inst);
            m_writer->emit(")");
            return;
        }
    }
    Super::emitSimpleValueImpl(inst);
}


void CUDASourceEmitter::emitSemanticsImpl(IRInst* inst, bool allowOffsetLayout)
{
    Super::emitSemanticsImpl(inst, allowOffsetLayout);
}

void CUDASourceEmitter::emitInterpolationModifiersImpl(
    IRInst* varInst,
    IRType* valueType,
    IRVarLayout* layout)
{
    Super::emitInterpolationModifiersImpl(varInst, valueType, layout);
}

void CUDASourceEmitter::emitVarDecorationsImpl(IRInst* varDecl)
{
    Super::emitVarDecorationsImpl(varDecl);
}

void CUDASourceEmitter::emitMatrixLayoutModifiersImpl(IRType* varType)
{
    Super::emitMatrixLayoutModifiersImpl(varType);
}

bool CUDASourceEmitter::tryEmitGlobalParamImpl(IRGlobalParam* varDecl, IRType* varType)
{
    // A global shader parameter in the IR for CUDA output will
    // either be the unique constant buffer that wraps all the
    // global-scope parameters in the original code (which is
    // handled as a special-case before this routine would be
    // called), or it is one of the system-defined varying inputs
    // like `threadIdx`. We won't need to emit anything in the
    // output code for the latter case, so we need to emit
    // nothing here and return `true` so that the base class
    // uses our logic instead of the default.
    //
    SLANG_UNUSED(varDecl);
    SLANG_UNUSED(varType);
    return true;
}


void CUDASourceEmitter::emitModuleImpl(IRModule* module, DiagnosticSink* sink)
{
    // Set up with all of the base types used in the module
    m_extensionTracker->requireBaseTypes(_findBaseTypesUsed(module));

    CLikeSourceEmitter::emitModuleImpl(module, sink);

    // Emit all witness table definitions.
    _emitWitnessTableDefinitions();
}


} // namespace Slang
