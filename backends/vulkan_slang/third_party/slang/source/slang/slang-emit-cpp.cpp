// slang-emit-cpp.cpp
#include "slang-emit-cpp.h"

#include "../compiler-core/slang-artifact-desc-util.h"
#include "../core/slang-token-reader.h"
#include "../core/slang-writer.h"
#include "slang-emit-source-writer.h"
#include "slang-ir-clone.h"
#include "slang-ir-util.h"
#include "slang-mangled-lexer.h"

#include <assert.h>

/*
ABI
---

In terms of ABI we need to discuss the variety of variables/resources that need to be defined by the
host for appropriate execution of the output code.

https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-variable-syntax

Broadly we could categorize these as..

1) Varying entry point parameters (or 'varying')
2) Uniform entry point parameters
3) Uniform globals
4) Thread shared (such as group shared) or ('thread shared')
5) Thread local ('static')

If we can invoke a bunch of threads as a single invocation we could effectively have the
ThreadShared not part of the ABI, but something that is say allocated on the stack before the
threads are kicked off. If we kick of threads individually then we would need to pass this in as
part of ABI. NOTE that it isn't right in so far as memory barriers etc couldn't work, as each thread
would run to completion, but we aren't going to worry about barriers for now.

On 1 - there could be potentially input and outputs (perhaps in out?). On CPU I guess that's fine.

On 2 and 3 they are effectively the same, and so for now 2+3 will be referred to together as
'uniforms'. They should be copied into a single structure that has a well known order.

On 1 these are parameters that vary on an invocation. Thus a caller might call many times with same
globals structure and different varying entry point parameters.

On 5 - This would be a global that can be set and then accessed within the context of single thread

So in order of rate of change

1 : Probably change on every invocation (in the future such an invocation might be behind the API)
2 + 3 : Changes per group of 'threads' executed together
4 : Does not change between invocations
5 : Could be placed on the stack, and so not necessarily part of the ABI

For now we are only going to implement something 'Compute shader'-like. Doing so makes the varying
parameter always the same.

So for now we would need to pass in

ComputeVaryingInput - Fixed because we are doing compute shader
Uniform             - All the uniform data in a big blob, both from uniform entry point parameters,
and uniform globals

When called we can have a structure that holds the thread local variables, and these two pointers.
*/

namespace Slang
{

static const char s_xyzwNames[] = "xyzw";

/* !!!!!!!!!!!!!!!!!!!!!!!! CPPEmitHandler !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

/* static */ UnownedStringSlice CPPSourceEmitter::getBuiltinTypeName(IROp op)
{
    switch (op)
    {
    case kIROp_VoidType:
        return UnownedStringSlice("void");
    case kIROp_BoolType:
        return UnownedStringSlice("bool");

    case kIROp_Int8Type:
        return UnownedStringSlice("int8_t");
    case kIROp_Int16Type:
        return UnownedStringSlice("int16_t");
    case kIROp_IntType:
        return UnownedStringSlice("int32_t");
    case kIROp_Int64Type:
        return UnownedStringSlice("int64_t");
    case kIROp_IntPtrType:
        return UnownedStringSlice("intptr_t");

    case kIROp_UInt8Type:
        return UnownedStringSlice("uint8_t");
    case kIROp_UInt16Type:
        return UnownedStringSlice("uint16_t");
    case kIROp_UIntType:
        return UnownedStringSlice("uint32_t");
    case kIROp_UInt64Type:
        return UnownedStringSlice("uint64_t");
    case kIROp_UIntPtrType:
        return UnownedStringSlice("uintptr_t");

        // Not clear just yet how we should handle half... we want all processing as float
        // probly, but when reading/writing to memory converting
    case kIROp_HalfType:
        return UnownedStringSlice("half");

    case kIROp_FloatType:
        return UnownedStringSlice("float");
    case kIROp_DoubleType:
        return UnownedStringSlice("double");
    case kIROp_CharType:
        return UnownedStringSlice("char");

    default:
        return UnownedStringSlice();
    }
}

UnownedStringSlice CPPSourceEmitter::_getTypeName(IRType* type)
{
    StringSlicePool::Handle handle = StringSlicePool::kNullHandle;
    if (m_typeNameMap.tryGetValue(type, handle))
    {
        return m_slicePool.getSlice(handle);
    }

    StringBuilder builder;
    if (SLANG_SUCCEEDED(calcTypeName(type, m_target, builder)))
    {
        handle = m_slicePool.add(builder);
    }

    m_typeNameMap.add(type, handle);

    SLANG_ASSERT(handle != StringSlicePool::kNullHandle);
    return m_slicePool.getSlice(handle);
}

SlangResult CPPSourceEmitter::_calcCPPTextureTypeName(
    IRTextureTypeBase* texType,
    StringBuilder& outName)
{
    switch (texType->getAccess())
    {
    case SLANG_RESOURCE_ACCESS_READ:
        break;
    case SLANG_RESOURCE_ACCESS_WRITE:
        outName << "RW";
        break;
    case SLANG_RESOURCE_ACCESS_READ_WRITE:
        outName << "RW";
        break;
    case SLANG_RESOURCE_ACCESS_RASTER_ORDERED:
        outName << "RasterizerOrdered";
        break;
    case SLANG_RESOURCE_ACCESS_APPEND:
        outName << "Append";
        break;
    case SLANG_RESOURCE_ACCESS_CONSUME:
        outName << "Consume";
        break;
    case SLANG_RESOURCE_ACCESS_FEEDBACK:
        outName << "Feedback";
        break;
    default:
        SLANG_DIAGNOSE_UNEXPECTED(getSink(), SourceLoc(), "unhandled resource access mode");
        return SLANG_FAIL;
    }

    switch (texType->GetBaseShape())
    {
    case SLANG_TEXTURE_1D:
        outName << "Texture1D";
        break;
    case SLANG_TEXTURE_2D:
        outName << "Texture2D";
        break;
    case SLANG_TEXTURE_3D:
        outName << "Texture3D";
        break;
    case SLANG_TEXTURE_CUBE:
        outName << "TextureCube";
        break;
    case SLANG_TEXTURE_BUFFER:
        outName << "Buffer";
        break;
    default:
        SLANG_DIAGNOSE_UNEXPECTED(getSink(), SourceLoc(), "unhandled resource shape");
        return SLANG_FAIL;
    }

    if (texType->isMultisample())
    {
        outName << "MS";
    }
    if (texType->isArray())
    {
        outName << "Array";
    }
    outName << "<" << _getTypeName(texType->getElementType()) << " >";

    return SLANG_OK;
}

static UnownedStringSlice _getResourceTypePrefix(IROp op)
{
    switch (op)
    {
    case kIROp_HLSLStructuredBufferType:
        return UnownedStringSlice::fromLiteral("StructuredBuffer");
    case kIROp_HLSLRWStructuredBufferType:
        return UnownedStringSlice::fromLiteral("RWStructuredBuffer");
    case kIROp_HLSLRWByteAddressBufferType:
        return UnownedStringSlice::fromLiteral("RWByteAddressBuffer");
    case kIROp_HLSLByteAddressBufferType:
        return UnownedStringSlice::fromLiteral("ByteAddressBuffer");
    case kIROp_SamplerStateType:
        return UnownedStringSlice::fromLiteral("SamplerState");
    case kIROp_SamplerComparisonStateType:
        return UnownedStringSlice::fromLiteral("SamplerComparisonState");
    case kIROp_HLSLRasterizerOrderedStructuredBufferType:
        return UnownedStringSlice::fromLiteral("RasterizerOrderedStructuredBuffer");
    case kIROp_HLSLAppendStructuredBufferType:
        return UnownedStringSlice::fromLiteral("AppendStructuredBuffer");
    case kIROp_HLSLConsumeStructuredBufferType:
        return UnownedStringSlice::fromLiteral("ConsumeStructuredBuffer");
    case kIROp_HLSLRasterizerOrderedByteAddressBufferType:
        return UnownedStringSlice::fromLiteral("RasterizerOrderedByteAddressBuffer");
    case kIROp_RaytracingAccelerationStructureType:
        return UnownedStringSlice::fromLiteral("RaytracingAccelerationStructure");

    default:
        return UnownedStringSlice();
    }
}

SlangResult CPPSourceEmitter::calcTypeName(IRType* type, CodeGenTarget target, StringBuilder& out)
{
    switch (type->getOp())
    {
    case kIROp_HalfType:
        {
            // Special case half
            out << getBuiltinTypeName(kIROp_FloatType);
            return SLANG_OK;
        }
    case kIROp_VectorType:
        {
            auto vecType = static_cast<IRVectorType*>(type);
            auto vecCount = int(getIntVal(vecType->getElementCount()));
            auto elemType = vecType->getElementType();

            out << "Vector<" << _getTypeName(elemType) << ", " << vecCount << ">";
            return SLANG_OK;
        }
    case kIROp_MatrixType:
        {
            auto matType = static_cast<IRMatrixType*>(type);

            auto elementType = matType->getElementType();
            const auto rowCount = int(getIntVal(matType->getRowCount()));
            const auto colCount = int(getIntVal(matType->getColumnCount()));

            out << "Matrix<" << _getTypeName(elementType) << ", " << rowCount << ", " << colCount
                << ">";

            return SLANG_OK;
        }
    case kIROp_WitnessTableType:
    case kIROp_WitnessTableIDType:
        {
            // A witness table typed value translates to a pointer to the
            // struct of function pointers corresponding to the interface type.
            auto witnessTableType = static_cast<IRWitnessTableType*>(type);
            auto baseType = cast<IRType>(witnessTableType->getOperand(0));
            SLANG_RETURN_ON_FAIL(calcTypeName(baseType, target, out));
            out << "*";
            return SLANG_OK;
        }
    case kIROp_RawPointerType:
    case kIROp_RTTIPointerType:
        {
            out << "void*";
            return SLANG_OK;
        }
    case kIROp_AnyValueType:
        {
            out << "AnyValue<";
            auto anyValueType = static_cast<IRAnyValueType*>(type);
            out << getIntVal(anyValueType->getSize());
            out << ">";
            return SLANG_OK;
        }
    case kIROp_ConstantBufferType:
    case kIROp_ParameterBlockType:
        {
            auto groupType = cast<IRParameterGroupType>(type);
            auto elementType = groupType->getElementType();

            SLANG_RETURN_ON_FAIL(calcTypeName(elementType, target, out));
            out << "*";
            return SLANG_OK;
        }
    case kIROp_NativePtrType:
    case kIROp_PtrType:
    case kIROp_ConstRefType:
        {
            auto elementType = (IRType*)type->getOperand(0);
            SLANG_RETURN_ON_FAIL(calcTypeName(elementType, target, out));
            out << "*";
            return SLANG_OK;
        }
    case kIROp_RTTIType:
        {
            out << "TypeInfo";
            return SLANG_OK;
        }
    case kIROp_RTTIHandleType:
        {
            out << "TypeInfo*";
            return SLANG_OK;
        }
    case kIROp_NativeStringType:
        {
            out << "const char*";
            return SLANG_OK;
        }
    case kIROp_StringType:
        {
            out << "String";
            return SLANG_OK;
        }
    case kIROp_ComPtrType:
        {
            auto comPtrType = static_cast<IRComPtrType*>(type);
            auto baseType = cast<IRType>(comPtrType->getOperand(0));

            out << "ComPtr<";
            SLANG_RETURN_ON_FAIL(calcTypeName(baseType, target, out));
            out << ">";
            return SLANG_OK;
        }
    case kIROp_ClassType:
        {
            out << "RefPtr<";
            out << getName(type);
            out << ">";
            return SLANG_OK;
        }
    case kIROp_TargetTupleType:
        {
            out << "std::tuple<";
            for (UInt i = 0; i < type->getOperandCount(); i++)
            {
                if (i > 0)
                    out << ", ";
                auto elementType = (IRType*)type->getOperand(i);
                SLANG_RETURN_ON_FAIL(calcTypeName(elementType, target, out));
            }
            out << ">";
            return SLANG_OK;
        }
    case kIROp_Specialize:
        {
            auto inner = getResolvedInstForDecorations(type);
            if (auto targetIntrinsic = findBestTargetIntrinsicDecoration(inner, getTargetCaps()))
            {
                out << targetIntrinsic->getDefinition();
                out << "<";
                for (UInt i = 1; i < type->getOperandCount(); i++)
                {
                    if (i > 1)
                        out << ", ";
                    auto elementType = (IRType*)type->getOperand(i);
                    SLANG_RETURN_ON_FAIL(calcTypeName(elementType, target, out));
                }
                out << ">";
                return SLANG_OK;
            }
            return SLANG_FAIL;
        }
    case kIROp_IntLit:
        {
            auto intLit = as<IRIntLit>(type);
            out << intLit->getValue();
            return SLANG_OK;
        }
    case kIROp_AtomicType:
        {
            return calcTypeName((IRType*)type->getOperand(0), target, out);
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
                return _calcCPPTextureTypeName(texType, out);
            }

            // If _getResourceTypePrefix returns something, we assume can output any specialization
            // after it in order.
            {
                UnownedStringSlice prefix = _getResourceTypePrefix(type->getOp());
                if (prefix.getLength() > 0)
                {
                    auto oldWriter = m_writer;
                    SourceManager* sourceManager = oldWriter->getSourceManager();

                    // TODO(JS): This is a bit of a hack. We don't want to emit the result here,
                    // so we replace the writer, write out the type, grab the contents, and restore
                    // the writer

                    SourceWriter writer(sourceManager, LineDirectiveMode::None, nullptr);
                    m_writer = &writer;

                    m_writer->emit(prefix);

                    // TODO(JS).
                    // Assumes ordering of types matches ordering of operands.

                    UInt operandCount = type->getOperandCount();
                    if (as<IRHLSLStructuredBufferTypeBase>(type))
                        operandCount = 1;

                    if (operandCount)
                    {
                        m_writer->emit("<");
                        for (UInt ii = 0; ii < operandCount; ++ii)
                        {
                            if (ii != 0)
                            {
                                m_writer->emit(", ");
                            }
                            emitVal(type->getOperand(ii), getInfo(EmitOp::General));
                        }
                        m_writer->emit(">");
                    }

                    out << writer.getContent();

                    m_writer = oldWriter;
                    return SLANG_OK;
                }
            }

            break;
        }
    }

    SLANG_DIAGNOSE_UNEXPECTED(getSink(), SourceLoc(), "unhandled type for C/C++ emit");
    return SLANG_FAIL;
}

void CPPSourceEmitter::useType(IRType* type)
{
    _getTypeName(type);
}

/* static */ CPPSourceEmitter::TypeDimension CPPSourceEmitter::_getTypeDimension(
    IRType* type,
    bool vecSwap)
{
    switch (type->getOp())
    {
    case kIROp_PtrType:
        {
            type = static_cast<IRPtrType*>(type)->getValueType();
            break;
        }
    case kIROp_RefType:
    case kIROp_ConstRefType:
        {
            type = static_cast<IRPtrTypeBase*>(type)->getValueType();
            break;
        }
    default:
        break;
    }

    switch (type->getOp())
    {
    case kIROp_VectorType:
        {
            auto vecType = static_cast<IRVectorType*>(type);

            IRBasicType* elemBasicType = as<IRBasicType>(vecType->getElementType());
            const BaseType baseType = elemBasicType->getBaseType();

            const int elemCount = int(getIntVal(vecType->getElementCount()));
            return (!vecSwap) ? TypeDimension{baseType, 1, elemCount}
                              : TypeDimension{baseType, elemCount, 1};
        }
    case kIROp_MatrixType:
        {
            auto matType = static_cast<IRMatrixType*>(type);
            const int colCount = int(getIntVal(matType->getColumnCount()));
            const int rowCount = int(getIntVal(matType->getRowCount()));

            IRBasicType* elemBasicType = as<IRBasicType>(matType->getElementType());
            const BaseType baseType = elemBasicType->getBaseType();

            return TypeDimension{baseType, rowCount, colCount};
        }
    default:
        {
            // Assume we don't know the type
            BaseType baseType = BaseType::Void;

            IRBasicType* basicType = as<IRBasicType>(type);
            if (basicType)
            {
                baseType = basicType->getBaseType();
            }

            return TypeDimension{baseType, 1, 1};
        }
    }
}

void CPPSourceEmitter::_emitAccess(
    const UnownedStringSlice& name,
    const TypeDimension& dimension,
    int row,
    int col,
    SourceWriter* writer)
{

    writer->emit(name);
    const int comb = (dimension.colCount > 1 ? 2 : 0) | (dimension.rowCount > 1 ? 1 : 0);
    switch (comb)
    {
    case 0:
        {
            break;
        }
    case 1:
        {
            // Vector, row count is biggest
            const UnownedStringSlice* elemNames =
                getVectorElementNames(dimension.elemType, dimension.rowCount);
            writer->emit(".");
            const int index = (row > col) ? row : col;
            writer->emit(elemNames[index]);
            break;
        }
    case 2:
        {
            // Vector cols biggest dimension
            const UnownedStringSlice* elemNames =
                getVectorElementNames(dimension.elemType, dimension.colCount);
            writer->emit(".");
            const int index = (row > col) ? row : col;
            writer->emit(elemNames[index]);
            break;
        }
    case 3:
        {
            // Matrix
            const UnownedStringSlice* elemNames =
                getVectorElementNames(dimension.elemType, dimension.colCount);

            writer->emit(".rows[");
            writer->emit(row);
            writer->emit("].");
            writer->emit(elemNames[col]);
            break;
        }
    }
}

/* !!!!!!!!!!!!!!!!!!!!!! CPPSourceEmitter !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

CPPSourceEmitter::CPPSourceEmitter(const Desc& desc)
    : Super(desc), m_slicePool(StringSlicePool::Style::Default)
{
    m_semanticUsedFlags = 0;
    // m_semanticUsedFlags = SemanticUsedFlag::GroupID | SemanticUsedFlag::GroupThreadID |
    // SemanticUsedFlag::DispatchThreadID;


    const auto artifactDesc = ArtifactDescUtil::makeDescForCompileTarget(asExternal(getTarget()));

    // If we have runtime library we can convert to a terminated string slice
    m_hasString = (artifactDesc.style == ArtifactStyle::Host);
}

void CPPSourceEmitter::emitParamTypeImpl(IRType* type, String const& name)
{
    emitType(type, name);
}

void CPPSourceEmitter::emitGlobalRTTISymbolPrefix()
{
    m_writer->emit("SLANG_PRELUDE_SHARED_LIB_EXPORT");
}

void CPPSourceEmitter::emitWitnessTable(IRWitnessTable* witnessTable)
{
    auto interfaceType = as<IRInterfaceType>(witnessTable->getConformanceType());

    if (!interfaceType)
        return;

    // Ignore witness tables for builtin interface types.
    if (isBuiltin(interfaceType))
        return;

    if (interfaceType->findDecoration<IRComInterfaceDecoration>())
    {
        pendingWitnessTableDefinitions.add(witnessTable);
        return;
    }

    // Declare a global variable for the witness table.
    m_writer->emit("extern \"C\" { ");
    emitGlobalRTTISymbolPrefix();
    m_writer->emit(" extern ");
    emitSimpleType(interfaceType);
    m_writer->emit(" ");
    m_writer->emit(getName(witnessTable));
    m_writer->emit("; }\n");

    // The actual definition of this witness table global variable
    // is deferred until the entire `Context` class is emitted, so
    // that the member functions are available for reference.
    // The witness table definition emission logic is defined in the
    // `_emitWitnessTableDefinitions` function.
    pendingWitnessTableDefinitions.add(witnessTable);
}

void CPPSourceEmitter::_emitWitnessTableDefinitions()
{
    for (auto witnessTable : pendingWitnessTableDefinitions)
    {
        auto interfaceType = cast<IRInterfaceType>(witnessTable->getConformanceType());
        if (interfaceType->findDecoration<IRComInterfaceDecoration>())
        {
            emitComWitnessTable(witnessTable);
            continue;
        }
        List<IRWitnessTableEntry*> sortedWitnessTableEntries =
            getSortedWitnessTableEntries(witnessTable);
        m_writer->emit("extern \"C\"\n{\n");
        m_writer->indent();
        emitGlobalRTTISymbolPrefix();
        m_writer->emit("\n");
        emitSimpleType(interfaceType);
        m_writer->emit(" ");
        m_writer->emit(getName(witnessTable));
        m_writer->emit(" = {\n");
        m_writer->indent();
        auto seqIdDecoration = witnessTable->findDecoration<IRSequentialIDDecoration>();
        if (seqIdDecoration)
            m_writer->emit((UInt)seqIdDecoration->getSequentialID());
        else
            m_writer->emit("0");
        for (Index i = 0; i < sortedWitnessTableEntries.getCount(); i++)
        {
            auto entry = sortedWitnessTableEntries[i];
            if (auto funcVal = as<IRFunc>(entry->satisfyingVal.get()))
            {
                m_writer->emit(",\n");
                m_writer->emit(getName(funcVal));
            }
            else if (auto witnessTableVal = as<IRWitnessTable>(entry->getSatisfyingVal()))
            {
                m_writer->emit(",\n");
                m_writer->emit("&");
                m_writer->emit(getName(witnessTableVal));
            }
            else if (
                entry->getSatisfyingVal() &&
                entry->getSatisfyingVal()->getDataType()->getOp() == kIROp_RTTIHandleType)
            {
                m_writer->emit(",\n");
                emitInstExpr(entry->getSatisfyingVal(), getInfo(EmitOp::General));
            }
            else
            {
                SLANG_UNEXPECTED("unknown witnesstable entry type");
            }
        }
        m_writer->dedent();
        m_writer->emit("\n};\n");
        m_writer->dedent();
        m_writer->emit("\n}\n");
    }
}

void CPPSourceEmitter::emitComInterface(IRInterfaceType* interfaceType)
{
    auto comDecoration = interfaceType->findDecoration<IRComInterfaceDecoration>();
    auto guidInst = as<IRStringLit>(comDecoration->getOperand(0));
    SLANG_RELEASE_ASSERT(guidInst);
    auto guid = guidInst->getStringSlice();
    SLANG_RELEASE_ASSERT(guid.getLength() == 32);

    m_writer->emit("struct ");
    emitSimpleType(interfaceType);
    m_writer->emit(" : ");
    // Emit base types.
    bool isFirst = true;
    for (UInt i = 0; i < interfaceType->getOperandCount(); i++)
    {
        auto entry = as<IRInterfaceRequirementEntry>(interfaceType->getOperand(i));
        if (auto witnessTableType = as<IRWitnessTableTypeBase>(entry->getRequirementVal()))
        {
            if (isFirst)
            {
                isFirst = false;
            }
            else
            {
                m_writer->emit(", ");
            }
            emitType((IRType*)witnessTableType->getConformanceType());
        }
    }
    if (isFirst)
    {
        m_writer->emit("ISlangUnknown");
    }

    // Emit methods.
    m_writer->emit("\n{\n");
    m_writer->indent();
    // Emit GUID.
    m_writer->emit("SLANG_COM_INTERFACE(0x");
    m_writer->emit(guid.subString(0, 8));
    m_writer->emit(", 0x");
    m_writer->emit(guid.subString(8, 4));
    m_writer->emit(", 0x");
    m_writer->emit(guid.subString(12, 4));
    m_writer->emit(", { ");
    for (UInt i = 0; i < 8; i++)
    {
        if (i > 0)
            m_writer->emit(", ");
        m_writer->emit("0x");
        m_writer->emit(guid.subString(16 + i * 2, 2));
    }
    m_writer->emit(" })\n");

    for (UInt i = 0; i < interfaceType->getOperandCount(); i++)
    {
        auto entry = as<IRInterfaceRequirementEntry>(interfaceType->getOperand(i));
        if (auto funcVal = as<IRFuncType>(entry->getRequirementVal()))
        {
            m_writer->emit("virtual SLANG_NO_THROW ");
            emitType(funcVal->getResultType());
            m_writer->emit(" SLANG_MCALL ");
            m_writer->emit(getName(entry->getRequirementKey()));
            m_writer->emit("(");
            bool isFirstParam = true;
            for (UInt p = 1; p < funcVal->getParamCount(); p++)
            {
                auto paramType = funcVal->getParamType(p);
                if (!isFirstParam)
                    m_writer->emit(", ");
                else
                    isFirstParam = false;

                emitParamType(paramType, String("param") + String(p));
            }
            m_writer->emit(") = 0;\n");
        }
    }
    m_writer->dedent();
    m_writer->emit("};\n");
}

void CPPSourceEmitter::emitInterface(IRInterfaceType* interfaceType)
{
    // Skip built-in interfaces.
    if (isBuiltin(interfaceType))
        return;

    if (interfaceType->findDecoration<IRComInterfaceDecoration>())
    {
        emitComInterface(interfaceType);
        return;
    }

    m_writer->emit("struct ");
    emitSimpleType(interfaceType);
    m_writer->emit("\n{\n");
    m_writer->indent();
    m_writer->emit("uint32_t sequentialID;\n");
    for (UInt i = 0; i < interfaceType->getOperandCount(); i++)
    {
        auto entry = as<IRInterfaceRequirementEntry>(interfaceType->getOperand(i));
        if (auto funcVal = as<IRFuncType>(entry->getRequirementVal()))
        {
            emitType(funcVal->getResultType());
            m_writer->emit(" (*");
            m_writer->emit(getName(entry->getRequirementKey()));
            m_writer->emit(")");
            m_writer->emit("(");
            bool isFirstParam = true;
            for (UInt p = 0; p < funcVal->getParamCount(); p++)
            {
                auto paramType = funcVal->getParamType(p);
                // Ingore TypeType-typed parameters for now.
                if (as<IRTypeType>(paramType))
                    continue;

                if (!isFirstParam)
                    m_writer->emit(", ");
                else
                    isFirstParam = false;

                emitParamType(paramType, String("param") + String(p));
            }
            m_writer->emit(");\n");
        }
        else if (auto witnessTableType = as<IRWitnessTableType>(entry->getRequirementVal()))
        {
            emitType((IRType*)witnessTableType->getConformanceType());
            m_writer->emit("* ");
            m_writer->emit(getName(entry->getRequirementKey()));
            m_writer->emit(";\n");
        }
        else if (entry->getRequirementVal()->getOp() == kIROp_RTTIHandleType)
        {
            m_writer->emit("TypeInfo* ");
            m_writer->emit(getName(entry->getRequirementKey()));
            m_writer->emit(";\n");
        }
    }
    m_writer->dedent();
    m_writer->emit("};\n");
}

void CPPSourceEmitter::emitRTTIObject(IRRTTIObject* rttiObject)
{
    m_writer->emit("extern \"C\" { ");
    emitGlobalRTTISymbolPrefix();
    m_writer->emit(" TypeInfo ");
    m_writer->emit(getName(rttiObject));
    m_writer->emit(" = {");
    auto typeSizeDecoration = rttiObject->findDecoration<IRRTTITypeSizeDecoration>();
    SLANG_ASSERT(typeSizeDecoration);
    m_writer->emit(typeSizeDecoration->getTypeSize());
    m_writer->emit("}; }\n");
}

bool CPPSourceEmitter::tryEmitGlobalParamImpl(IRGlobalParam* varDecl, IRType* varType)
{
    SLANG_UNUSED(varDecl);
    SLANG_UNUSED(varType);

    switch (varType->getOp())
    {
    case kIROp_StructType:
        {
            String name = getName(varDecl);

            UnownedStringSlice typeName = _getTypeName(varType);
            m_writer->emit(typeName);
            m_writer->emit("* ");
            m_writer->emit(name);
            m_writer->emit(";\n");
            return true;
        }
    }

    return false;
}

void CPPSourceEmitter::emitParameterGroupImpl(
    IRGlobalParam* varDecl,
    IRUniformParameterGroupType* type)
{
    // Output global parameters
    auto varLayout = getVarLayout(varDecl);
    SLANG_RELEASE_ASSERT(varLayout);

    String name = getName(varDecl);
    auto elementType = type->getElementType();

    switch (type->getOp())
    {
    case kIROp_ParameterBlockType:
    case kIROp_ConstantBufferType:
        {
            UnownedStringSlice typeName = _getTypeName(elementType);
            m_writer->emit(typeName);
            m_writer->emit("* ");
            m_writer->emit(name);
            m_writer->emit(";\n");
            break;
        }
    default:
        {
            emitType(elementType, name);
            m_writer->emit(";\n");
            break;
        }
    }
}

void CPPSourceEmitter::emitEntryPointAttributesImpl(
    IRFunc* irFunc,
    IREntryPointDecoration* entryPointDecor)
{
    SLANG_UNUSED(entryPointDecor);

    auto profile = m_effectiveProfile;
    auto stage = profile.getStage();

    switch (stage)
    {
    case Stage::Compute:
        {
            Int numThreads[kThreadGroupAxisCount];
            getComputeThreadGroupSize(irFunc, numThreads);

            // TODO(JS): We might want to store this information such that it can be used to execute
            m_writer->emit("// [numthreads(");
            for (int ii = 0; ii < kThreadGroupAxisCount; ++ii)
            {
                if (ii != 0)
                    m_writer->emit(", ");
                m_writer->emit(numThreads[ii]);
            }
            m_writer->emit(")]\n");
            break;
        }
    default:
        break;
    }

    m_writer->emit("SLANG_PRELUDE_EXPORT\n");
}

bool isPublicOrExportedFunc(IRFunc* func)
{
    for (auto decor : func->getDecorations())
    {
        switch (decor->getOp())
        {
        case kIROp_PublicDecoration:
        case kIROp_EntryPointDecoration:
        case kIROp_HLSLExportDecoration:
        case kIROp_DllExportDecoration:
        case kIROp_DllImportDecoration:
        case kIROp_CudaDeviceExportDecoration:
        case kIROp_CudaHostDecoration:
        case kIROp_CudaKernelDecoration:
            {
                return true;
            }
        default:
            break;
        }
    }
    return false;
}

void CPPSourceEmitter::emitSimpleFuncImpl(IRFunc* func)
{
    // Emit function decorations
    emitFuncDecorations(func);

    auto resultType = func->getResultType();

    auto name = getName(func);

    // Deal with decorations that need
    // to be emitted as attributes

    // If `func` is not public or exported, emit `static` to prevent linking clash.
    if (!isPublicOrExportedFunc(func))
    {
        m_writer->emit("static ");
    }
    // We start by emitting the result type and function name.
    //
    if (IREntryPointDecoration* const entryPointDecor =
            func->findDecoration<IREntryPointDecoration>())
    {
        // Note: we currently emit multiple functions to represent an entry point
        // on CPU/CUDA, and these all bottleneck through the actual `IRFunc`
        // here as a workhorse.
        //
        // Because the workhorse function doesn't have the right signature to service
        // general-purpose calls, it is being emitted with a `_` prefix.
        //
        StringBuilder prefixName;
        prefixName << "_" << name;
        emitType(resultType, prefixName);
    }
    else
    {
        emitType(resultType, name);
    }

    // Next we emit the parameter list of the function.
    //
    m_writer->emit("(");
    auto firstParam = func->getFirstParam();
    for (auto pp = firstParam; pp; pp = pp->getNextParam())
    {
        // Ingore TypeType-typed parameters for now.
        // In the future we will pass around runtime type info
        // for TypeType parameters.
        if (as<IRTypeType>(pp->getFullType()))
            continue;

        if (pp != firstParam)
            m_writer->emit(", ");

        emitSimpleFuncParamImpl(pp);
    }
    m_writer->emit(")");

    emitSemantics(func);

    // TODO: encode declaration vs. definition
    if (isDefinition(func))
    {
        m_writer->emit("\n{\n");
        m_writer->indent();

        // Need to emit the operations in the blocks of the function
        emitFunctionBody(func);

        m_writer->dedent();
        m_writer->emit("}\n\n");
    }
    else
    {
        m_writer->emit(";\n\n");
    }
}

void CPPSourceEmitter::emitSimpleValueImpl(IRInst* inst)
{
    if (inst->getOp() == kIROp_FloatLit)
    {
        IRConstant* constantInst = static_cast<IRConstant*>(inst);
        switch (constantInst->getFloatKind())
        {
        case IRConstant::FloatKind::Nan:
            {
                // TODO(JS):
                // It's not clear this will work on all targets.
                // In particular Visual Studio reports an error with this expression.
                m_writer->emit("(0.0 / 0.0)");
                break;
            }
        case IRConstant::FloatKind::PositiveInfinity:
            {
                m_writer->emit("SLANG_INFINITY");
                break;
            }
        case IRConstant::FloatKind::NegativeInfinity:
            {
                m_writer->emit("(-SLANG_INFINITY)");
                break;
            }
        default:
            {
                m_writer->emit(constantInst->value.floatVal);

                // If the literal is a float, then we need to add 'f' at end, as
                // without literal suffix the value defaults to double.
                IRType* type = constantInst->getDataType();
                if (type && type->getOp() == kIROp_FloatType)
                {
                    m_writer->emitChar('f');
                }
                break;
            }
        }
    }
    else
    {
        Super::emitSimpleValueImpl(inst);
    }
}

void CPPSourceEmitter::emitSimpleFuncParamImpl(IRParam* param)
{
    CLikeSourceEmitter::emitSimpleFuncParamImpl(param);
}

void CPPSourceEmitter::emitVectorTypeNameImpl(IRType* elementType, IRIntegerValue elementCount)
{
    m_writer->emit("Vector<");
    m_writer->emit(_getTypeName(elementType));
    m_writer->emit(", ");
    m_writer->emit(elementCount);
    m_writer->emit(">");
}

void CPPSourceEmitter::emitSimpleTypeImpl(IRType* inType)
{
    UnownedStringSlice slice = _getTypeName(inType);
    m_writer->emit(slice);
}

void CPPSourceEmitter::_emitType(IRType* type, DeclaratorInfo* declarator)
{
    switch (type->getOp())
    {
    default:
        CLikeSourceEmitter::_emitType(type, declarator);
        break;
    case kIROp_VectorType:
    case kIROp_MatrixType:
        {
            StringBuilder sb;
            calcTypeName(type, m_target, sb);
            m_writer->emit(sb.produceString());
            m_writer->emit(" ");
            emitDeclarator(declarator);
            break;
        }
    case kIROp_PtrType:
    case kIROp_InOutType:
    case kIROp_OutType:
        {
            auto ptrType = cast<IRPtrTypeBase>(type);
            PtrDeclaratorInfo ptrDeclarator(declarator);
            _emitType(ptrType->getValueType(), &ptrDeclarator);
        }
        break;
    case kIROp_RefType:
    case kIROp_ConstRefType:
        {
            auto ptrType = cast<IRPtrTypeBase>(type);
            PtrDeclaratorInfo refDeclarator(declarator);
            _emitType(ptrType->getValueType(), &refDeclarator);
        }
        break;
    case kIROp_ArrayType:
        {
            auto arrayType = static_cast<IRArrayType*>(type);
            auto elementType = arrayType->getElementType();
            int elementCount = int(getIntVal(arrayType->getElementCount()));

            m_writer->emit("FixedArray<");
            _emitType(elementType, nullptr);
            m_writer->emit(", ");
            m_writer->emit(elementCount);
            m_writer->emit("> ");
            emitDeclarator(declarator);
        }
        break;
    case kIROp_UnsizedArrayType:
        {
            auto arrayType = static_cast<IRUnsizedArrayType*>(type);
            auto elementType = arrayType->getElementType();

            m_writer->emit("Array<");
            _emitType(elementType, nullptr);
            m_writer->emit("> ");
            emitDeclarator(declarator);
        }
        break;
    case kIROp_FuncType:
        {
            auto funcType = cast<IRFuncType>(type);
            m_writer->emit("Slang_FuncType<");
            _emitType(funcType->getResultType(), nullptr);
            for (UInt i = 0; i < funcType->getParamCount(); i++)
            {
                m_writer->emit(", ");
                _emitType(funcType->getParamType(i), nullptr);
            }
            m_writer->emit("> ");
            emitDeclarator(declarator);
        }
        break;
    }
}

void CPPSourceEmitter::emitIntrinsicCallExprImpl(
    IRCall* inst,
    UnownedStringSlice intrinsicDefinition,
    IRInst* intrinsicInst,
    EmitOpInfo const& inOuterPrec)
{
    // TODO: Much of this logic duplicates code that is already
    // in `CLikeSourceEmitter::emitIntrinsicCallExpr`. The only
    // real difference is that when things bottom out on an ordinary
    // function call there is logic to look up a C/C++-backend-specific
    // opcode based on the function name, and emit using that.

    auto outerPrec = inOuterPrec;
    bool needClose = false;

    Index argCount = Index(inst->getArgCount());
    auto args = inst->getArgs();

    auto name = intrinsicDefinition;

    // We will special-case some names here, that
    // represent callable declarations that aren't
    // ordinary functions, and thus may use different
    // syntax.
    if (name == ".operator[]")
    {
        SLANG_ASSERT(argCount == 2 || argCount == 3);
        {
            // The user is invoking a built-in subscript operator

            // Determine if we are calling the `ref` accessor:
            // `ref` accessor returns a pointer of element type.
            auto ptrType = as<IRPtrType>(inst->getFullType());
            auto resourceType = inst->getOperand(1)->getFullType();
            auto elementType = resourceType ? resourceType->getOperand(0) : nullptr;
            bool isRef = ptrType && ptrType->getValueType() == elementType;

            auto emitSubscript = [this, &args](EmitOpInfo _outerPrec)
            {
                auto prec = getInfo(EmitOp::Postfix);
                bool needCloseSubscript = maybeEmitParens(_outerPrec, prec);
                emitOperand(args[0].get(), leftSide(_outerPrec, prec));
                m_writer->emit("[");
                emitOperand(args[1].get(), getInfo(EmitOp::General));
                m_writer->emit("]");
                maybeCloseParens(needCloseSubscript);
            };

            if (isRef)
            {
                auto prefixPrec = getInfo(EmitOp::Prefix);
                needClose = maybeEmitParens(outerPrec, prefixPrec);
                m_writer->emit("&");
                outerPrec = rightSide(outerPrec, prefixPrec);
            }
            emitSubscript(outerPrec);
            maybeCloseParens(needClose);
            if (argCount == 3)
            {
                m_writer->emit(" = ");
                emitOperand(args[2].get(), getInfo(EmitOp::General));
            }
        }

        return;
    }

    // Use default impl (which will do intrinsic special macro expansion as necessary)
    return Super::emitIntrinsicCallExprImpl(inst, intrinsicDefinition, intrinsicInst, inOuterPrec);
}

void CPPSourceEmitter::emitLoopControlDecorationImpl(IRLoopControlDecoration* decl)
{
    if (decl->getMode() == kIRLoopControl_Unroll)
    {
        // This relies on a suitable definition in slang-cpp-prelude.h or defined in C++ compiler
        // invocation.
        m_writer->emit("SLANG_UNROLL\n");
    }
}

const UnownedStringSlice* CPPSourceEmitter::getVectorElementNames(
    BaseType baseType,
    Index elemCount)
{
    SLANG_UNUSED(baseType);
    SLANG_UNUSED(elemCount);

    static const UnownedStringSlice elemNames[] = {
        UnownedStringSlice::fromLiteral("x"),
        UnownedStringSlice::fromLiteral("y"),
        UnownedStringSlice::fromLiteral("z"),
        UnownedStringSlice::fromLiteral("w"),
    };

    return elemNames;
}

const UnownedStringSlice* CPPSourceEmitter::getVectorElementNames(IRVectorType* vectorType)
{
    Index elemCount = Index(getIntVal(vectorType->getElementCount()));

    IRType* type = vectorType->getElementType()->getCanonicalType();
    IRBasicType* basicType = as<IRBasicType>(type);
    SLANG_ASSERT(basicType);
    return getVectorElementNames(basicType->getBaseType(), elemCount);
}

bool CPPSourceEmitter::tryEmitInstStmtImpl(IRInst* inst)
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
            m_writer->emit("uint2(");
            m_writer->emit(count);
            m_writer->emit(", ");
            m_writer->emit(stride);
            m_writer->emit(");\n");
            return true;
        }
    default:
        return false;
    }
}

bool CPPSourceEmitter::tryEmitInstExprImpl(IRInst* inst, const EmitOpInfo& inOuterPrec)
{
    switch (inst->getOp())
    {
    default:
        {
            return false;
        }

    case kIROp_InOutImplicitCast:
    case kIROp_OutImplicitCast:
        {
            // We'll just the LValue to be the desired type
            m_writer->emit("reinterpret_cast<");
            emitType(inst->getDataType());
            m_writer->emit(">(");

            emitOperand(inst->getOperand(0), getInfo(EmitOp::General));

            m_writer->emit(")");
            return true;
        }
    case kIROp_MakeVector:
        {
            IRType* retType = inst->getFullType();
            emitType(retType);
            m_writer->emit("(");
            bool isFirst = true;
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
                        m_writer->emitChar(s_xyzwNames[j]);
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
    case kIROp_MakeTargetTuple:
        {
            m_writer->emit("std::make_tuple(");
            for (UInt i = 0; i < inst->getOperandCount(); i++)
            {
                if (i > 0)
                    m_writer->emit(", ");
                auto arg = inst->getOperand(i);
                emitOperand(arg, getInfo(EmitOp::General));
            }
            m_writer->emit(")");
            return true;
        }
    case kIROp_GetTargetTupleElement:
        {
            auto outerPrec = getInfo(EmitOp::General);
            auto prec = getInfo(EmitOp::Postfix);
            m_writer->emit("std::get<");
            emitOperand(inst->getOperand(1), getInfo(EmitOp::General));
            m_writer->emit(">(");
            emitOperand(inst->getOperand(0), leftSide(outerPrec, prec));
            m_writer->emit(")");
            return true;
        }
    case kIROp_CastFloatToInt:
    case kIROp_CastIntToFloat:
    case kIROp_FloatCast:
    case kIROp_IntCast:
        {
            if (auto vectorType = as<IRVectorType>(inst->getDataType()))
            {
                emitType(vectorType);
                m_writer->emit("{");
                for (Index i = 0; i < cast<IRIntLit>(vectorType->getElementCount())->getValue();
                     i++)
                {
                    if (i > 0)
                        m_writer->emit(", ");
                    m_writer->emit("(");
                    emitType(vectorType->getElementType());
                    m_writer->emit(")_slang_vector_get_element(");
                    emitOperand(inst->getOperand(0), getInfo(EmitOp::General));
                    m_writer->emit(", ");
                    m_writer->emit(i);
                    m_writer->emit(")");
                }
                m_writer->emit("}");
                return true;
            }
            return false;
        }
    case kIROp_VectorReshape:
        {
            if (auto vectorType = as<IRVectorType>(inst->getDataType()))
            {
                m_writer->emit("_slang_vector_reshape<");
                emitType(vectorType->getElementType());
                m_writer->emit(", ");
                emitOperand(vectorType->getElementCount(), getInfo(EmitOp::General));
                m_writer->emit(">(");
                emitOperand(inst->getOperand(0), getInfo(EmitOp::General));
                m_writer->emit(")");
                return true;
            }
            return false;
        }
    case kIROp_GetElement:
        {
            auto getElementInst = static_cast<IRGetElement*>(inst);

            IRInst* baseInst = getElementInst->getBase();
            IRType* baseType = baseInst->getDataType();
            if (auto vectorBaseType = as<IRVectorType>(baseType))
            {
                if (auto intLitIndex = as<IRIntLit>(getElementInst->getIndex()))
                {
                    // For static index, we can emit simpler code using the `.x`, `.y` members.
                    auto outerPrec = getInfo(EmitOp::General);
                    auto prec = getInfo(EmitOp::Postfix);
                    emitOperand(baseInst, leftSide(outerPrec, prec));
                    m_writer->emit(".");
                    m_writer->emit(getVectorElementNames(vectorBaseType)[intLitIndex->getValue()]);
                }
                else
                {
                    // For dynamic index, we emit using `_slang_vector_get_element` intrinsics.
                    m_writer->emit("_slang_vector_get_element(");
                    emitOperand(baseInst, getInfo(EmitOp::General));
                    m_writer->emit(", ");
                    emitOperand(getElementInst->getIndex(), getInfo(EmitOp::General));
                    m_writer->emit(")");
                }
                return true;
            }
            else if (as<IRMatrixType>(baseType))
            {
                auto outerPrec = getInfo(EmitOp::General);
                auto prec = getInfo(EmitOp::Postfix);
                emitOperand(baseInst, leftSide(outerPrec, prec));
                m_writer->emit(".rows[");
                emitOperand(getElementInst->getIndex(), getInfo(EmitOp::General));
                m_writer->emit("]");
                return true;
            }
            return false;
        }
    case kIROp_GetElementPtr:
        {
            auto getElementInst = static_cast<IRGetElement*>(inst);

            IRInst* baseInst = getElementInst->getBase();
            IRType* baseType = as<IRPtrTypeBase>(baseInst->getDataType())->getValueType();
            if (auto vectorBaseType = as<IRVectorType>(baseType))
            {
                if (auto intLitIndex = as<IRIntLit>(getElementInst->getIndex()))
                {
                    // For static index, we can emit simpler code using the `.x`, `.y` members.
                    m_writer->emit("&(");
                    auto outerPrec = getInfo(EmitOp::General);
                    auto prec = getInfo(EmitOp::Postfix);
                    emitOperand(baseInst, leftSide(outerPrec, prec));
                    m_writer->emit("->");
                    m_writer->emit(getVectorElementNames(vectorBaseType)[intLitIndex->getValue()]);
                    m_writer->emit(")");
                }
                else
                {
                    m_writer->emit("_slang_vector_get_element_ptr(");
                    emitOperand(baseInst, getInfo(EmitOp::General));
                    m_writer->emit(", ");
                    emitOperand(getElementInst->getIndex(), getInfo(EmitOp::General));
                    m_writer->emit(")");
                }
                return true;
            }
            else if (as<IRMatrixType>(baseType))
            {
                m_writer->emit("(");
                auto outerPrec = getInfo(EmitOp::General);
                auto prec = getInfo(EmitOp::Postfix);
                emitOperand(baseInst, leftSide(outerPrec, prec));
                m_writer->emit("->rows + (");
                emitOperand(getElementInst->getIndex(), getInfo(EmitOp::General));
                m_writer->emit("))");
                return true;
            }
            return false;
        }
    case kIROp_RWStructuredBufferGetElementPtr:
        {
            m_writer->emit("(&(");
            auto base = inst->getOperand(0);
            auto outerPrec = getInfo(EmitOp::General);
            emitOperand(base, outerPrec);
            m_writer->emit("[");
            emitOperand(inst->getOperand(1), EmitOpInfo());
            m_writer->emit("]))");
            return true;
        }
    case kIROp_swizzle:
        {
            // For C++ we don't need to emit a swizzle function
            // For C we need a construction function
            auto swizzleInst = static_cast<IRSwizzle*>(inst);

            IRInst* baseInst = swizzleInst->getBase();
            IRType* baseType = baseInst->getDataType();

            // If we are swizzling from a built in type,
            if (as<IRBasicType>(baseType))
            {
                // We can swizzle a scalar type to be a vector, or just a scalar
                IRType* dstType = swizzleInst->getDataType();
                if (as<IRBasicType>(dstType))
                {
                    // If the output is a scalar, then could only have been a .x, which we can just
                    // ignore the '.x' part
                    emitOperand(baseInst, inOuterPrec);
                    return true;
                }
            }
            else
            {
                const Index elementCount = Index(swizzleInst->getElementCount());
                if (elementCount == 1)
                {
                    // If just one thing is extracted then the . syntax will just work
                    defaultEmitInstExpr(inst, inOuterPrec);
                    return true;
                }
            }

            {
                // Currently only works for C++ (we use {} constuction) - which means we don't need
                // to generate a function. For C we need to generate suitable construction function

                const Index elementCount = Index(swizzleInst->getElementCount());

                IRType* srcType = swizzleInst->getBase()->getDataType();
                IRVectorType* srcVecType = as<IRVectorType>(srcType);

                const UnownedStringSlice* elemNames = nullptr;
                if (srcVecType)
                    elemNames = getVectorElementNames(srcVecType);

                IRType* retType = swizzleInst->getFullType();
                emitType(retType);
                m_writer->emit("{");

                for (Index i = 0; i < elementCount; ++i)
                {
                    if (i > 0)
                    {
                        m_writer->emit(", ");
                    }

                    auto outerPrec = getInfo(EmitOp::General);

                    auto prec = getInfo(EmitOp::Postfix);
                    emitOperand(swizzleInst->getBase(), leftSide(outerPrec, prec));

                    if (srcVecType)
                    {
                        m_writer->emit(".");

                        IRInst* irElementIndex = swizzleInst->getElementIndex(i);
                        SLANG_RELEASE_ASSERT(irElementIndex->getOp() == kIROp_IntLit);
                        IRConstant* irConst = (IRConstant*)irElementIndex;
                        UInt elementIndex = (UInt)irConst->value.intVal;
                        SLANG_RELEASE_ASSERT(elementIndex < 4);

                        m_writer->emit(elemNames[elementIndex]);
                    }
                }

                m_writer->emit("}");
            }
            return true;
        }
    case kIROp_FRem:
        {
            if (auto basicType = as<IRBasicType>(inst->getDataType()))
            {
                switch (basicType->getOp())
                {
                case kIROp_HalfType:
                    m_writer->emit("F16_fmod(");
                    break;
                case kIROp_FloatType:
                    m_writer->emit("F32_fmod(");
                    break;
                case kIROp_DoubleType:
                    m_writer->emit("F64_fmod(");
                    break;
                default:
                    return false;
                }
                emitOperand(inst->getOperand(0), getInfo(EmitOp::General));
                m_writer->emit(", ");
                emitOperand(inst->getOperand(1), getInfo(EmitOp::General));
                m_writer->emit(")");
                return true;
            }
            return false;
        }
    case kIROp_Call:
        {
            auto funcValue = inst->getOperand(0);

            // Does this function declare any requirements.
            handleRequiredCapabilities(funcValue);

            // try doing automatically
            return false;
        }
    case kIROp_LookupWitness:
        {
            emitInstExpr(inst->getOperand(0), inOuterPrec);
            m_writer->emit("->");
            m_writer->emit(getName(inst->getOperand(1)));
            return true;
        }
    case kIROp_GetSequentialID:
        {
            emitInstExpr(inst->getOperand(0), inOuterPrec);
            m_writer->emit("->sequentialID");
            return true;
        }
    case kIROp_WitnessTable:
        {
            m_writer->emit("(&");
            m_writer->emit(getName(inst));
            m_writer->emit(")");
            return true;
        }
    case kIROp_GetAddr:
        {
            // Once we clean up the pointer emitting logic, we can
            // just use GetElementAddress instruction in place of
            // getAddr instruction, and this case can be removed.
            m_writer->emit("(&(");
            emitInstExpr(inst->getOperand(0), EmitOpInfo::get(EmitOp::General));
            m_writer->emit("))");
            return true;
        }
    case kIROp_RTTIObject:
        {
            m_writer->emit(getName(inst));
            return true;
        }
    case kIROp_Alloca:
        {
            m_writer->emit("alloca(");
            emitOperand(inst->getOperand(0), EmitOpInfo::get(EmitOp::Postfix));
            m_writer->emit("->typeSize)");
            return true;
        }
    case kIROp_BitCast:
        {
            m_writer->emit("(slang_bit_cast<");
            emitType(inst->getDataType());
            m_writer->emit(">(");
            emitOperand(inst->getOperand(0), getInfo(EmitOp::General));
            m_writer->emit("))");
            return true;
        }
    case kIROp_StringLit:
        {
            auto handler = StringEscapeUtil::getHandler(StringEscapeUtil::Style::Cpp);

            StringBuilder buf;
            const auto slice = as<IRStringLit>(inst)->getStringSlice();
            StringEscapeUtil::appendQuoted(handler, slice, buf);

            if (m_hasString)
            {
                m_writer->emit("Slang::toTerminatedSlice(");
                m_writer->emit(buf);
                m_writer->emit(")");
            }
            else
            {
                m_writer->emit(buf);
            }

            return true;
        }
    case kIROp_PtrLit:
        {
            auto ptrVal = as<IRPtrLit>(inst)->value.ptrVal;
            if (ptrVal == nullptr)
            {
                m_writer->emit("nullptr");
            }
            else
            {
                m_writer->emit("reinterpret_cast<");
                emitType(inst->getFullType());
                m_writer->emit(">(");
                m_writer->emitUInt64((uint64_t)ptrVal);
                m_writer->emit(")");
            }
            return true;
        }
    case kIROp_MakeExistential:
    case kIROp_MakeExistentialWithRTTI:
        {
            auto rsType = cast<IRComPtrType>(inst->getDataType());
            m_writer->emit("ComPtr<");
            m_writer->emit(getName(rsType->getOperand(0)));
            m_writer->emit(">(");
            m_writer->emit("static_cast<");
            m_writer->emit(getName(rsType->getOperand(0)));
            m_writer->emit("*>(");
            auto prec = getInfo(EmitOp::Postfix);
            emitOperand(inst->getOperand(0), leftSide(getInfo(EmitOp::General), prec));
            m_writer->emit(".get()");
            m_writer->emit("))");
            return true;
        }
    case kIROp_GetValueFromBoundInterface:
        {
            m_writer->emit("static_cast<");
            m_writer->emit(getName(inst->getFullType()));
            m_writer->emit("*>(");
            auto prec = getInfo(EmitOp::Postfix);
            emitOperand(inst->getOperand(0), leftSide(getInfo(EmitOp::General), prec));
            m_writer->emit(".get()");
            m_writer->emit(")");
            return true;
        }
    case kIROp_Select:
        {
            m_writer->emit("_slang_select(");
            emitOperand(inst->getOperand(0), getInfo(EmitOp::General));
            m_writer->emit(", ");
            emitOperand(inst->getOperand(1), getInfo(EmitOp::General));
            m_writer->emit(",");
            emitOperand(inst->getOperand(2), getInfo(EmitOp::General));
            m_writer->emit(")");
            return true;
        }
    }
}

void CPPSourceEmitter::emitPreModuleImpl()
{
    if (m_target == CodeGenTarget::CPPSource)
    {
        // TODO(JS): Previously this opened an anonymous scope for all generated functions
        // Unfortunately this is a problem if we are just emitting code that is externally available
        // and is not only accessible through entry points. So for now we disable

        // that this opens an anonymous scope.
        // The scope is closed in `emitModuleImpl`

        // m_writer->emit("namespace { // anonymous \n\n");

        // When generating kernel code in C++, put all into an anonymous namespace
        // This includes any generated types, and generated intrinsics.
        m_writer->emit("#ifdef SLANG_PRELUDE_NAMESPACE\n");
        m_writer->emit("using namespace SLANG_PRELUDE_NAMESPACE;\n");
        m_writer->emit("#endif\n\n");
    }
    else if (m_target == CodeGenTarget::HostCPPSource)
    {
        m_writer->emit("namespace Slang{ inline void handleSignal(SignalType, char const*) {} }\n");
    }
    Super::emitPreModuleImpl();
}


void CPPSourceEmitter::emitGlobalInstImpl(IRInst* inst)
{
    if (as<IRGlobalVar>(inst) && inst->findDecoration<IRExternCppDecoration>())
    {
        // JS:
        // Turns out just doing extern "C" means something different on a variable
        // So we need to wrap in extern "C" { }
        m_writer->emit("extern \"C\" {\n");
        Super::emitGlobalInstImpl(inst);
        m_writer->emit("\n}\n");
    }
    else
    {
        Super::emitGlobalInstImpl(inst);
    }
}

bool CPPSourceEmitter::shouldFoldInstIntoUseSites(IRInst* inst)
{
    bool result = Super::shouldFoldInstIntoUseSites(inst);
    if (!result)
        return result;
    if (as<IRVectorType>(inst->getDataType()) || as<IRMatrixType>(inst->getDataType()))
    {
        // If a vector value is being used in a reshape/cast,
        // we should not fold it because the implementation of cast will have multiple references to
        // it.
        for (auto use = inst->firstUse; use; use = use->nextUse)
        {
            switch (use->getUser()->getOp())
            {
            case kIROp_MatrixReshape:
            case kIROp_VectorReshape:
            case kIROp_IntCast:
            case kIROp_FloatCast:
            case kIROp_CastIntToFloat:
            case kIROp_CastFloatToInt:
                return false;
            default:
                break;
            }
        }
        switch (inst->getOp())
        {
        case kIROp_MatrixReshape:
        case kIROp_VectorReshape:
        case kIROp_IntCast:
        case kIROp_FloatCast:
        case kIROp_CastIntToFloat:
        case kIROp_CastFloatToInt:
            return false;
        default:
            break;
        }
    }
    return true;
}

static bool _isExported(IRInst* inst)
{
    for (auto decoration : inst->getDecorations())
    {
        const auto op = decoration->getOp();
        if (op == kIROp_PublicDecoration || op == kIROp_HLSLExportDecoration)
        {
            return true;
        }
    }
    return false;
}

void CPPSourceEmitter::emitVarDecorationsImpl(IRInst* inst)
{
    if (as<IRGlobalVar>(inst) && _isExported(inst))
    {
        m_writer->emit("SLANG_PRELUDE_SHARED_LIB_EXPORT\n");
    }

    Super::emitVarDecorationsImpl(inst);
}

void CPPSourceEmitter::_getExportStyle(IRInst* inst, bool& outIsExport, bool& outIsExternC)
{
    outIsExport = false;
    outIsExternC = false;
    // Specially handle export, as we don't want to emit it multiple times
    if (getTargetProgram()->getOptionSet().getBoolOption(CompilerOptionName::GenerateWholeProgram))
    {
        if (auto nameHint = inst->findDecoration<IRNameHintDecoration>())
        {
            if (nameHint->getName() == "main")
            {
                // Don't output any decorations on main function.
                return;
            }
        }

        // If public/export made it externally visible
        for (auto decoration : inst->getDecorations())
        {
            const auto op = decoration->getOp();
            if (op == kIROp_ExternCppDecoration)
            {
                outIsExternC = true;
            }
            else if (op == kIROp_PublicDecoration || op == kIROp_HLSLExportDecoration)
            {
                outIsExport = true;
            }
        }
    }
}

void CPPSourceEmitter::_maybeEmitExportLike(IRInst* inst)
{
    bool isExternC = false;
    bool isExported = false;
    _getExportStyle(inst, isExternC, isExported);

    // TODO(JS): Currently export *also* implies it's extern "C" and we can't list twice
    if (isExported)
    {
        m_writer->emit("SLANG_PRELUDE_EXPORT\n");
    }
    else if (isExternC)
    {
        // It's name is not manged.
        m_writer->emit("extern \"C\"\n");
    }
}

/* virtual */ void CPPSourceEmitter::emitFuncDecorationsImpl(IRFunc* func)
{
    _maybeEmitExportLike(func);

    // Use the default for others
    Super::emitFuncDecorationsImpl(func);
}

void CPPSourceEmitter::emitOperandImpl(IRInst* inst, EmitOpInfo const& outerPrec)
{
    if (shouldFoldInstIntoUseSites(inst))
    {
        emitInstExpr(inst, outerPrec);
        return;
    }

    switch (inst->getOp())
    {
    case kIROp_Var:
    case kIROp_GlobalVar:
        emitVarExpr(inst, outerPrec);
        break;
    default:
        m_writer->emit(getName(inst));
        break;
    }
}

/* static */ bool CPPSourceEmitter::_isVariable(IROp op)
{
    switch (op)
    {
    case kIROp_GlobalVar:
    case kIROp_GlobalParam:
        // case kIROp_Var:
        {
            return true;
        }
    default:
        return false;
    }
}

static bool _isFunction(IROp op)
{
    return op == kIROp_Func;
}

void CPPSourceEmitter::_emitEntryPointDefinitionStart(
    IRFunc* func,
    const String& funcName,
    const UnownedStringSlice& varyingTypeName)
{
    auto resultType = func->getResultType();

    auto entryPointDecl = func->findDecoration<IREntryPointDecoration>();
    SLANG_ASSERT(entryPointDecl);

    // Emit the actual function
    emitEntryPointAttributes(func, entryPointDecl);
    emitType(resultType, funcName);

    m_writer->emit("(");
    m_writer->emit(varyingTypeName);
    m_writer->emit("* varyingInput, void* entryPointParams, void* globalParams)");
    emitSemantics(func);
    m_writer->emit("\n{\n");

    m_writer->indent();
}

void CPPSourceEmitter::_emitEntryPointDefinitionEnd(IRFunc* func)
{
    SLANG_UNUSED(func);
    m_writer->dedent();
    m_writer->emit("}\n");
}

namespace
{ // anonymous

struct AxisWithSize
{
    typedef AxisWithSize ThisType;
    bool operator<(const ThisType& rhs) const
    {
        return size < rhs.size || (size == rhs.size && axis < rhs.axis);
    }

    int axis;
    Int size;
};

} // namespace

static void _calcAxisOrder(
    const Int sizeAlongAxis[CLikeSourceEmitter::kThreadGroupAxisCount],
    bool allowSingle,
    List<AxisWithSize>& out)
{
    out.clear();
    // Add in order z,y,x, so by default (if we don't sort), x will be the inner loop
    for (int i = CLikeSourceEmitter::kThreadGroupAxisCount - 1; i >= 0; --i)
    {
        if (allowSingle || sizeAlongAxis[i] > 1)
        {
            AxisWithSize axisWithSize;
            axisWithSize.axis = i;
            axisWithSize.size = sizeAlongAxis[i];
            out.add(axisWithSize);
        }
    }

    // The sort here works to make the axis with the highest value the inner most loop.
    // Disabled for now to make the order well defined as x, y, z
    // axes.sort();
}

void CPPSourceEmitter::_emitEntryPointGroup(
    const Int sizeAlongAxis[kThreadGroupAxisCount],
    const String& funcName)
{
    List<AxisWithSize> axes;
    _calcAxisOrder(sizeAlongAxis, false, axes);

    // Open all the loops
    StringBuilder builder;
    for (Index i = 0; i < axes.getCount(); ++i)
    {
        const auto& axis = axes[i];
        builder.clear();
        const char elem[2] = {s_xyzwNames[axis.axis], 0};
        builder << "for (uint32_t " << elem << " = 0; " << elem << " < " << axis.size << "; ++"
                << elem << ")\n{\n";
        m_writer->emit(builder);
        m_writer->indent();

        builder.clear();
        builder << "threadInput.groupThreadID." << elem << " = " << elem << ";\n";
        m_writer->emit(builder);
    }

    // just call at inner loop point
    m_writer->emit("_");
    m_writer->emit(funcName);
    m_writer->emit("(&threadInput, entryPointParams, globalParams);\n");

    // Close all the loops
    for (Index i = Index(axes.getCount() - 1); i >= 0; --i)
    {
        m_writer->dedent();
        m_writer->emit("}\n");
    }
}

void CPPSourceEmitter::_emitEntryPointGroupRange(
    const Int sizeAlongAxis[kThreadGroupAxisCount],
    const String& funcName)
{
    List<AxisWithSize> axes;
    _calcAxisOrder(sizeAlongAxis, true, axes);

    // Open all the loops
    StringBuilder builder;
    for (Index i = 0; i < axes.getCount(); ++i)
    {
        const auto& axis = axes[i];
        builder.clear();
        const char elem[2] = {s_xyzwNames[axis.axis], 0};

        builder << "for (uint32_t " << elem << " = vi.startGroupID." << elem << "; " << elem
                << " < vi.endGroupID." << elem << "; ++" << elem << ")\n{\n";
        m_writer->emit(builder);
        m_writer->indent();

        m_writer->emit("groupVaryingInput.startGroupID.");
        m_writer->emit(elem);
        m_writer->emit(" = ");
        m_writer->emit(elem);
        m_writer->emit(";\n");
    }

    // just call at inner loop point
    m_writer->emit(funcName);
    m_writer->emit("_Group(&groupVaryingInput, entryPointParams, globalParams);\n");

    // Close all the loops
    for (Index i = Index(axes.getCount() - 1); i >= 0; --i)
    {
        m_writer->dedent();
        m_writer->emit("}\n");
    }
}
void CPPSourceEmitter::_emitInitAxisValues(
    const Int sizeAlongAxis[kThreadGroupAxisCount],
    const UnownedStringSlice& mulName,
    const UnownedStringSlice& addName)
{
    StringBuilder builder;

    m_writer->emit("{\n");
    m_writer->indent();
    for (int i = 0; i < kThreadGroupAxisCount; ++i)
    {
        builder.clear();
        const char elem[2] = {s_xyzwNames[i], 0};
        builder << mulName << "." << elem << " * " << sizeAlongAxis[i];
        if (addName.getLength() > 0)
        {
            builder << " + " << addName << "." << elem;
        }
        if (i < kThreadGroupAxisCount - 1)
        {
            builder << ",";
        }
        builder << "\n";
        m_writer->emit(builder);
    }
    m_writer->dedent();
    m_writer->emit("};\n");
}

void CPPSourceEmitter::_emitForwardDeclarations(const List<EmitAction>& actions)
{
    // Emit forward declarations. Don't emit variables that need to be grouped or function
    // definitions (which will ref those types)
    for (auto action : actions)
    {
        switch (action.level)
        {
        case EmitAction::Level::ForwardDeclaration:
            {
                switch (action.inst->getOp())
                {
                case kIROp_Func:
                case kIROp_StructType:
                case kIROp_InterfaceType:
                    emitForwardDeclaration(action.inst);
                    break;
                default:
                    break;
                }
            }
            break;

        case EmitAction::Level::Definition:
            if (_isVariable(action.inst->getOp()) || _isFunction(action.inst->getOp()))
            {
                // Don't emit functions or variables that have to be grouped into structures yet
            }
            else
            {
                emitGlobalInst(action.inst);
            }
            break;
        }
    }
}

void CPPSourceEmitter::emitModuleImpl(IRModule* module, DiagnosticSink* sink)
{
    SLANG_UNUSED(sink);

    List<EmitAction> actions;
    computeEmitActions(module, actions);

    _emitForwardDeclarations(actions);

    {
        // Output all the thread locals
        for (auto action : actions)
        {
            if (action.level == EmitAction::Level::Definition &&
                action.inst->getOp() == kIROp_GlobalVar)
            {
                emitGlobalInst(action.inst);
            }
        }

        // Finally output the functions as methods on the context
        for (auto action : actions)
        {
            if (action.level == EmitAction::Level::Definition && _isFunction(action.inst->getOp()))
            {
                emitGlobalInst(action.inst);
            }
        }
    }

    // Emit all witness table definitions.
    _emitWitnessTableDefinitions();

    // TODO(JS):
    // Previously output code was placed in an anonymous namespace
    // Now that we can have any function available externally (not just entry points)
    // this doesn't work.

    // if (m_target == CodeGenTarget::CPPSource)
    //{
    //  Need to close the anonymous namespace when outputting for C++ kernel.
    // m_writer->emit("} // anonymous\n\n");
    //}

    // Finally we need to output dll entry points

    for (auto action : actions)
    {
        if (action.level == EmitAction::Level::Definition && _isFunction(action.inst->getOp()))
        {
            IRFunc* func = as<IRFunc>(action.inst);

            IREntryPointDecoration* entryPointDecor =
                func->findDecoration<IREntryPointDecoration>();

            if (entryPointDecor && entryPointDecor->getProfile().getStage() == Stage::Compute)
            {
                Int groupThreadSize[kThreadGroupAxisCount];
                getComputeThreadGroupSize(func, groupThreadSize);

                String funcName = getName(func);

                {
                    StringBuilder builder;
                    builder << funcName << "_Thread";

                    String threadFuncName = builder;

                    _emitEntryPointDefinitionStart(
                        func,
                        threadFuncName,
                        UnownedStringSlice::fromLiteral("ComputeThreadVaryingInput"));

                    m_writer->emit("_");
                    m_writer->emit(funcName);
                    m_writer->emit("(varyingInput, entryPointParams, globalParams);\n");

                    _emitEntryPointDefinitionEnd(func);
                }

                // Emit the group version which runs for all elements in *single* thread group
                {
                    StringBuilder builder;
                    builder << getName(func);
                    builder << "_Group";

                    String groupFuncName = builder;

                    _emitEntryPointDefinitionStart(
                        func,
                        groupFuncName,
                        UnownedStringSlice::fromLiteral("ComputeVaryingInput"));

                    m_writer->emit("ComputeThreadVaryingInput threadInput = {};\n");
                    m_writer->emit("threadInput.groupID = varyingInput->startGroupID;\n");

                    _emitEntryPointGroup(groupThreadSize, funcName);
                    _emitEntryPointDefinitionEnd(func);
                }

                // Emit the main version - which takes a dispatch size
                {
                    _emitEntryPointDefinitionStart(
                        func,
                        funcName,
                        UnownedStringSlice::fromLiteral("ComputeVaryingInput"));

                    m_writer->emit("ComputeVaryingInput vi = *varyingInput;\n");
                    m_writer->emit("ComputeVaryingInput groupVaryingInput = {};\n");

                    _emitEntryPointGroupRange(groupThreadSize, funcName);
                    _emitEntryPointDefinitionEnd(func);
                }
            }
        }
    }
}

} // namespace Slang
