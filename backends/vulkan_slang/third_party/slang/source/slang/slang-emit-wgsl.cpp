#include "slang-emit-wgsl.h"

#include "slang-ir-layout.h"
#include "slang-ir-util.h"

// A note on row/column "terminology reversal".
//
// This is an "terminology reversing" implementation in the sense that
// * "column" in Slang code maps to "row" in the generated WGSL code, and
// * "row" in Slang code maps to "column" in the generated WGSL code.
//
// This means that matrices in Slang code end up getting translated to
// matrices that actually represent the transpose of what the Slang matrix
// represented.
// Both API's adopt the standard matrix multiplication convention whereby the
// column count of the matrix on the left hand side needs to match row count of
// the matrix on the right hand side.
// For these reasons, and due to the fact that (M_1 ... M_n)^T = M_n^T ... M_1^T,
// the order of matrix (and vector-matrix products) products must also reversed
// in the WGSL code.
//
// This may lead to confusion (which is why this note is referenced in several
// places), but the benefit of doing this is that the generated WGSL code is
// simpler to generate and should be faster to compile.
// A "terminology preserving" implementation would have to generate lots of
// 'transpose' calls, or else perform more complicated transformations that
// end up duplicating expressions many times.

namespace Slang
{

// In WGSL, expression of "1.0/0.0" is not allowed, it will report compile error,
// so to construct infinity or nan, we have to assign the float literal to a variable
// and then use it to bypass the compile error.
static const char* kWGSLBuiltinPreludeGetInfinity = R"(
fn _slang_getInfinity(positive: bool) -> f32
{
    let a = select(f32(-1.0), f32(1.0), positive);
    let b = f32(0.0);
    return a / b;
}
)";

static const char* kWGSLBuiltinPreludeGetNan = R"(
fn _slang_getNan() -> f32
{
    let a = f32(0.0);
    let b = f32(0.0);
    return a / b;
}
)";

WGSLSourceEmitter::WGSLSourceEmitter(const Desc& desc)
    : CLikeSourceEmitter(desc)
{
    m_extensionTracker =
        dynamicCast<ShaderExtensionTracker>(desc.codeGenContext->getExtensionTracker());
    SLANG_ASSERT(m_extensionTracker);
}

void WGSLSourceEmitter::emitSwitchCaseSelectorsImpl(
    const SwitchRegion::Case* const currentCase,
    const bool isDefault)
{
    // WGSL has special syntax for blocks sharing case labels:
    // "case 2, 3, 4: ...;" instead of the C-like syntax
    // "case 2: case 3: case 4: ...;".

    m_writer->emit("case ");
    for (auto caseVal : currentCase->values)
    {
        emitOperand(caseVal, getInfo(EmitOp::General));
        m_writer->emit(", ");
    }
    if (isDefault)
    {
        m_writer->emit("default, ");
    }
    m_writer->emit(":\n");
}

void WGSLSourceEmitter::emitParameterGroupImpl(
    IRGlobalParam* varDecl,
    IRUniformParameterGroupType* type)
{
    auto varLayout = getVarLayout(varDecl);
    SLANG_RELEASE_ASSERT(varLayout);

    EmitVarChain blockChain(varLayout);

    EmitVarChain containerChain = blockChain;
    EmitVarChain elementChain = blockChain;

    auto typeLayout = varLayout->getTypeLayout()->unwrapArray();
    if (auto parameterGroupTypeLayout = as<IRParameterGroupTypeLayout>(typeLayout))
    {
        containerChain =
            EmitVarChain(parameterGroupTypeLayout->getContainerVarLayout(), &blockChain);
        elementChain = EmitVarChain(parameterGroupTypeLayout->getElementVarLayout(), &blockChain);

        typeLayout = parameterGroupTypeLayout->getElementVarLayout()->getTypeLayout();
    }

    for (auto attr : containerChain.varLayout->getOffsetAttrs())
    {
        const LayoutResourceKind kind = attr->getResourceKind();
        switch (kind)
        {
        case LayoutResourceKind::VaryingInput:
        case LayoutResourceKind::VaryingOutput:
            m_writer->emit("@location(");
            m_writer->emit(attr->getOffset());
            m_writer->emit(")");
            if (attr->getSpace())
            {
                // TODO: Not sure what 'space' should map to in WGSL
                SLANG_ASSERT(false);
            }
            break;

        case LayoutResourceKind::SpecializationConstant:
            // TODO:
            // Consider moving to a differently named function.
            // This is not technically an attribute, but a declaration.
            //
            // https://www.w3.org/TR/WGSL/#override-decls
            m_writer->emit("override");
            break;

        case LayoutResourceKind::Uniform:
        case LayoutResourceKind::ConstantBuffer:
        case LayoutResourceKind::ShaderResource:
        case LayoutResourceKind::UnorderedAccess:
        case LayoutResourceKind::SamplerState:
        case LayoutResourceKind::DescriptorTableSlot:
            {
                auto kinds = LayoutResourceKindFlag::make(LayoutResourceKind::DescriptorTableSlot);
                m_writer->emit("@binding(");
                auto index = getBindingOffsetForKinds(&containerChain, kinds);
                m_writer->emit(index);
                m_writer->emit(") ");
                m_writer->emit("@group(");
                auto space = getBindingSpaceForKinds(&containerChain, kinds);
                m_writer->emit(space);
                m_writer->emit(") ");
            }
            break;
        }
    }

    auto elementType = type->getElementType();
    m_writer->emit("var<uniform> ");
    m_writer->emit(getName(varDecl));
    m_writer->emit(" : ");
    emitType(elementType);
    m_writer->emit(";\n");
}

void WGSLSourceEmitter::emitEntryPointAttributesImpl(
    IRFunc* irFunc,
    IREntryPointDecoration* entryPointDecor)
{
    auto stage = entryPointDecor->getProfile().getStage();

    switch (stage)
    {

    case Stage::Fragment:
        m_writer->emit("@fragment\n");
        break;
    case Stage::Vertex:
        m_writer->emit("@vertex\n");
        break;

    case Stage::Compute:
        {
            m_writer->emit("@compute\n");

            {
                Int sizeAlongAxis[kThreadGroupAxisCount];
                getComputeThreadGroupSize(irFunc, sizeAlongAxis);

                m_writer->emit("@workgroup_size(");
                for (int ii = 0; ii < kThreadGroupAxisCount; ++ii)
                {
                    if (ii != 0)
                        m_writer->emit(", ");
                    m_writer->emit(sizeAlongAxis[ii]);
                }
                m_writer->emit(")\n");
            }
        }
        break;

    default:
        SLANG_ABORT_COMPILATION("unsupported stage.");
    }
}

// This is 'function_header' from the WGSL specification
void WGSLSourceEmitter::emitFuncHeaderImpl(IRFunc* func)
{
    Slang::IRType* resultType = func->getResultType();
    auto name = getName(func);

    m_writer->emit("fn ");
    m_writer->emit(name);

    emitSimpleFuncParamsImpl(func);

    // An absence of return type is expressed by skipping the optional '->' part of the
    // header.
    if (resultType->getOp() != kIROp_VoidType)
    {
        m_writer->emit(" -> ");
        emitType(resultType);
    }
}

void WGSLSourceEmitter::emitSimpleFuncParamImpl(IRParam* param)
{
    if (auto sysSemanticDecor = param->findDecoration<IRTargetSystemValueDecoration>())
    {
        m_writer->emit("@builtin(");
        m_writer->emit(sysSemanticDecor->getSemantic());
        m_writer->emit(")");
    }

    CLikeSourceEmitter::emitSimpleFuncParamImpl(param);
}

void WGSLSourceEmitter::emitMatrixType(
    IRType* const elementType,
    const IRIntegerValue& rowCountWGSL,
    const IRIntegerValue& colCountWGSL)
{
    // WGSL uses CxR convention
    m_writer->emit("mat");
    m_writer->emit(colCountWGSL);
    m_writer->emit("x");
    m_writer->emit(rowCountWGSL);
    m_writer->emit("<");
    emitType(elementType);
    m_writer->emit(">");
}

void WGSLSourceEmitter::emitStructDeclarationSeparatorImpl()
{
    m_writer->emit(",");
}

static bool isPowerOf2(const uint32_t n)
{
    return (n != 0U) && ((n - 1U) & n) == 0U;
}

bool WGSLSourceEmitter::maybeEmitSystemSemantic(IRInst* inst)
{
    if (auto sysSemanticDecor = inst->findDecoration<IRTargetSystemValueDecoration>())
    {
        m_writer->emit("@builtin(");
        m_writer->emit(sysSemanticDecor->getSemantic());
        m_writer->emit(")");
        return true;
    }
    return false;
}

void WGSLSourceEmitter::emitSemanticsPrefixImpl(IRInst* inst)
{
    if (!maybeEmitSystemSemantic(inst))
    {
        if (auto semanticDecoration = inst->findDecoration<IRSemanticDecoration>())
        {
            m_writer->emit("@location(");
            m_writer->emit(semanticDecoration->getSemanticIndex());
            m_writer->emit(")");
            return;
        }
    }
}

void WGSLSourceEmitter::emitStructFieldAttributes(
    IRStructType* structType,
    IRStructField* field,
    bool allowOffsetLayout)
{
    SLANG_UNUSED(allowOffsetLayout);

    // Tint emits errors unless we explicitly spell out the layout in some cases, so emit
    // offset and align attribtues for all fields.
    IRSizeAndAlignmentDecoration* const sizeAndAlignmentDecoration =
        structType->findDecoration<IRSizeAndAlignmentDecoration>();
    // NullDifferential struct doesn't have size and alignment decoration
    if (sizeAndAlignmentDecoration == nullptr)
        return;
    SLANG_ASSERT(sizeAndAlignmentDecoration->getAlignment() > IRIntegerValue{0});
    SLANG_ASSERT(sizeAndAlignmentDecoration->getAlignment() <= IRIntegerValue{UINT32_MAX});
    const uint32_t structAlignment =
        static_cast<uint32_t>(sizeAndAlignmentDecoration->getAlignment());
    IROffsetDecoration* const fieldOffsetDecoration = field->findDecoration<IROffsetDecoration>();
    SLANG_ASSERT(fieldOffsetDecoration->getOffset() >= IRIntegerValue{0});
    SLANG_ASSERT(fieldOffsetDecoration->getOffset() <= IRIntegerValue{UINT32_MAX});
    SLANG_ASSERT(isPowerOf2(structAlignment));
    const uint32_t fieldOffset = static_cast<uint32_t>(fieldOffsetDecoration->getOffset());
    // Alignment is GCD(fieldOffset, structAlignment)
    // TODO: Use builtin/intrinsic (e.g. __builtin_ffs)
    uint32_t fieldAlignment = 1U;
    while (((fieldAlignment & (structAlignment | fieldOffset)) == 0U))
        fieldAlignment = fieldAlignment << 1U;

    m_writer->emit("@align(");
    m_writer->emit(fieldAlignment);
    m_writer->emit(")");
}

void WGSLSourceEmitter::emit(const AddressSpace addressSpace)
{
    switch (addressSpace)
    {
    case AddressSpace::Uniform:
        m_writer->emit("uniform");
        break;

    case AddressSpace::StorageBuffer:
        m_writer->emit("storage");
        break;

    case AddressSpace::Generic:
        m_writer->emit("function");
        break;

    case AddressSpace::ThreadLocal:
        m_writer->emit("private");
        break;

    case AddressSpace::GroupShared:
        m_writer->emit("workgroup");
        break;
    }
}

const char* WGSLSourceEmitter::getWgslImageFormat(IRTextureTypeBase* type)
{
    // You can find the supported WGSL texel format from the URL:
    // https://www.w3.org/TR/WGSL/#storage-texel-formats
    //
    ImageFormat imageFormat =
        type->hasFormat() ? (ImageFormat)type->getFormat() : ImageFormat::unknown;

    if (imageFormat == ImageFormat::unknown)
    {
        // WGSL doesn't have a texel format for "unknown" so we try to infer float types that
        // normally just resolve to unknown.
        auto elementType = type->getElementType();
        Int vectorWidth = 1;
        if (auto elementVecType = as<IRVectorType>(elementType))
        {
            if (auto intLitVal = as<IRIntLit>(elementVecType->getElementCount()))
            {
                vectorWidth = (Int)intLitVal->getValue();
            }
            else
            {
                vectorWidth = 0;
            }
            elementType = elementVecType->getElementType();
        }
        if (auto basicType = as<IRBasicType>(elementType))
        {
            switch (basicType->getBaseType())
            {
            case BaseType::Float:
                switch (vectorWidth)
                {
                case 1:
                    return "r32float";
                case 2:
                    return "rg32float";
                case 4:
                    return "rgba32float";
                }
                break;
            }
        }
    }

    switch (imageFormat)
    {
    case ImageFormat::rgba8:
        return "rgba8unorm";
    case ImageFormat::rgba8_snorm:
        return "rgba8snorm";
    case ImageFormat::rgba8ui:
        return "rgba8uint";
    case ImageFormat::rgba8i:
        return "rgba8sint";
    case ImageFormat::rgba16ui:
        return "rgba16uint";
    case ImageFormat::rgba16i:
        return "rgba16sint";
    case ImageFormat::rgba16f:
        return "rgba16float";
    case ImageFormat::r32ui:
        return "r32uint";
    case ImageFormat::r32i:
        return "r32sint";
    case ImageFormat::r32f:
        return "r32float";
    case ImageFormat::rg32ui:
        return "rg32uint";
    case ImageFormat::rg32i:
        return "rg32sint";
    case ImageFormat::rg32f:
        return "rg32float";
    case ImageFormat::rgba32ui:
        return "rgba32uint";
    case ImageFormat::rgba32i:
        return "rgba32sint";
    case ImageFormat::rgba32f:
        return "rgba32float";
    case ImageFormat::bgra8:
        return "bgra8unorm";
    case ImageFormat::unknown:
        // Unlike SPIR-V, WGSL doesn't have a texel format for "unknown".
        return "rgba32float";
    default:
        const auto imageFormatInfo = getImageFormatInfo(imageFormat);
        getSink()->diagnose(
            SourceLoc(),
            Diagnostics::imageFormatUnsupportedByBackend,
            imageFormatInfo.name,
            "WGSL",
            "rgba32float");
        return "rgba32float";
    }
}

void WGSLSourceEmitter::emitSimpleTypeImpl(IRType* type)
{
    switch (type->getOp())
    {

    case kIROp_HLSLRWStructuredBufferType:
    case kIROp_HLSLStructuredBufferType:
    case kIROp_HLSLRasterizerOrderedStructuredBufferType:
        {
            auto structuredBufferType = as<IRHLSLStructuredBufferTypeBase>(type);
            m_writer->emit("array");
            m_writer->emit("<");
            emitType(structuredBufferType->getElementType());
            m_writer->emit(">");
        }
        break;

    case kIROp_HLSLByteAddressBufferType:
    case kIROp_HLSLRWByteAddressBufferType:
        {
            m_writer->emit("array<u32>");
        }
        break;

    case kIROp_VoidType:
        {
            // There is no void type in WGSL.
            // A return type of "void" is expressed by skipping the end part of the
            // 'function_header' term:
            // "
            // function_header :
            //   'fn' ident '(' param_list ? ')'
            //       ( '->' attribute * template_elaborated_ident ) ?
            // "
            // In other words, in WGSL we should never even get to the point where we're
            // asking to emit 'void'.
            SLANG_UNEXPECTED("'void' type emitted");
            return;
        }

    case kIROp_FloatType:
        m_writer->emit("f32");
        break;
    case kIROp_DoubleType:
        // There is no "f64" type in WGSL
        SLANG_UNEXPECTED("'double' type emitted");
        break;
    case kIROp_Int8Type:
    case kIROp_UInt8Type:
        // There is no "[i|u]8" type in WGSL
        SLANG_UNEXPECTED("8 bit integer type emitted");
        break;
    case kIROp_HalfType:
        m_f16ExtensionEnabled = true;
        m_writer->emit("f16");
        break;
    case kIROp_BoolType:
        m_writer->emit("bool");
        break;
    case kIROp_IntType:
        m_writer->emit("i32");
        break;
    case kIROp_UIntType:
        m_writer->emit("u32");
        break;
    case kIROp_UInt64Type:
        {
            m_writer->emit(getDefaultBuiltinTypeName(type->getOp()));
            return;
        }
    case kIROp_Int16Type:
    case kIROp_UInt16Type:
        SLANG_UNEXPECTED("16 bit integer value emitted");
        return;
    case kIROp_Int64Type:
    case kIROp_IntPtrType:
        m_writer->emit("i64");
        return;
    case kIROp_UIntPtrType:
        m_writer->emit("u64");
        return;
    case kIROp_StructType:
        m_writer->emit(getName(type));
        return;

    case kIROp_VectorType:
        {
            auto vecType = (IRVectorType*)type;
            emitVectorTypeNameImpl(
                vecType->getElementType(),
                getIntVal(vecType->getElementCount()));
            return;
        }
    case kIROp_MatrixType:
        {
            auto matType = (IRMatrixType*)type;
            // We map matrices in Slang to WGSL matrices that represent the transpose.
            // (See note on "terminology reversal".)
            const IRIntegerValue colCountWGSL = getIntVal(matType->getRowCount());
            const IRIntegerValue rowCountWGSL = getIntVal(matType->getColumnCount());
            emitMatrixType(matType->getElementType(), rowCountWGSL, colCountWGSL);
            return;
        }
    case kIROp_SamplerStateType:
        {
            m_writer->emit("sampler");
            return;
        }

    case kIROp_SamplerComparisonStateType:
        {
            m_writer->emit("sampler_comparison");
            return;
        }

    case kIROp_PtrType:
    case kIROp_InOutType:
    case kIROp_OutType:
    case kIROp_RefType:
    case kIROp_ConstRefType:
        {
            auto ptrType = cast<IRPtrTypeBase>(type);
            m_writer->emit("ptr<");
            emit((AddressSpace)ptrType->getAddressSpace());
            m_writer->emit(", ");
            emitType((IRType*)ptrType->getValueType());
            m_writer->emit(">");
            return;
        }

    case kIROp_ArrayType:
        {
            m_writer->emit("array<");
            emitType((IRType*)type->getOperand(0));
            m_writer->emit(", ");
            emitVal(type->getOperand(1), getInfo(EmitOp::General));
            m_writer->emit(">");
            return;
        }
    case kIROp_UnsizedArrayType:
        {
            m_writer->emit("array<");
            emitType((IRType*)type->getOperand(0));
            m_writer->emit(">");
            return;
        }
    case kIROp_TextureType:
        if (auto texType = as<IRTextureType>(type))
        {
            switch (texType->getAccess())
            {
            case SLANG_RESOURCE_ACCESS_WRITE:
            case SLANG_RESOURCE_ACCESS_READ_WRITE:
                m_writer->emit("texture_storage");
                break;
            default:
                m_writer->emit("texture");
                break;
            }

            if (texType->isShadow())
            {
                m_writer->emit("_depth");
            }

            if (texType->isMultisample())
            {
                m_writer->emit("_multisampled");
            }

            switch (texType->GetBaseShape())
            {
            case SLANG_TEXTURE_1D:
                m_writer->emit("_1d");
                break;
            case SLANG_TEXTURE_2D:
                m_writer->emit("_2d");
                break;
            case SLANG_TEXTURE_3D:
                m_writer->emit("_3d");
                break;
            case SLANG_TEXTURE_CUBE:
                m_writer->emit("_cube");
                break;
            }

            if (texType->isArray())
                m_writer->emit("_array");

            if (!texType->isShadow())
            {
                m_writer->emit("<");

                auto elemType = texType->getElementType();

                switch (texType->getAccess())
                {
                case SLANG_RESOURCE_ACCESS_READ_WRITE:
                    m_writer->emit(getWgslImageFormat(texType));
                    m_writer->emit(", read_write");
                    break;
                case SLANG_RESOURCE_ACCESS_WRITE:
                    m_writer->emit(getWgslImageFormat(texType));
                    m_writer->emit(", write");
                    break;
                default:
                    if (auto vecElemType = as<IRVectorType>(elemType))
                        emitSimpleType(vecElemType->getElementType());
                    else
                        emitType(elemType);
                    break;
                }

                m_writer->emit(">");
            }
        }
        return;

    case kIROp_AtomicType:
        {
            m_writer->emit("atomic<");
            emitType(cast<IRAtomicType>(type)->getElementType());
            m_writer->emit(">");
            return;
        }
    case kIROp_ConstantBufferType:
        {
            emitType((IRType*)type->getOperand(0));
            return;
        }
    default:
        break;
    }
}

void WGSLSourceEmitter::emitGlobalParamDefaultVal(IRGlobalParam* varDecl)
{
    auto layout = getVarLayout(varDecl);
    if (!layout)
        return;
    if (layout->findOffsetAttr(LayoutResourceKind::SpecializationConstant))
    {
        if (auto defaultValDecor = varDecl->findDecoration<IRDefaultValueDecoration>())
        {
            m_writer->emit(" = ");
            emitInstExpr(defaultValDecor->getOperand(0), EmitOpInfo());
        }
    }
}

void WGSLSourceEmitter::emitLayoutQualifiersImpl(IRVarLayout* layout)
{

    for (auto attr : layout->getOffsetAttrs())
    {
        LayoutResourceKind kind = attr->getResourceKind();

        // TODO:
        // This is not correct. For the moment this is just here as a hack to make
        // @binding and @group unique, so that we can pass WGSL compile tests.
        // This will have to be revisited when we actually want to supply resources to
        // shaders.
        if (kind == LayoutResourceKind::DescriptorTableSlot)
        {
            m_writer->emit("@binding(");
            m_writer->emit(attr->getOffset());
            m_writer->emit(") ");

            EmitVarChain chain = {};
            chain.varLayout = layout;
            auto space = getBindingSpaceForKinds(&chain, LayoutResourceKindFlag::make(kind));
            m_writer->emit("@group(");
            m_writer->emit(space);
            m_writer->emit(") ");

            return;
        }
        else if (kind == LayoutResourceKind::SpecializationConstant)
        {
            m_writer->emit("@id(");
            m_writer->emit(attr->getOffset());
            m_writer->emit(") ");

            return;
        }
    }
}

static bool isStaticConst(IRInst* inst)
{
    if (inst->getParent()->getOp() == kIROp_Module)
    {
        return true;
    }
    switch (inst->getOp())
    {
    case kIROp_MakeVector:
    case kIROp_swizzle:
    case kIROp_swizzleSet:
    case kIROp_IntCast:
    case kIROp_FloatCast:
    case kIROp_CastFloatToInt:
    case kIROp_CastIntToFloat:
    case kIROp_BitCast:
        {
            for (UInt i = 0; i < inst->getOperandCount(); i++)
            {
                if (!isStaticConst(inst->getOperand(i)))
                    return false;
            }
            return true;
        }
    }
    return false;
}

void WGSLSourceEmitter::emitVarKeywordImpl(IRType* type, IRInst* varDecl)
{
    switch (varDecl->getOp())
    {
    case kIROp_GlobalParam:
    case kIROp_GlobalVar:
    case kIROp_Var:
        {
            auto layout = getVarLayout(varDecl);
            if (layout && layout->findOffsetAttr(LayoutResourceKind::SpecializationConstant))
            {
                m_writer->emit("override");
                break;
            }
            m_writer->emit("var");
        }
        break;
    default:
        if (isStaticConst(varDecl))
            m_writer->emit("const");
        else
            m_writer->emit("var");
        break;
    }

    if (as<IRGroupSharedRate>(varDecl->getRate()))
    {
        m_writer->emit("<workgroup>");
    }
    else if (
        type->getOp() == kIROp_ArrayType &&
        type->getOperand(0)->getOp() == kIROp_ConstantBufferType)
    {
        // Arrays of constant buffers should use the uniform keyword.
        m_writer->emit("<uniform>");
    }
    else if (
        type->getOp() == kIROp_HLSLRWStructuredBufferType ||
        type->getOp() == kIROp_HLSLRasterizerOrderedStructuredBufferType ||
        type->getOp() == kIROp_HLSLRWByteAddressBufferType)
    {
        m_writer->emit("<");
        m_writer->emit("storage, read_write");
        m_writer->emit(">");
    }
    else if (
        type->getOp() == kIROp_HLSLStructuredBufferType ||
        type->getOp() == kIROp_HLSLByteAddressBufferType)
    {
        m_writer->emit("<");
        m_writer->emit("storage, read");
        m_writer->emit(">");
    }
    else if (varDecl->getOp() == kIROp_GlobalVar)
    {
        // Global ("module-scope") non-handle variables need to specify storage space

        // https://www.w3.org/TR/WGSL/#var-decls
        // "
        // Variables in the private, storage, uniform, workgroup, and handle address
        // spaces must only be declared in module scope, while variables in the function
        // address space must only be declared in function scope. The address space must
        // be specified for all address spaces except handle and function. The handle
        // address space must not be specified. Specifying the function address space is
        // optional.
        // "
        m_writer->emit("<private>");
    }
}

void WGSLSourceEmitter::_emitType(IRType* type, DeclaratorInfo* declarator)
{
    // C-like languages bake array-ness, pointer-ness and reference-ness into the
    // declarator, which happens in the default _emitType implementation.
    // WGSL on the other hand, don't have special syntax -- these are just types.
    switch (type->getOp())
    {
    case kIROp_ArrayType:
    case kIROp_AttributedType:
    case kIROp_UnsizedArrayType:
        emitSimpleTypeAndDeclarator(type, declarator);
        break;
    default:
        CLikeSourceEmitter::_emitType(type, declarator);
        break;
    }
}

void WGSLSourceEmitter::emitDeclaratorImpl(DeclaratorInfo* declarator)
{
    if (!declarator)
        return;

    m_writer->emit(" ");

    switch (declarator->flavor)
    {
    case DeclaratorInfo::Flavor::Name:
        {
            auto nameDeclarator = (NameDeclaratorInfo*)declarator;
            m_writer->emitName(*nameDeclarator->nameAndLoc);
        }
        break;

    case DeclaratorInfo::Flavor::SizedArray:
        {
            // Sized arrays are just types (array<T, N>) in WGSL -- they are not
            // supported at the syntax level
            // https://www.w3.org/TR/WGSL/#array
            SLANG_UNEXPECTED("Sized array declarator");
        }
        break;

    case DeclaratorInfo::Flavor::UnsizedArray:
        {
            // Unsized arrays are just types (array<T>) in WGSL -- they are not
            // supported at the syntax level
            // https://www.w3.org/TR/WGSL/#array
            SLANG_UNEXPECTED("Unsized array declarator");
        }
        break;

    case DeclaratorInfo::Flavor::Ptr:
        {
            // Pointers (ptr<AS,T,AM>) are just types in WGSL -- they are not supported at
            // the syntax level
            // https://www.w3.org/TR/WGSL/#ref-ptr-types
            SLANG_UNEXPECTED("Pointer declarator");
        }
        break;

    case DeclaratorInfo::Flavor::Ref:
        {
            // References (ref<AS,T,AM>) are just types in WGSL -- they are not supported
            // at the syntax level
            // https://www.w3.org/TR/WGSL/#ref-ptr-types
            SLANG_UNEXPECTED("Reference declarator");
        }
        break;

    case DeclaratorInfo::Flavor::LiteralSizedArray:
        {
            // Sized arrays are just types (array<T, N>) in WGSL -- they are not supported
            // at the syntax level
            // https://www.w3.org/TR/WGSL/#array
            SLANG_UNEXPECTED("Literal-sized array declarator");
        }
        break;

    case DeclaratorInfo::Flavor::Attributed:
        {
            auto attributedDeclarator = (AttributedDeclaratorInfo*)declarator;
            auto instWithAttributes = attributedDeclarator->instWithAttributes;
            for (auto attr : instWithAttributes->getAllAttrs())
            {
                _emitPostfixTypeAttr(attr);
            }
            emitDeclarator(attributedDeclarator->next);
        }
        break;

    default:
        SLANG_DIAGNOSE_UNEXPECTED(getSink(), SourceLoc(), "unknown declarator flavor");
        break;
    }
}

void WGSLSourceEmitter::emitOperandImpl(IRInst* operand, EmitOpInfo const& outerPrec)
{
    if (operand->getOp() == kIROp_Param && as<IRPtrTypeBase>(operand->getDataType()))
    {
        // If we are emitting a reference to a pointer typed operand, then
        // we should dereference it now since we want to treat all the remaining
        // part of wgsl as pointer-free target.
        m_writer->emit("(*");
        m_writer->emit(getName(operand));
        m_writer->emit(")");
    }
    else
    {
        CLikeSourceEmitter::emitOperandImpl(operand, outerPrec);
    }
}

void WGSLSourceEmitter::emitSimpleTypeAndDeclaratorImpl(IRType* type, DeclaratorInfo* declarator)
{
    if (declarator)
    {
        emitDeclarator(declarator);
        m_writer->emit(" : ");
    }
    emitSimpleType(type);
}

void WGSLSourceEmitter::emitSimpleValueImpl(IRInst* inst)
{
    switch (inst->getOp())
    {
    case kIROp_IntLit:
        {
            auto litInst = static_cast<IRConstant*>(inst);

            IRBasicType* type = as<IRBasicType>(inst->getDataType());
            if (type)
            {
                switch (type->getBaseType())
                {
                default:

                case BaseType::Int8:
                case BaseType::UInt8:
                    {
                        SLANG_UNEXPECTED("8 bit integer value emitted");
                        break;
                    }
                case BaseType::Int16:
                case BaseType::UInt16:
                    {
                        SLANG_UNEXPECTED("16 bit integer value emitted");
                        break;
                    }
                case BaseType::Int:
                    {
                        m_writer->emit("i32(");
                        m_writer->emit(int32_t(litInst->value.intVal));
                        m_writer->emit(")");
                        return;
                    }
                case BaseType::UInt:
                    {
                        m_writer->emit("u32(");
                        m_writer->emit(UInt(uint32_t(litInst->value.intVal)));
                        m_writer->emit(")");
                        break;
                    }
                case BaseType::Int64:
                    {
                        m_writer->emit("i64(");
                        m_writer->emitInt64(int64_t(litInst->value.intVal));
                        m_writer->emit(")");
                        break;
                    }
                case BaseType::UInt64:
                    {
                        m_writer->emit("u64(");
                        SLANG_COMPILE_TIME_ASSERT(
                            sizeof(litInst->value.intVal) >= sizeof(uint64_t));
                        m_writer->emitUInt64(uint64_t(litInst->value.intVal));
                        m_writer->emit(")");
                        break;
                    }
                case BaseType::IntPtr:
                    {
#if SLANG_PTR_IS_64
                        m_writer->emit("i64(");
                        m_writer->emitInt64(int64_t(litInst->value.intVal));
                        m_writer->emit(")");
#else
                        m_writer->emit("i32(");
                        m_writer->emit(int(litInst->value.intVal));
                        m_writer->emit(")");
#endif
                        break;
                    }
                case BaseType::UIntPtr:
                    {
#if SLANG_PTR_IS_64
                        m_writer->emit("u64(");
                        m_writer->emitUInt64(uint64_t(litInst->value.intVal));
                        m_writer->emit(")");
#else
                        m_writer->emit("u32(");
                        m_writer->emit(UInt(uint32_t(litInst->value.intVal)));
                        m_writer->emit(")");
#endif
                        break;
                    }
                }
            }
            else
            {
                // If no type... just output what we have
                m_writer->emit(litInst->value.intVal);
            }
            break;
        }

    case kIROp_FloatLit:
        {
            auto litInst = static_cast<IRConstant*>(inst);

            IRBasicType* type = as<IRBasicType>(inst->getDataType());
            if (type)
            {
                switch (type->getBaseType())
                {
                default:

                case BaseType::Half:
                    {
                        m_writer->emit(litInst->value.floatVal);
                        m_writer->emit("h");
                        m_f16ExtensionEnabled = true;
                    }
                    break;

                case BaseType::Float:
                    {
                        IRConstant::FloatKind kind = litInst->getFloatKind();
                        switch (kind)
                        {
                        case IRConstant::FloatKind::Nan:
                            {
                                ensurePrelude(kWGSLBuiltinPreludeGetNan);
                                m_writer->emit("_slang_getNan()");
                                break;
                            }
                        case IRConstant::FloatKind::PositiveInfinity:
                            {
                                ensurePrelude(kWGSLBuiltinPreludeGetInfinity);
                                m_writer->emit("_slang_getInfinity(true)");
                                break;
                            }
                        case IRConstant::FloatKind::NegativeInfinity:
                            {
                                ensurePrelude(kWGSLBuiltinPreludeGetInfinity);
                                m_writer->emit("_slang_getInfinity(false)");
                                break;
                            }
                        default:
                            m_writer->emit(litInst->value.floatVal);
                            m_writer->emit("f");
                            break;
                        }
                    }
                    break;

                case BaseType::Double:
                    {
                        // There is not "f64" in WGSL
                        SLANG_UNEXPECTED("'double' type emitted");
                    }
                    break;
                }
            }
            else
            {
                // If no type... just output what we have
                m_writer->emit(litInst->value.floatVal);
            }
        }
        break;

    case kIROp_BoolLit:
        {
            bool val = ((IRConstant*)inst)->value.intVal != 0;
            m_writer->emit(val ? "true" : "false");
        }
        break;

    default:
        SLANG_UNIMPLEMENTED_X("val case for emit");
        break;
    }
}

void WGSLSourceEmitter::emitParamTypeImpl(IRType* type, const String& name)
{
    emitType(type, name);
}

bool WGSLSourceEmitter::tryEmitInstStmtImpl(IRInst* inst)
{
    switch (inst->getOp())
    {
    default:
        return false;
    case kIROp_AtomicLoad:
        {
            emitInstResultDecl(inst);
            m_writer->emit("atomicLoad(&(");
            emitOperand(inst->getOperand(0), getInfo(EmitOp::General));
            m_writer->emit("));\n");
            return true;
        }
    case kIROp_AtomicStore:
        {
            m_writer->emit("atomicStore(&(");
            emitOperand(inst->getOperand(0), getInfo(EmitOp::General));
            m_writer->emit("), ");
            emitOperand(inst->getOperand(1), getInfo(EmitOp::General));
            m_writer->emit(");\n");
            return true;
        }
    case kIROp_AtomicExchange:
        {
            emitInstResultDecl(inst);
            m_writer->emit("atomicExchange(&(");
            emitOperand(inst->getOperand(0), getInfo(EmitOp::General));
            m_writer->emit("), ");
            emitOperand(inst->getOperand(1), getInfo(EmitOp::General));
            m_writer->emit(");\n");
            return true;
        }
    case kIROp_AtomicCompareExchange:
        {
            emitInstResultDecl(inst);
            m_writer->emit("atomicCompareExchangeWeak(&(");
            emitOperand(inst->getOperand(0), getInfo(EmitOp::General));
            m_writer->emit("), ");
            emitOperand(inst->getOperand(1), getInfo(EmitOp::General));
            m_writer->emit(", ");
            emitOperand(inst->getOperand(2), getInfo(EmitOp::General));
            m_writer->emit(").old_value;\n");
            return true;
        }
    case kIROp_AtomicAdd:
        {
            emitInstResultDecl(inst);
            m_writer->emit("atomicAdd(&(");
            emitOperand(inst->getOperand(0), getInfo(EmitOp::General));
            m_writer->emit("), ");
            emitOperand(inst->getOperand(1), getInfo(EmitOp::General));
            m_writer->emit(");\n");
            return true;
        }
    case kIROp_AtomicSub:
        {
            emitInstResultDecl(inst);
            m_writer->emit("atomicSub(&(");
            emitOperand(inst->getOperand(0), getInfo(EmitOp::General));
            m_writer->emit("), ");
            emitOperand(inst->getOperand(1), getInfo(EmitOp::General));
            m_writer->emit(");\n");
            return true;
        }
    case kIROp_AtomicAnd:
        {
            emitInstResultDecl(inst);
            m_writer->emit("atomicAnd(&(");
            emitOperand(inst->getOperand(0), getInfo(EmitOp::General));
            m_writer->emit("), ");
            emitOperand(inst->getOperand(1), getInfo(EmitOp::General));
            m_writer->emit(");\n");
            return true;
        }
    case kIROp_AtomicOr:
        {
            emitInstResultDecl(inst);
            m_writer->emit("atomicOr(&(");
            emitOperand(inst->getOperand(0), getInfo(EmitOp::General));
            m_writer->emit("), ");
            emitOperand(inst->getOperand(1), getInfo(EmitOp::General));
            m_writer->emit(");\n");
            return true;
        }
    case kIROp_AtomicXor:
        {
            emitInstResultDecl(inst);
            m_writer->emit("atomicXor(&(");
            emitOperand(inst->getOperand(0), getInfo(EmitOp::General));
            m_writer->emit("), ");
            emitOperand(inst->getOperand(1), getInfo(EmitOp::General));
            m_writer->emit(");\n");
            return true;
        }
    case kIROp_AtomicMin:
        {
            emitInstResultDecl(inst);
            m_writer->emit("atomicMin(&(");
            emitOperand(inst->getOperand(0), getInfo(EmitOp::General));
            m_writer->emit("), ");
            emitOperand(inst->getOperand(1), getInfo(EmitOp::General));
            m_writer->emit(");\n");
            return true;
        }
    case kIROp_AtomicMax:
        {
            emitInstResultDecl(inst);
            m_writer->emit("atomicMax(&(");
            emitOperand(inst->getOperand(0), getInfo(EmitOp::General));
            m_writer->emit("), ");
            emitOperand(inst->getOperand(1), getInfo(EmitOp::General));
            m_writer->emit(");\n");
            return true;
        }
    case kIROp_AtomicInc:
        {
            emitInstResultDecl(inst);
            m_writer->emit("atomicAdd(&(");
            emitOperand(inst->getOperand(0), getInfo(EmitOp::General));
            m_writer->emit("), ");
            emitType(inst->getDataType());
            m_writer->emit("(1));\n");
            return true;
        }
    case kIROp_AtomicDec:
        {
            emitInstResultDecl(inst);
            m_writer->emit("atomicSub(&(");
            emitOperand(inst->getOperand(0), getInfo(EmitOp::General));
            m_writer->emit("), ");
            emitType(inst->getDataType());
            m_writer->emit("(1));\n");
            return true;
        }
    case kIROp_StructuredBufferGetDimensions:
        {
            IRIntegerValue strideValue;
            auto dataType = inst->getOperand(0)->getDataType();
            auto structuredBufferType = as<IRHLSLStructuredBufferTypeBase>(dataType);
            if (structuredBufferType)
            {
                auto elementType = structuredBufferType->getElementType();
                auto sizeDecor = elementType->findDecoration<IRSizeAndAlignmentDecoration>();
                SLANG_ASSERT(sizeDecor);
                strideValue = align(sizeDecor->getSize(), (int)sizeDecor->getAlignment());
            }
            else
            {
                SLANG_ASSERT(as<IRByteAddressBufferTypeBase>(dataType));
                // ByteAddressBuffer(s) are an array of 32 bit integers, stride is 4 bytes.
                strideValue = 4;
            }

            emitInstResultDecl(inst);
            m_writer->emit("vec2<u32>(");
            m_writer->emit("arrayLength(&");
            emitOperand(inst->getOperand(0), getInfo(EmitOp::General));
            m_writer->emit(")");
            m_writer->emit(", ");
            m_writer->emit(strideValue);
            m_writer->emit(");\n");
            return true;
        }
    }
}

void WGSLSourceEmitter::emitCallArg(IRInst* inst)
{
    if (as<IRPtrTypeBase>(inst->getDataType()))
    {
        // If we are calling a function with a pointer-typed argument, we need to
        // explicitly prefix the argument with `&` to pass a pointer.
        //
        m_writer->emit("&(");
        emitOperand(inst, getInfo(EmitOp::General));
        m_writer->emit(")");
    }
    else
    {
        emitOperand(inst, getInfo(EmitOp::General));
    }
}

bool WGSLSourceEmitter::shouldFoldInstIntoUseSites(IRInst* inst)
{
    bool result = CLikeSourceEmitter::shouldFoldInstIntoUseSites(inst);
    if (result)
    {
        // If inst is a matrix, and is used in a component-wise multiply,
        // we need to not fold it.
        if (as<IRMatrixType>(inst->getDataType()))
        {
            for (auto use = inst->firstUse; use; use = use->nextUse)
            {
                auto user = use->getUser();
                if (user->getOp() == kIROp_Mul)
                {
                    if (as<IRMatrixType>(user->getOperand(0)->getDataType()) &&
                        as<IRMatrixType>(user->getOperand(1)->getDataType()))
                    {
                        return false;
                    }
                }
            }
        }
    }
    return result;
}


bool WGSLSourceEmitter::tryEmitInstExprImpl(IRInst* inst, const EmitOpInfo& inOuterPrec)
{
    EmitOpInfo outerPrec = inOuterPrec;

    switch (inst->getOp())
    {
    case kIROp_MakeVectorFromScalar:
        {
            // In WGSL this is done by calling the vec* overloads listed in [1]
            // [1] https://www.w3.org/TR/WGSL/#value-constructor-builtin-function
            emitType(inst->getDataType());
            m_writer->emit("(");
            auto prec = getInfo(EmitOp::Prefix);
            emitOperand(inst->getOperand(0), rightSide(outerPrec, prec));
            m_writer->emit(")");
            return true;
        }
        break;

    case kIROp_And:
    case kIROp_Or:
        {
            // WGSL doesn't have operator overloadings for `&&` and `||` when the operands are
            // non-scalar. Unlike HLSL, WGSL doesn't have `and()` and `or()`.
            auto vecType = as<IRVectorType>(inst->getDataType());
            if (!vecType)
                return false;

            // The function signature for `select` in WGSL is different from others:
            // @const @must_use fn select(f: T, t: T, cond: bool) -> T
            if (inst->getOp() == kIROp_And)
            {
                m_writer->emit("select(vec");
                m_writer->emit(getIntVal(vecType->getElementCount()));
                m_writer->emit("<bool>(false), ");
                emitOperand(inst->getOperand(1), getInfo(EmitOp::General));
                m_writer->emit(", ");
                emitOperand(inst->getOperand(0), getInfo(EmitOp::General));
                m_writer->emit(")");
            }
            else
            {
                m_writer->emit("select(");
                emitOperand(inst->getOperand(1), getInfo(EmitOp::General));
                m_writer->emit(", vec");
                m_writer->emit(getIntVal(vecType->getElementCount()));
                m_writer->emit("<bool>(true), ");
                emitOperand(inst->getOperand(0), getInfo(EmitOp::General));
                m_writer->emit(")");
            }
            return true;
        }

    case kIROp_BitCast:
        {
            // In WGSL there is a built-in bitcast function!
            // https://www.w3.org/TR/WGSL/#bitcast-builtin
            m_writer->emit("bitcast");
            m_writer->emit("<");
            emitType(inst->getDataType());
            m_writer->emit(">");
            m_writer->emit("(");
            emitOperand(inst->getOperand(0), getInfo(EmitOp::General));
            m_writer->emit(")");
            return true;
        }
        break;

    case kIROp_MakeArray:
    case kIROp_MakeStruct:
        {
            // It seems there are currently no designated initializers in WGSL.
            // Similarly for array initializers.
            // https://github.com/gpuweb/gpuweb/issues/4210

            // There is a constructor named like the struct/array type itself
            auto type = inst->getDataType();
            emitType(type);
            m_writer->emit("( ");
            UInt argCount = inst->getOperandCount();
            for (UInt aa = 0; aa < argCount; ++aa)
            {
                if (aa != 0)
                    m_writer->emit(", ");
                emitOperand(inst->getOperand(aa), getInfo(EmitOp::General));
            }
            m_writer->emit(" )");

            return true;
        }
        break;

    case kIROp_MakeArrayFromElement:
        {
            // It seems there are currently no array initializers in WGSL.

            // There is a constructor named like the array type itself
            auto type = inst->getDataType();
            emitType(type);
            m_writer->emit("(");
            UInt argCount =
                (UInt)cast<IRIntLit>(cast<IRArrayType>(inst->getDataType())->getElementCount())
                    ->getValue();
            for (UInt aa = 0; aa < argCount; ++aa)
            {
                if (aa != 0)
                    m_writer->emit(", ");
                emitOperand(inst->getOperand(0), getInfo(EmitOp::General));
            }
            m_writer->emit(")");
            return true;
        }
        break;

    case kIROp_StructuredBufferLoad:
    case kIROp_RWStructuredBufferLoad:
    case kIROp_RWStructuredBufferGetElementPtr:
        {
            emitOperand(inst->getOperand(0), leftSide(outerPrec, getInfo(EmitOp::Postfix)));
            m_writer->emit("[");
            emitOperand(inst->getOperand(1), getInfo(EmitOp::General));
            m_writer->emit("]");
            return true;
        }
        break;

    case kIROp_Rsh:
    case kIROp_Lsh:
        {
            // Shift amounts must be an unsigned type in WGSL.
            // We ensure this during legalization.
            // https://www.w3.org/TR/WGSL/#bit-expr
            SLANG_ASSERT(inst->getOperand(1)->getDataType()->getOp() != kIROp_IntType);

            // Dawn complains about mixing '<<' and '|', '^' and a bunch of other bit operators
            // without a paranthesis, so we'll always emit paranthesis around the shift amount.
            //

            m_writer->emit("(");

            const auto emitOp = getEmitOpForOp(inst->getOp());
            const auto info = getInfo(emitOp);

            const bool needClose = maybeEmitParens(outerPrec, info);
            emitOperand(inst->getOperand(0), leftSide(outerPrec, info));
            m_writer->emit(" ");
            m_writer->emit(info.op);
            m_writer->emit(" ");

            m_writer->emit("(");
            emitOperand(inst->getOperand(1), rightSide(outerPrec, info));
            m_writer->emit(")");

            maybeCloseParens(needClose);

            m_writer->emit(")");

            return true;
        }
    case kIROp_BitXor:
    case kIROp_BitOr:
    case kIROp_BitAnd:
        {
            // Emit bitwise operators with paranthesis to avoid precedence issues
            const auto emitOp = getEmitOpForOp(inst->getOp());
            const auto info = getInfo(emitOp);

            m_writer->emit("(");

            const bool needClose = maybeEmitParens(outerPrec, info);
            emitOperand(inst->getOperand(0), leftSide(outerPrec, info));
            m_writer->emit(" ");

            m_writer->emit(info.op);

            m_writer->emit(" (");
            emitOperand(inst->getOperand(1), rightSide(outerPrec, info));
            m_writer->emit(")");

            maybeCloseParens(needClose);

            m_writer->emit(")");
            return true;
        }
        break;

    case kIROp_ByteAddressBufferLoad:
        {
            // Indices in Slang code count bytes, but in WASM they count u32's since
            // byte address buffers translate to array<u32> in WASM, so divide by 4.
            emitOperand(inst->getOperand(0), getInfo(EmitOp::General));
            m_writer->emit("[(");
            emitOperand(inst->getOperand(1), getInfo(EmitOp::General));
            m_writer->emit(")/4]");
            return true;
        }
        break;

    case kIROp_ByteAddressBufferStore:
        {
            // Indices in Slang code count bytes, but in WASM they count u32's since
            // byte address buffers translate to array<u32> in WASM, so divide by 4.
            auto base = inst->getOperand(0);
            emitOperand(base, EmitOpInfo());
            m_writer->emit("[(");
            emitOperand(inst->getOperand(1), getInfo(EmitOp::General));
            m_writer->emit(")/4] = ");
            emitOperand(inst->getOperand(inst->getOperandCount() - 1), getInfo(EmitOp::General));
            return true;
        }
        break;

    case kIROp_GetStringHash:
        {
            auto getStringHashInst = as<IRGetStringHash>(inst);
            auto stringLit = getStringHashInst->getStringLit();

            if (stringLit)
            {
                auto slice = stringLit->getStringSlice();
                emitType(inst->getDataType());
                m_writer->emit("(");
                m_writer->emit((int)getStableHashCode32(slice.begin(), slice.getLength()).hash);
                m_writer->emit(")");
            }
            else
            {
                // Couldn't handle
                diagnoseUnhandledInst(inst);
            }
            return true;
        }

    case kIROp_Mul:
        {
            if (!as<IRMatrixType>(inst->getOperand(0)->getDataType()) ||
                !as<IRMatrixType>(inst->getOperand(1)->getDataType()))
            {
                return false;
            }
            // Mul(m1, m2) should be translated to component-wise multiplication in WGSL.
            auto matrixType = as<IRMatrixType>(inst->getDataType());
            auto rowCount = getIntVal(matrixType->getRowCount());
            emitType(inst->getDataType());
            m_writer->emit("(");
            for (IRIntegerValue i = 0; i < rowCount; i++)
            {
                if (i != 0)
                {
                    m_writer->emit(", ");
                }
                emitOperand(inst->getOperand(0), getInfo(EmitOp::Postfix));
                m_writer->emit("[");
                m_writer->emit(i);
                m_writer->emit("] * ");
                emitOperand(inst->getOperand(1), getInfo(EmitOp::Postfix));
                m_writer->emit("[");
                m_writer->emit(i);
                m_writer->emit("]");
            }
            m_writer->emit(")");

            return true;
        }

    case kIROp_Select:
        {
            m_writer->emit("select(");
            emitOperand(inst->getOperand(2), getInfo(EmitOp::General));
            m_writer->emit(", ");
            emitOperand(inst->getOperand(1), getInfo(EmitOp::General));
            m_writer->emit(", ");
            emitOperand(inst->getOperand(0), getInfo(EmitOp::General));
            m_writer->emit(")");
            return true;
        }
    }

    return false;
}

void WGSLSourceEmitter::emitVectorTypeNameImpl(IRType* elementType, IRIntegerValue elementCount)
{

    if (elementCount > 1)
    {
        m_writer->emit("vec");
        m_writer->emit(elementCount);
        m_writer->emit("<");
        emitSimpleType(elementType);
        m_writer->emit(">");
    }
    else
    {
        emitSimpleType(elementType);
    }
}

void WGSLSourceEmitter::emitFrontMatterImpl(TargetRequest* /* targetReq */)
{
    if (m_f16ExtensionEnabled)
    {
        m_writer->emit("enable f16;\n");
        m_writer->emit("\n");
    }

    StringBuilder builder;
    m_extensionTracker->appendExtensionRequireLinesForWGSL(builder);
    m_writer->emit(builder.getUnownedSlice());
}

void WGSLSourceEmitter::emitIntrinsicCallExprImpl(
    IRCall* inst,
    UnownedStringSlice intrinsicDefinition,
    IRInst* intrinsicInst,
    EmitOpInfo const& inOuterPrec)
{
    // The f16 constructor is generated for f32tof16
    if (intrinsicDefinition.startsWith("f16"))
    {
        m_f16ExtensionEnabled = true;
    }

    CLikeSourceEmitter::emitIntrinsicCallExprImpl(
        inst,
        intrinsicDefinition,
        intrinsicInst,
        inOuterPrec);
}

void WGSLSourceEmitter::emitInterpolationModifiersImpl(
    IRInst* varInst,
    IRType* /* valueType */,
    IRVarLayout* /* layout */)
{
    char const* interpolationType = nullptr;
    char const* interpolationSampling = nullptr;
    for (auto dd : varInst->getDecorations())
    {
        if (dd->getOp() != kIROp_InterpolationModeDecoration)
            continue;
        auto decoration = (IRInterpolationModeDecoration*)dd;
        IRInterpolationMode mode = decoration->getMode();
        switch (mode)
        {
        case IRInterpolationMode::NoInterpolation:
            interpolationType = "flat";
            break;
        case IRInterpolationMode::NoPerspective:
        case IRInterpolationMode::Linear:
            interpolationType = "linear";
            break;
        case IRInterpolationMode::Sample:
            interpolationSampling = "sample";
            break;
        case IRInterpolationMode::Centroid:
            interpolationSampling = "centroid";
            break;
        }
    }

    if (interpolationType)
    {
        m_writer->emit("@interpolate(");
        m_writer->emit(interpolationType);
        if (interpolationSampling)
        {
            m_writer->emit(", ");
            m_writer->emit(interpolationSampling);
        }
        m_writer->emit(") ");
    }

    // TODO: Check the following:
    // "User-defined vertex outputs and fragment inputs of scalar or vector
    //  integer type must always be specified with interpolation type flat."
    // https://www.w3.org/TR/WGSL/#interpolation
}

void WGSLSourceEmitter::_requireExtension(const UnownedStringSlice& name)
{
    m_extensionTracker->requireExtension(name);
}

void WGSLSourceEmitter::handleRequiredCapabilitiesImpl(IRInst* inst)
{
    for (auto decoration : inst->getDecorations())
    {
        if (const auto extensionDecoration = as<IRRequireWGSLExtensionDecoration>(decoration))
        {
            _requireExtension(extensionDecoration->getExtensionName());

            // TODO: Make this cleaner and only enable this extension if f16 is actually used on the
            // subgroup intrinsic. Check float type in meta file.
            if (m_f16ExtensionEnabled && extensionDecoration->getExtensionName() == "subgroups")
            {
                String extName = "subgroups_f16";
                _requireExtension(extName.getUnownedSlice());
            }
        }
    }
}

void WGSLSourceEmitter::emitRequireExtension(IRRequireTargetExtension* inst)
{
    _requireExtension(inst->getExtensionName());
}

} // namespace Slang
