// slang-emit-c-like.cpp
#include "slang-emit-c-like.h"

#include "../compiler-core/slang-name.h"
#include "../core/slang-stable-hash.h"
#include "../core/slang-writer.h"
#include "slang-emit-source-writer.h"
#include "slang-intrinsic-expand.h"
#include "slang-ir-bind-existentials.h"
#include "slang-ir-dce.h"
#include "slang-ir-entry-point-uniforms.h"
#include "slang-ir-glsl-legalize.h"
#include "slang-ir-link.h"
#include "slang-ir-restructure-scoping.h"
#include "slang-ir-specialize-resources.h"
#include "slang-ir-specialize.h"
#include "slang-ir-ssa.h"
#include "slang-ir-util.h"
#include "slang-ir-validate.h"
#include "slang-legalize-types.h"
#include "slang-lower-to-ir.h"
#include "slang-mangle.h"
#include "slang-mangled-lexer.h"
#include "slang-syntax.h"
#include "slang-type-layout.h"
#include "slang-visitor.h"
#include "slang/slang-ir.h"

#include <assert.h>

namespace Slang
{

bool isCPUTarget(TargetRequest* targetReq);
bool isCUDATarget(TargetRequest* targetReq);

struct CLikeSourceEmitter::ComputeEmitActionsContext
{
    IRInst* moduleInst;
    InstHashSet openInsts;
    Dictionary<IRInst*, EmitAction::Level> mapInstToLevel;
    List<EmitAction>* actions;
};

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!! CLikeSourceEmitter !!!!!!!!!!!!!!!!!!!!!!!!!! */

/* static */ SourceLanguage CLikeSourceEmitter::getSourceLanguage(CodeGenTarget target)
{
    switch (target)
    {
    default:
    case CodeGenTarget::Unknown:
    case CodeGenTarget::None:
        {
            return SourceLanguage::Unknown;
        }
    case CodeGenTarget::GLSL:
        {
            return SourceLanguage::GLSL;
        }
    case CodeGenTarget::HLSL:
        {
            return SourceLanguage::HLSL;
        }
    case CodeGenTarget::PTX:
    case CodeGenTarget::SPIRV:
    case CodeGenTarget::SPIRVAssembly:
    case CodeGenTarget::DXBytecode:
    case CodeGenTarget::DXBytecodeAssembly:
    case CodeGenTarget::DXIL:
    case CodeGenTarget::DXILAssembly:
    case CodeGenTarget::MetalLib:
    case CodeGenTarget::MetalLibAssembly:
        {
            return SourceLanguage::Unknown;
        }
    case CodeGenTarget::CSource:
        {
            return SourceLanguage::C;
        }
    case CodeGenTarget::CPPSource:
    case CodeGenTarget::HostCPPSource:
    case CodeGenTarget::PyTorchCppBinding:
        {
            return SourceLanguage::CPP;
        }
    case CodeGenTarget::CUDASource:
        {
            return SourceLanguage::CUDA;
        }
    case CodeGenTarget::Metal:
        {
            return SourceLanguage::Metal;
        }
    case CodeGenTarget::WGSL:
        {
            return SourceLanguage::WGSL;
        }
    }
}

CLikeSourceEmitter::CLikeSourceEmitter(const Desc& desc)
{
    m_writer = desc.sourceWriter;
    m_sourceLanguage = getSourceLanguage(desc.codeGenContext->getTargetFormat());
    SLANG_ASSERT(m_sourceLanguage != SourceLanguage::Unknown);

    m_target = desc.codeGenContext->getTargetFormat();
    m_codeGenContext = desc.codeGenContext;
    m_entryPointStage = desc.entryPointStage;
    m_effectiveProfile = desc.effectiveProfile;
}

SlangResult CLikeSourceEmitter::init()
{
    return SLANG_OK;
}

void CLikeSourceEmitter::emitFrontMatterImpl(TargetRequest* targetReq)
{
    SLANG_UNUSED(targetReq);
}

void CLikeSourceEmitter::emitPreModuleImpl()
{
    for (auto prelude : m_requiredPreludes)
    {
        m_writer->emit(prelude->getStringSlice());
        m_writer->emit("\n");
    }
}

//
// Types
//

void CLikeSourceEmitter::ensureTypePrelude(IRType* type)
{
    if (auto requirePreludeDecor = as<IRRequirePreludeDecoration>(
            findBestTargetDecoration<IRRequirePreludeDecoration>(type)))
    {
        auto preludeTextInst = as<IRStringLit>(requirePreludeDecor->getOperand(1));
        if (preludeTextInst)
            m_requiredPreludes.add(preludeTextInst);
    }
}

void CLikeSourceEmitter::emitDeclaratorImpl(DeclaratorInfo* declarator)
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
            auto arrayDeclarator = (SizedArrayDeclaratorInfo*)declarator;
            emitDeclarator(arrayDeclarator->next);
            m_writer->emit("[");
            if (auto elementCount = arrayDeclarator->elementCount)
            {
                emitVal(elementCount, getInfo(EmitOp::General));
            }
            m_writer->emit("]");
        }
        break;

    case DeclaratorInfo::Flavor::UnsizedArray:
        {
            auto arrayDeclarator = (UnsizedArrayDeclaratorInfo*)declarator;
            emitDeclarator(arrayDeclarator->next);
            m_writer->emit("[]");
        }
        break;

    case DeclaratorInfo::Flavor::Ptr:
        {
            // TODO: When there are both pointer and array declarators
            // as part of a type, paranetheses may be needed in order
            // to disambiguate between a pointer-to-array and an
            // array-of-poiners.
            //
            auto ptrDeclarator = (PtrDeclaratorInfo*)declarator;
            m_writer->emit("*");
            emitDeclarator(ptrDeclarator->next);
        }
        break;

    case DeclaratorInfo::Flavor::Ref:
        {
            auto refDeclarator = (RefDeclaratorInfo*)declarator;
            m_writer->emit("&");
            emitDeclarator(refDeclarator->next);
        }
        break;

    case DeclaratorInfo::Flavor::LiteralSizedArray:
        {
            auto arrayDeclarator = (LiteralSizedArrayDeclaratorInfo*)declarator;
            emitDeclarator(arrayDeclarator->next);
            m_writer->emit("[");
            m_writer->emit(arrayDeclarator->elementCount);
            m_writer->emit("]");
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

void CLikeSourceEmitter::emitSimpleType(IRType* type)
{
    emitSimpleTypeImpl(type);
}

/* static */ UnownedStringSlice CLikeSourceEmitter::getDefaultBuiltinTypeName(IROp op)
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
        return UnownedStringSlice("int");
    case kIROp_Int64Type:
        return UnownedStringSlice("int64_t");
    case kIROp_IntPtrType:
        return UnownedStringSlice("intptr_t");

    case kIROp_UInt8Type:
        return UnownedStringSlice("uint8_t");
    case kIROp_UInt16Type:
        return UnownedStringSlice("uint16_t");
    case kIROp_UIntType:
        return UnownedStringSlice("uint");
    case kIROp_UInt64Type:
        return UnownedStringSlice("uint64_t");
    case kIROp_UIntPtrType:
        return UnownedStringSlice("uintptr_t");

    case kIROp_HalfType:
        return UnownedStringSlice("half");

    case kIROp_FloatType:
        return UnownedStringSlice("float");
    case kIROp_DoubleType:
        return UnownedStringSlice("double");

    case kIROp_CharType:
        return UnownedStringSlice("uint8_t");
    default:
        return UnownedStringSlice();
    }
}


IRNumThreadsDecoration* CLikeSourceEmitter::getComputeThreadGroupSize(
    IRFunc* func,
    Int outNumThreads[kThreadGroupAxisCount])
{
    Int specializationConstantIds[kThreadGroupAxisCount];
    IRNumThreadsDecoration* decor =
        getComputeThreadGroupSize(func, outNumThreads, specializationConstantIds);

    for (auto id : specializationConstantIds)
    {
        if (id >= 0)
        {
            getSink()->diagnose(decor, Diagnostics::unsupportedSpecializationConstantForNumThreads);
            break;
        }
    }
    return decor;
}

/* static */ IRNumThreadsDecoration* CLikeSourceEmitter::getComputeThreadGroupSize(
    IRFunc* func,
    Int outNumThreads[kThreadGroupAxisCount],
    Int outSpecializationConstantIds[kThreadGroupAxisCount])
{
    IRNumThreadsDecoration* decor = func->findDecoration<IRNumThreadsDecoration>();
    for (int i = 0; i < kThreadGroupAxisCount; ++i)
    {
        if (!decor)
        {
            outNumThreads[i] = 1;
            outSpecializationConstantIds[i] = -1;
        }
        else if (auto specConst = as<IRGlobalParam>(decor->getOperand(i)))
        {
            outNumThreads[i] = 1;
            outSpecializationConstantIds[i] = getSpecializationConstantId(specConst);
        }
        else
        {
            outNumThreads[i] = Int(getIntVal(decor->getOperand(i)));
            outSpecializationConstantIds[i] = -1;
        }
    }
    return decor;
}

/* static */ IRWaveSizeDecoration* CLikeSourceEmitter::getComputeWaveSize(
    IRFunc* func,
    Int* outWaveSize)
{
    IRWaveSizeDecoration* decor = func->findDecoration<IRWaveSizeDecoration>();
    if (decor)
    {
        *outWaveSize = Int(getIntVal(decor->getOperand(0)));
    }
    return decor;
}

String CLikeSourceEmitter::getTargetBuiltinVarName(IRInst* inst, IRTargetBuiltinVarName builtinName)
{
    switch (builtinName)
    {
    case IRTargetBuiltinVarName::SpvInstanceIndex:
        return "gl_InstanceIndex";
    case IRTargetBuiltinVarName::SpvBaseInstance:
        return "gl_BaseInstance";
    }
    if (auto linkage = inst->findDecoration<IRLinkageDecoration>())
        return linkage->getMangledName();
    return generateName(inst);
}

List<IRWitnessTableEntry*> CLikeSourceEmitter::getSortedWitnessTableEntries(
    IRWitnessTable* witnessTable)
{
    List<IRWitnessTableEntry*> sortedWitnessTableEntries;
    auto interfaceType = cast<IRInterfaceType>(witnessTable->getConformanceType());
    auto witnessTableItems = witnessTable->getChildren();
    // Build a dictionary of witness table entries for fast lookup.
    Dictionary<IRInst*, IRWitnessTableEntry*> witnessTableEntryDictionary;
    for (auto item : witnessTableItems)
    {
        if (auto entry = as<IRWitnessTableEntry>(item))
        {
            witnessTableEntryDictionary[entry->getRequirementKey()] = entry;
        }
    }
    // Get a sorted list of entries using RequirementKeys defined in `interfaceType`.
    for (UInt i = 0; i < interfaceType->getOperandCount(); i++)
    {
        auto reqEntry = cast<IRInterfaceRequirementEntry>(interfaceType->getOperand(i));
        IRWitnessTableEntry* entry = nullptr;
        if (witnessTableEntryDictionary.tryGetValue(reqEntry->getRequirementKey(), entry))
        {
            sortedWitnessTableEntries.add(entry);
        }
        else
        {
            SLANG_UNREACHABLE("interface requirement key not found in witness table.");
        }
    }
    return sortedWitnessTableEntries;
}

void CLikeSourceEmitter::_emitPrefixTypeAttr(IRAttr* attr)
{
    SLANG_UNUSED(attr);

    // By defualt we will not emit any attributes.
    //
    // TODO: If `const` ever surfaces as a type attribute in our IR,
    // we may need to handle it here.
}

void CLikeSourceEmitter::_emitPostfixTypeAttr(IRAttr* attr)
{
    SLANG_UNUSED(attr);

    // By defualt we will not emit any attributes.
    //
    // TODO: If `const` ever surfaces as a type attribute in our IR,
    // we may need to handle it here.
}

void CLikeSourceEmitter::emitSimpleTypeAndDeclaratorImpl(IRType* type, DeclaratorInfo* declarator)
{
    emitSimpleType(type);
    emitDeclarator(declarator);
}

void CLikeSourceEmitter::_emitType(IRType* type, DeclaratorInfo* declarator)
{
    switch (type->getOp())
    {
    default:
        emitSimpleTypeAndDeclarator(type, declarator);
        break;

    case kIROp_RateQualifiedType:
        {
            auto rateQualifiedType = cast<IRRateQualifiedType>(type);
            _emitType(rateQualifiedType->getValueType(), declarator);
        }
        break;
    case kIROp_DescriptorHandleType:
        {
            // If the T is already bindless for target, emit it directly.
            auto resPtrType = cast<IRDescriptorHandleType>(type);
            if (isResourceTypeBindless(resPtrType->getResourceType()))
                _emitType(resPtrType->getResourceType(), declarator);
            else
            {
                // Otherwise, emit the DescriptorHandle<T> as uint2.
                IRBuilder builder(resPtrType);
                builder.setInsertBefore(resPtrType);
                emitSimpleTypeAndDeclarator(
                    builder.getVectorType(builder.getUIntType(), 2),
                    declarator);
            }
        }
        break;

    case kIROp_ArrayType:
        {
            auto arrayType = cast<IRArrayType>(type);
            SizedArrayDeclaratorInfo arrayDeclarator(declarator, arrayType->getElementCount());
            _emitType(arrayType->getElementType(), &arrayDeclarator);
        }
        break;

    case kIROp_UnsizedArrayType:
        {
            auto arrayType = cast<IRUnsizedArrayType>(type);
            UnsizedArrayDeclaratorInfo arrayDeclarator(declarator);
            _emitType(arrayType->getElementType(), &arrayDeclarator);
        }
        break;

    case kIROp_AttributedType:
        {
            auto attributedType = cast<IRAttributedType>(type);
            for (auto attr : attributedType->getAllAttrs())
            {
                _emitPrefixTypeAttr(attr);
            }
            AttributedDeclaratorInfo attributedDeclarator(declarator, attributedType);
            _emitType(attributedType->getBaseType(), &attributedDeclarator);
        }
        break;
    }
}

void CLikeSourceEmitter::_emitSwizzleStorePerElement(IRInst* inst)
{
    auto subscriptOuter = getInfo(EmitOp::General);
    auto subscriptPrec = getInfo(EmitOp::Postfix);

    auto ii = cast<IRSwizzledStore>(inst);

    UInt elementCount = ii->getElementCount();
    UInt dstIndex = 0;
    for (UInt ee = 0; ee < elementCount; ++ee)
    {
        bool needCloseSubscript = maybeEmitParens(subscriptOuter, subscriptPrec);

        emitDereferenceOperand(ii->getDest(), leftSide(subscriptOuter, subscriptPrec));
        m_writer->emit(".");

        IRInst* irElementIndex = ii->getElementIndex(ee);
        SLANG_RELEASE_ASSERT(irElementIndex->getOp() == kIROp_IntLit);

        IRConstant* irConst = (IRConstant*)irElementIndex;

        UInt elementIndex = (UInt)irConst->value.intVal;
        SLANG_RELEASE_ASSERT(elementIndex < 4);

        char const* kComponents[] = {"x", "y", "z", "w"};
        m_writer->emit(kComponents[elementIndex]);

        maybeCloseParens(needCloseSubscript);

        m_writer->emit(" = ");
        emitOperand(ii->getSource(), getInfo(EmitOp::General));
        m_writer->emit(".");
        m_writer->emit(kComponents[dstIndex++]);
        m_writer->emit(";\n");
    }
}

void CLikeSourceEmitter::emitWitnessTable(IRWitnessTable* witnessTable)
{
    SLANG_UNUSED(witnessTable);
}

void CLikeSourceEmitter::emitComWitnessTable(IRWitnessTable* witnessTable)
{
    auto classType = witnessTable->getConcreteType();
    for (auto ent : witnessTable->getEntries())
    {
        auto req = ent->getRequirementKey();
        auto func = as<IRFunc>(ent->getSatisfyingVal());
        if (!func)
            continue;

        auto resultType = func->getResultType();

        auto name = getName(classType) + "::" + getName(req);

        emitFuncDecorations(func);

        emitType(resultType, name);
        m_writer->emit("(");
        // Skip declaration of `this` parameter.
        auto firstParam = func->getFirstParam()->getNextParam();
        for (auto pp = firstParam; pp; pp = pp->getNextParam())
        {
            if (pp != firstParam)
                m_writer->emit(", ");

            emitSimpleFuncParamImpl(pp);
        }
        m_writer->emit(")");
        m_writer->emit("\n{\n");
        m_writer->indent();

        // emit definition for `this` param.
        m_writer->emit("auto ");
        m_writer->emit(getName(func->getFirstParam()));
        m_writer->emit(" = this;\n");

        // Need to emit the operations in the blocks of the function
        emitFunctionBody(func);

        m_writer->dedent();
        m_writer->emit("}\n\n");
    }
}

void CLikeSourceEmitter::emitInterface(IRInterfaceType* interfaceType)
{
    SLANG_UNUSED(interfaceType);
    // By default, don't emit anything for interface types.
    // This behavior is overloaded by concrete emitters.
}

void CLikeSourceEmitter::emitRTTIObject(IRRTTIObject* rttiObject)
{
    SLANG_UNUSED(rttiObject);
    // Ignore rtti object by default.
    // This is only used in targets that support dynamic dispatching.
}

void CLikeSourceEmitter::defaultEmitInstStmt(IRInst* inst)
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
            m_writer->emit(".GetDimensions(");
            m_writer->emit(count);
            m_writer->emit(", ");
            m_writer->emit(stride);
            m_writer->emit(");\n");
            emitInstResultDecl(inst);
            m_writer->emit("uint2(");
            m_writer->emit(count);
            m_writer->emit(", ");
            m_writer->emit(stride);
            m_writer->emit(");\n");
        }
        break;
    case kIROp_discard:
        m_writer->emit("discard;\n");
        break;
    default:
        diagnoseUnhandledInst(inst);
    }
}


void CLikeSourceEmitter::emitTypeImpl(IRType* type, const StringSliceLoc* nameAndLoc)
{
    if (nameAndLoc)
    {
        // We advance here, such that if there is a #line directive to output it will
        // be done so before the type name appears.
        m_writer->advanceToSourceLocationIfValid(nameAndLoc->loc);

        NameDeclaratorInfo nameDeclarator(nameAndLoc);
        _emitType(type, &nameDeclarator);
    }
    else
    {
        _emitType(type, nullptr);
    }
}

void CLikeSourceEmitter::emitType(IRType* type, Name* name)
{
    SLANG_ASSERT(name);
    StringSliceLoc nameAndLoc(name->text.getUnownedSlice());
    emitType(type, &nameAndLoc);
}

void CLikeSourceEmitter::emitType(IRType* type, const String& name)
{
    StringSliceLoc nameAndLoc(name.getUnownedSlice());
    emitType(type, &nameAndLoc);
}

void CLikeSourceEmitter::emitType(IRType* type)
{
    emitType(type, (StringSliceLoc*)nullptr);
}

void CLikeSourceEmitter::emitType(IRType* type, Name* name, SourceLoc const& nameLoc)
{
    SLANG_ASSERT(name);

    StringSliceLoc nameAndLoc;
    nameAndLoc.loc = nameLoc;
    nameAndLoc.name = name->text.getUnownedSlice();

    emitType(type, &nameAndLoc);
}

void CLikeSourceEmitter::emitType(IRType* type, NameLoc const& nameAndLoc)
{
    emitType(type, nameAndLoc.name, nameAndLoc.loc);
}


void CLikeSourceEmitter::emitLivenessImpl(IRInst* inst)
{

    auto liveMarker = as<IRLiveRangeMarker>(inst);
    if (!liveMarker)
    {
        return;
    }

    IRInst* referenced = liveMarker->getReferenced();
    SLANG_ASSERT(referenced);

    UnownedStringSlice text;
    switch (inst->getOp())
    {
    case kIROp_LiveRangeStart:
        {
            text = UnownedStringSlice::fromLiteral("SLANG_LIVE_START");
            break;
        }
    case kIROp_LiveRangeEnd:
        {
            text = UnownedStringSlice::fromLiteral("SLANG_LIVE_END");
            break;
        }
    default:
        break;
    }

    m_writer->emit(text);
    m_writer->emit("(");

    emitOperand(referenced, getInfo(EmitOp::General));

    m_writer->emit(")\n");
}

//
// Expressions
//

static bool isBitLogicalOrRelationalOrEquality(EPrecedence prec)
{
    switch (prec)
    {
    case EPrecedence::kEPrecedence_And_Left:
    case EPrecedence::kEPrecedence_And_Right:
    case EPrecedence::kEPrecedence_BitAnd_Left:
    case EPrecedence::kEPrecedence_BitAnd_Right:
    case EPrecedence::kEPrecedence_BitOr_Left:
    case EPrecedence::kEPrecedence_BitOr_Right:
    case EPrecedence::kEPrecedence_BitXor_Left:
    case EPrecedence::kEPrecedence_BitXor_Right:
    case EPrecedence::kEPrecedence_Or_Left:
    case EPrecedence::kEPrecedence_Or_Right:
    case EPrecedence::kEPrecedence_Relational_Left:
    case EPrecedence::kEPrecedence_Relational_Right:
    case EPrecedence::kEPrecedence_Shift_Left:
    case EPrecedence::kEPrecedence_Shift_Right:
    case EPrecedence::kEPrecedence_Equality_Left:
    case EPrecedence::kEPrecedence_Equality_Right:
        return true;

    default:
        break;
    }

    return false;
}

bool CLikeSourceEmitter::maybeEmitParens(EmitOpInfo& outerPrec, const EmitOpInfo& prec)
{
    bool needParens = (prec.leftPrecedence <= outerPrec.leftPrecedence) ||
                      (prec.rightPrecedence <= outerPrec.rightPrecedence);

    // While Slang correctly removes some of parentheses, many compilers print warnings
    // for common mistakes when parentheses are not used with certain combinations
    // of the operations. We emit parentheses to avoid the warnings.
    //

    if (isBitLogicalOrRelationalOrEquality(prec.leftPrecedence) &&
        (outerPrec.leftPrecedence > kEPrecedence_Assign_Left))
        needParens = true;
    if (isBitLogicalOrRelationalOrEquality(outerPrec.leftPrecedence) ||
        isBitLogicalOrRelationalOrEquality(outerPrec.rightPrecedence))
        needParens = true;

    if (needParens)
    {
        m_writer->emit("(");

        outerPrec = getInfo(EmitOp::None);
    }
    return needParens;
}

void CLikeSourceEmitter::maybeCloseParens(bool needClose)
{
    if (needClose)
        m_writer->emit(")");
}

void CLikeSourceEmitter::emitStringLiteral(String const& value)
{
    m_writer->emit("\"");
    for (auto c : value)
    {
        // TODO: This needs a more complete implementation,
        // especially if we want to support Unicode.

        char buffer[] = {c, 0};
        switch (c)
        {
        default:
            m_writer->emit(buffer);
            break;

        case '\"':
            m_writer->emit("\\\"");
            break;
        case '\'':
            m_writer->emit("\\\'");
            break;
        case '\\':
            m_writer->emit("\\\\");
            break;
        case '\n':
            m_writer->emit("\\n");
            break;
        case '\r':
            m_writer->emit("\\r");
            break;
        case '\t':
            m_writer->emit("\\t");
            break;
        }
    }
    m_writer->emit("\"");
}

void CLikeSourceEmitter::emitVal(IRInst* val, EmitOpInfo const& outerPrec)
{
    if (auto type = as<IRType>(val))
    {
        emitType(type);
    }
    else
    {
        emitInstExpr(val, outerPrec);
    }
}

UInt CLikeSourceEmitter::getBindingOffsetForKinds(
    EmitVarChain* chain,
    LayoutResourceKindFlags kindFlags)
{
    UInt offset = 0;
    for (auto cc = chain; cc; cc = cc->next)
    {
        for (auto offsetAttr : cc->varLayout->getOffsetAttrs())
        {
            // Accumulate offset for all matching kind
            if (LayoutResourceKindFlag::make(offsetAttr->getResourceKind()) & kindFlags)
            {
                offset += offsetAttr->getOffset();
            }
        }
    }

    return offset;
}

Index findRegisterSpaceResourceInfo(IRVarLayout* layout);

UInt CLikeSourceEmitter::getBindingSpaceForKinds(
    EmitVarChain* chain,
    LayoutResourceKindFlags kindFlags)
{
    UInt space = 0;

    bool useSubElementSpace = false;

    for (auto cc = chain; cc; cc = cc->next)
    {
        auto varLayout = cc->varLayout;

        for (auto offsetAttr : cc->varLayout->getOffsetAttrs())
        {
            // Accumulate offset for all matching kinds
            if (LayoutResourceKindFlag::make(offsetAttr->getResourceKind()) & kindFlags)
            {
                space += offsetAttr->getSpace();
            }
        }
        if (!useSubElementSpace)
        {
            auto spaceOffset = findRegisterSpaceResourceInfo(varLayout);
            if (spaceOffset != -1)
            {
                space += spaceOffset;
                useSubElementSpace = true;
            }
        }
        else
        {
            if (auto resInfo =
                    varLayout->findOffsetAttr(LayoutResourceKind::SubElementRegisterSpace))
            {
                space += resInfo->getOffset();
            }
        }
    }
    return space;
}

UInt CLikeSourceEmitter::getBindingOffset(EmitVarChain* chain, LayoutResourceKind kind)
{
    UInt offset = 0;

    for (auto cc = chain; cc; cc = cc->next)
    {
        if (auto resInfo = cc->varLayout->findOffsetAttr(kind))
        {
            offset += resInfo->getOffset();
        }
    }
    return offset;
}

UInt CLikeSourceEmitter::getBindingSpace(EmitVarChain* chain, LayoutResourceKind kind)
{
    UInt space = 0;
    bool useSubElementSpace = false;
    for (auto cc = chain; cc; cc = cc->next)
    {
        auto varLayout = cc->varLayout;
        if (auto resInfo = varLayout->findOffsetAttr(kind))
        {
            space += resInfo->getSpace();
        }
        if (!useSubElementSpace)
        {
            auto spaceOffset = findRegisterSpaceResourceInfo(varLayout);
            if (spaceOffset != -1)
            {
                space += spaceOffset;
                useSubElementSpace = true;
            }
        }
        else
        {
            if (auto resInfo =
                    varLayout->findOffsetAttr(LayoutResourceKind::SubElementRegisterSpace))
            {
                space += resInfo->getOffset();
            }
        }
    }
    return space;
}

UInt CLikeSourceEmitter::allocateUniqueID()
{
    return m_uniqueIDCounter++;
}

// IR-level emit logic

UInt CLikeSourceEmitter::getID(IRInst* value)
{
    auto& mapIRValueToID = m_mapIRValueToID;

    UInt id = 0;
    if (mapIRValueToID.tryGetValue(value, id))
        return id;

    id = allocateUniqueID();
    mapIRValueToID.add(value, id);
    return id;
}

void CLikeSourceEmitter::appendScrubbedName(const UnownedStringSlice& name, StringBuilder& out)
{
    // We will use a plain `U` as a dummy character to insert
    // whenever we need to insert things to make a string into
    // valid name.
    //
    const char dummyChar = 'U';

    // Special case a name that is the empty string, just in case.
    if (name.getLength() == 0)
    {
        out.appendChar(dummyChar);
        return;
    }

    // Otherwise, we are going to walk over the name byte by byte
    // and write some legal characters to the output as we go.

    if (getSourceLanguage() == SourceLanguage::GLSL)
    {
        // GLSL reserves all names that start with `gl_`,
        // so if we are in danger of collision, then make
        // our name start with a dummy character instead.
        if (name.startsWith("gl_"))
        {
            out.appendChar(dummyChar);
        }
    }

    // We will also detect user-defined names that
    // might overlap with our convention for mangled names,
    // to avoid an possible collision.
    if (name.startsWith("_S"))
    {
        out.appendChar(dummyChar);
    }

    // TODO: This is where we might want to consult
    // a dictionary of reserved words for the chosen target
    //
    //  if(isReservedWord(name)) { sb.Append(dummyChar); }
    //

    // We need to track the previous byte in
    // order to detect consecutive underscores for GLSL.
    int prevChar = -1;

    for (auto c : name)
    {
        // We will treat a dot character or any path separator
        // just like an underscore for the purposes of producing
        // a scrubbed name, so that we translate `SomeType.someMethod`
        // into `SomeType_someMethod`. This increases the readability
        // of output code when the input used lots of nesting of
        // code under types/namespaces/etc.
        //
        // By handling this case at the top of this loop, we
        // ensure that a `.`-turned-`_` is handled just like
        // a `_` in the original name, and will be properly
        // scrubbed for GLSL output.
        //
        switch (c)
        {
        default:
            break;

        case '.':
        case '\\':
        case '/':
            c = '_';
            break;
        }

        if (((c >= 'a') && (c <= 'z')) || ((c >= 'A') && (c <= 'Z')))
        {
            // Ordinary ASCII alphabetic characters are assumed
            // to always be okay.
        }
        else if ((c >= '0') && (c <= '9'))
        {
            // We don't want to allow a digit as the first
            // byte in a name, since the result wouldn't
            // be a valid identifier in many target languages.
            if (prevChar == -1)
            {
                out.appendChar(dummyChar);
            }
        }
        else if (c == '_')
        {
            // We will collapse any consecutive sequence of `_`
            // characters into a single one (this means that
            // some names that were unique in the original
            // code might not resolve to unique names after
            // scrubbing, but that was true in general).

            if (prevChar == '_')
            {
                // Skip this underscore, so we don't output
                // more than one in a row.
                continue;
            }
        }
        else
        {
            // If we run into a character that wouldn't normally
            // be allowed in an identifier, we need to translate
            // it into something that *is* valid.
            //
            // Our solution for now will be very clumsy: we will
            // emit `x` and then the hexadecimal version of
            // the byte we were given.
            out.appendChar('x');
            out.append(uint32_t((unsigned char)c), 16);

            // We don't want to apply the default handling below,
            // so skip to the top of the loop now.
            prevChar = c;
            continue;
        }

        out.appendChar(c);
        prevChar = c;
    }

    if (getSourceLanguage() == SourceLanguage::GLSL)
    {
        // It looks like the default glslang name limit is 1024, but let's go a little less so there
        // is some wiggle room
        const Index maxTokenLength = 1024 - 8;

        const Index length = out.getLength();

        if (length > maxTokenLength)
        {
            // We are going to output with a prefix and a hash of the full name
            const auto hash = getStableHashCode64(out.getBuffer(), length);
            // Two hex chars per byte
            const Index hashSize = sizeof(hash) * 2;

            // Work out a size that is within range taking into account the hash size and extra
            // chars
            Index reducedBaseLength = maxTokenLength - hashSize - 1;
            // If it has a trailing _ remove it.
            // We know because of scrubbing there can only be single _
            reducedBaseLength -= Index(out[reducedBaseLength - 1] == '_');

            // Reduce the length
            out.reduceLength(reducedBaseLength);
            // Let's add a _ to separate from the rest of the name
            out.appendChar('_');
            // Append the hash in hex
            out.append(hash);

            SLANG_ASSERT(out.getLength() <= maxTokenLength);
        }
    }
}

String CLikeSourceEmitter::generateEntryPointNameImpl(IREntryPointDecoration* entryPointDecor)
{
    return entryPointDecor->getName()->getStringSlice();
}

String CLikeSourceEmitter::_generateUniqueName(const UnownedStringSlice& name)
{
    //
    // We need to be careful that the name follows the rules of the target language,
    // so there is a "scrubbing" step that needs to be applied here.
    //
    // We also need to make sure that the name won't collide with other declarations
    // that might have the same name hint applied, so we will still unique
    // them by appending the numeric ID of the instruction.
    //
    // TODO: Find cases where we can drop the suffix safely.
    //
    // TODO: When we start having to handle symbols with external linkage for
    // things like DXIL libraries, we will need to *not* use the friendly
    // names for stuff that should be link-able.
    //
    // The name we output will basically be:
    //
    //      <name>_<uniqueID>
    //
    // Except that we will "scrub" the name first,
    // and we will omit the underscore if the (scrubbed)
    // name hint already ends with one.

    StringBuilder sb;

    appendScrubbedName(name, sb);

    // Avoid introducing a double underscore
    if (!sb.endsWith("_"))
    {
        sb.append("_");
    }

    String key = sb.produceString();

    UInt& countRef = m_uniqueNameCounters.getOrAddValue(key, 0);
    const UInt count = countRef;
    countRef = count + 1;

    sb.append(Int32(count));
    return sb.produceString();
}

String CLikeSourceEmitter::generateName(IRInst* inst)
{
    // If the instruction names something
    // that should be emitted as a target intrinsic,
    // then use that name instead.
    UnownedStringSlice intrinsicDef;
    IRInst* intrinsicInst = nullptr;
    if (findTargetIntrinsicDefinition(inst, intrinsicDef, intrinsicInst))
    {
        return String(intrinsicDef);
    }

    // If the instruction reprsents one of the "magic" declarations
    // that makes the NVAPI library work, then we want to make sure
    // it uses the original name it was declared with, so that our
    // generated code will work correctly with either a Slang-compiled
    // or directly `#include`d version of those declarations during
    // downstream compilation.
    //
    if (auto nvapiDecor = inst->findDecoration<IRNVAPIMagicDecoration>())
    {
        return String(nvapiDecor->getName());
    }

    auto entryPointDecor = inst->findDecoration<IREntryPointDecoration>();
    if (entryPointDecor)
    {
        if (getSourceLanguage() == SourceLanguage::GLSL)
        {
            // GLSL will always need to use `main` as the
            // name for an entry-point function, but other
            // targets should try to use the original name.
            //
            // TODO: always use the original name, and
            // use the appropriate options for glslang to
            // make it support a non-`main` name.
            //
            // A function may have an entry-point deocration if it
            // is declared by the user as an entry-point function.
            // However it may not actually be compiled as an entry-point
            // when generating code for targets that doesn't support
            // multiple entry-points.
            // We only want to emit "main" for user-marked entrypoint
            // functions that are actually being selected as entrypoint
            // for current compilation. We can do so by checking if
            // a layout decoration existed on the function.
            if (inst->findDecoration<IRLayoutDecoration>())
            {
                return "main";
            }
        }

        return generateEntryPointNameImpl(entryPointDecor);
    }

    // If the instruction has a linkage decoration, just use that.
    if (auto externCppDecoration = inst->findDecoration<IRExternCppDecoration>())
    {
        // Just use the linkages mangled name directly.
        return externCppDecoration->getName();
    }

    if (auto builtinTargetVarDecoration = inst->findDecoration<IRTargetBuiltinVarDecoration>())
    {
        return getTargetBuiltinVarName(inst, builtinTargetVarDecoration->getBuiltinVarName());
    }

    // If we have a name hint on the instruction, then we will try to use that
    // to provide the basis for the actual name in the output code.
    if (auto nameHintDecoration = inst->findDecoration<IRNameHintDecoration>())
    {
        return _generateUniqueName(nameHintDecoration->getName());
    }

    // If the instruction has a linkage decoration, just use that.
    if (auto linkageDecoration = inst->findDecoration<IRLinkageDecoration>())
    {
        // Just use the linkages mangled name directly.
        return linkageDecoration->getMangledName();
    }

    if (auto ptrType = as<IRPtrType>(inst))
    {
        if (ptrType->getAddressSpace() == AddressSpace::UserPointer)
        {
            StringBuilder sb;
            sb << "BufferPointer_";
            sb << getName(inst->getOperand(0));
            sb << "_" << Int32(getID(inst));
            return sb.produceString();
        }
    }

    // Otherwise fall back to a construct temporary name
    // for the instruction.
    StringBuilder sb;
    sb << "_S";
    sb << Int32(getID(inst));

    return sb.produceString();
}

String CLikeSourceEmitter::getName(IRInst* inst)
{
    String name;
    if (!m_mapInstToName.tryGetValue(inst, name))
    {
        name = generateName(inst);
        m_mapInstToName.add(inst, name);
    }
    return name;
}

String CLikeSourceEmitter::getUnmangledName(IRInst* inst)
{
    if (auto nameHintDecor = inst->findDecoration<IRNameHintDecoration>())
        return nameHintDecor->getName();
    return getName(inst);
}

void CLikeSourceEmitter::emitSimpleValueImpl(IRInst* inst)
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
                    {
                        m_writer->emit("int8_t(");
                        m_writer->emit(int8_t(litInst->value.intVal));
                        m_writer->emit(")");
                        return;
                    }
                case BaseType::UInt8:
                    {
                        m_writer->emit("uint8_t(");
                        m_writer->emit(UInt(uint8_t(litInst->value.intVal)));
                        m_writer->emit("U");
                        m_writer->emit(")");
                        break;
                    }
                case BaseType::Int16:
                    {
                        m_writer->emit("int16_t(");
                        m_writer->emit(int16_t(litInst->value.intVal));
                        m_writer->emit(")");
                        return;
                    }
                case BaseType::UInt16:
                    {
                        m_writer->emit("uint16_t(");
                        m_writer->emit(UInt(uint16_t(litInst->value.intVal)));
                        m_writer->emit("U");
                        m_writer->emit(")");
                        break;
                    }
                case BaseType::Int:
                    {
                        m_writer->emit("int(");
                        m_writer->emit(int32_t(litInst->value.intVal));
                        m_writer->emit(")");
                        return;
                    }
                case BaseType::UInt:
                    {
                        m_writer->emit(UInt(uint32_t(litInst->value.intVal)));
                        m_writer->emit("U");
                        break;
                    }
                case BaseType::Int64:
                    {
                        m_writer->emitInt64(int64_t(litInst->value.intVal));
                        m_writer->emit("LL");
                        break;
                    }
                case BaseType::UInt64:
                    {
                        SLANG_COMPILE_TIME_ASSERT(
                            sizeof(litInst->value.intVal) >= sizeof(uint64_t));
                        m_writer->emitUInt64(uint64_t(litInst->value.intVal));
                        m_writer->emit("ULL");
                        break;
                    }
                case BaseType::IntPtr:
                    {
#if SLANG_PTR_IS_64
                        m_writer->emit("int64_t(");
                        m_writer->emitInt64(int64_t(litInst->value.intVal));
                        m_writer->emit(")");
#else
                        m_writer->emit("int(");
                        m_writer->emit(int(litInst->value.intVal));
                        m_writer->emit(")");
#endif
                        break;
                    }
                case BaseType::UIntPtr:
                    {
#if SLANG_PTR_IS_64
                        m_writer->emit("uint64_t(");
                        m_writer->emitUInt64(uint64_t(litInst->value.intVal));
                        m_writer->emit(")");
#else
                        m_writer->emit(UInt(uint32_t(litInst->value.intVal)));
                        m_writer->emit("U");
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
        m_writer->emit(((IRConstant*)inst)->value.floatVal);
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

bool CLikeSourceEmitter::shouldFoldInstIntoUseSites(IRInst* inst)
{
    // Certain opcodes should never/always be folded in
    switch (inst->getOp())
    {
    default:
        break;

    // Never fold these in, because they represent declarations
    //
    case kIROp_Var:
    case kIROp_GlobalVar:
    case kIROp_GlobalConstant:
    case kIROp_GlobalParam:
    case kIROp_Param:
    case kIROp_Func:
    case kIROp_Alloca:
    case kIROp_Store:
        return false;

    // Never fold these, because their result cannot be computed
    // as a sub-expression (they must be emitted as a declaration
    // or statement).
    case kIROp_UpdateElement:
    case kIROp_DefaultConstruct:
    case kIROp_MetalCastToDepthTexture:
        return false;

    // Always fold these in, because they are trivial
    //
    case kIROp_IntLit:
    case kIROp_FloatLit:
    case kIROp_BoolLit:
    case kIROp_CapabilityConjunction:
    case kIROp_CapabilityDisjunction:
        return true;

    // Always fold these in, because their results
    // cannot be represented in the type system of
    // our current targets.
    //
    // TODO: when we add C/C++ as an optional target,
    // we could consider lowering insts that result
    // in pointers directly.
    //
    case kIROp_FieldAddress:
    case kIROp_GetElementPtr:
    case kIROp_Specialize:
    case kIROp_LookupWitness:
    case kIROp_GetValueFromBoundInterface:
        return true;

    case kIROp_GetVulkanRayTracingPayloadLocation:
        return true;

    case kIROp_NonUniformResourceIndex:
        return true;
    }

    // Layouts and attributes are only present to annotate other
    // instructions, and should not be emitted as anything in
    // source code.
    //
    if (as<IRLayout>(inst))
        return true;
    if (as<IRAttr>(inst))
        return true;

    switch (inst->getOp())
    {
    default:
        break;

    // HACK: don't fold these in because we currently lower
    // them to initializer lists, which aren't allowed in
    // general expression contexts.
    //
    case kIROp_MakeStruct:
    case kIROp_MakeArray:
    case kIROp_swizzleSet:
    case kIROp_MakeArrayFromElement:
    case kIROp_MakeCoopVector:

        return false;
    }

    // Instructions with specific result *types* will usually
    // want to be folded in, because they aren't allowed as types
    // for temporary variables.
    auto type = inst->getDataType();

    // We treat instructions that yield a type as things we should *always* fold.
    //
    // TODO: In general, at the point where we emit code we do not expect to
    // find types being constructed locally (inside function bodies), but this
    // can end up happening because of interaction between different features.
    // Notably, if a generic function gets force-inlined early in codegen,
    // then any types it constructs will be inlined into the body of the caller
    // by default.
    //
    if (as<IRType>(inst) || as<IRTypeKind>(type))
        return true;

    // Unwrap any layers of array-ness from the type, so that
    // we can look at the underlying data type, in case we
    // should *never* expose a value of that type
    while (auto arrayType = as<IRArrayTypeBase>(type))
    {
        type = arrayType->getElementType();
    }

    // Don't allow temporaries of pointer types to be created,
    // if target langauge doesn't support pointers.
    if (as<IRPtrTypeBase>(type))
    {
        if (!doesTargetSupportPtrTypes())
            return true;
    }

    // First we check for uniform parameter groups,
    // because a `cbuffer` or GLSL `uniform` block
    // does not have a first-class type that we can
    // pass around.
    //
    // TODO: We need to ensure that type legalization
    // cleans up cases where we use a parameter group
    // or parameter block type as a function parameter...
    //
    if (as<IRUniformParameterGroupType>(type))
    {
        // TODO: we need to be careful here, because
        // HLSL shader model 6 allows these as explicit
        // types.
        return true;
    }
    //
    // The stream-output and patch types need to be handled
    // too, because they are not really first class (especially
    // not in GLSL, but they also seem to confuse the HLSL
    // compiler when they get used as temporaries).
    //
    else if (as<IRHLSLStreamOutputType>(type))
    {
        return true;
    }
    else if (as<IRHLSLPatchType>(type))
    {
        return true;
    }

    // GLSL doesn't allow texture/resource types to
    // be used as first-class values, so we need
    // to fold them into their use sites in all cases
    if (getSourceLanguage() == SourceLanguage::GLSL)
    {
        if (as<IRResourceTypeBase>(type))
        {
            return true;
        }
        else if (as<IRHLSLStructuredBufferTypeBase>(type))
        {
            return true;
        }
        else if (as<IRUntypedBufferResourceType>(type))
        {
            return true;
        }
        else if (as<IRSamplerStateTypeBase>(type))
        {
            return true;
        }
        else if (as<IRMeshOutputType>(type))
        {
            return true;
        }
        if (as<IRHitObjectType>(type))
        {
            return true;
        }
    }

    // If the instruction is at global scope, then it might represent
    // a constant (e.g., the value of an enum case).
    //
    if (as<IRModuleInst>(inst->getParent()))
    {
        if (!inst->mightHaveSideEffects())
            return true;
    }

    if (auto load = as<IRLoad>(inst))
    {
        // Loads from a constref global param should always be folded.
        auto ptrType = load->getPtr()->getDataType();
        if (load->getPtr()->getOp() == kIROp_GlobalParam)
        {
            if (ptrType->getOp() == kIROp_ConstRefType)
                return true;
            if (auto ptrTypeBase = as<IRPtrTypeBase>(ptrType))
            {
                auto addrSpace = ptrTypeBase->getAddressSpace();
                switch (addrSpace)
                {
                case Slang::AddressSpace::Uniform:
                case Slang::AddressSpace::Input:
                case Slang::AddressSpace::BuiltinInput:
                    return true;
                default:
                    break;
                }
            }
        }
    }

    // Always hold if inst is a call into an [__alwaysFoldIntoUseSite] function.
    if (auto call = as<IRCall>(inst))
    {
        auto callee = call->getCallee();
        if (getResolvedInstForDecorations(callee)
                ->findDecoration<IRAlwaysFoldIntoUseSiteDecoration>())
        {
            return true;
        }
    }

    // Having dealt with all of the cases where we *must* fold things
    // above, we can now deal with the more general cases where we
    // *should not* fold things.
    // Don't fold something with no users:
    if (!inst->hasUses())
        return false;


    // Don't fold something that has multiple users:
    if (inst->hasMoreThanOneUse())
        return false;

    // Don't fold something that might have side effects:
    if (inst->mightHaveSideEffects())
        return false;

    // Don't fold instructions that are marked `[precise]`.
    // This could in principle be extended to any other
    // decorations that affect the semantics of an instruction
    // in ways that require a temporary to be introduced.
    //
    if (inst->findDecoration<IRPreciseDecoration>())
        return false;

    // In general, undefined value should be emitted as an uninitialized
    // variable, so we shouldn't fold it.
    // However, we cannot emit all undefined values a separate variable
    // definition for certain types on certain targets (e.g. `out TriangleStream<T>`
    // for GLSL), so we check this only after all those special cases are
    // considered.
    //
    if (inst->getOp() == kIROp_undefined)
        return false;

    // Okay, at this point we know our instruction must have a single use.
    auto use = inst->firstUse;
    SLANG_ASSERT(use);
    SLANG_ASSERT(!use->nextUse);

    auto user = use->getUser();

    // Check if the use is a call using a target intrinsic that uses the parameter more than once
    // in the intrinsic definition.
    if (auto callInst = as<IRCall>(user))
    {
        const auto funcValue = callInst->getCallee();

        // Let's see if this instruction is a intrinsic call
        // This is significant, because we can within a target intrinsics definition multiple
        // accesses to the same parameter. This is not indicated into the call, and can lead to
        // output code computes something multiple times as it is folding into the expression of the
        // the target intrinsic, which we don't want.
        UnownedStringSlice intrinsicDef;
        IRInst* intrinsicInst;
        if (findTargetIntrinsicDefinition(funcValue, intrinsicDef, intrinsicInst))
        {
            // Find the index of the original instruction, to see if it's multiply used.
            IRUse* args = callInst->getArgs();
            const Index paramIndex = Index(use - args);
            SLANG_ASSERT(paramIndex >= 0 && paramIndex < Index(callInst->getArgCount()));

            // Look through the slice to seeing how many times this parameters is used (signified
            // via the $0...$9)
            {
                UnownedStringSlice slice = intrinsicDef;

                const char* cur = slice.begin();
                const char* end = slice.end();

                // Count the amount of uses
                Index useCount = 0;
                while (cur < end)
                {
                    const char c = *cur;
                    if (c == '$' && cur + 1 < end && cur[1] >= '0' && cur[1] <= '9')
                    {
                        const Index index = Index(cur[1] - '0');
                        useCount += Index(index == paramIndex);
                        cur += 2;
                    }
                    else
                    {
                        cur++;
                    }
                }

                // If there is more than one use can't fold.
                if (useCount > 1)
                {
                    return false;
                }
            }
        }
    }

    // If this is a call to a ResourceType's member function, don't fold for readability.
    if (auto call = as<IRCall>(inst))
    {
        auto callee = getResolvedInstForDecorations(call->getCallee());
        if (callee->findDecoration<IRTargetIntrinsicDecoration>())
        {
            auto funcType = as<IRFuncType>(callee->getDataType());
            if (funcType)
            {
                if (funcType->getParamCount() > 0)
                {
                    auto firstParamType = funcType->getParamType(0);
                    if (as<IRResourceTypeBase>(firstParamType))
                        return false;
                    if (as<IRHLSLStructuredBufferTypeBase>(firstParamType))
                        return false;
                    if (as<IRUntypedBufferResourceType>(firstParamType))
                        return false;
                    if (as<IRSamplerStateTypeBase>(firstParamType))
                        return false;
                }
            }
        }
    }

    // The cpp, cuda and wgsl targets don't support swizzle on the left-hand-side
    // variable, e.g. vec4.xy = vec2 is not allowed.
    // Therefore, we don't want to fold the right-hand-side expression.
    // Instead, the right-hand-side expression should be generated as a separable
    // statement and stored in a temporary varible, then assign to the left-hand-side
    // variable per element. E.g. vec4.x = vec2.x; vec4.y = vec2.y.
    if (as<IRSwizzledStore>(user))
    {
        if (isCPUTarget(getTargetReq()) || isCUDATarget(getTargetReq()) ||
            isWGPUTarget(getTargetReq()))
            return false;
    }

    // We'd like to figure out if it is safe to fold our instruction into `user`

    // First, let's make sure they are in the same block/parent:
    if (inst->getParent() != user->getParent())
        return false;


    // Now let's look at all the instructions between this instruction
    // and the user. If any of them might have side effects, then lets
    // bail out now.
    for (auto ii = inst->getNextInst(); ii != user; ii = ii->getNextInst())
    {
        if (!ii)
        {
            // We somehow reached the end of the block without finding
            // the user, which doesn't make sense if uses dominate
            // defs. Let's just play it safe and bail out.
            return false;
        }

        if (ii->mightHaveSideEffects())
            return false;
    }

    // As a safeguard, we should not allow an instruction that references
    // a block parameter to be folded into a unconcditonal branch
    // (which includes arguments for the parameters of the target block).
    //
    // For simplicity, we will just disallow folding of intructions
    // into an unconditonal branch completely, and leave a more refined
    // version of this check for later.
    //
    if (as<IRUnconditionalBranch>(user))
        return false;

    // Okay, if we reach this point then the user comes later in
    // the same block, and there are no instructions with side
    // effects in between, so it seems safe to fold things in.
    return true;
}

void CLikeSourceEmitter::emitDereferenceOperand(IRInst* inst, EmitOpInfo const& outerPrec)
{
    EmitOpInfo newOuterPrec = outerPrec;

    if (doesTargetSupportPtrTypes())
    {
        switch (inst->getOp())
        {
        case kIROp_Var:
            // If `inst` is a variable, dereferencing it is equivalent to just
            // emit its name. i.e. *&var ==> var.
            // We apply this peep hole optimization here to reduce the clutter of
            // resulting code.
            m_writer->emit(getName(inst));
            return;
        case kIROp_FieldAddress:
            {
                auto innerPrec = getInfo(EmitOp::Postfix);
                bool innerNeedClose = maybeEmitParens(newOuterPrec, innerPrec);
                auto ii = as<IRFieldAddress>(inst);
                auto base = ii->getBase();
                if (isPtrToClassType(base->getDataType()))
                    emitDereferenceOperand(base, leftSide(newOuterPrec, innerPrec));
                else
                    emitOperand(base, leftSide(newOuterPrec, innerPrec));
                m_writer->emit("->");
                m_writer->emit(getName(ii->getField()));
                maybeCloseParens(innerNeedClose);
                return;
            }
        case kIROp_GetElementPtr:
            {
                const auto info = getInfo(EmitOp::Prefix);
                IRVectorType* vectorType = nullptr;
                if (auto ptrType = as<IRPtrTypeBase>(inst->getOperand(0)->getDataType()))
                {
                    vectorType = as<IRVectorType>(ptrType->getValueType());
                }
                if (vectorType)
                {
                    // Can't use simplified emit logic for get vector element operations on CUDA
                    // targets.
                    if (isCUDATarget(m_codeGenContext->getTargetReq()))
                        break;
                }

                auto rightSidePrec = rightSide(outerPrec, info);
                auto postfixInfo = getInfo(EmitOp::Postfix);
                bool rightSideNeedClose = maybeEmitParens(rightSidePrec, postfixInfo);
                emitDereferenceOperand(inst->getOperand(0), leftSide(rightSidePrec, postfixInfo));
                bool emitBracketPostfix = true;
                if (vectorType)
                {
                    // Simplify the emitted code if we are referencing a known vector element.
                    if (auto intLit = as<IRIntLit>(inst->getOperand(1)))
                    {
                        emitBracketPostfix = false;
                        switch (intLit->getValue())
                        {
                        case 0:
                            m_writer->emit(".x");
                            break;
                        case 1:
                            m_writer->emit(".y");
                            break;
                        case 2:
                            m_writer->emit(".z");
                            break;
                        case 3:
                            m_writer->emit(".w");
                            break;
                        default:
                            emitBracketPostfix = true;
                            break;
                        }
                    }
                }
                if (emitBracketPostfix)
                {
                    m_writer->emit("[");
                    emitOperand(inst->getOperand(1), getInfo(EmitOp::General));
                    m_writer->emit("]");
                }
                maybeCloseParens(rightSideNeedClose);
                return;
            }
        default:
            break;
        }

        auto dereferencePrec = EmitOpInfo::get(EmitOp::Prefix);
        bool needClose = maybeEmitParens(newOuterPrec, dereferencePrec);
        m_writer->emit("*");
        emitOperand(inst, rightSide(newOuterPrec, dereferencePrec));
        maybeCloseParens(needClose);
    }
    else
    {
        emitOperand(inst, outerPrec);
    }
}

void CLikeSourceEmitter::emitVarExpr(IRInst* inst, EmitOpInfo const& outerPrec)
{
    if (doesTargetSupportPtrTypes())
    {
        auto prec = getInfo(EmitOp::Prefix);
        auto newOuterPrec = outerPrec;
        bool needClose = maybeEmitParens(newOuterPrec, prec);
        m_writer->emit("&");
        m_writer->emit(getName(inst));
        maybeCloseParens(needClose);
    }
    else
    {
        m_writer->emit(getName(inst));
    }
}

void CLikeSourceEmitter::emitOperandImpl(IRInst* inst, EmitOpInfo const& outerPrec)
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

void CLikeSourceEmitter::emitArgs(IRInst* inst)
{
    UInt argCount = inst->getOperandCount();
    IRUse* args = inst->getOperands();

    m_writer->emit("(");
    for (UInt aa = 0; aa < argCount; ++aa)
    {
        if (aa != 0)
            m_writer->emit(", ");
        emitOperand(args[aa].get(), getInfo(EmitOp::General));
    }
    m_writer->emit(")");
}

void CLikeSourceEmitter::emitRateQualifiers(IRInst* value)
{
    const auto rate = value->getRate();
    if (rate)
    {
        emitRateQualifiersAndAddressSpaceImpl(rate, AddressSpace::Generic);
    }
}

void CLikeSourceEmitter::emitRateQualifiersAndAddressSpace(IRInst* value)
{
    const auto rate = value->getRate();
    const auto ptrTy = composeGetters<IRPtrTypeBase>(value, &IRInst::getDataType);
    const auto addressSpace = ptrTy ? ptrTy->getAddressSpace() : AddressSpace::Generic;
    if (rate || addressSpace != AddressSpace::Generic)
    {
        emitRateQualifiersAndAddressSpaceImpl(rate, addressSpace);
    }
}

void CLikeSourceEmitter::emitInstResultDecl(IRInst* inst)
{
    auto type = inst->getDataType();
    if (!type)
        return;

    if (as<IRVoidType>(type))
        return;

    emitTempModifiers(inst);

    emitRateQualifiers(inst);

    if (as<IRModuleInst>(inst->getParent()))
    {
        // "Ordinary" instructions at module scope are constants

        switch (getSourceLanguage())
        {
        case SourceLanguage::CUDA:
        case SourceLanguage::HLSL:
        case SourceLanguage::C:
        case SourceLanguage::CPP:
            m_writer->emit("static const ");
            break;
        case SourceLanguage::Metal:
            m_writer->emit("constant ");
            break;
        case SourceLanguage::WGSL:
            // This is handled by emitVarKeyword, below
            break;
        default:
            m_writer->emit("const ");
            break;
        }
    }

    emitVarKeyword(type, inst);

    emitType(type, getName(inst));
    m_writer->emit(" = ");
}

template<typename T>
IRTargetSpecificDecoration* CLikeSourceEmitter::findBestTargetDecoration(IRInst* inInst)
{
    return Slang::findBestTargetDecoration<T>(inInst, getTargetCaps());
}

IRTargetIntrinsicDecoration* CLikeSourceEmitter::_findBestTargetIntrinsicDecoration(IRInst* inInst)
{
    return as<IRTargetIntrinsicDecoration>(
        findBestTargetDecoration<IRTargetSpecificDefinitionDecoration>(inInst));
}

/* static */ bool CLikeSourceEmitter::isOrdinaryName(UnownedStringSlice const& name)
{
    char const* cursor = name.begin();
    char const* const end = name.end();

    // Consume an optional `.` at the start, which indicates
    // the ordinary name is for a member function.
    if (cursor < end && *cursor == '.')
        cursor++;

    // Must have at least one char, and first char can't be a digit
    if (cursor >= end || CharUtil::isDigit(cursor[0]))
        return false;

    for (; cursor < end; ++cursor)
    {
        const auto c = *cursor;
        if (CharUtil::isAlphaOrDigit(c) || c == '_')
        {
            continue;
        }

        // We allow :: for scope
        if (c == ':' && cursor + 1 < end && cursor[1] == ':')
        {
            ++cursor;
            continue;
        }

        return false;
    }
    return true;
}


void CLikeSourceEmitter::emitIntrinsicCallExpr(
    IRCall* inst,
    UnownedStringSlice intrinsicDefinition,
    IRInst* intrinsicInst,
    EmitOpInfo const& inOuterPrec)
{
    emitIntrinsicCallExprImpl(inst, intrinsicDefinition, intrinsicInst, inOuterPrec);
}

void CLikeSourceEmitter::emitIntrinsicCallExprImpl(
    IRCall* inst,
    UnownedStringSlice intrinsicDefinition,
    IRInst* intrinsicInst,
    EmitOpInfo const& inOuterPrec)
{
    auto outerPrec = inOuterPrec;

    IRUse* args = inst->getOperands();
    Index argCount = inst->getOperandCount();

    // First operand was the function to be called
    args++;
    argCount--;

    auto name = intrinsicDefinition;

    if (isOrdinaryName(name))
    {
        // Simple case: it is just an ordinary name, so we call it like a builtin.
        auto prec = getInfo(EmitOp::Postfix);
        bool needClose = maybeEmitParens(outerPrec, prec);

        // The definition string may be an ordinary name prefixed with `.`
        // to indicate that the operation should be called as a member
        // function on its first operand.
        //
        if (name[0] == '.')
        {
            emitOperand(args[0].get(), leftSide(outerPrec, prec));
            m_writer->emit(".");

            name = UnownedStringSlice(name.begin() + 1, name.end());
            args++;
            argCount--;
        }

        m_writer->emit(name);
        m_writer->emit("(");
        for (Index aa = 0; aa < argCount; ++aa)
        {
            if (aa != 0)
                m_writer->emit(", ");
            emitOperand(args[aa].get(), getInfo(EmitOp::General));
        }
        m_writer->emit(")");

        maybeCloseParens(needClose);
        return;
    }
    else if (name == ".operator[]")
    {
        // The user is invoking a built-in subscript operator
        //
        // TODO: We might want to remove this bit of special-casing
        // in favor of making all subscript operations in the standard
        // library explicitly declare how they lower. On the flip
        // side, that would require modifications to a very large
        // number of declarations.

        auto prec = getInfo(EmitOp::Postfix);
        bool needClose = maybeEmitParens(outerPrec, prec);

        Int argIndex = 0;

        emitOperand(args[argIndex++].get(), leftSide(outerPrec, prec));
        m_writer->emit("[");
        emitOperand(args[argIndex++].get(), getInfo(EmitOp::General));
        m_writer->emit("]");

        if (argIndex < argCount)
        {
            m_writer->emit(" = ");
            emitOperand(args[argIndex++].get(), getInfo(EmitOp::General));
        }

        maybeCloseParens(needClose);
        return;
    }
    else
    {
        IntrinsicExpandContext context(this);
        context.emit(inst, args, argCount, name, intrinsicInst);
    }
}

void CLikeSourceEmitter::emitCallArg(IRInst* inst)
{
    emitOperand(inst, getInfo(EmitOp::General));
}

void CLikeSourceEmitter::_emitCallArgList(IRCall* inst, int startingOperandIndex)
{
    bool isFirstArg = true;
    m_writer->emit("(");
    UInt argCount = inst->getOperandCount();
    for (UInt aa = startingOperandIndex; aa < argCount; ++aa)
    {
        auto operand = inst->getOperand(aa);
        if (as<IRVoidType>(operand->getDataType()))
            continue;

        // TODO: [generate dynamic dispatch code for generics]
        // Pass RTTI object here. Ignore type argument for now.
        if (as<IRType>(operand))
            continue;

        if (!isFirstArg)
            m_writer->emit(", ");
        else
            isFirstArg = false;
        emitCallArg(inst->getOperand(aa));
    }
    m_writer->emit(")");
}

void CLikeSourceEmitter::emitComInterfaceCallExpr(IRCall* inst, EmitOpInfo const& inOuterPrec)
{
    auto funcValue = inst->getOperand(0);
    auto object = funcValue->getOperand(0);
    auto methodKey = funcValue->getOperand(1);
    auto prec = getInfo(EmitOp::Postfix);

    auto outerPrec = inOuterPrec;
    bool needClose = maybeEmitParens(outerPrec, prec);

    emitOperand(object, leftSide(outerPrec, prec));
    m_writer->emit("->");
    m_writer->emit(getName(methodKey));
    _emitCallArgList(inst, 2);
    maybeCloseParens(needClose);
}

bool CLikeSourceEmitter::findTargetIntrinsicDefinition(
    IRInst* callee,
    UnownedStringSlice& outDefinition,
    IRInst*& outInst)
{
    return Slang::findTargetIntrinsicDefinition(callee, getTargetCaps(), outDefinition, outInst);
}

void CLikeSourceEmitter::emitCallExpr(IRCall* inst, EmitOpInfo outerPrec)
{
    auto funcValue = inst->getOperand(0);

    // Does this function declare any requirements.
    handleRequiredCapabilities(funcValue);

    // Detect if this is a call into a COM interface method.
    if (funcValue->getOp() == kIROp_LookupWitness)
    {
        auto operand0Type = funcValue->getOperand(0)->getDataType();
        switch (operand0Type->getOp())
        {
        case kIROp_WitnessTableIDType:
        case kIROp_WitnessTableType:
            if (as<IRWitnessTableTypeBase>(operand0Type)
                    ->getConformanceType()
                    ->findDecoration<IRComInterfaceDecoration>())
            {
                emitComInterfaceCallExpr(inst, outerPrec);
                return;
            }
            break;
        case kIROp_ComPtrType:
        case kIROp_PtrType:
        case kIROp_NativePtrType:
            emitComInterfaceCallExpr(inst, outerPrec);
            return;
        }
    }

    // We want to detect any call to an intrinsic operation,
    // that we can emit it directly without mangling, etc.
    UnownedStringSlice intrinsicDefinition;
    IRInst* intrinsicInst;
    auto resolvedFunc = getResolvedInstForDecorations(funcValue);
    if (findTargetIntrinsicDefinition(resolvedFunc, intrinsicDefinition, intrinsicInst))
    {
        // Make sure we register all required preludes for emit.
        if (auto func = as<IRFunc>(resolvedFunc))
        {
            for (auto block : func->getBlocks())
            {
                for (auto ii : block->getChildren())
                {
                    if (auto requirePrelude = as<IRRequirePrelude>(ii))
                    {
                        auto preludeTextInst = as<IRStringLit>(requirePrelude->getOperand(0));
                        if (preludeTextInst)
                            m_requiredPreludes.add(preludeTextInst);
                    }
                }
            }
        }
        emitIntrinsicCallExpr(inst, intrinsicDefinition, intrinsicInst, outerPrec);
    }
    else
    {
        auto prec = getInfo(EmitOp::Postfix);
        bool needClose = maybeEmitParens(outerPrec, prec);

        emitOperand(funcValue, leftSide(outerPrec, prec));
        _emitCallArgList(inst);
        maybeCloseParens(needClose);
    }
}

void CLikeSourceEmitter::emitInstExpr(IRInst* inst, const EmitOpInfo& inOuterPrec)
{
    // Try target specific impl first
    if (tryEmitInstExprImpl(inst, inOuterPrec))
    {
        return;
    }
    defaultEmitInstExpr(inst, inOuterPrec);
}

void CLikeSourceEmitter::emitInstStmt(IRInst* inst)
{
    // Try target specific impl first
    if (tryEmitInstStmtImpl(inst))
    {
        return;
    }
    defaultEmitInstStmt(inst);
}

void CLikeSourceEmitter::diagnoseUnhandledInst(IRInst* inst)
{
    getSink()->diagnose(inst, Diagnostics::unimplemented, "unexpected IR opcode during code emit");
}

bool CLikeSourceEmitter::hasExplicitConstantBufferOffset(IRInst* cbufferType)
{
    auto type = as<IRUniformParameterGroupType>(cbufferType);
    if (!type)
        return false;
    if (as<IRGLSLShaderStorageBufferType>(cbufferType))
        return false;
    auto structType = as<IRStructType>(type->getElementType());
    if (!structType)
        return false;
    for (auto ff : structType->getFields())
    {
        if (ff->getKey()->findDecoration<IRPackOffsetDecoration>())
            return true;
    }
    return false;
}

bool CLikeSourceEmitter::isSingleElementConstantBuffer(IRInst* cbufferType)
{
    auto type = as<IRUniformParameterGroupType>(cbufferType);
    if (!type)
        return false;
    if (as<IRGLSLShaderStorageBufferType>(cbufferType))
        return false;
    auto structType = as<IRStructType>(type->getElementType());
    if (structType)
        return false;
    return true;
}

bool CLikeSourceEmitter::shouldForceUnpackConstantBufferElements(IRInst* cbufferType)
{
    if (getTargetReq()->getTarget() != CodeGenTarget::HLSL)
        return false;
    if (!getTargetProgram()->getOptionSet().getBoolOption(
            CompilerOptionName::NoHLSLPackConstantBufferElements))
        return false;
    auto type = as<IRUniformParameterGroupType>(cbufferType);
    if (!type)
        return false;
    auto structType = as<IRStructType>(type->getElementType());
    if (!structType)
        return false;
    return true;
}

void CLikeSourceEmitter::defaultEmitInstExpr(IRInst* inst, const EmitOpInfo& inOuterPrec)
{
    EmitOpInfo outerPrec = inOuterPrec;
    bool needClose = false;
    switch (inst->getOp())
    {
    case kIROp_GlobalHashedStringLiterals:
        /* Don't need to to output anything for this instruction - it's used for reflecting
        string literals that are hashed with 'getStringHash' */
        break;
    case kIROp_RTTIPointerType:
        break;

    case kIROp_undefined:
    case kIROp_DefaultConstruct:
        m_writer->emit(getName(inst));
        break;

    case kIROp_IntLit:
    case kIROp_FloatLit:
    case kIROp_BoolLit:
        emitSimpleValue(inst);
        break;

    case kIROp_MakeCoopVector:
    case kIROp_MakeVector:
    case kIROp_MakeMatrix:
    case kIROp_VectorReshape:
    case kIROp_CastFloatToInt:
    case kIROp_CastIntToFloat:
    case kIROp_IntCast:
    case kIROp_FloatCast:
        // Simple constructor call
        emitType(inst->getDataType());
        emitArgs(inst);
        break;
    case kIROp_MakeMatrixFromScalar:
        {
            emitType(inst->getDataType());
            auto matrixType = as<IRMatrixType>(inst->getDataType());
            SLANG_RELEASE_ASSERT(matrixType);
            auto columnCount = as<IRIntLit>(matrixType->getColumnCount());
            SLANG_RELEASE_ASSERT(columnCount);
            auto rowCount = as<IRIntLit>(matrixType->getRowCount());
            SLANG_RELEASE_ASSERT(rowCount);
            m_writer->emit("(");
            for (IRIntegerValue i = 0; i < rowCount->getValue() * columnCount->getValue(); i++)
            {
                if (i != 0)
                    m_writer->emit(", ");
                emitOperand(inst->getOperand(0), getInfo(EmitOp::General));
            }
            m_writer->emit(")");
        }
        break;
    case kIROp_AllocObj:
        m_writer->emit("new ");
        m_writer->emit(getName(inst->getDataType()));
        m_writer->emit("()");
        break;
    case kIROp_MakeUInt64:
        m_writer->emit("((");
        emitType(inst->getDataType());
        m_writer->emit("(");
        emitOperand(inst->getOperand(1), getInfo(EmitOp::General));
        m_writer->emit(") << 32) + ");
        emitOperand(inst->getOperand(0), getInfo(EmitOp::General));
        m_writer->emit(")");
        break;
    case kIROp_MakeVectorFromScalar:
    case kIROp_MatrixReshape:
    case kIROp_CastPtrToInt:
    case kIROp_CastIntToPtr:
    case kIROp_PtrCast:
        {
            // Simple constructor call
            auto prec = getInfo(EmitOp::Prefix);
            needClose = maybeEmitParens(outerPrec, prec);

            m_writer->emit("(");
            emitType(inst->getDataType());
            m_writer->emit(")");

            emitOperand(inst->getOperand(0), rightSide(outerPrec, prec));
            break;
        }
    case kIROp_FieldExtract:
        {
            // Extract field from aggregate
            IRFieldExtract* fieldExtract = (IRFieldExtract*)inst;

            auto prec = getInfo(EmitOp::Postfix);
            needClose = maybeEmitParens(outerPrec, prec);

            auto base = fieldExtract->getBase();
            emitOperand(base, leftSide(outerPrec, prec));
            if (base->getDataType()->getOp() == kIROp_ClassType)
                m_writer->emit("->");
            else
                m_writer->emit(".");
            m_writer->emit(getName(fieldExtract->getField()));
            break;
        }
    case kIROp_FieldAddress:
        {
            // Extract field "address" from aggregate

            IRFieldAddress* ii = (IRFieldAddress*)inst;

            if (doesTargetSupportPtrTypes())
            {
                auto prec = getInfo(EmitOp::Prefix);
                needClose = maybeEmitParens(outerPrec, prec);
                m_writer->emit("&");
                outerPrec = rightSide(outerPrec, prec);
                auto innerPrec = getInfo(EmitOp::Postfix);
                bool innerNeedClose = maybeEmitParens(outerPrec, innerPrec);
                auto base = ii->getBase();
                if (isPtrToClassType(base->getDataType()))
                    emitDereferenceOperand(base, leftSide(outerPrec, innerPrec));
                else
                    emitOperand(base, leftSide(outerPrec, innerPrec));
                m_writer->emit("->");
                m_writer->emit(getName(ii->getField()));
                maybeCloseParens(innerNeedClose);
            }
            else
            {
                auto prec = getInfo(EmitOp::Postfix);
                needClose = maybeEmitParens(outerPrec, prec);
                bool skipBase =
                    (isD3DTarget(getTargetReq()) &&
                     hasExplicitConstantBufferOffset(ii->getBase()->getDataType())) ||
                    shouldForceUnpackConstantBufferElements(ii->getBase()->getDataType());
                if (!skipBase)
                {
                    auto base = ii->getBase();
                    emitOperand(base, leftSide(outerPrec, prec));
                    m_writer->emit(".");
                }
                m_writer->emit(getName(ii->getField()));
            }
            break;
        }

    // Comparisons
    case kIROp_Eql:
    case kIROp_Neq:
    case kIROp_Greater:
    case kIROp_Less:
    case kIROp_Geq:
    case kIROp_Leq:
        {
            const auto emitOp = getEmitOpForOp(inst->getOp());

            auto prec = getInfo(emitOp);
            needClose = maybeEmitParens(outerPrec, prec);

            emitOperand(inst->getOperand(0), leftSide(outerPrec, prec));
            m_writer->emit(" ");
            m_writer->emit(prec.op);
            m_writer->emit(" ");
            emitOperand(inst->getOperand(1), rightSide(outerPrec, prec));
            break;
        }
    case kIROp_CastDescriptorHandleToUInt2:
    case kIROp_CastUInt2ToDescriptorHandle:
    case kIROp_CastDescriptorHandleToResource:
        emitOperand(inst->getOperand(0), outerPrec);
        break;
    // Binary ops
    case kIROp_Add:
    case kIROp_Sub:
    case kIROp_Div:
    case kIROp_IRem:
    case kIROp_FRem:
    case kIROp_Lsh:
    case kIROp_Rsh:
    case kIROp_BitXor:
    case kIROp_BitOr:
    case kIROp_BitAnd:
    case kIROp_And:
    case kIROp_Or:
    case kIROp_Mul:
        {
            const auto emitOp = getEmitOpForOp(inst->getOp());
            const auto info = getInfo(emitOp);

            needClose = maybeEmitParens(outerPrec, info);
            emitOperand(inst->getOperand(0), leftSide(outerPrec, info));
            m_writer->emit(" ");
            m_writer->emit(info.op);
            m_writer->emit(" ");
            emitOperand(inst->getOperand(1), rightSide(outerPrec, info));
            break;
        }
    // Unary
    case kIROp_Not:
    case kIROp_Neg:
    case kIROp_BitNot:
        {
            IRInst* operand = inst->getOperand(0);

            const auto emitOp = getEmitOpForOp(inst->getOp());
            const auto prec = getInfo(emitOp);

            needClose = maybeEmitParens(outerPrec, prec);

            switch (inst->getOp())
            {
            case kIROp_BitNot:
                {
                    // If it's a BitNot, but the data type is bool special case to !
                    m_writer->emit(as<IRBoolType>(inst->getDataType()) ? "!" : prec.op);
                    break;
                }
            case kIROp_Not:
                {
                    m_writer->emit(prec.op);
                    break;
                }
            case kIROp_Neg:
                {
                    // Emit a space after the unary -, so if we are followed by a negative literal
                    // we don't end up with -- which some downstream compilers determine to be
                    // decrement.
                    m_writer->emit("- ");
                    break;
                }
            }

            emitOperand(operand, rightSide(outerPrec, prec));
            break;
        }
    case kIROp_Load:
        {
            auto base = inst->getOperand(0);
            emitDereferenceOperand(base, outerPrec);
            if (isKhronosTarget(getTargetReq()) &&
                isSingleElementConstantBuffer(base->getDataType()))
            {
                m_writer->emit("._data");
            }
        }
        break;

    case kIROp_StructuredBufferLoad:
    case kIROp_RWStructuredBufferLoad:
        {
            auto base = inst->getOperand(0);
            emitOperand(base, outerPrec);
            m_writer->emit(".Load(");
            emitOperand(inst->getOperand(1), EmitOpInfo());
            m_writer->emit(")");
        }
        break;

    case kIROp_StructuredBufferLoadStatus:
    case kIROp_RWStructuredBufferLoadStatus:
        {
            auto base = inst->getOperand(0);
            emitOperand(base, outerPrec);
            m_writer->emit(".Load(");
            emitOperand(inst->getOperand(1), EmitOpInfo());
            m_writer->emit(", ");
            emitOperand(inst->getOperand(2), EmitOpInfo());
            m_writer->emit(")");
        }
        break;

    case kIROp_RWStructuredBufferGetElementPtr:
        {
            auto base = inst->getOperand(0);
            emitOperand(base, outerPrec);
            m_writer->emit("[");
            emitOperand(inst->getOperand(1), EmitOpInfo());
            m_writer->emit("]");
        }
        break;

    case kIROp_GetEquivalentStructuredBuffer:
        {
            auto base = inst->getOperand(0);
            emitOperand(base, outerPrec);
            m_writer->emit(".asStructuredBuffer<");
            emitType(as<IRHLSLStructuredBufferTypeBase>(inst->getDataType())->getElementType());
            m_writer->emit(">()");
        }
        break;

    case kIROp_RWStructuredBufferStore:
        {
            auto base = inst->getOperand(0);
            emitOperand(base, EmitOpInfo());
            m_writer->emit(".Store(");
            emitOperand(inst->getOperand(1), EmitOpInfo());
            m_writer->emit(", ");
            emitOperand(inst->getOperand(2), EmitOpInfo());
            m_writer->emit(")");
        }
        break;

    case kIROp_StructuredBufferAppend:
        {
            auto outer = getInfo(EmitOp::General);
            emitOperand(inst->getOperand(0), leftSide(outer, getInfo(EmitOp::Postfix)));
            m_writer->emit(".Append(");
            emitOperand(inst->getOperand(1), getInfo(EmitOp::General));
            m_writer->emit(")");
        }
        break;
    case kIROp_StructuredBufferConsume:
        {
            auto outer = getInfo(EmitOp::General);
            emitOperand(inst->getOperand(0), leftSide(outer, getInfo(EmitOp::Postfix)));
            m_writer->emit(".Consume()");
        }
        break;

    case kIROp_Call:
        {
            emitCallExpr((IRCall*)inst, outerPrec);
        }
        break;

    case kIROp_GroupMemoryBarrierWithGroupSync:
        m_writer->emit("GroupMemoryBarrierWithGroupSync()");
        break;

    case kIROp_NonUniformResourceIndex:
        emitOperand(
            inst->getOperand(0),
            getInfo(EmitOp::General)); // Directly emit NonUniformResourceIndex Operand0;
        break;

    case kIROp_getNativeStr:
        {
            auto prec = getInfo(EmitOp::Postfix);
            needClose = maybeEmitParens(outerPrec, prec);
            emitOperand(inst->getOperand(0), leftSide(outerPrec, prec));
            if (as<IRPtrTypeBase>(inst->getOperand(0)->getDataType()))
                m_writer->emit("->");
            else
                m_writer->emit(".");
            m_writer->emit("getBuffer()");
            break;
        }
    case kIROp_MakeString:
        {
            m_writer->emit("String(");
            emitOperand(inst->getOperand(0), EmitOpInfo());
            m_writer->emit(")");
            break;
        }
    case kIROp_GetNativePtr:
        {
            auto prec = getInfo(EmitOp::Postfix);
            needClose = maybeEmitParens(outerPrec, prec);
            emitOperand(inst->getOperand(0), leftSide(outerPrec, prec));
            m_writer->emit(".get()");
            break;
        }
    case kIROp_GetManagedPtrWriteRef:
        {
            auto prec = getInfo(EmitOp::Postfix);
            needClose = maybeEmitParens(outerPrec, prec);
            emitDereferenceOperand(inst->getOperand(0), leftSide(outerPrec, prec));
            m_writer->emit(".writeRef()");
            break;
        }
    case kIROp_ManagedPtrAttach:
        {
            auto prec = getInfo(EmitOp::Postfix);
            needClose = maybeEmitParens(outerPrec, prec);
            emitDereferenceOperand(inst->getOperand(0), leftSide(outerPrec, prec));
            m_writer->emit(".attach(");
            emitOperand(inst->getOperand(1), EmitOpInfo());
            m_writer->emit(")");
            break;
        }
    case kIROp_ManagedPtrDetach:
        {
            auto prec = getInfo(EmitOp::Postfix);
            needClose = maybeEmitParens(outerPrec, prec);
            emitOperand(inst->getOperand(0), leftSide(outerPrec, prec));
            m_writer->emit(".detach()");
            break;
        }
    case kIROp_GetOffsetPtr:
        {
            auto prec = getInfo(EmitOp::Add);
            needClose = maybeEmitParens(outerPrec, prec);
            emitOperand(inst->getOperand(0), leftSide(outerPrec, prec));
            m_writer->emit(" + ");
            emitOperand(inst->getOperand(1), rightSide(outerPrec, prec));
            break;
        }

    case kIROp_ImageSubscript:
        // We should have legalized ImageSubscript before emit for metal targets
        if (isMetalTarget(this->getTargetReq()))
            getSink()->diagnose(
                inst,
                Diagnostics::unimplemented,
                "kIROp_ImageSubscript is unimplemented for Metal, expected legalization "
                "beforehand");
        [[fallthrough]];
    case kIROp_GetElement:
    case kIROp_MeshOutputRef:
    case kIROp_GetElementPtr:
        // HACK: deal with translation of GLSL geometry shader input arrays.
        if (auto decoration = inst->getOperand(0)->findDecoration<IRGLSLOuterArrayDecoration>())
        {
            auto prec = getInfo(EmitOp::Postfix);
            needClose = maybeEmitParens(outerPrec, prec);

            m_writer->emit(decoration->getOuterArrayName());
            m_writer->emit("[");
            emitOperand(inst->getOperand(1), getInfo(EmitOp::General));
            m_writer->emit("].");
            emitOperand(inst->getOperand(0), rightSide(outerPrec, prec));
            break;
        }
        else
        {
            if (inst->getOp() == kIROp_GetElementPtr && doesTargetSupportPtrTypes())
            {
                const auto info = getInfo(EmitOp::Prefix);
                needClose = maybeEmitParens(outerPrec, info);
                m_writer->emit("&");
                auto rightSidePrec = rightSide(outerPrec, info);
                auto postfixInfo = getInfo(EmitOp::Postfix);
                bool rightSideNeedClose = maybeEmitParens(rightSidePrec, postfixInfo);
                emitDereferenceOperand(inst->getOperand(0), leftSide(rightSidePrec, postfixInfo));
                m_writer->emit("[");
                emitOperand(inst->getOperand(1), getInfo(EmitOp::General));
                m_writer->emit("]");
                maybeCloseParens(rightSideNeedClose);
                break;
            }
            else
            {
                auto prec = getInfo(EmitOp::Postfix);
                needClose = maybeEmitParens(outerPrec, prec);

                emitOperand(inst->getOperand(0), leftSide(outerPrec, prec));
                m_writer->emit("[");
                emitOperand(inst->getOperand(1), getInfo(EmitOp::General));
                m_writer->emit("]");
            }
        }
        break;

    case kIROp_swizzle:
        {
            auto prec = getInfo(EmitOp::Postfix);
            needClose = maybeEmitParens(outerPrec, prec);

            auto ii = (IRSwizzle*)inst;
            emitOperand(ii->getBase(), leftSide(outerPrec, prec));
            m_writer->emit(".");
            const Index elementCount = Index(ii->getElementCount());
            for (Index ee = 0; ee < elementCount; ++ee)
            {
                IRInst* irElementIndex = ii->getElementIndex(ee);
                SLANG_RELEASE_ASSERT(irElementIndex->getOp() == kIROp_IntLit);
                IRConstant* irConst = (IRConstant*)irElementIndex;

                UInt elementIndex = (UInt)irConst->value.intVal;
                SLANG_RELEASE_ASSERT(elementIndex < 4);

                char const* kComponents[] = {"x", "y", "z", "w"};
                m_writer->emit(kComponents[elementIndex]);
            }
        }
        break;

    case kIROp_Specialize:
        {
            emitOperand(inst->getOperand(0), outerPrec);
        }
        break;

    case kIROp_WrapExistential:
        {
            // Normally `WrapExistential` shouldn't exist in user code at this point.
            // The only exception is when the user is calling a core module generic
            // function that has an existential type argument, for example
            // `StructuredBuffer<ISomething>.Load()`.
            // We can safely ignore the `wrapExistential` operation in this case.
            emitOperand(inst->getOperand(0), outerPrec);
        }
        break;

    case kIROp_Select:
        {
            auto prec = getInfo(EmitOp::Conditional);
            needClose = maybeEmitParens(outerPrec, prec);

            emitOperand(inst->getOperand(0), leftSide(outerPrec, prec));
            m_writer->emit(" ? ");
            emitOperand(inst->getOperand(1), prec);
            m_writer->emit(" : ");
            emitOperand(inst->getOperand(2), rightSide(outerPrec, prec));
        }
        break;

    case kIROp_Param:
        m_writer->emit(getName(inst));
        break;

    case kIROp_MakeArray:
    case kIROp_MakeStruct:
        {
            // TODO: initializer-list syntax may not always
            // be appropriate, depending on the context
            // of the expression.

            m_writer->emit("{ ");
            UInt argCount = inst->getOperandCount();
            for (UInt aa = 0; aa < argCount; ++aa)
            {
                if (aa != 0)
                    m_writer->emit(", ");
                emitOperand(inst->getOperand(aa), getInfo(EmitOp::General));
            }
            m_writer->emit(" }");
        }
        break;
    case kIROp_MakeArrayFromElement:
        {
            // TODO: initializer-list syntax may not always
            // be appropriate, depending on the context
            // of the expression.

            m_writer->emit("{ ");
            UInt argCount =
                (UInt)cast<IRIntLit>(cast<IRArrayType>(inst->getDataType())->getElementCount())
                    ->getValue();
            for (UInt aa = 0; aa < argCount; ++aa)
            {
                if (aa != 0)
                    m_writer->emit(", ");
                emitOperand(inst->getOperand(0), getInfo(EmitOp::General));
            }
            m_writer->emit(" }");
        }
        break;
    case kIROp_BitCast:
        {
            // Note: we are currently emitting casts as plain old
            // C-style casts, which may not always perform a bitcast.
            //
            // TODO: This operation should map to an intrinsic to be
            // provided in a prelude for C/C++, so that the target
            // can easily emit code for whatever the best possible
            // bitcast is on the platform.

            auto prec = getInfo(EmitOp::Prefix);
            needClose = maybeEmitParens(outerPrec, prec);

            m_writer->emit("(");
            emitType(inst->getDataType());
            m_writer->emit(")");
            m_writer->emit("(");
            emitOperand(inst->getOperand(0), getInfo(EmitOp::General));
            m_writer->emit(")");
        }
        break;
    case kIROp_GlobalConstant:
    case kIROp_GetValueFromBoundInterface:
        emitOperand(inst->getOperand(0), outerPrec);
        break;

    case kIROp_ByteAddressBufferLoad:
        {
            m_writer->emit("(");
            emitOperand(inst->getOperand(0), getInfo(EmitOp::General));
            m_writer->emit(").Load<");
            emitType(inst->getDataType());
            m_writer->emit(" >(");
            emitOperand(inst->getOperand(1), getInfo(EmitOp::General));
            m_writer->emit(")");
            break;
        }

    case kIROp_ByteAddressBufferStore:
        {
            auto prec = getInfo(EmitOp::Postfix);
            needClose = maybeEmitParens(outerPrec, prec);

            emitOperand(inst->getOperand(0), leftSide(outerPrec, prec));
            m_writer->emit(".Store(");
            emitOperand(inst->getOperand(1), getInfo(EmitOp::General));
            m_writer->emit(",");
            emitOperand(inst->getOperand(inst->getOperandCount() - 1), getInfo(EmitOp::General));
            m_writer->emit(")");
            break;
        }
    case kIROp_BitfieldExtract:
        {
            emitBitfieldExtractImpl(inst);
            break;
        }
    case kIROp_BitfieldInsert:
        {
            emitBitfieldInsertImpl(inst);
            break;
        }
    case kIROp_PackAnyValue:
        {
            m_writer->emit("packAnyValue<");
            m_writer->emit(getIntVal(cast<IRAnyValueType>(inst->getDataType())->getSize()));
            m_writer->emit(",");
            emitType(inst->getOperand(0)->getDataType());
            m_writer->emit(">(");
            emitOperand(inst->getOperand(0), getInfo(EmitOp::General));
            m_writer->emit(")");
            break;
        }
    case kIROp_UnpackAnyValue:
        {
            m_writer->emit("unpackAnyValue<");
            m_writer->emit(
                getIntVal(cast<IRAnyValueType>(inst->getOperand(0)->getDataType())->getSize()));
            m_writer->emit(",");
            emitType(inst->getDataType());
            m_writer->emit(">(");
            emitOperand(inst->getOperand(0), getInfo(EmitOp::General));
            m_writer->emit(")");
            break;
        }
    case kIROp_GpuForeach:
        {
            auto operand = inst->getOperand(2);
            if (as<IRFunc>(operand))
            {
                // emitOperand(operand->findDecoration<IREntryPointDecoration>(),
                // getInfo(EmitOp::General));
                emitOperand(operand, getInfo(EmitOp::General));
            }
            else
            {
                SLANG_UNEXPECTED("Expected 3rd operand to be a function");
            }
            m_writer->emit("_wrapper(");
            emitOperand(inst->getOperand(0), getInfo(EmitOp::General));
            m_writer->emit(", ");
            emitOperand(inst->getOperand(1), getInfo(EmitOp::General));
            UInt argCount = inst->getOperandCount();
            for (UInt aa = 3; aa < argCount; ++aa)
            {
                m_writer->emit(", ");
                emitOperand(inst->getOperand(aa), getInfo(EmitOp::General));
            }
            m_writer->emit(")");
            break;
        }
    case kIROp_GetStringHash:
        {
            auto getStringHashInst = as<IRGetStringHash>(inst);
            auto stringLit = getStringHashInst->getStringLit();

            if (stringLit)
            {
                auto slice = stringLit->getStringSlice();
                m_writer->emit(getStableHashCode32(slice.begin(), slice.getLength()).hash);
            }
            else
            {
                // Couldn't handle
                diagnoseUnhandledInst(inst);
            }
            break;
        }
    case kIROp_Printf:
        {
            m_writer->emit("printf(");
            emitOperand(inst->getOperand(0), getInfo(EmitOp::General));
            if (inst->getOperandCount() == 2)
            {
                auto operand = inst->getOperand(1);
                if (auto makeStruct = as<IRMakeStruct>(operand))
                {
                    // Flatten the tuple resulting from the variadic pack.
                    for (UInt bb = 0; bb < makeStruct->getOperandCount(); ++bb)
                    {
                        m_writer->emit(", ");
                        emitOperand(makeStruct->getOperand(bb), getInfo(EmitOp::General));
                    }
                }
            }
            m_writer->emit(")");
            break;
        }
    case kIROp_RequirePrelude:
        {
            auto preludeTextInst = as<IRStringLit>(inst->getOperand(0));
            if (preludeTextInst)
                m_requiredPreludes.add(preludeTextInst);
            break;
        }
    case kIROp_RequireComputeDerivative:
        {
            break; // should already have been parsed and used.
        }
    case kIROp_GlobalValueRef:
        {
            emitOperand(as<IRGlobalValueRef>(inst)->getOperand(0), getInfo(EmitOp::General));
            break;
        }
    case kIROp_RequireTargetExtension:
        {
            emitRequireExtension(as<IRRequireTargetExtension>(inst));
            break;
        }
    default:
        diagnoseUnhandledInst(inst);
        break;
    }
    maybeCloseParens(needClose);
}

void CLikeSourceEmitter::emitInst(IRInst* inst)
{
    try
    {
        _emitInst(inst);
    }
    // Don't emit any context message for an explicit `AbortCompilationException`
    // because it should only happen when an error is already emitted.
    catch (const AbortCompilationException&)
    {
        throw;
    }
    catch (...)
    {
        noteInternalErrorLoc(inst->sourceLoc);
        throw;
    }
}

void CLikeSourceEmitter::_emitInst(IRInst* inst)
{
    if (shouldFoldInstIntoUseSites(inst))
    {
        return;
    }

    // Specially handle params. The issue here is around PHI nodes, and that they do not
    // have source loc information, by default, but we don't want to force outputting a #line.
    if (inst->getOp() == kIROp_Param)
    {
        m_writer->advanceToSourceLocationIfValid(inst->sourceLoc);
    }
    else
    {
        m_writer->advanceToSourceLocation(inst->sourceLoc);
    }

    if (auto coopVecType = as<IRCoopVectorType>(inst->getDataType()))
    {
        switch (inst->getOp())
        {
        case kIROp_MakeCoopVector:
            {
                emitType(coopVecType, getName(inst));
                m_writer->emit(";\n");

                auto elemCount = as<IRIntLit>(coopVecType->getOperand(1));
                IRIntegerValue elemCountValue = elemCount->getValue();
                for (IRIntegerValue i = 0; i < elemCountValue; ++i)
                {
                    m_writer->emit(getName(inst));
                    m_writer->emit(".WriteToIndex(");
                    m_writer->emit(i);
                    m_writer->emit(", ");
                    emitDereferenceOperand(inst->getOperand(i), getInfo(EmitOp::General));
                    m_writer->emit(");\n");
                }
                return;
            }
        case kIROp_Call:
            emitType(coopVecType, getName(inst));
            m_writer->emit(";\n");

            m_writer->emit(getName(inst));
            m_writer->emit(".CopyFrom(");
            emitCallExpr((IRCall*)inst, getInfo(EmitOp::General));
            m_writer->emit(");\n");
            return;
        case kIROp_Load:
            emitType(coopVecType, getName(inst));
            m_writer->emit(";\n");

            m_writer->emit(getName(inst));
            m_writer->emit(".CopyFrom(");
            emitDereferenceOperand(inst->getOperand(0), getInfo(EmitOp::General));
            m_writer->emit(");\n");
            return;
        default:
            break;
        }
    }

    switch (inst->getOp())
    {
    default:
        emitInstResultDecl(inst);
        emitInstExpr(inst, getInfo(EmitOp::General));
        m_writer->emit(";\n");
        break;

    case kIROp_DebugSource:
    case kIROp_DebugLine:
    case kIROp_DebugVar:
    case kIROp_DebugValue:
        break;

    case kIROp_Unmodified:
        break;

        // Insts that needs to be emitted as code blocks.
    case kIROp_CudaKernelLaunch:
    case kIROp_AtomicLoad:
    case kIROp_AtomicStore:
    case kIROp_AtomicInc:
    case kIROp_AtomicDec:
    case kIROp_AtomicAdd:
    case kIROp_AtomicSub:
    case kIROp_AtomicAnd:
    case kIROp_AtomicOr:
    case kIROp_AtomicXor:
    case kIROp_AtomicMin:
    case kIROp_AtomicMax:
    case kIROp_AtomicExchange:
    case kIROp_AtomicCompareExchange:
    case kIROp_StructuredBufferGetDimensions:
    case kIROp_MetalAtomicCast:
    case kIROp_MetalCastToDepthTexture:
        emitInstStmt(inst);
        break;

    case kIROp_LiveRangeStart:
    case kIROp_LiveRangeEnd:
        emitLiveness(inst);
        break;
    case kIROp_undefined:
    case kIROp_DefaultConstruct:
        {
            auto type = inst->getDataType();
            _emitInstAsDefaultInitializedVar(inst, type);
        }
        break;

    case kIROp_AllocateOpaqueHandle:
        break;
    case kIROp_Var:
        {
            auto var = cast<IRVar>(inst);
            emitVar(var);
        }
        break;

    case kIROp_Store:
        {
            auto store = cast<IRStore>(inst);
            emitStore(store);
        }
        break;

    case kIROp_Param:
        // Don't emit parameters, since they are declared as part of the function.
        break;

    case kIROp_FieldAddress:
        // skip during code emit, since it should be
        // folded into use site(s)
        break;

    case kIROp_Return:
        m_writer->emit("return");
        if (((IRReturn*)inst)->getVal()->getOp() != kIROp_VoidLit)
        {
            m_writer->emit(" ");
            emitOperand(((IRReturn*)inst)->getVal(), getInfo(EmitOp::General));
        }
        m_writer->emit(";\n");
        break;

    case kIROp_discard:
        emitInstStmt(inst);
        break;

    case kIROp_swizzleSet:
        {
            auto ii = (IRSwizzleSet*)inst;
            emitInstResultDecl(inst);
            emitOperand(inst->getOperand(0), getInfo(EmitOp::General));
            m_writer->emit(";\n");

            auto subscriptOuter = getInfo(EmitOp::General);
            auto subscriptPrec = getInfo(EmitOp::Postfix);
            bool needCloseSubscript = maybeEmitParens(subscriptOuter, subscriptPrec);

            emitOperand(inst, leftSide(subscriptOuter, subscriptPrec));
            m_writer->emit(".");
            UInt elementCount = ii->getElementCount();
            for (UInt ee = 0; ee < elementCount; ++ee)
            {
                IRInst* irElementIndex = ii->getElementIndex(ee);
                SLANG_RELEASE_ASSERT(irElementIndex->getOp() == kIROp_IntLit);
                IRConstant* irConst = (IRConstant*)irElementIndex;

                UInt elementIndex = (UInt)irConst->value.intVal;
                SLANG_RELEASE_ASSERT(elementIndex < 4);

                char const* kComponents[] = {"x", "y", "z", "w"};
                m_writer->emit(kComponents[elementIndex]);
            }
            maybeCloseParens(needCloseSubscript);

            m_writer->emit(" = ");
            emitOperand(inst->getOperand(1), getInfo(EmitOp::General));
            m_writer->emit(";\n");
        }
        break;

    case kIROp_SwizzledStore:
        {
            // cpp, cuda and wgsl targets don't support swizzle on the left handside, so we
            // have to assign the element one by one.
            if (isCPUTarget(getTargetReq()) || isCUDATarget(getTargetReq()) ||
                isWGPUTarget(getTargetReq()))
            {
                _emitSwizzleStorePerElement(inst);
            }
            else
            {

                auto subscriptOuter = getInfo(EmitOp::General);
                auto subscriptPrec = getInfo(EmitOp::Postfix);
                bool needCloseSubscript = maybeEmitParens(subscriptOuter, subscriptPrec);

                auto ii = cast<IRSwizzledStore>(inst);
                emitDereferenceOperand(ii->getDest(), leftSide(subscriptOuter, subscriptPrec));
                m_writer->emit(".");
                UInt elementCount = ii->getElementCount();
                for (UInt ee = 0; ee < elementCount; ++ee)
                {
                    IRInst* irElementIndex = ii->getElementIndex(ee);
                    SLANG_RELEASE_ASSERT(irElementIndex->getOp() == kIROp_IntLit);
                    IRConstant* irConst = (IRConstant*)irElementIndex;

                    UInt elementIndex = (UInt)irConst->value.intVal;
                    SLANG_RELEASE_ASSERT(elementIndex < 4);

                    char const* kComponents[] = {"x", "y", "z", "w"};
                    m_writer->emit(kComponents[elementIndex]);
                }
                maybeCloseParens(needCloseSubscript);

                m_writer->emit(" = ");
                emitOperand(ii->getSource(), getInfo(EmitOp::General));
                m_writer->emit(";\n");
            }
        }
        break;

    case kIROp_UpdateElement:
        {
            auto ii = (IRUpdateElement*)inst;
            auto subscriptOuter = getInfo(EmitOp::General);
            auto subscriptPrec = getInfo(EmitOp::Postfix);
            emitInstResultDecl(inst);
            if (auto arrayType = as<IRArrayType>(inst->getDataType()))
            {
                auto arraySize = as<IRIntLit>(arrayType->getElementCount());
                SLANG_RELEASE_ASSERT(arraySize);
                m_writer->emit("{");
                for (UInt i = 0; i < (UInt)arraySize->getValue(); i++)
                {
                    if (i > 0)
                        m_writer->emit(", ");
                    emitOperand(ii->getOldValue(), leftSide(subscriptOuter, subscriptPrec));
                    m_writer->emit("[");
                    m_writer->emit(i);
                    m_writer->emit("]");
                }
                m_writer->emit("}");
            }
            else
            {
                emitOperand(ii->getOldValue(), getInfo(EmitOp::General));
            }
            m_writer->emit(";\n");

            emitOperand(ii, leftSide(subscriptOuter, subscriptPrec));
            for (UInt i = 0; i < ii->getAccessKeyCount(); i++)
            {
                auto key = ii->getAccessKey(i);
                if (as<IRStructKey>(key))
                {
                    m_writer->emit(".");
                    m_writer->emit(getName(key));
                }
                else
                {
                    m_writer->emit("[");
                    emitOperand(key, getInfo(EmitOp::General));
                    m_writer->emit("]");
                }
            }
            m_writer->emit(" = ");
            emitOperand(ii->getElementValue(), getInfo(EmitOp::General));
            m_writer->emit(";\n");
        }
        break;
    case kIROp_MeshOutputSet:
        {
            auto ii = (IRMeshOutputSet*)inst;
            auto subscriptOuter = getInfo(EmitOp::General);
            auto subscriptPrec = getInfo(EmitOp::Postfix);
            emitOperand(ii->getBase(), leftSide(subscriptOuter, subscriptPrec));
            m_writer->emit("[");
            emitOperand(ii->getIndex(), getInfo(EmitOp::General));
            m_writer->emit("]");
            m_writer->emit(" = ");
            emitOperand(ii->getElementValue(), getInfo(EmitOp::General));
            m_writer->emit(";\n");
        }
        break;
    }
}

void CLikeSourceEmitter::emitStore(IRStore* store)
{
    if (store->getPrevInst() == store->getOperand(0) && store->getOperand(0)->getOp() == kIROp_Var)
    {
        // If we are storing into a `var` that is defined right before the store, we have
        // already folded the store in the initialization of the `var`, so we can skip here.
        //
        return;
    }
    _emitStoreImpl(store);
}

void CLikeSourceEmitter::_emitStoreImpl(IRStore* store)
{
    auto srcVal = store->getVal();
    auto dstPtr = store->getPtr();
    if (isPointerOfType(dstPtr->getDataType(), kIROp_CoopVectorType))
    {
        emitDereferenceOperand(dstPtr, getInfo(EmitOp::General));
        m_writer->emit(".CopyFrom(");
        emitDereferenceOperand(srcVal, getInfo(EmitOp::General));
        m_writer->emit(");\n");
    }
    else
    {
        auto prec = getInfo(EmitOp::Assign);
        emitDereferenceOperand(dstPtr, leftSide(getInfo(EmitOp::General), prec));
        m_writer->emit(" = ");
        emitOperand(srcVal, rightSide(prec, getInfo(EmitOp::General)));
        m_writer->emit(";\n");
    }
}

void CLikeSourceEmitter::_emitInstAsDefaultInitializedVar(IRInst* inst, IRType* type)
{
    emitVarKeyword(type, inst);

    emitType(type, getName(inst));

    // On targets that support empty initializers, we will emit it.
    switch (this->getTarget())
    {
    case CodeGenTarget::CPPSource:
    case CodeGenTarget::HostCPPSource:
    case CodeGenTarget::PyTorchCppBinding:
    case CodeGenTarget::CUDASource:
        m_writer->emit(" = {}");
        break;
    }
    m_writer->emit(";\n");
}

void CLikeSourceEmitter::emitSemanticsUsingVarLayout(IRVarLayout* varLayout)
{
    if (auto semanticAttr = varLayout->findAttr<IRSemanticAttr>())
    {
        // Note: We force the semantic name stored in the IR to
        // upper-case here because that is what existing Slang
        // tests had assumed and continue to rely upon.
        //
        // The original rationale for switching to uppercase was
        // canonicalization for reflection (users can't accidentally
        // write code that works for `COLOR` but not for `Color`),
        // but it would probably be more ideal for our output code
        // to give the semantic name as close to how it was originally spelled
        // spelled as possible.
        //
        // TODO: Try removing this step and fixing up the test cases
        // to see if we are happier with an approach that doesn't
        // force uppercase.
        //
        String name = semanticAttr->getName();
        name = name.toUpper();

        m_writer->emit(" : ");
        m_writer->emit(name);
        if (auto index = semanticAttr->getIndex())
        {
            m_writer->emit(index);
        }
    }
}

void CLikeSourceEmitter::emitSemanticsPrefix(IRInst* inst)
{
    emitSemanticsPrefixImpl(inst);
}

void CLikeSourceEmitter::emitSemantics(IRInst* inst, bool allowOffsetLayout)
{
    emitSemanticsImpl(inst, allowOffsetLayout);
}

void CLikeSourceEmitter::emitDecorationLayoutSemantics(
    IRInst* inst,
    char const* uniformSemanticSpelling)
{
    emitLayoutSemanticsImpl(inst, uniformSemanticSpelling, EmitLayoutSemanticOption::kPreType);
}

void CLikeSourceEmitter::emitLayoutSemantics(IRInst* inst, char const* uniformSemanticSpelling)
{
    emitLayoutSemanticsImpl(inst, uniformSemanticSpelling, EmitLayoutSemanticOption::kPostType);
}

void CLikeSourceEmitter::emitSwitchCaseSelectorsImpl(
    const SwitchRegion::Case* currentCase,
    bool isDefault)
{
    for (auto caseVal : currentCase->values)
    {
        m_writer->emit("case ");
        emitOperand(caseVal, getInfo(EmitOp::General));
        m_writer->emit(":\n");
    }
    if (isDefault)
    {
        m_writer->emit("default:\n");
    }
}

void CLikeSourceEmitter::emitRegion(Region* inRegion)
{
    // We will use a loop so that we can process sequential (simple)
    // regions iteratively rather than recursively.
    // This is effectively an emulation of tail recursion.
    Region* region = inRegion;
    while (region)
    {
        // What flavor of region are we trying to emit?
        switch (region->getFlavor())
        {
        case Region::Flavor::Simple:
            {
                // A simple region consists of a basic block followed
                // by another region.
                //
                auto simpleRegion = (SimpleRegion*)region;

                // We start by outputting all of the non-terminator
                // instructions in the block.
                //
                auto block = simpleRegion->block;
                auto terminator = block->getTerminator();
                for (auto inst = block->getFirstInst(); inst != terminator;
                     inst = inst->getNextInst())
                {
                    emitInst(inst);
                }

                // Next we have to deal with the terminator instruction
                // itself. In many cases, the terminator will have been
                // turned into a block of its own, but certain cases
                // of terminators are simple enough that we just fold
                // them into the current block.
                //
                m_writer->advanceToSourceLocation(terminator->sourceLoc);
                switch (terminator->getOp())
                {
                default:
                    // Don't do anything with the terminator, and assume
                    // its behavior has been folded into the next region.
                    break;

                case kIROp_Return:
                case kIROp_discard:
                    // For extremely simple terminators, we just handle
                    // them here, so that we don't have to allocate
                    // separate `Region`s for them.
                    emitInst(terminator);
                    break;
                }

                // If the terminator required a full region to represent
                // its behavior in a structured form, then we will move
                // along to that region now.
                //
                // We do this iteratively rather than recursively, by
                // jumping back to the top of our loop with a new
                // value for `region`.
                //
                region = simpleRegion->nextRegion;
                continue;
            }

        // Break and continue regions are trivial to handle, as long as we
        // don't need to consider multi-level break/continue (which we
        // don't for now).
        case Region::Flavor::Break:
            m_writer->emit("break;\n");
            break;
        case Region::Flavor::Continue:
            m_writer->emit("continue;\n");
            break;

        case Region::Flavor::If:
            {
                auto ifRegion = (IfRegion*)region;

                emitIfDecorationsImpl(ifRegion->ifElseInst);

                // TODO: consider simplifying the code in
                // the case where `ifRegion == null`
                // so that we output `if(!condition) { elseRegion }`
                // instead of the current `if(condition) {} else { elseRegion }`

                m_writer->emit("if(");
                emitOperand(ifRegion->getCondition(), getInfo(EmitOp::General));
                m_writer->emit(")\n{\n");
                m_writer->indent();
                emitRegion(ifRegion->thenRegion);
                m_writer->dedent();
                m_writer->emit("}\n");

                // Don't emit the `else` region if it would be empty
                //
                if (auto elseRegion = ifRegion->elseRegion)
                {
                    m_writer->emit("else\n{\n");
                    m_writer->indent();
                    emitRegion(elseRegion);
                    m_writer->dedent();
                    m_writer->emit("}\n");
                }

                // Continue with the region after the `if`.
                //
                // TODO: consider just constructing a `SimpleRegion`
                // around an `IfRegion` to handle this sequencing,
                // rather than making `IfRegion` serve as both a
                // conditional and a sequence.
                //
                region = ifRegion->nextRegion;
                continue;
            }
            break;

        case Region::Flavor::Loop:
            {
                auto loopRegion = (LoopRegion*)region;
                auto loopInst = loopRegion->loopInst;

                // If the user applied an explicit decoration to the loop,
                // to control its unrolling behavior, then pass that
                // along in the output code (if the target language
                // supports the semantics of the decoration).
                //
                if (auto loopControlDecoration =
                        loopInst->findDecoration<IRLoopControlDecoration>())
                {
                    emitLoopControlDecorationImpl(loopControlDecoration);
                }

                m_writer->emit("for(;;)\n{\n");
                m_writer->indent();
                emitRegion(loopRegion->body);
                m_writer->dedent();
                m_writer->emit("}\n");

                // Continue with the region after the loop
                region = loopRegion->nextRegion;
                continue;
            }

        case Region::Flavor::Switch:
            {
                auto switchRegion = (SwitchRegion*)region;

                emitSwitchDecorationsImpl(switchRegion->switchInst);

                // Emit the start of our statement.
                m_writer->emit("switch(");
                emitOperand(switchRegion->getCondition(), getInfo(EmitOp::General));
                m_writer->emit(")\n{\n");

                auto defaultCase = switchRegion->defaultCase;
                for (auto currentCase : switchRegion->cases)
                {
                    bool isDefault = (currentCase.Ptr() == defaultCase);
                    emitSwitchCaseSelectors(currentCase.Ptr(), isDefault);
                    m_writer->indent();
                    m_writer->emit("{\n");
                    m_writer->indent();
                    emitRegion(currentCase->body);
                    m_writer->dedent();
                    m_writer->emit("}\n");
                    m_writer->dedent();
                }

                m_writer->emit("}\n");

                // Continue with the region after the `switch`
                region = switchRegion->nextRegion;
                continue;
            }
            break;
        }
        break;
    }
}

void CLikeSourceEmitter::emitRegionTree(RegionTree* regionTree)
{
    emitRegion(regionTree->rootRegion);
}

bool CLikeSourceEmitter::isDefinition(IRFunc* func)
{
    // For now, we use a simple approach: a function is
    // a definition if it has any blocks, and a declaration otherwise.
    return func->getFirstBlock() != nullptr;
}

void CLikeSourceEmitter::emitEntryPointAttributes(
    IRFunc* irFunc,
    IREntryPointDecoration* entryPointDecor)
{
    emitEntryPointAttributesImpl(irFunc, entryPointDecor);
}

void CLikeSourceEmitter::emitFunctionBody(IRGlobalValueWithCode* code)
{
    // Compute a structured region tree that can represent
    // the control flow of our function.
    //
    RefPtr<RegionTree> regionTree = generateRegionTreeForFunc(code, getSink());

    // Now that we've computed the region tree, we have
    // an opportunity to perform some last-minute transformations
    // on the code to make sure it follows our rules.
    //
    // TODO: it would be better to do these transformations earlier,
    // so that we can, e.g., dump the final IR code *before* emission
    // starts, but that gets a bit complicated because we also want
    // to have the region tree available without having to recompute it.
    //
    // For now we are just going to do things the expedient way, but
    // eventually we should allow an IR module to have side-band
    // storage for derived structures like the region tree (and logic
    // for invalidating them when a transformation would break them).
    //
    fixValueScoping(regionTree, [this](IRInst* inst) { return shouldFoldInstIntoUseSites(inst); });

    // Now emit high-level code from that structured region tree.
    //
    emitRegionTree(regionTree);
}

void CLikeSourceEmitter::emitSimpleFuncParamImpl(IRParam* param)
{
    auto paramName = getName(param);
    auto paramType = param->getDataType();

    if (auto layoutDecoration = param->findDecoration<IRLayoutDecoration>())
    {
        auto layout = as<IRVarLayout>(layoutDecoration->getLayout());
        SLANG_ASSERT(layout);

        if (layout->usesResourceKind(LayoutResourceKind::VaryingInput) ||
            layout->usesResourceKind(LayoutResourceKind::VaryingOutput))
        {
            emitInterpolationModifiers(param, paramType, layout);
            emitMeshShaderModifiers(param);
        }
    }

    emitParamType(paramType, paramName);
    emitSemantics(param);
    emitPostDeclarationAttributesForType(paramType);
}

void CLikeSourceEmitter::emitSimpleFuncParamsImpl(IRFunc* func)
{
    m_writer->emit("(");

    auto firstParam = func->getFirstParam();
    for (auto pp = firstParam; pp; pp = pp->getNextParam())
    {
        if (pp != firstParam)
            m_writer->emit(", ");

        emitSimpleFuncParamImpl(pp);
    }

    m_writer->emit(")");
}

void CLikeSourceEmitter::emitFuncHeaderImpl(IRFunc* func)
{
    auto resultType = func->getResultType();
    auto name = getName(func);
    emitType(resultType, name);
    emitSimpleFuncParamsImpl(func);
}

void CLikeSourceEmitter::emitSimpleFuncImpl(IRFunc* func)
{

    // Deal with decorations that need
    // to be emitted as attributes
    IREntryPointDecoration* entryPointDecor = func->findDecoration<IREntryPointDecoration>();
    if (entryPointDecor)
    {
        emitEntryPointAttributes(func, entryPointDecor);
    }

    emitFunctionPreambleImpl(func);

    emitFuncDecorations(func);
    emitFuncHeader(func);
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

void CLikeSourceEmitter::emitParamTypeImpl(IRType* type, String const& name)
{
    // An `out` or `inout` parameter will have been
    // encoded as a parameter of pointer type, so
    // we need to decode that here.
    //
    if (auto outType = as<IROutType>(type))
    {
        m_writer->emit("out ");
        type = outType->getValueType();
    }
    else if (auto inOutType = as<IRInOutType>(type))
    {
        m_writer->emit("inout ");
        type = inOutType->getValueType();
    }
    else if (auto refType = as<IRRefType>(type))
    {
        // Note: There is no HLSL/GLSL equivalent for by-reference parameters,
        // so we don't actually expect to encounter these in user code.
        m_writer->emit("inout ");
        type = refType->getValueType();
    }
    else if (auto constRefType = as<IRConstRefType>(type))
    {
        type = constRefType->getValueType();
    }
    emitParamTypeModifier(type);
    emitType(type, name);
}

void CLikeSourceEmitter::emitFuncDecl(IRFunc* func)
{
    auto name = getName(func);
    emitFuncDecl(func, name);
}

void CLikeSourceEmitter::emitFuncDecl(IRFunc* func, const String& name)
{
    // We don't want to emit declarations for operations
    // that only appear in the IR as stand-ins for built-in
    // operations on that target.
    if (isTargetIntrinsic(func))
        return;

    // Finally, don't emit a declaration for an entry point,
    // because it might need meta-data attributes attached
    // to it, and the HLSL compiler will get upset if the
    // forward declaration doesn't *also* have those
    // attributes.
    if (asEntryPoint(func))
        return;


    // A function declaration doesn't have any IR basic blocks,
    // and as a result it *also* doesn't have the IR `param` instructions,
    // so we need to emit a declaration entirely from the type.

    auto funcType = func->getDataType();
    auto resultType = func->getResultType();

    emitFuncDecorations(func);
    emitType(resultType, name);

    m_writer->emit("(");
    auto paramCount = funcType->getParamCount();
    for (UInt pp = 0; pp < paramCount; ++pp)
    {
        if (pp != 0)
            m_writer->emit(", ");

        String paramName;
        paramName.append("_");
        paramName.append(Int32(pp));
        auto paramType = funcType->getParamType(pp);

        emitParamType(paramType, paramName);
    }
    m_writer->emit(");\n\n");
}

IREntryPointLayout* CLikeSourceEmitter::getEntryPointLayout(IRFunc* func)
{
    if (auto layoutDecoration = func->findDecoration<IRLayoutDecoration>())
    {
        return as<IREntryPointLayout>(layoutDecoration->getLayout());
    }
    return nullptr;
}

IREntryPointLayout* CLikeSourceEmitter::asEntryPoint(IRFunc* func)
{
    if (auto layoutDecoration = func->findDecoration<IRLayoutDecoration>())
    {
        if (auto entryPointLayout = as<IREntryPointLayout>(layoutDecoration->getLayout()))
        {
            return entryPointLayout;
        }
    }

    return nullptr;
}

bool CLikeSourceEmitter::isTargetIntrinsic(IRInst* inst)
{
    // A function is a target intrinsic if and only if
    // it has a suitable decoration marking it as a
    // target intrinsic for the current compilation target.
    //
    UnownedStringSlice intrinsicDef;
    IRInst* intrinsicInst;
    return findTargetIntrinsicDefinition(inst, intrinsicDef, intrinsicInst);
}

bool shouldWrapInExternCBlock(IRFunc* func)
{
    for (auto decor : func->getDecorations())
    {
        switch (decor->getOp())
        {
        case kIROp_ExternCDecoration:
            return true;
        }
    }
    return false;
}

void CLikeSourceEmitter::emitFunc(IRFunc* func)
{
    // Target-intrinsic functions should never be emitted
    // even if they happen to have a body.
    //
    if (isTargetIntrinsic(func))
        return;

    bool shouldCloseExternCBlock = shouldWrapInExternCBlock(func);
    if (shouldCloseExternCBlock)
    {
        // If this is a C++ `extern "C"` function, then we need to emit
        // it as a C function, since that is what the C++ compiler will
        // expect.
        //
        m_writer->emit("extern \"C\" {\n");
    }

    if (!isDefinition(func))
    {
        // This is just a function declaration,
        // and so we want to emit it as such.
        //
        emitFuncDecl(func);
    }
    else
    {
        // The common case is that what we
        // have is just an ordinary function,
        // and we can emit it as such.
        //
        emitSimpleFunc(func);
    }
    if (shouldCloseExternCBlock)
    {
        m_writer->emit("}\n");
    }
}

void CLikeSourceEmitter::emitFuncDecorationsImpl(IRFunc* func)
{
    for (auto decoration : func->getDecorations())
    {
        emitFuncDecorationImpl(decoration);
    }
}

bool CLikeSourceEmitter::tryGetIntInfo(IRType* elementType, bool& isSigned, int& bitWidth)
{
    Slang::IROp type = elementType->getOp();
    if (!(type >= kIROp_Int8Type && type <= kIROp_UInt64Type))
        return false;
    isSigned = (type >= kIROp_Int8Type && type <= kIROp_Int64Type);

    Slang::IROp stype = (isSigned) ? type : Slang::IROp(type - 4);
    bitWidth = 8 << (stype - kIROp_Int8Type);
    return true;
}

void CLikeSourceEmitter::emitVecNOrScalar(
    IRVectorType* vectorType,
    std::function<void()> emitComponentLogic)
{
    if (vectorType)
    {
        int N = int(getIntVal(vectorType->getElementCount()));
        Slang::IRType* elementType = vectorType->getElementType();

        // Special handling required for CUDA target
        if (isCUDATarget(getTargetReq()))
        {
            m_writer->emit("make_");

            switch (elementType->getOp())
            {
            case kIROp_Int8Type:
                m_writer->emit("char");
                break;
            case kIROp_Int16Type:
                m_writer->emit("short");
                break;
            case kIROp_IntType:
                m_writer->emit("int");
                break;
            case kIROp_Int64Type:
                m_writer->emit("longlong");
                break;
            case kIROp_UInt8Type:
                m_writer->emit("uchar");
                break;
            case kIROp_UInt16Type:
                m_writer->emit("ushort");
                break;
            case kIROp_UIntType:
                m_writer->emit("uint");
                break;
            case kIROp_UInt64Type:
                m_writer->emit("ulonglong");
                break;
            default:
                SLANG_ABORT_COMPILATION("Unhandled type emitting CUDA vector");
            }

            m_writer->emitRawText(std::to_string(N).c_str());
        }

        // In other languages, we can output the Slang vector type directly
        else
        {
            emitType(vectorType);
        }

        m_writer->emit("(");
        for (int i = 0; i < N; ++i)
        {
            emitType(elementType);
            m_writer->emit("(");
            emitComponentLogic();
            m_writer->emit(")");
            if (i != N - 1)
                m_writer->emit(", ");
        }
        m_writer->emit(")");
    }
    else
    {
        m_writer->emit("(");
        emitComponentLogic();
        m_writer->emit(")");
    }
}

String CLikeSourceEmitter::_emitLiteralOneWithType(int bitWidth)
{
    if (getTarget() == CodeGenTarget::WGSL)
    {
        if (bitWidth != 32)
        {
            SLANG_DIAGNOSE_UNEXPECTED(getSink(), SourceLoc(), "unexpected bit width");
            return String();
        }
        else
        {
            String one;
            one = "u32(1)";
            return one;
        }
    }

    String one;
    switch (bitWidth)
    {
    case 8:
        one = "uint8_t(1)";
        break;
    case 16:
        one = "uint16_t(1)";
        break;
    case 32:
        one = "uint32_t(1)";
        break;
    case 64:
        one = "uint64_t(1)";
        break;
    default:
        SLANG_DIAGNOSE_UNEXPECTED(getSink(), SourceLoc(), "unexpected bit width");
    }
    return one;
}

void CLikeSourceEmitter::emitBitfieldExtractImpl(IRInst* inst)
{
    // If unsigned, bfue := ((val>>off)&((1u<<bts)-1))
    // Else signed, bfse := (((val>>off)&((1u<<bts)-1))<<(nbts-bts)>>(nbts-bts));
    //
    // Note: In WGSL, the data type for bit operators are more restricted than in other languages.
    // The number of bits to shift must be a u32 or vecN<u32>, therefore we have to cast this
    // operand to u32 always. Another constraint is that for "&" and "|" operators, the operands
    // must have the same type.
    // TODO: We can consider to bring the logic to WGSLSourceEmitter::emitBitfieldExtractImpl so
    // that we don't have to have those special handling here.
    Slang::IRType* dataType = inst->getDataType();
    Slang::IRInst* val = inst->getOperand(0);
    Slang::IRInst* off = inst->getOperand(1);
    Slang::IRInst* bts = inst->getOperand(2);

    Slang::IRType* elementType = dataType;
    IRVectorType* vectorType = as<IRVectorType>(elementType);
    IRVectorType* vectorTypeForShiftNumber = nullptr;

    if (vectorType)
    {
        elementType = vectorType->getElementType();

        if (getTarget() == CodeGenTarget::WGSL)
        {
            IRBuilder builder(elementType);
            vectorTypeForShiftNumber =
                builder.getVectorType(builder.getUIntType(), vectorType->getElementCount());
        }
        else
        {
            vectorTypeForShiftNumber = vectorType;
        }
    }

    bool isSigned;
    int bitWidth;
    if (!tryGetIntInfo(elementType, isSigned, bitWidth))
    {
        SLANG_DIAGNOSE_UNEXPECTED(
            getSink(),
            SourceLoc(),
            "non-integer element type given to bitfieldExtract");
        return;
    }

    String one = _emitLiteralOneWithType(bitWidth);

    // Emit open paren and type cast for later sign extension
    if (isSigned)
    {
        m_writer->emit("(");
        emitType(inst->getDataType());
        m_writer->emit("(");
    }

    // Emit bitfield extraction ( (val >> off) & ((1u << bts) - 1) )
    m_writer->emit("(");

    // In WGSL, "&" operator requires the operands to have the same type, since the
    // right operand '((1u << bts) - 1)' is known to be u32, we need to cast the left operand to
    // u32.
    if (getTarget() == CodeGenTarget::WGSL)
    {
        (vectorTypeForShiftNumber != nullptr) ? emitType(vectorTypeForShiftNumber)
                                              : m_writer->emit("u32");
    }

    m_writer->emit("(");

    emitOperand(val, getInfo(EmitOp::General));
    m_writer->emit(">>");
    emitVecNOrScalar(
        vectorTypeForShiftNumber,
        [&]() { emitOperand(off, getInfo(EmitOp::General)); });

    m_writer->emit(")&(");
    emitVecNOrScalar(
        vectorTypeForShiftNumber,
        [&]()
        {
            m_writer->emit("((" + one + "<<");
            emitOperand(bts, getInfo(EmitOp::General));
            m_writer->emit(")-" + one + ")");
        });
    m_writer->emit("))");

    // Emit sign extension logic
    // ( type(bitfield << (numBits - bts) ) >> (numBits - bts) )
    //           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    if (isSigned)
    {
        m_writer->emit("<<");
        emitVecNOrScalar(
            vectorTypeForShiftNumber,
            [&]()
            {
                m_writer->emit("(");
                m_writer->emit(bitWidth);
                m_writer->emit("-");
                emitOperand(bts, getInfo(EmitOp::General));
                m_writer->emit(")");
            });
        m_writer->emit(")>>");
        emitVecNOrScalar(
            vectorTypeForShiftNumber,
            [&]()
            {
                m_writer->emit("(");
                m_writer->emit(bitWidth);
                m_writer->emit("-");
                emitOperand(bts, getInfo(EmitOp::General));
                m_writer->emit(")");
            });
        m_writer->emit(")");
    }
}

void CLikeSourceEmitter::emitBitfieldInsertImpl(IRInst* inst)
{
    // uint clearMask = ~(((1u << bits) - 1u) << offset);
    // uint clearedBase = base & clearMask;
    // uint maskedInsert = (insert & ((1u << bits) - 1u)) << offset;
    // BitfieldInsert := T(uint(clearedBase) | uint(maskedInsert));
    Slang::IRType* dataType = inst->getDataType();
    Slang::IRInst* base = inst->getOperand(0);
    Slang::IRInst* insert = inst->getOperand(1);
    Slang::IRInst* off = inst->getOperand(2);
    Slang::IRInst* bts = inst->getOperand(3);


    Slang::IRType* elementType = dataType;
    IRVectorType* vectorType = as<IRVectorType>(elementType);
    IRVectorType* vectorTypeForShiftNumber = nullptr;

    if (vectorType)
    {
        elementType = vectorType->getElementType();

        if (getTarget() == CodeGenTarget::WGSL)
        {
            IRBuilder builder(elementType);
            vectorTypeForShiftNumber =
                builder.getVectorType(builder.getUIntType(), vectorType->getElementCount());
        }
        else
        {
            vectorTypeForShiftNumber = vectorType;
        }
    }

    bool isSigned;
    int bitWidth;
    if (!tryGetIntInfo(elementType, isSigned, bitWidth))
    {
        SLANG_DIAGNOSE_UNEXPECTED(
            getSink(),
            SourceLoc(),
            "non-integer element type given to bitfieldInsert");
        return;
    }

    String one = _emitLiteralOneWithType(bitWidth);

    if (isSigned)
    {
        emitType(inst->getDataType());
        m_writer->emit("(");
    }
    m_writer->emit("(");

    // emit clearedBase := uint( base & ~( ((1u << bts) - 1u) << off ) )

    // In WGSL, "&" operator requires the operands to have the same type, since the
    // right operand '~( ((1u << bts) - 1u) << off )' is known to be u32, we need to
    // cast the left operand to u32.
    if (getTarget() == CodeGenTarget::WGSL)
    {
        (vectorTypeForShiftNumber != nullptr) ? emitType(vectorTypeForShiftNumber)
                                              : m_writer->emit("u32");
    }

    m_writer->emit("(");
    emitOperand(base, getInfo(EmitOp::General));
    m_writer->emit(")");

    m_writer->emit("&");
    emitVecNOrScalar(
        vectorTypeForShiftNumber,
        [&]()
        {
            m_writer->emit("~(((" + one + "<<");
            emitOperand(bts, getInfo(EmitOp::General));

            m_writer->emit(")-" + one + ")<<");

            emitOperand(off, getInfo(EmitOp::General));
            m_writer->emit(")");
        });

    // bitwise or clearedBase with maskedInsert
    m_writer->emit(")|(");

    // Emit maskedInsert := ((insert & ((1u << bits) - 1u)) << offset);

    // - first emit mask := (insert & ((1u << bits) - 1u))
    m_writer->emit("(");

    // For the same reason as above, we need to cast the left operand to u32 for WGSL target.
    if (getTarget() == CodeGenTarget::WGSL)
    {
        (vectorTypeForShiftNumber != nullptr) ? emitType(vectorTypeForShiftNumber)
                                              : m_writer->emit("u32");
    }
    m_writer->emit("(");
    emitOperand(insert, getInfo(EmitOp::General));
    m_writer->emit(")");

    m_writer->emit("&");
    emitVecNOrScalar(
        vectorTypeForShiftNumber,
        [&]()
        {
            m_writer->emit("(" + one + "<<");
            emitOperand(bts, getInfo(EmitOp::General));
            m_writer->emit(")-" + one);
        });
    m_writer->emit(")");

    // then emit shift := << offset
    m_writer->emit("<<");
    emitVecNOrScalar(
        vectorTypeForShiftNumber,
        [&]() { emitOperand(off, getInfo(EmitOp::General)); });
    m_writer->emit(")");

    if (isSigned)
    {
        m_writer->emit(")");
    }
}

void CLikeSourceEmitter::emitStruct(IRStructType* structType)
{
    ensureTypePrelude(structType);

    // If the selected `struct` type is actually an intrinsic
    // on our target, then we don't want to emit anything at all.
    if (isTargetIntrinsic(structType))
    {
        return;
    }

    m_writer->emit("struct ");

    emitPostKeywordTypeAttributes(structType);

    m_writer->emit(getName(structType));

    emitStructDeclarationsBlock(structType, false);
    m_writer->emit(";\n\n");
}

void CLikeSourceEmitter::emitStructDeclarationSeparatorImpl()
{
    m_writer->emit(";");
}

void CLikeSourceEmitter::emitStructDeclarationsBlock(
    IRStructType* structType,
    bool allowOffsetLayout)
{
    m_writer->emit("\n{\n");
    m_writer->indent();

    for (auto ff : structType->getFields())
    {
        auto fieldKey = ff->getKey();
        auto fieldType = ff->getFieldType();

        // Filter out fields with `void` type that might
        // have been introduced by legalization.
        if (as<IRVoidType>(fieldType))
            continue;

        // Note: GLSL doesn't support interpolation modifiers on `struct` fields
        if (getSourceLanguage() != SourceLanguage::GLSL)
        {
            emitInterpolationModifiers(fieldKey, fieldType, nullptr);
        }

        if (allowOffsetLayout)
        {
            if (auto packOffsetDecoration = fieldKey->findDecoration<IRPackOffsetDecoration>())
            {
                emitPackOffsetModifier(fieldKey, fieldType, packOffsetDecoration);
            }
        }
        emitSemanticsPrefix(fieldKey);
        emitStructFieldAttributes(structType, ff, allowOffsetLayout);
        emitMemoryQualifiers(fieldKey);
        emitType(fieldType, getName(fieldKey));
        emitSemantics(fieldKey, allowOffsetLayout);
        emitPostDeclarationAttributesForType(fieldType);
        emitStructDeclarationSeparator();
        m_writer->emit("\n");
    }

    m_writer->dedent();
    m_writer->emit("}");
}

void CLikeSourceEmitter::emitClass(IRClassType* classType)
{
    ensureTypePrelude(classType);

    // If the selected `class` type is actually an intrinsic
    // on our target, then we don't want to emit anything at all.
    if (isTargetIntrinsic(classType))
    {
        return;
    }
    List<IRWitnessTable*> comWitnessTables;
    for (auto child : classType->getDecorations())
    {
        if (auto decoration = as<IRCOMWitnessDecoration>(child))
        {
            comWitnessTables.add(cast<IRWitnessTable>(decoration->getWitnessTable()));
        }
    }
    m_writer->emit("class ");

    emitPostKeywordTypeAttributes(classType);

    m_writer->emit(getName(classType));
    if (comWitnessTables.getCount() == 0)
    {
        m_writer->emit(" : public RefObject");
    }
    else
    {
        m_writer->emit(" : public ComObject");
        for (auto wt : comWitnessTables)
        {
            m_writer->emit(", public ");
            m_writer->emit(getName(wt->getConformanceType()));
        }
    }
    m_writer->emit("\n{\n");
    m_writer->emit("public:\n");
    m_writer->indent();

    if (comWitnessTables.getCount())
    {
        m_writer->emit("SLANG_COM_OBJECT_IUNKNOWN_ALL\n");
        m_writer->emit("void* getInterface(const Guid & uuid)\n{\n");
        m_writer->indent();
        m_writer->emit("if (uuid == ISlangUnknown::getTypeGuid()) return "
                       "static_cast<ISlangUnknown*>(this);\n");
        for (auto wt : comWitnessTables)
        {
            auto interfaceName = getName(wt->getConformanceType());
            m_writer->emit("if (uuid == ");
            m_writer->emit(interfaceName);
            m_writer->emit("::getTypeGuid())\n");
            m_writer->indent();
            m_writer->emit("return static_cast<");
            m_writer->emit(interfaceName);
            m_writer->emit("*>(this);\n");
            m_writer->dedent();
        }
        m_writer->emit("return nullptr;\n");
        m_writer->dedent();
        m_writer->emit("}\n");
    }

    for (auto ff : classType->getFields())
    {
        auto fieldKey = ff->getKey();
        auto fieldType = ff->getFieldType();

        // Filter out fields with `void` type that might
        // have been introduced by legalization.
        if (as<IRVoidType>(fieldType))
            continue;

        emitInterpolationModifiers(fieldKey, fieldType, nullptr);

        emitType(fieldType, getName(fieldKey));
        emitSemantics(fieldKey);
        emitPostDeclarationAttributesForType(fieldType);
        m_writer->emit(";\n");
    }

    // Emit COM method declarations.
    for (auto wt : comWitnessTables)
    {
        for (auto wtEntry : wt->getChildren())
        {
            auto req = as<IRWitnessTableEntry>(wtEntry);
            if (!req)
                continue;
            auto func = as<IRFunc>(req->getSatisfyingVal());
            if (!func)
                continue;
            m_writer->emit("virtual SLANG_NO_THROW ");
            emitType(func->getResultType(), "SLANG_MCALL " + getName(req->getRequirementKey()));
            m_writer->emit("(");
            auto param = func->getFirstParam();
            param = param->getNextParam();
            for (; param; param = param->getNextParam())
            {
                emitParamType(param->getFullType(), getName(param));
            }
            m_writer->emit(") override;\n");
        }
    }

    m_writer->dedent();
    m_writer->emit("};\n\n");
}

void CLikeSourceEmitter::emitInterpolationModifiers(
    IRInst* varInst,
    IRType* valueType,
    IRVarLayout* layout)
{
    emitInterpolationModifiersImpl(varInst, valueType, layout);
}

void CLikeSourceEmitter::emitMeshShaderModifiers(IRInst* varInst)
{
    emitMeshShaderModifiersImpl(varInst);
}

/// Emit modifiers that should apply even for a declaration of an SSA temporary.
void CLikeSourceEmitter::emitTempModifiers(IRInst* temp)
{
    if (temp->findDecoration<IRPreciseDecoration>())
    {
        m_writer->emit("precise ");
    }
}

void CLikeSourceEmitter::emitVarModifiers(IRVarLayout* layout, IRInst* varDecl, IRType* varType)
{
    // TODO(JS): We could push all of this onto the target impls, and then not need so many virtual
    // hooks.
    emitVarDecorationsImpl(varDecl);
    emitTempModifiers(varDecl);

    if (!layout)
        return;

    emitMatrixLayoutModifiersImpl(varType);

    // Target specific modifier output
    emitImageFormatModifierImpl(varDecl, varType);

    if (layout->usesResourceKind(LayoutResourceKind::VaryingInput) ||
        layout->usesResourceKind(LayoutResourceKind::VaryingOutput))
    {
        emitInterpolationModifiers(varDecl, varType, layout);
        emitMeshShaderModifiers(varDecl);
    }

    // Output target specific qualifiers
    emitLayoutQualifiersImpl(layout);
}

void CLikeSourceEmitter::emitArrayBrackets(IRType* inType)
{
    // A declaration may require zero, one, or
    // more array brackets. When writing out array
    // brackets from left to right, they represent
    // the structure of the type from the "outside"
    // in (that is, if we have a 5-element array of
    // 3-element arrays we should output `[5][3]`),
    // because of C-style declarator rules.
    //
    // This conveniently means that we can print
    // out all the array brackets with a looping
    // rather than a recursive structure.
    //
    // We will peel the input type like an onion,
    // looking at one layer at a time until we
    // reach a non-array type in the middle.
    //
    IRType* type = inType;
    for (;;)
    {
        if (auto arrayType = as<IRArrayType>(type))
        {
            m_writer->emit("[");
            emitVal(arrayType->getElementCount(), getInfo(EmitOp::General));
            m_writer->emit("]");

            // Continue looping on the next layer in.
            //
            type = arrayType->getElementType();
        }
        else if (auto unsizedArrayType = as<IRUnsizedArrayType>(type))
        {
            m_writer->emit("[]");

            // Continue looping on the next layer in.
            //
            type = unsizedArrayType->getElementType();
        }
        else
        {
            // This layer wasn't an array, so we are done.
            //
            return;
        }
    }
}

void CLikeSourceEmitter::emitParameterGroup(
    IRGlobalParam* varDecl,
    IRUniformParameterGroupType* type)
{
    emitParameterGroupImpl(varDecl, type);
}

void CLikeSourceEmitter::emitVarKeywordImpl(IRType* /* type */, IRInst* /* varDecl */) {}

void CLikeSourceEmitter::emitVar(IRVar* varDecl)
{
    auto allocatedType = varDecl->getDataType();
    auto varType = allocatedType->getValueType();
    //        auto addressSpace = allocatedType->getAddressSpace();

#if 0
    switch( varType->op )
    {
    case kIROp_ConstantBufferType:
    case kIROp_TextureBufferType:
        emitIRParameterGroup(ctx, varDecl, (IRUniformBufferType*) varType);
        return;

    default:
        break;
    }
#endif

    // Need to emit appropriate modifiers here.

    auto layout = getVarLayout(varDecl);

    emitVarModifiers(layout, varDecl, varType);

#if 0
    switch (addressSpace)
    {
    default:
        break;

    case kIRAddressSpace_GroupShared:
        emit("groupshared ");
        break;
    }
#endif
    emitRateQualifiersAndAddressSpace(varDecl);

    emitVarKeyword(varType, varDecl);

    emitType(varType, getName(varDecl));

    emitSemantics(varDecl);
    emitLayoutSemantics(varDecl, "register");
    emitPostDeclarationAttributesForType(varType);

    // TODO: ideally this logic should scan ahead to see if it can find a `store`
    // instruction that writes to the `var`, within the same block, such that all
    // of the intervening instructions are safe to fold.
    //
    if (auto store = as<IRStore>(varDecl->getNextInst()))
    {
        if (store->getPtr() == varDecl)
        {
            const bool isCoopVectorType = varType->getOp() == kIROp_CoopVectorType;
            if (isCoopVectorType && store->getVal()->getOp() == kIROp_Load)
            {
                m_writer->emit(";\n");
                m_writer->emit(getName(varDecl));
                m_writer->emit(".CopyFrom(");
                emitDereferenceOperand(store->getVal()->getOperand(0), getInfo(EmitOp::General));
                m_writer->emit(")");
            }
            else if (isCoopVectorType && store->getVal()->getOp() == kIROp_Call)
            {
                m_writer->emit(";\n");
                m_writer->emit(getName(varDecl));
                m_writer->emit(".CopyFrom(");
                emitCallExpr((IRCall*)store->getVal(), getInfo(EmitOp::General));
                m_writer->emit(")");
            }
            else if (isCoopVectorType && store->getVal()->getOp() == kIROp_MakeCoopVector)
            {
                auto coopVecType = as<IRCoopVectorType>(store->getVal()->getDataType());
                auto elemCount = as<IRIntLit>(coopVecType->getOperand(1));
                IRIntegerValue elemCountValue = elemCount->getValue();
                for (IRIntegerValue i = 0; i < elemCountValue; ++i)
                {
                    m_writer->emit(";\n");
                    m_writer->emit(getName(varDecl));
                    m_writer->emit(".WriteToIndex(");
                    m_writer->emit(i);
                    m_writer->emit(", ");
                    emitDereferenceOperand(
                        store->getVal()->getOperand(i),
                        getInfo(EmitOp::General));
                    m_writer->emit(")");
                }
            }
            else
            {
                _emitInstAsVarInitializerImpl(store->getVal());
            }
        }
    }

    m_writer->emit(";\n");
}

void CLikeSourceEmitter::_emitInstAsVarInitializerImpl(IRInst* inst)
{
    m_writer->emit(" = ");
    emitOperand(inst, getInfo(EmitOp::General));
}

bool _isFoldableValue(IRInst* val)
{
    if (val->getParent() && val->getParent()->getOp() == kIROp_Module)
        return true;

    switch (val->getOp())
    {
    case kIROp_MakeArray:
    case kIROp_MakeVector:
    case kIROp_MakeMatrix:
    case kIROp_MakeStruct:
    case kIROp_MakeVectorFromScalar:
    case kIROp_MakeArrayFromElement:
    case kIROp_MakeMatrixFromScalar:
    case kIROp_CastIntToFloat:
    case kIROp_CastFloatToInt:
    case kIROp_IntCast:
    case kIROp_FloatCast:
        {
            for (UInt i = 0; i < val->getOperandCount(); i++)
                if (!_isFoldableValue(val->getOperand(i)))
                    return false;
            return true;
        }
    default:
        return false;
    }
}

void CLikeSourceEmitter::emitGlobalVar(IRGlobalVar* varDecl)
{
    auto allocatedType = varDecl->getDataType();
    auto varType = allocatedType->getValueType();

    String initFuncName;
    IRInst* initVal = nullptr;

    if (auto firstBlock = varDecl->getFirstBlock())
    {
        // A global variable with code means it has an initializer
        // associated with it.

        if (auto returnInst = as<IRReturn>(firstBlock->getTerminator()))
        {
            // If the initializer can be conveniently emitted as an
            // expression, we will do that.
            if (_isFoldableValue(returnInst->getVal()))
            {
                initVal = returnInst->getVal();
            }
        }
        if (!initVal)
        {
            emitFunctionPreambleImpl(varDecl);

            // If we can't emit the initializer as an expression,
            // we will emit it as a separate function.
            //
            // TODO: the C language does not allow defining
            // functions that return arrays, so if we have an
            // array type here, we are going to generate invalid
            // code.

            initFuncName = getName(varDecl);
            initFuncName.append("_init");

            m_writer->emit("\n");
            emitType(varType, initFuncName);
            m_writer->emit("()\n{\n");
            m_writer->indent();
            emitFunctionBody(varDecl);
            m_writer->dedent();
            m_writer->emit("}\n");
        }
    }

    // An ordinary global variable won't have a layout
    // associated with it, since it is not a shader
    // parameter.
    //
    SLANG_ASSERT(!getVarLayout(varDecl));
    IRVarLayout* layout = nullptr;

    // An ordinary global variable (which is not a
    // shader parameter) may need special
    // modifiers to indicate it as such.
    //
    switch (getSourceLanguage())
    {
    case SourceLanguage::HLSL:
        // HLSL requires the `static` modifier on any
        // global variables; otherwise they are assumed
        // to be uniforms.
        m_writer->emit("static ");
        break;

    default:
        break;
    }

    emitVarModifiers(layout, varDecl, varType);

    emitRateQualifiersAndAddressSpace(varDecl);
    emitVarKeyword(varType, varDecl);
    emitType(varType, getName(varDecl));

    // TODO: These shouldn't be needed for ordinary
    // global variables.
    //
    emitSemantics(varDecl);
    emitLayoutSemantics(varDecl, "register");

    if (varDecl->getFirstBlock())
    {
        m_writer->emit(" = ");
        if (initVal)
        {
            emitInstExpr(initVal, EmitOpInfo());
        }
        else
        {
            m_writer->emit(initFuncName);
            m_writer->emit("()");
        }
    }
    m_writer->emit(";\n\n");
}

void CLikeSourceEmitter::emitGlobalParam(IRGlobalParam* varDecl)
{
    auto rawType = varDecl->getDataType();

    auto varType = rawType;
    if (auto ptrType = as<IRPtrTypeBase>(varType))
    {
        switch (ptrType->getAddressSpace())
        {
        case AddressSpace::Input:
        case AddressSpace::Output:
        case AddressSpace::BuiltinInput:
        case AddressSpace::BuiltinOutput:
            varType = ptrType->getValueType();
            break;
        default:
            if (as<IROutTypeBase>(ptrType))
                varType = ptrType->getValueType();
            break;
        }
    }
    if (as<IRVoidType>(varType))
        return;

    emitMemoryQualifiers(varDecl);

    // When a global shader parameter represents a "parameter group"
    // (either a constant buffer or a parameter block with non-resource
    // data in it), we will prefer to emit it as an ordinary `cbuffer`
    // declaration or `uniform` block, even when emitting HLSL for
    // D3D profiles that support the explicit `ConstantBuffer<T>` type.
    //
    // Alternatively, we could make this choice based on profile, and
    // prefer `ConstantBuffer<T>` on profiles that support it and/or when
    // the input code used that syntax.
    //
    if (auto paramBlockType = as<IRUniformParameterGroupType>(varType))
    {
        emitParameterGroup(varDecl, paramBlockType);
        return;
    }

    // Try target specific ways to emit.
    if (tryEmitGlobalParamImpl(varDecl, varType))
    {
        return;
    }

    // Need to emit appropriate modifiers here.

    // We expect/require all shader parameters to
    // have some kind of layout information associated with them.
    //
    auto layout = getVarLayout(varDecl);
    SLANG_ASSERT(layout);

    emitVarModifiers(layout, varDecl, varType);

    emitDecorationLayoutSemantics(varDecl, "register");

    emitRateQualifiersAndAddressSpace(varDecl);
    emitVarKeyword(varType, varDecl);
    emitType(varType, getName(varDecl));

    emitSemantics(varDecl);

    emitLayoutSemantics(varDecl, "register");

    // If the parameter has a default value, we may need to emit it.
    emitGlobalParamDefaultVal(varDecl);

    // A shader parameter cannot have an initializer,
    // so we do need to consider emitting one here.

    m_writer->emit(";\n\n");
}

void CLikeSourceEmitter::emitGlobalInst(IRInst* inst)
{
    emitGlobalInstImpl(inst);
}

static bool _shouldSkipFuncEmit(IRInst* func)
{
    // Skip emitting a func if it is a COM interface wrapper implementation and used
    // only by the witness table. We will emit this func differently than normal funcs
    // and this is handled by `emitComWitnessTable`.

    if (func->hasMoreThanOneUse())
        return false;
    if (func->firstUse)
    {
        if (auto entry = as<IRWitnessTableEntry>(func->firstUse->getUser()))
        {
            if (auto table = as<IRWitnessTable>(entry->getParent()))
            {
                if (auto interfaceType = table->getConformanceType())
                {
                    if (interfaceType->findDecoration<IRComInterfaceDecoration>())
                    {
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

void CLikeSourceEmitter::emitGlobalInstImpl(IRInst* inst)
{
    m_writer->advanceToSourceLocation(inst->sourceLoc);

    // Deal with required features/capabilities of the global inst
    //
    handleRequiredCapabilitiesImpl(inst);

    switch (inst->getOp())
    {
    case kIROp_GlobalHashedStringLiterals:
        /* Don't need to to output anything for this instruction - it's used for reflecting
        string literals that are hashed with 'getStringHash' */
        break;

    case kIROp_InterfaceRequirementEntry:
        // Don't emit anything for interface requirement at global level.
        // They are handled in `emitInterface`.
        break;

    case kIROp_Func:
        if (!_shouldSkipFuncEmit(inst))
        {
            emitFunc((IRFunc*)inst);
        }
        break;

    case kIROp_GlobalVar:
        emitGlobalVar((IRGlobalVar*)inst);
        break;

    case kIROp_GlobalParam:
        emitGlobalParam((IRGlobalParam*)inst);
        break;

    case kIROp_Var:
        emitVar((IRVar*)inst);
        break;

    case kIROp_StructType:
        emitStruct(cast<IRStructType>(inst));
        break;
    case kIROp_ClassType:
        emitClass(cast<IRClassType>(inst));
        break;
    case kIROp_InterfaceType:
        emitInterface(cast<IRInterfaceType>(inst));
        break;
    case kIROp_WitnessTable:
        emitWitnessTable(cast<IRWitnessTable>(inst));
        break;

    case kIROp_RTTIObject:
        emitRTTIObject(cast<IRRTTIObject>(inst));
        break;

    default:
        // We have an "ordinary" instruction at the global
        // scope, and we should therefore emit it using the
        // rules for other ordinary instructions.
        //
        // Such an instruction represents (part of) the value
        // for a global constants.
        //
        emitInst(inst);
        break;
    }
}

void CLikeSourceEmitter::ensureInstOperand(
    ComputeEmitActionsContext* ctx,
    IRInst* inst,
    EmitAction::Level requiredLevel)
{
    if (!inst)
        return;

    if (inst->getParent() == ctx->moduleInst)
    {
        ensureGlobalInst(ctx, inst, requiredLevel);
    }
}

void CLikeSourceEmitter::ensureInstOperandsRec(ComputeEmitActionsContext* ctx, IRInst* inst)
{
    ensureInstOperand(ctx, inst->getFullType());

    UInt operandCount = inst->operandCount;
    auto requiredLevel = EmitAction::Definition;
    switch (inst->getOp())
    {
    case kIROp_PtrType:
        {
            auto ptrType = static_cast<IRPtrType*>(inst);
            auto valueType = ptrType->getValueType();

            if (ctx->openInsts.contains(valueType))
            {
                requiredLevel = EmitAction::ForwardDeclaration;
            }
            else
            {
                requiredLevel = EmitAction::Definition;
            }
            break;
        }
    case kIROp_NativePtrType:
        requiredLevel = EmitAction::ForwardDeclaration;
        break;
    case kIROp_LookupWitness:
    case kIROp_FieldExtract:
    case kIROp_FieldAddress:
        {
            auto opType = inst->getOperand(0)->getDataType();
            if (auto nativePtrType = as<IRNativePtrType>(opType))
            {
                ensureInstOperand(ctx, nativePtrType->getValueType(), requiredLevel);
            }
            break;
        }
    default:
        break;
    }

    if (auto comWitnessDecoration = as<IRCOMWitnessDecoration>(inst))
    {
        // A COMWitnessDecoration marks the interface inheritance of a class.
        // We need to make sure the implemented interface is emited before the class.
        // The witness table itself doesn't matter.
        if (auto witnessTable = as<IRWitnessTable>(comWitnessDecoration->getWitnessTable()))
        {
            ensureInstOperand(ctx, witnessTable->getConformanceType(), requiredLevel);
        }
        requiredLevel = EmitAction::ForwardDeclaration;
    }

    for (UInt ii = 0; ii < operandCount; ++ii)
    {
        // TODO: there are some special cases we can add here,
        // to avoid outputting full definitions in cases that
        // can get by with forward declarations.
        //
        // For example, true pointer types should (in principle)
        // only need the type they point to to be forward-declared.
        // Similarly, a `call` instruction only needs the callee
        // to be forward-declared, etc.

        ensureInstOperand(ctx, inst->getOperand(ii), requiredLevel);
    }

    for (auto child : inst->getDecorationsAndChildren())
    {
        ensureInstOperandsRec(ctx, child);
    }
}

void CLikeSourceEmitter::ensureGlobalInst(
    ComputeEmitActionsContext* ctx,
    IRInst* inst,
    EmitAction::Level requiredLevel)
{
    // Skip certain instructions that don't affect output.
    switch (inst->getOp())
    {
    case kIROp_Generic:
        return;
    case kIROp_ThisType:
        return;
    default:
        break;
    }
    if (as<IRBasicType>(inst))
        return;
    if (as<IRPtrLit>(inst))
        return;
    // Certain inst ops will always emit as definition.
    switch (inst->getOp())
    {
    case kIROp_NativePtrType:
        // Pointer type will have their value type emited as forward declaration,
        // but the pointer type itself should be considered emitted as definition.
        requiredLevel = EmitAction::Level::Definition;
        break;
    default:
        break;
    }

    // Have we already processed this instruction?
    EmitAction::Level existingLevel;
    if (ctx->mapInstToLevel.tryGetValue(inst, existingLevel))
    {
        // If we've already emitted it suitably,
        // then don't worry about it.
        if (existingLevel >= requiredLevel)
            return;
    }

    EmitAction action;
    action.level = requiredLevel;
    action.inst = inst;

    if (requiredLevel == EmitAction::Level::Definition)
    {
        if (ctx->openInsts.contains(inst))
        {
            SLANG_UNEXPECTED("circularity during codegen");
            return;
        }

        ctx->openInsts.add(inst);

        ensureInstOperandsRec(ctx, inst);

        ctx->openInsts.remove(inst);
    }

    ctx->mapInstToLevel[inst] = requiredLevel;

    // Skip instructions that don't correspond to an independent entity in output.
    switch (inst->getOp())
    {
    case kIROp_InterfaceRequirementEntry:
        {
            return;
        }

    default:
        break;
    }
    ctx->actions->add(action);
}

void CLikeSourceEmitter::computeEmitActions(IRModule* module, List<EmitAction>& ioActions)
{
    ComputeEmitActionsContext ctx;
    ctx.moduleInst = module->getModuleInst();
    ctx.actions = &ioActions;
    ctx.openInsts = InstHashSet(module);

    for (auto inst : module->getGlobalInsts())
    {
        // Emit all resource-typed objects first. This is to avoid an odd scenario in HLSL
        // where not using a resource type in a resource definition before the same type
        // is used for a function parameter causes HLSL to complain about an 'incomplete type'
        //
        if (isResourceType(inst->getDataType()))
        {
            ensureGlobalInst(&ctx, inst, EmitAction::Level::Definition);
        }
    }

    for (auto inst : module->getGlobalInsts())
    {
        // After emitting all structure types we need to emit all raytracing objects to
        // ensure they are emitted before the layout location is referenced, otherwise,
        // this can be a compile error if layout is emitted after location is referenced
        // this is required since in GLSL, it is likley in a programs life time a raytracing
        // object will never be referenced by name
        for (auto dec : inst->getDecorations())
        {
            switch (dec->getOp())
            {
            case kIROp_VulkanRayPayloadDecoration:
            case kIROp_VulkanRayPayloadInDecoration:
            case kIROp_VulkanHitObjectAttributesDecoration:
            case kIROp_VulkanCallablePayloadDecoration:
            case kIROp_VulkanCallablePayloadInDecoration:
            case kIROp_VulkanHitAttributesDecoration:
                ensureGlobalInst(&ctx, inst, EmitAction::Level::Definition);
            };
        }
    }
    for (auto inst : module->getGlobalInsts())
    {
        if (as<IRType>(inst))
        {
            // Don't emit a type unless it is actually used or is marked exported.
            if (!inst->findDecoration<IRHLSLExportDecoration>())
                continue;
        }

        // Skip resource types in this pass.
        if (isResourceType(inst->getDataType()))
            continue;

        ensureGlobalInst(&ctx, inst, EmitAction::Level::Definition);
    }
}

void CLikeSourceEmitter::emitForwardDeclaration(IRInst* inst)
{
    switch (inst->getOp())
    {
    case kIROp_Func:
        emitFuncDecl(cast<IRFunc>(inst));
        break;
    case kIROp_StructType:
        m_writer->emit("struct ");
        m_writer->emit(getName(inst));
        m_writer->emit(";\n");
        break;
    case kIROp_InterfaceType:
        {
            if (inst->findDecoration<IRComInterfaceDecoration>())
            {
                m_writer->emit("struct ");
                m_writer->emit(getName(inst));
                m_writer->emit(";\n");
            }
            break;
        }
    default:
        SLANG_UNREACHABLE("emit forward declaration");
    }
}

void CLikeSourceEmitter::executeEmitActions(List<EmitAction> const& actions)
{
    for (auto action : actions)
    {
        switch (action.level)
        {
        case EmitAction::Level::ForwardDeclaration:
            emitForwardDeclaration(action.inst);
            break;

        case EmitAction::Level::Definition:
            emitGlobalInst(action.inst);
            break;
        }
    }
}

void CLikeSourceEmitter::emitModuleImpl(IRModule* module, DiagnosticSink* sink)
{
    // The IR will usually come in an order that respects
    // dependencies between global declarations, but this
    // isn't guaranteed, so we need to be careful about
    // the order in which we emit things.

    SLANG_UNUSED(sink);

    List<EmitAction> actions;

    beforeComputeEmitActions(module);
    computeEmitActions(module, actions);
    executeEmitActions(actions);
}

void CLikeSourceEmitter::ensurePrelude(const char* preludeText)
{
    IRStringLit* stringLit;
    if (!m_builtinPreludes.tryGetValue(preludeText, stringLit))
    {
        IRBuilder builder(m_irModule);
        stringLit = builder.getStringValue(UnownedStringSlice(preludeText));
        m_builtinPreludes[preludeText] = stringLit;
    }
    m_requiredPreludes.add(stringLit);
}
} // namespace Slang
