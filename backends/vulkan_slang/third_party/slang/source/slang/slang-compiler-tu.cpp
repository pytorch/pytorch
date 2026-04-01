// slang-compiler-tu.cpp: Compiles translation units to target language
// and emit precompiled blobs into IR

#include "../core/slang-basic.h"
#include "slang-capability.h"
#include "slang-check-impl.h"
#include "slang-compiler.h"
#include "slang-ir-insts.h"
#include "slang-ir-util.h"

namespace Slang
{
// Only attempt to precompile functions:
// 1) With function bodies (not just empty decls)
// 2) Not marked with unsafeForceInlineDecoration
// 3) Have a simple HLSL data type as the return or parameter type
static bool attemptPrecompiledExport(IRInst* inst)
{
    if (inst->getOp() != kIROp_Func)
    {
        return false;
    }

    // Skip functions with no body
    bool hasBody = false;
    for (auto child : inst->getChildren())
    {
        if (child->getOp() == kIROp_Block)
        {
            hasBody = true;
            break;
        }
    }
    if (!hasBody)
    {
        return false;
    }

    // Skip functions marked with unsafeForceInlineDecoration
    if (inst->findDecoration<IRUnsafeForceInlineEarlyDecoration>())
    {
        return false;
    }

    // Skip non-simple HLSL data types, filters out generics
    if (!isSimpleHLSLDataType(inst))
    {
        return false;
    }

    return true;
}

/*
 * Precompile the module for the given target.
 *
 * This function creates a target program and emits the precompiled blob as
 * an embedded blob in the module IR, e.g. DXIL, SPIR-V.
 * Because the IR for the Slang Module may violate the restrictions of the
 * target language, the emitted target blob may not be able to include the
 * full module, but rather only the subset that can be precompiled. For
 * example, DXIL libraries do not allow resources like structured buffers
 * to appear in the library interface. Also, no target languages allow
 * generics to be precompiled.
 *
 * Some restrictions can be enforced up front before linking, but some are
 * done during target generation in between IR linking+legalization and
 * target source emission.
 *
 * Functions which can be rejected up front:
 * - Functions with no body
 * - Functions marked with unsafeForceInlineDecoration
 * - Functions that define or use generics
 *
 * The functions not rejected up front are marked with
 * DownstreamModuleExportDecoration which indicates functions we're trying to
 * export for precompilation, and this also helps to identify the functions
 * in the linked IR which survived the additional pruning.
 *
 * Functions that are rejected after linking+legalization (inside
 * emitPrecompiledDownstreamIR):
 * - (DXIL) Functions that return or take a HLSLStructuredBufferType
 * - (DXIL) Functions that return or take a Matrix type
 *
 * emitPrecompiled* produces the output artifact containing target language
 * blob, and as metadata, the list of functions which survived the second
 * phase of filtering.
 *
 * The original module IR functions matching those are then marked with
 * "AvailableInDownstreamIRDecoration" to indicate to future
 * module users which functions are present in the precompiled blob.
 */
SLANG_NO_THROW SlangResult SLANG_MCALL
Module::precompileForTarget(SlangCompileTarget target, slang::IBlob** outDiagnostics)
{
    CodeGenTarget targetEnum = CodeGenTarget(target);

    // Don't precompile twice for the same target
    for (auto globalInst : getIRModule()->getModuleInst()->getChildren())
    {
        if (auto inst = as<IREmbeddedDownstreamIR>(globalInst))
        {
            if (inst->getTarget() == targetEnum)
            {
                return SLANG_OK;
            }
        }
    }

    auto module = getIRModule();
    auto linkage = getLinkage();
    auto builder = IRBuilder(module);

    DiagnosticSink sink(linkage->getSourceManager(), Lexer::sourceLocationLexer);
    applySettingsToDiagnosticSink(&sink, &sink, linkage->m_optionSet);
    applySettingsToDiagnosticSink(&sink, &sink, m_optionSet);

    RefPtr<TargetRequest> targetReq = new TargetRequest(linkage, targetEnum);

    List<RefPtr<ComponentType>> allComponentTypes;
    allComponentTypes.add(this); // Add Module as a component type

    for (auto entryPoint : this->getEntryPoints())
    {
        allComponentTypes.add(entryPoint); // Add the entry point as a component type
    }

    auto composite = CompositeComponentType::create(linkage, allComponentTypes);

    composite = fillRequirements(composite);

    TargetProgram tp(composite, targetReq);
    tp.getOrCreateLayout(&sink);
    Slang::Index const entryPointCount = m_entryPoints.getCount();
    tp.getOptionSet().add(CompilerOptionName::GenerateWholeProgram, true);

    switch (targetReq->getTarget())
    {
    case CodeGenTarget::DXIL:
        tp.getOptionSet().add(CompilerOptionName::Profile, Profile::RawEnum::DX_Lib_6_6);
        break;
    case CodeGenTarget::SPIRV:
        break;
    default:
        return SLANG_FAIL;
    }

    tp.getOptionSet().add(CompilerOptionName::EmbedDownstreamIR, true);

    CodeGenContext::EntryPointIndices entryPointIndices;

    entryPointIndices.setCount(entryPointCount);
    for (Index i = 0; i < entryPointCount; i++)
        entryPointIndices[i] = i;
    CodeGenContext::Shared sharedCodeGenContext(&tp, entryPointIndices, &sink, nullptr);
    CodeGenContext codeGenContext(&sharedCodeGenContext);

    // Mark all public functions as exported, ensure there's at least one. Store a mapping
    // of function name to IRInst* for later reference. After linking is done, we'll scan
    // the linked result to see which functions survived the pruning and are included in the
    // precompiled blob.
    Dictionary<String, IRInst*> nameToFunction;
    bool hasAtLeastOneFunction = false;
    for (auto inst : module->getGlobalInsts())
    {
        if (attemptPrecompiledExport(inst))
        {
            hasAtLeastOneFunction = true;
            builder.addDecoration(inst, kIROp_DownstreamModuleExportDecoration);
            nameToFunction[inst->findDecoration<IRExportDecoration>()->getMangledName()] = inst;
        }
    }

    // Bail if there are no functions to export. That's not treated as an error
    // because it's possible that the module just doesn't have any simple HLSL.
    if (!hasAtLeastOneFunction)
    {
        return SLANG_OK;
    }

    ComPtr<IArtifact> outArtifact;
    SlangResult res = codeGenContext.emitPrecompiledDownstreamIR(outArtifact);

    sink.getBlobIfNeeded(outDiagnostics);
    if (res != SLANG_OK)
    {
        return res;
    }

    auto metadata = findAssociatedRepresentation<IArtifactPostEmitMetadata>(outArtifact);
    if (!metadata)
    {
        return SLANG_E_NOT_AVAILABLE;
    }

    for (const auto& mangledName : metadata->getExportedFunctionMangledNames())
    {
        auto moduleInst = nameToFunction[mangledName];
        builder.addDecoration(
            moduleInst,
            kIROp_AvailableInDownstreamIRDecoration,
            builder.getIntValue(builder.getIntType(), (int)targetReq->getTarget()));
        auto moduleDec = moduleInst->findDecoration<IRDownstreamModuleExportDecoration>();
        moduleDec->removeAndDeallocate();
    }

    // Finally, clean up the transient export decorations left over in the module. These are
    // represent functions that were pruned from the IR after linking, before target generation.
    for (auto moduleInst : module->getGlobalInsts())
    {
        if (moduleInst->getOp() == kIROp_Func)
        {
            if (auto dec = moduleInst->findDecoration<IRDownstreamModuleExportDecoration>())
            {
                dec->removeAndDeallocate();
            }
        }
    }

    ComPtr<ISlangBlob> blob;
    outArtifact->loadBlob(ArtifactKeep::Yes, blob.writeRef());

    // Add the precompiled blob to the module
    builder.setInsertInto(module);

    builder.emitEmbeddedDownstreamIR(targetReq->getTarget(), blob);
    return SLANG_OK;
}

SLANG_NO_THROW SlangResult SLANG_MCALL Module::getPrecompiledTargetCode(
    SlangCompileTarget target,
    slang::IBlob** outCode,
    slang::IBlob** outDiagnostics)
{
    SLANG_UNUSED(outDiagnostics);
    for (auto globalInst : getIRModule()->getModuleInst()->getChildren())
    {
        if (auto inst = as<IREmbeddedDownstreamIR>(globalInst))
        {
            static_assert(CodeGenTarget::DXIL == static_cast<CodeGenTarget>(SLANG_DXIL));
            static_assert(CodeGenTarget::SPIRV == static_cast<CodeGenTarget>(SLANG_SPIRV));
            if (inst->getTarget() == static_cast<CodeGenTarget>(target))
            {
                auto slice = inst->getBlob()->getStringSlice();
                auto blob = StringBlob::create(slice);
                *outCode = blob.detach();
                return SLANG_OK;
            }
        }
    }
    return SLANG_FAIL;
}

SLANG_NO_THROW SlangInt SLANG_MCALL Module::getModuleDependencyCount()
{
    return 0;
}

SLANG_NO_THROW SlangResult SLANG_MCALL Module::getModuleDependency(
    SlangInt dependencyIndex,
    IModule** outModule,
    slang::IBlob** outDiagnostics)
{
    SLANG_UNUSED(dependencyIndex);
    SLANG_UNUSED(outModule);
    SLANG_UNUSED(outDiagnostics);
    return SLANG_OK;
}

// ComponentType

SLANG_NO_THROW SlangResult SLANG_MCALL
ComponentType::precompileForTarget(SlangCompileTarget target, slang::IBlob** outDiagnostics)
{
    SLANG_UNUSED(target);
    SLANG_UNUSED(outDiagnostics);
    return SLANG_FAIL;
}

SLANG_NO_THROW SlangResult SLANG_MCALL ComponentType::getPrecompiledTargetCode(
    SlangCompileTarget target,
    slang::IBlob** outCode,
    slang::IBlob** outDiagnostics)
{
    SLANG_UNUSED(target);
    SLANG_UNUSED(outCode);
    SLANG_UNUSED(outDiagnostics);
    return SLANG_FAIL;
}

SLANG_NO_THROW SlangInt SLANG_MCALL ComponentType::getModuleDependencyCount()
{
    return getModuleDependencies().getCount();
}

SLANG_NO_THROW SlangResult SLANG_MCALL ComponentType::getModuleDependency(
    SlangInt dependencyIndex,
    slang::IModule** outModule,
    slang::IBlob** outDiagnostics)
{
    SLANG_UNUSED(outDiagnostics);
    if (dependencyIndex < 0 || dependencyIndex >= getModuleDependencies().getCount())
    {
        return SLANG_E_INVALID_ARG;
    }
    getModuleDependencies()[dependencyIndex]->addRef();
    *outModule = getModuleDependencies()[dependencyIndex];
    return SLANG_OK;
}
} // namespace Slang
