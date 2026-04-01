// slang-support.cpp

#define _CRT_SECURE_NO_WARNINGS 1

#include "slang-support.h"

#include "../../source/compiler-core/slang-artifact-desc-util.h"
#include "../../source/core/slang-file-system.h"
#include "../../source/core/slang-string-util.h"
#include "../../source/core/slang-test-tool-util.h"
#include "options.h"

#include <assert.h>
#include <stdio.h>

namespace renderer_test
{
using namespace Slang;

// Entry point name to use for vertex/fragment shader
static const char vertexEntryPointName[] = "vertexMain";
static const char fragmentEntryPointName[] = "fragmentMain";
static const char computeEntryPointName[] = "computeMain";
static const char rtEntryPointName[] = "raygenMain";
static const char taskEntryPointName[] = "taskMain";
static const char meshEntryPointName[] = "meshMain";

void ShaderCompilerUtil::Output::set(slang::IComponentType* inSlangProgram)
{
    slangProgram = inSlangProgram;
    desc.slangGlobalScope = inSlangProgram;
}

void ShaderCompilerUtil::Output::reset()
{
    {
        desc.slangGlobalScope = nullptr;
    }

    globalSession = nullptr;
    m_session = nullptr;
}

static SlangResult _compileProgramImpl(
    slang::IGlobalSession* globalSession,
    const Options& options,
    const ShaderCompilerUtil::Input& input,
    const ShaderCompileRequest& request,
    ShaderCompilerUtil::Output& out)
{
    out.reset();

    List<const char*> args;
    for (const auto& arg : options.downstreamArgs.getArgsByName("slang"))
    {
        args.add(arg.value.getBuffer());
        // The -load-repro feature is not maintained, and not supported by the new compile API.
        // TODO: Remove this when the feature has been deprecated.
        SLANG_ASSERT(arg.value != "-load-repro");
    }

    slang::TargetDesc sessionTargetDesc = {};
    slang::SessionDesc sessionDesc = {};
    ComPtr<ISlangUnknown> sessionDescMemory;
    // If there are additional args parse them
    if (args.getCount())
    {
        const auto res = globalSession->parseCommandLineArguments(
            int(args.getCount()),
            args.getBuffer(),
            &sessionDesc,
            sessionDescMemory.writeRef());
        // If there is a parse failure and diagnostic, output it
        if (SLANG_FAILED(res))
        {
            fprintf(stderr, "error: Failed to parse command line arguments: %d\n", int(res));
            return res;
        }
        // We're setting the targets ourselves, below.
        // To simplify that, we're currently not expecting targets to be added by the command line
        // arguments.
        if (sessionDesc.targetCount > 0)
        {
            fprintf(stderr, "error: Command line arguments added targets.\n");
            return SLANG_FAIL;
        }
    }

    // Argument parsing may have already added options, so add those first.
    // For module reference options there are two cases:
    // 1. If it's a slang module, then record the path and later create an IModule from that.
    // 2. If not, then propagate the option.
    // The reason to propagate the option in case 2 is that there is not currently a way of
    // representing a module for a downstream compiler in the compilation API.
    List<slang::CompilerOptionEntry> sessionOptionEntries;
    List<Slang::String> referencedSlangModulePaths;
    for (int optionIndex = 0; optionIndex < sessionDesc.compilerOptionEntryCount; optionIndex++)
    {
        slang::CompilerOptionEntry& option = sessionDesc.compilerOptionEntries[optionIndex];
        if (option.name == slang::CompilerOptionName::ReferenceModule)
        {
            SLANG_ASSERT(option.value.kind == slang::CompilerOptionValueKind::String);
            const char* path = option.value.stringValue0;
            auto desc = Slang::ArtifactDescUtil::getDescFromPath(Slang::UnownedStringSlice(path));
            switch (desc.payload)
            {
            case Slang::ArtifactDesc::Payload::SlangIR:
            case Slang::ArtifactDesc::Payload::Slang:
                referencedSlangModulePaths.add(option.value.stringValue0);
                break;
            case Slang::ArtifactDesc::Payload::DXIL:
                sessionOptionEntries.add(option);
                break;
            default:
                {
                    fprintf(
                        stderr,
                        "error: Unexpected artifact payload type: %d\n",
                        (int)desc.payload);
                    return SLANG_FAIL;
                }
            }
        }
        else
        {
            sessionOptionEntries.add(option);
        }
    }

    List<slang::PreprocessorMacroDesc> macros;

    // Define a macro so that shader code in a test can detect what language we
    // are nominally working with.
    char const* langDefine = nullptr;
    switch (input.sourceLanguage)
    {
    case SLANG_SOURCE_LANGUAGE_GLSL:
        macros.add({"__GLSL__", "1"});
        break;

    case SLANG_SOURCE_LANGUAGE_SLANG:
        macros.add({"__SLANG__", "1"});
        // fall through
    case SLANG_SOURCE_LANGUAGE_HLSL:
        macros.add({"__HLSL__", "1"});
        break;
    case SLANG_SOURCE_LANGUAGE_C:
        macros.add({"__C__", "1"});
        break;
    case SLANG_SOURCE_LANGUAGE_CPP:
        macros.add({"__CPP__", "1"});
        break;
    case SLANG_SOURCE_LANGUAGE_CUDA:
        macros.add({"__CUDA__", "1"});
        break;
    case SLANG_SOURCE_LANGUAGE_WGSL:
        macros.add({"__WGSL__", "1"});
        break;

    default:
        assert(!"unexpected");
        break;
    }

    {
        slang::CompilerOptionEntry entry;
        entry.name = slang::CompilerOptionName::AllowGLSL;
        entry.value.kind = slang::CompilerOptionValueKind::Int;
        entry.value.intValue0 = int(options.allowGLSL);
        sessionOptionEntries.add(entry);
    }

    {
        slang::CompilerOptionEntry entry;
        entry.name = slang::CompilerOptionName::PassThrough;
        entry.value.kind = slang::CompilerOptionValueKind::Int;
        entry.value.intValue0 = int(input.passThrough);
        sessionOptionEntries.add(entry);
    }

    {
        slang::CompilerOptionEntry entry;
        entry.name = slang::CompilerOptionName::LineDirectiveMode;
        entry.value.kind = slang::CompilerOptionValueKind::Int;
        entry.value.intValue0 = int(SlangLineDirectiveMode::SLANG_LINE_DIRECTIVE_MODE_NONE);
        sessionOptionEntries.add(entry);
    }

    sessionTargetDesc.format = input.target;
    if (input.profile.getLength()) // do not set profile unless requested
        sessionTargetDesc.profile = globalSession->findProfile(input.profile.getBuffer());
    if (options.generateSPIRVDirectly)
        sessionTargetDesc.flags |= SLANG_TARGET_FLAG_GENERATE_SPIRV_DIRECTLY;
    else
        sessionTargetDesc.flags = 0;

    // Not expecting argument parsing to have added any targets
    SLANG_ASSERT(sessionDesc.targetCount == 0);
    sessionDesc.targetCount = 1;
    sessionDesc.targets = &sessionTargetDesc;

    sessionDesc.skipSPIRVValidation = options.skipSPIRVValidation;
    if (options.generateSPIRVDirectly)
    {
        slang::CompilerOptionEntry entry;
        entry.name = slang::CompilerOptionName::DebugInformation;
        entry.value.kind = slang::CompilerOptionValueKind::Int;
        entry.value.intValue0 =
            int(options.disableDebugInfo ? SlangDebugInfoLevel::SLANG_DEBUG_INFO_LEVEL_NONE
                                         : SlangDebugInfoLevel::SLANG_DEBUG_INFO_LEVEL_STANDARD);
        sessionOptionEntries.add(entry);
    }

    sessionDesc.compilerOptionEntryCount = sessionOptionEntries.getCount();
    sessionDesc.compilerOptionEntries = sessionOptionEntries.getBuffer();

    // Argument parsing should not have added macros.
    SLANG_ASSERT(sessionDesc.preprocessorMacroCount == 0);
    sessionDesc.preprocessorMacroCount = (SlangInt)macros.getCount();
    sessionDesc.preprocessorMacros = macros.getBuffer();

    ComPtr<slang::ISession> slangSession = nullptr;
    SLANG_RETURN_ON_FAIL(globalSession->createSession(sessionDesc, slangSession.writeRef()));
    out.m_session = slangSession;
    out.globalSession = globalSession;

    String source(request.source.dataBegin, request.source.dataEnd);
    ComPtr<slang::IBlob> diagnostics;
    ComPtr<slang::IModule> module(slangSession->loadModuleFromSourceString(
        "main",
        request.source.path,
        source.getBuffer(),
        diagnostics.writeRef()));
    if (!module)
    {
        fprintf(
            stderr,
            "error: Failed to load module: %s\n",
            diagnostics ? (char*)diagnostics->getBufferPointer() : "(no diagnostic output)");
        return SLANG_FAIL;
    }

    // Some tests are verifying that various warnings are printed, so print any diagnostics!
    if (diagnostics && (diagnostics->getBufferSize() > 0U))
        StdWriters::getError().print("%s", (char*)diagnostics->getBufferPointer());

    ComPtr<slang::IModule> specializedModule;
    List<ComPtr<slang::IEntryPoint>> specializedEntryPoints;
    List<slang::IComponentType*> componentsRawPtr;

    ComPtr<ISlangFileSystem> osFileSystem =
        ComPtr<ISlangFileSystem>(Slang::OSFileSystem::getExtSingleton());

    // This list is just kept so that the modules will be freed at scope exit
    List<ComPtr<slang::IModule>> referencedModules;
    for (auto& path : referencedSlangModulePaths)
    {
        auto desc =
            Slang::ArtifactDescUtil::getDescFromPath(Slang::UnownedStringSlice(path.getBuffer()));
        // If it's a GPU binary, then we'll assume it's a library
        if (ArtifactDescUtil::isGpuUsable(desc))
        {
            desc.kind = ArtifactKind::Library;
        }
        const String name = ArtifactDescUtil::getBaseNameFromPath(desc, path.getUnownedSlice());

        ComPtr<slang::IBlob> codeBlob;
        SlangResult result = osFileSystem->loadFile(path.getBuffer(), codeBlob.writeRef());
        if (SLANG_FAILED(result))
        {
            fprintf(stderr, "error: Failed to read referenced module file: %s\n", path.getBuffer());
            return SLANG_FAIL;
        }

        ComPtr<slang::IModule> module;
        switch (desc.payload)
        {
        case Slang::ArtifactDesc::Payload::Slang:
            {
                String sourceString(
                    (const char*)codeBlob->getBufferPointer(),
                    (const char*)codeBlob->getBufferPointer() + codeBlob->getBufferSize());
                module = ComPtr<slang::IModule>(slangSession->loadModuleFromSourceString(
                    name.getBuffer(),
                    path.getBuffer(),
                    sourceString.getBuffer(),
                    diagnostics.writeRef()));
                break;
            }
        case Slang::ArtifactDesc::Payload::SlangIR:
            {
                module = ComPtr<slang::IModule>(slangSession->loadModuleFromIRBlob(
                    name.getBuffer(),
                    path.getBuffer(),
                    codeBlob,
                    diagnostics.writeRef()));
                break;
            }
        default:
            {
                SLANG_UNREACHABLE("Unexpected artifact payload type");
            }
        }

        if (!module)
        {
            fprintf(
                stderr,
                "error: Failed to load referenced module: %s: %s\n",
                path.getBuffer(),
                diagnostics ? (char*)diagnostics->getBufferPointer() : "(no diagnostic output)");
            return SLANG_FAIL;
        }
        referencedModules.add(module);
        componentsRawPtr.add(module.get());
    }

    int globalSpecializationArgCount = int(request.globalSpecializationArgs.getCount());
    int moduleSpecializationArgCount = module->getSpecializationParamCount();
    if (globalSpecializationArgCount != moduleSpecializationArgCount)
    {
        fprintf(
            stderr,
            "error: The specialization argument count of the request (%d) does not match that of "
            "the module (%d)!\n",
            globalSpecializationArgCount,
            moduleSpecializationArgCount);
        return SLANG_FAIL;
    }
    List<slang::SpecializationArg> moduleSpecializationArgs;
    for (int ii = 0; ii < globalSpecializationArgCount; ++ii)
    {
        String specializedTypeName = request.globalSpecializationArgs[ii].getBuffer();
        slang::TypeReflection* typeReflection =
            module->getLayout()->findTypeByName(specializedTypeName.getBuffer());
        moduleSpecializationArgs.add(slang::SpecializationArg::fromType(typeReflection));
    }

    {
        ComPtr<slang::IBlob> diagnostics;
        auto res = module->specialize(
            moduleSpecializationArgs.getBuffer(),
            moduleSpecializationArgs.getCount(),
            (slang::IComponentType**)specializedModule.writeRef(),
            diagnostics.writeRef());
        if (SLANG_FAILED(res))
        {
            fprintf(
                stderr,
                "error: Failed to specialize module: %s\n",
                diagnostics ? (char*)diagnostics->getBufferPointer() : "(no diagnostic output)");
            return res;
        }
    }

    Index explicitEntryPointCount = request.entryPoints.getCount();
    for (Index ee = 0; ee < explicitEntryPointCount; ++ee)
    {
        if (options.dontAddDefaultEntryPoints)
        {
            // If default entry points are not to be added, then
            // the `request.entryPoints` array should have been
            // left empty.
            //
            SLANG_ASSERT(false);
        }

        auto& entryPointInfo = request.entryPoints[ee];

        ComPtr<slang::IEntryPoint> entryPoint;
        ComPtr<slang::IBlob> diagnostics;
        auto res = module->findAndCheckEntryPoint(
            entryPointInfo.name,
            entryPointInfo.slangStage,
            entryPoint.writeRef(),
            diagnostics.writeRef());
        if (SLANG_FAILED(res))
        {
            fprintf(
                stderr,
                "error: Failed to find entry point '%s': %s\n",
                entryPointInfo.name,
                diagnostics ? (char*)diagnostics->getBufferPointer() : "(no diagnostic output)");
            return res;
        }

        const int entryPointSpecializationArgCount =
            int(request.entryPointSpecializationArgs.getCount());
        if (entryPointSpecializationArgCount != entryPoint->getSpecializationParamCount())
        {
            fprintf(
                stderr,
                "error: %s\n",
                "The specialization argument count of the requested entry point does not match "
                "that of the entry point!");
            return SLANG_FAIL;
        }

        List<slang::SpecializationArg> entryPointSpecializationArgs;
        for (int ii = 0; ii < entryPointSpecializationArgCount; ++ii)
        {
            String specializedTypeName = request.entryPointSpecializationArgs[ii].getBuffer();
            slang::TypeReflection* typeReflection =
                module->getLayout()->findTypeByName(specializedTypeName.getBuffer());
            entryPointSpecializationArgs.add(slang::SpecializationArg::fromType(typeReflection));
        }

        ComPtr<slang::IEntryPoint> specializedEntryPoint;
        {
            ComPtr<slang::IBlob> diagnostics;
            auto res = entryPoint->specialize(
                entryPointSpecializationArgs.getBuffer(),
                entryPointSpecializationArgs.getCount(),
                (slang::IComponentType**)specializedEntryPoint.writeRef(),
                diagnostics.writeRef());
            if (SLANG_FAILED(res))
            {
                fprintf(
                    stderr,
                    "error: Failed to specialize entry point: %s\n",
                    diagnostics ? (char*)diagnostics->getBufferPointer()
                                : "(no diagnostic output)");
                return res;
            }
        }
        specializedEntryPoints.add(specializedEntryPoint);
    }

    if (input.passThrough == SLANG_PASS_THROUGH_NONE)
    {
        componentsRawPtr.add(specializedModule);
        for (auto& specializedEntryPoint : specializedEntryPoints)
            componentsRawPtr.add(specializedEntryPoint);
    }

    // This list just makes sure that the components get released
    List<ComPtr<slang::ITypeConformance>> typeConformanceComponents;
    if (request.typeConformances.getCount())
    {
        auto reflection = module->getLayout();
        for (auto& conformance : request.typeConformances)
        {
            ComPtr<ISlangBlob> outDiagnostic;
            auto derivedType = reflection->findTypeByName(conformance.derivedTypeName.getBuffer());
            auto baseType = reflection->findTypeByName(conformance.baseTypeName.getBuffer());
            ComPtr<slang::ITypeConformance> conformanceComponentType;
            SlangResult res = slangSession->createTypeConformanceComponentType(
                derivedType,
                baseType,
                conformanceComponentType.writeRef(),
                conformance.idOverride,
                outDiagnostic.writeRef());
            if (SLANG_FAILED(res))
            {
                fprintf(
                    stderr,
                    "error: Failed to handle type conformances: %s\n",
                    outDiagnostic ? (char*)outDiagnostic->getBufferPointer()
                                  : "(no diagnostic output)");
                return res;
            }
            typeConformanceComponents.add(conformanceComponentType);
            componentsRawPtr.add(conformanceComponentType);
        }
    }

    ComPtr<slang::IComponentType> linkedSlangProgram;
    if (componentsRawPtr.getCount() > 0)
    {
        ComPtr<slang::IComponentType> composite;
        ComPtr<ISlangBlob> outDiagnostic;
        SlangResult res = slangSession->createCompositeComponentType(
            componentsRawPtr.getBuffer(),
            componentsRawPtr.getCount(),
            composite.writeRef(),
            outDiagnostic.writeRef());
        if (SLANG_FAILED(res))
        {
            fprintf(
                stderr,
                "error: Failed to create composite: %s\n",
                outDiagnostic ? (char*)outDiagnostic->getBufferPointer()
                              : "(no diagnostic output)");
            return res;
        }
        res = composite->link(linkedSlangProgram.writeRef(), outDiagnostic.writeRef());
        if (SLANG_FAILED(res))
        {
            fprintf(
                stderr,
                "error: Failed to link program: %s\n",
                outDiagnostic ? (char*)outDiagnostic->getBufferPointer()
                              : "(no diagnostic output)");
        }
    }

    out.set(linkedSlangProgram);
    return SLANG_OK;
}

static SlangResult compileProgram(
    slang::IGlobalSession* globalSession,
    const Options& options,
    const ShaderCompilerUtil::Input& input,
    const ShaderCompileRequest& request,
    ShaderCompilerUtil::Output& out)
{
    if (input.passThrough == SLANG_PASS_THROUGH_NONE)
    {
        return _compileProgramImpl(globalSession, options, input, request, out);
    }
    else
    {
        bool canUseSlangForPrecompile = false;
        switch (input.passThrough)
        {
        case SLANG_PASS_THROUGH_DXC:
        case SLANG_PASS_THROUGH_FXC:
            canUseSlangForPrecompile = true;
            break;
        default:
            break;
        }
        // If we are doing a HLSL pass-through compilation, then we can't rely
        // on the downstream compiler for the reflection information that
        // will drive all of our parameter binding. As such, we will first
        // compile with Slang to get reflection information, and then
        // compile in another pass using the desired downstream compiler
        // so that we can get the refleciton information we need.
        //
        ShaderCompilerUtil::Output slangOutput;
        if (canUseSlangForPrecompile)
        {
            ShaderCompilerUtil::Input slangInput = input;
            slangInput.sourceLanguage = SLANG_SOURCE_LANGUAGE_SLANG;
            slangInput.passThrough = SLANG_PASS_THROUGH_NONE;
            // TODO: we want to pass along a flag to skip codegen...


            SLANG_RETURN_ON_FAIL(
                _compileProgramImpl(globalSession, options, slangInput, request, slangOutput));
        }

        // Now we have what we need to be able to do the downstream compile better.
        //
        // TODO: We should be able to use the output from the Slang compilation
        // to fill in the actual entry points to be used for this compilation,
        // so that discovery of entry points via `[shader(...)]` attributes will work.
        //
        SLANG_RETURN_ON_FAIL(_compileProgramImpl(globalSession, options, input, request, out));

        out.m_session = slangOutput.m_session;
        // slangOutput.desc.slangGlobalScope and slangOutput.slangProgram are the same object,
        // but the latter is a ComPtr while the former isn't. Therefore we need to detach so
        // that the object doesn't get destroyed.
        SLANG_ASSERT(slangOutput.desc.slangGlobalScope == slangOutput.slangProgram.get());
        out.desc.slangGlobalScope = slangOutput.slangProgram.detach();
        slangOutput.m_session = nullptr;
        return SLANG_OK;
    }
}

// Helper for compileWithLayout
/* static */ SlangResult readSource(const String& inSourcePath, List<char>& outSourceText)
{
    // Read in the source code
    FILE* sourceFile = fopen(inSourcePath.getBuffer(), "rb");
    if (!sourceFile)
    {
        fprintf(stderr, "error: failed to open '%s' for reading\n", inSourcePath.getBuffer());
        return SLANG_FAIL;
    }
    fseek(sourceFile, 0, SEEK_END);
    size_t sourceSize = ftell(sourceFile);
    fseek(sourceFile, 0, SEEK_SET);

    outSourceText.setCount(sourceSize + 1);
    if (fread(outSourceText.getBuffer(), sourceSize, 1, sourceFile) != 1)
    {
        fprintf(stderr, "error: failed to read from '%s'\n", inSourcePath.getBuffer());
        return SLANG_FAIL;
    }
    fclose(sourceFile);
    outSourceText[sourceSize] = 0;

    return SLANG_OK;
}

/* static */ SlangResult ShaderCompilerUtil::compileWithLayout(
    slang::IGlobalSession* globalSession,
    const Options& options,
    const Input& input,
    ShaderCompilerUtil::OutputAndLayout& output)
{
    String sourcePath = options.sourcePath;
    auto shaderType = options.shaderType;

    List<char> sourceText;
    SLANG_RETURN_ON_FAIL(readSource(sourcePath, sourceText));

    if (input.sourceLanguage == SLANG_SOURCE_LANGUAGE_CPP ||
        input.sourceLanguage == SLANG_SOURCE_LANGUAGE_C)
    {
        // Add an include of the prelude
        ComPtr<ISlangBlob> prelude;
        globalSession->getLanguagePrelude(input.sourceLanguage, prelude.writeRef());

        String preludeString = StringUtil::getString(prelude);

        // Add the prelude
        StringBuilder builder;
        builder << preludeString << "\n";
        builder << UnownedStringSlice(sourceText.getBuffer(), sourceText.getCount());

        sourceText.setCount(builder.getLength());
        memcpy(sourceText.getBuffer(), builder.getBuffer(), builder.getLength());
    }

    output.sourcePath = sourcePath;

    auto& layout = output.layout;

    // Default the amount of renderTargets based on shader type
    switch (shaderType)
    {
    default:
        layout.numRenderTargets = 1;
        break;

    case Options::ShaderProgramType::Compute:
    case Options::ShaderProgramType::RayTracing:
        layout.numRenderTargets = 0;
        break;
    }

    // Deterministic random generator
    RefPtr<RandomGenerator> rand = RandomGenerator::create(0x34234);

    // Parse the layout
    layout.parse(rand, sourceText.getBuffer());

    // Setup SourceInfo
    ShaderCompileRequest::SourceInfo sourceInfo;
    sourceInfo.path = sourcePath.getBuffer();
    sourceInfo.dataBegin = sourceText.getBuffer();
    // Subtract 1 because it's zero terminated
    sourceInfo.dataEnd = sourceText.getBuffer() + sourceText.getCount() - 1;

    ShaderCompileRequest compileRequest;

    compileRequest.source = sourceInfo;

    // Now we will add the "default" entry point names/stages that
    // are appropriate to the pipeline type being targetted, *unless*
    // the options specify that we should leave out the default
    // entry points and instead rely on the Slang compiler's built-in
    // mechanisms for discovering entry points (e.g., `[shader(...)]`
    // attributes).
    //
    if (!options.dontAddDefaultEntryPoints)
    {
        switch (shaderType)
        {
        case Options::ShaderProgramType::Graphics:
        case Options::ShaderProgramType::GraphicsCompute:
            {
                ShaderCompileRequest::EntryPoint vertexEntryPoint;
                vertexEntryPoint.name = vertexEntryPointName;
                vertexEntryPoint.slangStage = SLANG_STAGE_VERTEX;
                compileRequest.entryPoints.add(vertexEntryPoint);

                ShaderCompileRequest::EntryPoint fragmentEntryPoint;
                fragmentEntryPoint.name = fragmentEntryPointName;
                fragmentEntryPoint.slangStage = SLANG_STAGE_FRAGMENT;
                compileRequest.entryPoints.add(fragmentEntryPoint);
            }
            break;
        case Options::ShaderProgramType::GraphicsTaskMeshCompute:
            {
                ShaderCompileRequest::EntryPoint taskEntryPoint;
                taskEntryPoint.name = taskEntryPointName;
                taskEntryPoint.slangStage = SLANG_STAGE_AMPLIFICATION;
                compileRequest.entryPoints.add(taskEntryPoint);
            }
            [[fallthrough]];
        case Options::ShaderProgramType::GraphicsMeshCompute:
            {
                ShaderCompileRequest::EntryPoint meshEntryPoint;
                meshEntryPoint.name = meshEntryPointName;
                meshEntryPoint.slangStage = SLANG_STAGE_MESH;
                compileRequest.entryPoints.add(meshEntryPoint);

                ShaderCompileRequest::EntryPoint fragmentEntryPoint;
                fragmentEntryPoint.name = fragmentEntryPointName;
                fragmentEntryPoint.slangStage = SLANG_STAGE_FRAGMENT;
                compileRequest.entryPoints.add(fragmentEntryPoint);
            }
            break;
        case Options::ShaderProgramType::RayTracing:
            {
                // Note: Current GPU ray tracing pipelines allow for an
                // almost arbitrary mix of entry points for different stages
                // to be used together (e.g., a single "program" might
                // have multiple any-hit shaders, multiple miss shaders, etc.)
                //
                // Rather than try to define a fixed set of entry point
                // names and stages that the testing will support, we will
                // instead rely on `[shader(...)]` annotations to tell us
                // what entry points are present in the input code.
            }
            break;
        default:
            {
                ShaderCompileRequest::EntryPoint computeEntryPoint;
                computeEntryPoint.name = computeEntryPointName;
                computeEntryPoint.slangStage = SLANG_STAGE_COMPUTE;
                compileRequest.entryPoints.add(computeEntryPoint);
            }
        }
    }
    compileRequest.globalSpecializationArgs = layout.globalSpecializationArgs;
    compileRequest.entryPointSpecializationArgs = layout.entryPointSpecializationArgs;
    for (auto conformance : layout.typeConformances)
    {
        ShaderCompileRequest::TypeConformance c;
        c.derivedTypeName = conformance.derivedTypeName;
        c.baseTypeName = conformance.baseTypeName;
        c.idOverride = conformance.idOverride;
        compileRequest.typeConformances.add(c);
    }
    return compileProgram(globalSession, options, input, compileRequest, output.output);
}

} // namespace renderer_test
