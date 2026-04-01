// Compiler.cpp : Defines the entry point for the console application.
//
#include "slang-compiler.h"

#include "../compiler-core/slang-lexer.h"
#include "../core/slang-basic.h"
#include "../core/slang-castable.h"
#include "../core/slang-hex-dump-util.h"
#include "../core/slang-io.h"
#include "../core/slang-performance-profiler.h"
#include "../core/slang-platform.h"
#include "../core/slang-riff.h"
#include "../core/slang-string-util.h"
#include "../core/slang-type-convert-util.h"
#include "../core/slang-type-text-util.h"
#include "slang-check-impl.h"
#include "slang-check.h"

#include <chrono>

// Artifact
#include "../compiler-core/slang-artifact-associated.h"
#include "../compiler-core/slang-artifact-container-util.h"
#include "../compiler-core/slang-artifact-desc-util.h"
#include "../compiler-core/slang-artifact-diagnostic-util.h"
#include "../compiler-core/slang-artifact-impl.h"
#include "../compiler-core/slang-artifact-representation-impl.h"
#include "../compiler-core/slang-artifact-util.h"

// Artifact output
#include "slang-artifact-output-util.h"
#include "slang-emit-cuda.h"
#include "slang-extension-tracker.h"
#include "slang-lower-to-ir.h"
#include "slang-mangle.h"
#include "slang-parameter-binding.h"
#include "slang-parser.h"
#include "slang-preprocessor.h"
#include "slang-serialize-ast.h"
#include "slang-serialize-container.h"
#include "slang-type-layout.h"

namespace Slang
{

// !!!!!!!!!!!!!!!!!!!!!! free functions for DiagnosicSink !!!!!!!!!!!!!!!!!!!!!!!!!!!!!

bool isHeterogeneousTarget(CodeGenTarget target)
{
    return ArtifactDescUtil::makeDescForCompileTarget(asExternal(target)).style ==
           ArtifactStyle::Host;
}

void printDiagnosticArg(StringBuilder& sb, CodeGenTarget val)
{
    UnownedStringSlice name = TypeTextUtil::getCompileTargetName(asExternal(val));
    name = name.getLength() ? name : toSlice("<unknown>");
    sb << name;
}

void printDiagnosticArg(StringBuilder& sb, PassThroughMode val)
{
    sb << TypeTextUtil::getPassThroughName(SlangPassThrough(val));
}

//
// FrontEndEntryPointRequest
//

FrontEndEntryPointRequest::FrontEndEntryPointRequest(
    FrontEndCompileRequest* compileRequest,
    int translationUnitIndex,
    Name* name,
    Profile profile)
    : m_compileRequest(compileRequest)
    , m_translationUnitIndex(translationUnitIndex)
    , m_name(name)
    , m_profile(profile)
{
}


TranslationUnitRequest* FrontEndEntryPointRequest::getTranslationUnit()
{
    return getCompileRequest()->translationUnits[m_translationUnitIndex];
}

//
// EntryPoint
//

ISlangUnknown* EntryPoint::getInterface(const Guid& guid)
{
    if (guid == slang::IEntryPoint::getTypeGuid())
        return static_cast<slang::IEntryPoint*>(this);

    return Super::getInterface(guid);
}

RefPtr<EntryPoint> EntryPoint::create(
    Linkage* linkage,
    DeclRef<FuncDecl> funcDeclRef,
    Profile profile)
{
    RefPtr<EntryPoint> entryPoint =
        new EntryPoint(linkage, funcDeclRef.getName(), profile, funcDeclRef);
    entryPoint->m_mangledName = getMangledName(linkage->getASTBuilder(), funcDeclRef);
    return entryPoint;
}

RefPtr<EntryPoint> EntryPoint::createDummyForPassThrough(
    Linkage* linkage,
    Name* name,
    Profile profile)
{
    RefPtr<EntryPoint> entryPoint = new EntryPoint(linkage, name, profile, DeclRef<FuncDecl>());
    return entryPoint;
}

RefPtr<EntryPoint> EntryPoint::createDummyForDeserialize(
    Linkage* linkage,
    Name* name,
    Profile profile,
    String mangledName)
{
    RefPtr<EntryPoint> entryPoint = new EntryPoint(linkage, name, profile, DeclRef<FuncDecl>());
    entryPoint->m_mangledName = mangledName;
    return entryPoint;
}

EntryPoint::EntryPoint(Linkage* linkage, Name* name, Profile profile, DeclRef<FuncDecl> funcDeclRef)
    : ComponentType(linkage), m_name(name), m_profile(profile), m_funcDeclRef(funcDeclRef)
{
    // Collect any specialization parameters used by the entry point
    //
    _collectShaderParams();
}

Module* EntryPoint::getModule()
{
    return Slang::getModule(getFuncDecl());
}

Index EntryPoint::getSpecializationParamCount()
{
    return m_genericSpecializationParams.getCount() + m_existentialSpecializationParams.getCount();
}

SpecializationParam const& EntryPoint::getSpecializationParam(Index index)
{
    auto genericParamCount = m_genericSpecializationParams.getCount();
    if (index < genericParamCount)
    {
        return m_genericSpecializationParams[index];
    }
    else
    {
        return m_existentialSpecializationParams[index - genericParamCount];
    }
}

Index EntryPoint::getRequirementCount()
{
    // The only requirement of an entry point is the module that contains it.
    //
    // TODO: We will eventually want to support the case of an entry
    // point nested in a `struct` type, in which case there should be
    // a single requirement representing that outer type (so that multiple
    // entry points nested under the same type can share the storage
    // for parameters at that scope).

    // Note: the defensive coding is here because the
    // "dummy" entry points we create for pass-through
    // compilation will not have an associated module.
    //
    if (const auto module = getModule())
    {
        return 1;
    }
    return 0;
}

RefPtr<ComponentType> EntryPoint::getRequirement(Index index)
{
    SLANG_UNUSED(index);
    SLANG_ASSERT(index == 0);
    SLANG_ASSERT(getModule());
    return getModule();
}

String EntryPoint::getEntryPointMangledName(Index index)
{
    SLANG_UNUSED(index);
    SLANG_ASSERT(index == 0);

    return m_mangledName;
}

String EntryPoint::getEntryPointNameOverride(Index index)
{
    SLANG_UNUSED(index);
    SLANG_ASSERT(index == 0);

    return m_name ? m_name->text : "";
}

void EntryPoint::acceptVisitor(
    ComponentTypeVisitor* visitor,
    SpecializationInfo* specializationInfo)
{
    visitor->visitEntryPoint(this, as<EntryPointSpecializationInfo>(specializationInfo));
}

void EntryPoint::buildHash(DigestBuilder<SHA1>& builder)
{
    SLANG_UNUSED(builder);
}

List<Module*> const& EntryPoint::getModuleDependencies()
{
    if (auto module = getModule())
        return module->getModuleDependencies();

    static List<Module*> empty;
    return empty;
}

List<SourceFile*> const& EntryPoint::getFileDependencies()
{
    if (const auto module = getModule())
        return getModule()->getFileDependencies();

    static List<SourceFile*> empty;
    return empty;
}

TypeConformance::TypeConformance(
    Linkage* linkage,
    SubtypeWitness* witness,
    Int confomrmanceIdOverride,
    DiagnosticSink* sink)
    : ComponentType(linkage)
    , m_subtypeWitness(witness)
    , m_conformanceIdOverride(confomrmanceIdOverride)
{
    addDepedencyFromWitness(witness);
    m_irModule = generateIRForTypeConformance(this, m_conformanceIdOverride, sink);
}

void TypeConformance::addDepedencyFromWitness(SubtypeWitness* witness)
{
    if (auto declaredWitness = as<DeclaredSubtypeWitness>(witness))
    {
        auto declModule = getModule(declaredWitness->getDeclRef().getDecl());
        m_moduleDependencyList.addDependency(declModule);
        m_fileDependencyList.addDependency(declModule);
        if (m_requirementSet.add(declModule))
        {
            m_requirements.add(declModule);
        }
        // TODO: handle the specialization arguments in declaredWitness->declRef.substitutions.
    }
    else if (auto transitiveWitness = as<TransitiveSubtypeWitness>(witness))
    {
        addDepedencyFromWitness(transitiveWitness->getMidToSup());
        addDepedencyFromWitness(transitiveWitness->getSubToMid());
    }
    else if (auto conjunctionWitness = as<ConjunctionSubtypeWitness>(witness))
    {
        auto componentCount = conjunctionWitness->getComponentCount();
        for (Index i = 0; i < componentCount; ++i)
        {
            auto w = as<SubtypeWitness>(conjunctionWitness->getComponentWitness(i));
            if (w)
                addDepedencyFromWitness(w);
        }
    }
}

ISlangUnknown* TypeConformance::getInterface(const Guid& guid)
{
    if (guid == slang::ITypeConformance::getTypeGuid())
        return static_cast<slang::ITypeConformance*>(this);

    return Super::getInterface(guid);
}

void TypeConformance::buildHash(DigestBuilder<SHA1>& builder)
{
    // TODO: Implement some kind of hashInto for Val then replace this
    auto subtypeWitness = m_subtypeWitness->toString();

    builder.append(subtypeWitness);
    builder.append(m_conformanceIdOverride);
}

List<Module*> const& TypeConformance::getModuleDependencies()
{
    return m_moduleDependencyList.getModuleList();
}

List<SourceFile*> const& TypeConformance::getFileDependencies()
{
    return m_fileDependencyList.getFileList();
}

Index TypeConformance::getRequirementCount()
{
    return m_requirements.getCount();
}

RefPtr<ComponentType> TypeConformance::getRequirement(Index index)
{
    return m_requirements[index];
}

void TypeConformance::acceptVisitor(
    ComponentTypeVisitor* visitor,
    ComponentType::SpecializationInfo* specializationInfo)
{
    SLANG_UNUSED(specializationInfo);
    visitor->visitTypeConformance(this);
}

RefPtr<ComponentType::SpecializationInfo> TypeConformance::_validateSpecializationArgsImpl(
    SpecializationArg const* args,
    Index argCount,
    DiagnosticSink* sink)
{
    SLANG_UNUSED(args);
    SLANG_UNUSED(argCount);
    SLANG_UNUSED(sink);
    return nullptr;
}

//

Profile Profile::lookUp(UnownedStringSlice const& name)
{
#define PROFILE(TAG, NAME, STAGE, VERSION)           \
    if (name == UnownedTerminatedStringSlice(#NAME)) \
        return Profile::TAG;
#define PROFILE_ALIAS(TAG, DEF, NAME)                \
    if (name == UnownedTerminatedStringSlice(#NAME)) \
        return Profile::TAG;
#include "slang-profile-defs.h"

    return Profile::Unknown;
}

Profile Profile::lookUp(char const* name)
{
    return lookUp(UnownedTerminatedStringSlice(name));
}

CapabilitySet Profile::getCapabilityName()
{
    List<CapabilityName> result;
    switch (getVersion())
    {
#define PROFILE_VERSION(TAG, NAME)       \
    case ProfileVersion::TAG:            \
        result.add(CapabilityName::TAG); \
        break;
#include "slang-profile-defs.h"
    default:
        break;
    }
    switch (getStage())
    {
#define PROFILE_STAGE(TAG, NAME, VAL)     \
    case Stage::TAG:                      \
        result.add(CapabilityName::NAME); \
        break;
#include "slang-profile-defs.h"
    default:
        break;
    }

    CapabilitySet resultSet = CapabilitySet(result);
    for (auto i : this->additionalCapabilities)
        resultSet.join(i);
    return resultSet;
}

char const* Profile::getName()
{
    switch (raw)
    {
    default:
        return "unknown";

#define PROFILE(TAG, NAME, STAGE, VERSION) \
    case Profile::TAG:                     \
        return #NAME;
#define PROFILE_ALIAS(TAG, DEF, NAME) /* empty */
#include "slang-profile-defs.h"
    }
}

static const StageInfo kStages[] = {
#define PROFILE_STAGE(ID, NAME, ENUM) {#NAME, Stage::ID},

#define PROFILE_STAGE_ALIAS(ID, NAME, VAL) {#NAME, Stage::ID},

#include "slang-profile-defs.h"
};

ConstArrayView<StageInfo> getStageInfos()
{
    return makeConstArrayView(kStages);
}

Stage findStageByName(String const& name)
{
    for (auto entry : kStages)
    {
        if (name == entry.name)
        {
            return entry.stage;
        }
    }

    return Stage::Unknown;
}

UnownedStringSlice getStageText(Stage stage)
{
    for (auto entry : kStages)
    {
        if (stage == entry.stage)
        {
            return UnownedStringSlice(entry.name);
        }
    }
    return UnownedStringSlice();
}

Stage getStageFromAtom(CapabilityAtom atom)
{
    switch (atom)
    {
    case CapabilityAtom::vertex:
        return Stage::Vertex;
    case CapabilityAtom::hull:
        return Stage::Hull;
    case CapabilityAtom::domain:
        return Stage::Domain;
    case CapabilityAtom::geometry:
        return Stage::Geometry;
    case CapabilityAtom::fragment:
        return Stage::Fragment;
    case CapabilityAtom::compute:
        return Stage::Compute;
    case CapabilityAtom::_mesh:
        return Stage::Mesh;
    case CapabilityAtom::_amplification:
        return Stage::Amplification;
    case CapabilityAtom::_anyhit:
        return Stage::AnyHit;
    case CapabilityAtom::_closesthit:
        return Stage::ClosestHit;
    case CapabilityAtom::_intersection:
        return Stage::Intersection;
    case CapabilityAtom::_raygen:
        return Stage::RayGeneration;
    case CapabilityAtom::_miss:
        return Stage::Miss;
    case CapabilityAtom::_callable:
        return Stage::Callable;
    case CapabilityAtom::dispatch:
        return Stage::Dispatch;
    default:
        SLANG_UNEXPECTED("unknown stage atom");
        UNREACHABLE_RETURN(Stage::Unknown);
    }
}

SlangResult checkExternalCompilerSupport(Session* session, PassThroughMode passThrough)
{
    // Check if the type is supported on this compile
    if (passThrough == PassThroughMode::None)
    {
        // If no pass through -> that will always work!
        return SLANG_OK;
    }

    return session->getOrLoadDownstreamCompiler(passThrough, nullptr) ? SLANG_OK
                                                                      : SLANG_E_NOT_FOUND;
}

SourceLanguage getDefaultSourceLanguageForDownstreamCompiler(PassThroughMode compiler)
{
    switch (compiler)
    {
    case PassThroughMode::None:
        {
            return SourceLanguage::Unknown;
        }
    case PassThroughMode::Fxc:
    case PassThroughMode::Dxc:
        {
            return SourceLanguage::HLSL;
        }
    case PassThroughMode::Glslang:
        {
            return SourceLanguage::GLSL;
        }
    case PassThroughMode::LLVM:
    case PassThroughMode::Clang:
    case PassThroughMode::VisualStudio:
    case PassThroughMode::Gcc:
    case PassThroughMode::GenericCCpp:
        {
            // These could ingest C, but we only have this function to work out a
            // 'default' language to ingest.
            return SourceLanguage::CPP;
        }
    case PassThroughMode::NVRTC:
        {
            return SourceLanguage::CUDA;
        }
    case PassThroughMode::Tint:
        {
            return SourceLanguage::WGSL;
        }
    case PassThroughMode::SpirvDis:
        {
            return SourceLanguage::SPIRV;
        }
    case PassThroughMode::MetalC:
        {
            return SourceLanguage::Metal;
        }
    default:
        break;
    }
    SLANG_ASSERT(!"Unknown compiler");
    return SourceLanguage::Unknown;
}

PassThroughMode getDownstreamCompilerRequiredForTarget(CodeGenTarget target)
{
    switch (target)
    {
    // Don't *require* a downstream compiler for source output
    case CodeGenTarget::GLSL:
    case CodeGenTarget::HLSL:
    case CodeGenTarget::CUDASource:
    case CodeGenTarget::CPPSource:
    case CodeGenTarget::HostCPPSource:
    case CodeGenTarget::PyTorchCppBinding:
    case CodeGenTarget::CSource:
    case CodeGenTarget::Metal:
    case CodeGenTarget::WGSL:
        {
            return PassThroughMode::None;
        }
    case CodeGenTarget::None:
        {
            return PassThroughMode::None;
        }
    case CodeGenTarget::WGSLSPIRVAssembly:
    case CodeGenTarget::SPIRVAssembly:
    case CodeGenTarget::SPIRV:
        {
            return PassThroughMode::SpirvDis;
        }
    case CodeGenTarget::DXBytecode:
    case CodeGenTarget::DXBytecodeAssembly:
        {
            return PassThroughMode::Fxc;
        }
    case CodeGenTarget::DXIL:
    case CodeGenTarget::DXILAssembly:
        {
            return PassThroughMode::Dxc;
        }
    case CodeGenTarget::MetalLib:
    case CodeGenTarget::MetalLibAssembly:
        {
            return PassThroughMode::MetalC;
        }
    case CodeGenTarget::ShaderHostCallable:
    case CodeGenTarget::ShaderSharedLibrary:
    case CodeGenTarget::HostExecutable:
    case CodeGenTarget::HostHostCallable:
    case CodeGenTarget::HostSharedLibrary:
        {
            // We need some C/C++ compiler
            return PassThroughMode::GenericCCpp;
        }
    case CodeGenTarget::PTX:
        {
            return PassThroughMode::NVRTC;
        }
    case CodeGenTarget::WGSLSPIRV:
        {
            return PassThroughMode::Tint;
        }
    default:
        break;
    }

    SLANG_ASSERT(!"Unhandled target");
    return PassThroughMode::None;
}

EndToEndCompileRequest* CodeGenContext::isPassThroughEnabled()
{
    auto endToEndReq = isEndToEndCompile();

    // If there isn't an end-to-end compile going on,
    // there can be no pass-through.
    //
    if (!endToEndReq)
        return nullptr;

    // And if pass-through isn't set on that end-to-end compile,
    // then we clearly areb't doing a pass-through compile.
    //
    if (endToEndReq->m_passThrough == PassThroughMode::None)
        return nullptr;

    // If we have confirmed that pass-through compilation is going on,
    // we return the end-to-end request, because it has all the
    // relevant state that we need to implement pass-through mode.
    //
    return endToEndReq;
}

/// If there is a pass-through compile going on, find the translation unit for the given entry
/// point. Assumes isPassThroughEnabled has already been called
TranslationUnitRequest* getPassThroughTranslationUnit(
    EndToEndCompileRequest* endToEndReq,
    Int entryPointIndex)
{
    SLANG_ASSERT(endToEndReq);
    SLANG_ASSERT(endToEndReq->m_passThrough != PassThroughMode::None);
    auto frontEndReq = endToEndReq->getFrontEndReq();
    auto entryPointReq = frontEndReq->getEntryPointReq(entryPointIndex);
    auto translationUnit = entryPointReq->getTranslationUnit();
    return translationUnit;
}

TranslationUnitRequest* CodeGenContext::findPassThroughTranslationUnit(Int entryPointIndex)
{
    if (auto endToEndReq = isPassThroughEnabled())
        return getPassThroughTranslationUnit(endToEndReq, entryPointIndex);
    return nullptr;
}

static void _appendCodeWithPath(
    const UnownedStringSlice& filePath,
    const UnownedStringSlice& fileContent,
    StringBuilder& outCodeBuilder)
{
    outCodeBuilder << "#line 1 \"";
    auto handler = StringEscapeUtil::getHandler(StringEscapeUtil::Style::Cpp);
    handler->appendEscaped(filePath, outCodeBuilder);
    outCodeBuilder << "\"\n";
    outCodeBuilder << fileContent << "\n";
}

void trackGLSLTargetCaps(ShaderExtensionTracker* extensionTracker, CapabilitySet const& caps)
{
    for (auto& conjunctions : caps.getAtomSets())
    {
        for (auto atom : conjunctions)
        {
            switch (asAtom(atom))
            {
            default:
                break;

            case CapabilityAtom::glsl_spirv_1_0:
                extensionTracker->requireSPIRVVersion(SemanticVersion(1, 0));
                break;
            case CapabilityAtom::glsl_spirv_1_1:
                extensionTracker->requireSPIRVVersion(SemanticVersion(1, 1));
                break;
            case CapabilityAtom::glsl_spirv_1_2:
                extensionTracker->requireSPIRVVersion(SemanticVersion(1, 2));
                break;
            case CapabilityAtom::glsl_spirv_1_3:
                extensionTracker->requireSPIRVVersion(SemanticVersion(1, 3));
                break;
            case CapabilityAtom::glsl_spirv_1_4:
                extensionTracker->requireSPIRVVersion(SemanticVersion(1, 4));
                break;
            case CapabilityAtom::glsl_spirv_1_5:
                extensionTracker->requireSPIRVVersion(SemanticVersion(1, 5));
                break;
            case CapabilityAtom::glsl_spirv_1_6:
                extensionTracker->requireSPIRVVersion(SemanticVersion(1, 6));
                break;
            }
        }
    }
}

SlangResult CodeGenContext::requireTranslationUnitSourceFiles()
{
    if (auto endToEndReq = isPassThroughEnabled())
    {
        for (auto entryPointIndex : getEntryPointIndices())
        {
            auto translationUnit = getPassThroughTranslationUnit(endToEndReq, entryPointIndex);
            SLANG_ASSERT(translationUnit);
            /// Make sure we have the source files
            SLANG_RETURN_ON_FAIL(translationUnit->requireSourceFiles());
        }
    }

    return SLANG_OK;
}

#if SLANG_VC
// TODO(JS): This is a workaround
// In debug VS builds there is a warning on line about it being unreachable.
// for (auto entryPointIndex : getEntryPointIndices())
// It's not clear how that could possibly be unreachable
#pragma warning(push)
#pragma warning(disable : 4702)
#endif
SlangResult CodeGenContext::emitEntryPointsSource(ComPtr<IArtifact>& outArtifact)
{
    outArtifact.setNull();

    SLANG_RETURN_ON_FAIL(requireTranslationUnitSourceFiles());

    auto endToEndReq = isPassThroughEnabled();
    if (endToEndReq)
    {
        for (auto entryPointIndex : getEntryPointIndices())
        {
            auto translationUnit = getPassThroughTranslationUnit(endToEndReq, entryPointIndex);
            SLANG_ASSERT(translationUnit);

            /// Make sure we have the source files
            SLANG_RETURN_ON_FAIL(translationUnit->requireSourceFiles());

            // Generate a string that includes the content of
            // the source file(s), along with a line directive
            // to ensure that we get reasonable messages
            // from the downstream compiler when in pass-through
            // mode.

            StringBuilder codeBuilder;
            if (getTargetFormat() == CodeGenTarget::GLSL)
            {
                // Special case GLSL
                int translationUnitCounter = 0;
                for (auto sourceFile : translationUnit->getSourceFiles())
                {
                    int translationUnitIndex = translationUnitCounter++;

                    // We want to output `#line` directives, but we need
                    // to skip this for the first file, since otherwise
                    // some GLSL implementations will get tripped up by
                    // not having the `#version` directive be the first
                    // thing in the file.
                    if (translationUnitIndex != 0)
                    {
                        codeBuilder << "#line 1 " << translationUnitIndex << "\n";
                    }
                    codeBuilder << sourceFile->getContent() << "\n";
                }
            }
            else
            {
                for (auto sourceFile : translationUnit->getSourceFiles())
                {
                    _appendCodeWithPath(
                        sourceFile->getPathInfo().foundPath.getUnownedSlice(),
                        sourceFile->getContent(),
                        codeBuilder);
                }
            }

            auto artifact =
                ArtifactUtil::createArtifactForCompileTarget(asExternal(getTargetFormat()));
            artifact->addRepresentationUnknown(StringBlob::moveCreate(codeBuilder));

            outArtifact.swap(artifact);
            return SLANG_OK;
        }
        return SLANG_OK;
    }
    else
    {
        return emitEntryPointsSourceFromIR(outArtifact);
    }
}
#if SLANG_VC
#pragma warning(pop)
#endif

SlangResult CodeGenContext::emitPrecompiledDownstreamIR(ComPtr<IArtifact>& outArtifact)
{
    return _emitEntryPoints(outArtifact);
}

String GetHLSLProfileName(Profile profile)
{
    switch (profile.getFamily())
    {
    case ProfileFamily::DX:
        // Profile version is a DX one, so stick with it.
        break;

    default:
        // Profile is a non-DX profile family, so we need to try
        // to clobber it with something to get a default.
        //
        // TODO: This is a huge hack...
        profile.setVersion(ProfileVersion::DX_5_1);
        break;
    }

    char const* stagePrefix = nullptr;
    switch (profile.getStage())
    {
        // Note: All of the raytracing-related stages require
        // compiling for a `lib_*` profile, even when only a
        // single entry point is present.
        //
        // We also go ahead and use this target in any case
        // where we don't know the actual stage to compiel for,
        // as a fallback option.
        //
        // TODO: We also want to use this option when compiling
        // multiple entry points to a DXIL library.
        //
    default:
        stagePrefix = "lib";
        break;

        // The traditional rasterization pipeline and compute
        // shaders all have custom profile names that identify
        // both the stage and shader model, which need to be
        // used when compiling a single entry point.
        //
#define CASE(NAME, PREFIX)     \
    case Stage::NAME:          \
        stagePrefix = #PREFIX; \
        break
        CASE(Vertex, vs);
        CASE(Hull, hs);
        CASE(Domain, ds);
        CASE(Geometry, gs);
        CASE(Fragment, ps);
        CASE(Compute, cs);
        CASE(Amplification, as);
        CASE(Mesh, ms);
#undef CASE
    }

    char const* versionSuffix = nullptr;
    switch (profile.getVersion())
    {
#define CASE(TAG, SUFFIX)        \
    case ProfileVersion::TAG:    \
        versionSuffix = #SUFFIX; \
        break
        CASE(DX_4_0, _4_0);
        CASE(DX_4_1, _4_1);
        CASE(DX_5_0, _5_0);
        CASE(DX_5_1, _5_1);
        CASE(DX_6_0, _6_0);
        CASE(DX_6_1, _6_1);
        CASE(DX_6_2, _6_2);
        CASE(DX_6_3, _6_3);
        CASE(DX_6_4, _6_4);
        CASE(DX_6_5, _6_5);
        CASE(DX_6_6, _6_6);
        CASE(DX_6_7, _6_7);
        CASE(DX_6_8, _6_8);
        CASE(DX_6_9, _6_9);
#undef CASE

    default:
        return "unknown";
    }

    String result;
    result.append(stagePrefix);
    result.append(versionSuffix);
    return result;
}

void reportExternalCompileError(
    const char* compilerName,
    Severity severity,
    SlangResult res,
    const UnownedStringSlice& diagnostic,
    DiagnosticSink* sink)
{
    StringBuilder builder;
    if (compilerName)
    {
        builder << compilerName << ": ";
    }

    if (SLANG_FAILED(res) && res != SLANG_FAIL)
    {
        {
            char tmp[17];
            sprintf_s(tmp, SLANG_COUNT_OF(tmp), "0x%08x", uint32_t(res));
            builder << "Result(" << tmp << ") ";
        }

        PlatformUtil::appendResult(res, builder);
    }

    if (diagnostic.getLength() > 0)
    {
        builder.append(diagnostic);
        if (!diagnostic.endsWith("\n"))
        {
            builder.append("\n");
        }
    }

    sink->diagnoseRaw(severity, builder.getUnownedSlice());
}

void reportExternalCompileError(
    const char* compilerName,
    SlangResult res,
    const UnownedStringSlice& diagnostic,
    DiagnosticSink* sink)
{
    // TODO(tfoley): need a better policy for how we translate diagnostics
    // back into the Slang world (although we should always try to generate
    // HLSL that doesn't produce any diagnostics...)
    reportExternalCompileError(
        compilerName,
        SLANG_FAILED(res) ? Severity::Error : Severity::Warning,
        res,
        diagnostic,
        sink);
}

static String _getDisplayPath(DiagnosticSink* sink, SourceFile* sourceFile)
{
    if (sink->isFlagSet(DiagnosticSink::Flag::VerbosePath))
    {
        return sourceFile->calcVerbosePath();
    }
    else
    {
        return sourceFile->getPathInfo().foundPath;
    }
}

String CodeGenContext::calcSourcePathForEntryPoints()
{
    String failureMode = "slang-generated";
    if (getEntryPointCount() != 1)
        return failureMode;
    auto entryPointIndex = getSingleEntryPointIndex();
    auto translationUnitRequest = findPassThroughTranslationUnit(entryPointIndex);
    if (!translationUnitRequest)
        return failureMode;

    const auto& sourceFiles = translationUnitRequest->getSourceFiles();

    auto sink = getSink();

    const Index numSourceFiles = sourceFiles.getCount();

    switch (numSourceFiles)
    {
    case 0:
        return "unknown";
    case 1:
        return _getDisplayPath(sink, sourceFiles[0]);
    default:
        {
            StringBuilder builder;
            builder << _getDisplayPath(sink, sourceFiles[0]);
            for (int i = 1; i < numSourceFiles; ++i)
            {
                builder << ";" << _getDisplayPath(sink, sourceFiles[i]);
            }
            return builder;
        }
    }
}

// Helper function for cases where we can assume a single entry point
Int assertSingleEntryPoint(List<Int> const& entryPointIndices)
{
    SLANG_ASSERT(entryPointIndices.getCount() == 1);
    return *entryPointIndices.begin();
}

// True if it's best to use 'emitted' source for complication. For a downstream compiler
// that is not file based, this is always ok.
///
/// If the downstream compiler is file system based, we may want to just use the file that was
/// passed to be compiled. That the downstream compiler can determine if it will then save the file
/// or not based on if it's a match - and generally there will not be a match with emitted source.
///
/// This test is only used for pass through mode.
static bool _useEmittedSource(
    IDownstreamCompiler* compiler,
    TranslationUnitRequest* translationUnit)
{
    // We only bother if it's a file based compiler.
    if (compiler->isFileBased())
    {
        // It can only have *one* source file as otherwise we have to combine to make a new source
        // file anyway
        return translationUnit->getSourceArtifacts().getCount() != 1;
    }
    return true;
}

static Severity _getDiagnosticSeverity(ArtifactDiagnostic::Severity severity)
{
    switch (severity)
    {
    case ArtifactDiagnostic::Severity::Warning:
        return Severity::Warning;
    case ArtifactDiagnostic::Severity::Info:
        return Severity::Note;
    default:
        return Severity::Error;
    }
}

static RefPtr<ExtensionTracker> _newExtensionTracker(CodeGenTarget target)
{
    switch (target)
    {
    case CodeGenTarget::PTX:
    case CodeGenTarget::CUDASource:
        {
            return new CUDAExtensionTracker;
        }
    case CodeGenTarget::SPIRV:
    case CodeGenTarget::GLSL:
    case CodeGenTarget::WGSL:
    case CodeGenTarget::WGSLSPIRV:
    case CodeGenTarget::WGSLSPIRVAssembly:
        {
            return new ShaderExtensionTracker;
        }
    default:
        return nullptr;
    }
}

static CodeGenTarget _getDefaultSourceForTarget(CodeGenTarget target)
{
    switch (target)
    {
    case CodeGenTarget::ShaderHostCallable:
    case CodeGenTarget::ShaderSharedLibrary:
        {
            return CodeGenTarget::CPPSource;
        }
    case CodeGenTarget::HostHostCallable:
    case CodeGenTarget::HostExecutable:
    case CodeGenTarget::HostSharedLibrary:
        {
            return CodeGenTarget::HostCPPSource;
        }
    case CodeGenTarget::PTX:
        return CodeGenTarget::CUDASource;
    case CodeGenTarget::DXBytecode:
        return CodeGenTarget::HLSL;
    case CodeGenTarget::DXIL:
        return CodeGenTarget::HLSL;
    case CodeGenTarget::SPIRV:
        return CodeGenTarget::GLSL;
    case CodeGenTarget::MetalLib:
        return CodeGenTarget::Metal;
    case CodeGenTarget::WGSLSPIRV:
        return CodeGenTarget::WGSL;
    default:
        break;
    }
    return CodeGenTarget::Unknown;
}

static bool _isCPUHostTarget(CodeGenTarget target)
{
    auto desc = ArtifactDescUtil::makeDescForCompileTarget(asExternal(target));
    return desc.style == ArtifactStyle::Host;
}

static bool _shouldSetEntryPointName(TargetProgram* targetProgram)
{
    if (!isKhronosTarget(targetProgram->getTargetReq()))
        return true;
    if (targetProgram->getOptionSet().getBoolOption(CompilerOptionName::VulkanUseEntryPointName))
        return true;
    return false;
}

SlangResult passthroughDownstreamDiagnostics(
    DiagnosticSink* sink,
    IDownstreamCompiler* compiler,
    IArtifact* artifact)
{
    auto diagnostics = findAssociatedRepresentation<IArtifactDiagnostics>(artifact);

    if (!diagnostics)
        return SLANG_OK;

    if (diagnostics->getCount())
    {
        StringBuilder compilerText;
        DownstreamCompilerUtil::appendAsText(compiler->getDesc(), compilerText);

        StringBuilder builder;

        auto const diagnosticCount = diagnostics->getCount();
        for (Index i = 0; i < diagnosticCount; ++i)
        {
            const auto& diagnostic = *diagnostics->getAt(i);

            builder.clear();

            const Severity severity = _getDiagnosticSeverity(diagnostic.severity);

            if (diagnostic.filePath.count == 0 && diagnostic.location.line == 0 &&
                severity == Severity::Note)
            {
                // If theres no filePath line number and it's info, output severity and text alone
                builder << getSeverityName(severity) << " : ";
            }
            else
            {
                if (diagnostic.filePath.count)
                {
                    builder << asStringSlice(diagnostic.filePath);
                }

                if (diagnostic.location.line)
                {
                    builder << "(" << diagnostic.location.line << ")";
                }

                builder << ": ";

                if (diagnostic.stage == ArtifactDiagnostic::Stage::Link)
                {
                    builder << "link ";
                }

                builder << getSeverityName(severity);
                builder << " " << asStringSlice(diagnostic.code) << ": ";
            }

            builder << asStringSlice(diagnostic.text);
            reportExternalCompileError(
                compilerText.getBuffer(),
                severity,
                SLANG_OK,
                builder.getUnownedSlice(),
                sink);
        }
    }

    // If any errors are emitted, then we are done
    if (diagnostics->hasOfAtLeastSeverity(ArtifactDiagnostic::Severity::Error))
    {
        return SLANG_FAIL;
    }

    return SLANG_OK;
}

SlangResult CodeGenContext::emitWithDownstreamForEntryPoints(ComPtr<IArtifact>& outArtifact)
{
    outArtifact.setNull();

    auto sink = getSink();
    auto session = getSession();

    CodeGenTarget sourceTarget = CodeGenTarget::None;
    SourceLanguage sourceLanguage = SourceLanguage::Unknown;

    auto target = getTargetFormat();
    RefPtr<ExtensionTracker> extensionTracker = _newExtensionTracker(target);
    PassThroughMode compilerType;

    SliceAllocator allocator;

    if (auto endToEndReq = isPassThroughEnabled())
    {
        compilerType = endToEndReq->m_passThrough;
    }
    else
    {
        // If we are not in pass through, lookup the default compiler for the emitted source type

        // Get the default source codegen type for a given target
        sourceTarget = _getDefaultSourceForTarget(target);
        compilerType = (PassThroughMode)session->getDownstreamCompilerForTransition(
            (SlangCompileTarget)sourceTarget,
            (SlangCompileTarget)target);
        // We should have a downstream compiler set at this point
        if (compilerType == PassThroughMode::None)
        {
            auto sourceName = TypeTextUtil::getCompileTargetName(SlangCompileTarget(sourceTarget));
            auto targetName = TypeTextUtil::getCompileTargetName(SlangCompileTarget(target));

            sink->diagnose(
                SourceLoc(),
                Diagnostics::compilerNotDefinedForTransition,
                sourceName,
                targetName);
            return SLANG_FAIL;
        }
    }

    SLANG_ASSERT(compilerType != PassThroughMode::None);

    // Get the required downstream compiler
    IDownstreamCompiler* compiler = session->getOrLoadDownstreamCompiler(compilerType, sink);
    if (!compiler)
    {
        auto compilerName = TypeTextUtil::getPassThroughAsHumanText((SlangPassThrough)compilerType);
        sink->diagnose(SourceLoc(), Diagnostics::passThroughCompilerNotFound, compilerName);
        return SLANG_FAIL;
    }

    Dictionary<String, String> preprocessorDefinitions;
    List<String> includePaths;

    typedef DownstreamCompileOptions CompileOptions;
    CompileOptions options;

    List<DownstreamCompileOptions::CapabilityVersion> requiredCapabilityVersions;
    List<String> compilerSpecificArguments;
    List<ComPtr<IArtifact>> libraries;
    List<String> libraryPaths;

    // Set compiler specific args
    {
        auto name = TypeTextUtil::getPassThroughName((SlangPassThrough)compilerType);
        List<String> downstreamArgs = getTargetProgram()->getOptionSet().getDownstreamArgs(name);
        for (const auto& arg : downstreamArgs)
        {
            // We special case some kinds of args, that can be handled directly
            if (arg.startsWith("-I"))
            {
                // We handle the -I option, by just adding to the include paths
                includePaths.add(arg.getUnownedSlice().tail(2));
            }
            else
            {
                compilerSpecificArguments.add(arg);
            }
        }
    }

    ComPtr<IArtifact> sourceArtifact;

    /* This is more convoluted than the other scenarios, because when we invoke C/C++ compiler we
    would ideally like to use the original file. We want to do this because we want includes
    relative to the source file to work, and for that to work most easily we want to use the
    original file, if there is one */
    if (auto endToEndReq = isPassThroughEnabled())
    {
        // If we are pass through, we may need to set extension tracker state.
        if (ShaderExtensionTracker* glslTracker = as<ShaderExtensionTracker>(extensionTracker))
        {
            trackGLSLTargetCaps(glslTracker, getTargetCaps());
        }

        auto translationUnit =
            getPassThroughTranslationUnit(endToEndReq, getSingleEntryPointIndex());

        // We are just passing thru, so it's whatever it originally was
        sourceLanguage = translationUnit->sourceLanguage;

        // TODO(JS): This seems like a bit of a hack
        // That if a pass-through is being performed and the source language is Slang
        // no downstream compiler knows how to deal with that, so probably means 'HLSL'
        sourceLanguage =
            (sourceLanguage == SourceLanguage::Slang) ? SourceLanguage::HLSL : sourceLanguage;
        sourceTarget = CodeGenTarget(TypeConvertUtil::getCompileTargetFromSourceLanguage(
            (SlangSourceLanguage)sourceLanguage));

        // If it's pass through we accumulate the preprocessor definitions.
        for (const auto& define :
             endToEndReq->getOptionSet().getArray(CompilerOptionName::MacroDefine))
            preprocessorDefinitions.add(define.stringValue, define.stringValue2);
        for (const auto& define : translationUnit->preprocessorDefinitions)
            preprocessorDefinitions.add(define);

        {
            /* TODO(JS): Not totally clear what options should be set here. If we are using the pass
            through - then using say the defines/includes all makes total sense. If we are
            generating C++ code from slang, then should we really be using these values -> aren't
            they what is being set for the *slang* source, not for the C++ generated code. That
            being the case it implies that there needs to be a mechanism (if there isn't already) to
            specify such information on a particular pass/pass through etc.

            On invoking DXC for example include paths do not appear to be set at all (even with
            pass-through).
            */

            auto linkage = getLinkage();

            // Add all the search paths

            const auto searchDirectories = linkage->getSearchDirectories();
            const SearchDirectoryList* searchList = &searchDirectories;
            while (searchList)
            {
                for (const auto& searchDirectory : searchList->searchDirectories)
                {
                    includePaths.add(searchDirectory.path);
                }
                searchList = searchList->parent;
            }
        }

        // If emitted source is required, emit and set the path
        if (_useEmittedSource(compiler, translationUnit))
        {
            CodeGenContext sourceCodeGenContext(this, sourceTarget, extensionTracker);

            SLANG_RETURN_ON_FAIL(sourceCodeGenContext.emitEntryPointsSource(sourceArtifact));

            // If it's not file based we can set an appropriate path name, and it doesn't matter if
            // it doesn't exist on the file system. We set the name to the path as this will be used
            // for downstream reporting.
            auto sourcePath = calcSourcePathForEntryPoints();
            sourceArtifact->setName(sourcePath.getBuffer());

            sourceCodeGenContext.maybeDumpIntermediate(sourceArtifact);
        }
        else
        {
            // Special case if we have a single file, so that we pass the path, and the contents as
            // is.
            const auto& sourceArtifacts = translationUnit->getSourceArtifacts();
            SLANG_ASSERT(sourceArtifacts.getCount() == 1);

            sourceArtifact = sourceArtifacts[0];
            SLANG_ASSERT(sourceArtifact);
        }
    }
    else
    {
        CodeGenContext sourceCodeGenContext(this, sourceTarget, extensionTracker);

        sourceCodeGenContext.removeAvailableInDownstreamIR = true;

        SLANG_RETURN_ON_FAIL(sourceCodeGenContext.emitEntryPointsSource(sourceArtifact));
        sourceCodeGenContext.maybeDumpIntermediate(sourceArtifact);

        sourceLanguage = (SourceLanguage)TypeConvertUtil::getSourceLanguageFromTarget(
            (SlangCompileTarget)sourceTarget);
    }

    if (sourceArtifact)
    {
        // Set the source artifacts
        options.sourceArtifacts = makeSlice(sourceArtifact.readRef(), 1);
    }

    // Add any preprocessor definitions associated with the linkage
    {
        // TODO(JS): This is somewhat arguable - should defines passed to Slang really be
        // passed to downstream compilers? It does appear consistent with the behavior if
        // there is an endToEndReq.
        //
        // That said it's very convenient and provides way to control aspects
        // of downstream compilation.

        for (const auto& define :
             getTargetProgram()->getOptionSet().getArray(CompilerOptionName::MacroDefine))
        {
            preprocessorDefinitions.addIfNotExists(define.stringValue, define.stringValue2);
        }
    }


    // If we have an extension tracker, we may need to set options such as SPIR-V version
    // and CUDA Shader Model.
    if (extensionTracker)
    {
        // Look for the version
        if (auto cudaTracker = as<CUDAExtensionTracker>(extensionTracker))
        {
            cudaTracker->finalize();

            if (cudaTracker->m_smVersion.isSet())
            {
                DownstreamCompileOptions::CapabilityVersion version;
                version.kind = DownstreamCompileOptions::CapabilityVersion::Kind::CUDASM;
                version.version = cudaTracker->m_smVersion;

                requiredCapabilityVersions.add(version);
            }

            if (cudaTracker->isBaseTypeRequired(BaseType::Half))
            {
                options.flags |= CompileOptions::Flag::EnableFloat16;
            }
        }
        else if (ShaderExtensionTracker* glslTracker = as<ShaderExtensionTracker>(extensionTracker))
        {
            DownstreamCompileOptions::CapabilityVersion version;
            version.kind = DownstreamCompileOptions::CapabilityVersion::Kind::SPIRV;
            version.version = glslTracker->getSPIRVVersion();

            requiredCapabilityVersions.add(version);
        }
    }

    // Set the file sytem and source manager, as *may* be used by downstream compiler
    options.fileSystemExt = getFileSystemExt();
    options.sourceManager = getSourceManager();

    // Set the source type
    options.sourceLanguage = SlangSourceLanguage(sourceLanguage);

    switch (target)
    {
    case CodeGenTarget::ShaderHostCallable:
    case CodeGenTarget::ShaderSharedLibrary:
        // Disable exceptions and security checks
        options.flags &=
            ~(CompileOptions::Flag::EnableExceptionHandling |
              CompileOptions::Flag::EnableSecurityChecks);
        break;
    }

    Profile profile;

    if (compilerType == PassThroughMode::Fxc || compilerType == PassThroughMode::Dxc ||
        compilerType == PassThroughMode::Glslang)
    {
        const auto entryPointIndices = getEntryPointIndices();
        auto targetReq = getTargetReq();

        const auto entryPointIndicesCount = entryPointIndices.getCount();

        // Whole program means
        // * can have 0-N entry points
        // * 'doesn't build into an executable/kernel'
        //
        // So in some sense it is a library
        if (getTargetProgram()->getOptionSet().getBoolOption(
                CompilerOptionName::GenerateWholeProgram))
        {
            if (compilerType == PassThroughMode::Dxc)
            {
                // Can support no entry points on DXC because we can build libraries
                profile =
                    Profile(getTargetProgram()->getOptionSet().getEnumOption<Profile::RawEnum>(
                        CompilerOptionName::Profile));
            }
            else
            {
                auto downstreamCompilerName =
                    TypeTextUtil::getPassThroughName((SlangPassThrough)compilerType);

                sink->diagnose(
                    SourceLoc(),
                    Diagnostics::downstreamCompilerDoesntSupportWholeProgramCompilation,
                    downstreamCompilerName);
                return SLANG_FAIL;
            }
        }
        else if (entryPointIndicesCount == 1)
        {
            // All support a single entry point
            const Index entryPointIndex = entryPointIndices[0];

            auto entryPoint = getEntryPoint(entryPointIndex);
            profile = getEffectiveProfile(entryPoint, targetReq);

            if (_shouldSetEntryPointName(getTargetProgram()))
            {
                options.entryPointName = allocator.allocate(getText(entryPoint->getName()));
                auto entryPointNameOverride =
                    getProgram()->getEntryPointNameOverride(entryPointIndex);
                if (entryPointNameOverride.getLength() != 0)
                {
                    options.entryPointName = allocator.allocate(entryPointNameOverride);
                }
            }
        }
        else
        {
            // We only support a single entry point on this target
            SLANG_ASSERT(!"Can only compile with a single entry point on this target");
            return SLANG_FAIL;
        }

        options.stage = SlangStage(profile.getStage());

        if (compilerType == PassThroughMode::Dxc)
        {
            // We will enable the flag to generate proper code for 16 - bit types
            // by default, as long as the user is requesting a sufficiently
            // high shader model.
            //
            // TODO: Need to check that this is safe to enable in all cases,
            // or if it will make a shader demand hardware features that
            // aren't always present.
            //
            // TODO: Ideally the dxc back-end should be passed some information
            // on the "capabilities" that were used and/or requested in the code.
            //
            if (profile.getVersion() >= ProfileVersion::DX_6_2)
            {
                options.flags |= CompileOptions::Flag::EnableFloat16;
            }

            // Set the matrix layout
            options.matrixLayout =
                (SlangMatrixLayoutMode)getTargetProgram()->getOptionSet().getMatrixLayoutMode();
        }

        // Set the profile
        options.profileName = allocator.allocate(GetHLSLProfileName(profile));
    }

    // If we aren't using LLVM 'host callable', we want downstream compile to produce a shared
    // library
    if (compilerType != PassThroughMode::LLVM &&
        ArtifactDescUtil::makeDescForCompileTarget(asExternal(target)).kind ==
            ArtifactKind::HostCallable)
    {
        target = CodeGenTarget::ShaderSharedLibrary;
    }

    if (!isPassThroughEnabled())
    {
        if (_isCPUHostTarget(target))
        {
            libraryPaths.add(Path::getParentDirectory(Path::getExecutablePath()));
            libraryPaths.add(
                Path::combine(Path::getParentDirectory(Path::getExecutablePath()), "../lib"));

            // Set up the library artifact
            auto artifact = Artifact::create(
                ArtifactDesc::make(ArtifactKind::Library, Artifact::Payload::HostCPU),
                toSlice("slang-rt"));

            ComPtr<IOSFileArtifactRepresentation> fileRep(new OSFileArtifactRepresentation(
                IOSFileArtifactRepresentation::Kind::NameOnly,
                toSlice("slang-rt"),
                nullptr));
            artifact->addRepresentation(fileRep);

            libraries.add(artifact);
        }
    }

    options.targetType = (SlangCompileTarget)target;

    // Need to configure for the compilation

    {
        auto linkage = getLinkage();

        switch (getTargetProgram()->getOptionSet().getEnumOption<OptimizationLevel>(
            CompilerOptionName::Optimization))
        {
        case OptimizationLevel::None:
            options.optimizationLevel = DownstreamCompileOptions::OptimizationLevel::None;
            break;
        case OptimizationLevel::Default:
            options.optimizationLevel = DownstreamCompileOptions::OptimizationLevel::Default;
            break;
        case OptimizationLevel::High:
            options.optimizationLevel = DownstreamCompileOptions::OptimizationLevel::High;
            break;
        case OptimizationLevel::Maximal:
            options.optimizationLevel = DownstreamCompileOptions::OptimizationLevel::Maximal;
            break;
        default:
            SLANG_ASSERT(!"Unhandled optimization level");
            break;
        }

        switch (getTargetProgram()->getOptionSet().getEnumOption<DebugInfoLevel>(
            CompilerOptionName::DebugInformation))
        {
        case DebugInfoLevel::None:
            options.debugInfoType = DownstreamCompileOptions::DebugInfoType::None;
            break;
        case DebugInfoLevel::Minimal:
            options.debugInfoType = DownstreamCompileOptions::DebugInfoType::Minimal;
            break;

        case DebugInfoLevel::Standard:
            options.debugInfoType = DownstreamCompileOptions::DebugInfoType::Standard;
            break;
        case DebugInfoLevel::Maximal:
            options.debugInfoType = DownstreamCompileOptions::DebugInfoType::Maximal;
            break;
        default:
            SLANG_ASSERT(!"Unhandled debug level");
            break;
        }

        switch (getTargetProgram()->getOptionSet().getEnumOption<FloatingPointMode>(
            CompilerOptionName::FloatingPointMode))
        {
        case FloatingPointMode::Default:
            options.floatingPointMode = DownstreamCompileOptions::FloatingPointMode::Default;
            break;
        case FloatingPointMode::Precise:
            options.floatingPointMode = DownstreamCompileOptions::FloatingPointMode::Precise;
            break;
        case FloatingPointMode::Fast:
            options.floatingPointMode = DownstreamCompileOptions::FloatingPointMode::Fast;
            break;
        default:
            SLANG_ASSERT(!"Unhandled floating point mode");
        }

        {
            // We need to look at the stage of the entry point(s) we are
            // being asked to compile, since this will determine the
            // "pipeline" that the result should be compiled for (e.g.,
            // compute vs. ray tracing).
            //
            // TODO: This logic is kind of messy in that it assumes
            // a program to be compiled will only contain kernels for
            // a single pipeline type, but that invariant isn't expressed
            // at all in the front-end today. It also has no error
            // checking for the case where there are conflicts.
            //
            // HACK: Right now none of the above concerns matter
            // because we always perform code generation on a single
            // entry point at a time.
            //
            Index entryPointCount = getEntryPointCount();
            for (Index ee = 0; ee < entryPointCount; ++ee)
            {
                auto stage = getEntryPoint(ee)->getStage();
                switch (stage)
                {
                default:
                    break;

                case Stage::Compute:
                    options.pipelineType = DownstreamCompileOptions::PipelineType::Compute;
                    break;

                case Stage::Vertex:
                case Stage::Hull:
                case Stage::Domain:
                case Stage::Geometry:
                case Stage::Fragment:
                    options.pipelineType = DownstreamCompileOptions::PipelineType::Rasterization;
                    break;

                case Stage::RayGeneration:
                case Stage::Intersection:
                case Stage::AnyHit:
                case Stage::ClosestHit:
                case Stage::Miss:
                case Stage::Callable:
                    options.pipelineType = DownstreamCompileOptions::PipelineType::RayTracing;
                    break;
                }
            }
        }

        // Add all the search paths (as calculated earlier - they will only be set if this is a pass
        // through else will be empty)
        options.includePaths = allocator.allocate(includePaths);

        // Add the specified defines (as calculated earlier - they will only be set if this is a
        // pass through else will be empty)
        {
            const auto count = preprocessorDefinitions.getCount();
            auto dst = allocator.getArena().allocateArray<DownstreamCompileOptions::Define>(count);

            Index i = 0;

            for (const auto& [defKey, defValue] : preprocessorDefinitions)
            {
                auto& define = dst[i];

                define.nameWithSig = allocator.allocate(defKey);
                define.value = allocator.allocate(defValue);

                ++i;
            }
            options.defines = makeSlice(dst, count);
        }

        // Add all of the module libraries
        libraries.addRange(linkage->m_libModules.getBuffer(), linkage->m_libModules.getCount());
    }

    auto program = getProgram();

    // Load embedded precompiled libraries from IR into library artifacts
    program->enumerateIRModules(
        [&](IRModule* irModule)
        {
            for (auto globalInst : irModule->getModuleInst()->getChildren())
            {
                if (target == CodeGenTarget::DXILAssembly || target == CodeGenTarget::DXIL)
                {
                    if (auto inst = as<IREmbeddedDownstreamIR>(globalInst))
                    {
                        if (inst->getTarget() == CodeGenTarget::DXIL)
                        {
                            auto slice = inst->getBlob()->getStringSlice();
                            ArtifactDesc desc =
                                ArtifactDescUtil::makeDescForCompileTarget(SLANG_DXIL);
                            desc.kind = ArtifactKind::Library;

                            auto library = ArtifactUtil::createArtifact(desc);

                            library->addRepresentationUnknown(StringBlob::create(slice));
                            libraries.add(library);
                        }
                    }
                }
            }
        });

    options.compilerSpecificArguments = allocator.allocate(compilerSpecificArguments);
    options.requiredCapabilityVersions = SliceUtil::asSlice(requiredCapabilityVersions);
    options.libraries = SliceUtil::asSlice(libraries);
    options.libraryPaths = allocator.allocate(libraryPaths);

    if (m_targetProfile.getFamily() == ProfileFamily::DX)
    {
        options.enablePAQ = m_targetProfile.getVersion() >= ProfileVersion::DX_6_7;
    }

    // Compile
    ComPtr<IArtifact> artifact;
    auto downstreamStartTime = std::chrono::high_resolution_clock::now();
    SLANG_RETURN_ON_FAIL(compiler->compile(options, artifact.writeRef()));
    auto downstreamElapsedTime =
        (std::chrono::high_resolution_clock::now() - downstreamStartTime).count() * 0.000000001;
    getSession()->addDownstreamCompileTime(downstreamElapsedTime);

    SLANG_RETURN_ON_FAIL(passthroughDownstreamDiagnostics(getSink(), compiler, artifact));

    // Copy over all of the information associated with the source into the output
    if (sourceArtifact)
    {
        for (auto associatedArtifact : sourceArtifact->getAssociated())
        {
            artifact->addAssociated(associatedArtifact);
        }
    }

    // Set the artifact
    outArtifact.swap(artifact);
    return SLANG_OK;
}

SlangResult emitSPIRVForEntryPointsDirectly(
    CodeGenContext* codeGenContext,
    ComPtr<IArtifact>& outArtifact);

SlangResult emitHostVMCode(CodeGenContext* codeGenContext, ComPtr<IArtifact>& outArtifact);

static CodeGenTarget _getIntermediateTarget(CodeGenTarget target)
{
    switch (target)
    {
    case CodeGenTarget::DXBytecodeAssembly:
        return CodeGenTarget::DXBytecode;
    case CodeGenTarget::DXILAssembly:
        return CodeGenTarget::DXIL;
    case CodeGenTarget::SPIRVAssembly:
        return CodeGenTarget::SPIRV;
    case CodeGenTarget::WGSLSPIRVAssembly:
        return CodeGenTarget::WGSLSPIRV;
    default:
        return CodeGenTarget::None;
    }
}

/// Function to simplify the logic around emitting, and dissassembling
SlangResult CodeGenContext::_emitEntryPoints(ComPtr<IArtifact>& outArtifact)
{
    auto target = getTargetFormat();
    switch (target)
    {
    case CodeGenTarget::SPIRVAssembly:
    case CodeGenTarget::DXBytecodeAssembly:
    case CodeGenTarget::DXILAssembly:
    case CodeGenTarget::MetalLibAssembly:
    case CodeGenTarget::WGSLSPIRVAssembly:
        {
            // First compile to an intermediate target for the corresponding binary format.
            const CodeGenTarget intermediateTarget = _getIntermediateTarget(target);
            CodeGenContext intermediateContext(this, intermediateTarget);

            ComPtr<IArtifact> intermediateArtifact;

            SLANG_RETURN_ON_FAIL(intermediateContext._emitEntryPoints(intermediateArtifact));
            intermediateContext.maybeDumpIntermediate(intermediateArtifact);

            // Then disassemble the intermediate binary result to get the desired output
            // Output the disassemble
            ComPtr<IArtifact> disassemblyArtifact;
            SLANG_RETURN_ON_FAIL(ArtifactOutputUtil::dissassembleWithDownstream(
                getSession(),
                intermediateArtifact,
                getSink(),
                disassemblyArtifact.writeRef()));

            outArtifact.swap(disassemblyArtifact);
            return SLANG_OK;
        }
    case CodeGenTarget::SPIRV:
        if (getTargetProgram()->getOptionSet().shouldEmitSPIRVDirectly())
        {
            SLANG_RETURN_ON_FAIL(emitSPIRVForEntryPointsDirectly(this, outArtifact));
            return SLANG_OK;
        }
        [[fallthrough]];
    case CodeGenTarget::DXIL:
    case CodeGenTarget::DXBytecode:
    case CodeGenTarget::MetalLib:
    case CodeGenTarget::PTX:
    case CodeGenTarget::ShaderHostCallable:
    case CodeGenTarget::ShaderSharedLibrary:
    case CodeGenTarget::HostExecutable:
    case CodeGenTarget::HostHostCallable:
    case CodeGenTarget::HostSharedLibrary:
    case CodeGenTarget::WGSLSPIRV:
        SLANG_RETURN_ON_FAIL(emitWithDownstreamForEntryPoints(outArtifact));
        return SLANG_OK;
    case CodeGenTarget::HostVM:
        SLANG_RETURN_ON_FAIL(emitHostVMCode(this, outArtifact));
        return SLANG_OK;
    default:
        break;
    }

    return SLANG_FAIL;
}

// Helper class for recording compile time.
struct CompileTimerRAII
{
    std::chrono::high_resolution_clock::time_point startTime;
    Session* session;
    CompileTimerRAII(Session* inSession)
    {
        startTime = std::chrono::high_resolution_clock::now();
        session = inSession;
    }
    ~CompileTimerRAII()
    {
        double elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(
                                 std::chrono::high_resolution_clock::now() - startTime)
                                 .count() /
                             1e6;
        session->addTotalCompileTime(elapsedTime);
    }
};

// Do emit logic for a zero or more entry points
SlangResult CodeGenContext::emitEntryPoints(ComPtr<IArtifact>& outArtifact)
{
    CompileTimerRAII recordCompileTime(getSession());

    auto target = getTargetFormat();

    switch (target)
    {
    case CodeGenTarget::SPIRVAssembly:
    case CodeGenTarget::DXBytecodeAssembly:
    case CodeGenTarget::DXILAssembly:
    case CodeGenTarget::SPIRV:
    case CodeGenTarget::DXIL:
    case CodeGenTarget::DXBytecode:
    case CodeGenTarget::MetalLib:
    case CodeGenTarget::MetalLibAssembly:
    case CodeGenTarget::PTX:
    case CodeGenTarget::HostHostCallable:
    case CodeGenTarget::ShaderHostCallable:
    case CodeGenTarget::ShaderSharedLibrary:
    case CodeGenTarget::HostExecutable:
    case CodeGenTarget::HostSharedLibrary:
    case CodeGenTarget::WGSLSPIRVAssembly:
    case CodeGenTarget::HostVM:
        {
            SLANG_RETURN_ON_FAIL(_emitEntryPoints(outArtifact));

            maybeDumpIntermediate(outArtifact);
            return SLANG_OK;
        }
        break;
    case CodeGenTarget::GLSL:
    case CodeGenTarget::HLSL:
    case CodeGenTarget::CUDASource:
    case CodeGenTarget::CPPSource:
    case CodeGenTarget::HostCPPSource:
    case CodeGenTarget::PyTorchCppBinding:
    case CodeGenTarget::CSource:
    case CodeGenTarget::Metal:
    case CodeGenTarget::WGSL:
        {
            RefPtr<ExtensionTracker> extensionTracker = _newExtensionTracker(target);

            CodeGenContext subContext(this, target, extensionTracker);

            ComPtr<IArtifact> sourceArtifact;

            SLANG_RETURN_ON_FAIL(subContext.emitEntryPointsSource(sourceArtifact));

            subContext.maybeDumpIntermediate(sourceArtifact);
            outArtifact = sourceArtifact;
            return SLANG_OK;
        }
        break;

    case CodeGenTarget::None:
        // The user requested no output
        return SLANG_OK;

        // Note(tfoley): We currently hit this case when compiling the core module
    case CodeGenTarget::Unknown:
        return SLANG_OK;

    default:
        SLANG_UNEXPECTED("unhandled code generation target");
        break;
    }
    return SLANG_FAIL;
}

void EndToEndCompileRequest::writeArtifactToStandardOutput(
    IArtifact* artifact,
    DiagnosticSink* sink)
{
    // If it's host callable it's not available to write to output
    if (isDerivedFrom(artifact->getDesc().kind, ArtifactKind::HostCallable))
    {
        return;
    }

    auto session = getSession();
    ArtifactOutputUtil::maybeConvertAndWrite(
        session,
        artifact,
        sink,
        toSlice("stdout"),
        getWriter(WriterChannel::StdOutput));
}

String EndToEndCompileRequest::_getWholeProgramPath(TargetRequest* targetReq)
{
    RefPtr<EndToEndCompileRequest::TargetInfo> targetInfo;
    if (m_targetInfos.tryGetValue(targetReq, targetInfo))
    {
        return targetInfo->wholeTargetOutputPath;
    }
    return String();
}

String EndToEndCompileRequest::_getEntryPointPath(TargetRequest* targetReq, Index entryPointIndex)
{
    // It is possible that we are dynamically discovering entry
    // points (using `[shader(...)]` attributes), so that there
    // might be entry points added to the program that did not
    // get paths specified via command-line options.
    //
    RefPtr<EndToEndCompileRequest::TargetInfo> targetInfo;
    if (m_targetInfos.tryGetValue(targetReq, targetInfo))
    {
        String outputPath;
        if (targetInfo->entryPointOutputPaths.tryGetValue(entryPointIndex, outputPath))
        {
            return outputPath;
        }
    }

    return String();
}

SlangResult EndToEndCompileRequest::_writeArtifact(const String& path, IArtifact* artifact)
{
    if (path.getLength() > 0)
    {
        SLANG_RETURN_ON_FAIL(ArtifactOutputUtil::writeToFile(artifact, getSink(), path));
    }
    else if (m_containerFormat == ContainerFormat::None)
    {
        // If we aren't writing to a container and we didn't write to a file, we can output to
        // standard output
        writeArtifactToStandardOutput(artifact, getSink());
    }
    return SLANG_OK;
}

SlangResult EndToEndCompileRequest::_maybeWriteArtifact(const String& path, IArtifact* artifact)
{
    // We don't have to do anything if there is no artifact
    if (!artifact)
    {
        return SLANG_OK;
    }

    // If embedding is enabled...
    if (m_sourceEmbedStyle != SourceEmbedUtil::Style::None)
    {
        SourceEmbedUtil::Options options;

        options.style = m_sourceEmbedStyle;
        options.variableName = m_sourceEmbedName;
        options.language = (SlangSourceLanguage)m_sourceEmbedLanguage;

        ComPtr<IArtifact> embeddedArtifact;
        SLANG_RETURN_ON_FAIL(SourceEmbedUtil::createEmbedded(artifact, options, embeddedArtifact));

        if (!embeddedArtifact)
        {
            return SLANG_FAIL;
        }
        SLANG_RETURN_ON_FAIL(
            _writeArtifact(SourceEmbedUtil::getPath(path, options), embeddedArtifact));
        return SLANG_OK;
    }
    else
    {
        SLANG_RETURN_ON_FAIL(_writeArtifact(path, artifact));
    }

    return SLANG_OK;
}

IArtifact* TargetProgram::_createWholeProgramResult(
    DiagnosticSink* sink,
    EndToEndCompileRequest* endToEndReq)
{
    // We want to call `emitEntryPoints` function to generate code that contains
    // all the entrypoints defined in `m_program`.
    // The current logic of `emitEntryPoints` takes a list of entry-point indices to
    // emit code for, so we construct such a list first.
    List<Int> entryPointIndices;

    m_entryPointResults.setCount(m_program->getEntryPointCount());
    entryPointIndices.setCount(m_program->getEntryPointCount());
    for (Index i = 0; i < entryPointIndices.getCount(); i++)
        entryPointIndices[i] = i;

    CodeGenContext::Shared sharedCodeGenContext(this, entryPointIndices, sink, endToEndReq);
    CodeGenContext codeGenContext(&sharedCodeGenContext);

    if (SLANG_FAILED(codeGenContext.emitEntryPoints(m_wholeProgramResult)))
    {
        return nullptr;
    }

    return m_wholeProgramResult;
}

IArtifact* TargetProgram::_createEntryPointResult(
    Int entryPointIndex,
    DiagnosticSink* sink,
    EndToEndCompileRequest* endToEndReq)
{
    // It is possible that entry points got added to the `Program`
    // *after* we created this `TargetProgram`, so there might be
    // a request for an entry point that we didn't allocate space for.
    //
    // TODO: Change the construction logic so that a `Program` is
    // constructed all at once rather than incrementally, to avoid
    // this problem.
    //
    if (entryPointIndex >= m_entryPointResults.getCount())
        m_entryPointResults.setCount(entryPointIndex + 1);


    CodeGenContext::EntryPointIndices entryPointIndices;
    entryPointIndices.add(entryPointIndex);

    CodeGenContext::Shared sharedCodeGenContext(this, entryPointIndices, sink, endToEndReq);
    CodeGenContext codeGenContext(&sharedCodeGenContext);

    codeGenContext.emitEntryPoints(m_entryPointResults[entryPointIndex]);

    return m_entryPointResults[entryPointIndex];
}

IArtifact* TargetProgram::getOrCreateWholeProgramResult(DiagnosticSink* sink)
{
    if (m_wholeProgramResult)
        return m_wholeProgramResult;

    // If we haven't yet computed a layout for this target
    // program, we need to make sure that is done before
    // code generation.
    //
    if (!getOrCreateIRModuleForLayout(sink))
    {
        return nullptr;
    }

    return _createWholeProgramResult(sink);
}

IArtifact* TargetProgram::getOrCreateEntryPointResult(Int entryPointIndex, DiagnosticSink* sink)
{
    if (entryPointIndex >= m_entryPointResults.getCount())
        m_entryPointResults.setCount(entryPointIndex + 1);

    if (IArtifact* artifact = m_entryPointResults[entryPointIndex])
        return artifact;

    // If we haven't yet computed a layout for this target
    // program, we need to make sure that is done before
    // code generation.
    //
    if (!getOrCreateIRModuleForLayout(sink))
    {
        return nullptr;
    }

    return _createEntryPointResult(entryPointIndex, sink);
}

void EndToEndCompileRequest::generateOutput(TargetProgram* targetProgram)
{
    auto program = targetProgram->getProgram();

    // Generate target code any entry points that
    // have been requested for compilation.
    auto entryPointCount = program->getEntryPointCount();
    if (targetProgram->getOptionSet().getBoolOption(CompilerOptionName::GenerateWholeProgram))
    {
        targetProgram->_createWholeProgramResult(getSink(), this);
    }
    else
    {
        for (Index ii = 0; ii < entryPointCount; ++ii)
        {
            targetProgram->_createEntryPointResult(ii, getSink(), this);
        }
    }
}


bool _shouldWriteSourceLocs(Linkage* linkage)
{
    // If debug information or source manager are not avaiable we can't/shouldn't write out locs
    if (linkage->m_optionSet.getEnumOption<DebugInfoLevel>(CompilerOptionName::DebugInformation) ==
            DebugInfoLevel::None ||
        linkage->getSourceManager() == nullptr)
    {
        return false;
    }

    // Otherwise we do want to write out the locs
    return true;
}

SlangResult EndToEndCompileRequest::writeContainerToStream(Stream* stream)
{
    auto linkage = getLinkage();

    // Set up options
    SerialContainerUtil::WriteOptions options;

    // If debug information is enabled, enable writing out source locs
    if (_shouldWriteSourceLocs(linkage))
    {
        options.optionFlags |= SerialOptionFlag::SourceLocation;
        options.sourceManager = linkage->getSourceManager();
    }

    SLANG_RETURN_ON_FAIL(SerialContainerUtil::write(this, options, stream));

    return SLANG_OK;
}

static IBoxValue<SourceMap>* _getObfuscatedSourceMap(TranslationUnitRequest* translationUnit)
{
    if (auto module = translationUnit->getModule())
    {
        if (auto irModule = module->getIRModule())
        {
            return irModule->getObfuscatedSourceMap();
        }
    }
    return nullptr;
}

SlangResult EndToEndCompileRequest::maybeCreateContainer()
{
    m_containerArtifact.setNull();

    List<ComPtr<IArtifact>> artifacts;

    auto linkage = getLinkage();

    auto program = getSpecializedGlobalAndEntryPointsComponentType();

    for (auto targetReq : linkage->targets)
    {
        auto targetProgram = program->getTargetProgram(targetReq);

        if (targetProgram->getOptionSet().getBoolOption(CompilerOptionName::GenerateWholeProgram))
        {
            if (auto artifact = targetProgram->getExistingWholeProgramResult())
            {
                if (!targetProgram->getOptionSet().getBoolOption(
                        CompilerOptionName::EmbedDownstreamIR))
                {
                    artifacts.add(ComPtr<IArtifact>(artifact));
                }
            }
        }
        else
        {
            Index entryPointCount = program->getEntryPointCount();
            for (Index ee = 0; ee < entryPointCount; ++ee)
            {
                if (auto artifact = targetProgram->getExistingEntryPointResult(ee))
                {
                    artifacts.add(ComPtr<IArtifact>(artifact));
                }
            }
        }
    }

    // If IR emitting is enabled, add IR to the artifacts
    if (m_emitIr && (m_containerFormat == ContainerFormat::SlangModule))
    {
        OwnedMemoryStream stream(FileAccess::Write);
        SlangResult res = writeContainerToStream(&stream);
        if (SLANG_FAILED(res))
        {
            getSink()->diagnose(SourceLoc(), Diagnostics::unableToCreateModuleContainer);
            return res;
        }

        // Need to turn into a blob
        List<uint8_t> blobData;
        stream.swapContents(blobData);

        auto containerBlob = ListBlob::moveCreate(blobData);

        auto irArtifact = Artifact::create(ArtifactDesc::make(
            Artifact::Kind::CompileBinary,
            ArtifactPayload::SlangIR,
            ArtifactStyle::Unknown));
        irArtifact->addRepresentationUnknown(containerBlob);

        // Add the IR artifact
        artifacts.add(irArtifact);
    }

    // If there is only one artifact we can use that as the container
    if (artifacts.getCount() == 1)
    {
        m_containerArtifact = artifacts[0];
    }
    else
    {
        m_containerArtifact = ArtifactUtil::createArtifact(
            ArtifactDesc::make(ArtifactKind::Container, ArtifactPayload::CompileResults));

        for (IArtifact* childArtifact : artifacts)
        {
            m_containerArtifact->addChild(childArtifact);
        }
    }

    // Get all of the source obfuscated source maps and add those
    if (m_containerArtifact)
    {
        auto frontEndReq = getFrontEndReq();

        for (auto translationUnit : frontEndReq->translationUnits)
        {
            // Hmmm do I have to therefore add a map for all translation units(!)
            // I guess this is okay in so far as an association can always be looked up by name
            if (auto sourceMap = _getObfuscatedSourceMap(translationUnit))
            {
                auto artifactDesc = ArtifactDesc::make(
                    ArtifactKind::Json,
                    ArtifactPayload::SourceMap,
                    ArtifactStyle::Obfuscated);

                // Create the source map artifact
                auto sourceMapArtifact =
                    Artifact::create(artifactDesc, sourceMap->get().m_file.getUnownedSlice());

                // Add the repesentation
                sourceMapArtifact->addRepresentation(sourceMap);

                // Associate with the container
                m_containerArtifact->addAssociated(sourceMapArtifact);
            }
        }
    }

    return SLANG_OK;
}

CompilerOptionSet& EndToEndCompileRequest::getTargetOptionSet(TargetRequest* req)
{
    return req->getOptionSet();
}

CompilerOptionSet& EndToEndCompileRequest::getTargetOptionSet(Index targetIndex)
{
    return m_linkage->targets[targetIndex]->getOptionSet();
}

SlangResult EndToEndCompileRequest::maybeWriteContainer(const String& fileName)
{
    // If there is no container, or filename, don't write anything
    if (fileName.getLength() == 0 || !m_containerArtifact)
    {
        return SLANG_OK;
    }

    // Filter the containerArtifact into things that can be written
    ComPtr<IArtifact> writeArtifact;
    SLANG_RETURN_ON_FAIL(ArtifactContainerUtil::filter(m_containerArtifact, writeArtifact));

    // Only write if there is something to write
    if (writeArtifact)
    {
        SLANG_RETURN_ON_FAIL(ArtifactContainerUtil::writeContainer(writeArtifact, fileName));
    }

    return SLANG_OK;
}

static void _writeString(Stream& stream, const char* string)
{
    stream.write(string, strlen(string));
}

static void _escapeDependencyString(const char* string, StringBuilder& outBuilder)
{
    // make has unusual escaping rules, but we only care about characters that are acceptable in a
    // path
    for (const char* p = string; *p; ++p)
    {
        char c = *p;
        switch (c)
        {
        case ' ':
        case ':':
        case '#':
        case '[':
        case ']':
        case '\\':
            outBuilder.appendChar('\\');
            break;

        case '$':
            outBuilder.appendChar('$');
            break;
        }

        outBuilder.appendChar(c);
    }
}

// Writes a line to the file stream, formatted like this:
//   <output-file>: <dependency-file> <dependency-file...>
static void _writeDependencyStatement(
    Stream& stream,
    EndToEndCompileRequest* compileRequest,
    const String& outputPath)
{
    if (outputPath.getLength() == 0)
        return;

    StringBuilder builder;
    _escapeDependencyString(outputPath.begin(), builder);
    _writeString(stream, builder.begin());
    _writeString(stream, ": ");

    int dependencyCount = compileRequest->getDependencyFileCount();
    for (int dependencyIndex = 0; dependencyIndex < dependencyCount; ++dependencyIndex)
    {
        builder.clear();
        _escapeDependencyString(compileRequest->getDependencyFilePath(dependencyIndex), builder);
        _writeString(stream, builder.begin());
        _writeString(stream, (dependencyIndex + 1 < dependencyCount) ? " " : "\n");
    }
}

// Writes a file with dependency info, with one line in the output file per compile product.
static SlangResult _writeDependencyFile(EndToEndCompileRequest* compileRequest)
{
    if (compileRequest->m_dependencyOutputPath.getLength() == 0)
        return SLANG_OK;

    FileStream stream;
    SLANG_RETURN_ON_FAIL(stream.init(
        compileRequest->m_dependencyOutputPath,
        FileMode::Create,
        FileAccess::Write,
        FileShare::ReadWrite));

    auto linkage = compileRequest->getLinkage();
    auto program = compileRequest->getSpecializedGlobalAndEntryPointsComponentType();

    // Iterate over all the targets and their outputs
    for (const auto& targetReq : linkage->targets)
    {
        if (compileRequest->getTargetOptionSet(targetReq).getBoolOption(
                CompilerOptionName::GenerateWholeProgram))
        {
            RefPtr<EndToEndCompileRequest::TargetInfo> targetInfo;
            if (compileRequest->m_targetInfos.tryGetValue(targetReq, targetInfo))
            {
                _writeDependencyStatement(
                    stream,
                    compileRequest,
                    targetInfo->wholeTargetOutputPath);
            }
        }
        else
        {
            Index entryPointCount = program->getEntryPointCount();
            for (Index entryPointIndex = 0; entryPointIndex < entryPointCount; ++entryPointIndex)
            {
                RefPtr<EndToEndCompileRequest::TargetInfo> targetInfo;
                if (compileRequest->m_targetInfos.tryGetValue(targetReq, targetInfo))
                {
                    String outputPath;
                    if (targetInfo->entryPointOutputPaths.tryGetValue(entryPointIndex, outputPath))
                    {
                        _writeDependencyStatement(stream, compileRequest, outputPath);
                    }
                }
            }
        }
    }

    // When the output is a binary module, linkage->targets can be empty. So
    // we need to do their dependencies separately.
    if (compileRequest->m_containerFormat == ContainerFormat::SlangModule)
    {
        _writeDependencyStatement(stream, compileRequest, compileRequest->m_containerOutputPath);
    }

    return SLANG_OK;
}


void EndToEndCompileRequest::generateOutput(ComponentType* program)
{
    // When dynamic dispatch is disabled, the program must
    // be fully specialized by now. So we check if we still
    // have unspecialized generic/existential parameters,
    // and report them as an error.
    //
    auto specializationParamCount = program->getSpecializationParamCount();
    if (getOptionSet().getBoolOption(CompilerOptionName::DisableDynamicDispatch) &&
        specializationParamCount != 0)
    {
        auto sink = getSink();

        for (Index ii = 0; ii < specializationParamCount; ++ii)
        {
            auto specializationParam = program->getSpecializationParam(ii);
            if (auto decl = as<Decl>(specializationParam.object))
            {
                sink->diagnose(
                    specializationParam.loc,
                    Diagnostics::specializationParameterOfNameNotSpecialized,
                    decl);
            }
            else if (auto type = as<Type>(specializationParam.object))
            {
                sink->diagnose(
                    specializationParam.loc,
                    Diagnostics::specializationParameterOfNameNotSpecialized,
                    type);
            }
            else
            {
                sink->diagnose(
                    specializationParam.loc,
                    Diagnostics::specializationParameterNotSpecialized);
            }
        }

        return;
    }


    // Go through the code-generation targets that the user
    // has specified, and generate code for each of them.
    //
    auto linkage = getLinkage();
    for (auto targetReq : linkage->targets)
    {
        if (targetReq->getOptionSet().getBoolOption(CompilerOptionName::EmbedDownstreamIR))
            continue;

        auto targetProgram = program->getTargetProgram(targetReq);
        generateOutput(targetProgram);
    }
}

void EndToEndCompileRequest::generateOutput()
{
    SLANG_PROFILE;
    generateOutput(getSpecializedGlobalAndEntryPointsComponentType());

    // If we are in command-line mode, we might be expected to actually
    // write output to one or more files here.

    if (m_isCommandLineCompile && m_containerFormat == ContainerFormat::None)
    {
        auto linkage = getLinkage();
        auto program = getSpecializedGlobalAndEntryPointsComponentType();

        for (auto targetReq : linkage->targets)
        {
            auto targetProgram = program->getTargetProgram(targetReq);

            if (targetProgram->getOptionSet().getBoolOption(
                    CompilerOptionName::GenerateWholeProgram))
            {
                if (const auto artifact = targetProgram->getExistingWholeProgramResult())
                {
                    const auto path = _getWholeProgramPath(targetReq);

                    _maybeWriteArtifact(path, artifact);
                }
            }
            else
            {
                Index entryPointCount = program->getEntryPointCount();
                for (Index ee = 0; ee < entryPointCount; ++ee)
                {
                    if (const auto artifact = targetProgram->getExistingEntryPointResult(ee))
                    {
                        const auto path = _getEntryPointPath(targetReq, ee);

                        _maybeWriteArtifact(path, artifact);
                    }
                }
            }
        }
    }

    // Maybe create the container
    maybeCreateContainer();

    // If it's a command line compile we may need to write the container to a file
    if (m_isCommandLineCompile)
    {
        // TODO(JS):
        // We could write the container into a source embedded format potentially

        maybeWriteContainer(m_containerOutputPath);

        _writeDependencyFile(this);
    }
}

// Debug logic for dumping intermediate outputs


void CodeGenContext::_dumpIntermediateMaybeWithAssembly(IArtifact* artifact)
{
    _dumpIntermediate(artifact);

    ComPtr<IArtifact> assembly;
    ArtifactOutputUtil::maybeDisassemble(getSession(), artifact, nullptr, assembly);

    if (assembly)
    {
        _dumpIntermediate(assembly);
    }
}

void CodeGenContext::_dumpIntermediate(IArtifact* artifact)
{
    ComPtr<ISlangBlob> blob;
    if (SLANG_FAILED(artifact->loadBlob(ArtifactKeep::No, blob.writeRef())))
    {
        return;
    }
    _dumpIntermediate(artifact->getDesc(), blob->getBufferPointer(), blob->getBufferSize());
}

void CodeGenContext::_dumpIntermediate(const ArtifactDesc& desc, void const* data, size_t size)
{
    // Try to generate a unique ID for the file to dump,
    // even in cases where there might be multiple threads
    // doing compilation.
    //
    // This is primarily a debugging aid, so we don't
    // really need/want to do anything too elaborate

    static std::atomic<uint32_t> counter(0);

    const uint32_t id = ++counter;

    // Just use the counter for the 'base name'
    StringBuilder basename;

    // Add the prefix
    basename << getIntermediateDumpPrefix();

    // Add the id
    basename << int(id);

    // Work out the filename based on the desc and the basename
    StringBuilder filename;
    ArtifactDescUtil::calcNameForDesc(desc, basename.getUnownedSlice(), filename);

    // If didn't produce a filename, use basename with .unknown extension
    if (filename.getLength() == 0)
    {
        filename = basename;
        filename << ".unknown";
    }

    // Write to a file
    ArtifactOutputUtil::writeToFile(desc, data, size, filename);
}

void CodeGenContext::maybeDumpIntermediate(IArtifact* artifact)
{
    if (!shouldDumpIntermediates())
        return;


    _dumpIntermediateMaybeWithAssembly(artifact);
}

IRDumpOptions CodeGenContext::getIRDumpOptions()
{
    if (auto endToEndReq = isEndToEndCompile())
    {
        return endToEndReq->getFrontEndReq()->m_irDumpOptions;
    }
    return IRDumpOptions();
}

bool CodeGenContext::shouldValidateIR()
{
    return getTargetProgram()->getOptionSet().getBoolOption(CompilerOptionName::ValidateIr);
}

bool CodeGenContext::shouldSkipSPIRVValidation()
{
    return getTargetProgram()->getOptionSet().getBoolOption(
        CompilerOptionName::SkipSPIRVValidation);
}

bool CodeGenContext::shouldDumpIR()
{
    return getTargetProgram()->getOptionSet().getBoolOption(CompilerOptionName::DumpIr);
}

bool CodeGenContext::shouldSkipDownstreamLinking()
{
    return getTargetProgram()->getOptionSet().getBoolOption(
        CompilerOptionName::SkipDownstreamLinking);
}

bool CodeGenContext::shouldReportCheckpointIntermediates()
{
    return getTargetProgram()->getOptionSet().getBoolOption(
        CompilerOptionName::ReportCheckpointIntermediates);
}

bool CodeGenContext::shouldDumpIntermediates()
{
    return getTargetProgram()->getOptionSet().getBoolOption(CompilerOptionName::DumpIntermediates);
}

bool CodeGenContext::shouldTrackLiveness()
{
    return getTargetProgram()->getOptionSet().getBoolOption(CompilerOptionName::TrackLiveness);
}

String CodeGenContext::getIntermediateDumpPrefix()
{
    return getTargetProgram()->getOptionSet().getStringOption(
        CompilerOptionName::DumpIntermediatePrefix);
}

bool CodeGenContext::getUseUnknownImageFormatAsDefault()
{
    return getTargetProgram()->getOptionSet().getBoolOption(
        CompilerOptionName::DefaultImageFormatUnknown);
}

bool CodeGenContext::isSpecializationDisabled()
{
    return getTargetProgram()->getOptionSet().getBoolOption(
        CompilerOptionName::DisableSpecialization);
}

SLANG_NO_THROW SlangResult SLANG_MCALL Module::serialize(ISlangBlob** outSerializedBlob)
{
    SerialContainerUtil::WriteOptions writeOptions;
    writeOptions.sourceManager = getLinkage()->getSourceManager();
    OwnedMemoryStream memoryStream(FileAccess::Write);
    SLANG_RETURN_ON_FAIL(SerialContainerUtil::write(this, writeOptions, &memoryStream));
    *outSerializedBlob = RawBlob::create(
                             memoryStream.getContents().getBuffer(),
                             (size_t)memoryStream.getContents().getCount())
                             .detach();
    return SLANG_OK;
}

SLANG_NO_THROW SlangResult SLANG_MCALL Module::writeToFile(char const* fileName)
{
    SerialContainerUtil::WriteOptions writeOptions;
    writeOptions.sourceManager = getLinkage()->getSourceManager();
    FileStream fileStream;
    SLANG_RETURN_ON_FAIL(fileStream.init(fileName, FileMode::Create));
    return SerialContainerUtil::write(this, writeOptions, &fileStream);
}

SLANG_NO_THROW const char* SLANG_MCALL Module::getName()
{
    if (m_name)
        return m_name->text.getBuffer();
    return nullptr;
}

SLANG_NO_THROW const char* SLANG_MCALL Module::getFilePath()
{
    if (m_pathInfo.hasFoundPath())
        return m_pathInfo.foundPath.getBuffer();
    return nullptr;
}

SLANG_NO_THROW const char* SLANG_MCALL Module::getUniqueIdentity()
{
    if (m_pathInfo.hasUniqueIdentity())
        return m_pathInfo.getMostUniqueIdentity().getBuffer();
    return nullptr;
}

SLANG_NO_THROW SlangInt32 SLANG_MCALL Module::getDependencyFileCount()
{
    return (SlangInt32)getFileDependencies().getCount();
}

SLANG_NO_THROW char const* SLANG_MCALL Module::getDependencyFilePath(SlangInt32 index)
{
    SourceFile* sourceFile = getFileDependencies()[index];
    return sourceFile->getPathInfo().hasFoundPath()
               ? sourceFile->getPathInfo().getMostUniqueIdentity().getBuffer()
               : nullptr;
}

void validateEntryPoint(EntryPoint* entryPoint, DiagnosticSink* sink);

void Module::_discoverEntryPoints(DiagnosticSink* sink, const List<RefPtr<TargetRequest>>& targets)
{
    if (m_entryPoints.getCount() > 0)
        return;
    _discoverEntryPointsImpl(m_moduleDecl, sink, targets);
}
void Module::_discoverEntryPointsImpl(
    ContainerDecl* containerDecl,
    DiagnosticSink* sink,
    const List<RefPtr<TargetRequest>>& targets)
{
    for (auto globalDecl : containerDecl->members)
    {
        auto maybeFuncDecl = globalDecl;
        if (auto genericDecl = as<GenericDecl>(maybeFuncDecl))
        {
            maybeFuncDecl = genericDecl->inner;
        }

        if (as<NamespaceDeclBase>(globalDecl) || as<FileDecl>(globalDecl) ||
            as<StructDecl>(globalDecl))
        {
            _discoverEntryPointsImpl(as<ContainerDecl>(globalDecl), sink, targets);
            continue;
        }

        auto funcDecl = as<FuncDecl>(maybeFuncDecl);
        if (!funcDecl)
            continue;

        Profile profile;
        bool resolvedStageOfProfileWithEntryPoint = resolveStageOfProfileWithEntryPoint(
            profile,
            getLinkage()->m_optionSet,
            targets,
            funcDecl,
            sink);
        if (!resolvedStageOfProfileWithEntryPoint)
        {
            // If there isn't a [shader] attribute, look for a [numthreads] attribute
            // since that implicitly means a compute shader. We'll not do this when compiling for
            // CUDA/Torch since [numthreads] attributes are utilized differently for those targets.
            //

            bool allTargetsCUDARelated = true;
            for (auto target : targets)
            {
                if (!isCUDATarget(target) &&
                    target->getTarget() != CodeGenTarget::PyTorchCppBinding)
                {
                    allTargetsCUDARelated = false;
                    break;
                }
            }

            if (allTargetsCUDARelated && targets.getCount() > 0)
                continue;

            bool canDetermineStage = false;
            for (auto modifier : funcDecl->modifiers)
            {
                if (as<NumThreadsAttribute>(modifier))
                {
                    if (funcDecl->findModifier<OutputTopologyAttribute>())
                        profile.setStage(Stage::Mesh);
                    else
                        profile.setStage(Stage::Compute);
                    canDetermineStage = true;
                    break;
                }
                else if (as<PatchConstantFuncAttribute>(modifier))
                {
                    profile.setStage(Stage::Hull);
                    canDetermineStage = true;
                    break;
                }
            }
            if (!canDetermineStage)
                continue;
        }

        RefPtr<EntryPoint> entryPoint =
            EntryPoint::create(getLinkage(), makeDeclRef(funcDecl), profile);

        validateEntryPoint(entryPoint, sink);

        // Note: in the case that the user didn't explicitly
        // specify entry points and we are instead compiling
        // a shader "library," then we do not want to automatically
        // combine the entry points into groups in the generated
        // `Program`, since that would be slightly too magical.
        //
        // Instead, each entry point will end up in a singleton
        // group, so that its entry-point parameters lay out
        // independent of the others.
        //
        _addEntryPoint(entryPoint);
    }
}
} // namespace Slang
