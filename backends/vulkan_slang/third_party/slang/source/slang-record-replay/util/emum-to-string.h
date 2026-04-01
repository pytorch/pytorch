#include "slang.h"

namespace SlangRecord
{
static Slang::String SlangCompileTargetToString(const SlangCompileTarget target)
{
#define CASE(x) \
    case x:     \
        return #x

    switch (target)
    {
        CASE(SLANG_TARGET_UNKNOWN);
        CASE(SLANG_GLSL);
        CASE(SLANG_GLSL_VULKAN_DEPRECATED);
        CASE(SLANG_GLSL_VULKAN_ONE_DESC_DEPRECATED);
        CASE(SLANG_HLSL);
        CASE(SLANG_SPIRV);
        CASE(SLANG_SPIRV_ASM);
        CASE(SLANG_DXBC);
        CASE(SLANG_DXIL);
        CASE(SLANG_DXIL_ASM);
        CASE(SLANG_C_SOURCE);
        CASE(SLANG_CPP_SOURCE);
        CASE(SLANG_HOST_EXECUTABLE);
        CASE(SLANG_SHADER_SHARED_LIBRARY);
        CASE(SLANG_SHADER_HOST_CALLABLE);
        CASE(SLANG_CUDA_SOURCE);
        CASE(SLANG_PTX);
        CASE(SLANG_CUDA_OBJECT_CODE);
        CASE(SLANG_OBJECT_CODE);
        CASE(SLANG_HOST_CPP_SOURCE);
        CASE(SLANG_HOST_HOST_CALLABLE);
        CASE(SLANG_CPP_PYTORCH_BINDING);
        CASE(SLANG_METAL);
        CASE(SLANG_METAL_LIB);
        CASE(SLANG_METAL_LIB_ASM);
        CASE(SLANG_HOST_SHARED_LIBRARY);
        CASE(SLANG_WGSL);
        CASE(SLANG_TARGET_COUNT_OF);
    default:
        Slang::StringBuilder str;
        str << "Unknown SlangCompileTarget: " << static_cast<uint32_t>(target);
        return str.toString();
    }
#undef CASE
}

static Slang::String SlangProfileIDToString(const SlangProfileID profile)
{
    switch (profile)
    {
    case SLANG_PROFILE_UNKNOWN:
        return "SLANG_PROFILE_UNKNOWN";
    default:
        Slang::StringBuilder str;
        str << "Unknown SlangProfileID: " << static_cast<uint32_t>(profile);
        return str.toString();
    }
}

static Slang::String SlangTargetFlagsToString(const SlangTargetFlags flags)
{
    switch (flags)
    {
    case SLANG_TARGET_FLAG_PARAMETER_BLOCKS_USE_REGISTER_SPACES:
        return "SLANG_TARGET_FLAG_PARAMETER_BLOCKS_USE_REGISTER_SPACES";
    case SLANG_TARGET_FLAG_GENERATE_WHOLE_PROGRAM:
        return "SLANG_TARGET_FLAG_GENERATE_WHOLE_PROGRAM";
    case SLANG_TARGET_FLAG_DUMP_IR:
        return "SLANG_TARGET_FLAG_DUMP_IR";
    case SLANG_TARGET_FLAG_GENERATE_SPIRV_DIRECTLY:
        return "SLANG_TARGET_FLAG_GENERATE_SPIRV_DIRECTLY";
    default:
        Slang::StringBuilder str;
        str << "Unknown SlangTargetFlags: " << static_cast<uint32_t>(flags);
        return str.toString();
    }
}

static Slang::String SlangFloatingPointModeToString(const SlangFloatingPointMode mode)
{
    switch (mode)
    {
    case SLANG_FLOATING_POINT_MODE_DEFAULT:
        return "SLANG_FLOATING_POINT_MODE_DEFAULT";
    case SLANG_FLOATING_POINT_MODE_FAST:
        return "SLANG_FLOATING_POINT_MODE_FAST";
    case SLANG_FLOATING_POINT_MODE_PRECISE:
        return "SLANG_FLOATING_POINT_MODE_PRECISE";
    default:
        Slang::StringBuilder str;
        str << "Unknown SlangFloatingPointMode: " << static_cast<uint32_t>(mode);
        return str.toString();
    }
}

static Slang::String SlangLineDirectiveModeToString(const SlangLineDirectiveMode mode)
{
    switch (mode)
    {
    case SLANG_LINE_DIRECTIVE_MODE_DEFAULT:
        return "SLANG_LINE_DIRECTIVE_MODE_DEFAULT";
    case SLANG_LINE_DIRECTIVE_MODE_NONE:
        return "SLANG_LINE_DIRECTIVE_MODE_NONE";
    case SLANG_LINE_DIRECTIVE_MODE_STANDARD:
        return "SLANG_LINE_DIRECTIVE_MODE_STANDARD";
    case SLANG_LINE_DIRECTIVE_MODE_GLSL:
        return "SLANG_LINE_DIRECTIVE_MODE_GLSL";
    case SLANG_LINE_DIRECTIVE_MODE_SOURCE_MAP:
        return "SLANG_LINE_DIRECTIVE_MODE_SOURCE_MAP";
    default:
        Slang::StringBuilder str;
        str << "Unknown SlangLineDirectiveMode: " << static_cast<uint32_t>(mode);
        return str.toString();
    }
}

static Slang::String CompilerOptionNameToString(const slang::CompilerOptionName name)
{
#define CASE(x)                 \
    case CompilerOptionName::x: \
        return #x

    using namespace slang;
    switch (name)
    {
        CASE(MacroDefine);
        CASE(DepFile);
        CASE(EntryPointName);
        CASE(Specialize);
        CASE(Help);
        CASE(HelpStyle);
        CASE(Include);
        CASE(Language);
        CASE(MatrixLayoutColumn);
        CASE(MatrixLayoutRow);
        CASE(ZeroInitialize);
        CASE(IgnoreCapabilities);
        CASE(RestrictiveCapabilityCheck);
        CASE(ModuleName);
        CASE(Output);
        CASE(Profile);
        CASE(Stage);
        CASE(Target);
        CASE(Version);
        CASE(WarningsAsErrors);
        CASE(DisableWarnings);
        CASE(EnableWarning);
        CASE(DisableWarning);
        CASE(DumpWarningDiagnostics);
        CASE(InputFilesRemain);
        CASE(EmitIr);
        CASE(ReportDownstreamTime);
        CASE(ReportPerfBenchmark);
        CASE(ReportCheckpointIntermediates);
        CASE(SkipSPIRVValidation);
        CASE(SourceEmbedStyle);
        CASE(SourceEmbedName);
        CASE(SourceEmbedLanguage);
        CASE(DisableShortCircuit);
        CASE(MinimumSlangOptimization);
        CASE(DisableNonEssentialValidations);
        CASE(DisableSourceMap);
        CASE(UnscopedEnum);
        CASE(PreserveParameters);
        CASE(Capability);
        CASE(DefaultImageFormatUnknown);
        CASE(DisableDynamicDispatch);
        CASE(DisableSpecialization);
        CASE(FloatingPointMode);
        CASE(DebugInformation);
        CASE(LineDirectiveMode);
        CASE(Optimization);
        CASE(Obfuscate);
        CASE(VulkanBindShift);
        CASE(VulkanBindGlobals);
        CASE(VulkanInvertY);
        CASE(VulkanUseDxPositionW);
        CASE(VulkanUseEntryPointName);
        CASE(VulkanUseGLLayout);
        CASE(VulkanEmitReflection);
        CASE(GLSLForceScalarLayout);
        CASE(ForceDXLayout);
        CASE(EnableEffectAnnotations);
        CASE(EmitSpirvViaGLSL);
        CASE(EmitSpirvDirectly);
        CASE(SPIRVCoreGrammarJSON);
        CASE(IncompleteLibrary);
        CASE(CompilerPath);
        CASE(DefaultDownstreamCompiler);
        CASE(DownstreamArgs);
        CASE(PassThrough);
        CASE(DumpRepro);
        CASE(DumpReproOnError);
        CASE(ExtractRepro);
        CASE(LoadRepro);
        CASE(LoadReproDirectory);
        CASE(ReproFallbackDirectory);
        CASE(DumpAst);
        CASE(DumpIntermediatePrefix);
        CASE(DumpIntermediates);
        CASE(DumpIr);
        CASE(DumpIrIds);
        CASE(PreprocessorOutput);
        CASE(OutputIncludes);
        CASE(ReproFileSystem);
        CASE(SerialIr);
        CASE(SkipCodeGen);
        CASE(ValidateIr);
        CASE(VerbosePaths);
        CASE(VerifyDebugSerialIr);
        CASE(NoCodeGen);
        CASE(FileSystem);
        CASE(Heterogeneous);
        CASE(NoMangle);
        CASE(NoHLSLBinding);
        CASE(NoHLSLPackConstantBufferElements);
        CASE(ValidateUniformity);
        CASE(AllowGLSL);
        CASE(ArchiveType);
        CASE(CompileCoreModule);
        CASE(Doc);
        CASE(IrCompression);
        CASE(LoadCoreModule);
        CASE(ReferenceModule);
        CASE(SaveCoreModule);
        CASE(SaveCoreModuleBinSource);
        CASE(SaveGLSLModuleBinSource);
        CASE(TrackLiveness);
        CASE(LoopInversion);
        CASE(CountOfParsableOptions);
        CASE(DebugInformationFormat);
        CASE(VulkanBindShiftAll);
        CASE(GenerateWholeProgram);
        CASE(UseUpToDateBinaryModule);
        CASE(CountOf);
    default:
        Slang::StringBuilder str;
        str << "Unknown CompilerOptionName: " << static_cast<uint32_t>(name);
        return str.toString();
    }
#undef CASE
}

static Slang::String CompilerOptionValueKindToString(const slang::CompilerOptionValueKind kind)
{
    using namespace slang;
    switch (kind)
    {
    case CompilerOptionValueKind::Int:
        return "Int";
    case CompilerOptionValueKind::String:
        return "String";
    default:
        Slang::StringBuilder str;
        str << "Unknown CompilerOptionValueKind: " << static_cast<uint32_t>(kind);
        return str.toString();
    }
}

static Slang::String SessionFlagsToString(const slang::SessionFlags flags)
{
    using namespace slang;
    switch (flags)
    {
    case kSessionFlags_None:
        return "kSessionFlags_None";
    default:
        Slang::StringBuilder str;
        str << "Unknown SessionFlags: " << static_cast<uint32_t>(flags);
        return str.toString();
    }
}

static Slang::String SlangMatrixLayoutModeToString(const SlangMatrixLayoutMode mode)
{
    switch (mode)
    {
    case SLANG_MATRIX_LAYOUT_MODE_UNKNOWN:
        return "SLANG_MATRIX_LAYOUT_MODE_UNKNOWN";
    case SLANG_MATRIX_LAYOUT_ROW_MAJOR:
        return "SLANG_MATRIX_LAYOUT_ROW_MAJOR";
    case SLANG_MATRIX_LAYOUT_COLUMN_MAJOR:
        return "SLANG_MATRIX_LAYOUT_COLUMN_MAJOR";
    default:
        Slang::StringBuilder str;
        str << "Unknown SlangMatrixLayoutMode: " << static_cast<uint32_t>(mode);
        return str.toString();
    }
}

static Slang::String SlangPassThroughToString(const SlangPassThrough passThrough)
{
#define CASE(x) \
    case x:     \
        return #x

    switch (passThrough)
    {
        CASE(SLANG_PASS_THROUGH_NONE);
        CASE(SLANG_PASS_THROUGH_FXC);
        CASE(SLANG_PASS_THROUGH_DXC);
        CASE(SLANG_PASS_THROUGH_GLSLANG);
        CASE(SLANG_PASS_THROUGH_SPIRV_DIS);
        CASE(SLANG_PASS_THROUGH_CLANG);
        CASE(SLANG_PASS_THROUGH_VISUAL_STUDIO);
        CASE(SLANG_PASS_THROUGH_GCC);
        CASE(SLANG_PASS_THROUGH_GENERIC_C_CPP);
        CASE(SLANG_PASS_THROUGH_NVRTC);
        CASE(SLANG_PASS_THROUGH_LLVM);
        CASE(SLANG_PASS_THROUGH_SPIRV_OPT);
        CASE(SLANG_PASS_THROUGH_METAL);
        CASE(SLANG_PASS_THROUGH_COUNT_OF);
    default:
        Slang::StringBuilder str;
        str << "Unknown SlangPassThrough: " << static_cast<uint32_t>(passThrough);
        return str.toString();
    }
#undef CASE
}

static Slang::String SlangSourceLanguageToString(const SlangSourceLanguage language)
{
#define CASE(x) \
    case x:     \
        return #x

    switch (language)
    {
        CASE(SLANG_SOURCE_LANGUAGE_UNKNOWN);
        CASE(SLANG_SOURCE_LANGUAGE_SLANG);
        CASE(SLANG_SOURCE_LANGUAGE_HLSL);
        CASE(SLANG_SOURCE_LANGUAGE_GLSL);
        CASE(SLANG_SOURCE_LANGUAGE_C);
        CASE(SLANG_SOURCE_LANGUAGE_CPP);
        CASE(SLANG_SOURCE_LANGUAGE_CUDA);
        CASE(SLANG_SOURCE_LANGUAGE_SPIRV);
        CASE(SLANG_SOURCE_LANGUAGE_METAL);
        CASE(SLANG_SOURCE_LANGUAGE_COUNT_OF);
    default:
        Slang::StringBuilder str;
        str << "Unknown SlangSourceLanguage: " << static_cast<uint32_t>(language);
        return str.toString();
    }
}

static Slang::String CompileCoreModuleFlagsToString(const slang::CompileCoreModuleFlags flags)
{
    using namespace slang;
    switch (flags)
    {
    case CompileCoreModuleFlag::WriteDocumentation:
        return "WriteDocumentation";
    default:
        Slang::StringBuilder str;
        str << "Unknown CompileCoreModuleFlags: " << static_cast<uint32_t>(flags);
        return str.toString();
    }
}

static Slang::String SlangArchiveTypeToString(const SlangArchiveType type)
{
#define CASE(x) \
    case x:     \
        return #x
    switch (type)
    {
        CASE(SLANG_ARCHIVE_TYPE_UNDEFINED);
        CASE(SLANG_ARCHIVE_TYPE_ZIP);
        CASE(SLANG_ARCHIVE_TYPE_RIFF);
        CASE(SLANG_ARCHIVE_TYPE_RIFF_DEFLATE);
        CASE(SLANG_ARCHIVE_TYPE_RIFF_LZ4);
        CASE(SLANG_ARCHIVE_TYPE_COUNT_OF);
    default:
        Slang::StringBuilder str;
        str << "Unknown SlangArchiveType: " << static_cast<uint32_t>(type);
        return str.toString();
    }
}

static Slang::String SpecializationArgKindToString(const slang::SpecializationArg::Kind kind)
{
    using namespace slang;
    switch (kind)
    {
    case SpecializationArg::Kind::Unknown:
        return "Unknown";
    case SpecializationArg::Kind::Type:
        return "Type";
    default:
        Slang::StringBuilder str;
        str << "Unknown SpecializationArg::Kind: " << static_cast<uint32_t>(kind);
        return str.toString();
    }
}

static Slang::String LayoutRulesToString(const slang::LayoutRules rules)
{
    using namespace slang;
    switch (rules)
    {
    case LayoutRules::Default:
        return "Default";
    case LayoutRules::MetalArgumentBufferTier2:
        return "MetalArgumentBufferTier2";
    default:
        Slang::StringBuilder str;
        str << "Unknown LayoutRules: " << static_cast<uint32_t>(rules);
        return str.toString();
    }
}

static Slang::String SlangStageToString(const SlangStage stage)
{
#define CASE(x) \
    case x:     \
        return #x

    switch (stage)
    {
        CASE(SLANG_STAGE_NONE);
        CASE(SLANG_STAGE_VERTEX);
        CASE(SLANG_STAGE_HULL);
        CASE(SLANG_STAGE_DOMAIN);
        CASE(SLANG_STAGE_GEOMETRY);
        CASE(SLANG_STAGE_FRAGMENT);
        CASE(SLANG_STAGE_COMPUTE);
        CASE(SLANG_STAGE_RAY_GENERATION);
        CASE(SLANG_STAGE_INTERSECTION);
        CASE(SLANG_STAGE_ANY_HIT);
        CASE(SLANG_STAGE_CLOSEST_HIT);
        CASE(SLANG_STAGE_MISS);
        CASE(SLANG_STAGE_CALLABLE);
        CASE(SLANG_STAGE_MESH);
        CASE(SLANG_STAGE_AMPLIFICATION);
    default:
        Slang::StringBuilder str;
        str << "Unknown SlangStage: " << static_cast<uint32_t>(stage);
        return str.toString();
    }
#undef CASE
}

static Slang::String ContainerTypeToString(const slang::ContainerType type)
{
    using namespace slang;
    switch (type)
    {
    case ContainerType::None:
        return "None";
    case ContainerType::UnsizedArray:
        return "UnsizedArray";
    case ContainerType::StructuredBuffer:
        return "StructuredBuffer";
    case ContainerType::ConstantBuffer:
        return "ConstantBuffer";
    case ContainerType::ParameterBlock:
        return "ParameterBlock";
    default:
        Slang::StringBuilder str;
        str << "Unknown ContainerType: " << static_cast<uint32_t>(type);
        return str.toString();
    }
}
} // namespace SlangRecord
