// slang-options.cpp

// Implementation of options parsing for `slangc` command line,
// and also for API interface that takes command-line argument strings.

#include "slang-options.h"

#include "../compiler-core/slang-artifact-desc-util.h"
#include "../compiler-core/slang-artifact-impl.h"
#include "../compiler-core/slang-artifact-representation-impl.h"
#include "../compiler-core/slang-command-line-args.h"
#include "../compiler-core/slang-core-diagnostics.h"
#include "../compiler-core/slang-name-convention-util.h"
#include "../compiler-core/slang-source-embed-util.h"
#include "../core/slang-castable.h"
#include "../core/slang-char-util.h"
#include "../core/slang-command-options-writer.h"
#include "../core/slang-file-system.h"
#include "../core/slang-hex-dump-util.h"
#include "../core/slang-name-value.h"
#include "../core/slang-string-slice-pool.h"
#include "../core/slang-type-text-util.h"
#include "slang-compiler-options.h"
#include "slang-compiler.h"
#include "slang-hlsl-to-vulkan-layout-options.h"
#include "slang-profile.h"
#include "slang-repro.h"
#include "slang-serialize-ir.h"
#include "slang.h"

#include <assert.h>

namespace Slang
{

namespace
{ // anonymous

// All of the options are given an unique enum
typedef CompilerOptionName OptionKind;

struct Option
{
    OptionKind optionKind;
    const char* name;
    const char* usage = nullptr;
    const char* description = nullptr;
};

enum class ValueCategory
{
    Compiler,
    Target,
    Language,
    FloatingPointMode,
    ArchiveType,
    Stage,
    LineDirectiveMode,
    DebugInfoFormat,
    HelpStyle,
    OptimizationLevel,
    DebugLevel,
    FileSystemType,
    VulkanShift,
    SourceEmbedStyle,

    CountOf,
};

template<typename T>
struct GetValueCategory;

#define SLANG_GET_VALUE_CATEGORY(cat, type)   \
    template<>                                \
    struct GetValueCategory<type>             \
    {                                         \
        enum                                  \
        {                                     \
            Value = Index(ValueCategory::cat) \
        };                                    \
    };

SLANG_GET_VALUE_CATEGORY(Compiler, SlangPassThrough)
SLANG_GET_VALUE_CATEGORY(ArchiveType, SlangArchiveType)
SLANG_GET_VALUE_CATEGORY(LineDirectiveMode, SlangLineDirectiveMode)
SLANG_GET_VALUE_CATEGORY(FloatingPointMode, FloatingPointMode)
SLANG_GET_VALUE_CATEGORY(FileSystemType, TypeTextUtil::FileSystemType)
SLANG_GET_VALUE_CATEGORY(HelpStyle, CommandOptionsWriter::Style)
SLANG_GET_VALUE_CATEGORY(OptimizationLevel, SlangOptimizationLevel)
SLANG_GET_VALUE_CATEGORY(VulkanShift, HLSLToVulkanLayoutOptions::Kind)
SLANG_GET_VALUE_CATEGORY(SourceEmbedStyle, SourceEmbedUtil::Style)
SLANG_GET_VALUE_CATEGORY(Language, SourceLanguage)

} // namespace

static void _addOptions(const ConstArrayView<Option>& options, CommandOptions& cmdOptions)
{
    for (auto& opt : options)
    {
        cmdOptions
            .add(opt.name, opt.usage, opt.description, CommandOptions::UserValue(opt.optionKind));
    }
}

void initCommandOptions(CommandOptions& options)
{
    typedef CommandOptions::Flag::Enum Flag;
    typedef CommandOptions::CategoryKind CategoryKind;
    typedef CommandOptions::UserValue UserValue;

    // Add all the option categories

    options.addCategory(CategoryKind::Option, "General", "General options");
    options.addCategory(CategoryKind::Option, "Target", "Target code generation options");
    options.addCategory(CategoryKind::Option, "Downstream", "Downstream compiler options");
    options.addCategory(
        CategoryKind::Option,
        "Debugging",
        "Compiler debugging/instrumentation options");
    options.addCategory(CategoryKind::Option, "Repro", "Slang repro system related");
    options.addCategory(
        CategoryKind::Option,
        "Experimental",
        "Experimental options (use at your own risk)");
    options.addCategory(
        CategoryKind::Option,
        "Internal",
        "Internal-use options (use at your own risk)");
    options.addCategory(
        CategoryKind::Option,
        "Deprecated",
        "Deprecated options (allowed but ignored; may be removed in future)");

    // Do the easy ones
    {
        options.addCategory(
            CategoryKind::Value,
            "compiler",
            "Downstream Compilers (aka Pass through)",
            UserValue(ValueCategory::Compiler));
        options.addValues(TypeTextUtil::getCompilerInfos());

        options.addCategory(
            CategoryKind::Value,
            "language",
            "Language",
            UserValue(ValueCategory::Language));
        options.addValues(TypeTextUtil::getLanguageInfos());

        options.addCategory(
            CategoryKind::Value,
            "archive-type",
            "Archive Type",
            UserValue(ValueCategory::ArchiveType));
        options.addValues(TypeTextUtil::getArchiveTypeInfos());

        options.addCategory(
            CategoryKind::Value,
            "line-directive-mode",
            "Line Directive Mode",
            UserValue(ValueCategory::LineDirectiveMode));
        options.addValues(TypeTextUtil::getLineDirectiveInfos());

        options.addCategory(
            CategoryKind::Value,
            "debug-info-format",
            "Debug Info Format",
            UserValue(ValueCategory::DebugInfoFormat));
        options.addValues(TypeTextUtil::getDebugInfoFormatInfos());

        options.addCategory(
            CategoryKind::Value,
            "fp-mode",
            "Floating Point Mode",
            UserValue(ValueCategory::FloatingPointMode));
        options.addValues(TypeTextUtil::getFloatingPointModeInfos());

        options.addCategory(
            CategoryKind::Value,
            "help-style",
            "Help Style",
            UserValue(ValueCategory::HelpStyle));
        options.addValues(CommandOptionsWriter::getStyleInfos());

        options.addCategory(
            CategoryKind::Value,
            "optimization-level",
            "Optimization Level",
            UserValue(ValueCategory::OptimizationLevel));
        options.addValues(TypeTextUtil::getOptimizationLevelInfos());

        options.addCategory(
            CategoryKind::Value,
            "debug-level",
            "Debug Level",
            UserValue(ValueCategory::DebugLevel));
        options.addValues(TypeTextUtil::getDebugLevelInfos());

        options.addCategory(
            CategoryKind::Value,
            "file-system-type",
            "File System Type",
            UserValue(ValueCategory::FileSystemType));
        options.addValues(TypeTextUtil::getFileSystemTypeInfos());

        options.addCategory(
            CategoryKind::Value,
            "source-embed-style",
            "Source Embed Style",
            UserValue(ValueCategory::SourceEmbedStyle));
        options.addValues(SourceEmbedUtil::getStyleInfos());
    }

    /* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! target !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

    {
        options
            .addCategory(CategoryKind::Value, "target", "Target", UserValue(ValueCategory::Target));
        for (auto opt : TypeTextUtil::getCompileTargetInfos())
        {
            options.addValue(opt.names, opt.description, UserValue(opt.target));
        }
    }

    /* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! stage !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

    {
        options.addCategory(CategoryKind::Value, "stage", "Stage", UserValue(ValueCategory::Stage));
        List<NameValue> opts;
        for (auto& info : getStageInfos())
        {
            opts.add({ValueInt(info.stage), info.name});
        }
        options.addValuesWithAliases(opts.getArrayView());
    }

    /* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! vulkan-shift !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

    {
        options.addCategory(
            CategoryKind::Value,
            "vulkan-shift",
            "Vulkan Shift",
            UserValue(ValueCategory::VulkanShift));
        options.addValues(HLSLToVulkanLayoutOptions::getKindInfos());
    }

    /* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! capabilities !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

    {
        options.addCategory(
            CategoryKind::Value,
            "capability",
            "A capability describes an optional feature that a target may or "
            "may not support. When a -capability is specified, the compiler "
            "may assume that the target supports that capability, and generate "
            "code accordingly.");

        List<UnownedStringSlice> names;
        getCapabilityNames(names);

        // We'll just add to keep the list more simple...
        options.addValue("spirv_1_{ 0,1,2,3,4,5 }", "minimum supported SPIR - V version");

        for (auto name : names)
        {
            if (name.startsWith("__") || name.startsWith("spirv_1_") || name.startsWith("_"))
            {
                continue;
            }
            else if (name.startsWith("GL_") || name.startsWith("SPV_") || name.startsWith("GLSL_"))
            {
                // We'll assume it is an extension..
                StringBuilder buf;
                buf << "enables the " << name << " extension";
                options.addValue(name, buf.getUnownedSlice());
            }
            else
            {
                options.addValue(name);
            }
        }
    }

    /* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! extension !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

    {
        options.addCategory(
            CategoryKind::Value,
            "file-extension",
            "A <language>, <format>, and/or <stage> may be inferred from the "
            "extension of an input or -o path");

        // TODO(JS): It's concevable that these are enumerated via some other system
        // rather than just being listed here

        const CommandOptions::ValuePair pairs[] = {
            {"hlsl,fx", "hlsl"},
            {"dxbc", nullptr},
            {"dxbc-asm", "dxbc-assembly"},
            {"dxil", nullptr},
            {"dxil-asm", "dxil-assembly"},
            {"glsl", nullptr},
            {"vert", "glsl (vertex)"},
            {"frag", "glsl (fragment)"},
            {"geom", "glsl (geoemtry)"},
            {"tesc", "glsl (hull)"},
            {"tese", "glsl (domain)"},
            {"comp", "glsl (compute)"},
            {"slang", nullptr},
            {"spv", "SPIR-V"},
            {"spv-asm", "SPIR-V assembly"},
            {"c", nullptr},
            {"cpp,c++,cxx", "C++"},
            {"exe", "executable"},
            {"dll,so", "sharedlibrary/dll"},
            {"cu", "CUDA"},
            {"ptx", "PTX"},
            {"obj,o", "object-code"},
            {"zip", "container"},
            {"slang-module,slang-library", "Slang Module/Library"},
            {"dir", "Container as a directory"},
        };
        options.addValues(pairs, SLANG_COUNT_OF(pairs));
    }

    /* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! General !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

    options.setCategory("General");

    const Option generalOpts[] = {
        {OptionKind::MacroDefine,
         "-D?...",
         "-D<name>[=<value>], -D <name>[=<value>]",
         "Insert a preprocessor macro.\n"
         "The space between - D and <name> is optional. If no <value> is specified, Slang will "
         "define the macro with an empty value."},
        {OptionKind::DepFile,
         "-depfile",
         "-depfile <path>",
         "Save the source file dependency list in a file."},
        {OptionKind::EntryPointName,
         "-entry",
         "-entry <name>",
         "Specify the name of an entry-point function.\n"
         "When compiling from a single file, this defaults to main if you specify a stage using "
         "-stage.\n"
         "Multiple -entry options may be used in a single invocation. "
         "When they do, the file associated with the entry point will be the first one found when "
         "searching to the left in the command line.\n"
         "If no -entry options are given, compiler will use [shader(...)] "
         "attributes to detect entry points."},
        {OptionKind::Specialize,
         "-specialize",
         "-specialize <typename>",
         "Specialize the last entrypoint with <typename>.\n"},
        {OptionKind::EmitIr,
         "-emit-ir",
         nullptr,
         "Emit IR typically as a '.slang-module' when outputting to a container."},
        {OptionKind::Help,
         "-h,-help,--help",
         "-h or -h <help-category>",
         "Print this message, or help in specified category."},
        {OptionKind::HelpStyle, "-help-style", "-help-style <help-style>", "Help formatting style"},
        {OptionKind::Include,
         "-I?...",
         "-I<path>, -I <path>",
         "Add a path to be used in resolving '#include' "
         "and 'import' operations."},
        {OptionKind::Language,
         "-lang",
         "-lang <language>",
         "Set the language for the following input files."},
        {OptionKind::MatrixLayoutColumn,
         "-matrix-layout-column-major",
         nullptr,
         "Set the default matrix layout to column-major."},
        {OptionKind::MatrixLayoutRow,
         "-matrix-layout-row-major",
         nullptr,
         "Set the default matrix layout to row-major."},
        {OptionKind::RestrictiveCapabilityCheck,
         "-restrictive-capability-check",
         nullptr,
         "Many capability warnings will become an error."},
        {OptionKind::IgnoreCapabilities,
         "-ignore-capabilities",
         nullptr,
         "Do not warn or error if capabilities are violated"},
        {OptionKind::MinimumSlangOptimization,
         "-minimum-slang-optimization",
         nullptr,
         "Perform minimum code optimization in Slang to favor compilation time."},
        {OptionKind::DisableNonEssentialValidations,
         "-disable-non-essential-validations",
         nullptr,
         "Disable non-essential IR validations such as use of uninitialized variables."},
        {OptionKind::DisableSourceMap,
         "-disable-source-map",
         nullptr,
         "Disable source mapping in the Obfuscation."},
        {OptionKind::ModuleName,
         "-module-name",
         "-module-name <name>",
         "Set the module name to use when compiling multiple .slang source files into a single "
         "module."},
        {OptionKind::Output,
         "-o",
         "-o <path>",
         "Specify a path where generated output should be written.\n"
         "If no -target or -stage is specified, one may be inferred "
         "from file extension (see <file-extension>). "
         "If multiple -target options and a single -entry are present, each -o "
         "associates with the first -target to its left. "
         "Otherwise, if multiple -entry options are present, each -o associates "
         "with the first -entry to its left, and with the -target that matches "
         "the one inferred from <path>."},
        {OptionKind::Profile,
         "-profile",
         "-profile <profile>[+<capability>...]",
         "Specify the shader profile for code generation.\n"
         "Accepted profiles are:\n"
         "* sm_{4_0,4_1,5_0,5_1,6_0,6_1,6_2,6_3,6_4,6_5,6_6}\n"
         "* glsl_{110,120,130,140,150,330,400,410,420,430,440,450,460}\n"
         "Additional profiles that include -stage information:\n"
         "* {vs,hs,ds,gs,ps}_<version>\n"
         "See -capability for information on <capability>\n"
         "When multiple -target options are present, each -profile associates "
         "with the first -target to its left."},
        {OptionKind::Stage,
         "-stage",
         "-stage <stage>",
         "Specify the stage of an entry-point function.\n"
         "When multiple -entry options are present, each -stage associated with "
         "the first -entry to its left.\n"
         "May be omitted if entry-point function has a [shader(...)] attribute; "
         "otherwise required for each -entry option."},
        {OptionKind::Target,
         "-target",
         "-target <target>",
         "Specifies the format in which code should be generated."},
        {OptionKind::Version,
         "-v,-version",
         nullptr,
         "Display the build version. This is the contents of git describe --tags.\n"
         "It is typically only set from automated builds(such as distros available on github).A "
         "user build will by default be 'unknown'."},
        {OptionKind::WarningsAsErrors,
         "-warnings-as-errors",
         "-warnings-as-errors all or -warnings-as-errors <id>[,<id>...]",
         "all - Treat all warnings as errors.\n"
         "<id>[,<id>...]: Treat specific warning ids as errors.\n"},
        {OptionKind::DisableWarnings,
         "-warnings-disable",
         "-warnings-disable <id>[,<id>...]",
         "Disable specific warning ids."},
        {OptionKind::EnableWarning, "-W...", "-W<id>", "Enable a warning with the specified id."},
        {OptionKind::DisableWarning, "-Wno-...", "-Wno-<id>", "Disable warning with <id>"},
        {OptionKind::DumpWarningDiagnostics,
         "-dump-warning-diagnostics",
         nullptr,
         "Dump to output list of warning diagnostic numeric and name ids."},
        {OptionKind::InputFilesRemain,
         "--",
         nullptr,
         "Treat the rest of the command line as input files."},
        {OptionKind::ReportDownstreamTime,
         "-report-downstream-time",
         nullptr,
         "Reports the time spent in the downstream compiler."},
        {OptionKind::ReportPerfBenchmark,
         "-report-perf-benchmark",
         nullptr,
         "Reports compiler performance benchmark results."},
        {OptionKind::ReportCheckpointIntermediates,
         "-report-checkpoint-intermediates",
         nullptr,
         "Reports information about checkpoint contexts used for reverse-mode automatic "
         "differentiation."},
        {OptionKind::SkipSPIRVValidation,
         "-skip-spirv-validation",
         nullptr,
         "Skips spirv validation."},
        {OptionKind::SourceEmbedStyle,
         "-source-embed-style",
         "-source-embed-style <source-embed-style>",
         "If source embedding is enabled, defines the style used. When enabled (with any style "
         "other than `none`), "
         "will write compile results into embeddable source for the target language. "
         "If no output file is specified, the output is written to stdout. If an output file is "
         "specified "
         "it is written either to that file directly (if it is appropriate for the target "
         "language), "
         "or it will be output to the filename with an appropriate extension.\n\n"
         "Note for C/C++ with u16/u32/u64 types it is necessary to have \"#include <stdint.h>\" "
         "before the generated file.\n"},
        {OptionKind::SourceEmbedName,
         "-source-embed-name",
         "-source-embed-name <name>",
         "The name used as the basis for variables output for source embedding."},
        {OptionKind::SourceEmbedLanguage,
         "-source-embed-language",
         "-source-embed-language <language>",
         "The language to be used for source embedding. Defaults to C/C++. Currently only C/C++ "
         "are supported"},
        {OptionKind::DisableShortCircuit,
         "-disable-short-circuit",
         nullptr,
         "Disable short-circuiting for \"&&\" and \"||\" operations"},
        {OptionKind::UnscopedEnum,
         "-unscoped-enum",
         nullptr,
         "Treat enums types as unscoped by default."},
        {OptionKind::PreserveParameters,
         "-preserve-params",
         nullptr,
         "Preserve all resource parameters in the output code, even if they are not used by the "
         "shader."},
        {OptionKind::EmitReflectionJSON,
         "-reflection-json",
         "reflection-json <path>",
         "Emit reflection data in JSON format to a file."}};

    _addOptions(makeConstArrayView(generalOpts), options);

    /* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Target !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

    options.setCategory("Target");

    StringBuilder vkShiftNames;
    {
        for (auto nameSlice : NameValueUtil::getNames(
                 NameValueUtil::NameKind::All,
                 HLSLToVulkanLayoutOptions::getKindInfos()))
        {
            // -fvk-{b|s|t|u}-shift
            vkShiftNames << "-fvk-" << nameSlice << "-shift,";
        }
        // remove last ,
        vkShiftNames.reduceLength(vkShiftNames.getLength() - 1);
    }

    const Option targetOpts[] = {
        {OptionKind::Capability,
         "-capability",
         "-capability <capability>[+<capability>...]",
         "Add optional capabilities to a code generation target. See Capabilities below."},
        {OptionKind::DefaultImageFormatUnknown,
         "-default-image-format-unknown",
         nullptr,
         "Set the format of R/W images with unspecified format to 'unknown'. Otherwise try to "
         "guess the format."},
        {OptionKind::DisableDynamicDispatch,
         "-disable-dynamic-dispatch",
         nullptr,
         "Disables generating dynamic dispatch code."},
        {OptionKind::DisableSpecialization,
         "-disable-specialization",
         nullptr,
         "Disables generics and specialization pass."},
        {OptionKind::FloatingPointMode,
         "-fp-mode,-floating-point-mode",
         "-fp-mode <fp-mode>, -floating-point-mode <fp-mode>",
         "Control floating point optimizations"},
        {OptionKind::DebugInformation,
         "-g...",
         "-g, -g<debug-info-format>, -g<debug-level>",
         "Include debug information in the generated code, where possible.\n"
         "<debug-level> is the amount of information, 0..3, unspecified means 2\n"
         "<debug-info-format> specifies a debugging info format\n"
         "It is valid to have multiple -g options, such as a <debug-level> and a "
         "<debug-info-format>"},
        {OptionKind::LineDirectiveMode,
         "-line-directive-mode",
         "-line-directive-mode <line-directive-mode>",
         "Sets how the `#line` directives should be produced. Available options are:\n"
         "If not specified, default behavior is to use C-style `#line` directives "
         "for HLSL and C/C++ output, and traditional GLSL-style `#line` directives "
         "for GLSL output."},
        {OptionKind::Optimization,
         "-O...",
         "-O<optimization-level>",
         "Set the optimization level."},
        {OptionKind::Obfuscate,
         "-obfuscate",
         nullptr,
         "Remove all source file information from outputs."},
        {OptionKind::GLSLForceScalarLayout,
         "-force-glsl-scalar-layout,-fvk-use-scalar-layout",
         nullptr,
         "Make data accessed through ConstantBuffer, ParameterBlock, StructuredBuffer, "
         "ByteAddressBuffer and general pointers follow the 'scalar' layout when targeting GLSL or "
         "SPIRV."},
        {OptionKind::ForceDXLayout,
         "-fvk-use-dx-layout",
         nullptr,
         "Pack members using FXCs member packing rules when targeting GLSL or SPIRV."},
        {OptionKind::VulkanBindShift,
         vkShiftNames.getBuffer(),
         "-fvk-<vulkan-shift>-shift <N> <space>",
         "For example '-fvk-b-shift <N> <space>' shifts by N the inferred binding numbers for all "
         "resources in 'b' registers of space <space>. "
         "For a resource attached with :register(bX, <space>) but not [vk::binding(...)], "
         "sets its Vulkan descriptor set to <space> and binding number to X + N. If you need to "
         "shift the "
         "inferred binding numbers for more than one space, provide more than one such option. "
         "If more than one such option is provided for the same space, the last one takes effect. "
         "If you need to shift the inferred binding numbers for all sets, use 'all' as <space>. "
         "\n"
         "* [DXC "
         "description](https://github.com/Microsoft/DirectXShaderCompiler/blob/main/docs/"
         "SPIR-V.rst#implicit-binding-number-assignment)\n"
         "* [GLSL "
         "wiki](https://github.com/KhronosGroup/glslang/wiki/"
         "HLSL-FAQ#auto-mapped-binding-numbers)\n"},
        {OptionKind::VulkanBindGlobals,
         "-fvk-bind-globals",
         "-fvk-bind-globals <N> <descriptor-set>",
         "Places the $Globals cbuffer at descriptor set <descriptor-set> and binding <N>.\n"
         "It lets you specify the descriptor for the source at a certain register.\n"
         "* [DXC "
         "description](https://github.com/Microsoft/DirectXShaderCompiler/blob/main/docs/"
         "SPIR-V.rst#implicit-binding-number-assignment)\n"},
        {OptionKind::VulkanInvertY,
         "-fvk-invert-y",
         nullptr,
         "Negates (additively inverts) SV_Position.y before writing to stage output."},
        {OptionKind::VulkanUseDxPositionW,
         "-fvk-use-dx-position-w",
         nullptr,
         "Reciprocates (multiplicatively inverts) SV_Position.w after reading from stage input. "
         "For use in fragment shaders only."},
        {OptionKind::VulkanUseEntryPointName,
         "-fvk-use-entrypoint-name",
         nullptr,
         "Uses the entrypoint name from the source instead of 'main' in the spirv output."},
        {OptionKind::VulkanUseGLLayout,
         "-fvk-use-gl-layout",
         nullptr,
         "Use std430 layout instead of D3D buffer layout for raw buffer load/stores."},
        {OptionKind::VulkanEmitReflection,
         "-fspv-reflect",
         nullptr,
         "Include reflection decorations in the resulting SPIRV for shader parameters."},
        {OptionKind::EnableEffectAnnotations,
         "-enable-effect-annotations",
         nullptr,
         "Enables support for legacy effect annotation syntax."},
        {OptionKind::EmitSpirvViaGLSL,
         "-emit-spirv-via-glsl",
         nullptr,
         "Generate SPIR-V output by compiling generated GLSL with glslang"},
        {OptionKind::EmitSpirvDirectly,
         "-emit-spirv-directly",
         nullptr,
         "Generate SPIR-V output directly (default)"},
        {OptionKind::SPIRVCoreGrammarJSON,
         "-spirv-core-grammar",
         nullptr,
         "A path to a specific spirv.core.grammar.json to use when generating SPIR-V output"},
        {OptionKind::IncompleteLibrary,
         "-incomplete-library",
         nullptr,
         "Allow generating code from incomplete libraries with unresolved external functions"},
        {OptionKind::BindlessSpaceIndex,
         "-bindless-space-index",
         "-bindless-space-index <index>",
         "Specify the space index for the system defined global bindless resource array."}};

    _addOptions(makeConstArrayView(targetOpts), options);

    /* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Downstream !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

    options.setCategory("Downstream");

    {
        auto namesList = NameValueUtil::getNames(
            NameValueUtil::NameKind::First,
            TypeTextUtil::getCompilerInfos());
        StringBuilder names;
        for (auto name : namesList)
        {
            names << "-" << name << "-path,";
        }
        // remove last ,
        names.reduceLength(names.getLength() - 1);

        options.add(
            names.getBuffer(),
            "-<compiler>-path <path>",
            "Specify path to a downstream <compiler> "
            "executable or library.\n",
            UserValue(OptionKind::CompilerPath));
    }

    const Option downstreamOpts[] = {
        {OptionKind::DefaultDownstreamCompiler,
         "-default-downstream-compiler",
         "-default-downstream-compiler <language> <compiler>",
         "Set a default compiler for the given language. See -lang for the list of languages."},
        {OptionKind::DownstreamArgs,
         "-X...",
         "-X<compiler> <option> -X<compiler>... <options> -X.",
         "Pass arguments to downstream <compiler>. Just -X<compiler> passes just the next argument "
         "to the downstream compiler. -X<compiler>... options -X. will pass *all* of the options "
         "inbetween the opening -X and -X. to the downstream compiler."},
        {OptionKind::PassThrough,
         "-pass-through",
         "-pass-through <compiler>",
         "Pass the input through mostly unmodified to the "
         "existing compiler <compiler>.\n"
         "These are intended for debugging/testing purposes, when you want to be able to see what "
         "these existing compilers do with the \"same\" input and options"},
    };

    _addOptions(makeConstArrayView(downstreamOpts), options);

    /* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Repro !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

    options.setCategory("Repro");

    const Option reproOpts[] = {
        {OptionKind::DumpReproOnError,
         "-dump-repro-on-error",
         nullptr,
         "Dump `.slang-repro` file on any compilation error."},
        {OptionKind::ExtractRepro,
         "-extract-repro",
         "-extract-repro <name>",
         "Extract the repro files into a folder."},
        {OptionKind::LoadReproDirectory,
         "-load-repro-directory",
         "-load-repro-directory <path>",
         "Use repro along specified path"},
        {OptionKind::LoadRepro, "-load-repro", "-load-repro <name>", "Load repro"},
        {OptionKind::ReproFileSystem,
         "-repro-file-system",
         "-repro-file-system <name>",
         "Use a repro as a file system"},
        {OptionKind::DumpRepro,
         "-dump-repro",
         nullptr,
         "Dump a `.slang-repro` file that can be used to reproduce "
         "a compilation on another machine.\n"},
        {OptionKind::ReproFallbackDirectory,
         "-repro-fallback-directory <path>",
         "Specify a directory to use if a file isn't found in a repro. Should be specified "
         "*before* any repro usage such as `load-repro`. \n"
         "There are two *special* directories: \n\n"
         " * 'none:' indicates no fallback, so if the file isn't found in the repro compliation "
         "will fail\n"
         " * 'default:' is the default (which is the OS file system)"}};

    _addOptions(makeConstArrayView(reproOpts), options);

    /* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Debugging !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

    options.setCategory("Debugging");

    const Option debuggingOpts[] = {
        {OptionKind::DumpAst,
         "-dump-ast",
         nullptr,
         "Dump the AST to a .slang-ast file next to the input."},
        {OptionKind::DumpIntermediatePrefix,
         "-dump-intermediate-prefix",
         "-dump-intermediate-prefix <prefix>",
         "File name prefix for -dump-intermediates outputs, default is 'slang-dump-'"},
        {OptionKind::DumpIntermediates,
         "-dump-intermediates",
         nullptr,
         "Dump intermediate outputs for debugging."},
        {OptionKind::DumpIr, "-dump-ir", nullptr, "Dump the IR for debugging."},
        {OptionKind::DumpIrIds,
         "-dump-ir-ids",
         nullptr,
         "Dump the IDs with -dump-ir (debug builds only)"},
        {OptionKind::PreprocessorOutput,
         "-E,-output-preprocessor",
         nullptr,
         "Output the preprocessing result and exit."},
        {OptionKind::NoCodeGen,
         "-no-codegen",
         nullptr,
         "Skip the code generation step, just check the code and generate layout."},
        {OptionKind::OutputIncludes,
         "-output-includes",
         nullptr,
         "Print the hierarchy of the processed source files."},
        {OptionKind::SerialIr,
         "-serial-ir",
         nullptr,
         "Serialize the IR between front-end and back-end."},
        {OptionKind::SkipCodeGen, "-skip-codegen", nullptr, "Skip the code generation phase."},
        {OptionKind::ValidateIr, "-validate-ir", nullptr, "Validate the IR between the phases."},
        {OptionKind::VerbosePaths,
         "-verbose-paths",
         nullptr,
         "When displaying diagnostic output aim to display more detailed path information. "
         "In practice this is typically the complete 'canonical' path to the source file used."},
        {OptionKind::VerifyDebugSerialIr,
         "-verify-debug-serial-ir",
         nullptr,
         "Verify IR in the front-end."},
        {OptionKind::DumpModule, "-dump-module", nullptr, "Disassemble and print the module IR."}};
    _addOptions(makeConstArrayView(debuggingOpts), options);

    /* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Experimental !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

    options.setCategory("Experimental");

    const Option experimentalOpts[] = {
        {OptionKind::FileSystem,
         "-file-system",
         "-file-system <file-system-type>",
         "Set the filesystem hook to use for a compile request."},
        {OptionKind::Heterogeneous,
         "-heterogeneous",
         nullptr,
         "Output heterogeneity-related code."},
        {OptionKind::NoMangle,
         "-no-mangle",
         nullptr,
         "Do as little mangling of names as possible."},
        {OptionKind::NoHLSLBinding,
         "-no-hlsl-binding",
         nullptr,
         "Do not include explicit parameter binding semantics in the output HLSL code,"
         "except for parameters that has explicit bindings in the input source."},
        {OptionKind::NoHLSLPackConstantBufferElements,
         "-no-hlsl-pack-constant-buffer-elements",
         nullptr,
         "Do not pack elements of constant buffers into structs in the output HLSL code."},
        {OptionKind::ValidateUniformity,
         "-validate-uniformity",
         nullptr,
         "Perform uniformity validation analysis."},
        {OptionKind::AllowGLSL, "-allow-glsl", nullptr, "Enable GLSL as an input language."},
        {OptionKind::EnableExperimentalPasses,
         "-enable-experimental-passes",
         nullptr,
         "Enable experimental compiler passes"},
        {OptionKind::EmbedDownstreamIR,
         "-embed-downstream-ir",
         nullptr,
         "Embed downstream IR into emitted slang IR"},
    };
    _addOptions(makeConstArrayView(experimentalOpts), options);

    /* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Internal !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

    options.setCategory("Internal");

    const Option internalOpts[] = {
        {OptionKind::ArchiveType,
         "-archive-type",
         "-archive-type <archive-type>",
         "Set the archive type for -save-core-module. Default is zip."},
        {OptionKind::CompileCoreModule,
         "-compile-core-module",
         nullptr,
         "Compile the core module from embedded sources. "
         "Will return a failure if there is already a core module available."},
        {OptionKind::Doc, "-doc", nullptr, "Write documentation for -compile-core-module"},
        {OptionKind::IrCompression,
         "-ir-compression",
         "-ir-compression <type>",
         "Set compression for IR and AST outputs.\n"
         "Accepted compression types: none, lite"},
        {OptionKind::LoadCoreModule,
         "-load-core-module",
         "-load-core-module <filename>",
         "Load the core module from file."},
        {OptionKind::ReferenceModule, "-r", "-r <name>", "reference module <name>"},
        {OptionKind::SaveCoreModule,
         "-save-core-module",
         "-save-core-module <filename>",
         "Save the core module to an archive file."},
        {OptionKind::SaveCoreModuleBinSource,
         "-save-core-module-bin-source",
         "-save-core-module-bin-source <filename>",
         "Same as -save-core-module but output "
         "the data as a C array.\n"},
        {OptionKind::SaveGLSLModuleBinSource,
         "-save-glsl-module-bin-source",
         "-save-glsl-module-bin-source <filename>",
         "Save the serialized glsl module "
         "as a C array.\n"},
        {OptionKind::TrackLiveness,
         "-track-liveness",
         nullptr,
         "Enable liveness tracking. Places SLANG_LIVE_START, and SLANG_LIVE_END in output source "
         "to indicate value liveness."},
        {OptionKind::LoopInversion,
         "-loop-inversion",
         nullptr,
         "Enable loop inversion in the code-gen optimization. Default is off"},
    };
    _addOptions(makeConstArrayView(internalOpts), options);

    /* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Deprecated !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

    options.setCategory("Deprecated");

    const Option deprecatedOpts[] = {
        {OptionKind::ParameterBlocksUseRegisterSpaces,
         "-parameter-blocks-use-register-spaces",
         nullptr,
         "Parameter blocks will use register spaces"},
        {OptionKind::ZeroInitialize,
         "-zero-initialize",
         nullptr,
         "Initialize all variables to zero."
         "Structs will set all struct-fields without an init expression to 0."
         "All variables will call their default constructor if not explicitly initialized as "
         "usual."},
    };
    _addOptions(makeConstArrayView(deprecatedOpts), options);

    // We can now check that the whole range is available. If this fails it means there
    // is an enum in the list that hasn't been setup as an option!
    SLANG_ASSERT(options.hasContiguousUserValueRange(
        CommandOptions::LookupKind::Option,
        UserValue(0),
        UserValue(OptionKind::CountOfParsableOptions)));
    SLANG_ASSERT(options.hasContiguousUserValueRange(
        CommandOptions::LookupKind::Category,
        UserValue(0),
        UserValue(ValueCategory::CountOf)));
}

SlangResult _addLibraryReference(
    EndToEndCompileRequest* req,
    String path,
    IArtifact* artifact,
    bool includeEntryPoint);

class ReproPathVisitor : public Slang::Path::Visitor
{
public:
    virtual void accept(Slang::Path::Type type, const Slang::UnownedStringSlice& filename)
        SLANG_OVERRIDE
    {
        if (type == Path::Type::File && Path::getPathExt(filename) == "slang-repro")
        {
            m_filenames.add(filename);
        }
    }

    Slang::List<String> m_filenames;
};

struct OptionsParser
{
    // A "translation unit" represents one or more source files
    // that are processed as a single entity when it comes to
    // semantic checking.
    //
    // For languages like HLSL, GLSL, and C, a translation unit
    // is usually a single source file (which can then go on
    // to `#include` other files into the same translation unit).
    //
    // For Slang, we support having multiple source files in
    // a single translation unit, and indeed command-line `slangc`
    // will always put all the source files into a single translation
    // unit.
    //
    // We track information on the translation units that we
    // create during options parsing, so that we can assocaite
    // other entities with these translation units:
    //
    struct RawTranslationUnit
    {
        // What language is the translation unit using?
        //
        // Note: We do not support translation units that mix
        // languages.
        //
        SlangSourceLanguage sourceLanguage;

        // Certain naming conventions imply a stage for
        // a file with only a single entry point, and in
        // those cases we will try to infer the stage from
        // the file when it is possible.
        //
        Stage impliedStage;

        // We retain the Slang API level translation unit index,
        // which we will call an "ID" inside the options parsing code.
        //
        // This will almost always be the index into the
        // `rawTranslationUnits` array below, but could conceivably,
        // be mismatched if we were parsing options for a compile
        // request that already had some translation unit(s) added
        // manually.
        //
        int translationUnitID;
    };

    // An entry point represents a function to be checked and possibly have
    // code generated in one of our translation units. An entry point
    // needs to have an associated stage, which might come via the
    // `-stage` command line option, or a `[shader("...")]` attribute
    // in the source code.
    //
    struct RawEntryPoint
    {
        String name;
        Stage stage = Stage::Unknown;
        int translationUnitIndex = -1;
        int entryPointID = -1;
        List<String> specializationArgs;
        // State for tracking command-line errors
        bool conflictingStagesSet = false;
        bool redundantStageSet = false;
    };

    struct RawOutput
    {
        String path;
        CodeGenTarget impliedFormat = CodeGenTarget::Unknown;
        int targetIndex = -1;
        int entryPointIndex = -1;
        bool isWholeProgram = false;
    };

    struct RawTarget
    {
        CodeGenTarget format = CodeGenTarget::Unknown;
        int targetID = -1;
        CompilerOptionSet optionSet;

        // State for tracking command-line errors
        bool conflictingProfilesSet = false;
        bool redundantProfileSet = false;
    };

    int addTranslationUnit(SlangSourceLanguage language, Stage impliedStage);

    void addInputSlangPath(String const& path);

    void addInputForeignShaderPath(
        String const& path,
        SlangSourceLanguage language,
        Stage impliedStage);

    static Profile::RawVal findGlslProfileFromPath(const String& path);

    SlangResult addInputPath(
        char const* inPath,
        SourceLanguage langOverride = SourceLanguage::Unknown);

    void addOutputPath(String const& path, CodeGenTarget impliedFormat);

    void addOutputPath(char const* inPath);
    RawEntryPoint* getCurrentEntryPoint();

    void setStage(RawEntryPoint* rawEntryPoint, Stage stage);

    RawTarget* getCurrentTarget();
    void setProfileVersion(RawTarget* rawTarget, ProfileVersion profileVersion);
    void setProfile(RawTarget* rawTarget, Profile profile);
    void addCapabilityAtom(RawTarget* rawTarget, CapabilityName atom);

    void setFloatingPointMode(RawTarget* rawTarget, FloatingPointMode mode);

    SlangResult parse(SlangCompileRequest* compileRequest, int argc, char const* const* argv);

    SlangResult _parse(int argc, char const* const* argv);

    static bool _passThroughRequiresStage(PassThroughMode passThrough);

    SlangResult _compileReproDirectory(
        SlangSession* session,
        EndToEndCompileRequest* originalRequest,
        const String& dir);

    // Pass Severity::Disabled to allow any original severity
    SlangResult _overrideDiagnostics(
        const UnownedStringSlice& identifierList,
        Severity originalSeverity,
        Severity overrideSeverity);

    // Pass Severity::Disabled to allow any original severity
    SlangResult _overrideDiagnostic(
        const UnownedStringSlice& identifier,
        Severity originalSeverity,
        Severity overrideSeverity);

    SlangResult _dumpDiagnostics(Severity originalSeverity);

    template<typename T>
    SlangResult _getValue(const CommandLineArg& arg, const UnownedStringSlice& name, T& ioValue)
    {
        CommandOptions::UserValue value;
        SLANG_RETURN_ON_FAIL(
            _getValue(ValueCategory(GetValueCategory<T>::Value), arg, name, value));
        ioValue = T(value);
        return SLANG_OK;
    }

    SlangResult _getValue(
        ValueCategory valueCategory,
        const CommandLineArg& arg,
        const UnownedStringSlice& name,
        CommandOptions::UserValue& outValue);
    SlangResult _getValue(
        ValueCategory valueCategory,
        const CommandLineArg& arg,
        CommandOptions::UserValue& outValue);
    SlangResult _getValue(
        const ConstArrayView<ValueCategory>& valueCategories,
        const CommandLineArg& arg,
        const UnownedStringSlice& name,
        ValueCategory& outCat,
        CommandOptions::UserValue& outValue);

    SlangResult _expectValue(ValueCategory valueCategory, CommandOptions::UserValue& outValue);
    SlangResult _expectInt(const CommandLineArg& arg, Int& outInt);

    template<typename T>
    SlangResult _expectValue(T& ioValue)
    {
        CommandOptions::UserValue value;
        SLANG_RETURN_ON_FAIL(_expectValue(ValueCategory(GetValueCategory<T>::Value), value));
        ioValue = T(value);
        return SLANG_OK;
    }

    void _appendUsageTitle(StringBuilder& out);
    void _appendMinimalUsage(StringBuilder& out);
    void _outputMinimalUsage();

    SlangResult addReferencedModule(String path, SourceLoc loc, bool includeEntryPoint);
    SlangResult _parseReferenceModule(const CommandLineArg& arg);
    SlangResult _parseReproFileSystem(const CommandLineArg& arg);
    SlangResult _parseLoadRepro(const CommandLineArg& arg);
    SlangResult _parseDebugInformation(const CommandLineArg& arg);
    SlangResult _parseProfile(const CommandLineArg& arg);
    SlangResult _parseHelp(const CommandLineArg& arg);

    SlangSession* m_session = nullptr;
    SlangCompileRequest* m_compileRequest = nullptr;

    Slang::EndToEndCompileRequest* m_requestImpl = nullptr;

    List<RawTarget> m_rawTargets;

    RawTarget m_defaultTarget;

    //
    // We collect the entry points in a "raw" array so that we can
    // possibly associate them with a stage or translation unit
    // after the fact.
    //
    List<RawEntryPoint> m_rawEntryPoints;

    // In the case where we have only a single entry point,
    // the entry point and its options might be specified out
    // of order, so we will keep a single `RawEntryPoint` around
    // and use it as the target for any state-setting options
    // before the first "proper" entry point is specified.
    RawEntryPoint m_defaultEntryPoint;

    List<RawTranslationUnit> m_rawTranslationUnits;

    // If we already have a translation unit for Slang code, then this will give its index.
    // If not, it will be `-1`.
    int m_slangTranslationUnitIndex = -1;

    int m_translationUnitCount = 0;
    int m_currentTranslationUnitIndex = -1;

    bool m_hasLoadedRepro = false;
    bool m_compileCoreModule = false;
    slang::CompileCoreModuleFlags m_compileCoreModuleFlags;

    SlangArchiveType m_archiveType = SLANG_ARCHIVE_TYPE_RIFF_LZ4;

    List<RawOutput> m_rawOutputs;

    DiagnosticSink m_parseSink;
    DiagnosticSink* m_sink = nullptr;

    FrontEndCompileRequest* m_frontEndReq = nullptr;

    CommandLineReader m_reader;

    CommandOptionsWriter::Style m_helpStyle = CommandOptionsWriter::Style::Text;

    CommandOptions* m_cmdOptions = nullptr;
    CommandLineContext* m_cmdLineContext = nullptr;
};

int OptionsParser::addTranslationUnit(SlangSourceLanguage language, Stage impliedStage)
{
    auto translationUnitIndex = m_rawTranslationUnits.getCount();
    auto translationUnitID = m_compileRequest->addTranslationUnit(language, nullptr);

    // As a sanity check: the API should be returning the same translation
    // unit index as we maintain internally. This invariant would only
    // be broken if we decide to support a mix of translation units specified
    // via API, and ones specified via command-line arguments.
    //
    SLANG_RELEASE_ASSERT(Index(translationUnitID) == translationUnitIndex);

    RawTranslationUnit rawTranslationUnit;
    rawTranslationUnit.sourceLanguage = language;
    rawTranslationUnit.translationUnitID = translationUnitID;
    rawTranslationUnit.impliedStage = impliedStage;

    m_rawTranslationUnits.add(rawTranslationUnit);

    return int(translationUnitIndex);
}

void OptionsParser::addInputSlangPath(String const& path)
{
    // All of the input .slang files will be grouped into a single logical translation unit,
    // which we create lazily when the first .slang file is encountered.
    if (m_slangTranslationUnitIndex == -1)
    {
        m_translationUnitCount++;
        m_slangTranslationUnitIndex =
            addTranslationUnit(SLANG_SOURCE_LANGUAGE_SLANG, Stage::Unknown);
    }

    m_compileRequest->addTranslationUnitSourceFile(
        m_rawTranslationUnits[m_slangTranslationUnitIndex].translationUnitID,
        path.begin());

    // Set the translation unit to be used by subsequent entry points
    m_currentTranslationUnitIndex = m_slangTranslationUnitIndex;
}

void OptionsParser::addInputForeignShaderPath(
    String const& path,
    SlangSourceLanguage language,
    Stage impliedStage)
{
    m_translationUnitCount++;
    m_currentTranslationUnitIndex = addTranslationUnit(language, impliedStage);

    m_compileRequest->addTranslationUnitSourceFile(
        m_rawTranslationUnits[m_currentTranslationUnitIndex].translationUnitID,
        path.begin());
}

/* static */ Profile::RawVal OptionsParser::findGlslProfileFromPath(const String& path)
{
    struct Entry
    {
        const char* ext;
        Profile::RawVal profileId;
    };

    static const Entry entries[] = {
        {".frag", Profile::GLSL_Fragment},
        {".geom", Profile::GLSL_Geometry},
        {".tesc", Profile::GLSL_TessControl},
        {".tese", Profile::GLSL_TessEval},
        {".comp", Profile::GLSL_Compute}};

    for (Index i = 0; i < SLANG_COUNT_OF(entries); ++i)
    {
        const Entry& entry = entries[i];
        if (path.endsWith(entry.ext))
        {
            return entry.profileId;
        }
    }
    return Profile::Unknown;
}

SlangSourceLanguage findSourceLanguageFromPath(const String& path, Stage& outImpliedStage)
{
    struct Entry
    {
        const char* ext;
        SlangSourceLanguage sourceLanguage;
        SlangStage impliedStage;
    };

    static const Entry entries[] = {
        {".slang", SLANG_SOURCE_LANGUAGE_SLANG, SLANG_STAGE_NONE},

        {".hlsl", SLANG_SOURCE_LANGUAGE_HLSL, SLANG_STAGE_NONE},
        {".fx", SLANG_SOURCE_LANGUAGE_HLSL, SLANG_STAGE_NONE},

        {".glsl", SLANG_SOURCE_LANGUAGE_GLSL, SLANG_STAGE_NONE},
        {".vert", SLANG_SOURCE_LANGUAGE_GLSL, SLANG_STAGE_VERTEX},
        {".frag", SLANG_SOURCE_LANGUAGE_GLSL, SLANG_STAGE_FRAGMENT},
        {".geom", SLANG_SOURCE_LANGUAGE_GLSL, SLANG_STAGE_GEOMETRY},
        {".tesc", SLANG_SOURCE_LANGUAGE_GLSL, SLANG_STAGE_HULL},
        {".tese", SLANG_SOURCE_LANGUAGE_GLSL, SLANG_STAGE_DOMAIN},
        {".comp", SLANG_SOURCE_LANGUAGE_GLSL, SLANG_STAGE_COMPUTE},
        {".mesh", SLANG_SOURCE_LANGUAGE_GLSL, SLANG_STAGE_MESH},
        {".task", SLANG_SOURCE_LANGUAGE_GLSL, SLANG_STAGE_AMPLIFICATION},
        {".rgen", SLANG_SOURCE_LANGUAGE_GLSL, SLANG_STAGE_RAY_GENERATION},
        {".rint", SLANG_SOURCE_LANGUAGE_GLSL, SLANG_STAGE_INTERSECTION},
        {".rahit", SLANG_SOURCE_LANGUAGE_GLSL, SLANG_STAGE_ANY_HIT},
        {".rchit", SLANG_SOURCE_LANGUAGE_GLSL, SLANG_STAGE_CLOSEST_HIT},
        {".rmiss", SLANG_SOURCE_LANGUAGE_GLSL, SLANG_STAGE_MISS},
        {".rcall", SLANG_SOURCE_LANGUAGE_GLSL, SLANG_STAGE_CALLABLE},

        {".c", SLANG_SOURCE_LANGUAGE_C, SLANG_STAGE_NONE},
        {".cpp", SLANG_SOURCE_LANGUAGE_CPP, SLANG_STAGE_NONE},
        {".cu", SLANG_SOURCE_LANGUAGE_CUDA, SLANG_STAGE_NONE},

        {".wgsl", SLANG_SOURCE_LANGUAGE_WGSL, SLANG_STAGE_NONE},
    };

    for (Index i = 0; i < SLANG_COUNT_OF(entries); ++i)
    {
        const Entry& entry = entries[i];
        if (path.endsWith(entry.ext))
        {
            outImpliedStage = Stage(entry.impliedStage);
            return entry.sourceLanguage;
        }
    }
    return SLANG_SOURCE_LANGUAGE_UNKNOWN;
}

SlangResult OptionsParser::addInputPath(char const* inPath, SourceLanguage langOverride)
{
    // look at the extension on the file name to determine
    // how we should handle it.
    String path = String(inPath);

    if (path.endsWith(".slang-module") || path.endsWith(".slang-lib"))
    {
        return addReferencedModule(path, SourceLoc(), false);
    }
    else if (path.endsWith(".slang") || langOverride == SourceLanguage::Slang)
    {
        // Plain old slang code
        addInputSlangPath(path);
        return SLANG_OK;
    }

    Stage impliedStage = Stage::Unknown;
    SlangSourceLanguage sourceLanguage = SlangSourceLanguage(langOverride);
    if (sourceLanguage == SLANG_SOURCE_LANGUAGE_UNKNOWN)
    {
        if (m_requestImpl->getLinkage()->m_optionSet.hasOption(CompilerOptionName::Language))
            sourceLanguage = SlangSourceLanguage(
                m_requestImpl->getLinkage()->m_optionSet.getEnumOption<SlangSourceLanguage>(
                    CompilerOptionName::Language));
        else
            sourceLanguage = findSourceLanguageFromPath(path, impliedStage);
    }
    if (sourceLanguage == SLANG_SOURCE_LANGUAGE_UNKNOWN)
    {
        m_requestImpl->getSink()->diagnose(
            SourceLoc(),
            Diagnostics::cannotDeduceSourceLanguage,
            inPath);
        return SLANG_FAIL;
    }

    addInputForeignShaderPath(path, sourceLanguage, impliedStage);

    return SLANG_OK;
}

void OptionsParser::addOutputPath(String const& path, CodeGenTarget impliedFormat)
{
    RawOutput rawOutput;
    rawOutput.path = path;
    rawOutput.impliedFormat = impliedFormat;
    m_rawOutputs.add(rawOutput);
}

void OptionsParser::addOutputPath(char const* inPath)
{
    String path = String(inPath);
    String ext = Path::getPathExt(path);

    if (ext == toSlice("slang-module") || ext == toSlice("slang-lib") || ext == toSlice("dir") ||
        ext == toSlice("zip"))
    {
        // These extensions don't indicate a artifact container, just that we want to emit IR
        // We want to emit IR
        m_requestImpl->m_emitIr = true;

        // We want to write out in an artfact "container", that can hold multiple artifacts.
        m_compileRequest->setOutputContainerFormat(SLANG_CONTAINER_FORMAT_SLANG_MODULE);

        m_requestImpl->m_containerOutputPath = path;
    }
    else
    {
        const SlangCompileTarget target =
            TypeTextUtil::findCompileTargetFromExtension(ext.getUnownedSlice());
        // If the target is not found the value returned is Unknown. This is okay because
        // we allow an unknown-format `-o`, assuming we get a target format
        // from another argument.
        addOutputPath(path, CodeGenTarget(target));
    }
}

OptionsParser::RawEntryPoint* OptionsParser::getCurrentEntryPoint()
{
    auto rawEntryPointCount = m_rawEntryPoints.getCount();
    return rawEntryPointCount ? &m_rawEntryPoints[rawEntryPointCount - 1] : &m_defaultEntryPoint;
}

void OptionsParser::setStage(RawEntryPoint* rawEntryPoint, Stage stage)
{
    if (rawEntryPoint->stage != Stage::Unknown)
    {
        rawEntryPoint->redundantStageSet = true;
        if (stage != rawEntryPoint->stage)
        {
            rawEntryPoint->conflictingStagesSet = true;
        }
    }
    rawEntryPoint->stage = stage;
}

OptionsParser::RawTarget* OptionsParser::getCurrentTarget()
{
    auto rawTargetCount = m_rawTargets.getCount();
    return rawTargetCount ? &m_rawTargets[rawTargetCount - 1] : &m_defaultTarget;
}

void OptionsParser::setProfileVersion(RawTarget* rawTarget, ProfileVersion profileVersion)
{
    if (rawTarget->optionSet.getProfileVersion() != ProfileVersion::Unknown)
    {
        rawTarget->redundantProfileSet = true;

        if (profileVersion != rawTarget->optionSet.getProfileVersion())
        {
            rawTarget->conflictingProfilesSet = true;
        }
    }
    rawTarget->optionSet.setProfileVersion(profileVersion);
}

void OptionsParser::setProfile(RawTarget* rawTarget, Profile profile)
{
    if (rawTarget->optionSet.getProfile() != Profile::Unknown)
    {
        rawTarget->redundantProfileSet = true;

        if (profile != rawTarget->optionSet.getProfile())
        {
            rawTarget->conflictingProfilesSet = true;
        }
    }
    rawTarget->optionSet.setProfile(profile);
}

void OptionsParser::addCapabilityAtom(RawTarget* rawTarget, CapabilityName atom)
{
    CapabilitySet capSet(atom);
    auto stageAtom = capSet.getUniquelyImpliedStageAtom();
    if (stageAtom != CapabilityAtom::Invalid)
    {
        Stage stage = getStageFromAtom(stageAtom);
        setStage(getCurrentEntryPoint(), stage);
    }
    rawTarget->optionSet.addCapabilityAtom(atom);
}

void OptionsParser::setFloatingPointMode(RawTarget* rawTarget, FloatingPointMode mode)
{
    rawTarget->optionSet.set(CompilerOptionName::FloatingPointMode, mode);
}

/* static */ bool OptionsParser::_passThroughRequiresStage(PassThroughMode passThrough)
{
    switch (passThrough)
    {
    case PassThroughMode::Glslang:
    case PassThroughMode::Dxc:
    case PassThroughMode::Fxc:
        {
            return true;
        }
    default:
        {
            return false;
        }
    }
}

static SlangResult _loadRepro(
    const String& path,
    DiagnosticSink* sink,
    EndToEndCompileRequest* request)
{
    List<uint8_t> buffer;
    SLANG_RETURN_ON_FAIL(ReproUtil::loadState(path, sink, buffer));

    auto requestState = ReproUtil::getRequest(buffer);
    MemoryOffsetBase base;
    base.set(buffer.getBuffer(), buffer.getCount());

    // If we can find a directory, that exists, we will set up a file system to load from that
    // directory
    ComPtr<ISlangFileSystem> optionalFileSystem;
    String dirPath;
    if (SLANG_SUCCEEDED(ReproUtil::calcDirectoryPathFromFilename(path, dirPath)))
    {
        SlangPathType pathType;
        if (SLANG_SUCCEEDED(Path::getPathType(dirPath, &pathType)) &&
            pathType == SLANG_PATH_TYPE_DIRECTORY)
        {
            optionalFileSystem = new RelativeFileSystem(OSFileSystem::getExtSingleton(), dirPath);
        }
    }

    SLANG_RETURN_ON_FAIL(ReproUtil::load(base, requestState, optionalFileSystem, request));

    return SLANG_OK;
}

SlangResult OptionsParser::_compileReproDirectory(
    SlangSession* session,
    EndToEndCompileRequest* originalRequest,
    const String& dir)
{
    auto stdOut = originalRequest->getWriter(WriterChannel::StdOutput);

    ReproPathVisitor visitor;
    Path::find(dir, nullptr, &visitor);

    for (auto filename : visitor.m_filenames)
    {
        // Create a fresh request
        ComPtr<slang::ICompileRequest> request;
        SLANG_ALLOW_DEPRECATED_BEGIN
        SLANG_RETURN_ON_FAIL(session->createCompileRequest(request.writeRef()));
        SLANG_ALLOW_DEPRECATED_END

        auto requestImpl = asInternal(request);

        // Copy over the fallback file system
        requestImpl->m_reproFallbackFileSystem = originalRequest->m_reproFallbackFileSystem;

        // Load the repro into it
        auto path = Path::combine(dir, filename);

        if (SLANG_FAILED(_loadRepro(path, m_sink, requestImpl)))
        {
            if (stdOut)
            {
                StringBuilder buf;
                buf << filename << " - Failed to load!\n";
            }
            continue;
        }

        if (stdOut)
        {
            StringBuilder buf;
            buf << filename << "\n";
            stdOut->write(buf.getBuffer(), buf.getLength());
        }

        StringBuilder bufs[Index(WriterChannel::CountOf)];
        ComPtr<ISlangWriter> writers[Index(WriterChannel::CountOf)];
        for (Index i = 0; i < Index(WriterChannel::CountOf); ++i)
        {
            writers[i] = new StringWriter(&bufs[0], 0);
            requestImpl->setWriter(WriterChannel(i), writers[i]);
        }

        if (SLANG_FAILED(requestImpl->compile()))
        {
            const char failed[] = "FAILED!\n";
            stdOut->write(failed, SLANG_COUNT_OF(failed) - 1);

            const auto& diagnostics = bufs[Index(WriterChannel::Diagnostic)];

            stdOut->write(diagnostics.getBuffer(), diagnostics.getLength());

            return SLANG_FAIL;
        }
    }

    if (stdOut)
    {
        const char end[] = "(END)\n";
        stdOut->write(end, SLANG_COUNT_OF(end) - 1);
    }

    return SLANG_OK;
}

SlangResult OptionsParser::_dumpDiagnostics(Severity originalSeverity)
{
    // Get the diagnostics and dump them
    auto diagnosticsLookup = getDiagnosticsLookup();

    StringBuilder buf;

    for (const auto& diagnostic : diagnosticsLookup->getDiagnostics())
    {
        if (originalSeverity != Severity::Disable && diagnostic->severity != originalSeverity)
        {
            continue;
        }

        buf.clear();

        buf << diagnostic->id << " : ";
        NameConventionUtil::convert(
            NameStyle::Camel,
            UnownedStringSlice(diagnostic->name),
            NameConvention::LowerKabab,
            buf);
        buf << "\n";
        m_sink->diagnoseRaw(Severity::Note, buf.getUnownedSlice());
    }

    return SLANG_OK;
}

void OptionsParser::_appendUsageTitle(StringBuilder& out)
{
    out << "Usage: slangc [options...] [--] <input files>\n\n";
}

void OptionsParser::_outputMinimalUsage()
{
    // Output usage info
    StringBuilder buf;
    _appendMinimalUsage(buf);

    m_sink->diagnoseRaw(Severity::Note, buf.getUnownedSlice());
}

void OptionsParser::_appendMinimalUsage(StringBuilder& out)
{
    _appendUsageTitle(out);
    out << "For help: slangc -h\n";
}


SlangResult OptionsParser::_getValue(
    ValueCategory valueCategory,
    const CommandLineArg& arg,
    const UnownedStringSlice& name,
    CommandOptions::UserValue& outValue)
{
    const auto optionIndex =
        m_cmdOptions->findOptionByCategoryUserValue(CommandOptions::UserValue(valueCategory), name);
    if (optionIndex < 0)
    {
        const auto categoryIndex =
            m_cmdOptions->findCategoryByUserValue(CommandOptions::UserValue(valueCategory));
        SLANG_ASSERT(categoryIndex >= 0);
        if (categoryIndex < 0)
        {
            return SLANG_FAIL;
        }

        List<UnownedStringSlice> names;
        m_cmdOptions->getCategoryOptionNames(categoryIndex, names);

        StringBuilder buf;
        StringUtil::join(names.getBuffer(), names.getCount(), toSlice(", "), buf);

        m_sink->diagnose(arg.loc, Diagnostics::unknownCommandLineValue, buf);
        return SLANG_FAIL;
    }

    outValue = m_cmdOptions->getOptionAt(optionIndex).userValue;
    return SLANG_OK;
}

SlangResult OptionsParser::_getValue(
    ValueCategory valueCategory,
    const CommandLineArg& arg,
    CommandOptions::UserValue& outValue)
{
    return _getValue(valueCategory, arg, arg.value.getUnownedSlice(), outValue);
}

SlangResult OptionsParser::_getValue(
    const ConstArrayView<ValueCategory>& valueCategories,
    const CommandLineArg& arg,
    const UnownedStringSlice& name,
    ValueCategory& outCat,
    CommandOptions::UserValue& outValue)
{
    auto& cmdOptions = asInternal(m_session)->m_commandOptions;

    for (auto valueCategory : valueCategories)
    {
        const auto optionIndex = cmdOptions.findOptionByCategoryUserValue(
            CommandOptions::UserValue(valueCategory),
            name);
        if (optionIndex >= 0)
        {
            outCat = valueCategory;
            outValue = cmdOptions.getOptionAt(optionIndex).userValue;
            return SLANG_OK;
        }
    }

    List<UnownedStringSlice> names;
    for (auto valueCategory : valueCategories)
    {
        const auto categoryIndex =
            cmdOptions.findCategoryByUserValue(CommandOptions::UserValue(valueCategory));
        SLANG_ASSERT(categoryIndex >= 0);
        if (categoryIndex < 0)
        {
            return SLANG_FAIL;
        }
        cmdOptions.appendCategoryOptionNames(categoryIndex, names);
    }

    StringBuilder buf;
    StringUtil::join(names.getBuffer(), names.getCount(), toSlice(", "), buf);

    m_sink->diagnose(arg.loc, Diagnostics::unknownCommandLineValue, buf);
    return SLANG_FAIL;
}

SlangResult OptionsParser::_expectValue(
    ValueCategory valueCategory,
    CommandOptions::UserValue& outValue)
{
    CommandLineArg arg;
    SLANG_RETURN_ON_FAIL(m_reader.expectArg(arg));
    SLANG_RETURN_ON_FAIL(_getValue(valueCategory, arg, outValue));
    return SLANG_OK;
}

SlangResult OptionsParser::_expectInt(const CommandLineArg& initArg, Int& outInt)
{
    SLANG_UNUSED(initArg);

    CommandLineArg arg;
    SLANG_RETURN_ON_FAIL(m_reader.expectArg(arg));

    if (SLANG_FAILED(StringUtil::parseInt(arg.value.getUnownedSlice(), outInt)))
    {
        m_sink->diagnose(arg.loc, Diagnostics::expectingAnInteger);
        return SLANG_FAIL;
    }
    return SLANG_OK;
}

SlangResult createArtifactFromReferencedModule(
    String path,
    SourceLoc loc,
    DiagnosticSink* sink,
    IArtifact** outArtifact)
{
    auto desc = ArtifactDescUtil::getDescFromPath(path.getUnownedSlice());

    if (desc.kind == ArtifactKind::Unknown)
    {
        sink->diagnose(loc, Diagnostics::unknownLibraryKind, Path::getPathExt(path));
        return SLANG_FAIL;
    }

    // If it's a GPU binary, then we'll assume it's a library
    if (ArtifactDescUtil::isGpuUsable(desc))
    {
        desc.kind = ArtifactKind::Library;
    }

    // If its a zip we'll *assume* its a zip holding compilation results
    if (desc.kind == ArtifactKind::Zip)
    {
        desc.payload = ArtifactPayload::CompileResults;
    }

    if (!ArtifactDescUtil::isLinkable(desc))
    {
        sink->diagnose(loc, Diagnostics::kindNotLinkable, Path::getPathExt(path));
        return SLANG_FAIL;
    }

    const String name = ArtifactDescUtil::getBaseNameFromPath(desc, path.getUnownedSlice());

    // Create the artifact
    auto artifact = Artifact::create(desc, name.getUnownedSlice());

    // There is a problem here if I want to reference a library that is a 'system' library or is not
    // directly a file In that case the path shouldn't be set and the name should completely define
    // the library. Seeing as on all targets the baseName doesn't have an extension, and all library
    // types do if the name doesn't have an extension we can assume there is no path to it.

    ComPtr<IOSFileArtifactRepresentation> fileRep;
    if (Path::getPathExt(path).getLength() <= 0)
    {
        // If there is no extension *assume* it is the name of a system level library
        fileRep = new OSFileArtifactRepresentation(
            IOSFileArtifactRepresentation::Kind::NameOnly,
            path.getUnownedSlice(),
            nullptr);
    }
    else
    {
        fileRep = new OSFileArtifactRepresentation(
            IOSFileArtifactRepresentation::Kind::Reference,
            path.getUnownedSlice(),
            nullptr);
        if (!fileRep->exists())
        {
            sink->diagnose(loc, Diagnostics::libraryDoesNotExist, path);
            return SLANG_FAIL;
        }
    }
    artifact->addRepresentation(fileRep);
    *outArtifact = artifact.detach();
    return SLANG_OK;
}

SlangResult OptionsParser::addReferencedModule(String path, SourceLoc loc, bool includeEntryPoint)
{
    ComPtr<IArtifact> artifact;
    SLANG_RETURN_ON_FAIL(
        createArtifactFromReferencedModule(path, loc, m_sink, artifact.writeRef()));

    SLANG_RETURN_ON_FAIL(_addLibraryReference(m_requestImpl, path, artifact, includeEntryPoint));
    for (Index i = m_rawTranslationUnits.getCount(); i < m_requestImpl->getTranslationUnitCount();
         i++)
    {
        RawTranslationUnit rawTU;
        rawTU.translationUnitID = (int)i;
        rawTU.impliedStage = Stage::Unknown;
        rawTU.sourceLanguage = SLANG_SOURCE_LANGUAGE_SLANG;
        m_rawTranslationUnits.add(rawTU);
    }
    m_currentTranslationUnitIndex = m_requestImpl->getTranslationUnitCount() - 1;
    m_slangTranslationUnitIndex = m_currentTranslationUnitIndex;
    return SLANG_OK;
}

SlangResult OptionsParser::_parseReferenceModule(const CommandLineArg& arg)
{
    SLANG_UNUSED(arg);

    CommandLineArg referenceModuleName;
    SLANG_RETURN_ON_FAIL(m_reader.expectArg(referenceModuleName));

    // Add the module to the request
    SLANG_RETURN_ON_FAIL(
        addReferencedModule(referenceModuleName.value, referenceModuleName.loc, true));

    // In addition to adding the module to the request, we also add to the options set, because
    // the same options parser is also used for IGlobalSession::parseCommandLineArguments, which
    // parses options via a dummy request that is destroyed once the command line options are
    // obtained. Therefore, also add the option here so that
    // IGlobalSession::parseCommandLineArguments can return them.
    m_requestImpl->getLinkage()->m_optionSet.add(
        CompilerOptionName::ReferenceModule,
        referenceModuleName.value);

    return SLANG_OK;
}

SlangResult OptionsParser::_parseReproFileSystem(const CommandLineArg& arg)
{
    SLANG_UNUSED(arg);

    CommandLineArg reproName;
    SLANG_RETURN_ON_FAIL(m_reader.expectArg(reproName));

    List<uint8_t> buffer;
    {
        const Result res = ReproUtil::loadState(reproName.value, m_sink, buffer);
        if (SLANG_FAILED(res))
        {
            m_sink->diagnose(reproName.loc, Diagnostics::unableToReadFile, reproName.value);
            return res;
        }
    }

    auto requestState = ReproUtil::getRequest(buffer);
    MemoryOffsetBase base;
    base.set(buffer.getBuffer(), buffer.getCount());

    // If we can find a directory, that exists, we will set up a file system to load from that
    // directory
    ComPtr<ISlangFileSystem> dirFileSystem;
    String dirPath;
    if (SLANG_SUCCEEDED(ReproUtil::calcDirectoryPathFromFilename(reproName.value, dirPath)))
    {
        SlangPathType pathType;
        if (SLANG_SUCCEEDED(Path::getPathType(dirPath, &pathType)) &&
            pathType == SLANG_PATH_TYPE_DIRECTORY)
        {
            dirFileSystem = new RelativeFileSystem(OSFileSystem::getExtSingleton(), dirPath, true);
        }
    }

    ComPtr<ISlangFileSystemExt> fileSystem;
    SLANG_RETURN_ON_FAIL(ReproUtil::loadFileSystem(base, requestState, dirFileSystem, fileSystem));

    auto cacheFileSystem = as<CacheFileSystem>(fileSystem);
    SLANG_ASSERT(cacheFileSystem);

    // I might want to make the dir file system the fallback file system...
    cacheFileSystem->setInnerFileSystem(
        dirFileSystem,
        cacheFileSystem->getUniqueIdentityMode(),
        cacheFileSystem->getPathStyle());

    // Set as the file system
    m_compileRequest->setFileSystem(fileSystem);
    return SLANG_OK;
}

SlangResult OptionsParser::_parseHelp(const CommandLineArg& arg)
{
    SLANG_UNUSED(arg);

    Index categoryIndex = -1;

    if (m_reader.hasArg())
    {
        auto catArg = m_reader.getArgAndAdvance();

        categoryIndex =
            m_cmdOptions->findCategoryByCaseInsensitiveName(catArg.value.getUnownedSlice());
        if (categoryIndex < 0)
        {
            m_sink->diagnose(catArg.loc, Diagnostics::unknownHelpCategory);
            return SLANG_FAIL;
        }
    }

    CommandOptionsWriter::Options writerOptions;
    writerOptions.style = m_helpStyle;

    auto writer = CommandOptionsWriter::create(writerOptions);

    auto& buf = writer->getBuilder();

    if (categoryIndex < 0)
    {
        // If it's the text style we can inject usage at the top
        if (m_helpStyle == CommandOptionsWriter::Style::Text)
        {
            _appendUsageTitle(buf);
        }
        else
        {
            // NOTE! We need this preamble because if we have links,
            // we have to make sure the first thing in markdown *isn't* <>

            buf << "# Slang Command Line Options\n\n";
            buf << "*Usage:*\n";
            buf << "```\n";
            buf << "slangc [options...] [--] <input files>\n\n";
            buf << "# For help\n";
            buf << "slangc -h\n\n";
            buf << "# To generate this file\n";
            buf << "slangc -help-style markdown -h\n";
            buf << "```\n";
        }

        writer->appendDescription(m_cmdOptions);
    }
    else
    {
        writer->appendDescriptionForCategory(m_cmdOptions, categoryIndex);
    }

    m_sink->diagnoseRaw(Severity::Note, buf.getBuffer());

    return SLANG_OK;
}

SlangResult OptionsParser::_parseLoadRepro(const CommandLineArg& arg)
{
    SLANG_UNUSED(arg);

    CommandLineArg reproName;
    SLANG_RETURN_ON_FAIL(m_reader.expectArg(reproName));

    if (SLANG_FAILED(_loadRepro(reproName.value, m_sink, m_requestImpl)))
    {
        m_sink->diagnose(reproName.loc, Diagnostics::unableToReadFile, reproName.value);
        return SLANG_FAIL;
    }

    m_hasLoadedRepro = true;
    return SLANG_OK;
}

SlangResult OptionsParser::_parseDebugInformation(const CommandLineArg& arg)
{
    auto name = arg.value.getUnownedSlice().tail(2);

    // Note: unlike with `-O` above, we have to consider that other
    // options might have names that start with `-g` and so cannot
    // just detect it as a prefix.
    if (name.getLength() == 0)
    {
        // The default is standard
        m_compileRequest->setDebugInfoLevel(SLANG_DEBUG_INFO_LEVEL_STANDARD);
    }
    else
    {
        CommandOptions::UserValue value;
        ValueCategory valueCat;
        ValueCategory valueCats[] = {ValueCategory::DebugLevel, ValueCategory::DebugInfoFormat};
        SLANG_RETURN_ON_FAIL(_getValue(makeConstArrayView(valueCats), arg, name, valueCat, value));

        if (valueCat == ValueCategory::DebugLevel)
        {
            const auto level = (SlangDebugInfoLevel)value;
            m_compileRequest->setDebugInfoLevel(level);
        }
        else
        {
            const auto debugFormat = (SlangDebugInfoFormat)value;
            m_compileRequest->setDebugInfoFormat(debugFormat);
        }
    }
    return SLANG_OK;
}


SlangResult OptionsParser::_parseProfile(const CommandLineArg& arg)
{
    SLANG_UNUSED(arg);

    // A "profile" can specify both a general capability level for
    // a target, and also (as a legacy/compatibility feature) a
    // specific stage to use for an entry point.

    CommandLineArg operand;
    SLANG_RETURN_ON_FAIL(m_reader.expectArg(operand));

    // A a convenience, the `-profile` option supports an operand that consists
    // of multiple tokens separated with `+`. The eventual goal is that each
    // of these tokens will represent a capability that should be assumed to
    // be present on the target.
    //
    List<UnownedStringSlice> slices;
    StringUtil::split(operand.value.getUnownedSlice(), '+', slices);
    Index sliceCount = slices.getCount();

    // For now, we will require that the *first* capability in the list is
    // special, and represents the traditional `Profile` to compile for in
    // the existing Slang model.
    //
    UnownedStringSlice profileName = sliceCount >= 1 ? slices[0] : UnownedTerminatedStringSlice("");

    SlangProfileID profileID = SlangProfileID(Slang::Profile::lookUp(profileName).raw);
    if (profileID == SLANG_PROFILE_UNKNOWN)
    {
        m_sink->diagnose(operand.loc, Diagnostics::unknownProfile, profileName);
        return SLANG_FAIL;
    }
    else
    {
        auto profile = Profile(profileID);

        setProfile(this->getCurrentTarget(), profile);

        auto stage = profile.getStage();
        if (stage != Stage::Unknown)
        {
            setStage(getCurrentEntryPoint(), stage);
        }
    }

    // Any additional capability tokens will be assumed to represent `CapabilityAtom`s.
    // Those atoms will need to be added to the supported capabilities of the target.
    //
    for (Index i = 1; i < sliceCount; ++i)
    {
        UnownedStringSlice atomName = slices[i];
        CapabilityName atom = findCapabilityName(atomName);
        if (atom == CapabilityName::Invalid)
        {
            m_sink->diagnose(operand.loc, Diagnostics::unknownProfile, atomName);
            return SLANG_FAIL;
        }

        addCapabilityAtom(getCurrentTarget(), atom);
    }

    return SLANG_OK;
}

SlangResult OptionsParser::_parse(int argc, char const* const* argv)
{
    // Set up the args
    CommandLineArgs args(m_cmdLineContext);

    // Converts input args into args in 'args'.
    // Doing so will allocate some SourceLoc space from the CommandLineContext.
    args.setArgs(argv, argc);

    auto linkage = m_requestImpl->getLinkage();

    // Before we do anything else lets strip out all of the downstream arguments.
    DownstreamArgs downstreamArgs(m_cmdLineContext);


    SLANG_RETURN_ON_FAIL(downstreamArgs.stripDownstreamArgs(args, 0, m_sink));
    for (auto& entry : downstreamArgs.m_entries)
    {
        String serializedArgs = entry.args.serialize();
        CompilerOptionValue v;
        v.kind = CompilerOptionValueKind::String;
        v.stringValue = entry.name;
        v.stringValue2 = serializedArgs;
        linkage->m_optionSet.add(CompilerOptionName::DownstreamArgs, v);
    }

    m_reader.init(&args, m_sink);

    while (m_reader.hasArg())
    {
        auto arg = m_reader.getArgAndAdvance();
        const auto& argValue = arg.value;

        // If it's not an option we assume it's a path
        if (argValue[0] != '-')
        {
            SLANG_RETURN_ON_FAIL(addInputPath(argValue.getBuffer()));
            continue;
        }

        const Index optionIndex = m_cmdOptions->findOptionByName(argValue.getUnownedSlice());

        if (optionIndex < 0)
        {
            m_sink->diagnose(arg.loc, Diagnostics::unknownCommandLineOption, argValue);
            _outputMinimalUsage();
            return SLANG_FAIL;
        }

        const auto optionKind = OptionKind(m_cmdOptions->getOptionAt(optionIndex).userValue);

        switch (optionKind)
        {
        case OptionKind::NoMangle:
        case OptionKind::ValidateUniformity:
        case OptionKind::AllowGLSL:
        case OptionKind::EnableExperimentalPasses:
        case OptionKind::EmitIr:
        case OptionKind::DumpIntermediates:
        case OptionKind::DumpReproOnError:
        case OptionKind::ReportDownstreamTime:
        case OptionKind::ReportPerfBenchmark:
        case OptionKind::ReportCheckpointIntermediates:
        case OptionKind::SkipSPIRVValidation:
        case OptionKind::DisableSpecialization:
        case OptionKind::DisableDynamicDispatch:
        case OptionKind::TrackLiveness:
        case OptionKind::SkipCodeGen:
        case OptionKind::ParameterBlocksUseRegisterSpaces:
        case OptionKind::ValidateIr:
        case OptionKind::DumpIr:
        case OptionKind::VulkanInvertY:
        case OptionKind::VulkanUseDxPositionW:
        case OptionKind::VulkanUseEntryPointName:
        case OptionKind::VulkanUseGLLayout:
        case OptionKind::VulkanEmitReflection:
        case OptionKind::IgnoreCapabilities:
        case OptionKind::RestrictiveCapabilityCheck:
        case OptionKind::MinimumSlangOptimization:
        case OptionKind::DisableNonEssentialValidations:
        case OptionKind::DisableSourceMap:
        case OptionKind::DefaultImageFormatUnknown:
        case OptionKind::Obfuscate:
        case OptionKind::OutputIncludes:
        case OptionKind::PreprocessorOutput:
        case OptionKind::DumpAst:
        case OptionKind::IncompleteLibrary:
        case OptionKind::NoHLSLBinding:
        case OptionKind::NoHLSLPackConstantBufferElements:
        case OptionKind::LoopInversion:
        case OptionKind::UnscopedEnum:
        case OptionKind::PreserveParameters:
            linkage->m_optionSet.set(optionKind, true);
            break;
        case OptionKind::MatrixLayoutRow:
        case OptionKind::MatrixLayoutColumn:
            linkage->m_optionSet.setMatrixLayoutMode(
                (optionKind == OptionKind::MatrixLayoutRow)
                    ? MatrixLayoutMode::kMatrixLayoutMode_RowMajor
                    : MatrixLayoutMode::kMatrixLayoutMode_ColumnMajor);
            break;
        case OptionKind::NoCodeGen:
            linkage->m_optionSet.set(OptionKind::SkipCodeGen, true);
            break;
            break;
        case OptionKind::LoadCoreModule:
            {
                CommandLineArg fileName;
                SLANG_RETURN_ON_FAIL(m_reader.expectArg(fileName));

                // Load the file
                ScopedAllocation contents;
                SLANG_RETURN_ON_FAIL(File::readAllBytes(fileName.value, contents));
                SLANG_RETURN_ON_FAIL(
                    m_session->loadCoreModule(contents.getData(), contents.getSizeInBytes()));

                // Ensure that the linkage's AST builder is up-to-date.
                linkage->getASTBuilder()->m_cachedNodes =
                    asInternal(m_session)->getGlobalASTBuilder()->m_cachedNodes;

                break;
            }
        case OptionKind::CompileCoreModule:
            m_compileCoreModule = true;
            break;
        case OptionKind::ArchiveType:
            {
                SLANG_RETURN_ON_FAIL(_expectValue(m_archiveType));
                break;
            }
        case OptionKind::SaveCoreModule:
            {
                CommandLineArg fileName;
                SLANG_RETURN_ON_FAIL(m_reader.expectArg(fileName));

                ComPtr<ISlangBlob> blob;

                SLANG_RETURN_ON_FAIL(m_session->saveCoreModule(m_archiveType, blob.writeRef()));
                SLANG_RETURN_ON_FAIL(File::writeAllBytes(
                    fileName.value,
                    blob->getBufferPointer(),
                    blob->getBufferSize()));
                break;
            }
        case OptionKind::SaveCoreModuleBinSource:
        case OptionKind::SaveGLSLModuleBinSource:
            {
                CommandLineArg fileName;
                SLANG_RETURN_ON_FAIL(m_reader.expectArg(fileName));

                ComPtr<ISlangBlob> blob;

                if (optionKind == OptionKind::SaveCoreModuleBinSource)
                {
                    SLANG_RETURN_ON_FAIL(m_session->saveCoreModule(m_archiveType, blob.writeRef()));
                }
                else
                {
                    SLANG_RETURN_ON_FAIL(m_session->saveBuiltinModule(
                        slang::BuiltinModuleName::GLSL,
                        m_archiveType,
                        blob.writeRef()));
                }
                StringBuilder builder;
                StringWriter writer(&builder, 0);

                SLANG_RETURN_ON_FAIL(HexDumpUtil::dumpSourceBytes(
                    (const uint8_t*)blob->getBufferPointer(),
                    blob->getBufferSize(),
                    16,
                    &writer));

                File::writeNativeText(fileName.value, builder.getBuffer(), builder.getLength());
                break;
            }
        case OptionKind::DumpIrIds:
            {
                m_frontEndReq->m_irDumpOptions.flags |= IRDumpOptions::Flag::DumpDebugIds;
                break;
            }
        case OptionKind::DumpIntermediatePrefix:
            {
                CommandLineArg prefix;
                SLANG_RETURN_ON_FAIL(m_reader.expectArg(prefix));
                linkage->m_optionSet.set(CompilerOptionName::DumpIntermediatePrefix, prefix.value);
                break;
            }
        case OptionKind::Doc:
            {
                // When compiling the core module, it will write out a documentation.
                m_compileCoreModuleFlags |= slang::CompileCoreModuleFlag::WriteDocumentation;

                // Enable writing out documentation on the req
                linkage->m_optionSet.set(CompilerOptionName::Doc, true);
                break;
            }
        case OptionKind::DumpRepro:
            {
                CommandLineArg dumpRepro;
                SLANG_RETURN_ON_FAIL(m_reader.expectArg(dumpRepro));
                linkage->m_optionSet.set(OptionKind::DumpRepro, dumpRepro.value);
                m_compileRequest->enableReproCapture();
                break;
            }
        case OptionKind::ExtractRepro:
            {
                CommandLineArg reproName;
                SLANG_RETURN_ON_FAIL(m_reader.expectArg(reproName));

                {
                    const Result res = ReproUtil::extractFilesToDirectory(reproName.value, m_sink);
                    if (SLANG_FAILED(res))
                    {
                        m_sink->diagnose(
                            reproName.loc,
                            Diagnostics::unableExtractReproToDirectory,
                            reproName.value);
                        return res;
                    }
                }
                break;
            }
        case OptionKind::ModuleName:
            {
                CommandLineArg moduleName;
                SLANG_RETURN_ON_FAIL(m_reader.expectArg(moduleName));

                m_compileRequest->setDefaultModuleName(moduleName.value.getBuffer());
                break;
            }
        case OptionKind::LoadRepro:
            SLANG_RETURN_ON_FAIL(_parseLoadRepro(arg));
            break;
        case OptionKind::LoadReproDirectory:
            {
                CommandLineArg reproDirectory;
                SLANG_RETURN_ON_FAIL(m_reader.expectArg(reproDirectory));

                SLANG_RETURN_ON_FAIL(
                    _compileReproDirectory(m_session, m_requestImpl, reproDirectory.value));
                break;
            }
        case OptionKind::ReproFallbackDirectory:
            {
                CommandLineArg reproDirectory;
                SLANG_RETURN_ON_FAIL(m_reader.expectArg(reproDirectory));

                if (reproDirectory.value == toSlice("default:"))
                {
                    // The default is to use the OS file system
                    m_requestImpl->m_reproFallbackFileSystem = OSFileSystem::getExtSingleton();
                }
                else if (reproDirectory.value == toSlice("none:"))
                {
                    // None, means that there isn't a fallback
                    m_requestImpl->m_reproFallbackFileSystem.setNull();
                }
                else
                {
                    auto osFileSystem = OSFileSystem::getExtSingleton();

                    SlangPathType pathType;
                    if (SLANG_FAILED(osFileSystem->getPathType(
                            reproDirectory.value.getBuffer(),
                            &pathType)) ||
                        pathType != SLANG_PATH_TYPE_DIRECTORY)
                    {
                        return SLANG_FAIL;
                    }
                    // Make the fallback directory use a relative file system, to the specified
                    // directory
                    m_requestImpl->m_reproFallbackFileSystem =
                        new RelativeFileSystem(osFileSystem, reproDirectory.value);
                }
                break;
            }
        case OptionKind::ReproFileSystem:
            SLANG_RETURN_ON_FAIL(_parseReproFileSystem(arg));
            break;
        case OptionKind::SerialIr:
            m_frontEndReq->useSerialIRBottleneck = true;
            break;
        case OptionKind::VerbosePaths:
            m_requestImpl->getSink()->setFlag(DiagnosticSink::Flag::VerbosePath);
            break;
        case OptionKind::DumpWarningDiagnostics:
            _dumpDiagnostics(Severity::Warning);
            break;
        case OptionKind::WarningsAsErrors:
            {
                CommandLineArg operand;
                SLANG_RETURN_ON_FAIL(m_reader.expectArg(operand));
                linkage->m_optionSet.add(
                    OptionKind::WarningsAsErrors,
                    operand.value.getUnownedSlice());
                break;
            }
        case OptionKind::DisableWarnings:
            {
                CommandLineArg operand;
                SLANG_RETURN_ON_FAIL(m_reader.expectArg(operand));
                // SLANG_RETURN_ON_FAIL(_overrideDiagnostics(operand.value.getUnownedSlice(),
                // Severity::Warning, Severity::Disable));
                linkage->m_optionSet.add(
                    OptionKind::DisableWarnings,
                    operand.value.getUnownedSlice());
                break;
            }
        case OptionKind::DisableWarning:
            {
                // 5 because -Wno-
                auto name = argValue.getUnownedSlice().tail(5);
                linkage->m_optionSet.add(OptionKind::DisableWarning, name);

                // SLANG_RETURN_ON_FAIL(_overrideDiagnostic(name, Severity::Warning,
                // Severity::Disable));
                break;
            }
        case OptionKind::EnableWarning:
            {
                // 2 because -W
                auto name = argValue.getUnownedSlice().tail(2);
                linkage->m_optionSet.add(OptionKind::EnableWarning, name);
                // Enable the warning
                // SLANG_RETURN_ON_FAIL(_overrideDiagnostic(name, Severity::Warning,
                // Severity::Warning));
                break;
            }
        case OptionKind::VerifyDebugSerialIr:
            m_frontEndReq->verifyDebugSerialization = true;
            break;
        case OptionKind::IrCompression:
            {
                CommandLineArg name;
                SLANG_RETURN_ON_FAIL(m_reader.expectArg(name));
                // TODO: warn that this option is deprecated
                break;
            }
        case OptionKind::EmbedDownstreamIR:
            {
                getCurrentTarget()->optionSet.add(CompilerOptionName::EmbedDownstreamIR, true);
                break;
            }
        case OptionKind::Target:
            {
                CommandLineArg name;
                SLANG_RETURN_ON_FAIL(m_reader.expectArg(name));

                const CodeGenTarget format = (CodeGenTarget)TypeTextUtil::findCompileTargetFromName(
                    name.value.getUnownedSlice());

                if (format == CodeGenTarget::Unknown)
                {
                    m_sink->diagnose(
                        name.loc,
                        Diagnostics::unknownCodeGenerationTarget,
                        name.value);
                    return SLANG_FAIL;
                }

                RawTarget rawTarget;
                rawTarget.format = CodeGenTarget(format);
                // Silently allow redundant targets if it is the same as the last specified target.
                if (m_rawTargets.getCount() != 0 &&
                    m_rawTargets.getLast().format == rawTarget.format)
                    break;
                m_rawTargets.add(rawTarget);
                break;
            }
        case OptionKind::VulkanBindShift:
            {
                // -fvk-{b|s|t|u}-shift <binding-shift> <set>
                const auto slice = arg.value.getUnownedSlice().subString(5, 1);
                HLSLToVulkanLayoutOptions::Kind kind;
                SLANG_RETURN_ON_FAIL(_getValue(arg, slice, kind));

                Int shift;
                SLANG_RETURN_ON_FAIL(_expectInt(arg, shift));

                if (m_reader.hasArg() && m_reader.peekArg().value == toSlice("all"))
                {
                    m_reader.advance();
                    linkage->m_optionSet.add(
                        CompilerOptionName::VulkanBindShiftAll,
                        (int)kind,
                        (int)shift);
                }
                else
                {
                    Int set;
                    SLANG_RETURN_ON_FAIL(_expectInt(arg, set));
                    linkage->m_optionSet.add(
                        CompilerOptionName::VulkanBindShift,
                        (uint8_t)kind,
                        (int)set,
                        (int)shift);
                }
                break;
            }
        case OptionKind::VulkanBindGlobals:
            {
                // -fvk-bind-globals <index> <set>
                Int binding, bindingSet;
                SLANG_RETURN_ON_FAIL(_expectInt(arg, binding));
                SLANG_RETURN_ON_FAIL(_expectInt(arg, bindingSet));
                linkage->m_optionSet.set(
                    OptionKind::VulkanBindGlobals,
                    (int)binding,
                    (int)bindingSet);
                break;
            }
        case OptionKind::Profile:
            SLANG_RETURN_ON_FAIL(_parseProfile(arg));
            break;
        case OptionKind::Capability:
            {
                // The `-capability` option is similar to `-profile` but does not set the actual
                // profile for a target (it just adds capabilities).
                //
                // TODO: Once profiles are treated as capabilities themselves, it might be possible
                // to treat `-profile` and `-capability` as aliases, although there might still be
                // value in only allowing a single `-profile` option per target while still allowing
                // zero or more `-capability` options.

                // Don't treat zero args as an error.
                if (!m_reader.hasArg())
                    break;

                CommandLineArg operand;
                SLANG_RETURN_ON_FAIL(m_reader.expectArg(operand));

                List<UnownedStringSlice> slices;
                StringUtil::split(operand.value.getUnownedSlice(), '+', slices);
                Index sliceCount = slices.getCount();
                for (Index i = 0; i < sliceCount; ++i)
                {
                    UnownedStringSlice atomName = slices[i];
                    CapabilityName atom = findCapabilityName(atomName);
                    if (atom == CapabilityName::Invalid)
                    {
                        m_sink->diagnose(operand.loc, Diagnostics::unknownProfile, atomName);
                        return SLANG_FAIL;
                    }

                    addCapabilityAtom(getCurrentTarget(), atom);
                }
                break;
            }
        case OptionKind::Stage:
            {
                CommandLineArg name;
                SLANG_RETURN_ON_FAIL(m_reader.expectArg(name));

                Stage stage = findStageByName(name.value);
                if (stage == Stage::Unknown)
                {
                    m_sink->diagnose(name.loc, Diagnostics::unknownStage, name.value);
                    return SLANG_FAIL;
                }
                else
                {
                    setStage(getCurrentEntryPoint(), stage);
                }
                break;
            }
        case OptionKind::GLSLForceScalarLayout:
            {
                getCurrentTarget()->optionSet.add(CompilerOptionName::GLSLForceScalarLayout, true);
                break;
            }
        case OptionKind::ForceDXLayout:
            {
                getCurrentTarget()->optionSet.add(CompilerOptionName::ForceDXLayout, true);
                break;
            }
        case OptionKind::EnableEffectAnnotations:
            {
                m_compileRequest->setEnableEffectAnnotations(true);
                break;
            }

        case OptionKind::EntryPointName:
            {
                CommandLineArg name;
                SLANG_RETURN_ON_FAIL(m_reader.expectArg(name));

                RawEntryPoint rawEntryPoint;
                rawEntryPoint.name = name.value;
                rawEntryPoint.translationUnitIndex = m_currentTranslationUnitIndex;
                // Silently allow duplicate entrypoints if it is the same as the last specified one.
                if (m_rawEntryPoints.getCount() != 0 &&
                    m_rawEntryPoints.getLast().name == rawEntryPoint.name)
                    break;
                m_rawEntryPoints.add(rawEntryPoint);
                break;
            }
        case OptionKind::Specialize:
            {
                for (;;)
                {
                    CommandLineArg name;
                    SLANG_RETURN_ON_FAIL(m_reader.expectArg(name));
                    if (m_rawEntryPoints.getCount() > 0)
                    {
                        auto& lastEntryPoint = m_rawEntryPoints.getLast();
                        lastEntryPoint.specializationArgs.add(name.value);
                    }
                    if (m_reader.hasArg() && m_reader.peekArg().value == ",")
                        m_reader.advance();
                    else
                        break;
                }
                break;
            }
        case OptionKind::Language:
            {
                CommandLineArg name;
                SLANG_RETURN_ON_FAIL(m_reader.expectArg(name));

                const SourceLanguage sourceLanguage =
                    (SourceLanguage)TypeTextUtil::findSourceLanguage(name.value.getUnownedSlice());

                if (sourceLanguage == SourceLanguage::Unknown)
                {
                    m_sink->diagnose(name.loc, Diagnostics::unknownSourceLanguage, name.value);
                    return SLANG_FAIL;
                }
                else
                {
                    while (m_reader.hasArg() && !m_reader.peekValue().startsWith("-"))
                    {
                        SLANG_RETURN_ON_FAIL(addInputPath(
                            m_reader.getValueAndAdvance().getBuffer(),
                            sourceLanguage));
                    }
                }
                linkage->m_optionSet.add(CompilerOptionName::Language, (int)sourceLanguage);
                break;
            }
        case OptionKind::PassThrough:
            {
                CommandLineArg name;
                SLANG_RETURN_ON_FAIL(m_reader.expectArg(name));

                SlangPassThrough passThrough = SLANG_PASS_THROUGH_NONE;
                if (SLANG_FAILED(
                        TypeTextUtil::findPassThrough(name.value.getUnownedSlice(), passThrough)))
                {
                    m_sink->diagnose(name.loc, Diagnostics::unknownPassThroughTarget, name.value);
                    return SLANG_FAIL;
                }

                m_compileRequest->setPassThrough(passThrough);
                break;
            }
        case OptionKind::MacroDefine:
            {
                // The value to be defined might be part of the same option, as in:
                //     -DFOO
                // or it might come separately, as in:
                //     -D FOO

                UnownedStringSlice slice = argValue.getUnownedSlice().tail(2);

                CommandLineArg nextArg;
                if (slice.getLength() <= 0)
                {
                    SLANG_RETURN_ON_FAIL(m_reader.expectArg(nextArg));
                    slice = nextArg.value.getUnownedSlice();
                }

                // The string that sets up the define can have an `=` between
                // the name to be defined and its value, so we search for one.
                const Index equalIndex = slice.indexOf('=');

                // Now set the preprocessor define

                if (equalIndex >= 0)
                {
                    // If we found an `=`, we split the string...
                    m_compileRequest->addPreprocessorDefine(
                        String(slice.head(equalIndex)).getBuffer(),
                        String(slice.tail(equalIndex + 1)).getBuffer());
                }
                else
                {
                    // If there was no `=`, then just #define it to an empty string
                    m_compileRequest->addPreprocessorDefine(String(slice).getBuffer(), "");
                }
                break;
            }
        case OptionKind::Include:
            {
                // The value to be defined might be part of the same option, as in:
                //     -IFOO
                // or it might come separately, as in:
                //     -I FOO
                // (see handling of `-D` above)
                UnownedStringSlice slice = argValue.getUnownedSlice().tail(2);

                CommandLineArg nextArg;
                if (slice.getLength() <= 0)
                {
                    // Need to read another argument from the command line
                    SLANG_RETURN_ON_FAIL(m_reader.expectArg(nextArg));
                    slice = nextArg.value.getUnownedSlice();
                }

                m_compileRequest->addSearchPath(String(slice).getBuffer());
                break;
            }
        case OptionKind::Output:
            {
                //
                // A `-o` option is used to specify a desired output file.
                CommandLineArg outputPath;
                SLANG_RETURN_ON_FAIL(m_reader.expectArg(outputPath));

                addOutputPath(outputPath.value.getBuffer());
                break;
            }
        case OptionKind::EmitReflectionJSON:
            {
                CommandLineArg outputPath;
                SLANG_RETURN_ON_FAIL(m_reader.expectArg(outputPath));

                linkage->m_optionSet.set(CompilerOptionName::EmitReflectionJSON, outputPath.value);
                break;
            }
        case OptionKind::DepFile:
            {
                CommandLineArg dependencyPath;
                SLANG_RETURN_ON_FAIL(m_reader.expectArg(dependencyPath));

                if (m_requestImpl->m_dependencyOutputPath.getLength() == 0)
                {
                    m_requestImpl->m_dependencyOutputPath = dependencyPath.value;
                }
                else
                {
                    m_sink->diagnose(
                        dependencyPath.loc,
                        Diagnostics::duplicateDependencyOutputPaths);
                    return SLANG_FAIL;
                }
                break;
            }
        case OptionKind::LineDirectiveMode:
            {
                SlangLineDirectiveMode value;
                SLANG_RETURN_ON_FAIL(_expectValue(value));
                m_compileRequest->setLineDirectiveMode(value);
                break;
            }
        case OptionKind::FloatingPointMode:
            {
                FloatingPointMode value;
                SLANG_RETURN_ON_FAIL(_expectValue(value));
                setFloatingPointMode(getCurrentTarget(), value);
                break;
            }
        case OptionKind::Optimization:
            {
                UnownedStringSlice levelSlice = argValue.getUnownedSlice().tail(2);
                SlangOptimizationLevel level = SLANG_OPTIMIZATION_LEVEL_DEFAULT;

                if (levelSlice.getLength())
                {
                    SLANG_RETURN_ON_FAIL(_getValue(arg, levelSlice, level));
                }

                m_compileRequest->setOptimizationLevel(level);
                break;
            }
        case OptionKind::DebugInformation:
            SLANG_RETURN_ON_FAIL(_parseDebugInformation(arg));
            break;
        case OptionKind::FileSystem:
            {
                typedef TypeTextUtil::FileSystemType FileSystemType;
                FileSystemType value;
                SLANG_RETURN_ON_FAIL(_expectValue(value));

                switch (value)
                {
                case FileSystemType::Default:
                    m_compileRequest->setFileSystem(nullptr);
                    break;
                case FileSystemType::LoadFile:
                    m_compileRequest->setFileSystem(OSFileSystem::getLoadSingleton());
                    break;
                case FileSystemType::Os:
                    m_compileRequest->setFileSystem(OSFileSystem::getExtSingleton());
                    break;
                }
                break;
            }
        case OptionKind::ReferenceModule:
            SLANG_RETURN_ON_FAIL(_parseReferenceModule(arg));
            break;
        case OptionKind::Version:
            {
                m_sink->diagnoseRaw(Severity::Note, m_session->getBuildTagString());
                break;
            }
        case OptionKind::HelpStyle:
            SLANG_RETURN_ON_FAIL(_expectValue(m_helpStyle));
            break;
        case OptionKind::Help:
            {
                SLANG_RETURN_ON_FAIL(_parseHelp(arg));
                return SLANG_OK;
            }
        case OptionKind::EmitSpirvViaGLSL:
        case OptionKind::EmitSpirvDirectly:
            {
                SlangEmitSpirvMethod selectMethod = (optionKind == OptionKind::EmitSpirvViaGLSL)
                                                        ? SLANG_EMIT_SPIRV_VIA_GLSL
                                                        : SLANG_EMIT_SPIRV_DIRECTLY;

                SlangEmitSpirvMethod currentMethod =
                    getCurrentTarget()->optionSet.getEnumOption<SlangEmitSpirvMethod>(
                        OptionKind::EmitSpirvMethod);
                // When both flag turns on, spirv-direcly mode will always take higher priority.
                // By default (value 0), spirv-via-glsl mode is used, and any input flag can
                // override the default value.
                if (selectMethod > currentMethod)
                {
                    getCurrentTarget()->optionSet.set(OptionKind::EmitSpirvMethod, selectMethod);
                }
            }
            break;
        case OptionKind::SPIRVCoreGrammarJSON:
            {
                CommandLineArg path;
                SLANG_RETURN_ON_FAIL(m_reader.expectArg(path));
                m_session->setSPIRVCoreGrammar(path.value.getBuffer());
            }
            break;

        case OptionKind::DefaultDownstreamCompiler:
            {
                CommandLineArg sourceLanguageArg, compilerArg;
                SLANG_RETURN_ON_FAIL(m_reader.expectArg(sourceLanguageArg));
                SLANG_RETURN_ON_FAIL(m_reader.expectArg(compilerArg));

                SlangSourceLanguage sourceLanguage =
                    TypeTextUtil::findSourceLanguage(sourceLanguageArg.value.getUnownedSlice());
                if (sourceLanguage == SLANG_SOURCE_LANGUAGE_UNKNOWN)
                {
                    m_sink->diagnose(
                        sourceLanguageArg.loc,
                        Diagnostics::unknownSourceLanguage,
                        sourceLanguageArg.value);
                    return SLANG_FAIL;
                }

                SlangPassThrough compiler;
                if (SLANG_FAILED(TypeTextUtil::findPassThrough(
                        compilerArg.value.getUnownedSlice(),
                        compiler)))
                {
                    m_sink->diagnose(
                        compilerArg.loc,
                        Diagnostics::unknownPassThroughTarget,
                        compilerArg.value);
                    return SLANG_FAIL;
                }

                if (SLANG_FAILED(m_session->setDefaultDownstreamCompiler(sourceLanguage, compiler)))
                {
                    m_sink->diagnose(
                        arg.loc,
                        Diagnostics::unableToSetDefaultDownstreamCompiler,
                        compilerArg.value,
                        sourceLanguageArg.value);
                    return SLANG_FAIL;
                }
                break;
            }
        case OptionKind::CompilerPath:
            {
                const Index index = argValue.lastIndexOf('-');
                if (index >= 0)
                {
                    CommandLineArg name;
                    SLANG_RETURN_ON_FAIL(m_reader.expectArg(name));

                    UnownedStringSlice passThroughSlice =
                        argValue.getUnownedSlice().head(index).tail(1);

                    // Skip the initial -, up to the last -
                    SlangPassThrough passThrough = SLANG_PASS_THROUGH_NONE;
                    if (SLANG_SUCCEEDED(
                            TypeTextUtil::findPassThrough(passThroughSlice, passThrough)))
                    {
                        m_session->setDownstreamCompilerPath(passThrough, name.value.getBuffer());
                        continue;
                    }
                    else
                    {
                        m_sink->diagnose(
                            arg.loc,
                            Diagnostics::unknownDownstreamCompiler,
                            passThroughSlice);
                        return SLANG_FAIL;
                    }
                }
                break;
            }
        case OptionKind::InputFilesRemain:
            {
                // The `--` option causes us to stop trying to parse options,
                // and treat the rest of the command line as input file names:
                while (m_reader.hasArg())
                {
                    SLANG_RETURN_ON_FAIL(addInputPath(m_reader.getValueAndAdvance().getBuffer()));
                }
                break;
            }
        case OptionKind::SourceEmbedStyle:
            {
                SLANG_RETURN_ON_FAIL(_expectValue(m_requestImpl->m_sourceEmbedStyle));
                break;
            }
        case OptionKind::SourceEmbedName:
            {
                CommandLineArg name;
                SLANG_RETURN_ON_FAIL(m_reader.expectArg(name));
                m_requestImpl->m_sourceEmbedName = name.value;
                break;
            }
        case OptionKind::SourceEmbedLanguage:
            {
                SLANG_RETURN_ON_FAIL(_expectValue(m_requestImpl->m_sourceEmbedLanguage));

                if (!SourceEmbedUtil::isSupported(
                        (SlangSourceLanguage)m_requestImpl->m_sourceEmbedLanguage))
                {
                    m_sink->diagnose(arg.loc, Diagnostics::unhandledLanguageForSourceEmbedding);
                    return SLANG_FAIL;
                }

                break;
            }
        case OptionKind::DisableShortCircuit:
            {
                linkage->m_optionSet.add(OptionKind::DisableShortCircuit, true);
                break;
            }
        case OptionKind::BindlessSpaceIndex:
            {
                Int index = 0;
                SLANG_RETURN_ON_FAIL(_expectInt(arg, index));
                linkage->m_optionSet.add(OptionKind::BindlessSpaceIndex, (int)index);
                break;
            }
        case OptionKind::DumpModule:
            {
                CommandLineArg fileName;
                SLANG_RETURN_ON_FAIL(m_reader.expectArg(fileName));
                auto desc = slang::SessionDesc();
                ComPtr<slang::ISession> session;
                m_session->createSession(desc, session.writeRef());
                ComPtr<slang::IBlob> diagnostics;

                // Coerce Slang to load from the given file, without letting it automatically
                // choose .slang-module files over .slang files.
                // First try to load as source string, and fall back to loading as an IR Blob.
                // Avoid guessing based on filename or inspecting the file contents.
                FileStream file;
                if (SLANG_FAILED(file.init(
                        fileName.value,
                        FileMode::Open,
                        FileAccess::Read,
                        FileShare::None)))
                {
                    m_sink->diagnose(arg.loc, Diagnostics::cannotOpenFile, fileName.value);
                    return SLANG_FAIL;
                }

                List<uint8_t> buffer;
                file.seek(SeekOrigin::End, 0);
                const Int64 size = file.getPosition();
                buffer.setCount(size + 1);
                file.seek(SeekOrigin::Start, 0);
                SLANG_RETURN_ON_FAIL(file.readExactly(buffer.getBuffer(), (size_t)size));
                buffer[size] = 0;
                file.close();

                ComPtr<slang::IModule> module;
                module = session->loadModuleFromSourceString(
                    "module",
                    "path",
                    (const char*)buffer.getBuffer(),
                    diagnostics.writeRef());
                if (!module)
                {
                    // Load buffer as an IR blob
                    ComPtr<slang::IBlob> blob;
                    blob = RawBlob::create(buffer.getBuffer(), size);

                    module = session->loadModuleFromIRBlob(
                        "module",
                        "path",
                        blob,
                        diagnostics.writeRef());
                }

                if (module)
                {
                    ComPtr<slang::IBlob> disassemblyBlob;
                    if (SLANG_FAILED(module->disassemble(disassemblyBlob.writeRef())))
                    {
                        m_sink->diagnose(arg.loc, Diagnostics::cannotDisassemble, fileName.value);
                        return SLANG_FAIL;
                    }
                    else
                    {
                        // success, print out the disassembly in a way that slang-test can read
                        m_sink->diagnoseRaw(
                            Severity::Note,
                            (const char*)disassemblyBlob->getBufferPointer());
                    }
                }
                else
                {
                    if (diagnostics)
                    {
                        m_sink->diagnoseRaw(
                            Severity::Error,
                            (const char*)diagnostics->getBufferPointer());
                    }
                    return SLANG_FAIL;
                }


                break;
            }
        default:
            {
                // Hmmm, we looked up and produced a valid enum, but it wasn't handled in the
                // switch...
                m_sink->diagnose(arg.loc, Diagnostics::unknownCommandLineOption, argValue);

                _outputMinimalUsage();
                return SLANG_FAIL;
            }
        }
    }

    if (m_compileCoreModule)
    {
        SLANG_RETURN_ON_FAIL(m_session->compileCoreModule(m_compileCoreModuleFlags));
    }

    // TODO(JS): This is a restriction because of how setting of state works for load repro
    // If a repro has been loaded, then many of the following options will overwrite
    // what was set up. So for now they are ignored, and only parameters set as part
    // of the loop work if they are after -load-repro
    if (!m_hasLoadedRepro)
    {
        // As a compatability feature, if the user didn't list any explicit entry
        // point names, *and* they are compiling a single translation unit, *and* they
        // have either specified a stage, or we can assume one from the naming
        // of the translation unit, then we assume they wanted to compile a single
        // entry point named `main`.
        //
        if (m_rawEntryPoints.getCount() == 0 && m_rawTranslationUnits.getCount() == 1 &&
            (m_defaultEntryPoint.stage != Stage::Unknown ||
             m_rawTranslationUnits[0].impliedStage != Stage::Unknown))
        {
            RawEntryPoint entry;
            entry.name = "main";
            entry.translationUnitIndex = 0;
            m_rawEntryPoints.add(entry);
        }

        // If the user (manually or implicitly) specified only a single entry point,
        // then we allow the associated stage to be specified either before or after
        // the entry point. This means that if there is a stage attached
        // to the "default" entry point, we should copy it over to the
        // explicit one.
        //
        if (m_rawEntryPoints.getCount() == 1)
        {
            if (m_defaultEntryPoint.stage != Stage::Unknown)
            {
                setStage(getCurrentEntryPoint(), m_defaultEntryPoint.stage);
            }

            if (m_defaultEntryPoint.redundantStageSet)
                getCurrentEntryPoint()->redundantStageSet = true;
            if (m_defaultEntryPoint.conflictingStagesSet)
                getCurrentEntryPoint()->conflictingStagesSet = true;
        }
        else
        {
            // If the "default" entry point has had a stage (or
            // other state, if we add other per-entry-point state)
            // specified, but there is more than one entry point,
            // then that state doesn't apply to anything and we
            // should issue an error to tell the user something
            // funky is going on.
            //
            if (m_defaultEntryPoint.stage != Stage::Unknown)
            {
                if (m_rawEntryPoints.getCount() == 0)
                {
                    m_sink->diagnose(
                        SourceLoc(),
                        Diagnostics::stageSpecificationIgnoredBecauseNoEntryPoints);
                }
                else
                {
                    m_sink->diagnose(
                        SourceLoc(),
                        Diagnostics::stageSpecificationIgnoredBecauseBeforeAllEntryPoints);
                }
            }
        }

        // Slang requires that every explicit entry point indicate the translation
        // unit it comes from. If there is only one translation unit specified,
        // then implicitly all entry points come from it.
        //
        if (m_translationUnitCount == 1)
        {
            for (auto& entryPoint : m_rawEntryPoints)
            {
                entryPoint.translationUnitIndex = 0;
            }
        }
        else if (
            m_frontEndReq->additionalLoadedModules &&
            m_frontEndReq->additionalLoadedModules->getCount() == 0)
        {
            // Otherwise, we require that all entry points be specified after
            // the translation unit to which tye belong.
            bool anyEntryPointWithoutTranslationUnit = false;
            for (auto& entryPoint : m_rawEntryPoints)
            {
                // Skip entry points that are already associated with a translation unit...
                if (entryPoint.translationUnitIndex != -1)
                    continue;

                anyEntryPointWithoutTranslationUnit = true;
            }
            if (anyEntryPointWithoutTranslationUnit)
            {
                m_sink->diagnose(
                    SourceLoc(),
                    Diagnostics::entryPointsNeedToBeAssociatedWithTranslationUnits);
                return SLANG_FAIL;
            }
        }

        // Now that entry points are associated with translation units,
        // we can make one additional pass where if an entry point has
        // no specified stage, but the nameing of its translation unit
        // implies a stage, we will use that (a manual `-stage` annotation
        // will always win out in such a case).
        //
        for (auto& rawEntryPoint : m_rawEntryPoints)
        {
            // Skip entry points that already have a stage.
            if (rawEntryPoint.stage != Stage::Unknown)
                continue;

            // Sanity check: don't process entry points with no associated translation unit.
            if (rawEntryPoint.translationUnitIndex == -1)
                continue;

            auto impliedStage =
                m_rawTranslationUnits[rawEntryPoint.translationUnitIndex].impliedStage;
            if (impliedStage != Stage::Unknown)
                rawEntryPoint.stage = impliedStage;
        }

        // Note: it is possible that some entry points still won't have associated
        // stages at this point, but we don't want to error out here, because
        // those entry points might get stages later, as part of semantic checking,
        // if the corresponding function has a `[shader("...")]` attribute.

        // Now that we've tried to establish stages for entry points, we can
        // issue diagnostics for cases where stages were set redundantly or
        // in conflicting ways.
        //
        for (auto& rawEntryPoint : m_rawEntryPoints)
        {
            if (rawEntryPoint.conflictingStagesSet)
            {
                m_sink->diagnose(
                    SourceLoc(),
                    Diagnostics::conflictingStagesForEntryPoint,
                    rawEntryPoint.name);
            }
            else if (rawEntryPoint.redundantStageSet)
            {
                m_sink->diagnose(
                    SourceLoc(),
                    Diagnostics::sameStageSpecifiedMoreThanOnce,
                    rawEntryPoint.stage,
                    rawEntryPoint.name);
            }
            else if (rawEntryPoint.translationUnitIndex != -1)
            {
                // As a quality-of-life feature, if the file name implies a particular
                // stage, but the user manually specified something different for
                // their entry point, give a warning in case they made a mistake.

                auto& rawTranslationUnit =
                    m_rawTranslationUnits[rawEntryPoint.translationUnitIndex];
                if (rawTranslationUnit.impliedStage != Stage::Unknown &&
                    rawEntryPoint.stage != Stage::Unknown &&
                    rawTranslationUnit.impliedStage != rawEntryPoint.stage)
                {
                    m_sink->diagnose(
                        SourceLoc(),
                        Diagnostics::explicitStageDoesntMatchImpliedStage,
                        rawEntryPoint.name,
                        rawEntryPoint.stage,
                        rawTranslationUnit.impliedStage);
                }
            }
        }

        // If the user is requesting code generation via pass-through,
        // then any entry points they specify need to have a stage set,
        // because fxc/dxc/glslang don't have a facility for taking
        // a named entry point and pulling its stage from an attribute.
        //
        if (_passThroughRequiresStage(m_requestImpl->m_passThrough))
        {
            for (auto& rawEntryPoint : m_rawEntryPoints)
            {
                if (rawEntryPoint.stage == Stage::Unknown)
                {
                    m_sink->diagnose(
                        SourceLoc(),
                        Diagnostics::noStageSpecifiedInPassThroughMode,
                        rawEntryPoint.name);
                }
            }
        }

        // We now have inferred enough information to add the
        // entry points to our compile request.
        //
        for (auto& rawEntryPoint : m_rawEntryPoints)
        {
            if (rawEntryPoint.translationUnitIndex < 0)
                continue;

            auto translationUnitID =
                m_rawTranslationUnits[rawEntryPoint.translationUnitIndex].translationUnitID;

            List<const char*> specializationArgs;
            for (auto& arg : rawEntryPoint.specializationArgs)
                specializationArgs.add(arg.getBuffer());

            int entryPointID = m_compileRequest->addEntryPointEx(
                translationUnitID,
                rawEntryPoint.name.begin(),
                SlangStage(rawEntryPoint.stage),
                (int)specializationArgs.getCount(),
                specializationArgs.getBuffer());

            rawEntryPoint.entryPointID = entryPointID;
        }

        // We are going to build a mapping from target formats to the
        // target that handles that format.
        Dictionary<CodeGenTarget, int> mapFormatToTargetIndex;

        // If there was no explicit `-target` specified, then we will look
        // at the `-o` options to see what we can infer.
        //
        if (m_rawTargets.getCount() == 0)
        {
            // If there are no targets and no outputs
            if (m_rawOutputs.getCount() == 0)
            {
                m_requestImpl->m_emitIr = true;
            }
            else
            {
                for (auto& rawOutput : m_rawOutputs)
                {
                    // Some outputs don't imply a target format, and we shouldn't use those for
                    // inference.
                    auto impliedFormat = rawOutput.impliedFormat;
                    if (impliedFormat == CodeGenTarget::Unknown)
                        continue;

                    int targetIndex = 0;
                    if (!mapFormatToTargetIndex.tryGetValue(impliedFormat, targetIndex))
                    {
                        targetIndex = (int)m_rawTargets.getCount();

                        RawTarget rawTarget;
                        rawTarget.format = impliedFormat;
                        m_rawTargets.add(rawTarget);

                        mapFormatToTargetIndex[impliedFormat] = targetIndex;
                    }

                    rawOutput.targetIndex = targetIndex;
                }
            }
        }
        else
        {
            // If there were explicit targets, then we will use those, but still
            // build up our mapping. We should object if the same target format
            // is specified more than once (just because of the ambiguities
            // it will create).
            //
            int targetCount = (int)m_rawTargets.getCount();
            for (int targetIndex = 0; targetIndex < targetCount; ++targetIndex)
            {
                auto format = m_rawTargets[targetIndex].format;

                if (mapFormatToTargetIndex.containsKey(format))
                {
                    m_sink->diagnose(SourceLoc(), Diagnostics::duplicateTargets, format);
                }
                else
                {
                    mapFormatToTargetIndex[format] = targetIndex;
                }
            }
        }

        // If we weren't able to infer any targets from output paths (perhaps
        // because there were no output paths), but there was a profile specified,
        // then we can try to infer a target from the profile.
        //
        if (m_rawTargets.getCount() == 0 &&
            m_defaultTarget.optionSet.getProfileVersion() != ProfileVersion::Unknown &&
            !m_defaultTarget.conflictingProfilesSet)
        {
            // Let's see if the chosen profile allows us to infer
            // the code gen target format that the user probably meant.
            //
            CodeGenTarget inferredFormat = CodeGenTarget::Unknown;
            auto profileVersion = m_defaultTarget.optionSet.getProfileVersion();
            switch (Profile(profileVersion).getFamily())
            {
            default:
                break;

                // For GLSL profile versions, we will assume SPIR-V
                // is the output format the user intended.
            case ProfileFamily::GLSL:
                inferredFormat = CodeGenTarget::SPIRV;
                break;

                // For DX profile versions, we will assume that the
                // user wants DXIL for Shader Model 6.0 and up,
                // and DXBC for all earlier versions.
                //
                // Note: There is overlap where both DXBC and DXIL
                // nominally support SM 5.1, but in general we
                // expect users to prefer to make a clean break
                // at SM 6.0. Anybody who cares about the overlap
                // cases should manually specify `-target dxil`.
                //
            case ProfileFamily::DX:
                if (profileVersion >= ProfileVersion::DX_6_0)
                {
                    inferredFormat = CodeGenTarget::DXIL;
                }
                else
                {
                    inferredFormat = CodeGenTarget::DXBytecode;
                }
                break;
            }

            if (inferredFormat != CodeGenTarget::Unknown)
            {
                RawTarget rawTarget;
                rawTarget.format = inferredFormat;
                m_rawTargets.add(rawTarget);
            }
        }

        // Similar to the case for entry points, if there is a single target,
        // then we allow some of its options to come from the "default"
        // target state.
        auto defaultTargetFloatingPointMode =
            m_defaultTarget.optionSet.getEnumOption<FloatingPointMode>(
                CompilerOptionName::FloatingPointMode);

        if (m_rawTargets.getCount() == 1)
        {
            m_rawTargets[0].optionSet.overrideWith(m_defaultTarget.optionSet);
        }
        else
        {
            // If the "default" target has had a profile (or other state)
            // specified, but there is != 1 taget, then that state doesn't
            // apply to anythign and we should give the user an error.
            //
            if (m_defaultTarget.optionSet.getProfileVersion() != ProfileVersion::Unknown)
            {
                if (m_rawTargets.getCount() == 0)
                {
                    // This should only happen if there were multiple `-profile` options,
                    // so we didn't try to infer a target, or if the `-profile` option
                    // somehow didn't imply a target.
                    //
                    m_sink->diagnose(
                        SourceLoc(),
                        Diagnostics::profileSpecificationIgnoredBecauseNoTargets);
                }
                else
                {
                    m_sink->diagnose(
                        SourceLoc(),
                        Diagnostics::profileSpecificationIgnoredBecauseBeforeAllTargets);
                }
            }

            if (defaultTargetFloatingPointMode != FloatingPointMode::Default)
            {
                if (m_rawTargets.getCount() == 0)
                {
                    m_sink->diagnose(SourceLoc(), Diagnostics::targetFlagsIgnoredBecauseNoTargets);
                }
                else
                {
                    m_sink->diagnose(
                        SourceLoc(),
                        Diagnostics::targetFlagsIgnoredBecauseBeforeAllTargets);
                }
            }
        }
        for (auto& rawTarget : m_rawTargets)
        {
            if (rawTarget.conflictingProfilesSet)
            {
                m_sink->diagnose(
                    SourceLoc(),
                    Diagnostics::conflictingProfilesSpecifiedForTarget,
                    rawTarget.format);
            }
            else if (rawTarget.redundantProfileSet)
            {
                m_sink->diagnose(
                    SourceLoc(),
                    Diagnostics::sameProfileSpecifiedMoreThanOnce,
                    rawTarget.optionSet.getProfileVersion(),
                    rawTarget.format);
            }
        }

        // TODO: do we need to require that a target must have a profile specified,
        // or will we continue to allow the profile to be inferred from the target?

        // We now have enough information to go ahead and declare the targets
        // through the Slang API:
        //
        for (auto& rawTarget : m_rawTargets)
        {
            int targetID = m_compileRequest->addCodeGenTarget(SlangCompileTarget(rawTarget.format));
            rawTarget.targetID = targetID;

            if (rawTarget.optionSet.getProfileVersion() != ProfileVersion::Unknown)
            {
                m_compileRequest->setTargetProfile(
                    targetID,
                    SlangProfileID(Profile(rawTarget.optionSet.getProfileVersion()).raw));
            }
            for (auto atom : rawTarget.optionSet.getArray(CompilerOptionName::Capability))
            {
                m_requestImpl->addTargetCapability(targetID, SlangCapabilityID(atom.intValue));
            }

            auto floatingPointMode = rawTarget.optionSet.getEnumOption<FloatingPointMode>(
                CompilerOptionName::FloatingPointMode);
            if (floatingPointMode != FloatingPointMode::Default)
            {
                m_compileRequest->setTargetFloatingPointMode(
                    targetID,
                    SlangFloatingPointMode(floatingPointMode));
            }

            if (rawTarget.optionSet.shouldUseScalarLayout())
            {
                m_compileRequest->setTargetForceGLSLScalarBufferLayout(targetID, true);
            }

            if (rawTarget.optionSet.shouldUseDXLayout())
            {
                m_compileRequest->setTargetForceDXLayout(targetID, true);
            }

            if (rawTarget.optionSet.getBoolOption(CompilerOptionName::GenerateWholeProgram))
            {
                m_compileRequest->setTargetGenerateWholeProgram(targetID, true);
            }

            if (rawTarget.optionSet.getBoolOption(CompilerOptionName::EmbedDownstreamIR))
            {
                m_compileRequest->setTargetEmbedDownstreamIR(targetID, true);
            }
        }

        // Next we need to sort out the output files specified with `-o`, and
        // figure out which entry point and/or target they apply to.
        //
        // If there is only a single entry point, then that is automatically
        // the entry point that should be associated with all outputs.
        //
        if (m_rawEntryPoints.getCount() == 1)
        {
            for (auto& rawOutput : m_rawOutputs)
            {
                rawOutput.entryPointIndex = 0;
            }
        }
        //
        // Similarly, if there is only one target, then all outputs must
        // implicitly appertain to that target.
        //
        if (m_rawTargets.getCount() == 1)
        {
            for (auto& rawOutput : m_rawOutputs)
            {
                rawOutput.targetIndex = 0;
            }
        }

        // If we don't have any raw outputs but do have a raw target,
        // add an empty' rawOutput for certain targets where the expected behavior is obvious.
        if (m_rawOutputs.getCount() == 0 && m_rawTargets.getCount() == 1 &&
            (m_rawTargets[0].format == CodeGenTarget::HostCPPSource ||
             m_rawTargets[0].format == CodeGenTarget::PyTorchCppBinding ||
             m_rawTargets[0].format == CodeGenTarget::CUDASource ||
             m_rawTargets[0].format == CodeGenTarget::SPIRV ||
             m_rawTargets[0].format == CodeGenTarget::SPIRVAssembly ||
             m_rawTargets[0].format == CodeGenTarget::Metal ||
             m_rawTargets[0].format == CodeGenTarget::MetalLib ||
             m_rawTargets[0].format == CodeGenTarget::MetalLibAssembly ||
             ArtifactDescUtil::makeDescForCompileTarget(asExternal(m_rawTargets[0].format)).kind ==
                 ArtifactKind::HostCallable))
        {
            RawOutput rawOutput;
            rawOutput.impliedFormat = m_rawTargets[0].format;
            rawOutput.targetIndex = 0;
            m_rawOutputs.add(rawOutput);
        }

        // Consider the output files specified via `-o` and try to figure
        // out how to deal with them.
        //
        for (auto& rawOutput : m_rawOutputs)
        {
            // For now, most output formats need to be tightly bound to
            // both a target and an entry point.

            // If an output doesn't have a target associated with
            // it, then search for the target with the matching format.
            if (rawOutput.targetIndex == -1)
            {
                auto impliedFormat = rawOutput.impliedFormat;
                int targetIndex = -1;

                if (impliedFormat == CodeGenTarget::Unknown)
                {

                    // If we hit this case, then it means that we need to pick the
                    // target to assocaite with this output based on its implied
                    // format, but the file path doesn't direclty imply a format
                    // (it doesn't have a suffix like `.spv` that tells us what to write).
                    //
                    m_sink->diagnose(
                        SourceLoc(),
                        Diagnostics::cannotDeduceOutputFormatFromPath,
                        rawOutput.path);
                }
                else if (mapFormatToTargetIndex.tryGetValue(rawOutput.impliedFormat, targetIndex))
                {
                    rawOutput.targetIndex = targetIndex;
                }
                else
                {
                    m_sink->diagnose(
                        SourceLoc(),
                        Diagnostics::cannotMatchOutputFileToTarget,
                        rawOutput.path,
                        rawOutput.impliedFormat);
                }
            }

            // We won't do any searching to match an output file
            // with an entry point, since the case of a single entry
            // point was handled above, and the user is expected to
            // follow the ordering rules when using multiple entry points.
            if (rawOutput.entryPointIndex == -1)
            {
                if (rawOutput.targetIndex != -1)
                {
                    auto outputFormat = m_rawTargets[rawOutput.targetIndex].format;
                    // Here we check whether the given output format supports multiple entry points
                    // When we add targets with support for multiple entry points,
                    // we should update this switch with those new formats
                    switch (outputFormat)
                    {
                    case CodeGenTarget::CPPSource:
                    case CodeGenTarget::PTX:
                    case CodeGenTarget::CUDASource:

                    case CodeGenTarget::HostHostCallable:
                    case CodeGenTarget::ShaderHostCallable:
                    case CodeGenTarget::HostExecutable:
                    case CodeGenTarget::ShaderSharedLibrary:
                    case CodeGenTarget::HostSharedLibrary:
                    case CodeGenTarget::PyTorchCppBinding:
                    case CodeGenTarget::DXIL:
                    case CodeGenTarget::MetalLib:
                    case CodeGenTarget::MetalLibAssembly:
                    case CodeGenTarget::Metal:
                    case CodeGenTarget::WGSL:
                    case CodeGenTarget::HostVM:
                        rawOutput.isWholeProgram = true;
                        break;
                    case CodeGenTarget::SPIRV:
                    case CodeGenTarget::SPIRVAssembly:
                        if (getCurrentTarget()->optionSet.shouldEmitSPIRVDirectly())
                        {
                            rawOutput.isWholeProgram = true;
                            break;
                        }
                        else if (m_rawEntryPoints.getCount() != 0)
                        {
                            rawOutput.entryPointIndex = (int)m_rawEntryPoints.getCount() - 1;
                            break;
                        }
                        [[fallthrough]];
                    default:
                        if (rawOutput.path.getLength() != 0)
                        {
                            m_sink->diagnose(
                                SourceLoc(),
                                Diagnostics::cannotMatchOutputFileToEntryPoint,
                                rawOutput.path);
                        }
                        break;
                    }
                }
            }
        }
    }

    // Now that we've diagnosed the output paths, we can add them
    // to the compile request at the appropriate locations.
    //
    // We will consider the output files specified via `-o` and try to figure
    // out how to deal with them.
    //
    for (auto& rawOutput : m_rawOutputs)
    {
        if (rawOutput.targetIndex == -1)
            continue;
        auto targetID = m_rawTargets[rawOutput.targetIndex].targetID;
        auto target = m_requestImpl->getLinkage()->targets[targetID];
        RefPtr<EndToEndCompileRequest::TargetInfo> targetInfo;
        if (!m_requestImpl->m_targetInfos.tryGetValue(target, targetInfo))
        {
            targetInfo = new EndToEndCompileRequest::TargetInfo();
            m_requestImpl->m_targetInfos[target] = targetInfo;
        }
        target->getOptionSet().overrideWith(m_rawTargets[rawOutput.targetIndex].optionSet);
        if (rawOutput.isWholeProgram)
        {
            if (targetInfo->wholeTargetOutputPath != "")
            {
                m_sink->diagnose(
                    SourceLoc(),
                    Diagnostics::duplicateOutputPathsForTarget,
                    target->getTarget());
            }
            else
            {
                target->getOptionSet().addTargetFlags(SLANG_TARGET_FLAG_GENERATE_WHOLE_PROGRAM);
                targetInfo->wholeTargetOutputPath = rawOutput.path;
            }
        }
        else
        {
            if (rawOutput.entryPointIndex == -1)
                continue;

            auto entryPoint = m_rawEntryPoints[rawOutput.entryPointIndex];
            Int entryPointID = entryPoint.entryPointID;
            if (entryPointID == -1)
            {
                m_sink->diagnose(
                    SourceLoc(),
                    Diagnostics::entryPointFunctionNotFound,
                    entryPoint.name);
                continue;
            }
            auto entryPointReq = m_requestImpl->getFrontEndReq()->getEntryPointReqs()[entryPointID];

            // String outputPath;
            if (targetInfo->entryPointOutputPaths.containsKey(entryPointID))
            {
                m_sink->diagnose(
                    SourceLoc(),
                    Diagnostics::duplicateOutputPathsForEntryPointAndTarget,
                    entryPointReq->getName(),
                    target->getTarget());
            }
            else
            {
                targetInfo->entryPointOutputPaths[entryPointID] = rawOutput.path;
            }
        }
    }


    // Copy all settings from linkage to targets.
    for (auto target : linkage->targets)
    {
        target->getOptionSet().inheritFrom(linkage->m_optionSet);

        // If there is no target specified in command line, we should inherit the default target
        // options.
        if (m_rawTargets.getCount() == 0)
        {
            target->getOptionSet().inheritFrom(m_defaultTarget.optionSet);
        }
    }

    // If there are no targets specified in command line, and addCodeGenTarget() is not called
    // yet, the options for the default target will be gone after option parsing. We
    // should save the option for the future use when addCodeGenTarget() is called.
    if ((linkage->targets.getCount() == 0) && (m_rawTargets.getCount() == 0))
    {
        m_requestImpl->m_optionSetForDefaultTarget = m_defaultTarget.optionSet;
    }

    applySettingsToDiagnosticSink(m_requestImpl->getSink(), m_sink, linkage->m_optionSet);

    return (m_sink->getErrorCount() == 0) ? SLANG_OK : SLANG_FAIL;
}

SlangResult OptionsParser::parse(
    SlangCompileRequest* compileRequest,
    int argc,
    char const* const* argv)
{
    m_compileRequest = compileRequest;

    // Set up useful members
    m_requestImpl = asInternal(compileRequest);

    auto session = asInternal(m_requestImpl->getSession());

    m_session = session;
    m_frontEndReq = m_requestImpl->getFrontEndReq();

    m_cmdOptions = &session->m_commandOptions;
    m_cmdLineContext = m_requestImpl->getLinkage()->m_cmdLineContext.get();

    DiagnosticSink* requestSink = m_requestImpl->getSink();

    // Why create a new DiagnosticSink?
    // We *don't* want the lexer that comes as default (it's for Slang source!)
    // We may want to set flags that are different
    // We will need to use a new sourceManager that will just last for this parse and will map locs
    // to source lines.
    //
    // The *problem* is that we still need to communicate to the requestSink in some suitable way.
    //
    // 1) We could have some kind of scoping mechanism (and only one sink)
    // 2) We could have a 'parent' diagnostic sink, that if we set we route output too
    // 3) We use something like the ISlangWriter to always be the thing output too (this has
    // problems because some code assumes the diagnostics are accessible as a string)
    //
    // The solution used here is to have DiagnosticsSink have a 'parent' that also gets diagnostics
    // reported to.

    m_parseSink.init(m_cmdLineContext->getSourceManager(), nullptr);
    {
        m_parseSink.setFlags(requestSink->getFlags());
        // Allow HumaneLoc - it won't display much for command line parsing - just (1):
        // Leaving allows for diagnostics to be compatible with other Slang diagnostic parsing.
        // parseSink.resetFlag(DiagnosticSink::Flag::HumaneLoc);
        m_parseSink.setFlag(DiagnosticSink::Flag::SourceLocationLine);
    }

    // All diagnostics will also be sent to requestSink
    m_parseSink.setParentSink(requestSink);
    m_sink = &m_parseSink;

    Result res = _parse(argc, argv);

    m_sink = nullptr;

    if (m_parseSink.getErrorCount() > 0)
    {
        // Put the errors in the diagnostic
        m_requestImpl->m_diagnosticOutput = m_parseSink.outputBuffer.produceString();
    }

    return res;
}

SlangResult parseOptions(SlangCompileRequest* inCompileRequest, int argc, char const* const* argv)
{
    OptionsParser parser;
    return parser.parse(inCompileRequest, argc, argv);
}


} // namespace Slang
