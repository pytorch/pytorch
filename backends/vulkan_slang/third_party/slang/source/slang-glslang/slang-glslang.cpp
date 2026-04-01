// slang-glslang.cpp
#include "slang-glslang.h"

#include "SPIRV/GlslangToSpv.h"
#include "glslang/MachineIndependent/localintermediate.h"
#include "glslang/Public/ShaderLang.h"
#include "slang.h"
#include "spirv-tools/libspirv.h"
#include "spirv-tools/linker.hpp"
#include "spirv-tools/optimizer.hpp"

#ifdef _WIN32
#include <windows.h>
#endif

#include <cassert>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>

// This is a wrapper to allow us to run the `glslang` compiler
// in a controlled fashion.

#define UNLIMITED 9999

static TBuiltInResource _calcBuiltinResources()
{
    // NOTE! This is a bit of a hack - to set all the fields to true/UNLIMITED.
    // Care must be taken if new variables are introduced, the default may not be appropriate.

    // We are relying on limits being after the other fields.
    SLANG_COMPILE_TIME_ASSERT(SLANG_OFFSET_OF(TBuiltInResource, limits) > 0);
    // We are relying on maxLights being the first parameter, and all values will have the same type
    SLANG_COMPILE_TIME_ASSERT(SLANG_OFFSET_OF(TBuiltInResource, maxLights) == 0);

    TBuiltInResource resource;
    // Set up all the integer values.
    {

        auto* dst = &resource.maxLights;
        const size_t count = SLANG_OFFSET_OF(TBuiltInResource, limits) / sizeof(*dst);
        for (size_t i = 0; i < count; ++i)
        {
            dst[i] = UNLIMITED;
        }
    }

    // In the sea of variables there is a min value
    resource.minProgramTexelOffset = -UNLIMITED;

    // Set up the bools
    {
        TLimits* limits = &resource.limits;
        bool* dst = (bool*)limits;

        const size_t count = sizeof(TLimits) / sizeof(bool);
        for (size_t i = 0; i < count; ++i)
        {
            dst[i] = true;
        }
    }
    return resource;
}

static TBuiltInResource gResources = _calcBuiltinResources();

static void dump(
    void const* data,
    size_t size,
    glslang_OutputFunc outputFunc,
    void* outputUserData,
    FILE* fallbackStream)
{
    if (outputFunc)
    {
        outputFunc(data, size, outputUserData);
    }
    else
    {
        fwrite(data, 1, size, fallbackStream);

        // also output it for debug purposes
        std::string str((char const*)data, size);
#ifdef _WIN32
        OutputDebugStringA(str.c_str());
#else
        fprintf(stderr, "%s\n", str.c_str());
        ;
#endif
    }
}

static void dumpDiagnostics(const glslang_CompileRequest_1_2& request, std::string const& log)
{
    dump(log.c_str(), log.length(), request.diagnosticFunc, request.diagnosticUserData, stderr);
}

struct SPIRVOptimizationDiagnostic
{
    std::string toString() const
    {
        std::ostringstream out;

        switch (level)
        {
        case SPV_MSG_FATAL:
        case SPV_MSG_INTERNAL_ERROR:
        case SPV_MSG_ERROR:
            out << "error: ";
            break;
        case SPV_MSG_WARNING:
            out << "warning: ";
            break;
        case SPV_MSG_INFO:
        case SPV_MSG_DEBUG:
            out << "info: ";
            break;
        default:
            break;
        }
        if (source.length())
        {
            out << source << ":";
        }
        out << position.line << ":" << position.column << ":" << position.index << ":";
        if (message.length())
        {
            out << " " << message;
        }

        return out.str();
    }

    spv_message_level_t level;
    std::string source;
    spv_position_t position;
    std::string message;
};

// TODO: the actual printing should happen on the application side.
static void validationMessageConsumer(
    spv_message_level_t level,
    const char*,
    const spv_position_t& position,
    const char* message)
{
    switch (level)
    {
    case SPV_MSG_FATAL:
    case SPV_MSG_INTERNAL_ERROR:
    case SPV_MSG_ERROR:
        std::cerr << "error: line " << position.index << ": " << message << std::endl;
        break;
    case SPV_MSG_WARNING:
        std::cout << "warning: line " << position.index << ": " << message << std::endl;
        break;
    case SPV_MSG_INFO:
        std::cout << "info: line " << position.index << ": " << message << std::endl;
        break;
    default:
        break;
    }
}

// Validate the given SPIRV-ASM instructions.
extern "C"
#ifdef _MSC_VER
    _declspec(dllexport)
#else
    __attribute__((__visibility__("default")))
#endif
        bool glslang_validateSPIRV(const uint32_t* contents, int contentsSize)
{
    spv_target_env target_env = SPV_ENV_UNIVERSAL_1_6;

    spvtools::ValidatorOptions options;
    options.SetScalarBlockLayout(true);
    options.SetFriendlyNames(true);

    spvtools::SpirvTools tools(target_env);
    tools.SetMessageConsumer(validationMessageConsumer);

    return tools.Validate(contents, contentsSize, options);
}

// Disassemble the given SPIRV-ASM instructions and return the result as a string.
extern "C"
#ifdef _MSC_VER
    _declspec(dllexport)
#else
__attribute__((__visibility__("default")))
#endif
        bool glslang_disassembleSPIRVWithResult(
            const uint32_t* contents,
            int contentsSize,
            char** outString)
{
    static const auto kDefaultEnvironment = SPV_ENV_UNIVERSAL_1_5;
    spv_text text;

    uint32_t options = SPV_BINARY_TO_TEXT_OPTION_NONE;
    options |= SPV_BINARY_TO_TEXT_OPTION_COMMENT;
    options |= SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES;
    options |= SPV_BINARY_TO_TEXT_OPTION_INDENT;

    spv_diagnostic diagnostic = nullptr;
    spv_context context = spvContextCreate(kDefaultEnvironment);
    spv_result_t error =
        spvBinaryToText(context, contents, contentsSize, options, &text, &diagnostic);
    spvContextDestroy(context);
    if (error)
    {
        spvDiagnosticPrint(diagnostic);
        spvDiagnosticDestroy(diagnostic);
        return false;
    }
    else
    {
        if (outString)
        {
            // Allocate memory for the output string and copy the result
            size_t len = text->length + 1; // +1 for null terminator
            *outString = new char[len];
            memcpy(*outString, text->str, text->length);
            (*outString)[text->length] = '\0'; // Ensure null termination
        }

        spvTextDestroy(text);
        return true;
    }
}


// Disassemble the given SPIRV-ASM instructions.
extern "C"
#ifdef _MSC_VER
    _declspec(dllexport)
#else
__attribute__((__visibility__("default")))
#endif
        bool glslang_disassembleSPIRV(const uint32_t* contents, int contentsSize)
{
    return glslang_disassembleSPIRVWithResult(contents, contentsSize, nullptr);
}

// Apply the SPIRV-Tools optimizer to generated SPIR-V based on the desired optimization level
// TODO: add flag for optimizing SPIR-V size as well
static void glslang_optimizeSPIRV(
    spv_target_env targetEnv,
    const glslang_CompileRequest_1_2& request,
    std::vector<SPIRVOptimizationDiagnostic>& outDiags,
    std::vector<unsigned int>& ioSpirv)
{
    const auto optimizationLevel = request.optimizationLevel;

    // If there is no optimization then we are done
    if (optimizationLevel == SLANG_OPTIMIZATION_LEVEL_NONE)
    {
        return;
    }

    const auto debugInfoType = request.debugInfoType;

    spvtools::Optimizer optimizer(targetEnv);

    optimizer.SetMessageConsumer(
        [&](spv_message_level_t level,
            const char* source,
            const spv_position_t& position,
            const char* message)
        {
            SPIRVOptimizationDiagnostic diag;
            diag.level = level;
            if (source)
            {
                diag.source = source;
            }
            diag.position = position;
            if (message)
            {
                diag.message = message;
            }
            outDiags.push_back(diag);
        });

    // If debug info is being generated, propagate
    // line information into all SPIR-V instructions. This avoids loss of
    // information when instructions are deleted or moved. Later, remove
    // redundant information to minimize final SPRIR-V size.
    if (debugInfoType != SLANG_DEBUG_INFO_LEVEL_NONE)
    {
        optimizer.RegisterPass(spvtools::CreatePropagateLineInfoPass());
    }

    spvtools::OptimizerOptions spvOptOptions;

    // To compile some large shaders the default is not enough.
    // That although this limit is exceeded, the final optimized output is typically well
    // within the range.
    //
    // See kDefaultMaxIdBound for description of this limit.
    //
    // If a compilation produces a warning like
    // `0:0: ID overflow. Try running compact-ids.`
    // it might be fixable by raising the multiplier to a larger value.
    spvOptOptions.set_max_id_bound(kDefaultMaxIdBound * 4);

    // TODO confirm which passes we want to invoke for each level
    switch (optimizationLevel)
    {
    default:
    case SLANG_OPTIMIZATION_LEVEL_DEFAULT:
        {
            // Use a minimal set of performance settings
            // If we run CreateInlineExhaustivePass, We need to run CreateMergeReturnPass first.

#if 0
            // This is the previous 'default optimization' passes setting for glslang
            optimizer.RegisterPass(spvtools::CreateMergeReturnPass());
            optimizer.RegisterPass(spvtools::CreateInlineExhaustivePass());
            optimizer.RegisterPass(spvtools::CreateAggressiveDCEPass());
            optimizer.RegisterPass(spvtools::CreatePrivateToLocalPass());
            optimizer.RegisterPass(spvtools::CreateScalarReplacementPass(100));
            optimizer.RegisterPass(spvtools::CreateLocalAccessChainConvertPass());
            optimizer.RegisterPass(spvtools::CreateAggressiveDCEPass());
#elif 1
            // 6Mb 27 secs (all passes up to 9)
            // 9Mb 25 secs (all passes up to 7)
            // 8Mb 15 secs (all passes) -(5,6,7)
            // 6Mb 15 secs (all passes) -(6,7)

            // This list of passes takes the previous 'default optimization'
            // passes (as listed above) and tries to combine them in order with the 'new' passes
            // below. The issue with the passes below is that although it produces smaller SPIR-V
            // fairly quickly it can cause serious problem on some drivers.
            //
            // Across a wide range of compilations this produced SPIR-V that is less than half size
            // of the previous -O1 passes above.

            optimizer.RegisterPass(spvtools::CreateWrapOpKillPass());     // 1
            optimizer.RegisterPass(spvtools::CreateDeadBranchElimPass()); // 2

            optimizer.RegisterPass(spvtools::CreateMergeReturnPass());
            optimizer.RegisterPass(spvtools::CreateInlineExhaustivePass());

            optimizer.RegisterPass(spvtools::CreateEliminateDeadFunctionsPass()); // 3

            optimizer.RegisterPass(spvtools::CreateAggressiveDCEPass());
            optimizer.RegisterPass(spvtools::CreatePrivateToLocalPass());

            optimizer.RegisterPass(spvtools::CreateScalarReplacementPass(100));

            optimizer.RegisterPass(spvtools::CreateCCPPass());            // 4 *
            optimizer.RegisterPass(spvtools::CreateSimplificationPass()); // 5
            // optimizer.RegisterPass(spvtools::CreateIfConversionPass());         // 6
            // optimizer.RegisterPass(spvtools::CreateBlockMergePass());           // 7 *

            optimizer.RegisterPass(spvtools::CreateLocalAccessChainConvertPass());

            optimizer.RegisterPass(spvtools::CreateLocalSingleBlockLoadStoreElimPass()); // 8

            optimizer.RegisterPass(spvtools::CreateAggressiveDCEPass());

            optimizer.RegisterPass(spvtools::CreateVectorDCEPass()); // 9

#else
            // The following selection of passes was created by
            // 1) Taking the list of passes from optimizer.RegisterSizePasses
            // 2) Disable/enable passes to try to produce some reasonable combination of low SPIR-V
            // output size and compilation speed
            //
            // For a particularly difficult glsl shader this produced 1/3 SPIR-V code (against
            // previous -O1), in around 13th the time (against -O3 option) Over a wide range of
            // compiles the SPIR-V is around 6% larger than -O3

            // The following comments describe the path to finding this combination. The original
            // compilation produces 18Mb SPIR-V binaries in around 3 1/2 mins. The integer number
            // increases with the ordering of the test.
            //
            // With 5 47s
            // With 6 we have 6Mb, and 38 seconds
            // With 7 we have 6Mb and 26 seconds
            // With 8 we have 6Mb in 18 seconds
            // 9 didn't improve perf or size
            // With 10 we have 6Mb in 16.8
            // With 11 we have 6Mb in 16.1
            // With 12 we have 6Mb in 15.6
            // With 13 didn't improve
            // With 14 slightly larger, slightly smaller, so leave
            // Try 15 - Adding one and removing the other, makes things much worse
            // Without any SSA rewrite we are up to 6Mb. 48
            //
            // So (for test case) approximately 13x compilation speed.
            // Binary twice the size of smallest SPIR-V size and 1/3 the size of the previous -O
            // size
            optimizer.RegisterPass(spvtools::CreateWrapOpKillPass());
            optimizer.RegisterPass(spvtools::CreateDeadBranchElimPass()); // 15
            optimizer.RegisterPass(spvtools::CreateMergeReturnPass());
            optimizer.RegisterPass(spvtools::CreateInlineExhaustivePass());
            optimizer.RegisterPass(spvtools::CreateEliminateDeadFunctionsPass()); // 9
            optimizer.RegisterPass(spvtools::CreatePrivateToLocalPass());
            // optimizer.RegisterPass(spvtools::CreateScalarReplacementPass(0));   // 12
            // optimizer.RegisterPass(spvtools::CreateLocalMultiStoreElimPass());
            optimizer.RegisterPass(spvtools::CreateCCPPass());
            // optimizer.RegisterPass(spvtools::CreateLoopUnrollPass(true));     // 1
            // optimizer.RegisterPass(spvtools::CreateDeadBranchElimPass());     // 4
            // optimizer.RegisterPass(spvtools::CreateSimplificationPass());       // 11
            optimizer.RegisterPass(spvtools::CreateScalarReplacementPass(0));
            // optimizer.RegisterPass(spvtools::CreateLocalSingleStoreElimPass());
            // optimizer.RegisterPass(spvtools::CreateIfConversionPass());       // 7
            optimizer.RegisterPass(spvtools::CreateSimplificationPass()); // 13
            // optimizer.RegisterPass(spvtools::CreateAggressiveDCEPass());      // 10
            // optimizer.RegisterPass(spvtools::CreateDeadBranchElimPass());         // 6 + 15
            // optimizer.RegisterPass(spvtools::CreateBlockMergePass());             // 8
            optimizer.RegisterPass(spvtools::CreateLocalAccessChainConvertPass());
            optimizer.RegisterPass(spvtools::CreateLocalSingleBlockLoadStoreElimPass());
            optimizer.RegisterPass(spvtools::CreateAggressiveDCEPass()); // 5
            // optimizer.RegisterPass(spvtools::CreateCopyPropagateArraysPass());          // 1
            optimizer.RegisterPass(spvtools::CreateVectorDCEPass());
            optimizer.RegisterPass(spvtools::CreateDeadInsertElimPass());
            optimizer.RegisterPass(spvtools::CreateEliminateDeadMembersPass());
            // optimizer.RegisterPass(spvtools::CreateLocalSingleStoreElimPass());
            // optimizer.RegisterPass(spvtools::CreateBlockMergePass());                 // 3
            // optimizer.RegisterPass(spvtools::CreateLocalMultiStoreElimPass());        // 2
            // optimizer.RegisterPass(spvtools::CreateRedundancyEliminationPass());
            optimizer.RegisterPass(spvtools::CreateSimplificationPass()); // 14
            optimizer.RegisterPass(spvtools::CreateAggressiveDCEPass());
            optimizer.RegisterPass(spvtools::CreateCFGCleanupPass());
#endif

            break;
        }
    // TODO(JS): It would be better if we had some distinction here where 'high' meant optimize
    // 'in a reasonable time' for a better optimization, and 'maximal' meant compilation might
    // take a really long time... so only use it if it's really needed.
    //
    // Currently we just have high have the same meaning as 'maximal'.
    case SLANG_OPTIMIZATION_LEVEL_HIGH:
    case SLANG_OPTIMIZATION_LEVEL_MAXIMAL:
        {
            // Use the same passes when specifying the "-O" flag in spirv-opt
            // Roughly equivalent to `RegisterPerformancePasses`

            optimizer.RegisterPass(spvtools::CreateWrapOpKillPass());
            optimizer.RegisterPass(spvtools::CreateDeadBranchElimPass());
            optimizer.RegisterPass(spvtools::CreateMergeReturnPass());
            optimizer.RegisterPass(spvtools::CreateInlineExhaustivePass());
            optimizer.RegisterPass(spvtools::CreateEliminateDeadFunctionsPass());
            optimizer.RegisterPass(spvtools::CreateAggressiveDCEPass());
            optimizer.RegisterPass(spvtools::CreatePrivateToLocalPass());
            optimizer.RegisterPass(spvtools::CreateLocalSingleBlockLoadStoreElimPass());
            optimizer.RegisterPass(spvtools::CreateLocalSingleStoreElimPass());
            optimizer.RegisterPass(spvtools::CreateAggressiveDCEPass());
            optimizer.RegisterPass(spvtools::CreateScalarReplacementPass());
            optimizer.RegisterPass(spvtools::CreateLocalAccessChainConvertPass());
            optimizer.RegisterPass(spvtools::CreateLocalSingleBlockLoadStoreElimPass());
            optimizer.RegisterPass(spvtools::CreateLocalSingleStoreElimPass());
            optimizer.RegisterPass(spvtools::CreateAggressiveDCEPass());

            // We run CompactIdsPass here, because CreateLocalMultiStoreElimPass can explode
            // id usage (by a factor of 10), and compacting ids here has been shown to half
            // id usage with a complex shader.
            optimizer.RegisterPass(spvtools::CreateCompactIdsPass());

            // Note that CreateLocalMultiStoreElimPass really just does a SSARewritePass
            optimizer.RegisterPass(spvtools::CreateLocalMultiStoreElimPass());

            optimizer.RegisterPass(spvtools::CreateAggressiveDCEPass());
            optimizer.RegisterPass(spvtools::CreateCCPPass());
            optimizer.RegisterPass(spvtools::CreateAggressiveDCEPass());
            optimizer.RegisterPass(spvtools::CreateLoopUnrollPass(true));
            optimizer.RegisterPass(spvtools::CreateDeadBranchElimPass());
            optimizer.RegisterPass(spvtools::CreateRedundancyEliminationPass());
            optimizer.RegisterPass(spvtools::CreateCombineAccessChainsPass());
            optimizer.RegisterPass(spvtools::CreateSimplificationPass());
            optimizer.RegisterPass(spvtools::CreateScalarReplacementPass());
            optimizer.RegisterPass(spvtools::CreateLocalAccessChainConvertPass());
            optimizer.RegisterPass(spvtools::CreateLocalSingleBlockLoadStoreElimPass());
            optimizer.RegisterPass(spvtools::CreateLocalSingleStoreElimPass());
            optimizer.RegisterPass(spvtools::CreateAggressiveDCEPass());
            optimizer.RegisterPass(spvtools::CreateSSARewritePass());
            optimizer.RegisterPass(spvtools::CreateAggressiveDCEPass());
            optimizer.RegisterPass(spvtools::CreateVectorDCEPass());
            optimizer.RegisterPass(spvtools::CreateDeadInsertElimPass());
            optimizer.RegisterPass(spvtools::CreateDeadBranchElimPass());
            optimizer.RegisterPass(spvtools::CreateSimplificationPass());
            optimizer.RegisterPass(spvtools::CreateIfConversionPass());
            optimizer.RegisterPass(spvtools::CreateCopyPropagateArraysPass());
            optimizer.RegisterPass(spvtools::CreateReduceLoadSizePass());
            optimizer.RegisterPass(spvtools::CreateAggressiveDCEPass());
            optimizer.RegisterPass(spvtools::CreateBlockMergePass());
            optimizer.RegisterPass(spvtools::CreateRedundancyEliminationPass());
            optimizer.RegisterPass(spvtools::CreateDeadBranchElimPass());
            optimizer.RegisterPass(spvtools::CreateBlockMergePass());
            optimizer.RegisterPass(spvtools::CreateSimplificationPass());

            // We again run compaction to try and ensure the final output uses ids that are in
            // range. On a complex shader, this reduced the amount ids by 5.
            optimizer.RegisterPass(spvtools::CreateCompactIdsPass());

            break;
        }
    }

    if (debugInfoType != SLANG_DEBUG_INFO_LEVEL_NONE)
    {
        optimizer.RegisterPass(spvtools::CreateRedundantLineInfoElimPass());
    }

    spvOptOptions.set_run_validator(false); // Don't run the validator by default

    {
        // Put the output optimized spirv into optSpirv
        std::vector<unsigned int> optSpirv;

        // Optimize
        if (optimizer.Run(ioSpirv.data(), ioSpirv.size(), &optSpirv, spvOptOptions))
        {
            assert(optSpirv.size() > 0);
            // Make the ioSpirv the optimized spirv
            ioSpirv.swap(optSpirv);
        }
    }
}

static int spirv_Optimize_1_2(const glslang_CompileRequest_1_2& request)
{
    std::vector<SPIRVOptimizationDiagnostic> diagnostics;
    std::vector<uint32_t> spirvBuffer;
    size_t inputBlobSize = (char*)request.inputEnd - (char*)request.inputBegin;
    spirvBuffer.resize(inputBlobSize / sizeof(uint32_t));
    memcpy(spirvBuffer.data(), request.inputBegin, inputBlobSize);

    glslang_optimizeSPIRV(SPV_ENV_UNIVERSAL_1_5, request, diagnostics, spirvBuffer);
    if (request.outputFunc)
    {
        request.outputFunc(
            spirvBuffer.data(),
            spirvBuffer.size() * sizeof(uint32_t),
            request.outputUserData);
    }
    if (request.diagnosticFunc)
    {
        for (auto& diagnostic : diagnostics)
        {
            request.diagnosticFunc(
                (void*)diagnostic.message.c_str(),
                diagnostic.message.size() * sizeof(char),
                request.diagnosticUserData);
        }
    }
    return SLANG_OK;
}

static glslang::EShTargetLanguageVersion _makeTargetLanguageVersion(
    int majorVersion,
    int minorVersion)
{
    return glslang::EShTargetLanguageVersion(
        (uint32_t(majorVersion) << 16) | (uint32_t(minorVersion) << 8));
}

static glsl_SPIRVVersion _toSPIRVVersion(glslang::EShTargetLanguageVersion version)
{
    glsl_SPIRVVersion ver;
    ver.patch = 0;
    ver.major = uint8_t(uint32_t(version) >> 16);
    ver.minor = uint8_t(uint32_t(version) >> 8);
    return ver;
}

// For working out the targets based on SPIR-V target strings

namespace
{ // anonymous

struct SPRIVTargetInfo
{
    const char* name;
    spv_target_env targetEnv;
};

} // namespace

static const SPRIVTargetInfo kSpirvTargetInfos[] = {
    {"1.0", SPV_ENV_UNIVERSAL_1_0},
    {"vk1.0", SPV_ENV_VULKAN_1_0},
    {"1.1", SPV_ENV_UNIVERSAL_1_1},
    {"cl2.1", SPV_ENV_OPENCL_2_1},
    {"cl2.2", SPV_ENV_OPENCL_2_2},
    {"gl4.0", SPV_ENV_OPENGL_4_0},
    {"gl4.1", SPV_ENV_OPENGL_4_1},
    {"gl4.2", SPV_ENV_OPENGL_4_2},
    {"gl4.3", SPV_ENV_OPENGL_4_3},
    {"gl4.5", SPV_ENV_OPENGL_4_5},
    {"1.2", SPV_ENV_UNIVERSAL_1_2},
    {"cl1.2", SPV_ENV_OPENCL_1_2},
    {"cl_emb1.2", SPV_ENV_OPENCL_EMBEDDED_1_2},
    {"cl2.0", SPV_ENV_OPENCL_2_0},
    {"cl_emb2.0", SPV_ENV_OPENCL_EMBEDDED_2_0},
    {"cl_emb2.1", SPV_ENV_OPENCL_EMBEDDED_2_1},
    {"cl_emb2.2", SPV_ENV_OPENCL_EMBEDDED_2_2},
    {"1.3", SPV_ENV_UNIVERSAL_1_3},
    {"vk1.1", SPV_ENV_VULKAN_1_1},
    {"web_gpu1.0", SPV_ENV_WEBGPU_0},
    {"1.4", SPV_ENV_UNIVERSAL_1_4},
    {"vk1.1_spirv1.4", SPV_ENV_VULKAN_1_1_SPIRV_1_4},
    {"1.5", SPV_ENV_UNIVERSAL_1_5},
};

static int _findTargetIndex(const char* name)
{
    const int count = int(sizeof(kSpirvTargetInfos) / sizeof(kSpirvTargetInfos[0]));
    for (int i = 0; i < count; ++i)
    {
        const SPRIVTargetInfo& info = kSpirvTargetInfos[i];

        if (::strcmp(info.name, name) == 0)
        {
            return i;
        }
    }
    return -1;
}

static spv_target_env _getUniversalTargetEnv(glslang::EShTargetLanguageVersion inVersion)
{
    glsl_SPIRVVersion spirvVersion = _toSPIRVVersion(inVersion);
    uint32_t ver = (uint32_t(spirvVersion.major) << 8) | spirvVersion.minor;

    switch (ver)
    {
    case 0x100:
        return SPV_ENV_UNIVERSAL_1_0;
    case 0x101:
        return SPV_ENV_UNIVERSAL_1_1;
    case 0x102:
        return SPV_ENV_UNIVERSAL_1_2;
    case 0x103:
        return SPV_ENV_UNIVERSAL_1_3;
    case 0x104:
        return SPV_ENV_UNIVERSAL_1_4;
    case 0x105:
        return SPV_ENV_UNIVERSAL_1_5;
    case 0x106:
        return SPV_ENV_UNIVERSAL_1_6;
    default:
        {
            if (ver > 0x106)
            {
                // This is the highest we known for now..., so try that
                return SPV_ENV_UNIVERSAL_1_6;
            }
            break;
        }
    }
    // Just use the default...
    return SPV_ENV_UNIVERSAL_1_2;
}

static int glslang_compileGLSLToSPIRV(glslang_CompileRequest_1_2 request)
{
    // Check that the encoding matches
    assert(glslang::EShTargetSpv_1_4 == _makeTargetLanguageVersion(1, 4));

    EShLanguage glslangStage;
    switch (request.slangStage)
    {
#define CASE(SP, GL)                \
    case SLANG_STAGE_##SP:          \
        glslangStage = EShLang##GL; \
        break
        CASE(VERTEX, Vertex);
        CASE(FRAGMENT, Fragment);
        CASE(GEOMETRY, Geometry);
        CASE(HULL, TessControl);
        CASE(DOMAIN, TessEvaluation);
        CASE(COMPUTE, Compute);

        CASE(RAY_GENERATION, RayGenNV);
        CASE(INTERSECTION, IntersectNV);
        CASE(ANY_HIT, AnyHitNV);
        CASE(CLOSEST_HIT, ClosestHitNV);
        CASE(MISS, MissNV);
        CASE(CALLABLE, CallableNV);

        CASE(MESH, Mesh);
        CASE(AMPLIFICATION, Task);
#undef CASE

    default:
        dumpDiagnostics(request, "internal error: stage unsupported by glslang\n");
        return 1;
    }

    spv_target_env targetEnv = SPV_ENV_UNIVERSAL_1_2;
    glslang::EShTargetLanguageVersion targetLanguage = glslang::EShTargetLanguageVersion(0);

    int spirvTargetIndex = -1;
    if (request.spirvTargetName)
    {
        spirvTargetIndex = _findTargetIndex(request.spirvTargetName);
        if (spirvTargetIndex < 0)
        {
            dumpDiagnostics(request, "warning: unknown SPIR-V version\n");
        }
        else
        {
            targetEnv = kSpirvTargetInfos[spirvTargetIndex].targetEnv;
        }
    }

    // If a version is specified, and no target language is specified, set to universal version of
    // that SPIR-V version
    if (request.spirvVersion.major != 0 && targetLanguage == glslang::EShTargetLanguageVersion(0))
    {
        targetLanguage =
            _makeTargetLanguageVersion(request.spirvVersion.major, request.spirvVersion.minor);
    }

    // If we don't have a target, but do have a language, use that to determine a universal target
    if (spirvTargetIndex < 0 && targetLanguage != glslang::EShTargetLanguageVersion(0))
    {
        // We can just use the appropriate universal based on the target language
        targetEnv = _getUniversalTargetEnv(targetLanguage);
    }

    // TODO: compute glslang stage to use

    glslang::TShader* shader = new glslang::TShader(glslangStage);
    auto shaderPtr = std::unique_ptr<glslang::TShader>(shader);

    // Only set the target language if one is determined
    if (targetLanguage != glslang::EShTargetLanguageVersion(0))
    {
        shader->setEnvTarget(glslang::EShTargetSpv, targetLanguage);
    }

    glslang::TProgram* program = new glslang::TProgram();
    auto programPtr = std::unique_ptr<glslang::TProgram>(program);

    char const* sourceText = (char const*)request.inputBegin;
    char const* sourceTextEnd = (char const*)request.inputEnd;

    int sourceTextLength = (int)(sourceTextEnd - sourceText);

    shader->setPreamble("#extension GL_GOOGLE_cpp_style_line_directive : require\n");
    shader->setStringsWithLengthsAndNames(&sourceText, &sourceTextLength, &request.sourcePath, 1);

    // Options for compilation of glsl to Spv

    // spvOptions ctors with default options (this it the same as passing nullptr to GlslangToSpv)
    glslang::SpvOptions spvOptions;

    const SlangDebugInfoLevel debugLevel = (SlangDebugInfoLevel)request.debugInfoType;

    // Enable generation of debug info, if any debug level other than none is requested
    if (debugLevel != SLANG_DEBUG_INFO_LEVEL_NONE)
    {
        spvOptions.generateDebugInfo = true;
        spvOptions.emitNonSemanticShaderDebugInfo = true;
        shader->setDebugInfo(true);
    }

    if (debugLevel == SLANG_DEBUG_INFO_LEVEL_MAXIMAL)
    {
        spvOptions.emitNonSemanticShaderDebugSource = true;
        spvOptions.disableOptimizer = true;
        request.optimizationLevel = SLANG_OPTIMIZATION_LEVEL_NONE;
    }

    // Link program
    {
        const EShMessages messages = EShMessages(EShMsgSpvRules | EShMsgVulkanRules);

        if (!shader->parse(&gResources, 110, false, messages))
        {
            dumpDiagnostics(request, shader->getInfoLog());
            return 1;
        }

        if (request.entryPointName && strlen(request.entryPointName))
            shader->setEntryPoint(request.entryPointName);

        program->addShader(shader);

        if (!program->link(messages))
        {
            dumpDiagnostics(request, program->getInfoLog());
            return 1;
        }

        if (!program->mapIO())
        {
            dumpDiagnostics(request, program->getInfoLog());
            return 1;
        }
    }

    for (int stage = 0; stage < EShLangCount; ++stage)
    {
        auto stageIntermediate = program->getIntermediate((EShLanguage)stage);
        if (!stageIntermediate)
            continue;
        if (debugLevel == SLANG_DEBUG_INFO_LEVEL_MAXIMAL)
        {
            stageIntermediate->addSourceText(sourceText, sourceTextLength);
        }

        std::vector<unsigned int> spirv;
        spv::SpvBuildLogger logger;

        // Copy options to make sure spvOptions not altered
        glslang::SpvOptions copySpvOptions(spvOptions);

        glslang::GlslangToSpv(*stageIntermediate, spirv, &logger, &copySpvOptions);

        int optErrorCount = 0;

        if (request.optimizationLevel != SLANG_OPTIMIZATION_LEVEL_NONE)
        {
            std::vector<SPIRVOptimizationDiagnostic> optDiags;
            glslang_optimizeSPIRV(targetEnv, request, optDiags, spirv);

            {
                for (const auto& diag : optDiags)
                {
                    // Count the number of errors
                    optErrorCount += int(diag.level <= SPV_MSG_ERROR);

                    // Note this string does not have \n.
                    std::string diagString = diag.toString();

                    // Dump
                    dump(
                        diagString.c_str(),
                        diagString.length(),
                        request.diagnosticFunc,
                        request.diagnosticUserData,
                        stderr);
                }
            }
        }

        dumpDiagnostics(request, logger.getAllMessages());

        dump(
            spirv.data(),
            spirv.size() * sizeof(unsigned int),
            request.outputFunc,
            request.outputUserData,
            stdout);

        if (optErrorCount > 0)
        {
            // It's an error...
            return 1;
        }
    }

    return 0;
}

static int glslang_dissassembleSPIRV(const glslang_CompileRequest_1_2& request)
{
    typedef unsigned int SPIRVWord;

    SPIRVWord const* spirvBegin = (SPIRVWord const*)request.inputBegin;
    SPIRVWord const* spirvEnd = (SPIRVWord const*)request.inputEnd;

    std::vector<SPIRVWord> spirv(spirvBegin, spirvEnd);

    std::string result;
    spvtools::SpirvTools spirvTools(SPV_ENV_UNIVERSAL_1_5);
    spirvTools.Disassemble(
        spirv,
        &result,
        SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES | SPV_BINARY_TO_TEXT_OPTION_COMMENT);

    dump(result.c_str(), result.length(), request.outputFunc, request.outputUserData, stdout);
    return 0;
}

// We need a per process initialization
class ProcessInitializer
{
public:
    ProcessInitializer() { m_isInitialized = false; }

    bool init()
    {
        std::lock_guard<std::mutex> guard(m_mutex);
        if (!m_isInitialized)
        {
            if (!glslang::InitializeProcess())
            {
                return false;
            }
            m_isInitialized = true;
        }
        return true;
    }

    ~ProcessInitializer()
    {
        // We *assume* will only be called once dll is detatched and that will be on a single thread
        if (m_isInitialized)
        {
            glslang::FinalizeProcess();
        }
    }

    std::mutex m_mutex;
    bool m_isInitialized = false;
};

static int _compile(const glslang_CompileRequest_1_2& request)
{
    int result = 0;
    switch (request.action)
    {
    default:
        result = 1;
        break;

    case GLSLANG_ACTION_COMPILE_GLSL_TO_SPIRV:
        result = glslang_compileGLSLToSPIRV(request);
        break;

    case GLSLANG_ACTION_DISSASSEMBLE_SPIRV:
        result = glslang_dissassembleSPIRV(request);
        break;

    case GLSLANG_ACTION_OPTIMIZE_SPIRV:
        result = spirv_Optimize_1_2(request);
        break;
    }

    return result;
}

extern "C"
#ifdef _MSC_VER
    _declspec(dllexport)
#else
    __attribute__((__visibility__("default")))
#endif
        int glslang_compile_1_2(glslang_CompileRequest_1_2* inRequest)
{
    static ProcessInitializer g_processInitializer;
    if (!g_processInitializer.init())
    {
        // Failed
        return 1;
    }

    // If it's the right size just use it
    if (inRequest->sizeInBytes == sizeof(glslang_CompileRequest_1_2))
    {
        return _compile(*inRequest);
    }
    else
    {
        // NOTE! It could be larger, but here we'll assume thats ok, and copy and use.

        // Try to ensure some binary compatibility, by using sizeInBytes member, and copying

        glslang_CompileRequest_1_2 request;

        // Copy into request
        const size_t copySize =
            (inRequest->sizeInBytes > sizeof(request)) ? sizeof(request) : inRequest->sizeInBytes;
        ::memcpy(&request, inRequest, copySize);
        // Zero any remaining members
        memset(((uint8_t*)&request) + copySize, 0, sizeof(request) - copySize);

        return _compile(request);
    }
}

extern "C"
#ifdef _MSC_VER
    _declspec(dllexport)
#else
    __attribute__((__visibility__("default")))
#endif
        int glslang_compile_1_1(glslang_CompileRequest_1_1* inRequest)
{
    glslang_CompileRequest_1_2 request;
    memset(&request, 0, sizeof(request));
    request.sizeInBytes = sizeof(request);
    request.set(*inRequest);
    return glslang_compile_1_2(&request);
}

extern "C"
#ifdef _MSC_VER
    _declspec(dllexport)
#else
    __attribute__((__visibility__("default")))
#endif
        int glslang_compile(glslang_CompileRequest_1_0* inRequest)
{
    glslang_CompileRequest_1_1 request;
    memset(&request, 0, sizeof(request));
    request.sizeInBytes = sizeof(request);
    request.set(*inRequest);
    return glslang_compile_1_1(&request);
}

extern "C"
#ifdef _MSC_VER
    _declspec(dllexport)
#else
    __attribute__((__visibility__("default")))
#endif
        int glslang_linkSPIRV(glslang_LinkRequest* request)
{
    if (!request || !request->modules || request->linkResult)
        return false;

    try
    {
        spvtools::Context context(SPV_ENV_UNIVERSAL_1_5);
        spvtools::LinkerOptions options = {};

        options.SetUseHighestVersion(true);

        spvtools::MessageConsumer consumer = [](spv_message_level_t level,
                                                const char* source,
                                                const spv_position_t& position,
                                                const char* message)
        {
            printf("SPIRV-TOOLS: %s\n", message);
            printf("SPIRV-TOOLS: %s\n", source);
            printf("SPIRV-TOOLS: %zu:%zu\n", position.index, position.column);
        };
        context.SetMessageConsumer(consumer);

        std::vector<std::vector<uint32_t>> moduleVecs(request->moduleCount);
        std::vector<const uint32_t*> moduleData(request->moduleCount);
        std::vector<size_t> moduleSizes(request->moduleCount);

        for (size_t i = 0; i < request->moduleCount; ++i)
        {
            moduleData[i] = request->modules[i];
            moduleSizes[i] = request->moduleSizes[i];
        }

        std::vector<uint32_t> linkedBinary;
        spv_result_t success = spvtools::Link(
            context,
            moduleData.data(),
            moduleSizes.data(),
            request->moduleCount,
            &linkedBinary,
            options);

        if (success == SPV_SUCCESS)
        {
            request->linkResult = new uint32_t[linkedBinary.size()];
            memcpy(
                (void*)request->linkResult,
                linkedBinary.data(),
                linkedBinary.size() * sizeof(uint32_t));
            request->linkResultSize = linkedBinary.size();
        }

        return success == SPV_SUCCESS;
    }
    catch (...)
    {
        return false;
    }
}
