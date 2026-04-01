// options.cpp

#include "options.h"

#include "../../source/core/slang-list.h"
#include "../../source/core/slang-render-api-util.h"
#include "../../source/core/slang-string-util.h"
#include "../../source/core/slang-writer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// #include "../../source/core/slang-downstream-compiler.h"

#include "../../source/compiler-core/slang-command-line-args.h"
#include "../../source/core/slang-type-text-util.h"
#include "diagnostics.h"

namespace renderer_test
{
using namespace Slang;

static rhi::DeviceType _toRenderType(Slang::RenderApiType apiType)
{
    using namespace Slang;
    switch (apiType)
    {
    case RenderApiType::D3D11:
        return rhi::DeviceType::D3D11;
    case RenderApiType::D3D12:
        return rhi::DeviceType::D3D12;
    case RenderApiType::Vulkan:
        return rhi::DeviceType::Vulkan;
    case RenderApiType::Metal:
        return rhi::DeviceType::Metal;
    case RenderApiType::CPU:
        return rhi::DeviceType::CPU;
    case RenderApiType::CUDA:
        return rhi::DeviceType::CUDA;
    case RenderApiType::WebGPU:
        return rhi::DeviceType::WGPU;
    default:
        return rhi::DeviceType::Default;
    }
}

/* static */ SlangResult Options::parse(
    int argc,
    const char* const* argv,
    Slang::WriterHelper stdError,
    Options& outOptions)
{
    using namespace Slang;

    RefPtr<CommandLineContext> cmdLineContext(new CommandLineContext);

    DiagnosticSink sink(cmdLineContext->getSourceManager(), nullptr);
    sink.writer = stdError.getWriter();
    sink.setFlag(DiagnosticSink::Flag::SourceLocationLine);

    outOptions = Options();

    CommandLineArgs args(cmdLineContext);

    if (argc > 0)
    {
        // first argument is the application name
        outOptions.appName = argv[0];
        args.setArgs(argv + 1, argc - 1);
    }
    else
    {
        args.setArgs(argv, argc);
    }
    SLANG_RETURN_ON_FAIL(outOptions.downstreamArgs.stripDownstreamArgs(args, 0, &sink));

    CommandLineReader reader(&args, &sink);

    List<CommandLineArg> positionalArgs;

    typedef Options::ShaderProgramType ShaderProgramType;
    typedef Options::InputLanguageID InputLanguageID;

    // now iterate over arguments to collect options
    while (reader.hasArg())
    {
        CommandLineArg arg = reader.getArgAndAdvance();
        const auto& argValue = arg.value;

        if (!argValue.startsWith("-"))
        {
            positionalArgs.add(arg);
            continue;
        }

        if (argValue == "--")
        {
            while (reader.hasArg())
            {
                positionalArgs.add(reader.getArgAndAdvance());
            }
            break;
        }
        else if (argValue == "-o")
        {
            SLANG_RETURN_ON_FAIL(reader.expectArg(outOptions.outputPath));
        }
        else if (argValue == "-profile")
        {
            SLANG_RETURN_ON_FAIL(reader.expectArg(outOptions.profileName));
        }
        else if (argValue == "-render-features" || argValue == "-render-feature")
        {
            String features;
            SLANG_RETURN_ON_FAIL(reader.expectArg(features));

            List<UnownedStringSlice> values;
            StringUtil::split(features.getUnownedSlice(), ',', values);

            for (const auto& value : values)
            {
                outOptions.renderFeatures.add(value);
            }
        }
        else if (argValue == "-xslang" || argValue == "-compile-arg")
        {
            // This is legacy support, should use -Xslang now
            // This is an option that we want to pass along to Slang
            CommandLineArg slangArg;
            SLANG_RETURN_ON_FAIL(reader.expectArg(slangArg));
            outOptions.downstreamArgs.getArgsByName("slang").add(slangArg);
        }
        else if (argValue == "-compute")
        {
            outOptions.shaderType = ShaderProgramType::Compute;
        }
        else if (argValue == "-graphics")
        {
            outOptions.shaderType = ShaderProgramType::Graphics;
        }
        else if (argValue == "-gcompute")
        {
            outOptions.shaderType = ShaderProgramType::GraphicsCompute;
        }
        else if (argValue == "-rt")
        {
            outOptions.shaderType = ShaderProgramType::RayTracing;
        }
        else if (argValue == "-mesh")
        {
            outOptions.shaderType = ShaderProgramType::GraphicsMeshCompute;
        }
        else if (argValue == "-task")
        {
            outOptions.shaderType = ShaderProgramType::GraphicsTaskMeshCompute;
        }
        else if (argValue == "-use-dxil")
        {
            outOptions.useDXIL = true;
        }
        else if (argValue == "-skip-spirv-validation")
        {
            outOptions.skipSPIRVValidation = true;
        }
        else if (argValue == "-emit-spirv-directly")
        {
            outOptions.generateSPIRVDirectly = true;
        }
        else if (argValue == "-emit-spirv-via-glsl")
        {
            outOptions.generateSPIRVDirectly = false;
        }
        else if (argValue == "-only-startup")
        {
            outOptions.onlyStartup = true;
        }
        else if (argValue == "-performance-profile")
        {
            outOptions.performanceProfile = true;
        }
        else if (argValue == "-output-using-type")
        {
            outOptions.outputUsingType = true;
        }
        else if (argValue == "-compute-dispatch")
        {
            CommandLineArg dispatchSize;
            SLANG_RETURN_ON_FAIL(reader.expectArg(dispatchSize));

            List<UnownedStringSlice> slices;
            StringUtil::split(dispatchSize.value.getUnownedSlice(), ',', slices);
            if (slices.getCount() != 3)
            {
                sink.diagnose(
                    dispatchSize.loc,
                    RenderTestDiagnostics::expectingCommaComputeDispatch);
                return SLANG_FAIL;
            }

            String string;
            for (Index i = 0; i < 3; ++i)
            {
                string = slices[i];
                int v = stringToInt(string);
                if (v < 1)
                {
                    sink.diagnose(
                        dispatchSize.loc,
                        RenderTestDiagnostics::expectingPositiveComputeDispatch);
                    return SLANG_FAIL;
                }
                outOptions.computeDispatchSize[i] = v;
            }
        }
        else if (argValue == "-source-language")
        {
            CommandLineArg sourceLanguageName;
            SLANG_RETURN_ON_FAIL(reader.expectArg(sourceLanguageName));

            const SlangSourceLanguage sourceLanguage =
                TypeTextUtil::findSourceLanguage(sourceLanguageName.value.getUnownedSlice());
            if (sourceLanguage == SLANG_SOURCE_LANGUAGE_UNKNOWN)
            {
                sink.diagnose(sourceLanguageName.loc, RenderTestDiagnostics::unknownSourceLanguage);
                return SLANG_FAIL;
            }

            outOptions.sourceLanguage = sourceLanguage;
        }
        else if (argValue == "-no-default-entry-point")
        {
            outOptions.dontAddDefaultEntryPoints = true;
        }
        else if (argValue == "-nvapi-slot")
        {
            SLANG_RETURN_ON_FAIL(reader.expectArg(outOptions.nvapiExtnSlot));
        }
        else if (argValue == "-shaderobj")
        {
            // Note: We ignore this option because it is always enabled now.
            //
            // TODO: At some point we could warn/error and deprecate this option.
        }
        else if (argValue == "-g0")
        {
            outOptions.disableDebugInfo = true;
        }
        else if (argValue == "-allow-glsl")
        {
            outOptions.allowGLSL = true;
        }
        else if (argValue == "-entry")
        {
            SLANG_RETURN_ON_FAIL(reader.expectArg(outOptions.entryPointName));
        }
        else if (argValue == "-enable-debug-layers")
        {
            outOptions.enableDebugLayers = true;
        }
        else if (argValue == "-dx12-experimental")
        {
            outOptions.dx12Experimental = true;
        }
        else if (argValue == "-show-adapter-info")
        {
            outOptions.showAdapterInfo = true;
        }
        else
        {
            // Lookup
            Slang::UnownedStringSlice argSlice = arg.value.getUnownedSlice();
            if (argSlice.getLength() && argSlice[0] == '-')
            {
                // Look up the rendering API if set
                UnownedStringSlice argName = argSlice.tail(1);
                DeviceType deviceType = _toRenderType(RenderApiUtil::findApiTypeByName(argName));

                if (deviceType != DeviceType::Default)
                {
                    outOptions.deviceType = deviceType;
                    continue;
                }

                // Lookup the target language type
                DeviceType targetLanguageDeviceType =
                    _toRenderType(RenderApiUtil::findImplicitLanguageRenderApiType(argName));

                if (targetLanguageDeviceType != DeviceType::Default || argName == "glsl")
                {
                    outOptions.targetLanguageDeviceType = targetLanguageDeviceType;
                    outOptions.inputLanguageID =
                        (argName == "hlsl" || argName == "glsl" || argName == "cpp" ||
                         argName == "cxx" || argName == "c")
                            ? InputLanguageID::Native
                            : InputLanguageID::Slang;
                    continue;
                }
            }
            sink.diagnose(arg.loc, RenderTestDiagnostics::unknownCommandLineOption, arg.value);
            return SLANG_FAIL;
        }
    }

    // If a render option isn't set use defaultRenderType
    outOptions.deviceType = (outOptions.deviceType == DeviceType::Default)
                                ? outOptions.targetLanguageDeviceType
                                : outOptions.deviceType;

    // first positional argument is source shader path
    if (positionalArgs.getCount())
    {
        outOptions.sourcePath = positionalArgs[0].value;
        positionalArgs.removeAt(0);
    }

    // any remaining arguments represent an error
    if (positionalArgs.getCount() != 0)
    {
        sink.diagnose(positionalArgs[0].loc, RenderTestDiagnostics::unexpectedPositionalArg);
        return SLANG_FAIL;
    }

    return SLANG_OK;
}

} // namespace renderer_test
