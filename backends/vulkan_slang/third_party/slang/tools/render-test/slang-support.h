// slang-support.h
#pragma once

#include "options.h"
#include "shader-input-layout.h"
#include "slang.h"

#include <slang-rhi.h>

namespace renderer_test
{

struct ShaderCompileRequest
{
    struct SourceInfo
    {
        char const* path;

        // The data may either be source text (in which
        // case it can be assumed to be nul-terminated with
        // `dataEnd` pointing at the terminator), or
        // raw binary data (in which case `dataEnd` points
        // at the end of the buffer).
        char const* dataBegin;
        char const* dataEnd;
    };

    struct EntryPoint
    {
        char const* name = nullptr;
        SlangStage slangStage;
    };

    struct TypeConformance
    {
    public:
        Slang::String derivedTypeName;
        Slang::String baseTypeName;
        Slang::Int idOverride;
    };

    SourceInfo source;
    Slang::List<EntryPoint> entryPoints;

    Slang::List<Slang::String> globalSpecializationArgs;
    Slang::List<Slang::String> entryPointSpecializationArgs;
    Slang::List<TypeConformance> typeConformances;
};


struct ShaderCompilerUtil
{
    struct Input
    {
        SlangCompileTarget target;
        SlangSourceLanguage sourceLanguage;
        SlangPassThrough passThrough;
        Slang::String profile;
    };

    struct Output
    {
        void set(slang::IComponentType* slangProgram);
        void reset();
        ~Output() { reset(); }

        ComPtr<slang::IComponentType> slangProgram;
        ShaderProgramDesc desc = {};

        ComPtr<slang::ISession> m_session = nullptr;

        slang::IGlobalSession* globalSession = nullptr;
    };

    struct OutputAndLayout
    {
        Output output;
        ShaderInputLayout layout;
        Slang::String sourcePath;
    };

    // Wrapper for compileProgram
    static SlangResult compileWithLayout(
        slang::IGlobalSession* globalSession,
        const Options& options,
        const ShaderCompilerUtil::Input& input,
        OutputAndLayout& output);
};


} // namespace renderer_test
