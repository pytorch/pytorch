
#include "slang-type-text-util.h"

#include "slang-array-view.h"
#include "slang-string-util.h"

namespace Slang
{

namespace
{ // anonymous

// clang-format off
#define SLANG_SCALAR_TYPES(x) \
    x(None, none) \
    x(Void, void) \
    x(Bool, bool) \
    x(Float16, half) \
    x(UInt32, uint32_t) \
    x(Int32, int32_t) \
    x(Int64, int64_t) \
    x(UInt64, uint64_t) \
    x(Float32, float) \
    x(Float64, double)
// clang-format on

struct ScalarTypeInfo
{
    slang::TypeReflection::ScalarType type;
    UnownedStringSlice text;
};

static const ScalarTypeInfo s_scalarTypeInfos[] = {
#define SLANG_SCALAR_TYPE_INFO(value, text) \
    {slang::TypeReflection::ScalarType::value, UnownedStringSlice::fromLiteral(#text)},
    SLANG_SCALAR_TYPES(SLANG_SCALAR_TYPE_INFO)};

// Make sure to keep this table in sync with that in slang/slang-options.cpp getHelpText
static const TypeTextUtil::CompileTargetInfo s_compileTargetInfos[] = {
    {SLANG_TARGET_UNKNOWN, "", "unknown"},
    {SLANG_TARGET_NONE, "", "none"},
    {SLANG_HLSL, "hlsl,fx", "hlsl", "HLSL source code"},
    {SLANG_DXBC, "dxbc", "dxbc", "DirectX shader bytecode binary"},
    {SLANG_DXBC_ASM, "dxbc-asm", "dxbc-asm,dxbc-assembly", "DirectX shader bytecode assembly"},
    {SLANG_DXIL, "dxil", "dxil", "DirectX Intermediate Language binary"},
    {SLANG_DXIL_ASM,
     "dxil-asm",
     "dxil-asm,dxil-assembly",
     "DirectX Intermediate Language assembly"},
    {SLANG_GLSL, "glsl,vert,frag,geom,tesc,tese,comp", "glsl", "GLSL(Vulkan) source code"},
    {SLANG_SPIRV, "spv", "spirv", "SPIR-V binary"},
    {SLANG_SPIRV_ASM, "spv-asm", "spirv-asm,spirv-assembly", "SPIR-V assembly"},
    {SLANG_C_SOURCE, "c", "c", "C source code"},
    {SLANG_CPP_SOURCE, "cpp,c++,cxx", "cpp,c++,cxx", "C++ source code"},
    {SLANG_CPP_PYTORCH_BINDING,
     "cpp,c++,cxx",
     "torch,torch-binding,torch-cpp,torch-cpp-binding",
     "C++ for pytorch binding"},
    {SLANG_HOST_CPP_SOURCE,
     "cpp,c++,cxx",
     "host-cpp,host-c++,host-cxx",
     "C++ source for host execution"},
    {SLANG_HOST_EXECUTABLE, "exe", "exe,executable", "Executable binary"},
    {SLANG_SHADER_SHARED_LIBRARY,
     "shader-dll,shader-so",
     "shader-sharedlib,shader-sharedlibrary,shader-dll",
     "Shared library/Dll for shader kernel"},
    {SLANG_HOST_SHARED_LIBRARY,
     "dll,so",
     "sharedlib,sharedlibrary,dll",
     "Shared library/Dll for host execution"},
    {SLANG_CUDA_SOURCE, "cu", "cuda,cu", "CUDA source code"},
    {SLANG_PTX, "ptx", "ptx", "PTX assembly"},
    {SLANG_CUDA_OBJECT_CODE, "obj,o", "cuobj,cubin", "CUDA binary"},
    {SLANG_SHADER_HOST_CALLABLE, "", "host-callable,callable", "Host callable"},
    {SLANG_OBJECT_CODE, "obj,o", "object-code", "Object code"},
    {SLANG_HOST_HOST_CALLABLE, "", "host-host-callable", "Host callable for host execution"},
    {SLANG_METAL, "metal", "metal", "Metal shader source"},
    {SLANG_METAL_LIB, "metallib", "metallib", "Metal Library Bytecode"},
    {SLANG_METAL_LIB_ASM,
     "metallib-asm"
     "metallib-asm",
     "Metal Library Bytecode assembly"},
    {SLANG_WGSL, "wgsl", "wgsl", "WebGPU shading language source"},
    {SLANG_WGSL_SPIRV_ASM,
     "wgsl-spirv-asm",
     "wgsl-spirv-asm,wgsl-spirv-assembly",
     "SPIR-V assembly via WebGPU shading language"},
    {SLANG_WGSL_SPIRV, "wgsl-spirv", "wgsl-spirv", "SPIR-V via WebGPU shading language"},
    {SLANG_HOST_VM, "slang-vm", "slangvm,slang-vm", "Slang VM byte code"},
};

static const NamesDescriptionValue s_languageInfos[] = {
    {SLANG_SOURCE_LANGUAGE_C, "c,C", "C language"},
    {SLANG_SOURCE_LANGUAGE_CPP, "cpp,c++,C++,cxx", "C++ language"},
    {SLANG_SOURCE_LANGUAGE_SLANG, "slang", "Slang language"},
    {SLANG_SOURCE_LANGUAGE_GLSL, "glsl", "GLSL language"},
    {SLANG_SOURCE_LANGUAGE_HLSL, "hlsl", "HLSL language"},
    {SLANG_SOURCE_LANGUAGE_CUDA, "cu,cuda", "CUDA"},
};

static const NamesDescriptionValue s_compilerInfos[] = {
    {SLANG_PASS_THROUGH_NONE, "none", "Unknown"},
    {SLANG_PASS_THROUGH_FXC, "fxc", "FXC HLSL compiler"},
    {SLANG_PASS_THROUGH_DXC, "dxc", "DXC HLSL compiler"},
    {SLANG_PASS_THROUGH_GLSLANG, "glslang", "GLSLANG GLSL compiler"},
    {SLANG_PASS_THROUGH_SPIRV_DIS, "spirv-dis", "spirv-tools SPIRV disassembler"},
    {SLANG_PASS_THROUGH_CLANG, "clang", "Clang C/C++ compiler"},
    {SLANG_PASS_THROUGH_VISUAL_STUDIO, "visualstudio,vs", "Visual Studio C/C++ compiler"},
    {SLANG_PASS_THROUGH_GCC, "gcc", "GCC C/C++ compiler"},
    {SLANG_PASS_THROUGH_GENERIC_C_CPP,
     "genericcpp,c,cpp",
     "A generic C++ compiler (can be any one of visual studio, clang or gcc depending on system "
     "and availability)"},
    {SLANG_PASS_THROUGH_NVRTC, "nvrtc", "NVRTC CUDA compiler"},
    {SLANG_PASS_THROUGH_LLVM, "llvm", "LLVM/Clang `slang-llvm`"},
    {SLANG_PASS_THROUGH_SPIRV_OPT, "spirv-opt", "spirv-tools SPIRV optimizer"},
    {SLANG_PASS_THROUGH_METAL, "metal", "Metal shader compiler"},
    {SLANG_PASS_THROUGH_TINT, "tint", "Tint compiler"},
};

static const NamesDescriptionValue s_archiveTypeInfos[] = {
    {SLANG_ARCHIVE_TYPE_RIFF_DEFLATE, "riff-deflate", "Slang RIFF using deflate compression"},
    {SLANG_ARCHIVE_TYPE_RIFF_LZ4, "riff-lz4", "Slang RIFF using LZ4 compression"},
    {SLANG_ARCHIVE_TYPE_ZIP, "zip", "Zip file"},
    {SLANG_ARCHIVE_TYPE_RIFF, "riff", "Slang RIFF without compression"},
};

static const NamesDescriptionValue s_debugInfoFormatInfos[] = {
    {SLANG_DEBUG_INFO_FORMAT_DEFAULT,
     "default-format",
     "Use the default debugging format for the target"},
    {SLANG_DEBUG_INFO_FORMAT_C7,
     "c7",
     "CodeView C7 format (typically means debugging infomation is embedded in the binary)"},
    {SLANG_DEBUG_INFO_FORMAT_PDB, "pdb", "Program database"},
    {SLANG_DEBUG_INFO_FORMAT_STABS, "stabs", "STABS debug format"},
    {SLANG_DEBUG_INFO_FORMAT_COFF, "coff", "COFF debug format"},
    {SLANG_DEBUG_INFO_FORMAT_DWARF, "dwarf", "DWARF debug format"},
};

static const NamesDescriptionValue s_lineDirectiveInfos[] = {
    {SLANG_LINE_DIRECTIVE_MODE_NONE, "none", "Don't emit `#line` directives at all"},
    {SLANG_LINE_DIRECTIVE_MODE_SOURCE_MAP,
     "source-map",
     "Use source map to track line associations (doen't emit #line)"},
    {SLANG_LINE_DIRECTIVE_MODE_DEFAULT, "default", "Default behavior"},
    {SLANG_LINE_DIRECTIVE_MODE_STANDARD, "standard", "Emit standard C-style `#line` directives."},
    {SLANG_LINE_DIRECTIVE_MODE_GLSL,
     "glsl",
     "Emit GLSL-style directives with file *number* instead of name."},
};

static const NamesDescriptionValue s_floatingPointModes[] = {
    {SLANG_FLOATING_POINT_MODE_PRECISE,
     "precise",
     "Disable optimization that could change the output of floating-"
     "point computations, including around infinities, NaNs, denormalized "
     "values, and negative zero. Prefer the most precise versions of special "
     "functions supported by the target."},
    {SLANG_FLOATING_POINT_MODE_FAST,
     "fast",
     "Allow optimizations that may change results of floating-point "
     "computations. Prefer the fastest version of special functions supported "
     "by the target."},
    {SLANG_FLOATING_POINT_MODE_DEFAULT, "default", "Default floating point mode"}};

static const NamesDescriptionValue s_optimizationLevels[] = {
    {SLANG_OPTIMIZATION_LEVEL_NONE, "0,none", "Disable all optimizations"},
    {SLANG_OPTIMIZATION_LEVEL_DEFAULT,
     "1,default",
     "Enable a default level of optimization.This is the default if no -o options are used."},
    {SLANG_OPTIMIZATION_LEVEL_HIGH, "2,high", "Enable aggressive optimizations for speed."},
    {SLANG_OPTIMIZATION_LEVEL_MAXIMAL,
     "3,maximal",
     "Enable further optimizations, which might have a significant impact on compile time, or "
     "involve unwanted tradeoffs in terms of code size."},
};

static const NamesDescriptionValue s_debugLevels[] = {
    {SLANG_DEBUG_INFO_LEVEL_NONE, "0,none", "Don't emit debug information at all."},
    {SLANG_DEBUG_INFO_LEVEL_MINIMAL,
     "1,minimal",
     "Emit as little debug information as possible, while still supporting stack traces."},
    {SLANG_DEBUG_INFO_LEVEL_STANDARD,
     "2,standard",
     "Emit whatever is the standard level of debug information for each target."},
    {SLANG_DEBUG_INFO_LEVEL_MAXIMAL,
     "3,maximal",
     "Emit as much debug information as possible for each target."},
};

static const NamesDescriptionValue s_fileSystemTypes[] = {
    {ValueInt(TypeTextUtil::FileSystemType::Default), "default", "Default file system."},
    {ValueInt(TypeTextUtil::FileSystemType::LoadFile),
     "load-file",
     "Just implements loadFile interface, so will be wrapped with CacheFileSystem internally."},
    {ValueInt(TypeTextUtil::FileSystemType::Os),
     "os",
     "Use the OS based file system directly (without file system caching)"},
};

} // namespace

/* static */ ConstArrayView<NamesDescriptionValue> TypeTextUtil::getFileSystemTypeInfos()
{
    return makeConstArrayView(s_fileSystemTypes);
}

/* static */ ConstArrayView<TypeTextUtil::CompileTargetInfo> TypeTextUtil::getCompileTargetInfos()
{
    return makeConstArrayView(s_compileTargetInfos);
}

/* static */ ConstArrayView<NamesDescriptionValue> TypeTextUtil::getLanguageInfos()
{
    return makeConstArrayView(s_languageInfos);
}

/* static */ ConstArrayView<NamesDescriptionValue> TypeTextUtil::getCompilerInfos()
{
    return makeConstArrayView(s_compilerInfos);
}

/* static */ ConstArrayView<NamesDescriptionValue> TypeTextUtil::getArchiveTypeInfos()
{
    return makeConstArrayView(s_archiveTypeInfos);
}

/* static */ ConstArrayView<NamesDescriptionValue> TypeTextUtil::getDebugInfoFormatInfos()
{
    return makeConstArrayView(s_debugInfoFormatInfos);
}

/* static */ ConstArrayView<NamesDescriptionValue> TypeTextUtil::getLineDirectiveInfos()
{
    return makeConstArrayView(s_lineDirectiveInfos);
}

/* static */ ConstArrayView<NamesDescriptionValue> TypeTextUtil::getFloatingPointModeInfos()
{
    return makeConstArrayView(s_floatingPointModes);
}

/* static */ ConstArrayView<NamesDescriptionValue> TypeTextUtil::getOptimizationLevelInfos()
{
    return makeConstArrayView(s_optimizationLevels);
}

/* static */ ConstArrayView<NamesDescriptionValue> TypeTextUtil::getDebugLevelInfos()
{
    return makeConstArrayView(s_debugLevels);
}

/* static */ SlangArchiveType TypeTextUtil::findArchiveType(const UnownedStringSlice& slice)
{
    return NameValueUtil::findValue(getArchiveTypeInfos(), slice, SLANG_ARCHIVE_TYPE_UNDEFINED);
}

/* static */ SlangResult TypeTextUtil::findDebugInfoFormat(
    const Slang::UnownedStringSlice& text,
    SlangDebugInfoFormat& out)
{
    const ValueInt value = NameValueUtil::findValue(getDebugInfoFormatInfos(), text, -1);
    if (value >= 0)
    {
        out = SlangDebugInfoFormat(value);
        return SLANG_OK;
    }
    return SLANG_FAIL;
}

/* static */ UnownedStringSlice TypeTextUtil::getDebugInfoFormatName(SlangDebugInfoFormat format)
{
    return NameValueUtil::findName(getDebugInfoFormatInfos(), format, toSlice("unknown"));
}

/* static */ UnownedStringSlice TypeTextUtil::getScalarTypeName(
    slang::TypeReflection::ScalarType scalarType)
{
    typedef slang::TypeReflection::ScalarType ScalarType;
    switch (scalarType)
    {
#define SLANG_SCALAR_TYPE_TO_TEXT(value, text) \
    case ScalarType::value:                    \
        return UnownedStringSlice::fromLiteral(#text);
        SLANG_SCALAR_TYPES(SLANG_SCALAR_TYPE_TO_TEXT)
    default:
        break;
    }

    return UnownedStringSlice();
}

/* static */ slang::TypeReflection::ScalarType TypeTextUtil::findScalarType(
    const UnownedStringSlice& inText)
{
    for (Index i = 0; i < SLANG_COUNT_OF(s_scalarTypeInfos); ++i)
    {
        const auto& info = s_scalarTypeInfos[i];
        if (info.text == inText)
        {
            return info.type;
        }
    }
    return slang::TypeReflection::ScalarType::None;
}


/* static */ UnownedStringSlice TypeTextUtil::getPassThroughAsHumanText(SlangPassThrough type)
{
    return NameValueUtil::findName(getCompilerInfos(), type, toSlice("unknown"));
}

/* static */ SlangSourceLanguage TypeTextUtil::findSourceLanguage(const UnownedStringSlice& text)
{
    return NameValueUtil::findValue(getLanguageInfos(), text, SLANG_SOURCE_LANGUAGE_UNKNOWN);
}

/* static */ SlangPassThrough TypeTextUtil::findPassThrough(const UnownedStringSlice& slice)
{
    return NameValueUtil::findValue(getCompilerInfos(), slice, SLANG_PASS_THROUGH_NONE);
}

/* static */ SlangResult TypeTextUtil::findPassThrough(
    const UnownedStringSlice& slice,
    SlangPassThrough& outPassThrough)
{
    outPassThrough = findPassThrough(slice);
    // It could be none on error - if it's not equal to "none" then it must be an error
    if (outPassThrough == SLANG_PASS_THROUGH_NONE &&
        slice != UnownedStringSlice::fromLiteral("none"))
    {
        return SLANG_FAIL;
    }
    return SLANG_OK;
}

/* static */ UnownedStringSlice TypeTextUtil::getPassThroughName(SlangPassThrough passThru)
{
    return NameValueUtil::findName(getCompilerInfos(), passThru, toSlice("unknown"));
}

/* static */ SlangCompileTarget TypeTextUtil::findCompileTargetFromExtension(
    const UnownedStringSlice& slice)
{
    if (slice.getLength())
    {
        for (const auto& info : s_compileTargetInfos)
        {
            if (StringUtil::indexOfInSplit(UnownedStringSlice(info.extensions), ',', slice) >= 0)
            {
                return info.target;
            }
        }
    }
    return SLANG_TARGET_UNKNOWN;
}

/* static */ SlangCompileTarget TypeTextUtil::findCompileTargetFromName(
    const UnownedStringSlice& slice)
{
    if (slice.getLength())
    {
        for (const auto& info : s_compileTargetInfos)
        {
            if (StringUtil::indexOfInSplit(UnownedStringSlice(info.names), ',', slice) >= 0)
            {
                return info.target;
            }
        }
    }
    return SLANG_TARGET_UNKNOWN;
}

static Index _getTargetInfoIndex(SlangCompileTarget target)
{
    for (Index i = 0; i < SLANG_COUNT_OF(s_compileTargetInfos); ++i)
    {
        if (s_compileTargetInfos[i].target == target)
        {
            return i;
        }
    }
    return -1;
}

UnownedStringSlice TypeTextUtil::getCompileTargetName(SlangCompileTarget target)
{
    const Index index = _getTargetInfoIndex(target);
    // Return the first name
    return index >= 0 ? StringUtil::getAtInSplit(
                            UnownedStringSlice(s_compileTargetInfos[index].names),
                            ',',
                            0)
                      : UnownedStringSlice();
}

} // namespace Slang
