#ifndef SLANG_H
#define SLANG_H

#ifdef SLANG_USER_CONFIG
    #include SLANG_USER_CONFIG
#endif

/** \file slang.h

The Slang API provides services to compile, reflect, and specialize code
written in the Slang shading language.
*/

/*
The following section attempts to detect the compiler and version in use.

If an application defines `SLANG_COMPILER` before including this header,
they take responsibility for setting any compiler-dependent macros
used later in the file.

Most applications should not need to touch this section.
*/
#ifndef SLANG_COMPILER
    #define SLANG_COMPILER

    /*
    Compiler defines, see http://sourceforge.net/p/predef/wiki/Compilers/
    NOTE that SLANG_VC holds the compiler version - not just 1 or 0
    */
    #if defined(_MSC_VER)
        #if _MSC_VER >= 1900
            #define SLANG_VC 14
        #elif _MSC_VER >= 1800
            #define SLANG_VC 12
        #elif _MSC_VER >= 1700
            #define SLANG_VC 11
        #elif _MSC_VER >= 1600
            #define SLANG_VC 10
        #elif _MSC_VER >= 1500
            #define SLANG_VC 9
        #else
            #error "unknown version of Visual C++ compiler"
        #endif
    #elif defined(__clang__)
        #define SLANG_CLANG 1
    #elif defined(__SNC__)
        #define SLANG_SNC 1
    #elif defined(__ghs__)
        #define SLANG_GHS 1
    #elif defined(__GNUC__) /* note: __clang__, __SNC__, or __ghs__ imply __GNUC__ */
        #define SLANG_GCC 1
    #else
        #error "unknown compiler"
    #endif
    /*
    Any compilers not detected by the above logic are now now explicitly zeroed out.
    */
    #ifndef SLANG_VC
        #define SLANG_VC 0
    #endif
    #ifndef SLANG_CLANG
        #define SLANG_CLANG 0
    #endif
    #ifndef SLANG_SNC
        #define SLANG_SNC 0
    #endif
    #ifndef SLANG_GHS
        #define SLANG_GHS 0
    #endif
    #ifndef SLANG_GCC
        #define SLANG_GCC 0
    #endif
#endif /* SLANG_COMPILER */

/*
The following section attempts to detect the target platform being compiled for.

If an application defines `SLANG_PLATFORM` before including this header,
they take responsibility for setting any compiler-dependent macros
used later in the file.

Most applications should not need to touch this section.
*/
#ifndef SLANG_PLATFORM
    #define SLANG_PLATFORM
    /**
    Operating system defines, see http://sourceforge.net/p/predef/wiki/OperatingSystems/
    */
    #if defined(WINAPI_FAMILY) && WINAPI_FAMILY == WINAPI_PARTITION_APP
        #define SLANG_WINRT 1 /* Windows Runtime, either on Windows RT or Windows 8 */
    #elif defined(XBOXONE)
        #define SLANG_XBOXONE 1
    #elif defined(_WIN64) /* note: XBOXONE implies _WIN64 */
        #define SLANG_WIN64 1
    #elif defined(_M_PPC)
        #define SLANG_X360 1
    #elif defined(_WIN32) /* note: _M_PPC implies _WIN32 */
        #define SLANG_WIN32 1
    #elif defined(__ANDROID__)
        #define SLANG_ANDROID 1
    #elif defined(__linux__) || defined(__CYGWIN__) /* note: __ANDROID__ implies __linux__ */
        #define SLANG_LINUX 1
    #elif defined(__APPLE__)
        #include "TargetConditionals.h"
        #if TARGET_OS_MAC
            #define SLANG_OSX 1
        #else
            #define SLANG_IOS 1
        #endif
    #elif defined(__CELLOS_LV2__)
        #define SLANG_PS3 1
    #elif defined(__ORBIS__)
        #define SLANG_PS4 1
    #elif defined(__SNC__) && defined(__arm__)
        #define SLANG_PSP2 1
    #elif defined(__ghs__)
        #define SLANG_WIIU 1
    #elif defined(__EMSCRIPTEN__)
        #define SLANG_WASM 1
    #else
        #error "unknown target platform"
    #endif
    /*
    Any platforms not detected by the above logic are now now explicitly zeroed out.
    */
    #ifndef SLANG_WINRT
        #define SLANG_WINRT 0
    #endif
    #ifndef SLANG_XBOXONE
        #define SLANG_XBOXONE 0
    #endif
    #ifndef SLANG_WIN64
        #define SLANG_WIN64 0
    #endif
    #ifndef SLANG_X360
        #define SLANG_X360 0
    #endif
    #ifndef SLANG_WIN32
        #define SLANG_WIN32 0
    #endif
    #ifndef SLANG_ANDROID
        #define SLANG_ANDROID 0
    #endif
    #ifndef SLANG_LINUX
        #define SLANG_LINUX 0
    #endif
    #ifndef SLANG_IOS
        #define SLANG_IOS 0
    #endif
    #ifndef SLANG_OSX
        #define SLANG_OSX 0
    #endif
    #ifndef SLANG_PS3
        #define SLANG_PS3 0
    #endif
    #ifndef SLANG_PS4
        #define SLANG_PS4 0
    #endif
    #ifndef SLANG_PSP2
        #define SLANG_PSP2 0
    #endif
    #ifndef SLANG_WIIU
        #define SLANG_WIIU 0
    #endif
#endif /* SLANG_PLATFORM */

/* Shorthands for "families" of compilers/platforms */
#define SLANG_GCC_FAMILY (SLANG_CLANG || SLANG_SNC || SLANG_GHS || SLANG_GCC)
#define SLANG_WINDOWS_FAMILY (SLANG_WINRT || SLANG_WIN32 || SLANG_WIN64)
#define SLANG_MICROSOFT_FAMILY (SLANG_XBOXONE || SLANG_X360 || SLANG_WINDOWS_FAMILY)
#define SLANG_LINUX_FAMILY (SLANG_LINUX || SLANG_ANDROID)
#define SLANG_APPLE_FAMILY (SLANG_IOS || SLANG_OSX) /* equivalent to #if __APPLE__ */
#define SLANG_UNIX_FAMILY \
    (SLANG_LINUX_FAMILY || SLANG_APPLE_FAMILY) /* shortcut for unix/posix platforms */

/* Macros concerning DirectX */
#if !defined(SLANG_CONFIG_DX_ON_VK) || !SLANG_CONFIG_DX_ON_VK
    #define SLANG_ENABLE_DXVK 0
    #define SLANG_ENABLE_VKD3D 0
#else
    #define SLANG_ENABLE_DXVK 1
    #define SLANG_ENABLE_VKD3D 1
#endif

#if SLANG_WINDOWS_FAMILY
    #define SLANG_ENABLE_DIRECTX 1
    #define SLANG_ENABLE_DXGI_DEBUG 1
    #define SLANG_ENABLE_DXBC_SUPPORT 1
    #define SLANG_ENABLE_PIX 1
#elif SLANG_LINUX_FAMILY
    #define SLANG_ENABLE_DIRECTX (SLANG_ENABLE_DXVK || SLANG_ENABLE_VKD3D)
    #define SLANG_ENABLE_DXGI_DEBUG 0
    #define SLANG_ENABLE_DXBC_SUPPORT 0
    #define SLANG_ENABLE_PIX 0
#else
    #define SLANG_ENABLE_DIRECTX 0
    #define SLANG_ENABLE_DXGI_DEBUG 0
    #define SLANG_ENABLE_DXBC_SUPPORT 0
    #define SLANG_ENABLE_PIX 0
#endif

/* Macro for declaring if a method is no throw. Should be set before the return parameter. */
#ifndef SLANG_NO_THROW
    #if SLANG_WINDOWS_FAMILY && !defined(SLANG_DISABLE_EXCEPTIONS)
        #define SLANG_NO_THROW __declspec(nothrow)
    #endif
#endif
#ifndef SLANG_NO_THROW
    #define SLANG_NO_THROW
#endif

/* The `SLANG_STDCALL` and `SLANG_MCALL` defines are used to set the calling
convention for interface methods.
*/
#ifndef SLANG_STDCALL
    #if SLANG_MICROSOFT_FAMILY
        #define SLANG_STDCALL __stdcall
    #else
        #define SLANG_STDCALL
    #endif
#endif
#ifndef SLANG_MCALL
    #define SLANG_MCALL SLANG_STDCALL
#endif


#if !defined(SLANG_STATIC) && !defined(SLANG_DYNAMIC)
    #define SLANG_DYNAMIC
#endif

#if defined(_MSC_VER)
    #define SLANG_DLL_EXPORT __declspec(dllexport)
#else
    #if SLANG_WINDOWS_FAMILY
        #define SLANG_DLL_EXPORT \
            __attribute__((dllexport)) __attribute__((__visibility__("default")))
    #else
        #define SLANG_DLL_EXPORT __attribute__((__visibility__("default")))
    #endif
#endif

#if defined(SLANG_DYNAMIC)
    #if defined(_MSC_VER)
        #ifdef SLANG_DYNAMIC_EXPORT
            #define SLANG_API SLANG_DLL_EXPORT
        #else
            #define SLANG_API __declspec(dllimport)
        #endif
    #else
        // TODO: need to consider compiler capabilities
        // #     ifdef SLANG_DYNAMIC_EXPORT
        #define SLANG_API SLANG_DLL_EXPORT
    // #     endif
    #endif
#endif

#ifndef SLANG_API
    #define SLANG_API
#endif

// GCC Specific
#if SLANG_GCC_FAMILY
    #define SLANG_NO_INLINE __attribute__((noinline))
    #define SLANG_FORCE_INLINE inline __attribute__((always_inline))
    #define SLANG_BREAKPOINT(id) __builtin_trap();
    #define SLANG_ALIGN_OF(T) __alignof__(T)
#endif // SLANG_GCC_FAMILY

#if SLANG_GCC_FAMILY || defined(__clang__)
    // Use the builtin directly so we don't need to have an include of stddef.h
    #define SLANG_OFFSET_OF(T, ELEMENT) __builtin_offsetof(T, ELEMENT)
#endif

#ifndef SLANG_OFFSET_OF
    #define SLANG_OFFSET_OF(T, ELEMENT) (size_t(&((T*)1)->ELEMENT) - 1)
#endif

// Microsoft VC specific
#if SLANG_VC
    #define SLANG_NO_INLINE __declspec(noinline)
    #define SLANG_FORCE_INLINE __forceinline
    #define SLANG_BREAKPOINT(id) __debugbreak();
    #define SLANG_ALIGN_OF(T) __alignof(T)

    #define SLANG_INT64(x) (x##i64)
    #define SLANG_UINT64(x) (x##ui64)
#endif // SLANG_MICROSOFT_FAMILY

#ifndef SLANG_FORCE_INLINE
    #define SLANG_FORCE_INLINE inline
#endif
#ifndef SLANG_NO_INLINE
    #define SLANG_NO_INLINE
#endif

#ifndef SLANG_COMPILE_TIME_ASSERT
    #define SLANG_COMPILE_TIME_ASSERT(x) static_assert(x)
#endif

#ifndef SLANG_BREAKPOINT
    // Make it crash with a write to 0!
    #define SLANG_BREAKPOINT(id) (*((int*)0) = int(id));
#endif

// Use for getting the amount of members of a standard C array.
// Use 0[x] here to catch the case where x has an overloaded subscript operator
#define SLANG_COUNT_OF(x) (SlangSSizeT(sizeof(x) / sizeof(0 [x])))
/// SLANG_INLINE exists to have a way to inline consistent with SLANG_ALWAYS_INLINE
#define SLANG_INLINE inline

// If explicitly disabled and not set, set to not available
#if !defined(SLANG_HAS_EXCEPTIONS) && defined(SLANG_DISABLE_EXCEPTIONS)
    #define SLANG_HAS_EXCEPTIONS 0
#endif

// If not set, the default is exceptions are available
#ifndef SLANG_HAS_EXCEPTIONS
    #define SLANG_HAS_EXCEPTIONS 1
#endif

// Other defines
#define SLANG_STRINGIZE_HELPER(X) #X
#define SLANG_STRINGIZE(X) SLANG_STRINGIZE_HELPER(X)

#define SLANG_CONCAT_HELPER(X, Y) X##Y
#define SLANG_CONCAT(X, Y) SLANG_CONCAT_HELPER(X, Y)

#ifndef SLANG_UNUSED
    #define SLANG_UNUSED(v) (void)v;
#endif

#if defined(__llvm__)
    #define SLANG_MAYBE_UNUSED [[maybe_unused]]
#else
    #define SLANG_MAYBE_UNUSED
#endif

// Used for doing constant literals
#ifndef SLANG_INT64
    #define SLANG_INT64(x) (x##ll)
#endif
#ifndef SLANG_UINT64
    #define SLANG_UINT64(x) (x##ull)
#endif


#ifdef __cplusplus
    #define SLANG_EXTERN_C extern "C"
#else
    #define SLANG_EXTERN_C
#endif

#ifdef __cplusplus
    // C++ specific macros
    // Clang
    #if SLANG_CLANG
        #if (__clang_major__ * 10 + __clang_minor__) >= 33
            #define SLANG_HAS_MOVE_SEMANTICS 1
            #define SLANG_HAS_ENUM_CLASS 1
            #define SLANG_OVERRIDE override
        #endif

    // Gcc
    #elif SLANG_GCC_FAMILY
        // Check for C++11
        #if (__cplusplus >= 201103L)
            #if (__GNUC__ * 100 + __GNUC_MINOR__) >= 405
                #define SLANG_HAS_MOVE_SEMANTICS 1
            #endif
            #if (__GNUC__ * 100 + __GNUC_MINOR__) >= 406
                #define SLANG_HAS_ENUM_CLASS 1
            #endif
            #if (__GNUC__ * 100 + __GNUC_MINOR__) >= 407
                #define SLANG_OVERRIDE override
            #endif
        #endif
    #endif // SLANG_GCC_FAMILY

    // Visual Studio

    #if SLANG_VC
        // C4481: nonstandard extension used: override specifier 'override'
        #if _MSC_VER < 1700
            #pragma warning(disable : 4481)
        #endif
        #define SLANG_OVERRIDE override
        #if _MSC_VER >= 1600
            #define SLANG_HAS_MOVE_SEMANTICS 1
        #endif
        #if _MSC_VER >= 1700
            #define SLANG_HAS_ENUM_CLASS 1
        #endif
    #endif // SLANG_VC

    // Set non set
    #ifndef SLANG_OVERRIDE
        #define SLANG_OVERRIDE
    #endif
    #ifndef SLANG_HAS_ENUM_CLASS
        #define SLANG_HAS_ENUM_CLASS 0
    #endif
    #ifndef SLANG_HAS_MOVE_SEMANTICS
        #define SLANG_HAS_MOVE_SEMANTICS 0
    #endif

#endif // __cplusplus

/* Macros for detecting processor */
#if defined(_M_ARM) || defined(__ARM_EABI__)
    // This is special case for nVidia tegra
    #define SLANG_PROCESSOR_ARM 1
#elif defined(__i386__) || defined(_M_IX86)
    #define SLANG_PROCESSOR_X86 1
#elif defined(_M_AMD64) || defined(_M_X64) || defined(__amd64) || defined(__x86_64)
    #define SLANG_PROCESSOR_X86_64 1
#elif defined(_PPC_) || defined(__ppc__) || defined(__POWERPC__) || defined(_M_PPC)
    #if defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__) || \
        defined(__64BIT__) || defined(_LP64) || defined(__LP64__)
        #define SLANG_PROCESSOR_POWER_PC_64 1
    #else
        #define SLANG_PROCESSOR_POWER_PC 1
    #endif
#elif defined(__arm__)
    #define SLANG_PROCESSOR_ARM 1
#elif defined(_M_ARM64) || defined(__aarch64__)
    #define SLANG_PROCESSOR_ARM_64 1
#elif defined(__EMSCRIPTEN__)
    #define SLANG_PROCESSOR_WASM 1
#endif

#ifndef SLANG_PROCESSOR_ARM
    #define SLANG_PROCESSOR_ARM 0
#endif

#ifndef SLANG_PROCESSOR_ARM_64
    #define SLANG_PROCESSOR_ARM_64 0
#endif

#ifndef SLANG_PROCESSOR_X86
    #define SLANG_PROCESSOR_X86 0
#endif

#ifndef SLANG_PROCESSOR_X86_64
    #define SLANG_PROCESSOR_X86_64 0
#endif

#ifndef SLANG_PROCESSOR_POWER_PC
    #define SLANG_PROCESSOR_POWER_PC 0
#endif

#ifndef SLANG_PROCESSOR_POWER_PC_64
    #define SLANG_PROCESSOR_POWER_PC_64 0
#endif

// Processor families

#define SLANG_PROCESSOR_FAMILY_X86 (SLANG_PROCESSOR_X86_64 | SLANG_PROCESSOR_X86)
#define SLANG_PROCESSOR_FAMILY_ARM (SLANG_PROCESSOR_ARM | SLANG_PROCESSOR_ARM_64)
#define SLANG_PROCESSOR_FAMILY_POWER_PC (SLANG_PROCESSOR_POWER_PC_64 | SLANG_PROCESSOR_POWER_PC)

// Pointer size
#define SLANG_PTR_IS_64 \
    (SLANG_PROCESSOR_ARM_64 | SLANG_PROCESSOR_X86_64 | SLANG_PROCESSOR_POWER_PC_64)
#define SLANG_PTR_IS_32 (SLANG_PTR_IS_64 ^ 1)

// Processor features
#if SLANG_PROCESSOR_FAMILY_X86
    #define SLANG_LITTLE_ENDIAN 1
    #define SLANG_UNALIGNED_ACCESS 1
#elif SLANG_PROCESSOR_FAMILY_ARM
    #if defined(__ARMEB__)
        #define SLANG_BIG_ENDIAN 1
    #else
        #define SLANG_LITTLE_ENDIAN 1
    #endif
#elif SLANG_PROCESSOR_FAMILY_POWER_PC
    #define SLANG_BIG_ENDIAN 1
#elif SLANG_WASM
    #define SLANG_LITTLE_ENDIAN 1
#endif

#ifndef SLANG_LITTLE_ENDIAN
    #define SLANG_LITTLE_ENDIAN 0
#endif

#ifndef SLANG_BIG_ENDIAN
    #define SLANG_BIG_ENDIAN 0
#endif

#ifndef SLANG_UNALIGNED_ACCESS
    #define SLANG_UNALIGNED_ACCESS 0
#endif

// One endianness must be set
#if ((SLANG_BIG_ENDIAN | SLANG_LITTLE_ENDIAN) == 0)
    #error "Couldn't determine endianness"
#endif

#ifndef SLANG_NO_INTTYPES
    #include <inttypes.h>
#endif // ! SLANG_NO_INTTYPES

#ifndef SLANG_NO_STDDEF
    #include <stddef.h>
#endif // ! SLANG_NO_STDDEF

#ifdef __cplusplus
extern "C"
{
#endif
    /*!
    @mainpage Introduction

    API Reference: slang.h

    @file slang.h
    */

    typedef uint32_t SlangUInt32;
    typedef int32_t SlangInt32;

    // Use SLANG_PTR_ macros to determine SlangInt/SlangUInt types.
    // This is used over say using size_t/ptrdiff_t/intptr_t/uintptr_t, because on some targets,
    // these types are distinct from their uint_t/int_t equivalents and so produce ambiguity with
    // function overloading.
    //
    // SlangSizeT is helpful as on some compilers size_t is distinct from a regular integer type and
    // so overloading doesn't work. Casting to SlangSizeT works around this.
#if SLANG_PTR_IS_64
    typedef int64_t SlangInt;
    typedef uint64_t SlangUInt;

    typedef int64_t SlangSSizeT;
    typedef uint64_t SlangSizeT;
#else
typedef int32_t SlangInt;
typedef uint32_t SlangUInt;

typedef int32_t SlangSSizeT;
typedef uint32_t SlangSizeT;
#endif

    typedef bool SlangBool;


    /*!
    @brief Severity of a diagnostic generated by the compiler.
    Values come from the enum below, with higher values representing more severe
    conditions, and all values >= SLANG_SEVERITY_ERROR indicating compilation
    failure.
    */
    typedef int SlangSeverityIntegral;
    enum SlangSeverity : SlangSeverityIntegral
    {
        SLANG_SEVERITY_DISABLED = 0, /**< A message that is disabled, filtered out. */
        SLANG_SEVERITY_NOTE,         /**< An informative message. */
        SLANG_SEVERITY_WARNING,      /**< A warning, which indicates a possible problem. */
        SLANG_SEVERITY_ERROR,        /**< An error, indicating that compilation failed. */
        SLANG_SEVERITY_FATAL,    /**< An unrecoverable error, which forced compilation to abort. */
        SLANG_SEVERITY_INTERNAL, /**< An internal error, indicating a logic error in the compiler.
                                  */
    };

    typedef int SlangDiagnosticFlags;
    enum
    {
        SLANG_DIAGNOSTIC_FLAG_VERBOSE_PATHS = 0x01,
        SLANG_DIAGNOSTIC_FLAG_TREAT_WARNINGS_AS_ERRORS = 0x02
    };

    typedef int SlangBindableResourceIntegral;
    enum SlangBindableResourceType : SlangBindableResourceIntegral
    {
        SLANG_NON_BINDABLE = 0,
        SLANG_TEXTURE,
        SLANG_SAMPLER,
        SLANG_UNIFORM_BUFFER,
        SLANG_STORAGE_BUFFER,
    };

    /* NOTE! To keep binary compatibility care is needed with this enum!

    * To add value, only add at the bottom (before COUNT_OF)
    * To remove a value, add _DEPRECATED as a suffix, but leave in the list

    This will make the enum values stable, and compatible with libraries that might not use the
    latest enum values.
    */
    typedef int SlangCompileTargetIntegral;
    enum SlangCompileTarget : SlangCompileTargetIntegral
    {
        SLANG_TARGET_UNKNOWN,
        SLANG_TARGET_NONE,
        SLANG_GLSL,
        SLANG_GLSL_VULKAN_DEPRECATED,          //< deprecated and removed: just use `SLANG_GLSL`.
        SLANG_GLSL_VULKAN_ONE_DESC_DEPRECATED, //< deprecated and removed.
        SLANG_HLSL,
        SLANG_SPIRV,
        SLANG_SPIRV_ASM,
        SLANG_DXBC,
        SLANG_DXBC_ASM,
        SLANG_DXIL,
        SLANG_DXIL_ASM,
        SLANG_C_SOURCE,              ///< The C language
        SLANG_CPP_SOURCE,            ///< C++ code for shader kernels.
        SLANG_HOST_EXECUTABLE,       ///< Standalone binary executable (for hosting CPU/OS)
        SLANG_SHADER_SHARED_LIBRARY, ///< A shared library/Dll for shader kernels (for hosting
                                     ///< CPU/OS)
        SLANG_SHADER_HOST_CALLABLE,  ///< A CPU target that makes the compiled shader code available
                                     ///< to be run immediately
        SLANG_CUDA_SOURCE,           ///< Cuda source
        SLANG_PTX,                   ///< PTX
        SLANG_CUDA_OBJECT_CODE,      ///< Object code that contains CUDA functions.
        SLANG_OBJECT_CODE,           ///< Object code that can be used for later linking
        SLANG_HOST_CPP_SOURCE,       ///< C++ code for host library or executable.
        SLANG_HOST_HOST_CALLABLE,    ///< Host callable host code (ie non kernel/shader)
        SLANG_CPP_PYTORCH_BINDING,   ///< C++ PyTorch binding code.
        SLANG_METAL,                 ///< Metal shading language
        SLANG_METAL_LIB,             ///< Metal library
        SLANG_METAL_LIB_ASM,         ///< Metal library assembly
        SLANG_HOST_SHARED_LIBRARY,   ///< A shared library/Dll for host code (for hosting CPU/OS)
        SLANG_WGSL,                  ///< WebGPU shading language
        SLANG_WGSL_SPIRV_ASM,        ///< SPIR-V assembly via WebGPU shading language
        SLANG_WGSL_SPIRV,            ///< SPIR-V via WebGPU shading language

        SLANG_HOST_VM, ///< Bytecode that can be interpreted by the Slang VM
        SLANG_TARGET_COUNT_OF,
    };

    /* A "container format" describes the way that the outputs
    for multiple files, entry points, targets, etc. should be
    combined into a single artifact for output. */
    typedef int SlangContainerFormatIntegral;
    enum SlangContainerFormat : SlangContainerFormatIntegral
    {
        /* Don't generate a container. */
        SLANG_CONTAINER_FORMAT_NONE,

        /* Generate a container in the `.slang-module` format,
        which includes reflection information, compiled kernels, etc. */
        SLANG_CONTAINER_FORMAT_SLANG_MODULE,
    };

    typedef int SlangPassThroughIntegral;
    enum SlangPassThrough : SlangPassThroughIntegral
    {
        SLANG_PASS_THROUGH_NONE,
        SLANG_PASS_THROUGH_FXC,
        SLANG_PASS_THROUGH_DXC,
        SLANG_PASS_THROUGH_GLSLANG,
        SLANG_PASS_THROUGH_SPIRV_DIS,
        SLANG_PASS_THROUGH_CLANG,         ///< Clang C/C++ compiler
        SLANG_PASS_THROUGH_VISUAL_STUDIO, ///< Visual studio C/C++ compiler
        SLANG_PASS_THROUGH_GCC,           ///< GCC C/C++ compiler
        SLANG_PASS_THROUGH_GENERIC_C_CPP, ///< Generic C or C++ compiler, which is decided by the
                                          ///< source type
        SLANG_PASS_THROUGH_NVRTC,         ///< NVRTC Cuda compiler
        SLANG_PASS_THROUGH_LLVM,          ///< LLVM 'compiler' - includes LLVM and Clang
        SLANG_PASS_THROUGH_SPIRV_OPT,     ///< SPIRV-opt
        SLANG_PASS_THROUGH_METAL,         ///< Metal compiler
        SLANG_PASS_THROUGH_TINT,          ///< Tint WGSL compiler
        SLANG_PASS_THROUGH_SPIRV_LINK,    ///< SPIRV-link
        SLANG_PASS_THROUGH_COUNT_OF,
    };

    /* Defines an archive type used to holds a 'file system' type structure. */
    typedef int SlangArchiveTypeIntegral;
    enum SlangArchiveType : SlangArchiveTypeIntegral
    {
        SLANG_ARCHIVE_TYPE_UNDEFINED,
        SLANG_ARCHIVE_TYPE_ZIP,
        SLANG_ARCHIVE_TYPE_RIFF, ///< Riff container with no compression
        SLANG_ARCHIVE_TYPE_RIFF_DEFLATE,
        SLANG_ARCHIVE_TYPE_RIFF_LZ4,
        SLANG_ARCHIVE_TYPE_COUNT_OF,
    };

    /*!
    Flags to control compilation behavior.
    */
    typedef unsigned int SlangCompileFlags;
    enum
    {
        /* Do as little mangling of names as possible, to try to preserve original names */
        SLANG_COMPILE_FLAG_NO_MANGLING = 1 << 3,

        /* Skip code generation step, just check the code and generate layout */
        SLANG_COMPILE_FLAG_NO_CODEGEN = 1 << 4,

        /* Obfuscate shader names on release products */
        SLANG_COMPILE_FLAG_OBFUSCATE = 1 << 5,

        /* Deprecated flags: kept around to allow existing applications to
        compile. Note that the relevant features will still be left in
        their default state. */
        SLANG_COMPILE_FLAG_NO_CHECKING = 0,
        SLANG_COMPILE_FLAG_SPLIT_MIXED_TYPES = 0,
    };

    /*!
    @brief Flags to control code generation behavior of a compilation target */
    typedef unsigned int SlangTargetFlags;
    enum
    {
        /* When compiling for a D3D Shader Model 5.1 or higher target, allocate
           distinct register spaces for parameter blocks.

           @deprecated This behavior is now enabled unconditionally.
        */
        SLANG_TARGET_FLAG_PARAMETER_BLOCKS_USE_REGISTER_SPACES = 1 << 4,

        /* When set, will generate target code that contains all entrypoints defined
           in the input source or specified via the `spAddEntryPoint` function in a
           single output module (library/source file).
        */
        SLANG_TARGET_FLAG_GENERATE_WHOLE_PROGRAM = 1 << 8,

        /* When set, will dump out the IR between intermediate compilation steps.*/
        SLANG_TARGET_FLAG_DUMP_IR = 1 << 9,

        /* When set, will generate SPIRV directly rather than via glslang. */
        // This flag will be deprecated, use CompilerOption instead.
        SLANG_TARGET_FLAG_GENERATE_SPIRV_DIRECTLY = 1 << 10,
    };
    constexpr static SlangTargetFlags kDefaultTargetFlags =
        SLANG_TARGET_FLAG_GENERATE_SPIRV_DIRECTLY;

    /*!
    @brief Options to control floating-point precision guarantees for a target.
    */
    typedef unsigned int SlangFloatingPointModeIntegral;
    enum SlangFloatingPointMode : SlangFloatingPointModeIntegral
    {
        SLANG_FLOATING_POINT_MODE_DEFAULT = 0,
        SLANG_FLOATING_POINT_MODE_FAST,
        SLANG_FLOATING_POINT_MODE_PRECISE,
    };

    /*!
    @brief Options to control emission of `#line` directives
    */
    typedef unsigned int SlangLineDirectiveModeIntegral;
    enum SlangLineDirectiveMode : SlangLineDirectiveModeIntegral
    {
        SLANG_LINE_DIRECTIVE_MODE_DEFAULT =
            0,                              /**< Default behavior: pick behavior base on target. */
        SLANG_LINE_DIRECTIVE_MODE_NONE,     /**< Don't emit line directives at all. */
        SLANG_LINE_DIRECTIVE_MODE_STANDARD, /**< Emit standard C-style `#line` directives. */
        SLANG_LINE_DIRECTIVE_MODE_GLSL, /**< Emit GLSL-style directives with file *number* instead
                                           of name */
        SLANG_LINE_DIRECTIVE_MODE_SOURCE_MAP, /**< Use a source map to track line mappings (ie no
                                                 #line will appear in emitting source) */
    };

    typedef int SlangSourceLanguageIntegral;
    enum SlangSourceLanguage : SlangSourceLanguageIntegral
    {
        SLANG_SOURCE_LANGUAGE_UNKNOWN,
        SLANG_SOURCE_LANGUAGE_SLANG,
        SLANG_SOURCE_LANGUAGE_HLSL,
        SLANG_SOURCE_LANGUAGE_GLSL,
        SLANG_SOURCE_LANGUAGE_C,
        SLANG_SOURCE_LANGUAGE_CPP,
        SLANG_SOURCE_LANGUAGE_CUDA,
        SLANG_SOURCE_LANGUAGE_SPIRV,
        SLANG_SOURCE_LANGUAGE_METAL,
        SLANG_SOURCE_LANGUAGE_WGSL,
        SLANG_SOURCE_LANGUAGE_COUNT_OF,
    };

    typedef unsigned int SlangProfileIDIntegral;
    enum SlangProfileID : SlangProfileIDIntegral
    {
        SLANG_PROFILE_UNKNOWN,
    };


    typedef SlangInt32 SlangCapabilityIDIntegral;
    enum SlangCapabilityID : SlangCapabilityIDIntegral
    {
        SLANG_CAPABILITY_UNKNOWN = 0,
    };

    typedef unsigned int SlangMatrixLayoutModeIntegral;
    enum SlangMatrixLayoutMode : SlangMatrixLayoutModeIntegral
    {
        SLANG_MATRIX_LAYOUT_MODE_UNKNOWN = 0,
        SLANG_MATRIX_LAYOUT_ROW_MAJOR,
        SLANG_MATRIX_LAYOUT_COLUMN_MAJOR,
    };

    typedef SlangUInt32 SlangStageIntegral;
    enum SlangStage : SlangStageIntegral
    {
        SLANG_STAGE_NONE,
        SLANG_STAGE_VERTEX,
        SLANG_STAGE_HULL,
        SLANG_STAGE_DOMAIN,
        SLANG_STAGE_GEOMETRY,
        SLANG_STAGE_FRAGMENT,
        SLANG_STAGE_COMPUTE,
        SLANG_STAGE_RAY_GENERATION,
        SLANG_STAGE_INTERSECTION,
        SLANG_STAGE_ANY_HIT,
        SLANG_STAGE_CLOSEST_HIT,
        SLANG_STAGE_MISS,
        SLANG_STAGE_CALLABLE,
        SLANG_STAGE_MESH,
        SLANG_STAGE_AMPLIFICATION,
        SLANG_STAGE_DISPATCH,
        //
        SLANG_STAGE_COUNT,

        // alias:
        SLANG_STAGE_PIXEL = SLANG_STAGE_FRAGMENT,
    };

    typedef SlangUInt32 SlangDebugInfoLevelIntegral;
    enum SlangDebugInfoLevel : SlangDebugInfoLevelIntegral
    {
        SLANG_DEBUG_INFO_LEVEL_NONE = 0, /**< Don't emit debug information at all. */
        SLANG_DEBUG_INFO_LEVEL_MINIMAL,  /**< Emit as little debug information as possible, while
                                            still supporting stack trackers. */
        SLANG_DEBUG_INFO_LEVEL_STANDARD, /**< Emit whatever is the standard level of debug
                                            information for each target. */
        SLANG_DEBUG_INFO_LEVEL_MAXIMAL,  /**< Emit as much debug information as possible for each
                                            target. */
    };

    /* Describes the debugging information format produced during a compilation. */
    typedef SlangUInt32 SlangDebugInfoFormatIntegral;
    enum SlangDebugInfoFormat : SlangDebugInfoFormatIntegral
    {
        SLANG_DEBUG_INFO_FORMAT_DEFAULT, ///< Use the default debugging format for the target
        SLANG_DEBUG_INFO_FORMAT_C7,  ///< CodeView C7 format (typically means debugging information
                                     ///< is embedded in the binary)
        SLANG_DEBUG_INFO_FORMAT_PDB, ///< Program database

        SLANG_DEBUG_INFO_FORMAT_STABS, ///< Stabs
        SLANG_DEBUG_INFO_FORMAT_COFF,  ///< COFF debug info
        SLANG_DEBUG_INFO_FORMAT_DWARF, ///< DWARF debug info (we may want to support specifying the
                                       ///< version)

        SLANG_DEBUG_INFO_FORMAT_COUNT_OF,
    };

    typedef SlangUInt32 SlangOptimizationLevelIntegral;
    enum SlangOptimizationLevel : SlangOptimizationLevelIntegral
    {
        SLANG_OPTIMIZATION_LEVEL_NONE = 0, /**< Don't optimize at all. */
        SLANG_OPTIMIZATION_LEVEL_DEFAULT,  /**< Default optimization level: balance code quality and
                                              compilation time. */
        SLANG_OPTIMIZATION_LEVEL_HIGH,     /**< Optimize aggressively. */
        SLANG_OPTIMIZATION_LEVEL_MAXIMAL, /**< Include optimizations that may take a very long time,
                                             or may involve severe space-vs-speed tradeoffs */
    };

    enum SlangEmitSpirvMethod
    {
        SLANG_EMIT_SPIRV_DEFAULT = 0,
        SLANG_EMIT_SPIRV_VIA_GLSL,
        SLANG_EMIT_SPIRV_DIRECTLY,
    };

    // All compiler option names supported by Slang.
    namespace slang
    {
    enum class CompilerOptionName
    {
        MacroDefine, // stringValue0: macro name;  stringValue1: macro value
        DepFile,
        EntryPointName,
        Specialize,
        Help,
        HelpStyle,
        Include, // stringValue: additional include path.
        Language,
        MatrixLayoutColumn,         // bool
        MatrixLayoutRow,            // bool
        ZeroInitialize,             // bool
        IgnoreCapabilities,         // bool
        RestrictiveCapabilityCheck, // bool
        ModuleName,                 // stringValue0: module name.
        Output,
        Profile, // intValue0: profile
        Stage,   // intValue0: stage
        Target,  // intValue0: CodeGenTarget
        Version,
        WarningsAsErrors, // stringValue0: "all" or comma separated list of warning codes or names.
        DisableWarnings,  // stringValue0: comma separated list of warning codes or names.
        EnableWarning,    // stringValue0: warning code or name.
        DisableWarning,   // stringValue0: warning code or name.
        DumpWarningDiagnostics,
        InputFilesRemain,
        EmitIr,                        // bool
        ReportDownstreamTime,          // bool
        ReportPerfBenchmark,           // bool
        ReportCheckpointIntermediates, // bool
        SkipSPIRVValidation,           // bool
        SourceEmbedStyle,
        SourceEmbedName,
        SourceEmbedLanguage,
        DisableShortCircuit,            // bool
        MinimumSlangOptimization,       // bool
        DisableNonEssentialValidations, // bool
        DisableSourceMap,               // bool
        UnscopedEnum,                   // bool
        PreserveParameters, // bool: preserve all resource parameters in the output code.

        // Target

        Capability,                // intValue0: CapabilityName
        DefaultImageFormatUnknown, // bool
        DisableDynamicDispatch,    // bool
        DisableSpecialization,     // bool
        FloatingPointMode,         // intValue0: FloatingPointMode
        DebugInformation,          // intValue0: DebugInfoLevel
        LineDirectiveMode,
        Optimization, // intValue0: OptimizationLevel
        Obfuscate,    // bool

        VulkanBindShift, // intValue0 (higher 8 bits): kind; intValue0(lower bits): set; intValue1:
                         // shift
        VulkanBindGlobals,       // intValue0: index; intValue1: set
        VulkanInvertY,           // bool
        VulkanUseDxPositionW,    // bool
        VulkanUseEntryPointName, // bool
        VulkanUseGLLayout,       // bool
        VulkanEmitReflection,    // bool

        GLSLForceScalarLayout,   // bool
        EnableEffectAnnotations, // bool

        EmitSpirvViaGLSL,     // bool (will be deprecated)
        EmitSpirvDirectly,    // bool (will be deprecated)
        SPIRVCoreGrammarJSON, // stringValue0: json path
        IncompleteLibrary,    // bool, when set, will not issue an error when the linked program has
                              // unresolved extern function symbols.

        // Downstream

        CompilerPath,
        DefaultDownstreamCompiler,
        DownstreamArgs, // stringValue0: downstream compiler name. stringValue1: argument list, one
                        // per line.
        PassThrough,

        // Repro

        DumpRepro,
        DumpReproOnError,
        ExtractRepro,
        LoadRepro,
        LoadReproDirectory,
        ReproFallbackDirectory,

        // Debugging

        DumpAst,
        DumpIntermediatePrefix,
        DumpIntermediates, // bool
        DumpIr,            // bool
        DumpIrIds,
        PreprocessorOutput,
        OutputIncludes,
        ReproFileSystem,
        SerialIr,    // bool
        SkipCodeGen, // bool
        ValidateIr,  // bool
        VerbosePaths,
        VerifyDebugSerialIr,
        NoCodeGen, // Not used.

        // Experimental

        FileSystem,
        Heterogeneous,
        NoMangle,
        NoHLSLBinding,
        NoHLSLPackConstantBufferElements,
        ValidateUniformity,
        AllowGLSL,
        EnableExperimentalPasses,
        BindlessSpaceIndex, // int

        // Internal

        ArchiveType,
        CompileCoreModule,
        Doc,

        IrCompression, //< deprecated

        LoadCoreModule,
        ReferenceModule,
        SaveCoreModule,
        SaveCoreModuleBinSource,
        TrackLiveness,
        LoopInversion, // bool, enable loop inversion optimization

        // Deprecated
        ParameterBlocksUseRegisterSpaces,

        CountOfParsableOptions,

        // Used in parsed options only.
        DebugInformationFormat,  // intValue0: DebugInfoFormat
        VulkanBindShiftAll,      // intValue0: kind; intValue1: shift
        GenerateWholeProgram,    // bool
        UseUpToDateBinaryModule, // bool, when set, will only load
                                 // precompiled modules if it is up-to-date with its source.
        EmbedDownstreamIR,       // bool
        ForceDXLayout,           // bool

        // Add this new option to the end of the list to avoid breaking ABI as much as possible.
        // Setting of EmitSpirvDirectly or EmitSpirvViaGLSL will turn into this option internally.
        EmitSpirvMethod, // enum SlangEmitSpirvMethod

        EmitReflectionJSON, // bool
        SaveGLSLModuleBinSource,

        SkipDownstreamLinking, // bool, experimental
        DumpModule,
        CountOf,
    };

    enum class CompilerOptionValueKind
    {
        Int,
        String
    };

    struct CompilerOptionValue
    {
        CompilerOptionValueKind kind = CompilerOptionValueKind::Int;
        int32_t intValue0 = 0;
        int32_t intValue1 = 0;
        const char* stringValue0 = nullptr;
        const char* stringValue1 = nullptr;
    };

    struct CompilerOptionEntry
    {
        CompilerOptionName name;
        CompilerOptionValue value;
    };
    } // namespace slang

    /** A result code for a Slang API operation.

    This type is generally compatible with the Windows API `HRESULT` type. In particular, negative
    values indicate failure results, while zero or positive results indicate success.

    In general, Slang APIs always return a zero result on success, unless documented otherwise.
    Strictly speaking a negative value indicates an error, a positive (or 0) value indicates
    success. This can be tested for with the macros SLANG_SUCCEEDED(x) or SLANG_FAILED(x).

    It can represent if the call was successful or not. It can also specify in an extensible manner
    what facility produced the result (as the integral 'facility') as well as what caused it (as an
    integral 'code'). Under the covers SlangResult is represented as a int32_t.

    SlangResult is designed to be compatible with COM HRESULT.

    It's layout in bits is as follows

    Severity | Facility | Code
    ---------|----------|-----
    31       |    30-16 | 15-0

    Severity - 1 fail, 0 is success - as SlangResult is signed 32 bits, means negative number
    indicates failure. Facility is where the error originated from. Code is the code specific to the
    facility.

    Result codes have the following styles,
    1) SLANG_name
    2) SLANG_s_f_name
    3) SLANG_s_name

    where s is S for success, E for error
    f is the short version of the facility name

    Style 1 is reserved for SLANG_OK and SLANG_FAIL as they are so commonly used.

    It is acceptable to expand 'f' to a longer name to differentiate a name or drop if unique
    without it. ie for a facility 'DRIVER' it might make sense to have an error of the form
    SLANG_E_DRIVER_OUT_OF_MEMORY
    */

    typedef int32_t SlangResult;

    //! Use to test if a result was failure. Never use result != SLANG_OK to test for failure, as
    //! there may be successful codes != SLANG_OK.
#define SLANG_FAILED(status) ((status) < 0)
    //! Use to test if a result succeeded. Never use result == SLANG_OK to test for success, as will
    //! detect other successful codes as a failure.
#define SLANG_SUCCEEDED(status) ((status) >= 0)

    //! Get the facility the result is associated with
#define SLANG_GET_RESULT_FACILITY(r) ((int32_t)(((r) >> 16) & 0x7fff))
    //! Get the result code for the facility
#define SLANG_GET_RESULT_CODE(r) ((int32_t)((r) & 0xffff))

#define SLANG_MAKE_ERROR(fac, code) \
    ((((int32_t)(fac)) << 16) | ((int32_t)(code)) | int32_t(0x80000000))
#define SLANG_MAKE_SUCCESS(fac, code) ((((int32_t)(fac)) << 16) | ((int32_t)(code)))

    /*************************** Facilities ************************************/

    //! Facilities compatible with windows COM - only use if known code is compatible
#define SLANG_FACILITY_WIN_GENERAL 0
#define SLANG_FACILITY_WIN_INTERFACE 4
#define SLANG_FACILITY_WIN_API 7

    //! Base facility -> so as to not clash with HRESULT values (values in 0x200 range do not appear
    //! used)
#define SLANG_FACILITY_BASE 0x200

    /*! Facilities numbers must be unique across a project to make the resulting result a unique
    number. It can be useful to have a consistent short name for a facility, as used in the name
    prefix */
#define SLANG_FACILITY_CORE SLANG_FACILITY_BASE
    /* Facility for codes, that are not uniquely defined/protected. Can be used to pass back a
    specific error without requiring system wide facility uniqueness. Codes should never be part of
    a public API. */
#define SLANG_FACILITY_INTERNAL SLANG_FACILITY_BASE + 1

    /// Base for external facilities. Facilities should be unique across modules.
#define SLANG_FACILITY_EXTERNAL_BASE 0x210

    /* ************************ Win COM compatible Results ******************************/
    // https://msdn.microsoft.com/en-us/library/windows/desktop/aa378137(v=vs.85).aspx

    //! SLANG_OK indicates success, and is equivalent to
    //! SLANG_MAKE_SUCCESS(SLANG_FACILITY_WIN_GENERAL, 0)
#define SLANG_OK 0
    //! SLANG_FAIL is the generic failure code - meaning a serious error occurred and the call
    //! couldn't complete
#define SLANG_FAIL SLANG_MAKE_ERROR(SLANG_FACILITY_WIN_GENERAL, 0x4005)

#define SLANG_MAKE_WIN_GENERAL_ERROR(code) SLANG_MAKE_ERROR(SLANG_FACILITY_WIN_GENERAL, code)

    //! Functionality is not implemented
#define SLANG_E_NOT_IMPLEMENTED SLANG_MAKE_WIN_GENERAL_ERROR(0x4001)
    //! Interface not be found
#define SLANG_E_NO_INTERFACE SLANG_MAKE_WIN_GENERAL_ERROR(0x4002)
    //! Operation was aborted (did not correctly complete)
#define SLANG_E_ABORT SLANG_MAKE_WIN_GENERAL_ERROR(0x4004)

    //! Indicates that a handle passed in as parameter to a method is invalid.
#define SLANG_E_INVALID_HANDLE SLANG_MAKE_ERROR(SLANG_FACILITY_WIN_API, 6)
    //! Indicates that an argument passed in as parameter to a method is invalid.
#define SLANG_E_INVALID_ARG SLANG_MAKE_ERROR(SLANG_FACILITY_WIN_API, 0x57)
    //! Operation could not complete - ran out of memory
#define SLANG_E_OUT_OF_MEMORY SLANG_MAKE_ERROR(SLANG_FACILITY_WIN_API, 0xe)

    /* *************************** other Results **************************************/

#define SLANG_MAKE_CORE_ERROR(code) SLANG_MAKE_ERROR(SLANG_FACILITY_CORE, code)

    // Supplied buffer is too small to be able to complete
#define SLANG_E_BUFFER_TOO_SMALL SLANG_MAKE_CORE_ERROR(1)
    //! Used to identify a Result that has yet to be initialized.
    //! It defaults to failure such that if used incorrectly will fail, as similar in concept to
    //! using an uninitialized variable.
#define SLANG_E_UNINITIALIZED SLANG_MAKE_CORE_ERROR(2)
    //! Returned from an async method meaning the output is invalid (thus an error), but a result
    //! for the request is pending, and will be returned on a subsequent call with the async handle.
#define SLANG_E_PENDING SLANG_MAKE_CORE_ERROR(3)
    //! Indicates a file/resource could not be opened
#define SLANG_E_CANNOT_OPEN SLANG_MAKE_CORE_ERROR(4)
    //! Indicates a file/resource could not be found
#define SLANG_E_NOT_FOUND SLANG_MAKE_CORE_ERROR(5)
    //! An unhandled internal failure (typically from unhandled exception)
#define SLANG_E_INTERNAL_FAIL SLANG_MAKE_CORE_ERROR(6)
    //! Could not complete because some underlying feature (hardware or software) was not available
#define SLANG_E_NOT_AVAILABLE SLANG_MAKE_CORE_ERROR(7)
    //! Could not complete because the operation times out.
#define SLANG_E_TIME_OUT SLANG_MAKE_CORE_ERROR(8)

    /** A "Universally Unique Identifier" (UUID)

    The Slang API uses UUIDs to identify interfaces when
    using `queryInterface`.

    This type is compatible with the `GUID` type defined
    by the Component Object Model (COM), but Slang is
    not dependent on COM.
    */
    struct SlangUUID
    {
        uint32_t data1;
        uint16_t data2;
        uint16_t data3;
        uint8_t data4[8];
    };

// Place at the start of an interface with the guid.
// Guid should be specified as SLANG_COM_INTERFACE(0x00000000, 0x0000, 0x0000, { 0xC0, 0x00, 0x00,
// 0x00, 0x00, 0x00, 0x00, 0x46 }) NOTE: it's the typical guid struct definition, without the
// surrounding {} It is not necessary to use the multiple parameters (we can wrap in parens), but
// this is simple.
#define SLANG_COM_INTERFACE(a, b, c, d0, d1, d2, d3, d4, d5, d6, d7) \
public:                                                              \
    SLANG_FORCE_INLINE constexpr static SlangUUID getTypeGuid()      \
    {                                                                \
        return {a, b, c, d0, d1, d2, d3, d4, d5, d6, d7};            \
    }

// Sometimes it's useful to associate a guid with a class to identify it. This macro can used for
// this, and the guid extracted via the getTypeGuid() function defined in the type
#define SLANG_CLASS_GUID(a, b, c, d0, d1, d2, d3, d4, d5, d6, d7) \
    SLANG_FORCE_INLINE constexpr static SlangUUID getTypeGuid()   \
    {                                                             \
        return {a, b, c, d0, d1, d2, d3, d4, d5, d6, d7};         \
    }

// Helper to fill in pairs of GUIDs and return pointers. This ensures that the
// type of the GUID passed matches the pointer type, and that it is derived
// from ISlangUnknown,
// TODO(c++20): would is_derived_from be more appropriate here for private inheritance of
// ISlangUnknown?
//
// with     : void createFoo(SlangUUID, void**);
//            Slang::ComPtr<Bar> myBar;
// call with: createFoo(SLANG_IID_PPV_ARGS(myBar.writeRef()))
// to call  : createFoo(Bar::getTypeGuid(), (void**)(myBar.writeRef()))
#define SLANG_IID_PPV_ARGS(ppType)                                                         \
    std::decay_t<decltype(**(ppType))>::getTypeGuid(),                                     \
        (                                                                                  \
            (void)[] {                                                                     \
                static_assert(                                                             \
                    std::is_base_of_v<ISlangUnknown, std::decay_t<decltype(**(ppType))>>); \
            },                                                                             \
            reinterpret_cast<void**>(ppType))


    /** Base interface for components exchanged through the API.

    This interface definition is compatible with the COM `IUnknown`,
    and uses the same UUID, but Slang does not require applications
    to use or initialize COM.
    */
    struct ISlangUnknown
    {
        SLANG_COM_INTERFACE(
            0x00000000,
            0x0000,
            0x0000,
            {0xC0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46})

        virtual SLANG_NO_THROW SlangResult SLANG_MCALL
        queryInterface(SlangUUID const& uuid, void** outObject) = 0;
        virtual SLANG_NO_THROW uint32_t SLANG_MCALL addRef() = 0;
        virtual SLANG_NO_THROW uint32_t SLANG_MCALL release() = 0;

        /*
        Inline methods are provided to allow the above operations to be called
        using their traditional COM names/signatures:
        */
        SlangResult QueryInterface(struct _GUID const& uuid, void** outObject)
        {
            return queryInterface(*(SlangUUID const*)&uuid, outObject);
        }
        uint32_t AddRef() { return addRef(); }
        uint32_t Release() { return release(); }
    };
#define SLANG_UUID_ISlangUnknown ISlangUnknown::getTypeGuid()


    /* An interface to provide a mechanism to cast, that doesn't require ref counting
    and doesn't have to return a pointer to a ISlangUnknown derived class */
    class ISlangCastable : public ISlangUnknown
    {
        SLANG_COM_INTERFACE(
            0x87ede0e1,
            0x4852,
            0x44b0,
            {0x8b, 0xf2, 0xcb, 0x31, 0x87, 0x4d, 0xe2, 0x39});

        /// Can be used to cast to interfaces without reference counting.
        /// Also provides access to internal implementations, when they provide a guid
        /// Can simulate a 'generated' interface as long as kept in scope by cast from.
        virtual SLANG_NO_THROW void* SLANG_MCALL castAs(const SlangUUID& guid) = 0;
    };

    class ISlangClonable : public ISlangCastable
    {
        SLANG_COM_INTERFACE(
            0x1ec36168,
            0xe9f4,
            0x430d,
            {0xbb, 0x17, 0x4, 0x8a, 0x80, 0x46, 0xb3, 0x1f});

        /// Note the use of guid is for the desired interface/object.
        /// The object is returned *not* ref counted. Any type that can implements the interface,
        /// derives from ICastable, and so (not withstanding some other issue) will always return
        /// an ICastable interface which other interfaces/types are accessible from via castAs
        SLANG_NO_THROW virtual void* SLANG_MCALL clone(const SlangUUID& guid) = 0;
    };

    /** A "blob" of binary data.

    This interface definition is compatible with the `ID3DBlob` and `ID3D10Blob` interfaces.
    */
    struct ISlangBlob : public ISlangUnknown
    {
        SLANG_COM_INTERFACE(
            0x8BA5FB08,
            0x5195,
            0x40e2,
            {0xAC, 0x58, 0x0D, 0x98, 0x9C, 0x3A, 0x01, 0x02})

        virtual SLANG_NO_THROW void const* SLANG_MCALL getBufferPointer() = 0;
        virtual SLANG_NO_THROW size_t SLANG_MCALL getBufferSize() = 0;
    };
#define SLANG_UUID_ISlangBlob ISlangBlob::getTypeGuid()

    /* Can be requested from ISlangCastable cast to indicate the contained chars are null
     * terminated.
     */
    struct SlangTerminatedChars
    {
        SLANG_CLASS_GUID(
            0xbe0db1a8,
            0x3594,
            0x4603,
            {0xa7, 0x8b, 0xc4, 0x86, 0x84, 0x30, 0xdf, 0xbb});
        operator const char*() const { return chars; }
        char chars[1];
    };

    /** A (real or virtual) file system.

    Slang can make use of this interface whenever it would otherwise try to load files
    from disk, allowing applications to hook and/or override filesystem access from
    the compiler.

    It is the responsibility of
    the caller of any method that returns a ISlangBlob to release the blob when it is no
    longer used (using 'release').
    */

    struct ISlangFileSystem : public ISlangCastable
    {
        SLANG_COM_INTERFACE(
            0x003A09FC,
            0x3A4D,
            0x4BA0,
            {0xAD, 0x60, 0x1F, 0xD8, 0x63, 0xA9, 0x15, 0xAB})

        /** Load a file from `path` and return a blob of its contents
        @param path The path to load from, as a null-terminated UTF-8 string.
        @param outBlob A destination pointer to receive the blob of the file contents.
        @returns A `SlangResult` to indicate success or failure in loading the file.

        NOTE! This is a *binary* load - the blob should contain the exact same bytes
        as are found in the backing file.

        If load is successful, the implementation should create a blob to hold
        the file's content, store it to `outBlob`, and return 0.
        If the load fails, the implementation should return a failure status
        (any negative value will do).
        */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL
        loadFile(char const* path, ISlangBlob** outBlob) = 0;
    };
#define SLANG_UUID_ISlangFileSystem ISlangFileSystem::getTypeGuid()


    typedef void (*SlangFuncPtr)(void);

    /**
    (DEPRECATED) ISlangSharedLibrary
    */
    struct ISlangSharedLibrary_Dep1 : public ISlangUnknown
    {
        SLANG_COM_INTERFACE(
            0x9c9d5bc5,
            0xeb61,
            0x496f,
            {0x80, 0xd7, 0xd1, 0x47, 0xc4, 0xa2, 0x37, 0x30})

        virtual SLANG_NO_THROW void* SLANG_MCALL findSymbolAddressByName(char const* name) = 0;
    };
#define SLANG_UUID_ISlangSharedLibrary_Dep1 ISlangSharedLibrary_Dep1::getTypeGuid()

    /** An interface that can be used to encapsulate access to a shared library. An implementation
    does not have to implement the library as a shared library
    */
    struct ISlangSharedLibrary : public ISlangCastable
    {
        SLANG_COM_INTERFACE(
            0x70dbc7c4,
            0xdc3b,
            0x4a07,
            {0xae, 0x7e, 0x75, 0x2a, 0xf6, 0xa8, 0x15, 0x55})

        /** Get a function by name. If the library is unloaded will only return nullptr.
        @param name The name of the function
        @return The function pointer related to the name or nullptr if not found
        */
        SLANG_FORCE_INLINE SlangFuncPtr findFuncByName(char const* name)
        {
            return (SlangFuncPtr)findSymbolAddressByName(name);
        }

        /** Get a symbol by name. If the library is unloaded will only return nullptr.
        @param name The name of the symbol
        @return The pointer related to the name or nullptr if not found
        */
        virtual SLANG_NO_THROW void* SLANG_MCALL findSymbolAddressByName(char const* name) = 0;
    };
#define SLANG_UUID_ISlangSharedLibrary ISlangSharedLibrary::getTypeGuid()

    struct ISlangSharedLibraryLoader : public ISlangUnknown
    {
        SLANG_COM_INTERFACE(
            0x6264ab2b,
            0xa3e8,
            0x4a06,
            {0x97, 0xf1, 0x49, 0xbc, 0x2d, 0x2a, 0xb1, 0x4d})

        /** Load a shared library. In typical usage the library name should *not* contain any
        platform specific elements. For example on windows a dll name should *not* be passed with a
        '.dll' extension, and similarly on linux a shared library should *not* be passed with the
        'lib' prefix and '.so' extension
        @path path The unadorned filename and/or path for the shared library
        @ param sharedLibraryOut Holds the shared library if successfully loaded */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL
        loadSharedLibrary(const char* path, ISlangSharedLibrary** sharedLibraryOut) = 0;
    };
#define SLANG_UUID_ISlangSharedLibraryLoader ISlangSharedLibraryLoader::getTypeGuid()

    /* Type that identifies how a path should be interpreted */
    typedef unsigned int SlangPathTypeIntegral;
    enum SlangPathType : SlangPathTypeIntegral
    {
        SLANG_PATH_TYPE_DIRECTORY, /**< Path specified specifies a directory. */
        SLANG_PATH_TYPE_FILE,      /**< Path specified is to a file. */
    };

    /* Callback to enumerate the contents of of a directory in a ISlangFileSystemExt.
    The name is the name of a file system object (directory/file) in the specified path (ie it is
    without a path) */
    typedef void (
        *FileSystemContentsCallBack)(SlangPathType pathType, const char* name, void* userData);

    /* Determines how paths map to files on the OS file system */
    enum class OSPathKind : uint8_t
    {
        None,            ///< Paths do not map to the file system
        Direct,          ///< Paths map directly to the file system
        OperatingSystem, ///< Only paths gained via PathKind::OperatingSystem map to the operating
                         ///< system file system
    };

    /* Used to determine what kind of path is required from an input path */
    enum class PathKind
    {
        /// Given a path, returns a simplified version of that path.
        /// This typically means removing '..' and/or '.' from the path.
        /// A simplified path must point to the same object as the original.
        Simplified,

        /// Given a path, returns a 'canonical path' to the item.
        /// This may be the operating system 'canonical path' that is the unique path to the item.
        ///
        /// If the item exists the returned canonical path should always be usable to access the
        /// item.
        ///
        /// If the item the path specifies doesn't exist, the canonical path may not be returnable
        /// or be a path simplification.
        /// Not all file systems support canonical paths.
        Canonical,

        /// Given a path returns a path such that it is suitable to be displayed to the user.
        ///
        /// For example if the file system is a zip file - it might include the path to the zip
        /// container as well as the path to the specific file.
        ///
        /// NOTE! The display path won't necessarily work on the file system to access the item
        Display,

        /// Get the path to the item on the *operating system* file system, if available.
        OperatingSystem,

        CountOf,
    };

    /** An extended file system abstraction.

    Implementing and using this interface over ISlangFileSystem gives much more control over how
    paths are managed, as well as how it is determined if two files 'are the same'.

    All paths as input char*, or output as ISlangBlobs are always encoded as UTF-8 strings.
    Blobs that contain strings are always zero terminated.
    */
    struct ISlangFileSystemExt : public ISlangFileSystem
    {
        SLANG_COM_INTERFACE(
            0x5fb632d2,
            0x979d,
            0x4481,
            {0x9f, 0xee, 0x66, 0x3c, 0x3f, 0x14, 0x49, 0xe1})

        /** Get a uniqueIdentity which uniquely identifies an object of the file system.

        Given a path, returns a 'uniqueIdentity' which ideally is the same value for the same object
        on the file system.

        The uniqueIdentity is used to compare if two paths are the same - which amongst other things
        allows Slang to cache source contents internally. It is also used for #pragma once
        functionality.

        A *requirement* is for any implementation is that two paths can only return the same
        uniqueIdentity if the contents of the two files are *identical*. If an implementation breaks
        this constraint it can produce incorrect compilation. If an implementation cannot *strictly*
        identify *the same* files, this will only have an effect on #pragma once behavior.

        The string for the uniqueIdentity is held zero terminated in the ISlangBlob of
        outUniqueIdentity.

        Note that there are many ways a uniqueIdentity may be generated for a file. For example it
        could be the 'canonical path' - assuming it is available and unambiguous for a file system.
        Another possible mechanism could be to store the filename combined with the file date time
        to uniquely identify it.

        The client must ensure the blob be released when no longer used, otherwise memory will leak.

        NOTE! Ideally this method would be called 'getPathUniqueIdentity' but for historical reasons
        and backward compatibility it's name remains with 'File' even though an implementation
        should be made to work with directories too.

        @param path
        @param outUniqueIdentity
        @returns A `SlangResult` to indicate success or failure getting the uniqueIdentity.
        */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL
        getFileUniqueIdentity(const char* path, ISlangBlob** outUniqueIdentity) = 0;

        /** Calculate a path combining the 'fromPath' with 'path'

        The client must ensure the blob be released when no longer used, otherwise memory will leak.

        @param fromPathType How to interpret the from path - as a file or a directory.
        @param fromPath The from path.
        @param path Path to be determined relative to the fromPath
        @param pathOut Holds the string which is the relative path. The string is held in the blob
        zero terminated.
        @returns A `SlangResult` to indicate success or failure in loading the file.
        */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL calcCombinedPath(
            SlangPathType fromPathType,
            const char* fromPath,
            const char* path,
            ISlangBlob** pathOut) = 0;

        /** Gets the type of path that path is on the file system.
        @param path
        @param pathTypeOut
        @returns SLANG_OK if located and type is known, else an error. SLANG_E_NOT_FOUND if not
        found.
        */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL
        getPathType(const char* path, SlangPathType* pathTypeOut) = 0;

        /** Get a path based on the kind.

        @param kind The kind of path wanted
        @param path The input path
        @param outPath The output path held in a blob
        @returns SLANG_OK if successfully simplified the path (SLANG_E_NOT_IMPLEMENTED if not
        implemented, or some other error code)
        */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL
        getPath(PathKind kind, const char* path, ISlangBlob** outPath) = 0;

        /** Clears any cached information */
        virtual SLANG_NO_THROW void SLANG_MCALL clearCache() = 0;

        /** Enumerate the contents of the path

        Note that for normal Slang operation it isn't necessary to enumerate contents this can
        return SLANG_E_NOT_IMPLEMENTED.

        @param The path to enumerate
        @param callback This callback is called for each entry in the path.
        @param userData This is passed to the callback
        @returns SLANG_OK if successful
        */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL enumeratePathContents(
            const char* path,
            FileSystemContentsCallBack callback,
            void* userData) = 0;

        /** Returns how paths map to the OS file system

        @returns OSPathKind that describes how paths map to the Operating System file system
        */
        virtual SLANG_NO_THROW OSPathKind SLANG_MCALL getOSPathKind() = 0;
    };

#define SLANG_UUID_ISlangFileSystemExt ISlangFileSystemExt::getTypeGuid()

    struct ISlangMutableFileSystem : public ISlangFileSystemExt
    {
        SLANG_COM_INTERFACE(
            0xa058675c,
            0x1d65,
            0x452a,
            {0x84, 0x58, 0xcc, 0xde, 0xd1, 0x42, 0x71, 0x5})

        /** Write data to the specified path.

        @param path The path for data to be saved to
        @param data The data to be saved
        @param size The size of the data in bytes
        @returns SLANG_OK if successful (SLANG_E_NOT_IMPLEMENTED if not implemented, or some other
        error code)
        */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL
        saveFile(const char* path, const void* data, size_t size) = 0;

        /** Write data in the form of a blob to the specified path.

        Depending on the implementation writing a blob might be faster/use less memory. It is
        assumed the blob is *immutable* and that an implementation can reference count it.

        It is not guaranteed loading the same file will return the *same* blob - just a blob with
        same contents.

        @param path The path for data to be saved to
        @param dataBlob The data to be saved
        @returns SLANG_OK if successful (SLANG_E_NOT_IMPLEMENTED if not implemented, or some other
        error code)
        */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL
        saveFileBlob(const char* path, ISlangBlob* dataBlob) = 0;

        /** Remove the entry in the path (directory of file). Will only delete an empty directory,
        if not empty will return an error.

        @param path The path to remove
        @returns SLANG_OK if successful
        */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL remove(const char* path) = 0;

        /** Create a directory.

        The path to the directory must exist

        @param path To the directory to create. The parent path *must* exist otherwise will return
        an error.
        @returns SLANG_OK if successful
        */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL createDirectory(const char* path) = 0;
    };

#define SLANG_UUID_ISlangMutableFileSystem ISlangMutableFileSystem::getTypeGuid()

    /* Identifies different types of writer target*/
    typedef unsigned int SlangWriterChannelIntegral;
    enum SlangWriterChannel : SlangWriterChannelIntegral
    {
        SLANG_WRITER_CHANNEL_DIAGNOSTIC,
        SLANG_WRITER_CHANNEL_STD_OUTPUT,
        SLANG_WRITER_CHANNEL_STD_ERROR,
        SLANG_WRITER_CHANNEL_COUNT_OF,
    };

    typedef unsigned int SlangWriterModeIntegral;
    enum SlangWriterMode : SlangWriterModeIntegral
    {
        SLANG_WRITER_MODE_TEXT,
        SLANG_WRITER_MODE_BINARY,
    };

    /** A stream typically of text, used for outputting diagnostic as well as other information.
     */
    struct ISlangWriter : public ISlangUnknown
    {
        SLANG_COM_INTERFACE(
            0xec457f0e,
            0x9add,
            0x4e6b,
            {0x85, 0x1c, 0xd7, 0xfa, 0x71, 0x6d, 0x15, 0xfd})

        /** Begin an append buffer.
        NOTE! Only one append buffer can be active at any time.
        @param maxNumChars The maximum of chars that will be appended
        @returns The start of the buffer for appending to. */
        virtual SLANG_NO_THROW char* SLANG_MCALL beginAppendBuffer(size_t maxNumChars) = 0;
        /** Ends the append buffer, and is equivalent to a write of the append buffer.
        NOTE! That an endAppendBuffer is not necessary if there are no characters to write.
        @param buffer is the start of the data to append and must be identical to last value
        returned from beginAppendBuffer
        @param numChars must be a value less than or equal to what was returned from last call to
        beginAppendBuffer
        @returns Result, will be SLANG_OK on success */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL
        endAppendBuffer(char* buffer, size_t numChars) = 0;
        /** Write text to the writer
        @param chars The characters to write out
        @param numChars The amount of characters
        @returns SLANG_OK on success */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL
        write(const char* chars, size_t numChars) = 0;
        /** Flushes any content to the output */
        virtual SLANG_NO_THROW void SLANG_MCALL flush() = 0;
        /** Determines if the writer stream is to the console, and can be used to alter the output
        @returns Returns true if is a console writer */
        virtual SLANG_NO_THROW SlangBool SLANG_MCALL isConsole() = 0;
        /** Set the mode for the writer to use
        @param mode The mode to use
        @returns SLANG_OK on success */
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL setMode(SlangWriterMode mode) = 0;
    };

#define SLANG_UUID_ISlangWriter ISlangWriter::getTypeGuid()

    struct ISlangProfiler : public ISlangUnknown
    {
        SLANG_COM_INTERFACE(
            0x197772c7,
            0x0155,
            0x4b91,
            {0x84, 0xe8, 0x66, 0x68, 0xba, 0xff, 0x06, 0x19})
        virtual SLANG_NO_THROW size_t SLANG_MCALL getEntryCount() = 0;
        virtual SLANG_NO_THROW const char* SLANG_MCALL getEntryName(uint32_t index) = 0;
        virtual SLANG_NO_THROW long SLANG_MCALL getEntryTimeMS(uint32_t index) = 0;
        virtual SLANG_NO_THROW uint32_t SLANG_MCALL getEntryInvocationTimes(uint32_t index) = 0;
    };
#define SLANG_UUID_ISlangProfiler ISlangProfiler::getTypeGuid()

    namespace slang
    {
    struct IGlobalSession;
    struct ICompileRequest;

    } // namespace slang

    /*!
    @brief An instance of the Slang library.
    */
    typedef slang::IGlobalSession SlangSession;


    typedef struct SlangProgramLayout SlangProgramLayout;

    /*!
    @brief A request for one or more compilation actions to be performed.
    */
    typedef struct slang::ICompileRequest SlangCompileRequest;


    /*!
@brief Callback type used for diagnostic output.
*/
    typedef void (*SlangDiagnosticCallback)(char const* message, void* userData);

    /*!
    @brief Get the build version 'tag' string. The string is the same as
    produced via `git describe --tags --match v*` for the project. If such a
    version could not be determined at build time then the contents will be
    0.0.0-unknown. Any string can be set by passing
    -DSLANG_VERSION_FULL=whatever during the cmake invocation.

    This function will return exactly the same result as the method
    getBuildTagString on IGlobalSession.

    An advantage of using this function over the method is that doing so does
    not require the creation of a session, which can be a fairly costly
    operation.

    @return The build tag string
    */
    SLANG_API const char* spGetBuildTagString();

    /*
    Forward declarations of types used in the reflection interface;
    */

    typedef struct SlangProgramLayout SlangProgramLayout;
    typedef struct SlangEntryPoint SlangEntryPoint;
    typedef struct SlangEntryPointLayout SlangEntryPointLayout;

    typedef struct SlangReflectionDecl SlangReflectionDecl;
    typedef struct SlangReflectionModifier SlangReflectionModifier;
    typedef struct SlangReflectionType SlangReflectionType;
    typedef struct SlangReflectionTypeLayout SlangReflectionTypeLayout;
    typedef struct SlangReflectionVariable SlangReflectionVariable;
    typedef struct SlangReflectionVariableLayout SlangReflectionVariableLayout;
    typedef struct SlangReflectionTypeParameter SlangReflectionTypeParameter;
    typedef struct SlangReflectionUserAttribute SlangReflectionUserAttribute;
    typedef SlangReflectionUserAttribute SlangReflectionAttribute;
    typedef struct SlangReflectionFunction SlangReflectionFunction;
    typedef struct SlangReflectionGeneric SlangReflectionGeneric;

    union SlangReflectionGenericArg
    {
        SlangReflectionType* typeVal;
        int64_t intVal;
        bool boolVal;
    };

    enum SlangReflectionGenericArgType
    {
        SLANG_GENERIC_ARG_TYPE = 0,
        SLANG_GENERIC_ARG_INT = 1,
        SLANG_GENERIC_ARG_BOOL = 2
    };

    /*
    Type aliases to maintain backward compatibility.
    */
    typedef SlangProgramLayout SlangReflection;
    typedef SlangEntryPointLayout SlangReflectionEntryPoint;

    // type reflection

    typedef unsigned int SlangTypeKindIntegral;
    enum SlangTypeKind : SlangTypeKindIntegral
    {
        SLANG_TYPE_KIND_NONE,
        SLANG_TYPE_KIND_STRUCT,
        SLANG_TYPE_KIND_ARRAY,
        SLANG_TYPE_KIND_MATRIX,
        SLANG_TYPE_KIND_VECTOR,
        SLANG_TYPE_KIND_SCALAR,
        SLANG_TYPE_KIND_CONSTANT_BUFFER,
        SLANG_TYPE_KIND_RESOURCE,
        SLANG_TYPE_KIND_SAMPLER_STATE,
        SLANG_TYPE_KIND_TEXTURE_BUFFER,
        SLANG_TYPE_KIND_SHADER_STORAGE_BUFFER,
        SLANG_TYPE_KIND_PARAMETER_BLOCK,
        SLANG_TYPE_KIND_GENERIC_TYPE_PARAMETER,
        SLANG_TYPE_KIND_INTERFACE,
        SLANG_TYPE_KIND_OUTPUT_STREAM,
        SLANG_TYPE_KIND_MESH_OUTPUT,
        SLANG_TYPE_KIND_SPECIALIZED,
        SLANG_TYPE_KIND_FEEDBACK,
        SLANG_TYPE_KIND_POINTER,
        SLANG_TYPE_KIND_DYNAMIC_RESOURCE,
        SLANG_TYPE_KIND_COUNT,
    };

    typedef unsigned int SlangScalarTypeIntegral;
    enum SlangScalarType : SlangScalarTypeIntegral
    {
        SLANG_SCALAR_TYPE_NONE,
        SLANG_SCALAR_TYPE_VOID,
        SLANG_SCALAR_TYPE_BOOL,
        SLANG_SCALAR_TYPE_INT32,
        SLANG_SCALAR_TYPE_UINT32,
        SLANG_SCALAR_TYPE_INT64,
        SLANG_SCALAR_TYPE_UINT64,
        SLANG_SCALAR_TYPE_FLOAT16,
        SLANG_SCALAR_TYPE_FLOAT32,
        SLANG_SCALAR_TYPE_FLOAT64,
        SLANG_SCALAR_TYPE_INT8,
        SLANG_SCALAR_TYPE_UINT8,
        SLANG_SCALAR_TYPE_INT16,
        SLANG_SCALAR_TYPE_UINT16,
        SLANG_SCALAR_TYPE_INTPTR,
        SLANG_SCALAR_TYPE_UINTPTR
    };

    // abstract decl reflection
    typedef unsigned int SlangDeclKindIntegral;
    enum SlangDeclKind : SlangDeclKindIntegral
    {
        SLANG_DECL_KIND_UNSUPPORTED_FOR_REFLECTION,
        SLANG_DECL_KIND_STRUCT,
        SLANG_DECL_KIND_FUNC,
        SLANG_DECL_KIND_MODULE,
        SLANG_DECL_KIND_GENERIC,
        SLANG_DECL_KIND_VARIABLE,
        SLANG_DECL_KIND_NAMESPACE
    };

#ifndef SLANG_RESOURCE_SHAPE
    #define SLANG_RESOURCE_SHAPE
    typedef unsigned int SlangResourceShapeIntegral;
    enum SlangResourceShape : SlangResourceShapeIntegral
    {
        SLANG_RESOURCE_BASE_SHAPE_MASK = 0x0F,

        SLANG_RESOURCE_NONE = 0x00,

        SLANG_TEXTURE_1D = 0x01,
        SLANG_TEXTURE_2D = 0x02,
        SLANG_TEXTURE_3D = 0x03,
        SLANG_TEXTURE_CUBE = 0x04,
        SLANG_TEXTURE_BUFFER = 0x05,

        SLANG_STRUCTURED_BUFFER = 0x06,
        SLANG_BYTE_ADDRESS_BUFFER = 0x07,
        SLANG_RESOURCE_UNKNOWN = 0x08,
        SLANG_ACCELERATION_STRUCTURE = 0x09,
        SLANG_TEXTURE_SUBPASS = 0x0A,

        SLANG_RESOURCE_EXT_SHAPE_MASK = 0xF0,

        SLANG_TEXTURE_FEEDBACK_FLAG = 0x10,
        SLANG_TEXTURE_SHADOW_FLAG = 0x20,
        SLANG_TEXTURE_ARRAY_FLAG = 0x40,
        SLANG_TEXTURE_MULTISAMPLE_FLAG = 0x80,

        SLANG_TEXTURE_1D_ARRAY = SLANG_TEXTURE_1D | SLANG_TEXTURE_ARRAY_FLAG,
        SLANG_TEXTURE_2D_ARRAY = SLANG_TEXTURE_2D | SLANG_TEXTURE_ARRAY_FLAG,
        SLANG_TEXTURE_CUBE_ARRAY = SLANG_TEXTURE_CUBE | SLANG_TEXTURE_ARRAY_FLAG,

        SLANG_TEXTURE_2D_MULTISAMPLE = SLANG_TEXTURE_2D | SLANG_TEXTURE_MULTISAMPLE_FLAG,
        SLANG_TEXTURE_2D_MULTISAMPLE_ARRAY =
            SLANG_TEXTURE_2D | SLANG_TEXTURE_MULTISAMPLE_FLAG | SLANG_TEXTURE_ARRAY_FLAG,
        SLANG_TEXTURE_SUBPASS_MULTISAMPLE = SLANG_TEXTURE_SUBPASS | SLANG_TEXTURE_MULTISAMPLE_FLAG,
    };
#endif
    typedef unsigned int SlangResourceAccessIntegral;
    enum SlangResourceAccess : SlangResourceAccessIntegral
    {
        SLANG_RESOURCE_ACCESS_NONE,
        SLANG_RESOURCE_ACCESS_READ,
        SLANG_RESOURCE_ACCESS_READ_WRITE,
        SLANG_RESOURCE_ACCESS_RASTER_ORDERED,
        SLANG_RESOURCE_ACCESS_APPEND,
        SLANG_RESOURCE_ACCESS_CONSUME,
        SLANG_RESOURCE_ACCESS_WRITE,
        SLANG_RESOURCE_ACCESS_FEEDBACK,
        SLANG_RESOURCE_ACCESS_UNKNOWN = 0x7FFFFFFF,
    };

    typedef unsigned int SlangParameterCategoryIntegral;
    enum SlangParameterCategory : SlangParameterCategoryIntegral
    {
        SLANG_PARAMETER_CATEGORY_NONE,
        SLANG_PARAMETER_CATEGORY_MIXED,
        SLANG_PARAMETER_CATEGORY_CONSTANT_BUFFER,
        SLANG_PARAMETER_CATEGORY_SHADER_RESOURCE,
        SLANG_PARAMETER_CATEGORY_UNORDERED_ACCESS,
        SLANG_PARAMETER_CATEGORY_VARYING_INPUT,
        SLANG_PARAMETER_CATEGORY_VARYING_OUTPUT,
        SLANG_PARAMETER_CATEGORY_SAMPLER_STATE,
        SLANG_PARAMETER_CATEGORY_UNIFORM,
        SLANG_PARAMETER_CATEGORY_DESCRIPTOR_TABLE_SLOT,
        SLANG_PARAMETER_CATEGORY_SPECIALIZATION_CONSTANT,
        SLANG_PARAMETER_CATEGORY_PUSH_CONSTANT_BUFFER,

        // HLSL register `space`, Vulkan GLSL `set`
        SLANG_PARAMETER_CATEGORY_REGISTER_SPACE,

        // TODO: Ellie, Both APIs treat mesh outputs as more or less varying output,
        // Does it deserve to be represented here??

        // A parameter whose type is to be specialized by a global generic type argument
        SLANG_PARAMETER_CATEGORY_GENERIC,

        SLANG_PARAMETER_CATEGORY_RAY_PAYLOAD,
        SLANG_PARAMETER_CATEGORY_HIT_ATTRIBUTES,
        SLANG_PARAMETER_CATEGORY_CALLABLE_PAYLOAD,
        SLANG_PARAMETER_CATEGORY_SHADER_RECORD,

        // An existential type parameter represents a "hole" that
        // needs to be filled with a concrete type to enable
        // generation of specialized code.
        //
        // Consider this example:
        //
        //      struct MyParams
        //      {
        //          IMaterial material;
        //          ILight lights[3];
        //      };
        //
        // This `MyParams` type introduces two existential type parameters:
        // one for `material` and one for `lights`. Even though `lights`
        // is an array, it only introduces one type parameter, because
        // we need to have a *single* concrete type for all the array
        // elements to be able to generate specialized code.
        //
        SLANG_PARAMETER_CATEGORY_EXISTENTIAL_TYPE_PARAM,

        // An existential object parameter represents a value
        // that needs to be passed in to provide data for some
        // interface-type shader parameter.
        //
        // Consider this example:
        //
        //      struct MyParams
        //      {
        //          IMaterial material;
        //          ILight lights[3];
        //      };
        //
        // This `MyParams` type introduces four existential object parameters:
        // one for `material` and three for `lights` (one for each array
        // element). This is consistent with the number of interface-type
        // "objects" that are being passed through to the shader.
        //
        SLANG_PARAMETER_CATEGORY_EXISTENTIAL_OBJECT_PARAM,

        // The register space offset for the sub-elements that occupies register spaces.
        SLANG_PARAMETER_CATEGORY_SUB_ELEMENT_REGISTER_SPACE,

        // The input_attachment_index subpass occupancy tracker
        SLANG_PARAMETER_CATEGORY_SUBPASS,

        // Metal tier-1 argument buffer element [[id]].
        SLANG_PARAMETER_CATEGORY_METAL_ARGUMENT_BUFFER_ELEMENT,

        // Metal [[attribute]] inputs.
        SLANG_PARAMETER_CATEGORY_METAL_ATTRIBUTE,

        // Metal [[payload]] inputs
        SLANG_PARAMETER_CATEGORY_METAL_PAYLOAD,

        //
        SLANG_PARAMETER_CATEGORY_COUNT,

        // Aliases for Metal-specific categories.
        SLANG_PARAMETER_CATEGORY_METAL_BUFFER = SLANG_PARAMETER_CATEGORY_CONSTANT_BUFFER,
        SLANG_PARAMETER_CATEGORY_METAL_TEXTURE = SLANG_PARAMETER_CATEGORY_SHADER_RESOURCE,
        SLANG_PARAMETER_CATEGORY_METAL_SAMPLER = SLANG_PARAMETER_CATEGORY_SAMPLER_STATE,

        // DEPRECATED:
        SLANG_PARAMETER_CATEGORY_VERTEX_INPUT = SLANG_PARAMETER_CATEGORY_VARYING_INPUT,
        SLANG_PARAMETER_CATEGORY_FRAGMENT_OUTPUT = SLANG_PARAMETER_CATEGORY_VARYING_OUTPUT,
        SLANG_PARAMETER_CATEGORY_COUNT_V1 = SLANG_PARAMETER_CATEGORY_SUBPASS,
    };

    /** Types of API-managed bindings that a parameter might use.

    `SlangBindingType` represents the distinct types of binding ranges that might be
    understood by an underlying graphics API or cross-API abstraction layer.
    Several of the enumeration cases here correspond to cases of `VkDescriptorType`
    defined by the Vulkan API. Note however that the values of this enumeration
    are not the same as those of any particular API.

    The `SlangBindingType` enumeration is distinct from `SlangParameterCategory`
    because `SlangParameterCategory` differentiates the types of parameters for
    the purposes of layout, where the layout rules of some targets will treat
    parameters of different types as occupying the same binding space for layout
    (e.g., in SPIR-V both a `Texture2D` and `SamplerState` use the same space of
    `binding` indices, and are not allowed to overlap), while those same types
    map to different types of bindings in the API (e.g., both textures and samplers
    use different `VkDescriptorType` values).

    When you want to answer "what register/binding did this parameter use?" you
    should use `SlangParameterCategory`.

    When you want to answer "what type of descriptor range should this parameter use?"
    you should use `SlangBindingType`.
    */
    typedef SlangUInt32 SlangBindingTypeIntegral;
    enum SlangBindingType : SlangBindingTypeIntegral
    {
        SLANG_BINDING_TYPE_UNKNOWN = 0,

        SLANG_BINDING_TYPE_SAMPLER,
        SLANG_BINDING_TYPE_TEXTURE,
        SLANG_BINDING_TYPE_CONSTANT_BUFFER,
        SLANG_BINDING_TYPE_PARAMETER_BLOCK,
        SLANG_BINDING_TYPE_TYPED_BUFFER,
        SLANG_BINDING_TYPE_RAW_BUFFER,
        SLANG_BINDING_TYPE_COMBINED_TEXTURE_SAMPLER,
        SLANG_BINDING_TYPE_INPUT_RENDER_TARGET,
        SLANG_BINDING_TYPE_INLINE_UNIFORM_DATA,
        SLANG_BINDING_TYPE_RAY_TRACING_ACCELERATION_STRUCTURE,

        SLANG_BINDING_TYPE_VARYING_INPUT,
        SLANG_BINDING_TYPE_VARYING_OUTPUT,

        SLANG_BINDING_TYPE_EXISTENTIAL_VALUE,
        SLANG_BINDING_TYPE_PUSH_CONSTANT,

        SLANG_BINDING_TYPE_MUTABLE_FLAG = 0x100,

        SLANG_BINDING_TYPE_MUTABLE_TETURE =
            SLANG_BINDING_TYPE_TEXTURE | SLANG_BINDING_TYPE_MUTABLE_FLAG,
        SLANG_BINDING_TYPE_MUTABLE_TYPED_BUFFER =
            SLANG_BINDING_TYPE_TYPED_BUFFER | SLANG_BINDING_TYPE_MUTABLE_FLAG,
        SLANG_BINDING_TYPE_MUTABLE_RAW_BUFFER =
            SLANG_BINDING_TYPE_RAW_BUFFER | SLANG_BINDING_TYPE_MUTABLE_FLAG,

        SLANG_BINDING_TYPE_BASE_MASK = 0x00FF,
        SLANG_BINDING_TYPE_EXT_MASK = 0xFF00,
    };

    typedef SlangUInt32 SlangLayoutRulesIntegral;
    enum SlangLayoutRules : SlangLayoutRulesIntegral
    {
        SLANG_LAYOUT_RULES_DEFAULT,
        SLANG_LAYOUT_RULES_METAL_ARGUMENT_BUFFER_TIER_2,
    };

    typedef SlangUInt32 SlangModifierIDIntegral;
    enum SlangModifierID : SlangModifierIDIntegral
    {
        SLANG_MODIFIER_SHARED,
        SLANG_MODIFIER_NO_DIFF,
        SLANG_MODIFIER_STATIC,
        SLANG_MODIFIER_CONST,
        SLANG_MODIFIER_EXPORT,
        SLANG_MODIFIER_EXTERN,
        SLANG_MODIFIER_DIFFERENTIABLE,
        SLANG_MODIFIER_MUTATING,
        SLANG_MODIFIER_IN,
        SLANG_MODIFIER_OUT,
        SLANG_MODIFIER_INOUT
    };

    typedef SlangUInt32 SlangImageFormatIntegral;
    enum SlangImageFormat : SlangImageFormatIntegral
    {
#define SLANG_FORMAT(NAME, DESC) SLANG_IMAGE_FORMAT_##NAME,
#include "slang-image-format-defs.h"
#undef SLANG_FORMAT
    };

#define SLANG_UNBOUNDED_SIZE (~size_t(0))

    // Shader Parameter Reflection

    typedef SlangReflectionVariableLayout SlangReflectionParameter;

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
namespace slang
{
struct ISession;
}
#endif

#include "slang-deprecated.h"

#ifdef __cplusplus

/* Helper interfaces for C++ users */
namespace slang
{
struct BufferReflection;
struct DeclReflection;
struct TypeLayoutReflection;
struct TypeReflection;
struct VariableLayoutReflection;
struct VariableReflection;
struct FunctionReflection;
struct GenericReflection;

union GenericArgReflection
{
    TypeReflection* typeVal;
    int64_t intVal;
    bool boolVal;
};

struct Attribute
{
    char const* getName()
    {
        return spReflectionUserAttribute_GetName((SlangReflectionAttribute*)this);
    }
    uint32_t getArgumentCount()
    {
        return (uint32_t)spReflectionUserAttribute_GetArgumentCount(
            (SlangReflectionAttribute*)this);
    }
    TypeReflection* getArgumentType(uint32_t index)
    {
        return (TypeReflection*)spReflectionUserAttribute_GetArgumentType(
            (SlangReflectionAttribute*)this,
            index);
    }
    SlangResult getArgumentValueInt(uint32_t index, int* value)
    {
        return spReflectionUserAttribute_GetArgumentValueInt(
            (SlangReflectionAttribute*)this,
            index,
            value);
    }
    SlangResult getArgumentValueFloat(uint32_t index, float* value)
    {
        return spReflectionUserAttribute_GetArgumentValueFloat(
            (SlangReflectionAttribute*)this,
            index,
            value);
    }
    const char* getArgumentValueString(uint32_t index, size_t* outSize)
    {
        return spReflectionUserAttribute_GetArgumentValueString(
            (SlangReflectionAttribute*)this,
            index,
            outSize);
    }
};

typedef Attribute UserAttribute;

struct TypeReflection
{
    enum class Kind
    {
        None = SLANG_TYPE_KIND_NONE,
        Struct = SLANG_TYPE_KIND_STRUCT,
        Array = SLANG_TYPE_KIND_ARRAY,
        Matrix = SLANG_TYPE_KIND_MATRIX,
        Vector = SLANG_TYPE_KIND_VECTOR,
        Scalar = SLANG_TYPE_KIND_SCALAR,
        ConstantBuffer = SLANG_TYPE_KIND_CONSTANT_BUFFER,
        Resource = SLANG_TYPE_KIND_RESOURCE,
        SamplerState = SLANG_TYPE_KIND_SAMPLER_STATE,
        TextureBuffer = SLANG_TYPE_KIND_TEXTURE_BUFFER,
        ShaderStorageBuffer = SLANG_TYPE_KIND_SHADER_STORAGE_BUFFER,
        ParameterBlock = SLANG_TYPE_KIND_PARAMETER_BLOCK,
        GenericTypeParameter = SLANG_TYPE_KIND_GENERIC_TYPE_PARAMETER,
        Interface = SLANG_TYPE_KIND_INTERFACE,
        OutputStream = SLANG_TYPE_KIND_OUTPUT_STREAM,
        Specialized = SLANG_TYPE_KIND_SPECIALIZED,
        Feedback = SLANG_TYPE_KIND_FEEDBACK,
        Pointer = SLANG_TYPE_KIND_POINTER,
        DynamicResource = SLANG_TYPE_KIND_DYNAMIC_RESOURCE,
    };

    enum ScalarType : SlangScalarTypeIntegral
    {
        None = SLANG_SCALAR_TYPE_NONE,
        Void = SLANG_SCALAR_TYPE_VOID,
        Bool = SLANG_SCALAR_TYPE_BOOL,
        Int32 = SLANG_SCALAR_TYPE_INT32,
        UInt32 = SLANG_SCALAR_TYPE_UINT32,
        Int64 = SLANG_SCALAR_TYPE_INT64,
        UInt64 = SLANG_SCALAR_TYPE_UINT64,
        Float16 = SLANG_SCALAR_TYPE_FLOAT16,
        Float32 = SLANG_SCALAR_TYPE_FLOAT32,
        Float64 = SLANG_SCALAR_TYPE_FLOAT64,
        Int8 = SLANG_SCALAR_TYPE_INT8,
        UInt8 = SLANG_SCALAR_TYPE_UINT8,
        Int16 = SLANG_SCALAR_TYPE_INT16,
        UInt16 = SLANG_SCALAR_TYPE_UINT16,
    };

    Kind getKind() { return (Kind)spReflectionType_GetKind((SlangReflectionType*)this); }

    // only useful if `getKind() == Kind::Struct`
    unsigned int getFieldCount()
    {
        return spReflectionType_GetFieldCount((SlangReflectionType*)this);
    }

    VariableReflection* getFieldByIndex(unsigned int index)
    {
        return (
            VariableReflection*)spReflectionType_GetFieldByIndex((SlangReflectionType*)this, index);
    }

    bool isArray() { return getKind() == TypeReflection::Kind::Array; }

    TypeReflection* unwrapArray()
    {
        TypeReflection* type = this;
        while (type->isArray())
        {
            type = type->getElementType();
        }
        return type;
    }

    // only useful if `getKind() == Kind::Array`
    size_t getElementCount()
    {
        return spReflectionType_GetElementCount((SlangReflectionType*)this);
    }

    size_t getTotalArrayElementCount()
    {
        if (!isArray())
            return 0;
        size_t result = 1;
        TypeReflection* type = this;
        for (;;)
        {
            if (!type->isArray())
                return result;

            result *= type->getElementCount();
            type = type->getElementType();
        }
    }

    TypeReflection* getElementType()
    {
        return (TypeReflection*)spReflectionType_GetElementType((SlangReflectionType*)this);
    }

    unsigned getRowCount() { return spReflectionType_GetRowCount((SlangReflectionType*)this); }

    unsigned getColumnCount()
    {
        return spReflectionType_GetColumnCount((SlangReflectionType*)this);
    }

    ScalarType getScalarType()
    {
        return (ScalarType)spReflectionType_GetScalarType((SlangReflectionType*)this);
    }

    TypeReflection* getResourceResultType()
    {
        return (TypeReflection*)spReflectionType_GetResourceResultType((SlangReflectionType*)this);
    }

    SlangResourceShape getResourceShape()
    {
        return spReflectionType_GetResourceShape((SlangReflectionType*)this);
    }

    SlangResourceAccess getResourceAccess()
    {
        return spReflectionType_GetResourceAccess((SlangReflectionType*)this);
    }

    char const* getName() { return spReflectionType_GetName((SlangReflectionType*)this); }

    SlangResult getFullName(ISlangBlob** outNameBlob)
    {
        return spReflectionType_GetFullName((SlangReflectionType*)this, outNameBlob);
    }

    unsigned int getUserAttributeCount()
    {
        return spReflectionType_GetUserAttributeCount((SlangReflectionType*)this);
    }

    UserAttribute* getUserAttributeByIndex(unsigned int index)
    {
        return (UserAttribute*)spReflectionType_GetUserAttribute((SlangReflectionType*)this, index);
    }

    UserAttribute* findAttributeByName(char const* name)
    {
        return (UserAttribute*)spReflectionType_FindUserAttributeByName(
            (SlangReflectionType*)this,
            name);
    }

    UserAttribute* findUserAttributeByName(char const* name) { return findAttributeByName(name); }

    TypeReflection* applySpecializations(GenericReflection* generic)
    {
        return (TypeReflection*)spReflectionType_applySpecializations(
            (SlangReflectionType*)this,
            (SlangReflectionGeneric*)generic);
    }

    GenericReflection* getGenericContainer()
    {
        return (GenericReflection*)spReflectionType_GetGenericContainer((SlangReflectionType*)this);
    }
};

enum ParameterCategory : SlangParameterCategoryIntegral
{
    // TODO: these aren't scoped...
    None = SLANG_PARAMETER_CATEGORY_NONE,
    Mixed = SLANG_PARAMETER_CATEGORY_MIXED,
    ConstantBuffer = SLANG_PARAMETER_CATEGORY_CONSTANT_BUFFER,
    ShaderResource = SLANG_PARAMETER_CATEGORY_SHADER_RESOURCE,
    UnorderedAccess = SLANG_PARAMETER_CATEGORY_UNORDERED_ACCESS,
    VaryingInput = SLANG_PARAMETER_CATEGORY_VARYING_INPUT,
    VaryingOutput = SLANG_PARAMETER_CATEGORY_VARYING_OUTPUT,
    SamplerState = SLANG_PARAMETER_CATEGORY_SAMPLER_STATE,
    Uniform = SLANG_PARAMETER_CATEGORY_UNIFORM,
    DescriptorTableSlot = SLANG_PARAMETER_CATEGORY_DESCRIPTOR_TABLE_SLOT,
    SpecializationConstant = SLANG_PARAMETER_CATEGORY_SPECIALIZATION_CONSTANT,
    PushConstantBuffer = SLANG_PARAMETER_CATEGORY_PUSH_CONSTANT_BUFFER,
    RegisterSpace = SLANG_PARAMETER_CATEGORY_REGISTER_SPACE,
    GenericResource = SLANG_PARAMETER_CATEGORY_GENERIC,

    RayPayload = SLANG_PARAMETER_CATEGORY_RAY_PAYLOAD,
    HitAttributes = SLANG_PARAMETER_CATEGORY_HIT_ATTRIBUTES,
    CallablePayload = SLANG_PARAMETER_CATEGORY_CALLABLE_PAYLOAD,

    ShaderRecord = SLANG_PARAMETER_CATEGORY_SHADER_RECORD,

    ExistentialTypeParam = SLANG_PARAMETER_CATEGORY_EXISTENTIAL_TYPE_PARAM,
    ExistentialObjectParam = SLANG_PARAMETER_CATEGORY_EXISTENTIAL_OBJECT_PARAM,

    SubElementRegisterSpace = SLANG_PARAMETER_CATEGORY_SUB_ELEMENT_REGISTER_SPACE,

    InputAttachmentIndex = SLANG_PARAMETER_CATEGORY_SUBPASS,

    MetalBuffer = SLANG_PARAMETER_CATEGORY_CONSTANT_BUFFER,
    MetalTexture = SLANG_PARAMETER_CATEGORY_METAL_TEXTURE,
    MetalArgumentBufferElement = SLANG_PARAMETER_CATEGORY_METAL_ARGUMENT_BUFFER_ELEMENT,
    MetalAttribute = SLANG_PARAMETER_CATEGORY_METAL_ATTRIBUTE,
    MetalPayload = SLANG_PARAMETER_CATEGORY_METAL_PAYLOAD,

    // DEPRECATED:
    VertexInput = SLANG_PARAMETER_CATEGORY_VERTEX_INPUT,
    FragmentOutput = SLANG_PARAMETER_CATEGORY_FRAGMENT_OUTPUT,
};

enum class BindingType : SlangBindingTypeIntegral
{
    Unknown = SLANG_BINDING_TYPE_UNKNOWN,

    Sampler = SLANG_BINDING_TYPE_SAMPLER,
    Texture = SLANG_BINDING_TYPE_TEXTURE,
    ConstantBuffer = SLANG_BINDING_TYPE_CONSTANT_BUFFER,
    ParameterBlock = SLANG_BINDING_TYPE_PARAMETER_BLOCK,
    TypedBuffer = SLANG_BINDING_TYPE_TYPED_BUFFER,
    RawBuffer = SLANG_BINDING_TYPE_RAW_BUFFER,
    CombinedTextureSampler = SLANG_BINDING_TYPE_COMBINED_TEXTURE_SAMPLER,
    InputRenderTarget = SLANG_BINDING_TYPE_INPUT_RENDER_TARGET,
    InlineUniformData = SLANG_BINDING_TYPE_INLINE_UNIFORM_DATA,
    RayTracingAccelerationStructure = SLANG_BINDING_TYPE_RAY_TRACING_ACCELERATION_STRUCTURE,
    VaryingInput = SLANG_BINDING_TYPE_VARYING_INPUT,
    VaryingOutput = SLANG_BINDING_TYPE_VARYING_OUTPUT,
    ExistentialValue = SLANG_BINDING_TYPE_EXISTENTIAL_VALUE,
    PushConstant = SLANG_BINDING_TYPE_PUSH_CONSTANT,

    MutableFlag = SLANG_BINDING_TYPE_MUTABLE_FLAG,

    MutableTexture = SLANG_BINDING_TYPE_MUTABLE_TETURE,
    MutableTypedBuffer = SLANG_BINDING_TYPE_MUTABLE_TYPED_BUFFER,
    MutableRawBuffer = SLANG_BINDING_TYPE_MUTABLE_RAW_BUFFER,

    BaseMask = SLANG_BINDING_TYPE_BASE_MASK,
    ExtMask = SLANG_BINDING_TYPE_EXT_MASK,
};

struct TypeLayoutReflection
{
    TypeReflection* getType()
    {
        return (TypeReflection*)spReflectionTypeLayout_GetType((SlangReflectionTypeLayout*)this);
    }

    TypeReflection::Kind getKind()
    {
        return (TypeReflection::Kind)spReflectionTypeLayout_getKind(
            (SlangReflectionTypeLayout*)this);
    }

    size_t getSize(SlangParameterCategory category)
    {
        return spReflectionTypeLayout_GetSize((SlangReflectionTypeLayout*)this, category);
    }

    size_t getStride(SlangParameterCategory category)
    {
        return spReflectionTypeLayout_GetStride((SlangReflectionTypeLayout*)this, category);
    }

    int32_t getAlignment(SlangParameterCategory category)
    {
        return spReflectionTypeLayout_getAlignment((SlangReflectionTypeLayout*)this, category);
    }

    size_t getSize(slang::ParameterCategory category = slang::ParameterCategory::Uniform)
    {
        return spReflectionTypeLayout_GetSize(
            (SlangReflectionTypeLayout*)this,
            (SlangParameterCategory)category);
    }

    size_t getStride(slang::ParameterCategory category = slang::ParameterCategory::Uniform)
    {
        return spReflectionTypeLayout_GetStride(
            (SlangReflectionTypeLayout*)this,
            (SlangParameterCategory)category);
    }

    int32_t getAlignment(slang::ParameterCategory category = slang::ParameterCategory::Uniform)
    {
        return spReflectionTypeLayout_getAlignment(
            (SlangReflectionTypeLayout*)this,
            (SlangParameterCategory)category);
    }


    unsigned int getFieldCount()
    {
        return spReflectionTypeLayout_GetFieldCount((SlangReflectionTypeLayout*)this);
    }

    VariableLayoutReflection* getFieldByIndex(unsigned int index)
    {
        return (VariableLayoutReflection*)spReflectionTypeLayout_GetFieldByIndex(
            (SlangReflectionTypeLayout*)this,
            index);
    }

    SlangInt findFieldIndexByName(char const* nameBegin, char const* nameEnd = nullptr)
    {
        return spReflectionTypeLayout_findFieldIndexByName(
            (SlangReflectionTypeLayout*)this,
            nameBegin,
            nameEnd);
    }

    VariableLayoutReflection* getExplicitCounter()
    {
        return (VariableLayoutReflection*)spReflectionTypeLayout_GetExplicitCounter(
            (SlangReflectionTypeLayout*)this);
    }

    bool isArray() { return getType()->isArray(); }

    TypeLayoutReflection* unwrapArray()
    {
        TypeLayoutReflection* typeLayout = this;
        while (typeLayout->isArray())
        {
            typeLayout = typeLayout->getElementTypeLayout();
        }
        return typeLayout;
    }

    // only useful if `getKind() == Kind::Array`
    size_t getElementCount() { return getType()->getElementCount(); }

    size_t getTotalArrayElementCount() { return getType()->getTotalArrayElementCount(); }

    size_t getElementStride(SlangParameterCategory category)
    {
        return spReflectionTypeLayout_GetElementStride((SlangReflectionTypeLayout*)this, category);
    }

    TypeLayoutReflection* getElementTypeLayout()
    {
        return (TypeLayoutReflection*)spReflectionTypeLayout_GetElementTypeLayout(
            (SlangReflectionTypeLayout*)this);
    }

    VariableLayoutReflection* getElementVarLayout()
    {
        return (VariableLayoutReflection*)spReflectionTypeLayout_GetElementVarLayout(
            (SlangReflectionTypeLayout*)this);
    }

    VariableLayoutReflection* getContainerVarLayout()
    {
        return (VariableLayoutReflection*)spReflectionTypeLayout_getContainerVarLayout(
            (SlangReflectionTypeLayout*)this);
    }

    // How is this type supposed to be bound?
    ParameterCategory getParameterCategory()
    {
        return (ParameterCategory)spReflectionTypeLayout_GetParameterCategory(
            (SlangReflectionTypeLayout*)this);
    }

    unsigned int getCategoryCount()
    {
        return spReflectionTypeLayout_GetCategoryCount((SlangReflectionTypeLayout*)this);
    }

    ParameterCategory getCategoryByIndex(unsigned int index)
    {
        return (ParameterCategory)spReflectionTypeLayout_GetCategoryByIndex(
            (SlangReflectionTypeLayout*)this,
            index);
    }

    unsigned getRowCount() { return getType()->getRowCount(); }

    unsigned getColumnCount() { return getType()->getColumnCount(); }

    TypeReflection::ScalarType getScalarType() { return getType()->getScalarType(); }

    TypeReflection* getResourceResultType() { return getType()->getResourceResultType(); }

    SlangResourceShape getResourceShape() { return getType()->getResourceShape(); }

    SlangResourceAccess getResourceAccess() { return getType()->getResourceAccess(); }

    char const* getName() { return getType()->getName(); }

    SlangMatrixLayoutMode getMatrixLayoutMode()
    {
        return spReflectionTypeLayout_GetMatrixLayoutMode((SlangReflectionTypeLayout*)this);
    }

    int getGenericParamIndex()
    {
        return spReflectionTypeLayout_getGenericParamIndex((SlangReflectionTypeLayout*)this);
    }

    TypeLayoutReflection* getPendingDataTypeLayout()
    {
        return (TypeLayoutReflection*)spReflectionTypeLayout_getPendingDataTypeLayout(
            (SlangReflectionTypeLayout*)this);
    }

    VariableLayoutReflection* getSpecializedTypePendingDataVarLayout()
    {
        return (VariableLayoutReflection*)
            spReflectionTypeLayout_getSpecializedTypePendingDataVarLayout(
                (SlangReflectionTypeLayout*)this);
    }

    SlangInt getBindingRangeCount()
    {
        return spReflectionTypeLayout_getBindingRangeCount((SlangReflectionTypeLayout*)this);
    }

    BindingType getBindingRangeType(SlangInt index)
    {
        return (BindingType)spReflectionTypeLayout_getBindingRangeType(
            (SlangReflectionTypeLayout*)this,
            index);
    }

    bool isBindingRangeSpecializable(SlangInt index)
    {
        return (bool)spReflectionTypeLayout_isBindingRangeSpecializable(
            (SlangReflectionTypeLayout*)this,
            index);
    }

    SlangInt getBindingRangeBindingCount(SlangInt index)
    {
        return spReflectionTypeLayout_getBindingRangeBindingCount(
            (SlangReflectionTypeLayout*)this,
            index);
    }

    /*
    SlangInt getBindingRangeIndexOffset(SlangInt index)
    {
        return spReflectionTypeLayout_getBindingRangeIndexOffset(
            (SlangReflectionTypeLayout*) this,
            index);
    }

    SlangInt getBindingRangeSpaceOffset(SlangInt index)
    {
        return spReflectionTypeLayout_getBindingRangeSpaceOffset(
            (SlangReflectionTypeLayout*) this,
            index);
    }
    */

    SlangInt getFieldBindingRangeOffset(SlangInt fieldIndex)
    {
        return spReflectionTypeLayout_getFieldBindingRangeOffset(
            (SlangReflectionTypeLayout*)this,
            fieldIndex);
    }

    SlangInt getExplicitCounterBindingRangeOffset()
    {
        return spReflectionTypeLayout_getExplicitCounterBindingRangeOffset(
            (SlangReflectionTypeLayout*)this);
    }

    TypeLayoutReflection* getBindingRangeLeafTypeLayout(SlangInt index)
    {
        return (TypeLayoutReflection*)spReflectionTypeLayout_getBindingRangeLeafTypeLayout(
            (SlangReflectionTypeLayout*)this,
            index);
    }

    VariableReflection* getBindingRangeLeafVariable(SlangInt index)
    {
        return (VariableReflection*)spReflectionTypeLayout_getBindingRangeLeafVariable(
            (SlangReflectionTypeLayout*)this,
            index);
    }

    SlangImageFormat getBindingRangeImageFormat(SlangInt index)
    {
        return spReflectionTypeLayout_getBindingRangeImageFormat(
            (SlangReflectionTypeLayout*)this,
            index);
    }

    SlangInt getBindingRangeDescriptorSetIndex(SlangInt index)
    {
        return spReflectionTypeLayout_getBindingRangeDescriptorSetIndex(
            (SlangReflectionTypeLayout*)this,
            index);
    }

    SlangInt getBindingRangeFirstDescriptorRangeIndex(SlangInt index)
    {
        return spReflectionTypeLayout_getBindingRangeFirstDescriptorRangeIndex(
            (SlangReflectionTypeLayout*)this,
            index);
    }

    SlangInt getBindingRangeDescriptorRangeCount(SlangInt index)
    {
        return spReflectionTypeLayout_getBindingRangeDescriptorRangeCount(
            (SlangReflectionTypeLayout*)this,
            index);
    }

    SlangInt getDescriptorSetCount()
    {
        return spReflectionTypeLayout_getDescriptorSetCount((SlangReflectionTypeLayout*)this);
    }

    SlangInt getDescriptorSetSpaceOffset(SlangInt setIndex)
    {
        return spReflectionTypeLayout_getDescriptorSetSpaceOffset(
            (SlangReflectionTypeLayout*)this,
            setIndex);
    }

    SlangInt getDescriptorSetDescriptorRangeCount(SlangInt setIndex)
    {
        return spReflectionTypeLayout_getDescriptorSetDescriptorRangeCount(
            (SlangReflectionTypeLayout*)this,
            setIndex);
    }

    SlangInt getDescriptorSetDescriptorRangeIndexOffset(SlangInt setIndex, SlangInt rangeIndex)
    {
        return spReflectionTypeLayout_getDescriptorSetDescriptorRangeIndexOffset(
            (SlangReflectionTypeLayout*)this,
            setIndex,
            rangeIndex);
    }

    SlangInt getDescriptorSetDescriptorRangeDescriptorCount(SlangInt setIndex, SlangInt rangeIndex)
    {
        return spReflectionTypeLayout_getDescriptorSetDescriptorRangeDescriptorCount(
            (SlangReflectionTypeLayout*)this,
            setIndex,
            rangeIndex);
    }

    BindingType getDescriptorSetDescriptorRangeType(SlangInt setIndex, SlangInt rangeIndex)
    {
        return (BindingType)spReflectionTypeLayout_getDescriptorSetDescriptorRangeType(
            (SlangReflectionTypeLayout*)this,
            setIndex,
            rangeIndex);
    }

    ParameterCategory getDescriptorSetDescriptorRangeCategory(
        SlangInt setIndex,
        SlangInt rangeIndex)
    {
        return (ParameterCategory)spReflectionTypeLayout_getDescriptorSetDescriptorRangeCategory(
            (SlangReflectionTypeLayout*)this,
            setIndex,
            rangeIndex);
    }

    SlangInt getSubObjectRangeCount()
    {
        return spReflectionTypeLayout_getSubObjectRangeCount((SlangReflectionTypeLayout*)this);
    }

    SlangInt getSubObjectRangeBindingRangeIndex(SlangInt subObjectRangeIndex)
    {
        return spReflectionTypeLayout_getSubObjectRangeBindingRangeIndex(
            (SlangReflectionTypeLayout*)this,
            subObjectRangeIndex);
    }

    SlangInt getSubObjectRangeSpaceOffset(SlangInt subObjectRangeIndex)
    {
        return spReflectionTypeLayout_getSubObjectRangeSpaceOffset(
            (SlangReflectionTypeLayout*)this,
            subObjectRangeIndex);
    }

    VariableLayoutReflection* getSubObjectRangeOffset(SlangInt subObjectRangeIndex)
    {
        return (VariableLayoutReflection*)spReflectionTypeLayout_getSubObjectRangeOffset(
            (SlangReflectionTypeLayout*)this,
            subObjectRangeIndex);
    }
};

struct Modifier
{
    enum ID : SlangModifierIDIntegral
    {
        Shared = SLANG_MODIFIER_SHARED,
        NoDiff = SLANG_MODIFIER_NO_DIFF,
        Static = SLANG_MODIFIER_STATIC,
        Const = SLANG_MODIFIER_CONST,
        Export = SLANG_MODIFIER_EXPORT,
        Extern = SLANG_MODIFIER_EXTERN,
        Differentiable = SLANG_MODIFIER_DIFFERENTIABLE,
        Mutating = SLANG_MODIFIER_MUTATING,
        In = SLANG_MODIFIER_IN,
        Out = SLANG_MODIFIER_OUT,
        InOut = SLANG_MODIFIER_INOUT
    };
};

struct VariableReflection
{
    char const* getName() { return spReflectionVariable_GetName((SlangReflectionVariable*)this); }

    TypeReflection* getType()
    {
        return (TypeReflection*)spReflectionVariable_GetType((SlangReflectionVariable*)this);
    }

    Modifier* findModifier(Modifier::ID id)
    {
        return (Modifier*)spReflectionVariable_FindModifier(
            (SlangReflectionVariable*)this,
            (SlangModifierID)id);
    }

    unsigned int getUserAttributeCount()
    {
        return spReflectionVariable_GetUserAttributeCount((SlangReflectionVariable*)this);
    }

    Attribute* getUserAttributeByIndex(unsigned int index)
    {
        return (UserAttribute*)spReflectionVariable_GetUserAttribute(
            (SlangReflectionVariable*)this,
            index);
    }

    Attribute* findAttributeByName(SlangSession* globalSession, char const* name)
    {
        return (UserAttribute*)spReflectionVariable_FindUserAttributeByName(
            (SlangReflectionVariable*)this,
            globalSession,
            name);
    }

    Attribute* findUserAttributeByName(SlangSession* globalSession, char const* name)
    {
        return findAttributeByName(globalSession, name);
    }

    bool hasDefaultValue()
    {
        return spReflectionVariable_HasDefaultValue((SlangReflectionVariable*)this);
    }

    SlangResult getDefaultValueInt(int64_t* value)
    {
        return spReflectionVariable_GetDefaultValueInt((SlangReflectionVariable*)this, value);
    }

    GenericReflection* getGenericContainer()
    {
        return (GenericReflection*)spReflectionVariable_GetGenericContainer(
            (SlangReflectionVariable*)this);
    }

    VariableReflection* applySpecializations(GenericReflection* generic)
    {
        return (VariableReflection*)spReflectionVariable_applySpecializations(
            (SlangReflectionVariable*)this,
            (SlangReflectionGeneric*)generic);
    }
};

struct VariableLayoutReflection
{
    VariableReflection* getVariable()
    {
        return (VariableReflection*)spReflectionVariableLayout_GetVariable(
            (SlangReflectionVariableLayout*)this);
    }

    char const* getName() { return getVariable()->getName(); }

    Modifier* findModifier(Modifier::ID id) { return getVariable()->findModifier(id); }

    TypeLayoutReflection* getTypeLayout()
    {
        return (TypeLayoutReflection*)spReflectionVariableLayout_GetTypeLayout(
            (SlangReflectionVariableLayout*)this);
    }

    ParameterCategory getCategory() { return getTypeLayout()->getParameterCategory(); }

    unsigned int getCategoryCount() { return getTypeLayout()->getCategoryCount(); }

    ParameterCategory getCategoryByIndex(unsigned int index)
    {
        return getTypeLayout()->getCategoryByIndex(index);
    }


    size_t getOffset(SlangParameterCategory category)
    {
        return spReflectionVariableLayout_GetOffset((SlangReflectionVariableLayout*)this, category);
    }
    size_t getOffset(slang::ParameterCategory category = slang::ParameterCategory::Uniform)
    {
        return spReflectionVariableLayout_GetOffset(
            (SlangReflectionVariableLayout*)this,
            (SlangParameterCategory)category);
    }


    TypeReflection* getType() { return getVariable()->getType(); }

    unsigned getBindingIndex()
    {
        return spReflectionParameter_GetBindingIndex((SlangReflectionVariableLayout*)this);
    }

    unsigned getBindingSpace()
    {
        return spReflectionParameter_GetBindingSpace((SlangReflectionVariableLayout*)this);
    }

    size_t getBindingSpace(SlangParameterCategory category)
    {
        return spReflectionVariableLayout_GetSpace((SlangReflectionVariableLayout*)this, category);
    }
    size_t getBindingSpace(slang::ParameterCategory category)
    {
        return spReflectionVariableLayout_GetSpace(
            (SlangReflectionVariableLayout*)this,
            (SlangParameterCategory)category);
    }

    SlangImageFormat getImageFormat()
    {
        return spReflectionVariableLayout_GetImageFormat((SlangReflectionVariableLayout*)this);
    }

    char const* getSemanticName()
    {
        return spReflectionVariableLayout_GetSemanticName((SlangReflectionVariableLayout*)this);
    }

    size_t getSemanticIndex()
    {
        return spReflectionVariableLayout_GetSemanticIndex((SlangReflectionVariableLayout*)this);
    }

    SlangStage getStage()
    {
        return spReflectionVariableLayout_getStage((SlangReflectionVariableLayout*)this);
    }

    VariableLayoutReflection* getPendingDataLayout()
    {
        return (VariableLayoutReflection*)spReflectionVariableLayout_getPendingDataLayout(
            (SlangReflectionVariableLayout*)this);
    }
};

struct FunctionReflection
{
    char const* getName() { return spReflectionFunction_GetName((SlangReflectionFunction*)this); }

    TypeReflection* getReturnType()
    {
        return (TypeReflection*)spReflectionFunction_GetResultType((SlangReflectionFunction*)this);
    }

    unsigned int getParameterCount()
    {
        return spReflectionFunction_GetParameterCount((SlangReflectionFunction*)this);
    }

    VariableReflection* getParameterByIndex(unsigned int index)
    {
        return (VariableReflection*)spReflectionFunction_GetParameter(
            (SlangReflectionFunction*)this,
            index);
    }

    unsigned int getUserAttributeCount()
    {
        return spReflectionFunction_GetUserAttributeCount((SlangReflectionFunction*)this);
    }
    Attribute* getUserAttributeByIndex(unsigned int index)
    {
        return (
            Attribute*)spReflectionFunction_GetUserAttribute((SlangReflectionFunction*)this, index);
    }
    Attribute* findAttributeByName(SlangSession* globalSession, char const* name)
    {
        return (Attribute*)spReflectionFunction_FindUserAttributeByName(
            (SlangReflectionFunction*)this,
            globalSession,
            name);
    }
    Attribute* findUserAttributeByName(SlangSession* globalSession, char const* name)
    {
        return findAttributeByName(globalSession, name);
    }
    Modifier* findModifier(Modifier::ID id)
    {
        return (Modifier*)spReflectionFunction_FindModifier(
            (SlangReflectionFunction*)this,
            (SlangModifierID)id);
    }

    GenericReflection* getGenericContainer()
    {
        return (GenericReflection*)spReflectionFunction_GetGenericContainer(
            (SlangReflectionFunction*)this);
    }

    FunctionReflection* applySpecializations(GenericReflection* generic)
    {
        return (FunctionReflection*)spReflectionFunction_applySpecializations(
            (SlangReflectionFunction*)this,
            (SlangReflectionGeneric*)generic);
    }

    FunctionReflection* specializeWithArgTypes(unsigned int argCount, TypeReflection* const* types)
    {
        return (FunctionReflection*)spReflectionFunction_specializeWithArgTypes(
            (SlangReflectionFunction*)this,
            argCount,
            (SlangReflectionType* const*)types);
    }

    bool isOverloaded()
    {
        return spReflectionFunction_isOverloaded((SlangReflectionFunction*)this);
    }

    unsigned int getOverloadCount()
    {
        return spReflectionFunction_getOverloadCount((SlangReflectionFunction*)this);
    }

    FunctionReflection* getOverload(unsigned int index)
    {
        return (FunctionReflection*)spReflectionFunction_getOverload(
            (SlangReflectionFunction*)this,
            index);
    }
};

struct GenericReflection
{

    DeclReflection* asDecl()
    {
        return (DeclReflection*)spReflectionGeneric_asDecl((SlangReflectionGeneric*)this);
    }

    char const* getName() { return spReflectionGeneric_GetName((SlangReflectionGeneric*)this); }

    unsigned int getTypeParameterCount()
    {
        return spReflectionGeneric_GetTypeParameterCount((SlangReflectionGeneric*)this);
    }

    VariableReflection* getTypeParameter(unsigned index)
    {
        return (VariableReflection*)spReflectionGeneric_GetTypeParameter(
            (SlangReflectionGeneric*)this,
            index);
    }

    unsigned int getValueParameterCount()
    {
        return spReflectionGeneric_GetValueParameterCount((SlangReflectionGeneric*)this);
    }

    VariableReflection* getValueParameter(unsigned index)
    {
        return (VariableReflection*)spReflectionGeneric_GetValueParameter(
            (SlangReflectionGeneric*)this,
            index);
    }

    unsigned int getTypeParameterConstraintCount(VariableReflection* typeParam)
    {
        return spReflectionGeneric_GetTypeParameterConstraintCount(
            (SlangReflectionGeneric*)this,
            (SlangReflectionVariable*)typeParam);
    }

    TypeReflection* getTypeParameterConstraintType(VariableReflection* typeParam, unsigned index)
    {
        return (TypeReflection*)spReflectionGeneric_GetTypeParameterConstraintType(
            (SlangReflectionGeneric*)this,
            (SlangReflectionVariable*)typeParam,
            index);
    }

    DeclReflection* getInnerDecl()
    {
        return (DeclReflection*)spReflectionGeneric_GetInnerDecl((SlangReflectionGeneric*)this);
    }

    SlangDeclKind getInnerKind()
    {
        return spReflectionGeneric_GetInnerKind((SlangReflectionGeneric*)this);
    }

    GenericReflection* getOuterGenericContainer()
    {
        return (GenericReflection*)spReflectionGeneric_GetOuterGenericContainer(
            (SlangReflectionGeneric*)this);
    }

    TypeReflection* getConcreteType(VariableReflection* typeParam)
    {
        return (TypeReflection*)spReflectionGeneric_GetConcreteType(
            (SlangReflectionGeneric*)this,
            (SlangReflectionVariable*)typeParam);
    }

    int64_t getConcreteIntVal(VariableReflection* valueParam)
    {
        return spReflectionGeneric_GetConcreteIntVal(
            (SlangReflectionGeneric*)this,
            (SlangReflectionVariable*)valueParam);
    }

    GenericReflection* applySpecializations(GenericReflection* generic)
    {
        return (GenericReflection*)spReflectionGeneric_applySpecializations(
            (SlangReflectionGeneric*)this,
            (SlangReflectionGeneric*)generic);
    }
};

struct EntryPointReflection
{
    char const* getName()
    {
        return spReflectionEntryPoint_getName((SlangReflectionEntryPoint*)this);
    }

    char const* getNameOverride()
    {
        return spReflectionEntryPoint_getNameOverride((SlangReflectionEntryPoint*)this);
    }

    unsigned getParameterCount()
    {
        return spReflectionEntryPoint_getParameterCount((SlangReflectionEntryPoint*)this);
    }

    FunctionReflection* getFunction()
    {
        return (FunctionReflection*)spReflectionEntryPoint_getFunction(
            (SlangReflectionEntryPoint*)this);
    }

    VariableLayoutReflection* getParameterByIndex(unsigned index)
    {
        return (VariableLayoutReflection*)spReflectionEntryPoint_getParameterByIndex(
            (SlangReflectionEntryPoint*)this,
            index);
    }

    SlangStage getStage()
    {
        return spReflectionEntryPoint_getStage((SlangReflectionEntryPoint*)this);
    }

    void getComputeThreadGroupSize(SlangUInt axisCount, SlangUInt* outSizeAlongAxis)
    {
        return spReflectionEntryPoint_getComputeThreadGroupSize(
            (SlangReflectionEntryPoint*)this,
            axisCount,
            outSizeAlongAxis);
    }

    void getComputeWaveSize(SlangUInt* outWaveSize)
    {
        return spReflectionEntryPoint_getComputeWaveSize(
            (SlangReflectionEntryPoint*)this,
            outWaveSize);
    }

    bool usesAnySampleRateInput()
    {
        return 0 != spReflectionEntryPoint_usesAnySampleRateInput((SlangReflectionEntryPoint*)this);
    }

    VariableLayoutReflection* getVarLayout()
    {
        return (VariableLayoutReflection*)spReflectionEntryPoint_getVarLayout(
            (SlangReflectionEntryPoint*)this);
    }

    TypeLayoutReflection* getTypeLayout() { return getVarLayout()->getTypeLayout(); }

    VariableLayoutReflection* getResultVarLayout()
    {
        return (VariableLayoutReflection*)spReflectionEntryPoint_getResultVarLayout(
            (SlangReflectionEntryPoint*)this);
    }

    bool hasDefaultConstantBuffer()
    {
        return spReflectionEntryPoint_hasDefaultConstantBuffer((SlangReflectionEntryPoint*)this) !=
               0;
    }
};

typedef EntryPointReflection EntryPointLayout;

struct TypeParameterReflection
{
    char const* getName()
    {
        return spReflectionTypeParameter_GetName((SlangReflectionTypeParameter*)this);
    }
    unsigned getIndex()
    {
        return spReflectionTypeParameter_GetIndex((SlangReflectionTypeParameter*)this);
    }
    unsigned getConstraintCount()
    {
        return spReflectionTypeParameter_GetConstraintCount((SlangReflectionTypeParameter*)this);
    }
    TypeReflection* getConstraintByIndex(int index)
    {
        return (TypeReflection*)spReflectionTypeParameter_GetConstraintByIndex(
            (SlangReflectionTypeParameter*)this,
            index);
    }
};

enum class LayoutRules : SlangLayoutRulesIntegral
{
    Default = SLANG_LAYOUT_RULES_DEFAULT,
    MetalArgumentBufferTier2 = SLANG_LAYOUT_RULES_METAL_ARGUMENT_BUFFER_TIER_2,
};

typedef struct ShaderReflection ProgramLayout;
typedef enum SlangReflectionGenericArgType GenericArgType;

struct ShaderReflection
{
    unsigned getParameterCount() { return spReflection_GetParameterCount((SlangReflection*)this); }

    unsigned getTypeParameterCount()
    {
        return spReflection_GetTypeParameterCount((SlangReflection*)this);
    }

    slang::ISession* getSession() { return spReflection_GetSession((SlangReflection*)this); }

    TypeParameterReflection* getTypeParameterByIndex(unsigned index)
    {
        return (TypeParameterReflection*)spReflection_GetTypeParameterByIndex(
            (SlangReflection*)this,
            index);
    }

    TypeParameterReflection* findTypeParameter(char const* name)
    {
        return (
            TypeParameterReflection*)spReflection_FindTypeParameter((SlangReflection*)this, name);
    }

    VariableLayoutReflection* getParameterByIndex(unsigned index)
    {
        return (VariableLayoutReflection*)spReflection_GetParameterByIndex(
            (SlangReflection*)this,
            index);
    }

    static ProgramLayout* get(SlangCompileRequest* request)
    {
        return (ProgramLayout*)spGetReflection(request);
    }

    SlangUInt getEntryPointCount()
    {
        return spReflection_getEntryPointCount((SlangReflection*)this);
    }

    EntryPointReflection* getEntryPointByIndex(SlangUInt index)
    {
        return (
            EntryPointReflection*)spReflection_getEntryPointByIndex((SlangReflection*)this, index);
    }

    SlangUInt getGlobalConstantBufferBinding()
    {
        return spReflection_getGlobalConstantBufferBinding((SlangReflection*)this);
    }

    size_t getGlobalConstantBufferSize()
    {
        return spReflection_getGlobalConstantBufferSize((SlangReflection*)this);
    }

    TypeReflection* findTypeByName(const char* name)
    {
        return (TypeReflection*)spReflection_FindTypeByName((SlangReflection*)this, name);
    }

    FunctionReflection* findFunctionByName(const char* name)
    {
        return (FunctionReflection*)spReflection_FindFunctionByName((SlangReflection*)this, name);
    }

    FunctionReflection* findFunctionByNameInType(TypeReflection* type, const char* name)
    {
        return (FunctionReflection*)spReflection_FindFunctionByNameInType(
            (SlangReflection*)this,
            (SlangReflectionType*)type,
            name);
    }

    VariableReflection* findVarByNameInType(TypeReflection* type, const char* name)
    {
        return (VariableReflection*)spReflection_FindVarByNameInType(
            (SlangReflection*)this,
            (SlangReflectionType*)type,
            name);
    }

    TypeLayoutReflection* getTypeLayout(
        TypeReflection* type,
        LayoutRules rules = LayoutRules::Default)
    {
        return (TypeLayoutReflection*)spReflection_GetTypeLayout(
            (SlangReflection*)this,
            (SlangReflectionType*)type,
            SlangLayoutRules(rules));
    }

    EntryPointReflection* findEntryPointByName(const char* name)
    {
        return (
            EntryPointReflection*)spReflection_findEntryPointByName((SlangReflection*)this, name);
    }

    TypeReflection* specializeType(
        TypeReflection* type,
        SlangInt specializationArgCount,
        TypeReflection* const* specializationArgs,
        ISlangBlob** outDiagnostics)
    {
        return (TypeReflection*)spReflection_specializeType(
            (SlangReflection*)this,
            (SlangReflectionType*)type,
            specializationArgCount,
            (SlangReflectionType* const*)specializationArgs,
            outDiagnostics);
    }

    GenericReflection* specializeGeneric(
        GenericReflection* generic,
        SlangInt specializationArgCount,
        GenericArgType const* specializationArgTypes,
        GenericArgReflection const* specializationArgVals,
        ISlangBlob** outDiagnostics)
    {
        return (GenericReflection*)spReflection_specializeGeneric(
            (SlangReflection*)this,
            (SlangReflectionGeneric*)generic,
            specializationArgCount,
            (SlangReflectionGenericArgType const*)specializationArgTypes,
            (SlangReflectionGenericArg const*)specializationArgVals,
            outDiagnostics);
    }

    bool isSubType(TypeReflection* subType, TypeReflection* superType)
    {
        return spReflection_isSubType(
            (SlangReflection*)this,
            (SlangReflectionType*)subType,
            (SlangReflectionType*)superType);
    }

    SlangUInt getHashedStringCount() const
    {
        return spReflection_getHashedStringCount((SlangReflection*)this);
    }

    const char* getHashedString(SlangUInt index, size_t* outCount) const
    {
        return spReflection_getHashedString((SlangReflection*)this, index, outCount);
    }

    TypeLayoutReflection* getGlobalParamsTypeLayout()
    {
        return (TypeLayoutReflection*)spReflection_getGlobalParamsTypeLayout(
            (SlangReflection*)this);
    }

    VariableLayoutReflection* getGlobalParamsVarLayout()
    {
        return (VariableLayoutReflection*)spReflection_getGlobalParamsVarLayout(
            (SlangReflection*)this);
    }

    SlangResult toJson(ISlangBlob** outBlob)
    {
        return spReflection_ToJson((SlangReflection*)this, nullptr, outBlob);
    }
};


struct DeclReflection
{
    enum class Kind
    {
        Unsupported = SLANG_DECL_KIND_UNSUPPORTED_FOR_REFLECTION,
        Struct = SLANG_DECL_KIND_STRUCT,
        Func = SLANG_DECL_KIND_FUNC,
        Module = SLANG_DECL_KIND_MODULE,
        Generic = SLANG_DECL_KIND_GENERIC,
        Variable = SLANG_DECL_KIND_VARIABLE,
        Namespace = SLANG_DECL_KIND_NAMESPACE,
    };

    char const* getName() { return spReflectionDecl_getName((SlangReflectionDecl*)this); }

    Kind getKind() { return (Kind)spReflectionDecl_getKind((SlangReflectionDecl*)this); }

    unsigned int getChildrenCount()
    {
        return spReflectionDecl_getChildrenCount((SlangReflectionDecl*)this);
    }

    DeclReflection* getChild(unsigned int index)
    {
        return (DeclReflection*)spReflectionDecl_getChild((SlangReflectionDecl*)this, index);
    }

    TypeReflection* getType()
    {
        return (TypeReflection*)spReflection_getTypeFromDecl((SlangReflectionDecl*)this);
    }

    VariableReflection* asVariable()
    {
        return (VariableReflection*)spReflectionDecl_castToVariable((SlangReflectionDecl*)this);
    }

    FunctionReflection* asFunction()
    {
        return (FunctionReflection*)spReflectionDecl_castToFunction((SlangReflectionDecl*)this);
    }

    GenericReflection* asGeneric()
    {
        return (GenericReflection*)spReflectionDecl_castToGeneric((SlangReflectionDecl*)this);
    }

    DeclReflection* getParent()
    {
        return (DeclReflection*)spReflectionDecl_getParent((SlangReflectionDecl*)this);
    }

    template<Kind K>
    struct FilteredList
    {
        unsigned int count;
        DeclReflection* parent;

        struct FilteredIterator
        {
            DeclReflection* parent;
            unsigned int count;
            unsigned int index;

            DeclReflection* operator*() { return parent->getChild(index); }
            void operator++()
            {
                index++;
                while (index < count && !(parent->getChild(index)->getKind() == K))
                {
                    index++;
                }
            }
            bool operator!=(FilteredIterator const& other) { return index != other.index; }
        };

        // begin/end for range-based for that checks the kind
        FilteredIterator begin()
        {
            // Find the first child of the right kind
            unsigned int index = 0;
            while (index < count && !(parent->getChild(index)->getKind() == K))
            {
                index++;
            }
            return FilteredIterator{parent, count, index};
        }

        FilteredIterator end() { return FilteredIterator{parent, count, count}; }
    };

    template<Kind K>
    FilteredList<K> getChildrenOfKind()
    {
        return FilteredList<K>{getChildrenCount(), (DeclReflection*)this};
    }

    struct IteratedList
    {
        unsigned int count;
        DeclReflection* parent;

        struct Iterator
        {
            DeclReflection* parent;
            unsigned int count;
            unsigned int index;

            DeclReflection* operator*() { return parent->getChild(index); }
            void operator++() { index++; }
            bool operator!=(Iterator const& other) { return index != other.index; }
        };

        // begin/end for range-based for that checks the kind
        IteratedList::Iterator begin() { return IteratedList::Iterator{parent, count, 0}; }
        IteratedList::Iterator end() { return IteratedList::Iterator{parent, count, count}; }
    };

    IteratedList getChildren() { return IteratedList{getChildrenCount(), (DeclReflection*)this}; }
};

typedef uint32_t CompileCoreModuleFlags;
struct CompileCoreModuleFlag
{
    enum Enum : CompileCoreModuleFlags
    {
        WriteDocumentation = 0x1,
    };
};

typedef ISlangBlob IBlob;

struct IComponentType;
struct ITypeConformance;
struct IGlobalSession;
struct IModule;

struct SessionDesc;
struct SpecializationArg;
struct TargetDesc;

enum class BuiltinModuleName
{
    Core,
    GLSL
};

/** A global session for interaction with the Slang library.

An application may create and re-use a single global session across
multiple sessions, in order to amortize startups costs (in current
Slang this is mostly the cost of loading the Slang standard library).

The global session is currently *not* thread-safe and objects created from
a single global session should only be used from a single thread at
a time.
*/
struct IGlobalSession : public ISlangUnknown
{
    SLANG_COM_INTERFACE(0xc140b5fd, 0xc78, 0x452e, {0xba, 0x7c, 0x1a, 0x1e, 0x70, 0xc7, 0xf7, 0x1c})

    /** Create a new session for loading and compiling code.
     */
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    createSession(SessionDesc const& desc, ISession** outSession) = 0;

    /** Look up the internal ID of a profile by its `name`.

    Profile IDs are *not* guaranteed to be stable across versions
    of the Slang library, so clients are expected to look up
    profiles by name at runtime.
    */
    virtual SLANG_NO_THROW SlangProfileID SLANG_MCALL findProfile(char const* name) = 0;

    /** Set the path that downstream compilers (aka back end compilers) will
    be looked from.
    @param passThrough Identifies the downstream compiler
    @param path The path to find the downstream compiler (shared library/dll/executable)

    For back ends that are dlls/shared libraries, it will mean the path will
    be prefixed with the path when calls are made out to ISlangSharedLibraryLoader.
    For executables - it will look for executables along the path */
    virtual SLANG_NO_THROW void SLANG_MCALL
    setDownstreamCompilerPath(SlangPassThrough passThrough, char const* path) = 0;

    /** DEPRECATED: Use setLanguagePrelude

    Set the 'prelude' for generated code for a 'downstream compiler'.
    @param passThrough The downstream compiler for generated code that will have the prelude applied
    to it.
    @param preludeText The text added pre-pended verbatim before the generated source

    That for pass-through usage, prelude is not pre-pended, preludes are for code generation only.
    */
    virtual SLANG_NO_THROW void SLANG_MCALL
    setDownstreamCompilerPrelude(SlangPassThrough passThrough, const char* preludeText) = 0;

    /** DEPRECATED: Use getLanguagePrelude

    Get the 'prelude' for generated code for a 'downstream compiler'.
    @param passThrough The downstream compiler for generated code that will have the prelude applied
    to it.
    @param outPrelude  On exit holds a blob that holds the string of the prelude.
    */
    virtual SLANG_NO_THROW void SLANG_MCALL
    getDownstreamCompilerPrelude(SlangPassThrough passThrough, ISlangBlob** outPrelude) = 0;

    /** Get the build version 'tag' string. The string is the same as produced via `git describe
    --tags` for the project. If Slang is built separately from the automated build scripts the
    contents will by default be 'unknown'. Any string can be set by changing the contents of
    'slang-tag-version.h' file and recompiling the project.

    This method will return exactly the same result as the free function spGetBuildTagString.

    @return The build tag string
    */
    virtual SLANG_NO_THROW const char* SLANG_MCALL getBuildTagString() = 0;

    /* For a given source language set the default compiler.
    If a default cannot be chosen (for example the target cannot be achieved by the default),
    the default will not be used.

    @param sourceLanguage the source language
    @param defaultCompiler the default compiler for that language
    @return
    */
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL setDefaultDownstreamCompiler(
        SlangSourceLanguage sourceLanguage,
        SlangPassThrough defaultCompiler) = 0;

    /* For a source type get the default compiler

    @param sourceLanguage the source language
    @return The downstream compiler for that source language */
    virtual SlangPassThrough SLANG_MCALL
    getDefaultDownstreamCompiler(SlangSourceLanguage sourceLanguage) = 0;

    /* Set the 'prelude' placed before generated code for a specific language type.

    @param sourceLanguage The language the prelude should be inserted on.
    @param preludeText The text added pre-pended verbatim before the generated source

    Note! That for pass-through usage, prelude is not pre-pended, preludes are for code generation
    only.
    */
    virtual SLANG_NO_THROW void SLANG_MCALL
    setLanguagePrelude(SlangSourceLanguage sourceLanguage, const char* preludeText) = 0;

    /** Get the 'prelude' associated with a specific source language.
    @param sourceLanguage The language the prelude should be inserted on.
    @param outPrelude  On exit holds a blob that holds the string of the prelude.
    */
    virtual SLANG_NO_THROW void SLANG_MCALL
    getLanguagePrelude(SlangSourceLanguage sourceLanguage, ISlangBlob** outPrelude) = 0;

    /** Create a compile request.
     */
    [[deprecated]] virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    createCompileRequest(slang::ICompileRequest** outCompileRequest) = 0;

    /** Add new builtin declarations to be used in subsequent compiles.
     */
    virtual SLANG_NO_THROW void SLANG_MCALL
    addBuiltins(char const* sourcePath, char const* sourceString) = 0;

    /** Set the session shared library loader. If this changes the loader, it may cause shared
    libraries to be unloaded
    @param loader The loader to set. Setting nullptr sets the default loader.
    */
    virtual SLANG_NO_THROW void SLANG_MCALL
    setSharedLibraryLoader(ISlangSharedLibraryLoader* loader) = 0;

    /** Gets the currently set shared library loader
    @return Gets the currently set loader. If returns nullptr, it's the default loader
    */
    virtual SLANG_NO_THROW ISlangSharedLibraryLoader* SLANG_MCALL getSharedLibraryLoader() = 0;

    /** Returns SLANG_OK if the compilation target is supported for this session

    @param target The compilation target to test
    @return SLANG_OK if the target is available
    SLANG_E_NOT_IMPLEMENTED if not implemented in this build
    SLANG_E_NOT_FOUND if other resources (such as shared libraries) required to make target work
    could not be found SLANG_FAIL other kinds of failures */
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    checkCompileTargetSupport(SlangCompileTarget target) = 0;

    /** Returns SLANG_OK if the pass through support is supported for this session
    @param session Session
    @param target The compilation target to test
    @return SLANG_OK if the target is available
    SLANG_E_NOT_IMPLEMENTED if not implemented in this build
    SLANG_E_NOT_FOUND if other resources (such as shared libraries) required to make target work
    could not be found SLANG_FAIL other kinds of failures */
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    checkPassThroughSupport(SlangPassThrough passThrough) = 0;

    /** Compile from (embedded source) the core module on the session.
    Will return a failure if there is already a core module available
    NOTE! API is experimental and not ready for production code
    @param flags to control compilation
    */
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    compileCoreModule(CompileCoreModuleFlags flags) = 0;

    /** Load the core module. Currently loads modules from the file system.
    @param coreModule Start address of the serialized core module
    @param coreModuleSizeInBytes The size in bytes of the serialized core module

    NOTE! API is experimental and not ready for production code
    */
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    loadCoreModule(const void* coreModule, size_t coreModuleSizeInBytes) = 0;

    /** Save the core module to the file system
    @param archiveType The type of archive used to hold the core module
    @param outBlob The serialized blob containing the core module

    NOTE! API is experimental and not ready for production code  */
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    saveCoreModule(SlangArchiveType archiveType, ISlangBlob** outBlob) = 0;

    /** Look up the internal ID of a capability by its `name`.

    Capability IDs are *not* guaranteed to be stable across versions
    of the Slang library, so clients are expected to look up
    capabilities by name at runtime.
    */
    virtual SLANG_NO_THROW SlangCapabilityID SLANG_MCALL findCapability(char const* name) = 0;

    /** Set the downstream/pass through compiler to be used for a transition from the source type to
    the target type
    @param source The source 'code gen target'
    @param target The target 'code gen target'
    @param compiler The compiler/pass through to use for the transition from source to target
    */
    virtual SLANG_NO_THROW void SLANG_MCALL setDownstreamCompilerForTransition(
        SlangCompileTarget source,
        SlangCompileTarget target,
        SlangPassThrough compiler) = 0;

    /** Get the downstream/pass through compiler for a transition specified by source and target
    @param source The source 'code gen target'
    @param target The target 'code gen target'
    @return The compiler that is used for the transition. Returns SLANG_PASS_THROUGH_NONE it is not
    defined
    */
    virtual SLANG_NO_THROW SlangPassThrough SLANG_MCALL
    getDownstreamCompilerForTransition(SlangCompileTarget source, SlangCompileTarget target) = 0;

    /** Get the time in seconds spent in the slang and downstream compiler.
     */
    virtual SLANG_NO_THROW void SLANG_MCALL
    getCompilerElapsedTime(double* outTotalTime, double* outDownstreamTime) = 0;

    /** Specify a spirv.core.grammar.json file to load and use when
     * parsing and checking any SPIR-V code
     */
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL setSPIRVCoreGrammar(char const* jsonPath) = 0;

    /** Parse slangc command line options into a SessionDesc that can be used to create a session
     *   with all the compiler options specified in the command line.
     *   @param argc The number of command line arguments.
     *   @param argv An input array of command line arguments to parse.
     *   @param outSessionDesc A pointer to a SessionDesc struct to receive parsed session desc.
     *   @param outAuxAllocation Auxiliary memory allocated to hold data used in the session desc.
     */
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL parseCommandLineArguments(
        int argc,
        const char* const* argv,
        SessionDesc* outSessionDesc,
        ISlangUnknown** outAuxAllocation) = 0;

    /** Computes a digest that uniquely identifies the session description.
     */
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    getSessionDescDigest(SessionDesc* sessionDesc, ISlangBlob** outBlob) = 0;

    /** Compile from (embedded source) the builtin module on the session.
    Will return a failure if there is already a builtin module available.
    NOTE! API is experimental and not ready for production code.
    @param module The builtin module name.
    @param flags to control compilation
    */
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    compileBuiltinModule(BuiltinModuleName module, CompileCoreModuleFlags flags) = 0;

    /** Load a builtin module. Currently loads modules from the file system.
    @param module The builtin module name
    @param moduleData Start address of the serialized core module
    @param sizeInBytes The size in bytes of the serialized builtin module

    NOTE! API is experimental and not ready for production code
    */
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    loadBuiltinModule(BuiltinModuleName module, const void* moduleData, size_t sizeInBytes) = 0;

    /** Save the builtin module to the file system
    @param module The builtin module name
    @param archiveType The type of archive used to hold the builtin module
    @param outBlob The serialized blob containing the builtin module

    NOTE! API is experimental and not ready for production code  */
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL saveBuiltinModule(
        BuiltinModuleName module,
        SlangArchiveType archiveType,
        ISlangBlob** outBlob) = 0;
};

    #define SLANG_UUID_IGlobalSession IGlobalSession::getTypeGuid()

/** Description of a code generation target.
 */
struct TargetDesc
{
    /** The size of this structure, in bytes.
     */
    size_t structureSize = sizeof(TargetDesc);

    /** The target format to generate code for (e.g., SPIR-V, DXIL, etc.)
     */
    SlangCompileTarget format = SLANG_TARGET_UNKNOWN;

    /** The compilation profile supported by the target (e.g., "Shader Model 5.1")
     */
    SlangProfileID profile = SLANG_PROFILE_UNKNOWN;

    /** Flags for the code generation target. Currently unused. */
    SlangTargetFlags flags = kDefaultTargetFlags;

    /** Default mode to use for floating-point operations on the target.
     */
    SlangFloatingPointMode floatingPointMode = SLANG_FLOATING_POINT_MODE_DEFAULT;

    /** The line directive mode for output source code.
     */
    SlangLineDirectiveMode lineDirectiveMode = SLANG_LINE_DIRECTIVE_MODE_DEFAULT;

    /** Whether to force `scalar` layout for glsl shader storage buffers.
     */
    bool forceGLSLScalarBufferLayout = false;

    /** Pointer to an array of compiler option entries, whose size is compilerOptionEntryCount.
     */
    CompilerOptionEntry* compilerOptionEntries = nullptr;

    /** Number of additional compiler option entries.
     */
    uint32_t compilerOptionEntryCount = 0;
};

typedef uint32_t SessionFlags;
enum
{
    kSessionFlags_None = 0
};

struct PreprocessorMacroDesc
{
    const char* name;
    const char* value;
};

struct SessionDesc
{
    /** The size of this structure, in bytes.
     */
    size_t structureSize = sizeof(SessionDesc);

    /** Code generation targets to include in the session.
     */
    TargetDesc const* targets = nullptr;
    SlangInt targetCount = 0;

    /** Flags to configure the session.
     */
    SessionFlags flags = kSessionFlags_None;

    /** Default layout to assume for variables with matrix types.
     */
    SlangMatrixLayoutMode defaultMatrixLayoutMode = SLANG_MATRIX_LAYOUT_ROW_MAJOR;

    /** Paths to use when searching for `#include`d or `import`ed files.
     */
    char const* const* searchPaths = nullptr;
    SlangInt searchPathCount = 0;

    PreprocessorMacroDesc const* preprocessorMacros = nullptr;
    SlangInt preprocessorMacroCount = 0;

    ISlangFileSystem* fileSystem = nullptr;

    bool enableEffectAnnotations = false;
    bool allowGLSLSyntax = false;

    /** Pointer to an array of compiler option entries, whose size is compilerOptionEntryCount.
     */
    CompilerOptionEntry* compilerOptionEntries = nullptr;

    /** Number of additional compiler option entries.
     */
    uint32_t compilerOptionEntryCount = 0;

    /** Whether to skip SPIRV validation.
     */
    bool skipSPIRVValidation = false;
};

enum class ContainerType
{
    None,
    UnsizedArray,
    StructuredBuffer,
    ConstantBuffer,
    ParameterBlock
};

/** A session provides a scope for code that is loaded.

A session can be used to load modules of Slang source code,
and to request target-specific compiled binaries and layout
information.

In order to be able to load code, the session owns a set
of active "search paths" for resolving `#include` directives
and `import` declarations, as well as a set of global
preprocessor definitions that will be used for all code
that gets `import`ed in the session.

If multiple user shaders are loaded in the same session,
and import the same module (e.g., two source files do `import X`)
then there will only be one copy of `X` loaded within the session.

In order to be able to generate target code, the session
owns a list of available compilation targets, which specify
code generation options.

Code loaded and compiled within a session is owned by the session
and will remain resident in memory until the session is released.
Applications wishing to control the memory usage for compiled
and loaded code should use multiple sessions.
*/
struct ISession : public ISlangUnknown
{
    SLANG_COM_INTERFACE(0x67618701, 0xd116, 0x468f, {0xab, 0x3b, 0x47, 0x4b, 0xed, 0xce, 0xe, 0x3d})

    /** Get the global session thas was used to create this session.
     */
    virtual SLANG_NO_THROW IGlobalSession* SLANG_MCALL getGlobalSession() = 0;

    /** Load a module as it would be by code using `import`.
     */
    virtual SLANG_NO_THROW IModule* SLANG_MCALL
    loadModule(const char* moduleName, IBlob** outDiagnostics = nullptr) = 0;

    /** Load a module from Slang source code.
     */
    virtual SLANG_NO_THROW IModule* SLANG_MCALL loadModuleFromSource(
        const char* moduleName,
        const char* path,
        slang::IBlob* source,
        slang::IBlob** outDiagnostics = nullptr) = 0;

    /** Combine multiple component types to create a composite component type.

    The `componentTypes` array must contain `componentTypeCount` pointers
    to component types that were loaded or created using the same session.

    The shader parameters and specialization parameters of the composite will
    be the union of those in `componentTypes`. The relative order of child
    component types is significant, and will affect the order in which
    parameters are reflected and laid out.

    The entry-point functions of the composite will be the union of those in
    `componentTypes`, and will follow the ordering of `componentTypes`.

    The requirements of the composite component type will be a subset of
    those in `componentTypes`. If an entry in `componentTypes` has a requirement
    that can be satisfied by another entry, then the composition will
    satisfy the requirement and it will not appear as a requirement of
    the composite. If multiple entries in `componentTypes` have a requirement
    for the same type, then only the first such requirement will be retained
    on the composite. The relative ordering of requirements on the composite
    will otherwise match that of `componentTypes`.

    If any diagnostics are generated during creation of the composite, they
    will be written to `outDiagnostics`. If an error is encountered, the
    function will return null.

    It is an error to create a composite component type that recursively
    aggregates a single module more than once.
    */
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL createCompositeComponentType(
        IComponentType* const* componentTypes,
        SlangInt componentTypeCount,
        IComponentType** outCompositeComponentType,
        ISlangBlob** outDiagnostics = nullptr) = 0;

    /** Specialize a type based on type arguments.
     */
    virtual SLANG_NO_THROW TypeReflection* SLANG_MCALL specializeType(
        TypeReflection* type,
        SpecializationArg const* specializationArgs,
        SlangInt specializationArgCount,
        ISlangBlob** outDiagnostics = nullptr) = 0;


    /** Get the layout `type` on the chosen `target`.
     */
    virtual SLANG_NO_THROW TypeLayoutReflection* SLANG_MCALL getTypeLayout(
        TypeReflection* type,
        SlangInt targetIndex = 0,
        LayoutRules rules = LayoutRules::Default,
        ISlangBlob** outDiagnostics = nullptr) = 0;

    /** Get a container type from `elementType`. For example, given type `T`, returns
        a type that represents `StructuredBuffer<T>`.

        @param `elementType`: the element type to wrap around.
        @param `containerType`: the type of the container to wrap `elementType` in.
        @param `outDiagnostics`: a blob to receive diagnostic messages.
    */
    virtual SLANG_NO_THROW TypeReflection* SLANG_MCALL getContainerType(
        TypeReflection* elementType,
        ContainerType containerType,
        ISlangBlob** outDiagnostics = nullptr) = 0;

    /** Return a `TypeReflection` that represents the `__Dynamic` type.
        This type can be used as a specialization argument to indicate using
        dynamic dispatch.
    */
    virtual SLANG_NO_THROW TypeReflection* SLANG_MCALL getDynamicType() = 0;

    /** Get the mangled name for a type RTTI object.
     */
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    getTypeRTTIMangledName(TypeReflection* type, ISlangBlob** outNameBlob) = 0;

    /** Get the mangled name for a type witness.
     */
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL getTypeConformanceWitnessMangledName(
        TypeReflection* type,
        TypeReflection* interfaceType,
        ISlangBlob** outNameBlob) = 0;

    /** Get the sequential ID used to identify a type witness in a dynamic object.
     */
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL getTypeConformanceWitnessSequentialID(
        slang::TypeReflection* type,
        slang::TypeReflection* interfaceType,
        uint32_t* outId) = 0;

    /** Create a request to load/compile front-end code.
     */
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    createCompileRequest(SlangCompileRequest** outCompileRequest) = 0;


    /** Creates a `IComponentType` that represents a type's conformance to an interface.
        The retrieved `ITypeConformance` objects can be included in a composite `IComponentType`
        to explicitly specify which implementation types should be included in the final compiled
        code. For example, if an module defines `IMaterial` interface and `AMaterial`,
        `BMaterial`, `CMaterial` types that implements the interface, the user can exclude
        `CMaterial` implementation from the resulting shader code by explicitly adding
        `AMaterial:IMaterial` and `BMaterial:IMaterial` conformances to a composite
        `IComponentType` and get entry point code from it. The resulting code will not have
        anything related to `CMaterial` in the dynamic dispatch logic. If the user does not
        explicitly include any `TypeConformances` to an interface type, all implementations to
        that interface will be included by default. By linking a `ITypeConformance`, the user is
        also given the opportunity to specify the dispatch ID of the implementation type. If
        `conformanceIdOverride` is -1, there will be no override behavior and Slang will
        automatically assign IDs to implementation types. The automatically assigned IDs can be
        queried via `ISession::getTypeConformanceWitnessSequentialID`.

        Returns SLANG_OK if succeeds, or SLANG_FAIL if `type` does not conform to `interfaceType`.
    */
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL createTypeConformanceComponentType(
        slang::TypeReflection* type,
        slang::TypeReflection* interfaceType,
        ITypeConformance** outConformance,
        SlangInt conformanceIdOverride,
        ISlangBlob** outDiagnostics) = 0;

    /** Load a module from a Slang module blob.
     */
    virtual SLANG_NO_THROW IModule* SLANG_MCALL loadModuleFromIRBlob(
        const char* moduleName,
        const char* path,
        slang::IBlob* source,
        slang::IBlob** outDiagnostics = nullptr) = 0;

    virtual SLANG_NO_THROW SlangInt SLANG_MCALL getLoadedModuleCount() = 0;
    virtual SLANG_NO_THROW IModule* SLANG_MCALL getLoadedModule(SlangInt index) = 0;

    /** Checks if a precompiled binary module is up-to-date with the current compiler
     *   option settings and the source file contents.
     */
    virtual SLANG_NO_THROW bool SLANG_MCALL
    isBinaryModuleUpToDate(const char* modulePath, slang::IBlob* binaryModuleBlob) = 0;

    /** Load a module from a string.
     */
    virtual SLANG_NO_THROW IModule* SLANG_MCALL loadModuleFromSourceString(
        const char* moduleName,
        const char* path,
        const char* string,
        slang::IBlob** outDiagnostics = nullptr) = 0;
};

    #define SLANG_UUID_ISession ISession::getTypeGuid()

struct IMetadata : public ISlangCastable
{
    SLANG_COM_INTERFACE(0x8044a8a3, 0xddc0, 0x4b7f, {0xaf, 0x8e, 0x2, 0x6e, 0x90, 0x5d, 0x73, 0x32})

    /*
    Returns whether a resource parameter at the specified binding location is actually being used
    in the compiled shader.
    */
    virtual SlangResult isParameterLocationUsed(
        SlangParameterCategory category, // is this a `t` register? `s` register?
        SlangUInt spaceIndex,            // `space` for D3D12, `set` for Vulkan
        SlangUInt registerIndex,         // `register` for D3D12, `binding` for Vulkan
        bool& outUsed) = 0;
};
    #define SLANG_UUID_IMetadata IMetadata::getTypeGuid()

/** A component type is a unit of shader code layout, reflection, and linking.

A component type is a unit of shader code that can be included into
a linked and compiled shader program. Each component type may have:

* Zero or more uniform shader parameters, representing textures,
  buffers, etc. that the code in the component depends on.

* Zero or more *specialization* parameters, which are type or
  value parameters that can be used to synthesize specialized
  versions of the component type.

* Zero or more entry points, which are the individually invocable
  kernels that can have final code generated.

* Zero or more *requirements*, which are other component
  types on which the component type depends.

One example of a component type is a module of Slang code:

* The global-scope shader parameters declared in the module are
  the parameters when considered as a component type.

* Any global-scope generic or interface type parameters introduce
  specialization parameters for the module.

* A module does not by default include any entry points when
  considered as a component type (although the code of the
  module might *declare* some entry points).

* Any other modules that are `import`ed in the source code
  become requirements of the module, when considered as a
  component type.

An entry point is another example of a component type:

* The `uniform` parameters of the entry point function are
  its shader parameters when considered as a component type.

* Any generic or interface-type parameters of the entry point
  introduce specialization parameters.

* An entry point component type exposes a single entry point (itself).

* An entry point has one requirement for the module in which
  it was defined.

Component types can be manipulated in a few ways:

* Multiple component types can be combined into a composite, which
  combines all of their code, parameters, etc.

* A component type can be specialized, by "plugging in" types and
  values for its specialization parameters.

* A component type can be laid out for a particular target, giving
  offsets/bindings to the shader parameters it contains.

* Generated kernel code can be requested for entry points.

*/
struct IComponentType : public ISlangUnknown
{
    SLANG_COM_INTERFACE(0x5bc42be8, 0x5c50, 0x4929, {0x9e, 0x5e, 0xd1, 0x5e, 0x7c, 0x24, 0x1, 0x5f})

    /** Get the runtime session that this component type belongs to.
     */
    virtual SLANG_NO_THROW ISession* SLANG_MCALL getSession() = 0;

    /** Get the layout for this program for the chosen `targetIndex`.

    The resulting layout will establish offsets/bindings for all
    of the global and entry-point shader parameters in the
    component type.

    If this component type has specialization parameters (that is,
    it is not fully specialized), then the resulting layout may
    be incomplete, and plugging in arguments for generic specialization
    parameters may result in a component type that doesn't have
    a compatible layout. If the component type only uses
    interface-type specialization parameters, then the layout
    for a specialization should be compatible with an unspecialized
    layout (all parameters in the unspecialized layout will have
    the same offset/binding in the specialized layout).

    If this component type is combined into a composite, then
    the absolute offsets/bindings of parameters may not stay the same.
    If the shader parameters in a component type don't make
    use of explicit binding annotations (e.g., `register(...)`),
    then the *relative* offset of shader parameters will stay
    the same when it is used in a composition.
    */
    virtual SLANG_NO_THROW ProgramLayout* SLANG_MCALL
    getLayout(SlangInt targetIndex = 0, IBlob** outDiagnostics = nullptr) = 0;

    /** Get the number of (unspecialized) specialization parameters for the component type.
     */
    virtual SLANG_NO_THROW SlangInt SLANG_MCALL getSpecializationParamCount() = 0;

    /** Get the compiled code for the entry point at `entryPointIndex` for the chosen `targetIndex`

    Entry point code can only be computed for a component type that
    has no specialization parameters (it must be fully specialized)
    and that has no requirements (it must be fully linked).

    If code has not already been generated for the given entry point and target,
    then a compilation error may be detected, in which case `outDiagnostics`
    (if non-null) will be filled in with a blob of messages diagnosing the error.
    */
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL getEntryPointCode(
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        IBlob** outCode,
        IBlob** outDiagnostics = nullptr) = 0;

    /** Get the compilation result as a file system.

    Has the same requirements as getEntryPointCode.

    The result is not written to the actual OS file system, but is made available as an
    in memory representation.
    */
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL getResultAsFileSystem(
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        ISlangMutableFileSystem** outFileSystem) = 0;

    /** Compute a hash for the entry point at `entryPointIndex` for the chosen `targetIndex`.

    This computes a hash based on all the dependencies for this component type as well as the
    target settings affecting the compiler backend. The computed hash is used as a key for caching
    the output of the compiler backend to implement shader caching.
    */
    virtual SLANG_NO_THROW void SLANG_MCALL
    getEntryPointHash(SlangInt entryPointIndex, SlangInt targetIndex, IBlob** outHash) = 0;

    /** Specialize the component by binding its specialization parameters to concrete arguments.

    The `specializationArgs` array must have `specializationArgCount` entries, and
    this must match the number of specialization parameters on this component type.

    If any diagnostics (error or warnings) are produced, they will be written to `outDiagnostics`.
    */
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL specialize(
        SpecializationArg const* specializationArgs,
        SlangInt specializationArgCount,
        IComponentType** outSpecializedComponentType,
        ISlangBlob** outDiagnostics = nullptr) = 0;

    /** Link this component type against all of its unsatisfied dependencies.

    A component type may have unsatisfied dependencies. For example, a module
    depends on any other modules it `import`s, and an entry point depends
    on the module that defined it.

    A user can manually satisfy dependencies by creating a composite
    component type, and when doing so they retain full control over
    the relative ordering of shader parameters in the resulting layout.

    It is an error to try to generate/access compiled kernel code for
    a component type with unresolved dependencies, so if dependencies
    remain after whatever manual composition steps an application
    cares to perform, the `link()` function can be used to automatically
    compose in any remaining dependencies. The order of parameters
    (and hence the global layout) that results will be deterministic,
    but is not currently documented.
    */
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    link(IComponentType** outLinkedComponentType, ISlangBlob** outDiagnostics = nullptr) = 0;

    /** Get entry point 'callable' functions accessible through the ISlangSharedLibrary interface.

    The functions remain in scope as long as the ISlangSharedLibrary interface is in scope.

    NOTE! Requires a compilation target of SLANG_HOST_CALLABLE.

    @param entryPointIndex  The index of the entry point to get code for.
    @param targetIndex      The index of the target to get code for (default: zero).
    @param outSharedLibrary A pointer to a ISharedLibrary interface which functions can be queried
    on.
    @returns                A `SlangResult` to indicate success or failure.
    */
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL getEntryPointHostCallable(
        int entryPointIndex,
        int targetIndex,
        ISlangSharedLibrary** outSharedLibrary,
        slang::IBlob** outDiagnostics = 0) = 0;

    /** Get a new ComponentType object that represents a renamed entry point.

    The current object must be a single EntryPoint, or a CompositeComponentType or
    SpecializedComponentType that contains one EntryPoint component.
    */
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    renameEntryPoint(const char* newName, IComponentType** outEntryPoint) = 0;

    /** Link and specify additional compiler options when generating code
     *   from the linked program.
     */
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL linkWithOptions(
        IComponentType** outLinkedComponentType,
        uint32_t compilerOptionEntryCount,
        CompilerOptionEntry* compilerOptionEntries,
        ISlangBlob** outDiagnostics = nullptr) = 0;

    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    getTargetCode(SlangInt targetIndex, IBlob** outCode, IBlob** outDiagnostics = nullptr) = 0;

    virtual SLANG_NO_THROW SlangResult SLANG_MCALL getTargetMetadata(
        SlangInt targetIndex,
        IMetadata** outMetadata,
        IBlob** outDiagnostics = nullptr) = 0;

    virtual SLANG_NO_THROW SlangResult SLANG_MCALL getEntryPointMetadata(
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        IMetadata** outMetadata,
        IBlob** outDiagnostics = nullptr) = 0;
};
    #define SLANG_UUID_IComponentType IComponentType::getTypeGuid()

struct IEntryPoint : public IComponentType
{
    SLANG_COM_INTERFACE(0x8f241361, 0xf5bd, 0x4ca0, {0xa3, 0xac, 0x2, 0xf7, 0xfa, 0x24, 0x2, 0xb8})

    virtual SLANG_NO_THROW FunctionReflection* SLANG_MCALL getFunctionReflection() = 0;
};

    #define SLANG_UUID_IEntryPoint IEntryPoint::getTypeGuid()

struct ITypeConformance : public IComponentType
{
    SLANG_COM_INTERFACE(0x73eb3147, 0xe544, 0x41b5, {0xb8, 0xf0, 0xa2, 0x44, 0xdf, 0x21, 0x94, 0xb})
};
    #define SLANG_UUID_ITypeConformance ITypeConformance::getTypeGuid()

/** A module is the granularity of shader code compilation and loading.

In most cases a module corresponds to a single compile "translation unit."
This will often be a single `.slang` or `.hlsl` file and everything it
`#include`s.

Notably, a module `M` does *not* include the things it `import`s, as these
as distinct modules that `M` depends on. There is a directed graph of
module dependencies, and all modules in the graph must belong to the
same session (`ISession`).

A module establishes a namespace for looking up types, functions, etc.
*/
struct IModule : public IComponentType
{
    SLANG_COM_INTERFACE(0xc720e64, 0x8722, 0x4d31, {0x89, 0x90, 0x63, 0x8a, 0x98, 0xb1, 0xc2, 0x79})

    /// Find and an entry point by name.
    /// Note that this does not work in case the function is not explicitly designated as an entry
    /// point, e.g. using a `[shader("...")]` attribute. In such cases, consider using
    /// `IModule::findAndCheckEntryPoint` instead.
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    findEntryPointByName(char const* name, IEntryPoint** outEntryPoint) = 0;

    /// Get number of entry points defined in the module. An entry point defined in a module
    /// is by default not included in the linkage, so calls to `IComponentType::getEntryPointCount`
    /// on an `IModule` instance will always return 0. However `IModule::getDefinedEntryPointCount`
    /// will return the number of defined entry points.
    virtual SLANG_NO_THROW SlangInt32 SLANG_MCALL getDefinedEntryPointCount() = 0;
    /// Get the name of an entry point defined in the module.
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    getDefinedEntryPoint(SlangInt32 index, IEntryPoint** outEntryPoint) = 0;

    /// Get a serialized representation of the checked module.
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL serialize(ISlangBlob** outSerializedBlob) = 0;

    /// Write the serialized representation of this module to a file.
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL writeToFile(char const* fileName) = 0;

    /// Get the name of the module.
    virtual SLANG_NO_THROW const char* SLANG_MCALL getName() = 0;

    /// Get the path of the module.
    virtual SLANG_NO_THROW const char* SLANG_MCALL getFilePath() = 0;

    /// Get the unique identity of the module.
    virtual SLANG_NO_THROW const char* SLANG_MCALL getUniqueIdentity() = 0;

    /// Find and validate an entry point by name, even if the function is
    /// not marked with the `[shader("...")]` attribute.
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL findAndCheckEntryPoint(
        char const* name,
        SlangStage stage,
        IEntryPoint** outEntryPoint,
        ISlangBlob** outDiagnostics) = 0;

    /// Get the number of dependency files that this module depends on.
    /// This includes both the explicit source files, as well as any
    /// additional files that were transitively referenced (e.g., via
    /// a `#include` directive).
    virtual SLANG_NO_THROW SlangInt32 SLANG_MCALL getDependencyFileCount() = 0;

    /// Get the path to a file this module depends on.
    virtual SLANG_NO_THROW char const* SLANG_MCALL getDependencyFilePath(SlangInt32 index) = 0;

    virtual SLANG_NO_THROW DeclReflection* SLANG_MCALL getModuleReflection() = 0;

    /** Disassemble a module.
     */
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    disassemble(slang::IBlob** outDisassembledBlob) = 0;
};

    #define SLANG_UUID_IModule IModule::getTypeGuid()

/* Experimental interface for doing target precompilation of slang modules */
struct IModulePrecompileService_Experimental : public ISlangUnknown
{
    // uuidgen output:     8e12e8e3 -  5fcd -  433e -    afcb -      13a088bc5ee5
    SLANG_COM_INTERFACE(
        0x8e12e8e3,
        0x5fcd,
        0x433e,
        {0xaf, 0xcb, 0x13, 0xa0, 0x88, 0xbc, 0x5e, 0xe5})

    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    precompileForTarget(SlangCompileTarget target, ISlangBlob** outDiagnostics) = 0;

    virtual SLANG_NO_THROW SlangResult SLANG_MCALL getPrecompiledTargetCode(
        SlangCompileTarget target,
        IBlob** outCode,
        IBlob** outDiagnostics = nullptr) = 0;

    virtual SLANG_NO_THROW SlangInt SLANG_MCALL getModuleDependencyCount() = 0;

    virtual SLANG_NO_THROW SlangResult SLANG_MCALL getModuleDependency(
        SlangInt dependencyIndex,
        IModule** outModule,
        IBlob** outDiagnostics = nullptr) = 0;
};

    #define SLANG_UUID_IModulePrecompileService_Experimental \
        IModulePrecompileService_Experimental::getTypeGuid()

/** Argument used for specialization to types/values.
 */
struct SpecializationArg
{
    enum class Kind : int32_t
    {
        Unknown, /**< An invalid specialization argument. */
        Type,    /**< Specialize to a type. */
    };

    /** The kind of specialization argument. */
    Kind kind;
    union
    {
        /** A type specialization argument, used for `Kind::Type`. */
        TypeReflection* type;
    };

    static SpecializationArg fromType(TypeReflection* inType)
    {
        SpecializationArg rs;
        rs.kind = Kind::Type;
        rs.type = inType;
        return rs;
    }
};
} // namespace slang

    // Passed into functions to create globalSession to identify the API version client code is
    // using.
    #define SLANG_API_VERSION 0

enum SlangLanguageVersion
{
    SLANG_LANGUAGE_VERSION_2025 = 2025
};


/* Description of a Slang global session.
 */
struct SlangGlobalSessionDesc
{
    /// Size of this struct.
    uint32_t structureSize = sizeof(SlangGlobalSessionDesc);

    /// Slang API version.
    uint32_t apiVersion = SLANG_API_VERSION;

    /// Slang language version.
    uint32_t languageVersion = SLANG_LANGUAGE_VERSION_2025;

    /// Whether to enable GLSL support.
    bool enableGLSL = false;

    /// Reserved for future use.
    uint32_t reserved[16] = {};
};

/* Create a global session, with the built-in core module.

@param apiVersion Pass in SLANG_API_VERSION
@param outGlobalSession (out)The created global session.
*/
SLANG_EXTERN_C SLANG_API SlangResult
slang_createGlobalSession(SlangInt apiVersion, slang::IGlobalSession** outGlobalSession);


/* Create a global session, with the built-in core module.

@param desc Description of the global session.
@param outGlobalSession (out)The created global session.
*/
SLANG_EXTERN_C SLANG_API SlangResult slang_createGlobalSession2(
    const SlangGlobalSessionDesc* desc,
    slang::IGlobalSession** outGlobalSession);

/* Create a global session, but do not set up the core module. The core module can
then be loaded via loadCoreModule or compileCoreModule

@param apiVersion Pass in SLANG_API_VERSION
@param outGlobalSession (out)The created global session that doesn't have a core module setup.

NOTE! API is experimental and not ready for production code
*/
SLANG_EXTERN_C SLANG_API SlangResult slang_createGlobalSessionWithoutCoreModule(
    SlangInt apiVersion,
    slang::IGlobalSession** outGlobalSession);

/* Returns a blob that contains the serialized core module.
Returns nullptr if there isn't an embedded core module.

NOTE! API is experimental and not ready for production code
*/
SLANG_API ISlangBlob* slang_getEmbeddedCoreModule();


/* Cleanup all global allocations used by Slang, to prevent memory leak detectors from
 reporting them as leaks. This function should only be called after all Slang objects
 have been released. No other Slang functions such as `createGlobalSession`
 should be called after this function.
 */
SLANG_EXTERN_C SLANG_API void slang_shutdown();

/* Return the last signaled internal error message.
 */
SLANG_EXTERN_C SLANG_API const char* slang_getLastInternalErrorMessage();

// Slang VM
namespace slang
{

enum class OperandDataType
{
    General = 0, // General data type, can be any type.
    Int32 = 1,   // 32-bit integer.
    Int64 = 2,   // 64-bit integer.
    Float32 = 3, // 32-bit floating-point number.
    Float64 = 4, // 64-bit floating-point number.
    String = 5,  // String data type, represented as a pointer to a null-terminated string.
};

struct VMExecOperand
{
    uint8_t** section; // Pointer to the section start pointer.
    #if SLANG_PTR_IS_32
    uint32_t padding;
    #endif
    uint32_t type : 8; // type of the operand data.
    uint32_t size : 24;
    uint32_t offset;
    void* getPtr() const { return *section + offset; }
    OperandDataType getType() const { return (OperandDataType)type; }
};

struct VMExecInstHeader;
class IByteCodeRunner;

typedef void (*VMExtFunction)(IByteCodeRunner* context, VMExecInstHeader* inst, void* userData);
typedef void (*VMPrintFunc)(const char* message, void* userData);

struct VMExecInstHeader
{
    VMExtFunction functionPtr; // Pointer to the function that executes this instruction.
    #if SLANG_PTR_IS_32
    uint32_t padding;
    #endif
    uint32_t opcodeExtension;
    uint32_t operandCount;
    VMExecInstHeader* getNextInst()
    {
        return (VMExecInstHeader*)((VMExecOperand*)(this + 1) + operandCount);
    }
    VMExecOperand& getOperand(SlangInt index) const
    {
        return *((VMExecOperand*)(this + 1) + index);
    }
};

struct ByteCodeFuncInfo
{
    uint32_t parameterCount;
    uint32_t returnValueSize;
};

struct ByteCodeRunnerDesc
{
    /** The size of this structure, in bytes.
     */
    size_t structSize = sizeof(ByteCodeRunnerDesc);
};

/// Represents a byte code runner that can execute Slang byte code.
class IByteCodeRunner : public ISlangUnknown
{
public:
    // {AFDAB195-361F-42CB-9513-9006261DD8CD}
    SLANG_COM_INTERFACE(0xafdab195, 0x361f, 0x42cb, {0x95, 0x13, 0x90, 0x6, 0x26, 0x1d, 0xd8, 0xcd})

    /// Load a byte code module into the execution context.
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL loadModule(IBlob* moduleBlob) = 0;

    /// Select a function for execution.
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    selectFunctionByIndex(uint32_t functionIndex) = 0;

    virtual SLANG_NO_THROW int SLANG_MCALL findFunctionByName(const char* name) = 0;

    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    getFunctionInfo(uint32_t index, ByteCodeFuncInfo* outInfo) = 0;

    /// Obtain the current working set memory for the selected function.
    virtual SLANG_NO_THROW void* SLANG_MCALL getCurrentWorkingSet() = 0;

    /// Execute the selected function.
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    execute(void* argumentData, size_t argumentSize) = 0;

    /// Query the error string.
    virtual SLANG_NO_THROW void SLANG_MCALL getErrorString(IBlob** outBlob) = 0;

    /// Retrieve the return value of the last executed function.
    virtual SLANG_NO_THROW void* SLANG_MCALL getReturnValue(size_t* outValueSize) = 0;

    /// Set the user data for the external instruction handler.
    virtual SLANG_NO_THROW void SLANG_MCALL setExtInstHandlerUserData(void* userData) = 0;

    /// Register an external function that can be called from the byte code.
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    registerExtCall(const char* name, VMExtFunction functionPtr) = 0;

    /// Set a callback function to print messages from the byte code runner.
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    setPrintCallback(VMPrintFunc callback, void* userData) = 0;
};

} // namespace slang

/// Create a byte code runner that can execute Slang byte code.
SLANG_EXTERN_C SLANG_API SlangResult slang_createByteCodeRunner(
    const slang::ByteCodeRunnerDesc* desc,
    slang::IByteCodeRunner** outByteCodeRunner);

/// Disassemble a Slang byte code blob into human-readable text.
SLANG_EXTERN_C SLANG_API SlangResult
slang_disassembleByteCode(slang::IBlob* moduleBlob, slang::IBlob** outDisassemblyBlob);

namespace slang
{
inline SlangResult createGlobalSession(slang::IGlobalSession** outGlobalSession)
{
    SlangGlobalSessionDesc defaultDesc = {};
    return slang_createGlobalSession2(&defaultDesc, outGlobalSession);
}
inline SlangResult createGlobalSession(
    const SlangGlobalSessionDesc* desc,
    slang::IGlobalSession** outGlobalSession)
{
    return slang_createGlobalSession2(desc, outGlobalSession);
}
inline void shutdown()
{
    slang_shutdown();
}
inline const char* getLastInternalErrorMessage()
{
    return slang_getLastInternalErrorMessage();
}
} // namespace slang

#endif // C++ helpers

#define SLANG_ERROR_INSUFFICIENT_BUFFER SLANG_E_BUFFER_TOO_SMALL
#define SLANG_ERROR_INVALID_PARAMETER SLANG_E_INVALID_ARG

#endif
