#include "clang/Basic/Stack.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Basic/Version.h"
#include "clang/CodeGen/CodeGenAction.h"
#include "clang/CodeGen/ObjectFilePCHContainerOperations.h"
#include "clang/Config/config.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Options.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/TextDiagnosticBuffer.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Frontend/Utils.h"
#include "clang/FrontendTool/Utils.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/LinkAllPasses.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Support/BuryPointer.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

// Jit
#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/ExecutionEngine/JITLink/JITLinkMemoryManager.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IRReader/IRReader.h"

// Slang

#include "slang-com-helper.h"
#include "slang-com-ptr.h"
#include "slang.h"

#include <compiler-core/slang-artifact-associated-impl.h>
#include <compiler-core/slang-artifact-desc-util.h>
#include <compiler-core/slang-downstream-compiler.h>
#include <compiler-core/slang-slice-allocator.h>
#include <core/slang-com-object.h>
#include <core/slang-hash.h>
#include <core/slang-list.h>
#include <core/slang-shared-library.h>
#include <core/slang-string-util.h>
#include <core/slang-string.h>
#include <stdio.h>

// We want to make math functions available to the JIT
#if SLANG_GCC_FAMILY && __GNUC__ < 6
#include <cmath>
#define SLANG_LLVM_STD std::
#else
#include <math.h>
#define SLANG_LLVM_STD
#endif

#if SLANG_OSX
// For memset_pattern functions
// https://www.unix.com/man-page/osx/3/memset_pattern16/
#include <string.h>
#endif

#if SLANG_WINDOWS_FAMILY

/*
It's not clear if this function is needed for ARM WIN targets, but we'll assume it does for now.

https://learn.microsoft.com/en-us/windows/win32/devnotes/-win32-chkstk
https://www.betaarchive.com/wiki/index.php/Microsoft_KB_Archive/100775
https://codywu2010.wordpress.com/2010/10/04/__chkstk-and-stack-overflow/
*/

#if SLANG_PROCESSOR_X86
extern "C" void /* __declspec(naked)*/ __cdecl _chkstk();
#else
extern "C" void /* __declspec(naked)*/ __cdecl __chkstk();
#endif
#endif

// Predeclare. We'll use this symbol to lookup timestamp, if we don't have a hash.
extern "C" SLANG_DLL_EXPORT SlangResult
createLLVMDownstreamCompiler_V4(const SlangUUID& intfGuid, Slang::IDownstreamCompiler** out);

namespace slang_llvm
{

using namespace clang;

using namespace llvm::opt;
using namespace llvm;
using namespace llvm::orc;

using namespace Slang;

class LLVMDownstreamCompiler : public ComBaseObject, public IDownstreamCompiler
{
public:
    typedef ComBaseObject Super;

    // IUnknown
    SLANG_COM_BASE_IUNKNOWN_ALL

    // ICastable
    virtual SLANG_NO_THROW void* SLANG_MCALL castAs(const Guid& guid) SLANG_OVERRIDE;

    // IDownstreamCompiler
    virtual SLANG_NO_THROW const Desc& SLANG_MCALL getDesc() SLANG_OVERRIDE { return m_desc; }
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    compile(const CompileOptions& options, IArtifact** outArtifact) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW bool SLANG_MCALL
    canConvert(const ArtifactDesc& from, const ArtifactDesc& to) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    convert(IArtifact* from, const ArtifactDesc& to, IArtifact** outArtifact) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW bool SLANG_MCALL isFileBased() SLANG_OVERRIDE { return false; }
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL getVersionString(slang::IBlob** outVersionString)
        SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    validate(const uint32_t* contents, int contentsSize) SLANG_OVERRIDE
    {
        SLANG_UNUSED(contents);
        SLANG_UNUSED(contentsSize);
        return SLANG_FAIL;
    }
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    disassemble(const uint32_t* contents, int contentsSize) SLANG_OVERRIDE
    {
        SLANG_UNUSED(contents);
        SLANG_UNUSED(contentsSize);
        return SLANG_FAIL;
    }
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL disassembleWithResult(
        const uint32_t* contents,
        int contentsSize,
        String& outString) SLANG_OVERRIDE
    {
        SLANG_UNUSED(contents);
        SLANG_UNUSED(contentsSize);
        SLANG_UNUSED(outString);
        return SLANG_FAIL;
    }

    LLVMDownstreamCompiler()
        : m_desc(
              SLANG_PASS_THROUGH_LLVM,
              SemanticVersion(LLVM_VERSION_MAJOR, LLVM_VERSION_MINOR, LLVM_VERSION_PATCH))
    {
    }

    void* getInterface(const Guid& guid);
    void* getObject(const Guid& guid);

    Desc m_desc;
};


/* !!!!!!!!!!!!!!!!!!!!! LLVMJITSharedLibrary !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

/* This implementation uses atomic ref counting to ensure the shared libraries lifetime can outlive
the LLVMDownstreamCompileResult and the compilation that created it */
class LLVMJITSharedLibrary : public ISlangSharedLibrary, public ComBaseObject
{
public:
    // ISlangUnknown
    SLANG_COM_BASE_IUNKNOWN_ALL

    /// ICastable
    virtual SLANG_NO_THROW void* SLANG_MCALL castAs(const Guid& guid) SLANG_OVERRIDE;

    // ISlangSharedLibrary impl
    virtual SLANG_NO_THROW void* SLANG_MCALL findSymbolAddressByName(char const* name)
        SLANG_OVERRIDE;

    LLVMJITSharedLibrary(std::unique_ptr<llvm::orc::LLJIT> jit)
        : m_jit(std::move(jit))
    {
    }

protected:
    ISlangUnknown* getInterface(const SlangUUID& uuid);
    void* getObject(const SlangUUID& uuid);

    std::unique_ptr<llvm::orc::LLJIT> m_jit;
};

ISlangUnknown* LLVMJITSharedLibrary::getInterface(const SlangUUID& guid)
{
    if (guid == ISlangUnknown::getTypeGuid() || guid == ISlangCastable::getTypeGuid() ||
        guid == ISlangSharedLibrary::getTypeGuid())
    {
        return static_cast<ISlangSharedLibrary*>(this);
    }
    return nullptr;
}

void* LLVMJITSharedLibrary::getObject(const SlangUUID& uuid)
{
    SLANG_UNUSED(uuid);
    return nullptr;
}

void* LLVMJITSharedLibrary::castAs(const Guid& guid)
{
    if (auto ptr = getInterface(guid))
    {
        return ptr;
    }
    return getObject(guid);
}

void* LLVMJITSharedLibrary::findSymbolAddressByName(char const* name)
{
    auto fnExpected = m_jit->lookup(name);
    if (fnExpected)
    {
        auto fn = std::move(*fnExpected);
        return (void*)fn.getAddress();
    }
    return nullptr;
}


static void _ensureSufficientStack() {}

static void _llvmErrorHandler(void* userData, const std::string& message, bool genCrashDiag)
{
    // DiagnosticsEngine& diags = *static_cast<DiagnosticsEngine*>(userData);
    // diags.Report(diag::err_fe_error_backend) << message;

    printf("Clang/LLVM fatal error: %s\n", message.c_str());

    // Run the interrupt handlers to make sure any special cleanups get done, in
    // particular that we remove files registered with RemoveFileOnSignal.
    llvm::sys::RunInterruptHandlers();

    // We cannot recover from llvm errors.  (!)
    //
    // Returning nothing, will still cause LLVM to exit the process.
}

static Slang::ArtifactDiagnostic::Severity _getSeverity(DiagnosticsEngine::Level level)
{
    typedef ArtifactDiagnostic::Severity Severity;
    typedef DiagnosticsEngine::Level Level;
    switch (level)
    {
    default:
    case Level::Ignored:
    case Level::Note:
    case Level::Remark:
        {
            return Severity::Info;
        }
    case Level::Warning:
        {
            return Severity::Warning;
        }
    case Level::Error:
    case Level::Fatal:
        {
            return Severity::Error;
        }
    }
}

class BufferedDiagnosticConsumer : public clang::DiagnosticConsumer
{
public:
    BufferedDiagnosticConsumer(IArtifactDiagnostics* diagnostics)
        : m_diagnostics(diagnostics)
    {
    }

    void HandleDiagnostic(DiagnosticsEngine::Level level, const Diagnostic& info) override
    {
        SmallString<100> text;
        info.FormatDiagnostic(text);

        ArtifactDiagnostic diagnostic;
        diagnostic.severity = _getSeverity(level);
        diagnostic.stage = ArtifactDiagnostic::Stage::Compile;
        diagnostic.text = TerminatedCharSlice(text.c_str(), Count(text.size()));

        auto location = info.getLocation();

        // Work out what the location is
        auto& sourceManager = info.getSourceManager();

        // Gets the file/line number
        const bool useLineDirectives = true;
        const PresumedLoc presumedLoc = sourceManager.getPresumedLoc(location, useLineDirectives);

        diagnostic.location.line = presumedLoc.getLine();
        diagnostic.filePath = TerminatedCharSlice(presumedLoc.getFilename());

        m_diagnostics->add(diagnostic);
    }

    bool hasError() const
    {
        return m_diagnostics->getCountAtLeastSeverity(ArtifactDiagnostic::Severity::Error) > 0;
    }

    ComPtr<IArtifactDiagnostics> m_diagnostics;
};

/*
 * A question is how to make the prototypes available for these functions. They would need to be
 * defined before the the prelude - or potentially in the prelude.
 *
 * I could just define the prototypes in the prelude, and only impl, if needed. Here though I
 * require that all the functions implemented here, use C style names (ie unmanagled) to simplify
 * lookup.
 */

struct NameAndFunc
{
    typedef void (*Func)();

    const char* name;
    Func func;
};

#define SLANG_LLVM_EXPAND(x) x

#define SLANG_LLVM_FUNC(name, cppName, retType, paramTypes) \
    NameAndFunc{                                            \
        #name,                                              \
        (NameAndFunc::Func) static_cast<retType(*) paramTypes>(&SLANG_LLVM_EXPAND(cppName))},

// Implementations of maths functions available to JIT
static float F32_frexp(float x, int* e)
{
    float m = ::frexpf(x, e);
    return m;
}

static double F64_frexp(double x, int* e)
{
    double m = ::frexp(x, e);
    return m;
}

static void assertFailed(const char* msg)
{
    printf("Assert failed: %s\n", msg);
    SLANG_BREAKPOINT(0);
}

#if SLANG_OSX

namespace OSXSpecific
{

static void bzero(void* dst, size_t size)
{
    ::memset(dst, 0, size);
}

} // namespace OSXSpecific
#endif

#if SLANG_VC && SLANG_PTR_IS_32

namespace WinSpecific
{

// NOTE! These are functions used in 32 bit windows to enable 64 bit maths. This set is probably
// *not* complete. Check:

// https://source.winehq.org/source/dlls/ntdll/large_int.c

static int64_t __stdcall _alldiv(int64_t a, int64_t b)
{
    return a / b;
}

static int64_t __stdcall _allrem(int64_t a, int64_t b)
{
    return a % b;
}

static uint64_t __stdcall _aullrem(uint64_t a, uint64_t b)
{
    return a % b;
}

static uint64_t __stdcall _aulldiv(uint64_t a, uint64_t b)
{
    return a / b;
}

} // namespace WinSpecific

#endif


// These are only the functions that cannot be implemented with 'reasonable performance' in the
// prelude. It is assumed that calling from JIT to C function whilst not super expensive, is an
// issue.

// name, cppName, retType, paramTypes
// clang-format off
#define SLANG_LLVM_FUNCS(x) \
    x(F64_ceil, ceil, double, (double)) \
    x(F64_floor, floor, double, (double)) \
    x(F64_round, round, double, (double)) \
    x(F64_abs, fabs, double, (double)) \
    x(F64_sin, sin, double, (double)) \
    x(F64_cos, cos, double, (double)) \
    x(F64_tan, tan, double, (double)) \
    x(F64_asin, asin, double, (double)) \
    x(F64_acos, acos, double, (double)) \
    x(F64_atan, atan, double, (double)) \
    x(F64_sinh, sinh, double, (double)) \
    x(F64_cosh, cosh, double, (double)) \
    x(F64_tanh, tanh, double, (double)) \
    x(F64_log2, log2, double, (double)) \
    x(F64_log, log, double, (double)) \
    x(F64_log10, log10, double, (double)) \
    x(F64_exp2, exp2, double, (double)) \
    x(F64_exp, exp, double, (double)) \
    x(F64_fabs, fabs, double, (double)) \
    x(F64_trunc, trunc, double, (double)) \
    x(F64_sqrt, sqrt, double, (double)) \
    \
    x(F64_isnan, SLANG_LLVM_STD isnan, bool, (double)) \
    x(F64_isfinite, SLANG_LLVM_STD isfinite, bool, (double)) \
    x(F64_isinf, SLANG_LLVM_STD isinf, bool, (double)) \
    \
    x(F64_atan2, atan2, double, (double, double)) \
    \
    x(F64_frexp, F64_frexp, double, (double, int*)) \
    x(F64_pow, pow, double, (double, double)) \
    \
    x(F64_modf, modf, double, (double, double*)) \
    x(F64_fmod, fmod, double, (double, double)) \
    x(F64_remainder, remainder, double, (double, double)) \
    \
    x(F32_ceil, ceilf, float, (float)) \
    x(F32_floor, floorf, float, (float)) \
    x(F32_round, roundf, float, (float)) \
    x(F32_abs, fabsf, float, (float)) \
    x(F32_sin, sinf, float, (float)) \
    x(F32_cos, cosf, float, (float)) \
    x(F32_tan, tanf, float, (float)) \
    x(F32_asin, asinf, float, (float)) \
    x(F32_acos, acosf, float, (float)) \
    x(F32_atan, atanf, float, (float)) \
    x(F32_sinh, sinhf, float, (float)) \
    x(F32_cosh, coshf, float, (float)) \
    x(F32_tanh, tanhf, float, (float)) \
    x(F32_log2, log2f, float, (float)) \
    x(F32_log, logf, float, (float)) \
    x(F32_log10, log10f, float, (float)) \
    x(F32_exp2, exp2f, float, (float)) \
    x(F32_exp, expf, float, (float)) \
    x(F32_fabs, fabsf, float, (float)) \
    x(F32_trunc, truncf, float, (float)) \
    x(F32_sqrt, sqrtf, float, (float)) \
    \
    x(F32_isnan, SLANG_LLVM_STD isnan, bool, (float)) \
    x(F32_isfinite, SLANG_LLVM_STD isfinite, bool, (float)) \
    x(F32_isinf, SLANG_LLVM_STD isinf, bool, (float)) \
    \
    x(F32_atan2, atan2f, float, (float, float)) \
    \
    x(F32_frexp, F32_frexp, float, (float, int*)) \
    x(F32_pow, powf, float, (float, float)) \
    \
    x(F32_modf, modff, float, (float, float*)) \
    x(F32_fmod, fmodf, float, (float, float)) \
    x(F32_remainder, remainderf, float, (float, float)) \
    \
    x(assertFailed, assertFailed, void, (const char*)) \
    \
    x(memcpy, memcpy, void*, (void*, const void*, size_t)) \
    x(memmove, memmove, void*, (void*, const void*, size_t)) \
    x(memcmp, memcmp, int, (const void*, const void*, size_t)) \
    x(memset, memset, void*, (void*, int, size_t)) 

#if SLANG_OSX
#   define SLANG_PLATFORM_FUNCS(x) \
    x(memset_pattern4, memset_pattern4, void, (void*, const void*, size_t)) \
    x(memset_pattern8, memset_pattern8, void, (void*, const void*, size_t)) \
    x(memset_pattern16, memset_pattern16, void, (void*, const void*, size_t)) \
    \
    x(__bzero, OSXSpecific::bzero, void, (void*, size_t))
#endif
// clang-format on

#if SLANG_WINDOWS_FAMILY
#if SLANG_PROCESSOR_X86
#define SLANG_PLATFORM_FUNCS(x) x(_chkstk, _chkstk, void, ())
#else
#define SLANG_PLATFORM_FUNCS(x) x(__chkstk, __chkstk, void, ())
#endif
#endif

#ifndef SLANG_PLATFORM_FUNCS
#define SLANG_PLATFORM_FUNCS(x)
#endif

static int _getOptimizationLevel(DownstreamCompileOptions::OptimizationLevel level)
{
    typedef DownstreamCompileOptions::OptimizationLevel OptimizationLevel;
    switch (level)
    {
    case OptimizationLevel::None:
        return 0;
    default:
    case OptimizationLevel::Default:
        return 1;
    case OptimizationLevel::High:
        return 2;
    case OptimizationLevel::Maximal:
        return 3;
    }
}

static SlangResult _initLLVM()
{
    // Initialize targets first, so that --version shows registered targets.
#if 0
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmPrinters();
    llvm::InitializeAllAsmParsers();
#else
    // Just initialize items needed for this target.

    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();

    llvm::InitializeNativeTargetDisassembler();
#endif

    // Set an error handler, so that any LLVM backend diagnostics go through our
    // error handler.
    // llvm::install_fatal_error_handler(_llvmErrorHandler,
    // static_cast<void*>(&clang->getDiagnostics()));
    // NOTE! Can only be set once.
    llvm::install_fatal_error_handler(_llvmErrorHandler, nullptr);

    return SLANG_OK;
}


bool LLVMDownstreamCompiler::canConvert(const ArtifactDesc& from, const ArtifactDesc& to)
{
    return false;
}

SlangResult LLVMDownstreamCompiler::convert(
    IArtifact* from,
    const ArtifactDesc& to,
    IArtifact** outArtifact)
{
    return SLANG_E_NOT_IMPLEMENTED;
}

SlangResult LLVMDownstreamCompiler::getVersionString(slang::IBlob** outVersionString)
{
    StringBuilder versionString;
    // Append the version
    m_desc.version.append(versionString);

    // Really we should have a hash to identify the specific version.
    // For now we'll fall back to just using the timestamp

    {
        // If we don't have the commitHash, we use the library timestamp, to uniquely identify.
        versionString << " "
                      << SharedLibraryUtils::getSharedLibraryTimestamp(
                             (void*)createLLVMDownstreamCompiler_V4);
    }

    *outVersionString = StringBlob::moveCreate(versionString).detach();
    return SLANG_OK;
}

void* LLVMDownstreamCompiler::castAs(const Guid& guid)
{
    if (auto ptr = getInterface(guid))
    {
        return ptr;
    }
    return getObject(guid);
}

void* LLVMDownstreamCompiler::getInterface(const Guid& guid)
{
    if (guid == ISlangUnknown::getTypeGuid() || guid == ICastable::getTypeGuid() ||
        guid == IDownstreamCompiler::getTypeGuid())
    {
        return static_cast<IDownstreamCompiler*>(this);
    }
    return nullptr;
}

void* LLVMDownstreamCompiler::getObject(const Guid& guid)
{
    SLANG_UNUSED(guid);
    return nullptr;
}

SlangResult LLVMDownstreamCompiler::compile(
    const CompileOptions& inOptions,
    IArtifact** outArtifact)
{
    if (!isVersionCompatible(inOptions))
    {
        // Not possible to compile with this version of the interface.
        return SLANG_E_NOT_IMPLEMENTED;
    }

    CompileOptions options = getCompatibleVersion(&inOptions);

    // Currently supports single source file
    if (options.sourceArtifacts.count != 1)
    {
        return SLANG_FAIL;
    }
    IArtifact* sourceArtifact = options.sourceArtifacts[0];

    _ensureSufficientStack();

    static const SlangResult initLLVMResult = _initLLVM();
    SLANG_RETURN_ON_FAIL(initLLVMResult);

    std::unique_ptr<CompilerInstance> clang(new CompilerInstance());
    IntrusiveRefCntPtr<DiagnosticIDs> diagID(new DiagnosticIDs());

    // Register the support for object-file-wrapped Clang modules.
    auto pchOps = clang->getPCHContainerOperations();
    pchOps->registerWriter(std::make_unique<ObjectFilePCHContainerWriter>());
    pchOps->registerReader(std::make_unique<ObjectFilePCHContainerReader>());

    IntrusiveRefCntPtr<DiagnosticOptions> diagOpts = new DiagnosticOptions();

    ComPtr<IArtifactDiagnostics> diagnostics(new ArtifactDiagnostics);


    // TODO(JS): We might just want this to talk directly to the listener.
    // For now we just buffer up.
    BufferedDiagnosticConsumer diagsBuffer(diagnostics);

    IntrusiveRefCntPtr<DiagnosticsEngine> diags =
        new DiagnosticsEngine(diagID, diagOpts, &diagsBuffer, false);

    ComPtr<ISlangBlob> sourceBlob;
    SLANG_RETURN_ON_FAIL(sourceArtifact->loadBlob(ArtifactKeep::Yes, sourceBlob.writeRef()));

    const auto sourceSlice = StringUtil::getSlice(sourceBlob);
    StringRef sourceStringRef(sourceSlice.begin(), sourceSlice.getLength());

    auto sourceBuffer = llvm::MemoryBuffer::getMemBuffer(sourceStringRef);

    auto& invocation = clang->getInvocation();

    std::string verboseOutputString;

    // Capture all of the verbose output into a buffer, so not writen to stdout
    clang->setVerboseOutputStream(std::make_unique<llvm::raw_string_ostream>(verboseOutputString));

    SmallVector<char> output;
    clang->setOutputStream(std::make_unique<llvm::raw_svector_ostream>(output));

    frontend::ActionKind action = frontend::ActionKind::EmitLLVMOnly;

    // EmitCodeGenOnly doesn't appear to actually emit anything
    // EmitLLVM outputs LLVM assembly
    // EmitLLVMOnly doesn't 'emit' anything, but the IR that is produced is accessible, from the
    // 'action'.

    action = frontend::ActionKind::EmitLLVMOnly;

    // action = frontend::ActionKind::EmitBC;
    // action = frontend::ActionKind::EmitLLVM;
    //
    // action = frontend::ActionKind::EmitCodeGenOnly;
    // action = frontend::ActionKind::EmitObj;
    // action = frontend::ActionKind::EmitAssembly;

    Language language;
    LangStandard::Kind langStd;
    switch (options.sourceLanguage)
    {
    case SLANG_SOURCE_LANGUAGE_CPP:
        {
            language = Language::CXX;
            langStd = LangStandard::Kind::lang_cxx17;
            break;
        }
    case SLANG_SOURCE_LANGUAGE_C:
        {
            language = Language::C;
            langStd = LangStandard::Kind::lang_c17;
            break;
        }
    default:
        {
            return SLANG_E_NOT_AVAILABLE;
        }
    }

    const InputKind inputKind(language, InputKind::Format::Source);

    {
        auto& opts = invocation.getFrontendOpts();

        // Add the source
        // TODO(JS): For the moment this kind of include does *NOT* show a input source filename
        // not super surprising as one isn't set, but it's not clear how one would be set when the
        // input is a memory buffer. For Slang usage, this probably isn't an issue, because it's
        // *output* typically holds #line directives.
        {

            FrontendInputFile inputFile(*sourceBuffer, inputKind);
            opts.Inputs.push_back(inputFile);
        }

        opts.ProgramAction = action;
    }

    {
        auto& opts = invocation.getPreprocessorOpts();

        // Add definition so that 'LLVM/Clang' compilations can be recognized
        opts.addMacroDef("SLANG_LLVM");

        for (const auto& define : options.defines)
        {
            const Index index = asStringSlice(define.nameWithSig).indexOf('(');
            if (index >= 0)
            {
                // Interface does not support having a signature.
                return SLANG_E_NOT_AVAILABLE;
            }

            // TODO(JS): NOTE! The options do not support setting a *value* just that a macro is
            // defined. So strictly speaking, we should probably have a warning/error if the value
            // is not appropriate
            opts.addMacroDef(define.nameWithSig.begin());
        }
    }


    llvm::Triple targetTriple;
    {
        auto& opts = invocation.getTargetOpts();

        opts.Triple = LLVM_DEFAULT_TARGET_TRIPLE;

        // A code model isn't set by default, "default" seems to fit the bill here
        opts.CodeModel = "default";

        targetTriple = llvm::Triple(opts.Triple);
    }

    {
        auto opts = invocation.getLangOpts();

        std::vector<std::string> includes;
        for (const auto& includePath : options.includePaths)
        {
            includes.push_back(includePath.begin());
        }

        clang::CompilerInvocation::setLangDefaults(
            *opts,
            inputKind,
            targetTriple,
            includes,
            langStd);

        if (options.floatingPointMode == DownstreamCompileOptions::FloatingPointMode::Fast)
        {
            opts->FastMath = true;
        }
    }

    {
        auto& opts = invocation.getHeaderSearchOpts();

        // These only work if the resource directory is setup (or a virtual file system points to
        // it)
        opts.UseBuiltinIncludes = true;
        opts.UseStandardSystemIncludes = true;
        opts.UseStandardCXXIncludes = true;

        /// Use libc++ instead of the default libstdc++.
        // opts.UseLibcxx = true;
    }


    {
        auto& opts = invocation.getCodeGenOpts();

        // Set to -O optimization level
        opts.OptimizationLevel = _getOptimizationLevel(options.optimizationLevel);

        // Copy over the targets CodeModel
        opts.CodeModel = invocation.getTargetOpts().CodeModel;
    }

    // const llvm::opt::OptTable& opts = clang::driver::getDriverOptTable();

    // TODO(JS): Need a way to find in system search paths, for now we just don't bother
    //
    // The system search paths are for includes for compiler intrinsics it seems.
    // Infer the builtin include path if unspecified.
#if 0
    {
        auto& searchOpts = clang->getHeaderSearchOpts();
        if (searchOpts.UseBuiltinIncludes && searchOpts.ResourceDir.empty())
        {
            // TODO(JS): Hack - hard coded path such that we can test out the
            // resource directory functionality.

            StringRef binaryPath = "F:/dev/llvm-12.0/llvm-project-llvmorg-12.0.1/build.vs/Release/bin";

            // Dir is bin/ or lib/, depending on where BinaryPath is.

            // On Windows, libclang.dll is in bin/.
            // On non-Windows, libclang.so/.dylib is in lib/.
            // With a static-library build of libclang, LibClangPath will contain the
            // path of the embedding binary, which for LLVM binaries will be in bin/.
            // ../lib gets us to lib/ in both cases.
            SmallString<128> path = llvm::sys::path::parent_path(binaryPath);
            llvm::sys::path::append(path, Twine("lib") + CLANG_LIBDIR_SUFFIX, "clang", CLANG_VERSION_STRING);

            searchOpts.ResourceDir = path.c_str();
        }
    }
#endif

    // Create the actual diagnostics engine.
    clang->createDiagnostics();
    clang->setDiagnostics(diags.get());

    if (!clang->hasDiagnostics())
        return SLANG_FAIL;

    //
    clang->createFileManager();
    clang->createSourceManager(clang->getFileManager());


    std::unique_ptr<LLVMContext> llvmContext = std::make_unique<LLVMContext>();

    clang::CodeGenAction* codeGenAction = nullptr;
    std::unique_ptr<FrontendAction> act;

    {
        // If we are going to just emit IR, we need to have access to the underlying type
        if (action == frontend::ActionKind::EmitLLVMOnly)
        {
            EmitLLVMOnlyAction* llvmOnlyAction = new EmitLLVMOnlyAction(llvmContext.get());
            codeGenAction = llvmOnlyAction;
            // Make act the owning ptr
            act = std::unique_ptr<FrontendAction>(llvmOnlyAction);
        }
        else
        {
            act = CreateFrontendAction(*clang);
        }

        if (!act)
        {
            return SLANG_FAIL;
        }

        const bool compileSucceeded = clang->ExecuteAction(*act);

        // If the compilation failed make sure, we have an error
        if (!compileSucceeded)
        {
            diagnostics->requireErrorDiagnostic();
        }

        if (!compileSucceeded || diagsBuffer.hasError())
        {
            diagnostics->setResult(SLANG_FAIL);

            auto artifact = ArtifactUtil::createArtifact(
                ArtifactDesc::make(ArtifactKind::None, ArtifactPayload::None));
            ArtifactUtil::addAssociated(artifact, diagnostics);

            *outArtifact = artifact.detach();
            return SLANG_OK;
        }
    }

    std::unique_ptr<llvm::Module> module;

    switch (action)
    {
    case frontend::ActionKind::EmitLLVM:
        {
            // LLVM output is text, that must be zero terminated
            output.push_back(char(0));

            StringRef identifier;
            StringRef data(output.begin(), output.size() - 1);

            MemoryBufferRef memoryBufferRef(data, identifier);

            SMDiagnostic err;
            module = llvm::parseIR(memoryBufferRef, err, *llvmContext);
            break;
        }
    case frontend::ActionKind::EmitBC:
        {
            StringRef identifier;
            StringRef data(output.begin(), output.size());

            MemoryBufferRef memoryBufferRef(data, identifier);

            SMDiagnostic err;
            module = llvm::parseIR(memoryBufferRef, err, *llvmContext);
            break;
        }
    case frontend::ActionKind::EmitLLVMOnly:
        {
            // Get the module produced by the action
            module = codeGenAction->takeModule();
            break;
        }
    }

    switch (options.targetType)
    {
    // TODO(JS): Shared library may not be appropriate, but as long as the 'shared library' is
    // never accessed as a blob all is good.
    case SLANG_SHADER_SHARED_LIBRARY:

    // TODO(JS):
    // Hmm. What does this even mean?
    // I guess the idea is it's 'SHADER' style, but is runnable on the host.
    case SLANG_SHADER_HOST_CALLABLE:
        {
            // Try running something in the module on the JIT
            std::unique_ptr<llvm::orc::LLJIT> jit;
            {
                // Create the JIT

                LLJITBuilder jitBuilder;

                Expected<std::unique_ptr<llvm::orc::LLJIT>> expectJit = jitBuilder.create();
                if (!expectJit)
                {
                    /* JS: NOTE!

                    It is worth saying there can be some odd issues around creating the JIT - if
                    LLVM-C is linked against.

                    If it is then LLVM will likely startup saying LLVM-C isn't found.
                    BUT if you have LLVM *installed* on your system (as is reasonable to do from a
                    LLVM distro, then at startup it *MIGHT* find a LLVM-C dll in that installation
                    (ie nothing to do with the version of LLVM linked with). This will likely lead
                    to an odd error saying the 'triple can't be found' and that no targets are
                    registered.

                    Also note that the behavior *may* be different with Debug/Release - because of
                    how the linked resolves symbols that are multiply defined.

                    If there are problems creating the JIT, check that LLVM-C is not linked against
                    (it should be disabled in the premake).
                    */

                    auto err = expectJit.takeError();

                    std::string jitErrorString;
                    llvm::raw_string_ostream jitErrorStream(jitErrorString);

                    jitErrorStream << err;

                    ArtifactDiagnostic diagnostic;

                    StringBuilder buf;
                    buf << "Unable to create JIT engine: " << jitErrorString.c_str();

                    diagnostic.severity = ArtifactDiagnostic::Severity::Error;
                    diagnostic.stage = ArtifactDiagnostic::Stage::Link;
                    diagnostic.text = TerminatedCharSlice(buf.getBuffer(), buf.getLength());

                    // Add the error
                    diagnostics->add(diagnostic);
                    diagnostics->setResult(SLANG_FAIL);

                    auto artifact = ArtifactUtil::createArtifact(
                        ArtifactDesc::make(ArtifactKind::None, ArtifactPayload::None));
                    ArtifactUtil::addAssociated(artifact, diagnostics);

                    *outArtifact = artifact.detach();
                    return SLANG_OK;
                }
                jit = std::move(*expectJit);
            }

            // Used the following link to test this out
            // https://www.llvm.org/docs/ORCv2.html
            // https://www.llvm.org/docs/ORCv2.html#processandlibrarysymbols

            {
                auto& es = jit->getExecutionSession();

                const DataLayout& dl = jit->getDataLayout();
                MangleAndInterner mangler(es, dl);

                // The name of the lib must be unique. Should be here as we are only thing adding
                // libs
                auto stdcLibExpected = es.createJITDylib("stdc");

                if (stdcLibExpected)
                {
                    auto& stdcLib = *stdcLibExpected;

                    // Add all the symbolmap
                    SymbolMap symbolMap;

                    // symbolMap.insert(std::make_pair(mangler("sin"),
                    // JITEvaluatedSymbol::fromPointer(static_cast<double (*)(double)>(&sin))));

                    {
                        static const NameAndFunc funcs[] = {SLANG_LLVM_FUNCS(
                            SLANG_LLVM_FUNC) SLANG_PLATFORM_FUNCS(SLANG_LLVM_FUNC)};

                        for (auto& func : funcs)
                        {
                            symbolMap.insert(std::make_pair(
                                mangler(func.name),
                                JITEvaluatedSymbol::fromPointer(func.func)));
                        }
                    }

#if SLANG_PTR_IS_32 && SLANG_VC
                    {
                        // https://docs.microsoft.com/en-us/windows/win32/devnotes/-win32-alldiv
                        symbolMap.insert(std::make_pair(
                            mangler("_alldiv"),
                            JITEvaluatedSymbol::fromPointer(WinSpecific::_alldiv)));
                        symbolMap.insert(std::make_pair(
                            mangler("_allrem"),
                            JITEvaluatedSymbol::fromPointer(WinSpecific::_allrem)));
                        symbolMap.insert(std::make_pair(
                            mangler("_aullrem"),
                            JITEvaluatedSymbol::fromPointer(WinSpecific::_aullrem)));
                        symbolMap.insert(std::make_pair(
                            mangler("_aulldiv"),
                            JITEvaluatedSymbol::fromPointer(WinSpecific::_aulldiv)));
                    }
#endif

                    if (auto err = stdcLib.define(absoluteSymbols(symbolMap)))
                    {
                        return SLANG_FAIL;
                    }

                    // Required or the symbols won't be found
                    jit->getMainJITDylib().addToLinkOrder(stdcLib);
                }
            }

            ThreadSafeModule threadSafeModule(std::move(module), std::move(llvmContext));

            if (auto err = jit->addIRModule(std::move(threadSafeModule)))
            {
                return SLANG_FAIL;
            }

            if (auto err = jit->initialize(jit->getMainJITDylib()))
            {
                return SLANG_FAIL;
            }

            // Create the shared library
            ComPtr<ISlangSharedLibrary> sharedLibrary(new LLVMJITSharedLibrary(std::move(jit)));

            // Work out the ArtifactDesc
            const auto targetDesc = ArtifactDescUtil::makeDescForCompileTarget(options.targetType);

            auto artifact = ArtifactUtil::createArtifact(targetDesc);
            ArtifactUtil::addAssociated(artifact, diagnostics);

            artifact->addRepresentation(sharedLibrary);

            *outArtifact = artifact.detach();
            return SLANG_OK;
        }
    }

    return SLANG_FAIL;
}

} // namespace slang_llvm

extern "C" SLANG_DLL_EXPORT SlangResult
createLLVMDownstreamCompiler_V4(const SlangUUID& intfGuid, Slang::IDownstreamCompiler** out)
{
    Slang::ComPtr<slang_llvm::LLVMDownstreamCompiler> compiler(
        new slang_llvm::LLVMDownstreamCompiler);

    if (auto ptr = compiler->castAs(intfGuid))
    {
        compiler.detach();
        *out = (Slang::IDownstreamCompiler*)ptr;
        return SLANG_OK;
    }

    return SLANG_E_NO_INTERFACE;
}
