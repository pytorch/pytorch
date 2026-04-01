// unit-test-com-host-callable.cpp

#include "../../source/core/slang-byte-encode-util.h"
#include "../../source/core/slang-list.h"
#include "slang-com-helper.h"
#include "slang-com-ptr.h"
#include "slang.h"
#include "unit-test/slang-unit-test.h"

#include <stdio.h>
#include <stdlib.h>

namespace
{ // anonymous

// Slang namespace is used for elements support code (like core) which we use here
// for ComPtr<> and TestToolUtil
using namespace Slang;

// For the moment we have to explicitly write the Slang COM interface in C++ code. It *MUST* match
// the interface in the slang source
// As it stands all interfaces need to derive from ISlangUnknown (or IUnknown).
class IDoThings : public ISlangUnknown
{
public:
    virtual SLANG_NO_THROW int SLANG_MCALL doThing(int a, int b) = 0;
    virtual SLANG_NO_THROW int SLANG_MCALL calcHash(const char* in) = 0;
};

class ICountGood : public ISlangUnknown
{
public:
    virtual SLANG_NO_THROW int SLANG_MCALL nextCount() = 0;
};

static int _calcHash(const char* in)
{
    int hash = 0;
    for (; *in; ++in)
    {
        // A very poor hash function
        hash = hash * 13 + *in;
    }
    return hash;
}

class DoThings : public IDoThings
{
public:
    // We don't need queryInterface for this impl, or ref counting
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    queryInterface(SlangUUID const& uuid, void** outObject) SLANG_OVERRIDE
    {
        return SLANG_E_NOT_IMPLEMENTED;
    }
    virtual SLANG_NO_THROW uint32_t SLANG_MCALL addRef() SLANG_OVERRIDE { return 1; }
    virtual SLANG_NO_THROW uint32_t SLANG_MCALL release() SLANG_OVERRIDE { return 1; }

    // IDoThings
    virtual SLANG_NO_THROW int SLANG_MCALL doThing(int a, int b) SLANG_OVERRIDE
    {
        return a + b + 1;
    }
    virtual SLANG_NO_THROW int SLANG_MCALL calcHash(const char* in) SLANG_OVERRIDE
    {
        return (int)_calcHash(in);
    }
};

class CountGood : public ICountGood
{
public:
    // We don't need queryInterface for this impl, or ref counting
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    queryInterface(SlangUUID const& uuid, void** outObject) SLANG_OVERRIDE
    {
        return SLANG_E_NOT_IMPLEMENTED;
    }
    virtual SLANG_NO_THROW uint32_t SLANG_MCALL addRef() SLANG_OVERRIDE { return 1; }
    virtual SLANG_NO_THROW uint32_t SLANG_MCALL release() SLANG_OVERRIDE { return 1; }

    // ICountGood
    virtual SLANG_NO_THROW int SLANG_MCALL nextCount() SLANG_OVERRIDE { return m_count++; }

    int m_count = 0;
};

struct ComTestContext
{
    ComTestContext(UnitTestContext* context)
        : m_unitTestContext(context)
    {
        slang::IGlobalSession* slangSession = m_unitTestContext->slangGlobalSession;

        m_defaultCppCompiler =
            slangSession->getDefaultDownstreamCompiler(SLANG_SOURCE_LANGUAGE_CPP);

        m_hostHostCallableCompiler = slangSession->getDownstreamCompilerForTransition(
            SLANG_CPP_SOURCE,
            SLANG_HOST_HOST_CALLABLE);
        m_shaderHostCallableCompiler = slangSession->getDownstreamCompilerForTransition(
            SLANG_CPP_SOURCE,
            SLANG_SHADER_HOST_CALLABLE);
    }

    SlangResult runTests()
    {
        slang::IGlobalSession* slangSession = m_unitTestContext->slangGlobalSession;

        // TODO(JS):
        // Care is needed around this in normal testing. `slang-llvm` is whatever was asked for for
        // when premake was built when the target is specified. Otherwise it is the `default` which
        // is typically 64 bit during development.
        //
        // On CI we should be okay, because it should download the correct `slang-llvm` for the
        // build (as it packages up with it). But for normal development, that can easily not be the
        // case (for example changing to 32 bit build in VS is a problem).
        //
        // Make sure to run
        //
        // ```
        // premake --arch=x86 --deps=true
        // ```
        //
        // for the actual target/arch(!)

        const bool hasLlvm =
            SLANG_SUCCEEDED(slangSession->checkPassThroughSupport(SLANG_PASS_THROUGH_LLVM));

        SlangPassThrough cppCompiler = SLANG_PASS_THROUGH_NONE;

        {
            const SlangPassThrough cppCompilers[] = {
                SLANG_PASS_THROUGH_VISUAL_STUDIO,
                SLANG_PASS_THROUGH_GCC,
                SLANG_PASS_THROUGH_CLANG,
            };
            // Do we have a C++ compiler
            for (const auto compiler : cppCompilers)
            {
                if (SLANG_SUCCEEDED(slangSession->checkPassThroughSupport(compiler)))
                {
                    cppCompiler = compiler;
                    break;
                }
            }
        }

        // If we have an *actual* C++ compile rtest on that first
        if (cppCompiler != SLANG_PASS_THROUGH_NONE)
        {
            slangSession->setDefaultDownstreamCompiler(SLANG_SOURCE_LANGUAGE_CPP, cppCompiler);

            slangSession->setDownstreamCompilerForTransition(
                SLANG_CPP_SOURCE,
                SLANG_SHADER_HOST_CALLABLE,
                cppCompiler);
            slangSession->setDownstreamCompilerForTransition(
                SLANG_CPP_SOURCE,
                SLANG_HOST_HOST_CALLABLE,
                cppCompiler);

            SLANG_RETURN_ON_FAIL(_runTest());
        }

        // Reset the compiler that's used for host-callable
        _reset();

        // If we have Llvm it is the default host callable compiler
        if (hasLlvm)
        {
            // Should run via slang-llvm
            SLANG_RETURN_ON_FAIL(_runTest());
        }

        return SLANG_OK;
    }

    void _reset()
    {
        slang::IGlobalSession* slangSession = m_unitTestContext->slangGlobalSession;
        slangSession->setDefaultDownstreamCompiler(SLANG_SOURCE_LANGUAGE_CPP, m_defaultCppCompiler);

        slangSession->setDownstreamCompilerForTransition(
            SLANG_CPP_SOURCE,
            SLANG_SHADER_HOST_CALLABLE,
            m_shaderHostCallableCompiler);
        slangSession->setDownstreamCompilerForTransition(
            SLANG_CPP_SOURCE,
            SLANG_HOST_HOST_CALLABLE,
            m_hostHostCallableCompiler);
    }

    ~ComTestContext() { _reset(); }

    SlangResult _runTest();

    UnitTestContext* m_unitTestContext;

    SlangPassThrough m_defaultCppCompiler;
    SlangPassThrough m_hostHostCallableCompiler;
    SlangPassThrough m_shaderHostCallableCompiler;
};

SlangResult ComTestContext::_runTest()
{
    slang::IGlobalSession* slangSession = m_unitTestContext->slangGlobalSession;

    // Create a compile request
    Slang::ComPtr<slang::ICompileRequest> request;
    SLANG_ALLOW_DEPRECATED_BEGIN
    SLANG_RETURN_ON_FAIL(slangSession->createCompileRequest(request.writeRef()));
    SLANG_ALLOW_DEPRECATED_END

    // We want to compile to 'HOST_CALLABLE' here such that we can execute the Slang code.
    //
    // Note that it is possible to use HOST_HOST_CALLABLE, but this currently only works with
    // 'regular' C++ compilers not with `slang-llvm`.
    const int targetIndex = request->addCodeGenTarget(SLANG_SHADER_HOST_CALLABLE);

    // Set the target flag to indicate that we want to compile all into a library.
    request->setTargetFlags(targetIndex, SLANG_TARGET_FLAG_GENERATE_WHOLE_PROGRAM);

    request->setOptimizationLevel(SLANG_OPTIMIZATION_LEVEL_NONE);
    request->setDebugInfoLevel(SLANG_DEBUG_INFO_LEVEL_STANDARD);

    // Add the translation unit
    const int translationUnitIndex =
        request->addTranslationUnit(SLANG_SOURCE_LANGUAGE_SLANG, nullptr);

    // Set the source file for the translation unit
    request->addTranslationUnitSourceFile(
        translationUnitIndex,
        "tools/slang-unit-test/unit-test-com-host-callable.slang");

    const SlangResult compileRes = request->compile();

    // Even if there were no errors that forced compilation to fail, the
    // compiler may have produced "diagnostic" output such as warnings.
    // We will go ahead and print that output here.
    //
    if (auto diagnostics = request->getDiagnosticOutput())
    {
        printf("%s", diagnostics);
    }

    // Get the 'shared library' (note that this doesn't necessarily have to be implemented as a
    // shared library it's just an interface to executable code).
    ComPtr<ISlangSharedLibrary> sharedLibrary;
    SLANG_RETURN_ON_FAIL(request->getTargetHostCallable(0, sharedLibrary.writeRef()));

    {
        typedef const char* (*Func)(const char*);
        Func func = (Func)sharedLibrary->findFuncByName("getString");

        if (!func)
        {
            return SLANG_FAIL;
        }

        String text = "Hello World!";
        String returnedText = func(text.getBuffer());

        SLANG_CHECK(text == returnedText);
    }
    {
        typedef int (*Func)(const char* text, IDoThings* doThings);

        Func func = (Func)sharedLibrary->findFuncByName("calcHash");

        if (!func)
        {
            return SLANG_FAIL;
        }

        DoThings doThings;

        String text("Hello");

        const int hash = func(text.getBuffer(), &doThings);

        SLANG_CHECK(hash == _calcHash(text.getBuffer()));
    }

    // Check accessing a global
    {
        typedef void (*SetFunc)(int v);
        typedef int (*GetFunc)();

        const auto setGlobal = (SetFunc)sharedLibrary->findFuncByName("setGlobal");
        const auto getGlobal = (GetFunc)sharedLibrary->findFuncByName("getGlobal");

        if (setGlobal == nullptr || getGlobal == nullptr)
        {
            return SLANG_FAIL;
        }

        // In the slang source it is set a default value
        SLANG_CHECK(getGlobal() == 10);

        for (Index i = 0; i < 10; ++i)
        {
            setGlobal(int(i));
            SLANG_CHECK(getGlobal() == i);
        }
    }

    // Check using a global interface
    {

        typedef void (*SetCounterFunc)(ICountGood* counter);
        typedef int (*NextCountFunc)();

        const auto setCounter = (SetCounterFunc)sharedLibrary->findFuncByName("setCounter");
        const auto nextCount = (NextCountFunc)sharedLibrary->findFuncByName("nextCount");

        if (setCounter == nullptr || nextCount == nullptr)
        {
            return SLANG_FAIL;
        }

        CountGood counter;

        ICountGood* counterIntf = &counter;

        setCounter(counterIntf);

        auto counterPtr = (ICountGood**)sharedLibrary->findSymbolAddressByName("globalCounter");
        SLANG_CHECK(counterPtr);
        if (!counterPtr)
        {
            return SLANG_FAIL;
        }

        for (Index i = 0; i < 10; ++i)
        {
            SLANG_CHECK(*counterPtr == &counter);

            const auto v = nextCount();
            SLANG_CHECK(v == i);
        }
    }

    return SLANG_OK;
}

} // namespace

SLANG_UNIT_TEST(comHostCallable)
{
#if SLANG_PTR_IS_32 && !SLANG_MICROSOFT_FAMILY
    // TODO(JS):
    // We can't currently run this test reliably on targets other than windows
    // Visual Studio DownstreamCompiler has support for 32 bit builds
    // Other targets generally build for the native environment which is almost always 64 bit,
    // and it requires other features to build/test 32 bit binaries on such systems.
    //
    // So we disable for any 32 bit non MS target for now
    return;
#endif

    ComTestContext context(unitTestContext);

    const auto result = context.runTests();

    SLANG_CHECK(SLANG_SUCCEEDED(result));
}
