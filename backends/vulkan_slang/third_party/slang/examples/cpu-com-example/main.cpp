// main.cpp

#include "slang-com-helper.h"
#include "slang-com-ptr.h"
#include "slang.h"

#include <stdio.h>

// This includes a useful small function for setting up the prelude (described more further below).
#include "../../source/core/slang-test-tool-util.h"
#include "examples/example-base/example-base.h"

// Slang namespace is used for elements support code (like core) which we use here
// for ComPtr<> and TestToolUtil
using namespace Slang;

static const ExampleResources resourceBase("cpu-com-example");

// For the moment we have to explicitly write the Slang COM interface in C++ code. It *MUST* match
// the interface in the slang source
// As it stands all interfaces need to derive from ISlangUnknown (or IUnknown).
class IDoThings : public ISlangUnknown
{
public:
    virtual SLANG_NO_THROW int SLANG_MCALL doThing(int a, int b) = 0;
    virtual SLANG_NO_THROW int SLANG_MCALL calcHash(const char* in) = 0;
    virtual SLANG_NO_THROW void SLANG_MCALL printMessage(const char* in) = 0;
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
    virtual SLANG_NO_THROW void SLANG_MCALL printMessage(const char* in) SLANG_OVERRIDE
    {
        printf("%s\n", in);
    }
};

static SlangResult _innerMain(int argc, char** argv)
{
    // NOTE! This example only works if `slang-llvm` or a C++ compiler that Slang supports is
    // available.

    // Create the session
    ComPtr<slang::IGlobalSession> slangSession;
    slangSession.attach(spCreateSession(NULL));

    // Set up the prelude
    // NOTE: This isn't strictly necessary, as preludes are embedded in the binary.
    TestToolUtil::setSessionDefaultPreludeFromExePath(argv[0], slangSession);

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

    // Add the translation unit
    const int translationUnitIndex =
        request->addTranslationUnit(SLANG_SOURCE_LANGUAGE_SLANG, nullptr);

    // Set the source file for the translation unit
    Slang::String path = resourceBase.resolveResource("shader.slang");
    request->addTranslationUnitSourceFile(translationUnitIndex, path.getBuffer());

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

    DoThings doThings;

    {
        auto doThingsPtr = (IDoThings**)sharedLibrary->findSymbolAddressByName("globalDoThings");
        if (!doThingsPtr)
        {
            return SLANG_FAIL;
        }
        // Set the global interface
        *doThingsPtr = &doThings;
    }

    // Test a free function
    {
        typedef const char* (*Func)(const char*);
        Func func = (Func)sharedLibrary->findFuncByName("getString");

        if (!func)
        {
            return SLANG_FAIL;
        }

        String text = "Hello World!";
        String returnedText = func(text.getBuffer());

        SLANG_ASSERT(text == returnedText);
    }

    // Test hash
    {
        typedef int (*Func)(const char* text);
        Func func = (Func)sharedLibrary->findFuncByName("calcHash");
        if (!func)
        {
            return SLANG_FAIL;
        }

        String text("Hello");
        const int hash = func(text.getBuffer());
        SLANG_ASSERT(hash == _calcHash(text.getBuffer()));
    }

    // Test printing
    {
        typedef void (*Func)(const char* text);

        Func func = (Func)sharedLibrary->findFuncByName("printMessage");

        if (!func)
        {
            return SLANG_FAIL;
        }
        func("Hello World!");
    }

    return SLANG_OK;
}

int exampleMain(int argc, char** argv)
{
    return SLANG_SUCCEEDED(_innerMain(argc, argv)) ? 0 : -1;
}
