// main.cpp

// This file implements the entry point for `slangi`, an interpreter for the Slang language.

#include "../../source/core/slang-basic.h"
#include "core/slang-io.h"
#include "slang-com-ptr.h"
#include "slang.h"

using namespace Slang;
using namespace slang;

void printUsage()
{
    printf("Slang Interpreter (Experimental)\n");
    printf("Compile and interpret Slang code.\n");
    printf("Usage: slangi [options] <filename>\n");
    printf("Options:\n");
    printf("  -entry <name>   Specify the entry point function name to run. (default: main)\n");
    printf("  -disasm         Disassemble the bytecode after compilation.\n");
    printf("  -help           Show this help message\n");
}

void maybePrintDiagnostic(const ComPtr<slang::IBlob>& diagnosticBlob)
{
    if (diagnosticBlob)
    {
        const char* diagText = (const char*)diagnosticBlob->getBufferPointer();
        fprintf(stderr, "%s\n", diagText);
    }
}

SlangResult compileAndInterpret(
    UnownedStringSlice fileName,
    const char* entryPointName,
    bool disasm,
    int argc,
    const char* const* argv)
{
    ComPtr<slang::IGlobalSession> globalSession;
    SLANG_RETURN_ON_FAIL(slang_createGlobalSession(SLANG_API_VERSION, globalSession.writeRef()));
    slang::TargetDesc targetDesc = {};
    targetDesc.format = SLANG_HOST_VM;
    slang::SessionDesc sessionDesc = {};
    sessionDesc.targetCount = 1;
    sessionDesc.targets = &targetDesc;
    sessionDesc.compilerOptionEntryCount = 0;
    String pathName = Path::getParentDirectory(fileName);
    String moduleName = Path::getFileNameWithoutExt(fileName);
    const char* searchPaths[] = {pathName.getBuffer()};
    if (pathName.getLength())
    {
        sessionDesc.searchPathCount = 1;
        sessionDesc.searchPaths = searchPaths;
    }
    ComPtr<slang::ISession> session;
    SLANG_RETURN_ON_FAIL(globalSession->createSession(sessionDesc, session.writeRef()));

    ComPtr<slang::IBlob> diagnosticBlob;
    auto module = session->loadModule(moduleName.getBuffer(), diagnosticBlob.writeRef());
    if (!module)
    {
        maybePrintDiagnostic(diagnosticBlob);
        return SLANG_FAIL;
    }
    ComPtr<slang::IEntryPoint> entryPoint;
    if (SLANG_FAILED(module->findAndCheckEntryPoint(
            entryPointName,
            SLANG_STAGE_DISPATCH,
            entryPoint.writeRef(),
            diagnosticBlob.writeRef())))
    {
        maybePrintDiagnostic(diagnosticBlob);
        return SLANG_FAIL;
    }

    ComPtr<slang::IComponentType> compositeComponent;
    slang::IComponentType* components[] = {module, entryPoint.get()};
    if (SLANG_FAILED(session->createCompositeComponentType(
            components,
            2,
            compositeComponent.writeRef(),
            diagnosticBlob.writeRef())))
    {
        maybePrintDiagnostic(diagnosticBlob);
        return SLANG_FAIL;
    }

    ComPtr<slang::IComponentType> linkedProgram;
    if (SLANG_FAILED(compositeComponent->link(linkedProgram.writeRef(), diagnosticBlob.writeRef())))
    {
        maybePrintDiagnostic(diagnosticBlob);
        return SLANG_FAIL;
    }
    ComPtr<slang::IBlob> code;

    if (SLANG_FAILED(linkedProgram->getTargetCode(0, code.writeRef(), diagnosticBlob.writeRef())))
    {
        maybePrintDiagnostic(diagnosticBlob);
        return SLANG_FAIL;
    }

    if (code->getBufferSize() == 0)
    {
        return SLANG_FAIL;
    }

    if (disasm)
    {
        ComPtr<slang::IBlob> disasmBlob;
        if (SLANG_FAILED(slang_disassembleByteCode(code, disasmBlob.writeRef())))
        {
            maybePrintDiagnostic(diagnosticBlob);
            return SLANG_FAIL;
        }
        const char* disasmText = (const char*)disasmBlob->getBufferPointer();
        printf("%s\n", disasmText);
    }

    // Create a byte code runner and interpret the code.
    ComPtr<slang::IByteCodeRunner> runner;
    slang::ByteCodeRunnerDesc runnerDesc = {};
    SLANG_RETURN_ON_FAIL(slang_createByteCodeRunner(&runnerDesc, runner.writeRef()));
    if (SLANG_FAILED(runner->loadModule(code)))
    {
        runner->getErrorString(diagnosticBlob.writeRef());
        maybePrintDiagnostic(diagnosticBlob);
    }
    auto funcIndex = runner->findFunctionByName(entryPointName);
    if (funcIndex < 0)
    {
        printf("Function '%s' not found in byte code.\n", entryPointName);
        return SLANG_FAIL;
    }

    if (SLANG_FAILED(runner->selectFunctionByIndex((uint32_t)funcIndex)))
    {
        runner->getErrorString(diagnosticBlob.writeRef());
        maybePrintDiagnostic(diagnosticBlob);
        return SLANG_FAIL;
    }

    struct Arguments
    {
        uint32_t argc;
        const char* const* argv;
    };
    Arguments args;
    args.argc = argc;
    args.argv = argv;
    void* arguments = nullptr;
    size_t argSize = 0;
    slang::ByteCodeFuncInfo funcInfo;
    if (SLANG_FAILED(runner->getFunctionInfo((uint32_t)funcIndex, &funcInfo)))
    {
        runner->getErrorString(diagnosticBlob.writeRef());
        maybePrintDiagnostic(diagnosticBlob);
        return SLANG_FAIL;
    }
    if (funcInfo.parameterCount == 2)
    {
        arguments = &args;
        argSize = sizeof(Arguments);
    }
    if (SLANG_FAILED(runner->execute(arguments, argSize)))
    {
        runner->getErrorString(diagnosticBlob.writeRef());
        maybePrintDiagnostic(diagnosticBlob);
        return SLANG_FAIL;
    }
    size_t returnValueSize = 0;
    void* returnVal = runner->getReturnValue(&returnValueSize);
    SlangResult result = SLANG_OK;
    memcpy(&result, returnVal, returnValueSize);
    return result;
}

int main(int argc, const char* const* argv)
{
    String entryPointName = toSlice("main");
    UnownedStringSlice fileName;
    bool disasm = false;
    int innerArgIndex = 0;
    if (argc < 2)
    {
        printUsage();
        return 0;
    }
    for (auto i = 1; i < argc; i++)
    {
        auto arg = UnownedStringSlice(argv[i]);
        if (arg == "-entry")
        {
            entryPointName = UnownedStringSlice(argv[++i]);
        }
        else if (arg == "-help" || arg == "--help")
        {
            printUsage();
            return 0;
        }
        else if (arg == "-disasm")
        {
            disasm = true;
        }
        else if (arg.startsWith("-"))
        {
            fprintf(stderr, "Unknown option: %s\n", arg.begin());
            printUsage();
            return -1;
        }
        else
        {
            fileName = arg;
            innerArgIndex = i;
            break;
        }
    }
    if (!fileName.getLength())
    {
        printUsage();
        return 0;
    }

    auto result = compileAndInterpret(
        fileName,
        entryPointName.getBuffer(),
        disasm,
        argc - innerArgIndex,
        argv + innerArgIndex);
    slang::shutdown();
    return result;
}
