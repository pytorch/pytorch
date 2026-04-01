namespace SlangITool
{
static void printCallback(const char* message, void* userData)
{
    auto stdWriters = (StdWriters*)userData;
    if (stdWriters)
    {
        stdWriters->getOut().print("%s", message);
    }
}

static SlangResult compileAndInterpret(
    slang::IGlobalSession* sharedSession,
    StdWriters* stdWriters,
    UnownedStringSlice fileName,
    const char* entryPointName,
    bool disasm,
    int argc,
    const char* const* argv)
{
    auto maybePrintDiagnostic = [&](const ComPtr<slang::IBlob>& diagnosticBlob)
    {
        if (diagnosticBlob)
        {
            const char* diagText = (const char*)diagnosticBlob->getBufferPointer();
            stdWriters->getError().print("%s\n", diagText);
        }
    };

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
        stdWriters->getOut().print("%s\n", disasmText);
        return SLANG_OK;
    }

    // Create a byte code runner and interpret the code.
    ComPtr<slang::IByteCodeRunner> runner;
    slang::ByteCodeRunnerDesc runnerDesc = {};
    SLANG_RETURN_ON_FAIL(slang_createByteCodeRunner(&runnerDesc, runner.writeRef()));
    runner->setPrintCallback(printCallback, stdWriters);

    if (SLANG_FAILED(runner->loadModule(code)))
    {
        runner->getErrorString(diagnosticBlob.writeRef());
        maybePrintDiagnostic(diagnosticBlob);
    }
    auto funcIndex = runner->findFunctionByName(entryPointName);
    if (funcIndex < 0)
    {
        stdWriters->getError().print("Function '%s' not found in byte code.\n", entryPointName);
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
    return SLANG_OK;
}

SlangResult innerMain(
    StdWriters* stdWriters,
    slang::IGlobalSession* sharedSession,
    int argc,
    const char* const* argv)
{
    StdWriters::setSingleton(stdWriters);

    // Assume we will used the shared session
    ComPtr<slang::IGlobalSession> session(sharedSession);

    // The sharedSession always has a pre-loaded core module.
    // This differed test checks if the command line has an option to setup the core module.
    // If so we *don't* use the sharedSession, and create a new session without the core module just
    // for this compilation.
    if (TestToolUtil::hasDeferredCoreModule(Index(argc - 1), argv + 1))
    {
        SLANG_RETURN_ON_FAIL(
            slang_createGlobalSessionWithoutCoreModule(SLANG_API_VERSION, session.writeRef()));
    }

    String entryPointName = toSlice("main");
    UnownedStringSlice fileName;
    bool disasm = false;
    int innerArgIndex = 0;
    if (argc < 2)
    {
        return SLANG_FAIL;
    }
    for (auto i = 1; i < argc; i++)
    {
        auto arg = UnownedStringSlice(argv[i]);
        if (arg == "-entry")
        {
            entryPointName = UnownedStringSlice(argv[++i]);
        }
        else if (arg == "-disasm")
        {
            disasm = true;
        }
        else if (arg.startsWith("-"))
        {
            return SLANG_FAIL;
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
        return SLANG_FAIL;
    }

    auto result = compileAndInterpret(
        session,
        stdWriters,
        fileName,
        entryPointName.getBuffer(),
        disasm,
        argc - innerArgIndex,
        argv + innerArgIndex);

    return result;
}
} // namespace SlangITool
