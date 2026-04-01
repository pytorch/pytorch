#include "slang.h"

#include "../core/slang-archive-file-system.h"
#include "../core/slang-castable.h"
#include "../core/slang-io.h"
#include "../core/slang-performance-profiler.h"
#include "../core/slang-shared-library.h"
#include "../core/slang-string-util.h"
#include "../core/slang-type-convert-util.h"
#include "../core/slang-type-text-util.h"
// Artifact
#include "../compiler-core/slang-artifact-associated-impl.h"
#include "../compiler-core/slang-artifact-container-util.h"
#include "../compiler-core/slang-artifact-desc-util.h"
#include "../compiler-core/slang-artifact-impl.h"
#include "../compiler-core/slang-artifact-util.h"
#include "../compiler-core/slang-source-loc.h"
#include "../core/slang-file-system.h"
#include "../core/slang-memory-file-system.h"
#include "../core/slang-writer.h"
#include "core/slang-shared-library.h"
#include "slang-ast-dump.h"
#include "slang-check-impl.h"
#include "slang-check.h"
#include "slang-doc-ast.h"
#include "slang-doc-markdown-writer.h"
#include "slang-lookup.h"
#include "slang-lower-to-ir.h"
#include "slang-mangle.h"
#include "slang-module-library.h"
#include "slang-options.h"
#include "slang-parameter-binding.h"
#include "slang-parser.h"
#include "slang-preprocessor.h"
#include "slang-reflection-json.h"
#include "slang-repro.h"
#include "slang-serialize-ast.h"
#include "slang-serialize-container.h"
#include "slang-serialize-ir.h"
#include "slang-tag-version.h"
#include "slang-type-layout.h"

#include <sys/stat.h>

// Used to print exception type names in internal-compiler-error messages
#include <typeinfo>

extern Slang::String get_slang_cuda_prelude();
extern Slang::String get_slang_cpp_prelude();
extern Slang::String get_slang_hlsl_prelude();

namespace Slang
{


/* static */ const BaseTypeInfo BaseTypeInfo::s_info[Index(BaseType::CountOf)] = {
    {0, 0, uint8_t(BaseType::Void)},
    {uint8_t(sizeof(bool)), 0, uint8_t(BaseType::Bool)},
    {uint8_t(sizeof(int8_t)),
     BaseTypeInfo::Flag::Signed | BaseTypeInfo::Flag::Integer,
     uint8_t(BaseType::Int8)},
    {uint8_t(sizeof(int16_t)),
     BaseTypeInfo::Flag::Signed | BaseTypeInfo::Flag::Integer,
     uint8_t(BaseType::Int16)},
    {uint8_t(sizeof(int32_t)),
     BaseTypeInfo::Flag::Signed | BaseTypeInfo::Flag::Integer,
     uint8_t(BaseType::Int)},
    {uint8_t(sizeof(int64_t)),
     BaseTypeInfo::Flag::Signed | BaseTypeInfo::Flag::Integer,
     uint8_t(BaseType::Int64)},
    {uint8_t(sizeof(uint8_t)), BaseTypeInfo::Flag::Integer, uint8_t(BaseType::UInt8)},
    {uint8_t(sizeof(uint16_t)), BaseTypeInfo::Flag::Integer, uint8_t(BaseType::UInt16)},
    {uint8_t(sizeof(uint32_t)), BaseTypeInfo::Flag::Integer, uint8_t(BaseType::UInt)},
    {uint8_t(sizeof(uint64_t)), BaseTypeInfo::Flag::Integer, uint8_t(BaseType::UInt64)},
    {uint8_t(sizeof(uint16_t)), BaseTypeInfo::Flag::FloatingPoint, uint8_t(BaseType::Half)},
    {uint8_t(sizeof(float)), BaseTypeInfo::Flag::FloatingPoint, uint8_t(BaseType::Float)},
    {uint8_t(sizeof(double)), BaseTypeInfo::Flag::FloatingPoint, uint8_t(BaseType::Double)},
    {uint8_t(sizeof(char)),
     BaseTypeInfo::Flag::Signed | BaseTypeInfo::Flag::Integer,
     uint8_t(BaseType::Char)},
    {uint8_t(sizeof(intptr_t)),
     BaseTypeInfo::Flag::Signed | BaseTypeInfo::Flag::Integer,
     uint8_t(BaseType::IntPtr)},
    {uint8_t(sizeof(uintptr_t)), BaseTypeInfo::Flag::Integer, uint8_t(BaseType::UIntPtr)},
};

/* static */ bool BaseTypeInfo::check()
{
    for (Index i = 0; i < SLANG_COUNT_OF(s_info); ++i)
    {
        if (s_info[i].baseType != i)
        {
            SLANG_ASSERT(!"Inconsistency between the s_info table and BaseInfo");
            return false;
        }
    }
    return true;
}

/* static */ UnownedStringSlice BaseTypeInfo::asText(BaseType baseType)
{
    switch (baseType)
    {
    case BaseType::Void:
        return UnownedStringSlice::fromLiteral("void");
    case BaseType::Bool:
        return UnownedStringSlice::fromLiteral("bool");
    case BaseType::Int8:
        return UnownedStringSlice::fromLiteral("int8_t");
    case BaseType::Int16:
        return UnownedStringSlice::fromLiteral("int16_t");
    case BaseType::Int:
        return UnownedStringSlice::fromLiteral("int");
    case BaseType::Int64:
        return UnownedStringSlice::fromLiteral("int64_t");
    case BaseType::UInt8:
        return UnownedStringSlice::fromLiteral("uint8_t");
    case BaseType::UInt16:
        return UnownedStringSlice::fromLiteral("uint16_t");
    case BaseType::UInt:
        return UnownedStringSlice::fromLiteral("uint");
    case BaseType::UInt64:
        return UnownedStringSlice::fromLiteral("uint64_t");
    case BaseType::Half:
        return UnownedStringSlice::fromLiteral("half");
    case BaseType::Float:
        return UnownedStringSlice::fromLiteral("float");
    case BaseType::Double:
        return UnownedStringSlice::fromLiteral("double");
    case BaseType::Char:
        return UnownedStringSlice::fromLiteral("char");
    case BaseType::IntPtr:
        return UnownedStringSlice::fromLiteral("intptr_t");
    case BaseType::UIntPtr:
        return UnownedStringSlice::fromLiteral("uintptr_t");
    default:
        {
            SLANG_ASSERT(!"Unknown basic type");
            return UnownedStringSlice();
        }
    }
}

const char* getBuildTagString()
{
    if (UnownedStringSlice(SLANG_TAG_VERSION) == "0.0.0-unknown")
    {
        // If the tag is unknown, then we will try to get the timestamp of the shared library
        // and use that as the version string, so that we can at least return something
        // that uniquely identifies the build.
        static String timeStampString =
            String(SharedLibraryUtils::getSharedLibraryTimestamp((void*)spCreateSession));
        return timeStampString.getBuffer();
    }
    return SLANG_TAG_VERSION;
}


void Session::init()
{
    SLANG_ASSERT(BaseTypeInfo::check());


    _initCodeGenTransitionMap();

    ::memset(m_downstreamCompilerLocators, 0, sizeof(m_downstreamCompilerLocators));
    DownstreamCompilerUtil::setDefaultLocators(m_downstreamCompilerLocators);
    m_downstreamCompilerSet = new DownstreamCompilerSet;

    // Initialize name pool
    getNamePool()->setRootNamePool(getRootNamePool());
    m_completionTokenName = getNamePool()->getName("#?");

    m_sharedLibraryLoader = DefaultSharedLibraryLoader::getSingleton();

    // Set up the command line options
    initCommandOptions(m_commandOptions);

    // Set up shared AST builder
    m_sharedASTBuilder = new SharedASTBuilder;
    m_sharedASTBuilder->init(this);

    // And the global ASTBuilder
    auto builtinAstBuilder = m_sharedASTBuilder->getInnerASTBuilder();
    globalAstBuilder = builtinAstBuilder;

    // Make sure our source manager is initialized
    builtinSourceManager.initialize(nullptr, nullptr);

    // Built in linkage uses the built in builder
    m_builtinLinkage = new Linkage(this, builtinAstBuilder, nullptr);
    m_builtinLinkage->m_optionSet.set(CompilerOptionName::DebugInformation, DebugInfoLevel::None);

    // Because the `Session` retains the builtin `Linkage`,
    // we need to make sure that the parent pointer inside
    // `Linkage` doesn't create a retain cycle.
    //
    // This operation ensures that the parent pointer will
    // just be a raw pointer, so that the builtin linkage
    // doesn't keep the parent session alive.
    //
    m_builtinLinkage->_stopRetainingParentSession();

    // Create scopes for various language builtins.
    //
    // TODO: load these on-demand to avoid parsing
    // the core module code for languages the user won't use.

    baseLanguageScope = builtinAstBuilder->create<Scope>();

    // Will stay in scope as long as ASTBuilder
    baseModuleDecl =
        populateBaseLanguageModule(m_builtinLinkage->getASTBuilder(), baseLanguageScope);

    coreLanguageScope = builtinAstBuilder->create<Scope>();
    coreLanguageScope->nextSibling = baseLanguageScope;

    hlslLanguageScope = builtinAstBuilder->create<Scope>();
    hlslLanguageScope->nextSibling = coreLanguageScope;

    slangLanguageScope = builtinAstBuilder->create<Scope>();
    slangLanguageScope->nextSibling = hlslLanguageScope;

    glslLanguageScope = builtinAstBuilder->create<Scope>();
    glslLanguageScope->nextSibling = slangLanguageScope;

    glslModuleName = getNameObj("glsl");

    {
        for (Index i = 0; i < Index(SourceLanguage::CountOf); ++i)
        {
            m_defaultDownstreamCompilers[i] = PassThroughMode::None;
        }
        m_defaultDownstreamCompilers[Index(SourceLanguage::C)] = PassThroughMode::GenericCCpp;
        m_defaultDownstreamCompilers[Index(SourceLanguage::CPP)] = PassThroughMode::GenericCCpp;
        m_defaultDownstreamCompilers[Index(SourceLanguage::CUDA)] = PassThroughMode::NVRTC;
    }

    // Set up default prelude code for target languages that need a prelude
    m_languagePreludes[Index(SourceLanguage::CUDA)] = get_slang_cuda_prelude();
    m_languagePreludes[Index(SourceLanguage::CPP)] = get_slang_cpp_prelude();
    m_languagePreludes[Index(SourceLanguage::HLSL)] = get_slang_hlsl_prelude();

    if (!spirvCoreGrammarInfo)
        spirvCoreGrammarInfo = SPIRVCoreGrammarInfo::getEmbeddedVersion();
}

Module* Session::getBuiltinModule(slang::BuiltinModuleName name)
{
    auto info = getBuiltinModuleInfo(name);
    auto builtinLinkage = getBuiltinLinkage();
    auto moduleNameObj = builtinLinkage->getNamePool()->getName(info.name);
    RefPtr<Module> module;
    if (builtinLinkage->mapNameToLoadedModules.tryGetValue(moduleNameObj, module))
        return module.get();
    return nullptr;
}

void Session::_initCodeGenTransitionMap()
{
    // TODO(JS): Might want to do something about these in the future...

    // PassThroughMode getDownstreamCompilerRequiredForTarget(CodeGenTarget target);
    // SourceLanguage getDefaultSourceLanguageForDownstreamCompiler(PassThroughMode compiler);

    // Set up the default ways to do compilations between code gen targets
    auto& map = m_codeGenTransitionMap;

    // TODO(JS): There currently isn't a 'downstream compiler' for direct spirv output. If we did
    // it would presumably a transition from SlangIR to SPIRV.

    // For C and C++ we default to use the 'genericCCpp' compiler
    {
        const CodeGenTarget sources[] = {CodeGenTarget::CSource, CodeGenTarget::CPPSource};
        for (auto source : sources)
        {
            // We *don't* add a default for host callable, as we will determine what is suitable
            // depending on what is available. We prefer LLVM if that's available. If it's not we
            // can use generic C/C++ compiler

            map.addTransition(
                source,
                CodeGenTarget::ShaderSharedLibrary,
                PassThroughMode::GenericCCpp);
            map.addTransition(
                source,
                CodeGenTarget::HostSharedLibrary,
                PassThroughMode::GenericCCpp);
            map.addTransition(source, CodeGenTarget::HostExecutable, PassThroughMode::GenericCCpp);
            map.addTransition(source, CodeGenTarget::ObjectCode, PassThroughMode::GenericCCpp);
        }
    }


    // Add all the straightforward transitions
    map.addTransition(CodeGenTarget::CUDASource, CodeGenTarget::PTX, PassThroughMode::NVRTC);
    map.addTransition(CodeGenTarget::HLSL, CodeGenTarget::DXBytecode, PassThroughMode::Fxc);
    map.addTransition(CodeGenTarget::HLSL, CodeGenTarget::DXIL, PassThroughMode::Dxc);
    map.addTransition(CodeGenTarget::GLSL, CodeGenTarget::SPIRV, PassThroughMode::Glslang);
    map.addTransition(CodeGenTarget::Metal, CodeGenTarget::MetalLib, PassThroughMode::MetalC);
    map.addTransition(CodeGenTarget::WGSL, CodeGenTarget::WGSLSPIRV, PassThroughMode::Tint);
    // To assembly
    map.addTransition(CodeGenTarget::SPIRV, CodeGenTarget::SPIRVAssembly, PassThroughMode::Glslang);
    // We use glslang to turn SPIR-V into SPIR-V assembly.
    map.addTransition(
        CodeGenTarget::WGSLSPIRV,
        CodeGenTarget::WGSLSPIRVAssembly,
        PassThroughMode::Glslang);
    map.addTransition(CodeGenTarget::DXIL, CodeGenTarget::DXILAssembly, PassThroughMode::Dxc);
    map.addTransition(
        CodeGenTarget::DXBytecode,
        CodeGenTarget::DXBytecodeAssembly,
        PassThroughMode::Fxc);
    map.addTransition(
        CodeGenTarget::MetalLib,
        CodeGenTarget::MetalLibAssembly,
        PassThroughMode::MetalC);
}

void Session::addBuiltins(char const* sourcePath, char const* source)
{
    auto sourceBlob = StringBlob::moveCreate(String(source));

    // TODO(tfoley): Add ability to directly new builtins to the appropriate scope
    Module* module = nullptr;
    addBuiltinSource(coreLanguageScope, sourcePath, sourceBlob, module);
    if (module)
        coreModules.add(module);
}

void Session::setSharedLibraryLoader(ISlangSharedLibraryLoader* loader)
{
    // External API allows passing of nullptr to reset the loader
    loader = loader ? loader : DefaultSharedLibraryLoader::getSingleton();

    _setSharedLibraryLoader(loader);
}

ISlangSharedLibraryLoader* Session::getSharedLibraryLoader()
{
    return (m_sharedLibraryLoader == DefaultSharedLibraryLoader::getSingleton())
               ? nullptr
               : m_sharedLibraryLoader.get();
}

SlangResult Session::checkCompileTargetSupport(SlangCompileTarget inTarget)
{
    auto target = CodeGenTarget(inTarget);

    const PassThroughMode mode = getDownstreamCompilerRequiredForTarget(target);
    return (mode != PassThroughMode::None) ? checkPassThroughSupport(SlangPassThrough(mode))
                                           : SLANG_OK;
}

SlangResult Session::checkPassThroughSupport(SlangPassThrough inPassThrough)
{
    return checkExternalCompilerSupport(this, PassThroughMode(inPassThrough));
}

void Session::writeCoreModuleDoc(String config)
{
    ASTBuilder* astBuilder = getBuiltinLinkage()->getASTBuilder();
    SourceManager* sourceManager = getBuiltinSourceManager();

    DiagnosticSink sink(sourceManager, Lexer::sourceLocationLexer);

    List<String> docStrings;

    // For all the modules add their doc output to docStrings
    for (Module* m : coreModules)
    {
        RefPtr<ASTMarkup> markup(new ASTMarkup);
        ASTMarkupUtil::extract(m->getModuleDecl(), sourceManager, &sink, markup);

        DocMarkdownWriter writer(markup, astBuilder, &sink);
        auto rootPage = writer.writeAll(config.getUnownedSlice());
        File::writeAllText("toc.html", writer.writeTOC());
        rootPage->writeToDisk();
        rootPage->writeSummary(toSlice("summary.txt"));
    }
    ComPtr<ISlangBlob> diagnosticBlob;
    sink.getBlobIfNeeded(diagnosticBlob.writeRef());
    if (diagnosticBlob && diagnosticBlob->getBufferSize() != 0)
    {
        // Write the diagnostic blob to stdout.
        fprintf(stderr, "%s", (const char*)diagnosticBlob->getBufferPointer());
    }
}

const char* getBuiltinModuleNameStr(slang::BuiltinModuleName name)
{
    const char* result = nullptr;
    switch (name)
    {
    case slang::BuiltinModuleName::Core:
        result = "core";
        break;
    case slang::BuiltinModuleName::GLSL:
        result = "glsl";
        break;
    default:
        SLANG_UNEXPECTED("Unknown builtin module");
    }
    return result;
}

TypeCheckingCache* Session::getTypeCheckingCache()
{
    return static_cast<TypeCheckingCache*>(m_typeCheckingCache.get());
}

Session::BuiltinModuleInfo Session::getBuiltinModuleInfo(slang::BuiltinModuleName name)
{
    Session::BuiltinModuleInfo result;

    result.name = getBuiltinModuleNameStr(name);

    switch (name)
    {
    case slang::BuiltinModuleName::Core:
        result.languageScope = coreLanguageScope;
        break;
    case slang::BuiltinModuleName::GLSL:
        result.name = "glsl";
        result.languageScope = glslLanguageScope;
        break;
    default:
        SLANG_UNEXPECTED("Unknown builtin module");
    }
    return result;
}

SlangResult Session::compileCoreModule(slang::CompileCoreModuleFlags compileFlags)
{
    return compileBuiltinModule(slang::BuiltinModuleName::Core, compileFlags);
}

SlangResult Session::compileBuiltinModule(
    slang::BuiltinModuleName moduleName,
    slang::CompileCoreModuleFlags compileFlags)
{
    SLANG_AST_BUILDER_RAII(m_builtinLinkage->getASTBuilder());

#ifdef _DEBUG
    time_t beginTime = 0;
    if (moduleName == slang::BuiltinModuleName::Core)
    {
        // Print a message in debug builds to notice the user that compiling the core module
        // can take a while.
        time(&beginTime);
        fprintf(stderr, "Compiling core module on debug build, this can take a while.\n");
    }
#endif
    BuiltinModuleInfo builtinModuleInfo = getBuiltinModuleInfo(moduleName);
    auto moduleNameObj = m_builtinLinkage->getNamePool()->getName(builtinModuleInfo.name);
    if (m_builtinLinkage->mapNameToLoadedModules.tryGetValue(moduleNameObj))
    {
        // Already have the builtin module loaded
        return SLANG_FAIL;
    }

    StringBuilder moduleSrcBuilder;
    switch (moduleName)
    {
    case slang::BuiltinModuleName::Core:
        moduleSrcBuilder << (const char*)getCoreLibraryCode()->getBufferPointer()
                         << (const char*)getHLSLLibraryCode()->getBufferPointer()
                         << (const char*)getAutodiffLibraryCode()->getBufferPointer();
        break;
    case slang::BuiltinModuleName::GLSL:
        moduleSrcBuilder << (const char*)getGLSLLibraryCode()->getBufferPointer();
        break;
    }

    // TODO(JS): Could make this return a SlangResult as opposed to exception
    auto moduleSrcBlob = StringBlob::moveCreate(moduleSrcBuilder.produceString());
    Module* compiledModule = nullptr;
    addBuiltinSource(
        builtinModuleInfo.languageScope,
        builtinModuleInfo.name,
        moduleSrcBlob,
        compiledModule);

    if (moduleName == slang::BuiltinModuleName::Core)
    {
        // We need to retain this AST so that we can use it in other code
        // (Note that the `Scope` type does not retain the AST it points to)
        coreModules.add(compiledModule);
    }

    if (compileFlags & slang::CompileCoreModuleFlag::WriteDocumentation)
    {
        // Load config file first.
        String configText;
        if (SLANG_FAILED(File::readAllText("config.txt", configText)))
        {
            fprintf(
                stderr,
                "Error writing documentation: config file not found on current working "
                "directory.\n");
        }
        else
        {
            writeCoreModuleDoc(configText);
        }
    }

    finalizeSharedASTBuilder();

#ifdef _DEBUG
    if (moduleName == slang::BuiltinModuleName::Core)
    {
        time_t endTime;
        time(&endTime);
        fprintf(stderr, "Compiling core module took %.2f seconds.\n", difftime(endTime, beginTime));
    }
#endif
    return SLANG_OK;
}

SlangResult Session::loadCoreModule(const void* coreModule, size_t coreModuleSizeInBytes)
{
    return loadBuiltinModule(slang::BuiltinModuleName::Core, coreModule, coreModuleSizeInBytes);
}

SlangResult Session::loadBuiltinModule(
    slang::BuiltinModuleName moduleName,
    const void* moduleData,
    size_t sizeInBytes)
{
    SLANG_PROFILE;


    SLANG_AST_BUILDER_RAII(m_builtinLinkage->getASTBuilder());

    BuiltinModuleInfo builtinModuleInfo = getBuiltinModuleInfo(moduleName);
    auto nameObj = m_builtinLinkage->getNamePool()->getName(builtinModuleInfo.name);
    if (m_builtinLinkage->mapNameToLoadedModules.containsKey(nameObj))
    {
        // Already have a core module loaded
        return SLANG_FAIL;
    }

    // Make a file system to read it from
    ComPtr<ISlangFileSystemExt> fileSystem;
    SLANG_RETURN_ON_FAIL(loadArchiveFileSystem(moduleData, sizeInBytes, fileSystem));

    // Let's try loading serialized modules and adding them
    Module* module = nullptr;
    SLANG_RETURN_ON_FAIL(_readBuiltinModule(
        fileSystem,
        builtinModuleInfo.languageScope,
        builtinModuleInfo.name,
        module));

    if (moduleName == slang::BuiltinModuleName::Core)
    {
        // We need to retain this AST so that we can use it in other code
        // (Note that the `Scope` type does not retain the AST it points to)
        coreModules.add(module);
    }

    finalizeSharedASTBuilder();
    return SLANG_OK;
}

SlangResult Session::saveCoreModule(SlangArchiveType archiveType, ISlangBlob** outBlob)
{
    return saveBuiltinModule(slang::BuiltinModuleName::Core, archiveType, outBlob);
}

SlangResult Session::saveBuiltinModule(
    slang::BuiltinModuleName moduleTag,
    SlangArchiveType archiveType,
    ISlangBlob** outBlob)
{
    // If no builtin modules have been loaded, then there is
    // nothing to save, and we fail immediately.
    //
    if (m_builtinLinkage->mapNameToLoadedModules.getCount() == 0)
    {
        return SLANG_FAIL;
    }

    // The module will need to be looked up by its name, and
    // will also be serialized out to a path with a matching name.
    //
    BuiltinModuleInfo moduleInfo = getBuiltinModuleInfo(moduleTag);
    const char* moduleName = moduleInfo.name;

    // If we cannot find a loaded module in the linkage with
    // the appropriate name, then for some reason it hasn't
    // been loaded, and we fail.
    //
    RefPtr<Module> module;
    m_builtinLinkage->mapNameToLoadedModules.tryGetValue(
        getNameObj(UnownedStringSlice(moduleName)),
        module);
    if (!module)
    {
        return SLANG_FAIL;
    }

    // AST serialization needs access to an AST builder, so
    // we establish a current builder for the duration of
    // the serialization process.
    //
    SLANG_AST_BUILDER_RAII(m_builtinLinkage->getASTBuilder());

    // The serialized module will be represented as a logical
    // file in an archive, so we create a logical file system
    // to represent that archive.
    //
    ComPtr<ISlangMutableFileSystem> fileSystem;
    SLANG_RETURN_ON_FAIL(createArchiveFileSystem(archiveType, fileSystem));
    //
    // The created file system must support the `IArchiveFileSystem`
    // interface (since we created it with `createArchiveFileSystem`).
    //
    auto archiveFileSystem = as<IArchiveFileSystem>(fileSystem);
    if (!archiveFileSystem)
    {
        return SLANG_FAIL;
    }

    // The output file name that we'll write to in that file system
    // is just the builtin module name with a `.slang-module` suffix.
    //
    StringBuilder moduleFileName;
    moduleFileName << moduleName << ".slang-module";

    // The module serialization step has some options that we need
    // to configure appropriately.
    //
    SerialContainerUtil::WriteOptions options;
    //
    // We want builtin modules to be saved with their source location
    // information.
    //
    options.optionFlags |= SerialOptionFlag::SourceLocation;
    //
    // And in order to work with source locations, the serialization
    // process will also need access to the source manager that
    // can translate locations into their humane format.
    //
    options.sourceManager = m_builtinLinkage->getSourceManager();

    // At this point we can finally delegate down to the next level,
    // which handles the serialization of a Slang module into a
    // byte stream.
    //
    OwnedMemoryStream stream(FileAccess::Write);
    SLANG_RETURN_ON_FAIL(SerialContainerUtil::write(module, options, &stream));
    auto contents = stream.getContents();

    // Once the stream that represents the module has been written, we can
    // write it to a file in the logical file system.
    //
    // TODO(tfoley): why can't the file system let us open the file for output?
    //
    SLANG_RETURN_ON_FAIL(fileSystem->saveFile(
        moduleFileName.getBuffer(),
        contents.getBuffer(),
        contents.getCount()));

    // And finally, we can ask the archive file system to serialize itself
    // out as a blob of bytes, which yields the final serialized representation
    // of the module.
    //
    SLANG_RETURN_ON_FAIL(archiveFileSystem->storeArchive(
        // The `true` here indicates that the blob that gets created should own
        // its content, independent from the file system object itself; otherwise
        // the file system might return a blob that shares storage with itself.
        true,
        outBlob));

    return SLANG_OK;
}

SlangResult Session::_readBuiltinModule(
    ISlangFileSystem* fileSystem,
    Scope* scope,
    String moduleName,
    Module*& outModule)
{
    // Get the name of the module
    StringBuilder moduleFilename;
    moduleFilename << moduleName << ".slang-module";

    RiffContainer riffContainer;
    {
        // Load it
        ComPtr<ISlangBlob> blob;
        SLANG_RETURN_ON_FAIL(fileSystem->loadFile(moduleFilename.getBuffer(), blob.writeRef()));

        // Set up a stream
        MemoryStreamBase stream(FileAccess::Read, blob->getBufferPointer(), blob->getBufferSize());

        // Load the riff container
        SLANG_RETURN_ON_FAIL(RiffUtil::read(&stream, riffContainer));
    }

    Linkage* linkage = getBuiltinLinkage();
    SourceManager* sourceManager = getBuiltinSourceManager();
    NamePool* sessionNamePool = &namePool;

    auto moduleChunk = ModuleChunkRef::find(&riffContainer);
    if (!moduleChunk)
        return SLANG_FAIL;

    SHA1::Digest moduleDigest = moduleChunk.getDigest();

    auto irChunk = moduleChunk.findIR();
    if (!irChunk)
        return SLANG_FAIL;

    auto astChunk = moduleChunk.findAST();
    if (!astChunk)
        return SLANG_FAIL;

    // Source location information is stored as a distinct
    // chunk from the IR and AST, so we need to search for
    // that chunk and then set up the information for use
    // in the IR and AST deserialization (if we find anything).
    //
    RefPtr<SerialSourceLocReader> sourceLocReader;
    if (auto debugChunk = findDebugChunk(moduleChunk.ptr()))
    {
        SLANG_RETURN_ON_FAIL(
            readSourceLocationsFromDebugChunk(debugChunk, sourceManager, sourceLocReader));
    }

    // At this point we create the `Module` object that will
    // represent the builtin module we are reading, although
    // it is still possible that deserialization will fail
    // at one of the following steps.
    //
    auto astBuilder = linkage->getASTBuilder();
    RefPtr<Module> module(new Module(linkage, astBuilder));
    module->setName(moduleName);
    module->setDigest(moduleDigest);


    // Next, we set about deserializing the AST representation
    // of the module.
    //
    auto moduleDecl = readSerializedModuleAST(
        linkage,
        astBuilder,
        nullptr, // no sink
        astChunk,
        sourceLocReader,
        SourceLoc());
    if (!moduleDecl)
    {
        return SLANG_FAIL;
    }
    moduleDecl->module = module;
    module->setModuleDecl(moduleDecl);

    if (isFromCoreModule(moduleDecl))
    {
        registerBuiltinDecls(this, moduleDecl);
    }

    // After the AST module has been read in, we next look
    // to deserialize the IR module.
    //
    RefPtr<IRModule> irModule;
    SLANG_RETURN_ON_FAIL(decodeModuleIR(irModule, irChunk, this, sourceLocReader));

    irModule->setName(module->getNameObj());
    module->setIRModule(irModule);

    // Put in the loaded module map
    linkage->mapNameToLoadedModules.add(sessionNamePool->getName(moduleName), module);


    // Add the resulting code to the appropriate scope
    if (!scope->containerDecl)
    {
        // We are the first chunk of code to be loaded for this scope
        scope->containerDecl = moduleDecl;
    }
    else
    {
        // We need to create a new scope to link into the whole thing
        auto subScope = linkage->getASTBuilder()->create<Scope>();
        subScope->containerDecl = moduleDecl;
        subScope->nextSibling = scope->nextSibling;
        scope->nextSibling = subScope;
    }

    outModule = module.get();

    return SLANG_OK;
}

SLANG_NO_THROW SlangResult SLANG_MCALL
Session::queryInterface(SlangUUID const& uuid, void** outObject)
{
    if (uuid == Session::getTypeGuid())
    {
        addReference();
        *outObject = static_cast<Session*>(this);
        return SLANG_OK;
    }

    if (uuid == ISlangUnknown::getTypeGuid() || uuid == IGlobalSession::getTypeGuid())
    {
        addReference();
        *outObject = static_cast<slang::IGlobalSession*>(this);
        return SLANG_OK;
    }

    return SLANG_E_NO_INTERFACE;
}

static size_t _getStructureSize(const uint8_t* src)
{
    size_t size = 0;
    ::memcpy(&size, src, sizeof(size_t));
    return size;
}

template<typename T>
static T makeFromSizeVersioned(const uint8_t* src)
{
    // The structure size must be size_t
    SLANG_COMPILE_TIME_ASSERT(sizeof(((T*)src)->structureSize) == sizeof(size_t));

    // The structureSize field *must* be the first element of T
    // Ideally would use SLANG_COMPILE_TIME_ASSERT, but that doesn't work on gcc.
    // Can't just assert, because determined to be a constant expression
    {
        auto offset = SLANG_OFFSET_OF(T, structureSize);
        SLANG_ASSERT(offset == 0);
        // Needed because offset is only 'used' by an assert
        SLANG_UNUSED(offset);
    }

    // The source size is held in the first element of T, and will be in the first bytes of src.
    const size_t srcSize = _getStructureSize(src);
    const size_t dstSize = sizeof(T);

    // If they are the same size, and appropriate alignment we can just cast and return
    if (srcSize == dstSize && (size_t(src) & (SLANG_ALIGN_OF(T) - 1)) == 0)
    {
        return *(const T*)src;
    }

    // Assumes T can default constructed sensibly
    T dst;

    // It's structure size should be setup and should be dstSize
    SLANG_ASSERT(dst.structureSize == dstSize);

    // The size to copy is the minimum on the two sizes
    const auto copySize = std::min(srcSize, dstSize);
    ::memcpy(&dst, src, copySize);

    // The final struct size is the destination size
    dst.structureSize = dstSize;

    return dst;
}

SLANG_NO_THROW SlangResult SLANG_MCALL
Session::createSession(slang::SessionDesc const& inDesc, slang::ISession** outSession)
{
    RefPtr<ASTBuilder> astBuilder(new ASTBuilder(m_sharedASTBuilder, "Session::astBuilder"));
    slang::SessionDesc desc = makeFromSizeVersioned<slang::SessionDesc>((uint8_t*)&inDesc);

    RefPtr<Linkage> linkage = new Linkage(this, astBuilder, getBuiltinLinkage());

    if (desc.skipSPIRVValidation)
    {
        linkage->m_optionSet.set(CompilerOptionName::SkipSPIRVValidation, true);
    }

    {
        std::lock_guard<std::mutex> lock(m_typeCheckingCacheMutex);
        if (m_typeCheckingCache)
            linkage->m_typeCheckingCache =
                new TypeCheckingCache(*static_cast<TypeCheckingCache*>(m_typeCheckingCache.get()));
    }

    Int searchPathCount = desc.searchPathCount;
    for (Int ii = 0; ii < searchPathCount; ++ii)
    {
        linkage->addSearchPath(desc.searchPaths[ii]);
    }

    Int macroCount = desc.preprocessorMacroCount;
    for (Int ii = 0; ii < macroCount; ++ii)
    {
        auto& macro = desc.preprocessorMacros[ii];
        linkage->addPreprocessorDefine(macro.name, macro.value);
    }

    if (desc.fileSystem)
    {
        linkage->setFileSystem(desc.fileSystem);
    }

    if (desc.structureSize >= offsetof(slang::SessionDesc, enableEffectAnnotations))
    {
        linkage->m_optionSet.set(
            CompilerOptionName::EnableEffectAnnotations,
            desc.enableEffectAnnotations);
    }

    linkage->m_optionSet.load(desc.compilerOptionEntryCount, desc.compilerOptionEntries);

    if (!linkage->m_optionSet.hasOption(CompilerOptionName::MatrixLayoutColumn) &&
        !linkage->m_optionSet.hasOption(CompilerOptionName::MatrixLayoutRow))
        linkage->setMatrixLayoutMode(desc.defaultMatrixLayoutMode);

    {
        const Int targetCount = desc.targetCount;
        const uint8_t* targetDescPtr = reinterpret_cast<const uint8_t*>(desc.targets);
        for (Int ii = 0; ii < targetCount; ++ii, targetDescPtr += _getStructureSize(targetDescPtr))
        {
            const auto targetDesc = makeFromSizeVersioned<slang::TargetDesc>(targetDescPtr);
            linkage->addTarget(targetDesc);
        }
    }

    // If any target requires debug info, then we will need to enable debug info when lowering to
    // target-agnostic IR. The target-agnostic IR will only include debug info if the linkage IR
    // options specify that it should, so make sure the linkage debug info level is greater than or
    // equal to that of any target.
    DebugInfoLevel linkageDebugInfoLevel = linkage->m_optionSet.getDebugInfoLevel();
    for (auto target : linkage->targets)
        linkageDebugInfoLevel =
            Math::Max(linkageDebugInfoLevel, target->getOptionSet().getDebugInfoLevel());
    linkage->m_optionSet.set(CompilerOptionName::DebugInformation, linkageDebugInfoLevel);

    // Add any referenced modules to the linkage
    for (auto& option : linkage->m_optionSet.options)
    {
        if (option.key != CompilerOptionName::ReferenceModule)
            continue;
        for (auto& path : option.value)
        {
            DiagnosticSink sink;
            ComPtr<IArtifact> artifact;
            SlangResult result = createArtifactFromReferencedModule(
                path.stringValue,
                SourceLoc{},
                &sink,
                artifact.writeRef());
            if (SLANG_FAILED(result))
            {
                sink.diagnose(SourceLoc{}, Diagnostics::unableToReadFile, path.stringValue);
                return result;
            }
            linkage->m_libModules.add(artifact);
        }
    }

    *outSession = asExternal(linkage.detach());
    return SLANG_OK;
}

SLANG_NO_THROW SlangResult SLANG_MCALL
Session::createCompileRequest(slang::ICompileRequest** outCompileRequest)
{
    auto req = new EndToEndCompileRequest(this);

    // Give it a ref (for output)
    req->addRef();
    // Check it is what we think it should be
    SLANG_ASSERT(req->debugGetReferenceCount() == 1);

    *outCompileRequest = req;
    return SLANG_OK;
}

SLANG_NO_THROW SlangProfileID SLANG_MCALL Session::findProfile(char const* name)
{
    return SlangProfileID(Slang::Profile::lookUp(name).raw);
}

SLANG_NO_THROW SlangCapabilityID SLANG_MCALL Session::findCapability(char const* name)
{
    return SlangCapabilityID(Slang::findCapabilityName(UnownedTerminatedStringSlice(name)));
}

SLANG_NO_THROW void SLANG_MCALL
Session::setDownstreamCompilerPath(SlangPassThrough inPassThrough, char const* path)
{
    PassThroughMode passThrough = PassThroughMode(inPassThrough);
    SLANG_ASSERT(
        int(passThrough) > int(PassThroughMode::None) &&
        int(passThrough) < int(PassThroughMode::CountOf));

    if (m_downstreamCompilerPaths[int(passThrough)] != path)
    {
        // Make access redetermine compiler
        resetDownstreamCompiler(passThrough);
        // Set the path
        m_downstreamCompilerPaths[int(passThrough)] = path;
    }
}

SLANG_NO_THROW void SLANG_MCALL
Session::setDownstreamCompilerPrelude(SlangPassThrough inPassThrough, char const* prelude)
{
    PassThroughMode downstreamCompiler = PassThroughMode(inPassThrough);
    SLANG_ASSERT(
        int(downstreamCompiler) > int(PassThroughMode::None) &&
        int(downstreamCompiler) < int(PassThroughMode::CountOf));
    const SourceLanguage sourceLanguage =
        getDefaultSourceLanguageForDownstreamCompiler(downstreamCompiler);
    setLanguagePrelude(SlangSourceLanguage(sourceLanguage), prelude);
}

SLANG_NO_THROW void SLANG_MCALL
Session::getDownstreamCompilerPrelude(SlangPassThrough inPassThrough, ISlangBlob** outPrelude)
{
    PassThroughMode downstreamCompiler = PassThroughMode(inPassThrough);
    SLANG_ASSERT(
        int(downstreamCompiler) > int(PassThroughMode::None) &&
        int(downstreamCompiler) < int(PassThroughMode::CountOf));
    const SourceLanguage sourceLanguage =
        getDefaultSourceLanguageForDownstreamCompiler(downstreamCompiler);
    getLanguagePrelude(SlangSourceLanguage(sourceLanguage), outPrelude);
}

SLANG_NO_THROW void SLANG_MCALL
Session::setLanguagePrelude(SlangSourceLanguage inSourceLanguage, char const* prelude)
{
    SourceLanguage sourceLanguage = SourceLanguage(inSourceLanguage);
    SLANG_ASSERT(
        int(sourceLanguage) > int(SourceLanguage::Unknown) &&
        int(sourceLanguage) < int(SourceLanguage::CountOf));

    SLANG_ASSERT(sourceLanguage != SourceLanguage::Unknown);

    if (sourceLanguage != SourceLanguage::Unknown)
    {
        m_languagePreludes[int(sourceLanguage)] = prelude;
    }
}

SLANG_NO_THROW void SLANG_MCALL
Session::getLanguagePrelude(SlangSourceLanguage inSourceLanguage, ISlangBlob** outPrelude)
{
    SourceLanguage sourceLanguage = SourceLanguage(inSourceLanguage);

    *outPrelude = nullptr;
    if (sourceLanguage != SourceLanguage::Unknown)
    {
        SLANG_ASSERT(
            int(sourceLanguage) > int(SourceLanguage::Unknown) &&
            int(sourceLanguage) < int(SourceLanguage::CountOf));
        *outPrelude =
            Slang::StringUtil::createStringBlob(m_languagePreludes[int(sourceLanguage)]).detach();
    }
}

SLANG_NO_THROW const char* SLANG_MCALL Session::getBuildTagString()
{
    return ::Slang::getBuildTagString();
}

SLANG_NO_THROW SlangResult SLANG_MCALL Session::setDefaultDownstreamCompiler(
    SlangSourceLanguage sourceLanguage,
    SlangPassThrough defaultCompiler)
{
    if (DownstreamCompilerInfo::canCompile(defaultCompiler, sourceLanguage))
    {
        m_defaultDownstreamCompilers[int(sourceLanguage)] = PassThroughMode(defaultCompiler);
        return SLANG_OK;
    }
    return SLANG_FAIL;
}

SlangPassThrough SLANG_MCALL
Session::getDefaultDownstreamCompiler(SlangSourceLanguage inSourceLanguage)
{
    SLANG_ASSERT(inSourceLanguage >= 0 && inSourceLanguage < SLANG_SOURCE_LANGUAGE_COUNT_OF);
    auto sourceLanguage = SourceLanguage(inSourceLanguage);
    return SlangPassThrough(m_defaultDownstreamCompilers[int(sourceLanguage)]);
}

void Session::setDownstreamCompilerForTransition(
    SlangCompileTarget source,
    SlangCompileTarget target,
    SlangPassThrough compiler)
{
    if (compiler == SLANG_PASS_THROUGH_NONE)
    {
        // Removing the transition means a default can be used
        m_codeGenTransitionMap.removeTransition(CodeGenTarget(source), CodeGenTarget(target));
    }
    else
    {
        m_codeGenTransitionMap.addTransition(
            CodeGenTarget(source),
            CodeGenTarget(target),
            PassThroughMode(compiler));
    }
}

SlangPassThrough Session::getDownstreamCompilerForTransition(
    SlangCompileTarget inSource,
    SlangCompileTarget inTarget)
{
    const CodeGenTarget source = CodeGenTarget(inSource);
    const CodeGenTarget target = CodeGenTarget(inTarget);

    if (m_codeGenTransitionMap.hasTransition(source, target))
    {
        return (SlangPassThrough)m_codeGenTransitionMap.getTransition(source, target);
    }

    const auto desc = ArtifactDescUtil::makeDescForCompileTarget(inTarget);

    // Special case host-callable
    if ((desc.kind == ArtifactKind::HostCallable) &&
        (source == CodeGenTarget::CSource || source == CodeGenTarget::CPPSource))
    {
        // We prefer LLVM if it's available
        if (const auto llvm = getOrLoadDownstreamCompiler(PassThroughMode::LLVM, nullptr))
        {
            return SLANG_PASS_THROUGH_LLVM;
        }
    }

    // Use the legacy 'sourceLanguage' default mechanism.
    // This says nothing about the target type, so it is *assumed* the target type is possible
    // If not it will fail when trying to compile to an unknown target
    const SourceLanguage sourceLanguage =
        (SourceLanguage)TypeConvertUtil::getSourceLanguageFromTarget(inSource);
    if (sourceLanguage != SourceLanguage::Unknown)
    {
        return getDefaultDownstreamCompiler(SlangSourceLanguage(sourceLanguage));
    }

    // Unknwon
    return SLANG_PASS_THROUGH_NONE;
}

IDownstreamCompiler* Session::getDownstreamCompiler(CodeGenTarget source, CodeGenTarget target)
{
    PassThroughMode compilerType = (PassThroughMode)getDownstreamCompilerForTransition(
        SlangCompileTarget(source),
        SlangCompileTarget(target));
    return getOrLoadDownstreamCompiler(compilerType, nullptr);
}

SLANG_NO_THROW SlangResult SLANG_MCALL Session::setSPIRVCoreGrammar(char const* jsonPath)
{
    if (!jsonPath)
    {
        spirvCoreGrammarInfo = SPIRVCoreGrammarInfo::getEmbeddedVersion();
        SLANG_ASSERT(spirvCoreGrammarInfo);
    }
    else
    {
        SourceManager* sourceManager = getBuiltinSourceManager();
        SLANG_ASSERT(sourceManager);
        DiagnosticSink sink(sourceManager, Lexer::sourceLocationLexer);

        String contents;
        const auto readRes = File::readAllText(jsonPath, contents);
        if (SLANG_FAILED(readRes))
        {
            sink.diagnose(SourceLoc{}, Diagnostics::unableToReadFile, jsonPath);
            return readRes;
        }
        const auto pathInfo = PathInfo::makeFromString(jsonPath);
        const auto sourceFile = sourceManager->createSourceFileWithString(pathInfo, contents);
        const auto sourceView = sourceManager->createSourceView(sourceFile, nullptr, SourceLoc());
        spirvCoreGrammarInfo = SPIRVCoreGrammarInfo::loadFromJSON(*sourceView, sink);
    }
    return spirvCoreGrammarInfo ? SLANG_OK : SLANG_FAIL;
}

struct ParsedCommandLineData : public ISlangUnknown, public ComObject
{
    SLANG_COM_OBJECT_IUNKNOWN_ALL

    ISlangUnknown* getInterface(const Slang::Guid& guid)
    {
        if (guid == ISlangUnknown::getTypeGuid())
            return this;
        return nullptr;
    }
    List<SerializedOptionsData> options;
    List<slang::TargetDesc> targets;
};

SLANG_NO_THROW SlangResult SLANG_MCALL Session::parseCommandLineArguments(
    int argc,
    const char* const* argv,
    slang::SessionDesc* outDesc,
    ISlangUnknown** outAllocation)
{
    if (outDesc->structureSize < sizeof(slang::SessionDesc))
        return SLANG_E_BUFFER_TOO_SMALL;
    RefPtr<ParsedCommandLineData> outData = new ParsedCommandLineData();
    RefPtr<EndToEndCompileRequest> tempReq = new EndToEndCompileRequest(this);
    tempReq->processCommandLineArguments(argv, argc);
    outData->options.setCount(1 + tempReq->getLinkage()->targets.getCount());
    int optionDataIndex = 0;
    SerializedOptionsData& optionData = outData->options[optionDataIndex];
    optionDataIndex++;
    tempReq->getOptionSet().serialize(&optionData);
    tempReq->m_optionSetForDefaultTarget.serialize(&optionData);
    for (auto target : tempReq->getLinkage()->targets)
    {
        slang::TargetDesc tdesc;
        SerializedOptionsData& targetOptionData = outData->options[optionDataIndex];
        optionDataIndex++;
        tempReq->getTargetOptionSet(target).serialize(&targetOptionData);
        tdesc.compilerOptionEntryCount = (uint32_t)targetOptionData.entries.getCount();
        tdesc.compilerOptionEntries = targetOptionData.entries.getBuffer();
        outData->targets.add(tdesc);
    }
    outDesc->compilerOptionEntryCount = (uint32_t)optionData.entries.getCount();
    outDesc->compilerOptionEntries = optionData.entries.getBuffer();
    outDesc->targetCount = outData->targets.getCount();
    outDesc->targets = outData->targets.getBuffer();
    *outAllocation = outData.get();
    outData->addRef();
    return SLANG_OK;
}

SLANG_NO_THROW SlangResult SLANG_MCALL
Session::getSessionDescDigest(slang::SessionDesc* sessionDesc, ISlangBlob** outBlob)
{
    ComPtr<slang::ISession> tempSession;
    createSession(*sessionDesc, tempSession.writeRef());
    auto linkage = static_cast<Linkage*>(tempSession.get());
    DigestBuilder<SHA1> digestBuilder;
    linkage->buildHash(digestBuilder, -1);
    auto blob = digestBuilder.finalize().toBlob();
    *outBlob = blob.detach();
    return SLANG_OK;
}

Profile getEffectiveProfile(EntryPoint* entryPoint, TargetRequest* target)
{
    auto entryPointProfile = entryPoint->getProfile();
    auto targetProfile = target->getOptionSet().getProfile();

    // Depending on the target *format* we might have to restrict the
    // profile family to one that makes sense.
    //
    // TODO: Some of this should really be handled as validation at
    // the front-end. People shouldn't be allowed to ask for SPIR-V
    // output with Shader Model 5.0...
    switch (target->getTarget())
    {
    default:
        break;

    case CodeGenTarget::GLSL:
    case CodeGenTarget::SPIRV:
    case CodeGenTarget::SPIRVAssembly:
        if (targetProfile.getFamily() != ProfileFamily::GLSL)
        {
            targetProfile.setVersion(ProfileVersion::GLSL_150);
        }
        break;

    case CodeGenTarget::HLSL:
    case CodeGenTarget::DXBytecode:
    case CodeGenTarget::DXBytecodeAssembly:
    case CodeGenTarget::DXIL:
    case CodeGenTarget::DXILAssembly:
        if (targetProfile.getFamily() != ProfileFamily::DX)
        {
            targetProfile.setVersion(ProfileVersion::DX_5_1);
        }
        break;
    case CodeGenTarget::Metal:
    case CodeGenTarget::MetalLib:
    case CodeGenTarget::MetalLibAssembly:
        if (targetProfile.getFamily() != ProfileFamily::METAL)
        {
            targetProfile.setVersion(ProfileVersion::METAL_2_3);
        }
        break;
    }

    auto entryPointProfileVersion = entryPointProfile.getVersion();
    auto targetProfileVersion = targetProfile.getVersion();

    // Default to the entry point profile, since we know that has the right stage.
    Profile effectiveProfile = entryPointProfile;

    // Ignore the input from the target profile if it is missing.
    if (targetProfile.getFamily() != ProfileFamily::Unknown)
    {
        // If the target comes from a different profile family, *or* it is from
        // the same family but has a greater version number, then use the target's version.
        if (targetProfile.getFamily() != entryPointProfile.getFamily() ||
            (targetProfileVersion > entryPointProfileVersion))
        {
            effectiveProfile.setVersion(targetProfileVersion);
        }
    }

    // Now consider the possibility that the chosen stage might force an "upgrade"
    // to the profile level.
    ProfileVersion stageMinVersion = ProfileVersion::Unknown;
    switch (effectiveProfile.getFamily())
    {
    case ProfileFamily::DX:
        switch (effectiveProfile.getStage())
        {
        default:
            break;

        case Stage::RayGeneration:
        case Stage::Intersection:
        case Stage::ClosestHit:
        case Stage::AnyHit:
        case Stage::Miss:
        case Stage::Callable:
            // The DirectX ray tracing stages implicitly
            // require Shader Model 6.3 or later.
            //
            stageMinVersion = ProfileVersion::DX_6_3;
            break;

            //  TODO: Add equivalent logic for geometry, tessellation, and compute stages.
        }
        break;

    case ProfileFamily::GLSL:
        switch (effectiveProfile.getStage())
        {
        default:
            break;

        case Stage::RayGeneration:
        case Stage::Intersection:
        case Stage::ClosestHit:
        case Stage::AnyHit:
        case Stage::Miss:
        case Stage::Callable:
            stageMinVersion = ProfileVersion::GLSL_460;
            break;

            //  TODO: Add equivalent logic for geometry, tessellation, and compute stages.
        }
        break;

    default:
        break;
    }

    if (stageMinVersion > effectiveProfile.getVersion())
    {
        effectiveProfile.setVersion(stageMinVersion);
    }

    return effectiveProfile;
}


//

Linkage::Linkage(Session* session, ASTBuilder* astBuilder, Linkage* builtinLinkage)
    : m_session(session)
    , m_retainedSession(session)
    , m_sourceManager(&m_defaultSourceManager)
    , m_astBuilder(astBuilder)
    , m_cmdLineContext(new CommandLineContext())
{
    getNamePool()->setRootNamePool(session->getRootNamePool());

    m_defaultSourceManager.initialize(session->getBuiltinSourceManager(), nullptr);

    setFileSystem(nullptr);

    // Copy of the built in linkages modules
    if (builtinLinkage)
    {
        for (const auto& nameToMod : builtinLinkage->mapNameToLoadedModules)
            mapNameToLoadedModules.add(nameToMod);
    }

    m_semanticsForReflection = new SharedSemanticsContext(this, nullptr, nullptr);
}

SharedSemanticsContext* Linkage::getSemanticsForReflection()
{
    return m_semanticsForReflection.get();
}

ISlangUnknown* Linkage::getInterface(const Guid& guid)
{
    if (guid == ISlangUnknown::getTypeGuid() || guid == ISession::getTypeGuid())
        return asExternal(this);

    return nullptr;
}

Linkage::~Linkage()
{
    // Upstream type checking cache.
    if (m_typeCheckingCache)
    {
        auto globalSession = getSessionImpl();
        std::lock_guard<std::mutex> lock(globalSession->m_typeCheckingCacheMutex);
        if (!globalSession->m_typeCheckingCache ||
            globalSession->getTypeCheckingCache()->resolvedOperatorOverloadCache.getCount() <
                getTypeCheckingCache()->resolvedOperatorOverloadCache.getCount())
        {
            globalSession->m_typeCheckingCache = m_typeCheckingCache;
            getTypeCheckingCache()->version++;
        }
        destroyTypeCheckingCache();
    }
}

SearchDirectoryList& Linkage::getSearchDirectories()
{
    auto list = m_optionSet.getArray(CompilerOptionName::Include);
    if (list.getCount() != searchDirectoryCache.searchDirectories.getCount())
    {
        searchDirectoryCache.searchDirectories.clear();
        for (auto dir : list)
            searchDirectoryCache.searchDirectories.add(SearchDirectory(dir.stringValue));
    }
    return searchDirectoryCache;
}

TypeCheckingCache* Linkage::getTypeCheckingCache()
{
    if (!m_typeCheckingCache)
    {
        m_typeCheckingCache = new TypeCheckingCache();
    }
    return static_cast<TypeCheckingCache*>(m_typeCheckingCache.get());
}

void Linkage::destroyTypeCheckingCache()
{
    m_typeCheckingCache = nullptr;
}

SLANG_NO_THROW slang::IGlobalSession* SLANG_MCALL Linkage::getGlobalSession()
{
    return asExternal(getSessionImpl());
}

void Linkage::addTarget(slang::TargetDesc const& desc)
{
    SLANG_AST_BUILDER_RAII(getASTBuilder());

    auto targetIndex = addTarget(CodeGenTarget(desc.format));
    auto target = targets[targetIndex];

    auto& optionSet = target->getOptionSet();
    optionSet.inheritFrom(m_optionSet);

    optionSet.set(CompilerOptionName::FloatingPointMode, FloatingPointMode(desc.floatingPointMode));
    optionSet.addTargetFlags(desc.flags);
    optionSet.setProfile(Profile(desc.profile));
    optionSet.set(CompilerOptionName::LineDirectiveMode, LineDirectiveMode(desc.lineDirectiveMode));
    optionSet.set(CompilerOptionName::GLSLForceScalarLayout, desc.forceGLSLScalarBufferLayout);

    CompilerOptionSet targetOptions;
    targetOptions.load(desc.compilerOptionEntryCount, desc.compilerOptionEntries);
    optionSet.overrideWith(targetOptions);
}

#if 0
    SLANG_NO_THROW SlangInt SLANG_MCALL Linkage::getTargetCount()
    {
        return targets.getCount();
    }

    SLANG_NO_THROW slang::ITarget* SLANG_MCALL Linkage::getTargetByIndex(SlangInt index)
    {
        if (index < 0) return nullptr;
        if (index >= targets.getCount()) return nullptr;
        return asExternal(targets[index]);
    }
#endif

static void outputExceptionDiagnostic(
    const AbortCompilationException& exception,
    DiagnosticSink& sink,
    slang::IBlob** outDiagnostics)
{
    sink.diagnoseRaw(Severity::Error, exception.Message.getUnownedSlice());
    sink.getBlobIfNeeded(outDiagnostics);
}

static void outputExceptionDiagnostic(
    const Exception& exception,
    DiagnosticSink& sink,
    slang::IBlob** outDiagnostics)
{
    sink.diagnoseRaw(Severity::Internal, exception.Message.getUnownedSlice());
    sink.getBlobIfNeeded(outDiagnostics);
}

static void outputExceptionDiagnostic(DiagnosticSink& sink, slang::IBlob** outDiagnostics)
{
    sink.diagnoseRaw(Severity::Fatal, "An unknown exception occurred");
    sink.getBlobIfNeeded(outDiagnostics);
}

SLANG_NO_THROW slang::IModule* SLANG_MCALL
Linkage::loadModule(const char* moduleName, slang::IBlob** outDiagnostics)
{
    SLANG_AST_BUILDER_RAII(getASTBuilder());

    DiagnosticSink sink(getSourceManager(), Lexer::sourceLocationLexer);
    applySettingsToDiagnosticSink(&sink, &sink, m_optionSet);

    if (isInLanguageServer())
    {
        sink.setFlags(DiagnosticSink::Flag::HumaneLoc | DiagnosticSink::Flag::LanguageServer);
    }

    try
    {
        auto name = getNamePool()->getName(moduleName);

        auto module = findOrImportModule(name, SourceLoc(), &sink);
        sink.getBlobIfNeeded(outDiagnostics);

        return asExternal(module);
    }
    catch (const AbortCompilationException& e)
    {
        outputExceptionDiagnostic(e, sink, outDiagnostics);
        return nullptr;
    }
    catch (const Exception& e)
    {
        outputExceptionDiagnostic(e, sink, outDiagnostics);
        return nullptr;
    }
    catch (...)
    {
        outputExceptionDiagnostic(sink, outDiagnostics);
        return nullptr;
    }
}

slang::IModule* Linkage::loadModuleFromBlob(
    const char* moduleName,
    const char* path,
    slang::IBlob* source,
    ModuleBlobType blobType,
    slang::IBlob** outDiagnostics)
{
    SLANG_AST_BUILDER_RAII(getASTBuilder());

    DiagnosticSink sink(getSourceManager(), Lexer::sourceLocationLexer);
    applySettingsToDiagnosticSink(&sink, &sink, m_optionSet);

    if (isInLanguageServer())
    {
        sink.setFlags(DiagnosticSink::Flag::HumaneLoc | DiagnosticSink::Flag::LanguageServer);
    }


    try
    {
        auto getDigestStr = [](auto x)
        {
            DigestBuilder<SHA1> digestBuilder;
            digestBuilder.append(x);
            return digestBuilder.finalize().toString();
        };

        String moduleNameStr = moduleName;
        if (!moduleName)
            moduleNameStr = getDigestStr(source);

        auto name = getNamePool()->getName(moduleNameStr);
        RefPtr<LoadedModule> loadedModule;
        if (mapNameToLoadedModules.tryGetValue(name, loadedModule))
        {
            return loadedModule;
        }
        String pathStr = path;
        if (pathStr.getLength() == 0)
        {
            // If path is empty, use a digest from source as path.
            pathStr = getDigestStr(source);
        }
        auto pathInfo = PathInfo::makeFromString(pathStr);
        if (File::exists(pathStr))
        {
            String cannonicalPath;
            if (SLANG_SUCCEEDED(Path::getCanonical(pathStr, cannonicalPath)))
            {
                pathInfo = PathInfo::makeNormal(pathStr, cannonicalPath);
            }
        }
        RefPtr<Module> module =
            loadModuleImpl(name, pathInfo, source, SourceLoc(), &sink, nullptr, blobType);
        sink.getBlobIfNeeded(outDiagnostics);
        return asExternal(module.detach());
    }
    catch (const AbortCompilationException& e)
    {
        outputExceptionDiagnostic(e, sink, outDiagnostics);
        return nullptr;
    }
    catch (const Exception& e)
    {
        outputExceptionDiagnostic(e, sink, outDiagnostics);
        return nullptr;
    }
    catch (...)
    {
        outputExceptionDiagnostic(sink, outDiagnostics);
        return nullptr;
    }
}

SLANG_NO_THROW slang::IModule* SLANG_MCALL Linkage::loadModuleFromSource(
    const char* moduleName,
    const char* path,
    slang::IBlob* source,
    slang::IBlob** outDiagnostics)
{
    return loadModuleFromBlob(moduleName, path, source, ModuleBlobType::Source, outDiagnostics);
}

SLANG_NO_THROW slang::IModule* SLANG_MCALL Linkage::loadModuleFromSourceString(
    const char* moduleName,
    const char* path,
    const char* source,
    slang::IBlob** outDiagnostics)
{
    auto sourceBlob = StringBlob::create(UnownedStringSlice(source));
    return loadModuleFromSource(moduleName, path, sourceBlob.get(), outDiagnostics);
}

SLANG_NO_THROW slang::IModule* SLANG_MCALL Linkage::loadModuleFromIRBlob(
    const char* moduleName,
    const char* path,
    slang::IBlob* source,
    slang::IBlob** outDiagnostics)
{
    return loadModuleFromBlob(moduleName, path, source, ModuleBlobType::IR, outDiagnostics);
}

SLANG_NO_THROW SlangResult SLANG_MCALL Linkage::createCompositeComponentType(
    slang::IComponentType* const* componentTypes,
    SlangInt componentTypeCount,
    slang::IComponentType** outCompositeComponentType,
    ISlangBlob** outDiagnostics)
{
    if (outCompositeComponentType == nullptr)
        return SLANG_E_INVALID_ARG;

    SLANG_AST_BUILDER_RAII(getASTBuilder());

    // Attempting to create a "composite" of just one component type should
    // just return the component type itself, to avoid redundant work.
    //
    if (componentTypeCount == 1)
    {
        auto componentType = componentTypes[0];
        componentType->addRef();
        *outCompositeComponentType = componentType;
        return SLANG_OK;
    }

    DiagnosticSink sink(getSourceManager(), Lexer::sourceLocationLexer);
    applySettingsToDiagnosticSink(&sink, &sink, m_optionSet);

    List<RefPtr<ComponentType>> childComponents;
    for (Int cc = 0; cc < componentTypeCount; ++cc)
    {
        childComponents.add(asInternal(componentTypes[cc]));
    }

    RefPtr<ComponentType> composite = CompositeComponentType::create(this, childComponents);

    sink.getBlobIfNeeded(outDiagnostics);

    *outCompositeComponentType = asExternal(composite.detach());
    return SLANG_OK;
}

SLANG_NO_THROW slang::TypeReflection* SLANG_MCALL Linkage::specializeType(
    slang::TypeReflection* inUnspecializedType,
    slang::SpecializationArg const* specializationArgs,
    SlangInt specializationArgCount,
    ISlangBlob** outDiagnostics)
{
    SLANG_AST_BUILDER_RAII(getASTBuilder());

    auto unspecializedType = asInternal(inUnspecializedType);

    List<Type*> typeArgs;

    for (Int ii = 0; ii < specializationArgCount; ++ii)
    {
        auto& arg = specializationArgs[ii];
        if (arg.kind != slang::SpecializationArg::Kind::Type)
            return nullptr;

        typeArgs.add(asInternal(arg.type));
    }

    DiagnosticSink sink(getSourceManager(), Lexer::sourceLocationLexer);
    auto specializedType =
        specializeType(unspecializedType, typeArgs.getCount(), typeArgs.getBuffer(), &sink);
    sink.getBlobIfNeeded(outDiagnostics);

    return asExternal(specializedType);
}

DeclRef<GenericDecl> getGenericParentDeclRef(
    ASTBuilder* astBuilder,
    SemanticsVisitor* visitor,
    DeclRef<Decl> declRef)
{
    // Create substituted parent decl ref.
    auto decl = declRef.getDecl();

    while (decl && !as<GenericDecl>(decl))
    {
        decl = decl->parentDecl;
    }

    if (!decl)
    {
        // No generic parent
        return DeclRef<GenericDecl>();
    }

    auto genericDecl = as<GenericDecl>(decl);
    auto genericDeclRef =
        createDefaultSubstitutionsIfNeeded(astBuilder, visitor, DeclRef(genericDecl))
            .as<GenericDecl>();
    return substituteDeclRef(SubstitutionSet(declRef), astBuilder, genericDeclRef)
        .as<GenericDecl>();
}

bool Linkage::isSpecialized(DeclRef<Decl> declRef)
{
    // For now, we only support two 'states': fully applied or not at all.
    // If we add support for partial specialization, we will need to update this logic.
    //
    // If it's not specialized, then declRef will be the one with default substitutions.
    //
    SemanticsVisitor visitor(getSemanticsForReflection());

    auto decl = declRef.getDecl();
    while (decl && !as<GenericDecl>(decl))
    {
        decl = decl->parentDecl;
    }

    if (!decl)
        return true; // no generics => always specialized

    auto defaultArgs = getDefaultSubstitutionArgs(getASTBuilder(), &visitor, as<GenericDecl>(decl));
    auto currentArgs =
        SubstitutionSet(declRef).findGenericAppDeclRef(as<GenericDecl>(decl))->getArgs();

    if (defaultArgs.getCount() != currentArgs.getCount()) // should really never happen.
        return true;

    for (Index i = 0; i < defaultArgs.getCount(); ++i)
    {
        if (defaultArgs[i] != currentArgs[i])
            return true;
    }

    return false;
}

bool isFuncGeneric(DeclRef<Decl> declRef)
{
    if (auto funcDecl = as<FuncDecl>(declRef.getDecl()))
    {
        if (funcDecl->parentDecl && as<GenericDecl>(funcDecl->parentDecl))
        {
            return true;
        }
    }

    return false;
}

DeclRef<Decl> Linkage::specializeWithArgTypes(
    Expr* funcExpr,
    List<Type*> argTypes,
    DiagnosticSink* sink)
{
    SemanticsVisitor visitor(getSemanticsForReflection());
    SemanticsVisitor::ExprLocalScope scope;
    visitor = visitor.withSink(sink).withExprLocalScope(&scope);

    SLANG_AST_BUILDER_RAII(getASTBuilder());

    if (auto declRefFuncExpr = as<DeclRefExpr>(funcExpr))
    {
        if (isFuncGeneric(declRefFuncExpr->declRef) && !isSpecialized(declRefFuncExpr->declRef))
        {
            if (auto genericDeclRef = getGenericParentDeclRef(
                    getCurrentASTBuilder(),
                    &visitor,
                    declRefFuncExpr->declRef))
            {
                auto genericDeclRefExpr = getCurrentASTBuilder()->create<DeclRefExpr>();
                genericDeclRefExpr->declRef = genericDeclRef;
                funcExpr = genericDeclRefExpr;
            }
        }
    }

    List<Expr*> argExprs;
    for (SlangInt aa = 0; aa < argTypes.getCount(); ++aa)
    {
        auto argType = argTypes[aa];

        // Create an 'empty' expr with the given type. Ideally, the expression itself should not
        // matter only its checked type.
        //
        auto argExpr = getCurrentASTBuilder()->create<VarExpr>();
        argExpr->type = argType;
        argExpr->type.isLeftValue = true;
        argExprs.add(argExpr);
    }

    // Construct invoke expr.
    auto invokeExpr = getCurrentASTBuilder()->create<InvokeExpr>();
    invokeExpr->functionExpr = funcExpr;
    invokeExpr->arguments = argExprs;

    auto checkedInvokeExpr = visitor.CheckInvokeExprWithCheckedOperands(invokeExpr);

    return as<DeclRefExpr>(as<InvokeExpr>(checkedInvokeExpr)->functionExpr)->declRef;
}


DeclRef<Decl> Linkage::specializeGeneric(
    DeclRef<Decl> declRef,
    List<Expr*> argExprs,
    DiagnosticSink* sink)
{
    SLANG_AST_BUILDER_RAII(getASTBuilder());
    SLANG_ASSERT(declRef);

    SemanticsVisitor visitor(getSemanticsForReflection());
    visitor = visitor.withSink(sink);

    auto genericDeclRef = getGenericParentDeclRef(getASTBuilder(), &visitor, declRef);

    DeclRefExpr* declRefExpr = getASTBuilder()->create<DeclRefExpr>();
    declRefExpr->declRef = genericDeclRef;

    GenericAppExpr* genericAppExpr = getASTBuilder()->create<GenericAppExpr>();
    genericAppExpr->functionExpr = declRefExpr;
    genericAppExpr->arguments = argExprs;

    auto specializedDeclRef =
        as<DeclRefExpr>(visitor.checkGenericAppWithCheckedArgs(genericAppExpr))->declRef;

    return specializedDeclRef;
}

SLANG_NO_THROW slang::TypeLayoutReflection* SLANG_MCALL Linkage::getTypeLayout(
    slang::TypeReflection* inType,
    SlangInt targetIndex,
    slang::LayoutRules rules,
    ISlangBlob** outDiagnostics)
{
    SLANG_AST_BUILDER_RAII(getASTBuilder());

    auto type = asInternal(inType);

    if (targetIndex < 0 || targetIndex >= targets.getCount())
        return nullptr;

    auto target = targets[targetIndex];

    // TODO: We need a way to pass through the layout rules
    // that the user requested (e.g., constant buffers vs.
    // structured buffer rules). Right now the API only
    // exposes a single case, so this isn't a big deal.
    //
    SLANG_UNUSED(rules);

    auto typeLayout = target->getTypeLayout(type, rules);

    // TODO: We currently don't have a path for capturing
    // errors that occur during layout (e.g., types that
    // are invalid because of target-specific layout constraints).
    //
    SLANG_UNUSED(outDiagnostics);

    return asExternal(typeLayout);
}

SLANG_NO_THROW slang::TypeReflection* SLANG_MCALL Linkage::getContainerType(
    slang::TypeReflection* inType,
    slang::ContainerType containerType,
    ISlangBlob** outDiagnostics)
{
    SLANG_AST_BUILDER_RAII(getASTBuilder());

    auto type = asInternal(inType);

    Type* containerTypeReflection = nullptr;
    ContainerTypeKey key = {inType, containerType};
    if (!m_containerTypes.tryGetValue(key, containerTypeReflection))
    {
        switch (containerType)
        {
        case slang::ContainerType::ConstantBuffer:
            {
                SemanticsVisitor visitor(getSemanticsForReflection());
                auto layoutType = getASTBuilder()->getDefaultLayoutType();
                Type* cbType = visitor.getConstantBufferType(type, layoutType);
                containerTypeReflection = cbType;
            }
            break;
        case slang::ContainerType::ParameterBlock:
            {
                ParameterBlockType* pbType = getASTBuilder()->getParameterBlockType(type);
                containerTypeReflection = pbType;
            }
            break;
        case slang::ContainerType::StructuredBuffer:
            {
                HLSLStructuredBufferType* sbType = getASTBuilder()->getStructuredBufferType(type);
                containerTypeReflection = sbType;
            }
            break;
        case slang::ContainerType::UnsizedArray:
            {
                ArrayExpressionType* arrType = getASTBuilder()->getArrayType(type, nullptr);
                containerTypeReflection = arrType;
            }
            break;
        default:
            containerTypeReflection = type;
            break;
        }

        m_containerTypes.add(key, containerTypeReflection);
    }

    SLANG_UNUSED(outDiagnostics);

    return asExternal(containerTypeReflection);
}

SLANG_NO_THROW slang::TypeReflection* SLANG_MCALL Linkage::getDynamicType()
{
    SLANG_AST_BUILDER_RAII(getASTBuilder());

    return asExternal(getASTBuilder()->getSharedASTBuilder()->getDynamicType());
}

SLANG_NO_THROW SlangResult SLANG_MCALL
Linkage::getTypeRTTIMangledName(slang::TypeReflection* type, ISlangBlob** outNameBlob)
{
    SLANG_AST_BUILDER_RAII(getASTBuilder());

    auto internalType = asInternal(type);
    if (auto declRefType = as<DeclRefType>(internalType))
    {
        auto name = getMangledName(m_astBuilder, declRefType->getDeclRef());
        Slang::ComPtr<ISlangBlob> blob = Slang::StringUtil::createStringBlob(name);
        *outNameBlob = blob.detach();
        return SLANG_OK;
    }
    return SLANG_FAIL;
}

SLANG_NO_THROW SlangResult SLANG_MCALL Linkage::getTypeConformanceWitnessMangledName(
    slang::TypeReflection* type,
    slang::TypeReflection* interfaceType,
    ISlangBlob** outNameBlob)
{
    SLANG_AST_BUILDER_RAII(getASTBuilder());

    auto subType = asInternal(type);
    auto supType = asInternal(interfaceType);
    auto name = getMangledNameForConformanceWitness(m_astBuilder, subType, supType);
    Slang::ComPtr<ISlangBlob> blob = Slang::StringUtil::createStringBlob(name);
    *outNameBlob = blob.detach();
    return SLANG_OK;
}

SLANG_NO_THROW SlangResult SLANG_MCALL Linkage::getTypeConformanceWitnessSequentialID(
    slang::TypeReflection* type,
    slang::TypeReflection* interfaceType,
    uint32_t* outId)
{
    SLANG_AST_BUILDER_RAII(getASTBuilder());

    auto subType = asInternal(type);
    auto supType = asInternal(interfaceType);

    if (!subType || !supType)
        return SLANG_FAIL;

    auto name = getMangledNameForConformanceWitness(m_astBuilder, subType, supType);
    auto interfaceName = getMangledTypeName(m_astBuilder, supType);
    uint32_t resultIndex = 0;
    if (mapMangledNameToRTTIObjectIndex.tryGetValue(name, resultIndex))
    {
        if (outId)
            *outId = resultIndex;
        return SLANG_OK;
    }
    auto idAllocator = mapInterfaceMangledNameToSequentialIDCounters.tryGetValue(interfaceName);
    if (!idAllocator)
    {
        mapInterfaceMangledNameToSequentialIDCounters[interfaceName] = 0;
        idAllocator = mapInterfaceMangledNameToSequentialIDCounters.tryGetValue(interfaceName);
    }
    resultIndex = (*idAllocator);
    ++(*idAllocator);
    mapMangledNameToRTTIObjectIndex[name] = resultIndex;
    if (outId)
        *outId = resultIndex;
    return SLANG_OK;
}

SLANG_NO_THROW SlangResult SLANG_MCALL Linkage::createTypeConformanceComponentType(
    slang::TypeReflection* type,
    slang::TypeReflection* interfaceType,
    slang::ITypeConformance** outConformanceComponentType,
    SlangInt conformanceIdOverride,
    ISlangBlob** outDiagnostics)
{
    if (outConformanceComponentType == nullptr)
        return SLANG_E_INVALID_ARG;

    SLANG_AST_BUILDER_RAII(getASTBuilder());

    RefPtr<TypeConformance> result;
    DiagnosticSink sink;
    applySettingsToDiagnosticSink(&sink, &sink, m_optionSet);

    try
    {
        SemanticsVisitor visitor(getSemanticsForReflection());
        visitor = visitor.withSink(&sink);

        auto witness = visitor.isSubtype(
            (Slang::Type*)type,
            (Slang::Type*)interfaceType,
            IsSubTypeOptions::None);
        if (auto subtypeWitness = as<SubtypeWitness>(witness))
        {
            result = new TypeConformance(this, subtypeWitness, conformanceIdOverride, &sink);
        }
    }
    catch (...)
    {
    }
    sink.getBlobIfNeeded(outDiagnostics);
    bool success = (result != nullptr);
    *outConformanceComponentType = result.detach();
    return success ? SLANG_OK : SLANG_FAIL;
}

SLANG_NO_THROW SlangResult SLANG_MCALL
Linkage::createCompileRequest(SlangCompileRequest** outCompileRequest)
{
    auto compileRequest = new EndToEndCompileRequest(this);
    compileRequest->addRef();
    *outCompileRequest = asExternal(compileRequest);
    return SLANG_OK;
}

SLANG_NO_THROW SlangInt SLANG_MCALL Linkage::getLoadedModuleCount()
{
    return loadedModulesList.getCount();
}

SLANG_NO_THROW slang::IModule* SLANG_MCALL Linkage::getLoadedModule(SlangInt index)
{
    if (index >= 0 && index < loadedModulesList.getCount())
        return loadedModulesList[index].get();
    return nullptr;
}

void Linkage::buildHash(DigestBuilder<SHA1>& builder, SlangInt targetIndex)
{
    // Add the Slang compiler version to the hash
    auto version = String(getBuildTagString());
    builder.append(version);

    // Add compiler options, including search path, preprocessor includes, etc.
    m_optionSet.buildHash(builder);

    auto addTargetDigest = [&](TargetRequest* targetReq)
    {
        targetReq->getOptionSet().buildHash(builder);

        const PassThroughMode passThroughMode =
            getDownstreamCompilerRequiredForTarget(targetReq->getTarget());
        const SourceLanguage sourceLanguage =
            getDefaultSourceLanguageForDownstreamCompiler(passThroughMode);

        // Add prelude for the given downstream compiler.
        ComPtr<ISlangBlob> prelude;
        getGlobalSession()->getLanguagePrelude(
            (SlangSourceLanguage)sourceLanguage,
            prelude.writeRef());
        if (prelude)
        {
            builder.append(prelude);
        }

        // TODO: Downstream compilers (specifically dxc) can currently #include additional
        // dependencies. This is currently the case for NVAPI headers included in the prelude. These
        // dependencies are currently not picked up by the shader cache which is a significant
        // issue. This can only be fixed by running the preprocessor in the slang compiler so dxc
        // (or any other downstream compiler for that matter) isn't resolving any includes
        // implicitly.

        // Add the downstream compiler version (if it exists) to the hash
        auto downstreamCompiler =
            getSessionImpl()->getOrLoadDownstreamCompiler(passThroughMode, nullptr);
        if (downstreamCompiler)
        {
            ComPtr<ISlangBlob> versionString;
            if (SLANG_SUCCEEDED(downstreamCompiler->getVersionString(versionString.writeRef())))
            {
                builder.append(versionString);
            }
        }
    };

    // Add the target specified by targetIndex
    if (targetIndex == -1)
    {
        // -1 means all targets.
        for (auto targetReq : targets)
        {
            addTargetDigest(targetReq);
        }
    }
    else
    {
        auto targetReq = targets[targetIndex];
        addTargetDigest(targetReq);
    }
}

SlangResult Linkage::addSearchPath(char const* path)
{
    m_optionSet.add(CompilerOptionName::Include, String(path));
    return SLANG_OK;
}

SlangResult Linkage::addPreprocessorDefine(char const* name, char const* value)
{
    CompilerOptionValue val;
    val.kind = CompilerOptionValueKind::String;
    val.stringValue = name;
    val.stringValue2 = value;
    m_optionSet.add(CompilerOptionName::MacroDefine, val);
    return SLANG_OK;
}

SlangResult Linkage::setMatrixLayoutMode(SlangMatrixLayoutMode mode)
{
    m_optionSet.setMatrixLayoutMode((MatrixLayoutMode)mode);
    return SLANG_OK;
}

//
// TargetRequest
//

TargetRequest::TargetRequest(Linkage* linkage, CodeGenTarget format)
    : linkage(linkage)
{
    optionSet = linkage->m_optionSet;
    optionSet.add(CompilerOptionName::Target, format);
}

TargetRequest::TargetRequest(const TargetRequest& other)
    : RefObject(), linkage(other.linkage), optionSet(other.optionSet)
{
}


Session* TargetRequest::getSession()
{
    return linkage->getSessionImpl();
}

HLSLToVulkanLayoutOptions* TargetRequest::getHLSLToVulkanLayoutOptions()
{
    if (!hlslToVulkanOptions)
    {
        hlslToVulkanOptions = new HLSLToVulkanLayoutOptions();
        hlslToVulkanOptions->loadFromOptionSet(optionSet);
    }
    return hlslToVulkanOptions.get();
}

void TargetRequest::setTargetCaps(CapabilitySet capSet)
{
    cookedCapabilities = capSet;
}

CapabilitySet TargetRequest::getTargetCaps()
{
    if (!cookedCapabilities.isEmpty())
        return cookedCapabilities;

    // The full `CapabilitySet` for the target will be computed
    // from the combination of the code generation format, and
    // the profile.
    //
    // Note: the preofile might have been set in a way that is
    // inconsistent with the output code format of SPIR-V, but
    // a profile of Direct3D Shader Model 5.1. In those cases,
    // the format should always override the implications in
    // the profile.
    //
    // TODO: This logic isn't currently taking int account
    // the information in the profile, because the current
    // `CapabilityAtom`s that we support don't include any
    // of the details there (e.g., the shader model versions).
    //
    // Eventually, we'd want to have a rich set of capability
    // atoms, so that most of the information about what operations
    // are available where can be directly encoded on the declarations.

    List<CapabilityName> atoms;

    // If the user specified a explicit profile, we should pull
    // a corresponding atom representing the target version from the profile.
    CapabilitySet profileCaps = optionSet.getProfile().getCapabilityName();

    bool isGLSLTarget = false;
    switch (getTarget())
    {
    case CodeGenTarget::GLSL:
        isGLSLTarget = true;
        atoms.add(CapabilityName::glsl);
        break;
    case CodeGenTarget::SPIRV:
    case CodeGenTarget::SPIRVAssembly:
        if (getOptionSet().shouldEmitSPIRVDirectly())
        {
            // Default to SPIRV 1.5 if the user has not specified a target version.
            bool hasTargetVersionAtom = false;
            if (!profileCaps.isEmpty())
            {
                profileCaps.join(CapabilitySet(CapabilityName::spirv_1_0));
                for (auto profileCapAtomSet : profileCaps.getAtomSets())
                {
                    for (auto atom : profileCapAtomSet)
                    {
                        if (isTargetVersionAtom(asAtom(atom)))
                        {
                            atoms.add((CapabilityName)atom);
                            hasTargetVersionAtom = true;
                        }
                    }
                }
            }
            if (!hasTargetVersionAtom)
            {
                atoms.add(CapabilityName::spirv_1_5);
            }
            // If the user specified any SPIR-V extensions in the profile,
            // pull them in.
            for (auto profileCapAtomSet : profileCaps.getAtomSets())
            {
                for (auto atom : profileCapAtomSet)
                {
                    if (isSpirvExtensionAtom(asAtom(atom)))
                    {
                        atoms.add((CapabilityName)atom);
                        hasTargetVersionAtom = true;
                    }
                }
            }
        }
        else
        {
            isGLSLTarget = true;
            atoms.add(CapabilityName::glsl);
            profileCaps.addSpirvVersionFromOtherAsGlslSpirvVersion(profileCaps);
        }
        break;

    case CodeGenTarget::HLSL:
    case CodeGenTarget::DXBytecode:
    case CodeGenTarget::DXBytecodeAssembly:
    case CodeGenTarget::DXIL:
    case CodeGenTarget::DXILAssembly:
        atoms.add(CapabilityName::hlsl);
        break;

    case CodeGenTarget::CSource:
        atoms.add(CapabilityName::c);
        break;

    case CodeGenTarget::CPPSource:
    case CodeGenTarget::PyTorchCppBinding:
    case CodeGenTarget::HostExecutable:
    case CodeGenTarget::ShaderSharedLibrary:
    case CodeGenTarget::HostSharedLibrary:
    case CodeGenTarget::HostHostCallable:
    case CodeGenTarget::ShaderHostCallable:
        atoms.add(CapabilityName::cpp);
        break;

    case CodeGenTarget::CUDASource:
    case CodeGenTarget::PTX:
        atoms.add(CapabilityName::cuda);
        break;

    case CodeGenTarget::Metal:
    case CodeGenTarget::MetalLib:
    case CodeGenTarget::MetalLibAssembly:
        atoms.add(CapabilityName::metal);
        break;

    case CodeGenTarget::WGSLSPIRV:
    case CodeGenTarget::WGSLSPIRVAssembly:
    case CodeGenTarget::WGSL:
        atoms.add(CapabilityName::wgsl);
        break;

    default:
        break;
    }

    CapabilitySet targetCap = CapabilitySet(atoms);

    if (profileCaps.atLeastOneSetImpliedInOther(targetCap) ==
        CapabilitySet::ImpliesReturnFlags::Implied)
        targetCap.join(profileCaps);

    for (auto atomVal : optionSet.getArray(CompilerOptionName::Capability))
    {
        auto toAdd = CapabilitySet((CapabilityName)atomVal.intValue);

        if (isGLSLTarget)
            targetCap.addSpirvVersionFromOtherAsGlslSpirvVersion(toAdd);

        if (!targetCap.isIncompatibleWith(toAdd))
            targetCap.join(toAdd);
    }

    cookedCapabilities = targetCap;

    SLANG_ASSERT(!cookedCapabilities.isInvalid());

    return cookedCapabilities;
}


TypeLayout* TargetRequest::getTypeLayout(Type* type, slang::LayoutRules rules)
{
    SLANG_AST_BUILDER_RAII(getLinkage()->getASTBuilder());

    // TODO: We are not passing in a `ProgramLayout` here, although one
    // is nominally required to establish the global ordering of
    // generic type parameters, which might be referenced from field types.
    //
    // The solution here is to make sure that the reflection data for
    // uses of global generic/existential types does *not* include any
    // kind of index in that global ordering, and just refers to the
    // parameter instead (leaving the user to figure out how that
    // maps to the ordering via some API on the program layout).
    //
    auto layoutContext = getInitialLayoutContextForTarget(this, nullptr, rules);

    RefPtr<TypeLayout> result;
    auto key = TypeLayoutKey{type, rules};
    if (getTypeLayouts().tryGetValue(key, result))
        return result.Ptr();
    result = createTypeLayout(layoutContext, type);
    getTypeLayouts()[key] = result;
    return result.Ptr();
}

//
// TranslationUnitRequest
//

TranslationUnitRequest::TranslationUnitRequest(FrontEndCompileRequest* compileRequest)
    : compileRequest(compileRequest)
{
    module = new Module(compileRequest->getLinkage());
}

TranslationUnitRequest::TranslationUnitRequest(FrontEndCompileRequest* compileRequest, Module* m)
    : compileRequest(compileRequest), module(m), isChecked(true)
{
    moduleName = getNamePool()->getName(m->getName());
}

Session* TranslationUnitRequest::getSession()
{
    return compileRequest->getSession();
}

NamePool* TranslationUnitRequest::getNamePool()
{
    return compileRequest->getNamePool();
}

SourceManager* TranslationUnitRequest::getSourceManager()
{
    return compileRequest->getSourceManager();
}

Scope* TranslationUnitRequest::getLanguageScope()
{
    Scope* languageScope = nullptr;
    switch (sourceLanguage)
    {
    case SourceLanguage::HLSL:
        languageScope = getSession()->hlslLanguageScope;
        break;
    case SourceLanguage::GLSL:
        languageScope = getSession()->glslLanguageScope;
        break;
    case SourceLanguage::Slang:
    default:
        languageScope = getSession()->slangLanguageScope;
        break;
    }
    return languageScope;
}

Dictionary<String, String> TranslationUnitRequest::getCombinedPreprocessorDefinitions()
{
    Dictionary<String, String> combinedPreprocessorDefinitions;
    for (const auto& def : preprocessorDefinitions)
        combinedPreprocessorDefinitions.addIfNotExists(def);
    for (const auto& def : compileRequest->optionSet.getArray(CompilerOptionName::MacroDefine))
        combinedPreprocessorDefinitions.addIfNotExists(def.stringValue, def.stringValue2);

    // Define standard macros, if not already defined. This style assumes using `#if __SOME_VAR`
    // style, as in
    //
    // ```
    // #if __SLANG_COMPILER__
    // ```
    //
    // This choice is made because slang outputs a warning on using a variable in an #if if not
    // defined
    //
    // Of course this means using #ifndef/#ifdef/defined() is probably not appropraite with thes
    // variables.
    {
        // Used to identify level of HLSL language compatibility
        combinedPreprocessorDefinitions.addIfNotExists("__HLSL_VERSION", "2018");

        // Indicates this is being compiled by the slang *compiler*
        combinedPreprocessorDefinitions.addIfNotExists("__SLANG_COMPILER__", "1");

        // Set macro depending on source type
        switch (sourceLanguage)
        {
        case SourceLanguage::HLSL:
            // Used to indicate compiled as HLSL language
            combinedPreprocessorDefinitions.addIfNotExists("__HLSL__", "1");
            break;
        case SourceLanguage::Slang:
            // Used to indicate compiled as Slang language
            combinedPreprocessorDefinitions.addIfNotExists("__SLANG__", "1");
            break;
        default:
            break;
        }

        // If not set, define as 0.
        combinedPreprocessorDefinitions.addIfNotExists("__HLSL__", "0");
        combinedPreprocessorDefinitions.addIfNotExists("__SLANG__", "0");
    }

    return combinedPreprocessorDefinitions;
}

void TranslationUnitRequest::addSourceArtifact(IArtifact* sourceArtifact)
{
    SLANG_ASSERT(sourceArtifact);
    m_sourceArtifacts.add(ComPtr<IArtifact>(sourceArtifact));
}


void TranslationUnitRequest::addSource(IArtifact* sourceArtifact, SourceFile* sourceFile)
{
    SLANG_ASSERT(sourceArtifact && sourceFile);
    // Must be in sync!
    SLANG_ASSERT(m_sourceFiles.getCount() == m_sourceArtifacts.getCount());

    addSourceArtifact(sourceArtifact);
    _addSourceFile(sourceFile);
}

PathInfo TranslationUnitRequest::_findSourcePathInfo(IArtifact* artifact)
{
    auto pathRep = findRepresentation<IPathArtifactRepresentation>(artifact);

    if (pathRep && pathRep->getPathType() == SLANG_PATH_TYPE_FILE)
    {
        // See if we have a unique identity set with the path
        if (const auto uniqueIdentity = pathRep->getUniqueIdentity())
        {
            return PathInfo::makeNormal(pathRep->getPath(), uniqueIdentity);
        }

        // If we couldn't get a unique identity, just use the path
        return PathInfo::makePath(pathRep->getPath());
    }

    // If there isn't a path, we can try with the name
    const char* name = artifact->getName();
    if (name && name[0] != 0)
    {
        return PathInfo::makeFromString(name);
    }

    return PathInfo::makeUnknown();
}

SlangResult TranslationUnitRequest::requireSourceFiles()
{
    SLANG_ASSERT(m_sourceFiles.getCount() <= m_sourceArtifacts.getCount());

    if (m_sourceFiles.getCount() == m_sourceArtifacts.getCount())
    {
        return SLANG_OK;
    }

    auto sink = compileRequest->getSink();
    SourceManager* sourceManager = compileRequest->getSourceManager();

    for (Index i = m_sourceFiles.getCount(); i < m_sourceArtifacts.getCount(); ++i)
    {
        IArtifact* artifact = m_sourceArtifacts[i];

        const PathInfo pathInfo = _findSourcePathInfo(artifact);

        SourceFile* sourceFile = nullptr;
        ComPtr<ISlangBlob> blob;

        // If we have a unique identity see if we have it already
        if (pathInfo.hasUniqueIdentity())
        {
            // See if this an already loaded source file
            sourceFile = sourceManager->findSourceFileRecursively(pathInfo.uniqueIdentity);
            // If we have a sourceFile see if it has a blob
            if (sourceFile)
            {
                blob = sourceFile->getContentBlob();
            }
        }

        // If we *don't* have a blob try and get a blob from the artifact
        if (!blob)
        {
            const SlangResult res = artifact->loadBlob(ArtifactKeep::Yes, blob.writeRef());
            if (SLANG_FAILED(res))
            {
                // Report couldn't load
                sink->diagnose(SourceLoc(), Diagnostics::cannotOpenFile, pathInfo.getName());
                return res;
            }
        }

        // If we don't have a blob on the artifact we can now add the one we have
        if (!findRepresentation<ISlangBlob>(artifact))
        {
            artifact->addRepresentationUnknown(blob);
        }

        // If we have a sourceFile check if it has contents, and set the blob if doesn't
        if (sourceFile)
        {
            if (!sourceFile->getContentBlob())
            {
                sourceFile->setContents(blob);
            }
        }
        else
        {
            // Create a new source file, using the pathInfo and blob
            sourceFile = sourceManager->createSourceFileWithBlob(pathInfo, blob);
        }

        auto uniqueIdentity = pathInfo.getMostUniqueIdentity();
        if (uniqueIdentity.getLength())
            sourceManager->addSourceFileIfNotExist(uniqueIdentity, sourceFile);

        // Finally add the source file
        _addSourceFile(sourceFile);
    }

    return SLANG_OK;
}

void TranslationUnitRequest::_addSourceFile(SourceFile* sourceFile)
{
    m_sourceFiles.add(sourceFile);

    getModule()->addFileDependency(sourceFile);
    getModule()->getIncludedSourceFileMap().add(sourceFile, nullptr);
}

List<SourceFile*> const& TranslationUnitRequest::getSourceFiles()
{
    SLANG_ASSERT(m_sourceArtifacts.getCount() == m_sourceFiles.getCount());
    return m_sourceFiles;
}

EndToEndCompileRequest::~EndToEndCompileRequest()
{
    // Flush any writers associated with the request
    m_writers->flushWriters();

    m_linkage.setNull();
    m_frontEndReq.setNull();
}

static ISlangWriter* _getDefaultWriter(WriterChannel chan)
{
    static FileWriter stdOut(stdout, WriterFlag::IsStatic | WriterFlag::IsUnowned);
    static FileWriter stdError(stderr, WriterFlag::IsStatic | WriterFlag::IsUnowned);
    static NullWriter nullWriter(WriterFlag::IsStatic | WriterFlag::IsConsole);

    switch (chan)
    {
    case WriterChannel::StdError:
        return &stdError;
    case WriterChannel::StdOutput:
        return &stdOut;
    case WriterChannel::Diagnostic:
        return &nullWriter;
    default:
        {
            SLANG_ASSERT(!"Unknown type");
            return &stdError;
        }
    }
}

void EndToEndCompileRequest::setWriter(WriterChannel chan, ISlangWriter* writer)
{
    // If the user passed in null, we will use the default writer on that channel
    m_writers->setWriter(SlangWriterChannel(chan), writer ? writer : _getDefaultWriter(chan));

    // For diagnostic output, if the user passes in nullptr, we set on m_sink.writer as that enables
    // buffering on DiagnosticSink
    if (chan == WriterChannel::Diagnostic)
    {
        m_sink.writer = writer;
    }
}

SlangResult Linkage::loadFile(String const& path, PathInfo& outPathInfo, ISlangBlob** outBlob)
{
    outPathInfo.type = PathInfo::Type::Unknown;

    SLANG_RETURN_ON_FAIL(m_fileSystemExt->loadFile(path.getBuffer(), outBlob));

    ComPtr<ISlangBlob> uniqueIdentity;
    // Get the unique identity
    if (SLANG_FAILED(
            m_fileSystemExt->getFileUniqueIdentity(path.getBuffer(), uniqueIdentity.writeRef())))
    {
        // We didn't get a unique identity, so go with just a found path
        outPathInfo.type = PathInfo::Type::FoundPath;
        outPathInfo.foundPath = path;
    }
    else
    {
        outPathInfo = PathInfo::makeNormal(path, StringUtil::getString(uniqueIdentity));
    }
    return SLANG_OK;
}

Expr* Linkage::parseTermString(String typeStr, Scope* scope)
{
    // Create a SourceManager on the stack, so any allocations for 'SourceFile'/'SourceView' etc
    // will be cleaned up
    SourceManager localSourceManager;
    localSourceManager.initialize(getSourceManager(), nullptr);

    Slang::SourceFile* srcFile =
        localSourceManager.createSourceFileWithString(PathInfo::makeTypeParse(), typeStr);

    // We'll use a temporary diagnostic sink
    DiagnosticSink sink(&localSourceManager, nullptr);

    // RAII type to make make sure current SourceManager is restored after parse.
    // Use RAII - to make sure everything is reset even if an exception is thrown.
    struct ScopeReplaceSourceManager
    {
        ScopeReplaceSourceManager(Linkage* linkage, SourceManager* replaceManager)
            : m_linkage(linkage), m_originalSourceManager(linkage->getSourceManager())
        {
            linkage->setSourceManager(replaceManager);
        }

        ~ScopeReplaceSourceManager() { m_linkage->setSourceManager(m_originalSourceManager); }

    private:
        Linkage* m_linkage;
        SourceManager* m_originalSourceManager;
    };

    // We need to temporarily replace the SourceManager for this CompileRequest
    ScopeReplaceSourceManager scopeReplaceSourceManager(this, &localSourceManager);

    SourceLanguage sourceLanguage;

    auto tokens = preprocessSource(
        srcFile,
        &sink,
        nullptr,
        Dictionary<String, String>(),
        this,
        sourceLanguage);

    if (sourceLanguage == SourceLanguage::Unknown)
        sourceLanguage = SourceLanguage::Slang;

    return parseTermFromSourceFile(
        getASTBuilder(),
        tokens,
        &sink,
        scope,
        getNamePool(),
        sourceLanguage);
}

Type* checkProperType(Linkage* linkage, TypeExp typeExp, DiagnosticSink* sink);

Type* ComponentType::getTypeFromString(String const& typeStr, DiagnosticSink* sink)
{
    // If we've looked up this type name before,
    // then we can re-use it.
    //
    Type* type = nullptr;
    if (m_types.tryGetValue(typeStr, type))
        return type;


    // TODO(JS): For now just used the linkages ASTBuilder to keep on scope
    //
    // The parseTermString uses the linkage ASTBuilder for it's parsing.
    //
    // It might be possible to just create a temporary ASTBuilder - the worry though is
    // that the parsing sets a member variable in AST node to one of these scopes, and then
    // it become a dangling pointer. So for now we go with the linkages.
    auto astBuilder = getLinkage()->getASTBuilder();

    // Otherwise, we need to start looking in
    // the modules that were directly or
    // indirectly referenced.
    //
    Scope* scope = _getOrCreateScopeForLegacyLookup(astBuilder);

    auto linkage = getLinkage();

    SLANG_AST_BUILDER_RAII(linkage->getASTBuilder());

    Expr* typeExpr = linkage->parseTermString(typeStr, scope);
    type = checkProperType(linkage, TypeExp(typeExpr), sink);

    if (type)
    {
        m_types[typeStr] = type;
    }
    return type;
}

Expr* ComponentType::findDeclFromString(String const& name, DiagnosticSink* sink)
{
    // If we've looked up this type name before,
    // then we can re-use it.
    //
    Expr* result = nullptr;
    if (m_decls.tryGetValue(name, result))
        return result;


    // TODO(JS): For now just used the linkages ASTBuilder to keep on scope
    //
    // The parseTermString uses the linkage ASTBuilder for it's parsing.
    //
    // It might be possible to just create a temporary ASTBuilder - the worry though is
    // that the parsing sets a member variable in AST node to one of these scopes, and then
    // it become a dangling pointer. So for now we go with the linkages.
    auto astBuilder = getLinkage()->getASTBuilder();

    // Otherwise, we need to start looking in
    // the modules that were directly or
    // indirectly referenced.
    //
    Scope* scope = _getOrCreateScopeForLegacyLookup(astBuilder);

    auto linkage = getLinkage();

    SLANG_AST_BUILDER_RAII(linkage->getASTBuilder());

    Expr* expr = linkage->parseTermString(name, scope);

    SemanticsContext context(linkage->getSemanticsForReflection());
    context = context.allowStaticReferenceToNonStaticMember().withSink(sink);

    SemanticsVisitor visitor(context);

    auto checkedExpr = visitor.CheckTerm(expr);

    if (as<DeclRefExpr>(checkedExpr) || as<OverloadedExpr>(checkedExpr))
    {
        result = checkedExpr;
    }

    m_decls[name] = result;
    return result;
}

bool isSimpleName(String const& name)
{
    for (char c : name)
    {
        if (!CharUtil::isAlphaOrDigit(c) && c != '_' && c != '$')
            return false;
    }
    return true;
}

Expr* ComponentType::findDeclFromStringInType(
    Type* type,
    String const& name,
    LookupMask mask,
    DiagnosticSink* sink)
{
    // Only look up in the type if it is a DeclRefType
    if (!as<DeclRefType>(type))
        return nullptr;

    // TODO(JS): For now just used the linkages ASTBuilder to keep on scope
    //
    // The parseTermString uses the linkage ASTBuilder for it's parsing.
    //
    // It might be possible to just create a temporary ASTBuilder - the worry though is
    // that the parsing sets a member variable in AST node to one of these scopes, and then
    // it become a dangling pointer. So for now we go with the linkages.
    auto astBuilder = getLinkage()->getASTBuilder();

    // Otherwise, we need to start looking in
    // the modules that were directly or
    // indirectly referenced.
    //
    Scope* scope = _getOrCreateScopeForLegacyLookup(astBuilder);

    auto linkage = getLinkage();

    SLANG_AST_BUILDER_RAII(linkage->getASTBuilder());

    Expr* expr = nullptr;

    if (isSimpleName(name))
    {
        auto varExpr = astBuilder->create<VarExpr>();
        varExpr->scope = scope;
        varExpr->name = getLinkage()->getNamePool()->getName(name);
        expr = varExpr;
    }
    else
    {
        expr = linkage->parseTermString(name, scope);
    }
    SemanticsContext context(linkage->getSemanticsForReflection());
    context = context.allowStaticReferenceToNonStaticMember().withSink(sink);

    SemanticsVisitor visitor(context);

    GenericAppExpr* genericOuterExpr = nullptr;
    if (as<GenericAppExpr>(expr))
    {
        // Unwrap the generic application, and re-wrap it around the static-member expr
        genericOuterExpr = as<GenericAppExpr>(expr);
        expr = genericOuterExpr->functionExpr;
    }

    if (!as<VarExpr>(expr))
        return nullptr;

    auto rs = astBuilder->create<StaticMemberExpr>();
    auto typeExpr = astBuilder->create<SharedTypeExpr>();
    auto typetype = astBuilder->getOrCreate<TypeType>(type);
    typeExpr->type = typetype;
    rs->baseExpression = typeExpr;
    rs->name = as<VarExpr>(expr)->name;

    expr = rs;

    // If we have a generic-app expression, re-wrap the static-member expr
    if (genericOuterExpr)
    {
        genericOuterExpr->functionExpr = expr;
        expr = genericOuterExpr;
    }

    auto checkedTerm = visitor.CheckTerm(expr);
    auto resolvedTerm = visitor.maybeResolveOverloadedExpr(checkedTerm, mask, sink);


    if (auto overloadedExpr = as<OverloadedExpr>(resolvedTerm))
    {
        return overloadedExpr;
    }
    if (auto declRefExpr = as<DeclRefExpr>(resolvedTerm))
    {
        return declRefExpr;
    }

    return nullptr;
}

bool ComponentType::isSubType(Type* subType, Type* superType)
{
    SemanticsContext context(getLinkage()->getSemanticsForReflection());
    SemanticsVisitor visitor(context);

    return (visitor.isSubtype(subType, superType, IsSubTypeOptions::None) != nullptr);
}

static void collectExportedConstantInContainer(
    Dictionary<String, IntVal*>& dict,
    ASTBuilder* builder,
    ContainerDecl* containerDecl)
{
    for (auto m : containerDecl->members)
    {
        auto varMember = as<VarDeclBase>(m);
        if (!varMember)
            continue;
        if (!varMember->val)
            continue;
        bool isExported = false;
        bool isConst = false;
        bool isExtern = false;
        for (auto modifier : m->modifiers)
        {
            if (as<HLSLExportModifier>(modifier))
                isExported = true;
            if (as<ExternAttribute>(modifier) || as<ExternModifier>(modifier))
            {
                isExtern = true;
                isExported = true;
            }
            if (as<ConstModifier>(modifier))
                isConst = true;
        }
        if (isExported && isConst)
        {
            auto mangledName = getMangledName(builder, m);
            if (isExtern && dict.containsKey(mangledName))
                continue;
            dict[mangledName] = varMember->val;
        }
    }

    for (auto member : containerDecl->members)
    {
        if (as<NamespaceDecl>(member) || as<FileDecl>(member))
        {
            collectExportedConstantInContainer(dict, builder, (ContainerDecl*)member);
        }
    }
}

Dictionary<String, IntVal*>& ComponentType::getMangledNameToIntValMap()
{
    if (m_mapMangledNameToIntVal)
    {
        return *m_mapMangledNameToIntVal;
    }
    m_mapMangledNameToIntVal = std::make_unique<Dictionary<String, IntVal*>>();
    auto astBuilder = getLinkage()->getASTBuilder();
    SLANG_AST_BUILDER_RAII(astBuilder);
    Scope* scope = _getOrCreateScopeForLegacyLookup(astBuilder);
    for (; scope; scope = scope->nextSibling)
    {
        if (scope->containerDecl)
            collectExportedConstantInContainer(
                *m_mapMangledNameToIntVal,
                astBuilder,
                scope->containerDecl);
    }
    return *m_mapMangledNameToIntVal;
}

ConstantIntVal* ComponentType::tryFoldIntVal(IntVal* intVal)
{
    auto astBuilder = getLinkage()->getASTBuilder();
    SLANG_AST_BUILDER_RAII(astBuilder);
    return as<ConstantIntVal>(intVal->linkTimeResolve(getMangledNameToIntValMap()));
}

CompileRequestBase::CompileRequestBase(Linkage* linkage, DiagnosticSink* sink)
    : m_linkage(linkage), m_sink(sink)
{
}


FrontEndCompileRequest::FrontEndCompileRequest(
    Linkage* linkage,
    StdWriters* writers,
    DiagnosticSink* sink)
    : CompileRequestBase(linkage, sink), m_writers(writers)
{
    optionSet.inheritFrom(linkage->m_optionSet);
}

/// Handlers for preprocessor callbacks to use when doing ordinary front-end compilation
struct FrontEndPreprocessorHandler : PreprocessorHandler
{
public:
    FrontEndPreprocessorHandler(Module* module, ASTBuilder* astBuilder, DiagnosticSink* sink)
        : m_module(module), m_astBuilder(astBuilder), m_sink(sink)
    {
    }

protected:
    Module* m_module;
    ASTBuilder* m_astBuilder;
    DiagnosticSink* m_sink;

    // The first task that this handler tries to deal with is
    // capturing all the files on which a module is dependent.
    //
    // That information is exposed through public APIs and used
    // by applications to decide when they need to "hot reload"
    // their shader code.
    //
    void handleFileDependency(SourceFile* sourceFile) SLANG_OVERRIDE
    {
        m_module->addFileDependency(sourceFile);
    }

    // The second task that this handler deals with is detecting
    // whether any macro values were set in a given source file
    // that are semantically relevant to other stages of compilation.
    //
    void handleEndOfTranslationUnit(Preprocessor* preprocessor) SLANG_OVERRIDE
    {
        // We look at the preprocessor state after reading the entire
        // source file/string, in order to see if any macros have been
        // set that should be considered semantically relevant for
        // later stages of compilation.
        //
        // Note: Checking the macro environment *after* preprocessing is complete
        // means that we can treat macros introduced via `-D` options or the API
        // equivalently to macros introduced via `#define`s in user code.
        //
        // For now, the only case of semantically-relevant macros we need to worrry
        // about are the NVAPI macros used to establish the register/space to use.
        //
        static const char* kNVAPIRegisterMacroName = "NV_SHADER_EXTN_SLOT";
        static const char* kNVAPISpaceMacroName = "NV_SHADER_EXTN_REGISTER_SPACE";

        // For NVAPI use, the `NV_SHADER_EXTN_SLOT` macro is required to be defined.
        //
        String nvapiRegister;
        SourceLoc nvapiRegisterLoc;
        if (!SLANG_FAILED(findMacroValue(
                preprocessor,
                kNVAPIRegisterMacroName,
                nvapiRegister,
                nvapiRegisterLoc)))
        {
            // In contrast, NVAPI can be used without defining `NV_SHADER_EXTN_REGISTER_SPACE`,
            // which effectively defaults to `space0`.
            //
            String nvapiSpace = "space0";
            SourceLoc nvapiSpaceLoc;
            findMacroValue(preprocessor, kNVAPISpaceMacroName, nvapiSpace, nvapiSpaceLoc);

            // We are going to store the values of these macros on the AST-level `ModuleDecl`
            // so that they will be available to later processing stages.
            //
            auto moduleDecl = m_module->getModuleDecl();

            if (auto existingModifier = moduleDecl->findModifier<NVAPISlotModifier>())
            {
                // If there is already a modifier attached to the module (perhaps
                // because of preprocessing a different source file, or because
                // of settings established via command-line options), then we
                // need to validate that the values being set in this file
                // match those already set (or else there is likely to be
                // some kind of error in the user's code).
                //
                _validateNVAPIMacroMatch(
                    kNVAPIRegisterMacroName,
                    existingModifier->registerName,
                    nvapiRegister,
                    nvapiRegisterLoc);
                _validateNVAPIMacroMatch(
                    kNVAPISpaceMacroName,
                    existingModifier->spaceName,
                    nvapiSpace,
                    nvapiSpaceLoc);
            }
            else
            {
                // If there is no existing modifier on the module, then we
                // take responsibility for adding one, based on the macro
                // values we saw.
                //
                auto modifier = m_astBuilder->create<NVAPISlotModifier>();
                modifier->loc = nvapiRegisterLoc;
                modifier->registerName = nvapiRegister;
                modifier->spaceName = nvapiSpace;

                addModifier(moduleDecl, modifier);
            }
        }
    }

    /// Validate that a re-defintion of an NVAPI-related macro matches any previous definition
    void _validateNVAPIMacroMatch(
        char const* macroName,
        String const& existingValue,
        String const& newValue,
        SourceLoc loc)
    {
        if (existingValue != newValue)
        {
            m_sink->diagnose(
                loc,
                Diagnostics::nvapiMacroMismatch,
                macroName,
                existingValue,
                newValue);
        }
    }
};


// Holds the hierarchy of views, the children being views that were 'initiated' (have an initiating
// SourceLoc) in the parent.
typedef Dictionary<SourceView*, List<SourceView*>> ViewInitiatingHierarchy;

// Calculate the hierarchy from the sourceManager
static void _calcViewInitiatingHierarchy(
    SourceManager* sourceManager,
    ViewInitiatingHierarchy& outHierarchy)
{
    const List<SourceView*> emptyList;
    outHierarchy.clear();

    // Iterate over all managers
    for (SourceManager* curManager = sourceManager; curManager;
         curManager = curManager->getParent())
    {
        // Iterate over all views
        for (SourceView* view : curManager->getSourceViews())
        {
            if (view->getInitiatingSourceLoc().isValid())
            {
                // Look up the view it came from
                SourceView* parentView =
                    sourceManager->findSourceViewRecursively(view->getInitiatingSourceLoc());
                if (parentView)
                {
                    List<SourceView*>& children = outHierarchy.getOrAddValue(parentView, emptyList);
                    // It shouldn't have already been added
                    SLANG_ASSERT(children.indexOf(view) < 0);
                    children.add(view);
                }
            }
        }
    }

    // Order all the children, by their raw SourceLocs. This is desirable, so that a trivial
    // traversal will traverse children in the order they are initiated in the parent source. This
    // assumes they increase in SourceLoc implies an later within a source file - this is true
    // currently.
    for (auto& [_, value] : outHierarchy)
    {
        value.sort(
            [](SourceView* a, SourceView* b) -> bool {
                return a->getInitiatingSourceLoc().getRaw() < b->getInitiatingSourceLoc().getRaw();
            });
    }
}

// Given a source file, find the view that is the initial SourceView use of the source. It must have
// an initiating SourceLoc that is not valid.
static SourceView* _findInitialSourceView(SourceFile* sourceFile)
{
    // TODO(JS):
    // This might be overkill - presumably the SourceView would belong to the same manager as it's
    // SourceFile? That is not enforced by the SourceManager in any way though so we just search all
    // managers, and all views.
    for (SourceManager* sourceManager = sourceFile->getSourceManager(); sourceManager;
         sourceManager = sourceManager->getParent())
    {
        for (SourceView* view : sourceManager->getSourceViews())
        {
            if (view->getSourceFile() == sourceFile && !view->getInitiatingSourceLoc().isValid())
            {
                return view;
            }
        }
    }

    return nullptr;
}

static void _outputInclude(SourceFile* sourceFile, Index depth, DiagnosticSink* sink)
{
    StringBuilder buf;

    for (Index i = 0; i < depth; ++i)
    {
        buf << "  ";
    }

    // Output the found path for now
    // TODO(JS). We could use the verbose paths flag to control what path is output -> as it may be
    // useful to output the full path for example

    const PathInfo& pathInfo = sourceFile->getPathInfo();
    buf << "'" << pathInfo.foundPath << "'";

    // TODO(JS)?
    // You might want to know where this include was from.
    // If I output this though there will be a problem... as the indenting won't be clearly shown.
    // Perhaps I output in two sections, one the hierarchy and the other the locations of the
    // includes?

    sink->diagnose(SourceLoc(), Diagnostics::includeOutput, buf);
}

static void _outputIncludesRec(
    SourceView* sourceView,
    Index depth,
    ViewInitiatingHierarchy& hierarchy,
    DiagnosticSink* sink)
{
    SourceFile* sourceFile = sourceView->getSourceFile();
    const PathInfo& pathInfo = sourceFile->getPathInfo();

    switch (pathInfo.type)
    {
    case PathInfo::Type::TokenPaste:
    case PathInfo::Type::CommandLine:
    case PathInfo::Type::TypeParse:
        {
            // If any of these types we don't output
            return;
        }
    default:
        break;
    }

    // Okay output this file at the current depth
    _outputInclude(sourceFile, depth, sink);

    // Now recurse to all of the children at the next depth
    List<SourceView*>* children = hierarchy.tryGetValue(sourceView);
    if (children)
    {
        for (SourceView* child : *children)
        {
            _outputIncludesRec(child, depth + 1, hierarchy, sink);
        }
    }
}

static void _outputPreprocessorTokens(const TokenList& toks, ISlangWriter* writer)
{
    if (writer == nullptr)
    {
        return;
    }

    StringBuilder buf;
    for (const auto& tok : toks)
    {
        buf << tok.getContent();
        // We'll separate tokens with space for now
        buf.appendChar(' ');
    }

    buf.appendChar('\n');

    writer->write(buf.getBuffer(), buf.getLength());
}

static void _outputIncludes(
    const List<SourceFile*>& sourceFiles,
    SourceManager* sourceManager,
    DiagnosticSink* sink)
{
    // Set up the hierarchy to know how all the source views relate. This could be argued as
    // overkill, but makes recursive output pretty simple
    ViewInitiatingHierarchy hierarchy;
    _calcViewInitiatingHierarchy(sourceManager, hierarchy);

    // For all the source files
    for (SourceFile* sourceFile : sourceFiles)
    {
        // Find an initial view (this is the view of this file, that doesn't have an initiating loc)
        SourceView* sourceView = _findInitialSourceView(sourceFile);
        if (!sourceView)
        {
            // Okay, didn't find one, so just output the file
            _outputInclude(sourceFile, 0, sink);
        }
        else
        {
            // Output from this view recursively
            _outputIncludesRec(sourceView, 0, hierarchy, sink);
        }
    }
}

void FrontEndCompileRequest::parseTranslationUnit(TranslationUnitRequest* translationUnit)
{
    SLANG_PROFILE;
    if (translationUnit->isChecked)
        return;

    auto linkage = getLinkage();

    SLANG_AST_BUILDER_RAII(linkage->getASTBuilder());

    // TODO(JS): NOTE! Here we are using the searchDirectories on the linkage. This is because
    // currently the API only allows the setting search paths on linkage.
    //
    // Here we should probably be using the searchDirectories on the FrontEndCompileRequest.
    // If searchDirectories.parent pointed to the one in the Linkage would mean linkage paths
    // would be checked too (after those on the FrontEndCompileRequest).
    IncludeSystem includeSystem(
        &linkage->getSearchDirectories(),
        linkage->getFileSystemExt(),
        linkage->getSourceManager());

    auto combinedPreprocessorDefinitions = translationUnit->getCombinedPreprocessorDefinitions();

    auto module = translationUnit->getModule();

    ASTBuilder* astBuilder = module->getASTBuilder();

    ModuleDecl* translationUnitSyntax = astBuilder->create<ModuleDecl>();

    translationUnitSyntax->nameAndLoc.name = translationUnit->moduleName;
    translationUnitSyntax->module = module;
    module->setModuleDecl(translationUnitSyntax);

    // When compiling a module of code that belongs to the Slang
    // core module, we add a modifier to the module to act
    // as a marker, so that downstream code can detect declarations
    // that came from the core module (by walking up their
    // chain of ancestors and looking for the marker), and treat
    // them differently from user declarations.
    //
    // We are adding the marker here, before we even parse the
    // code in the module, in case the subsequent steps would
    // like to treat the core module differently. Alternatively
    // we could pass down the `m_isStandardLibraryCode` flag to
    // these passes.
    //
    if (m_isCoreModuleCode)
    {
        translationUnitSyntax->modifiers.first = astBuilder->create<FromCoreModuleModifier>();
    }

    // We use a custom handler for preprocessor callbacks, to
    // ensure that relevant state that is only visible during
    // preprocessoing can be communicated to later phases of
    // compilation.
    //
    FrontEndPreprocessorHandler preprocessorHandler(module, astBuilder, getSink());

    for (auto sourceFile : translationUnit->getSourceFiles())
    {
        module->getIncludedSourceFileMap().addIfNotExists(sourceFile, nullptr);
    }

    for (auto sourceFile : translationUnit->getSourceFiles())
    {
        SourceLanguage sourceLanguage = SourceLanguage::Unknown;
        auto tokens = preprocessSource(
            sourceFile,
            getSink(),
            &includeSystem,
            combinedPreprocessorDefinitions,
            getLinkage(),
            sourceLanguage,
            &preprocessorHandler);

        if (sourceLanguage == SourceLanguage::Unknown)
            sourceLanguage = translationUnit->sourceLanguage;

        Scope* languageScope = nullptr;
        switch (sourceLanguage)
        {
        case SourceLanguage::HLSL:
            languageScope = getSession()->hlslLanguageScope;
            break;
        case SourceLanguage::GLSL:
            languageScope = getSession()->glslLanguageScope;
            break;
        case SourceLanguage::Slang:
        default:
            languageScope = getSession()->slangLanguageScope;
            break;
        }

        if (optionSet.getBoolOption(CompilerOptionName::OutputIncludes))
        {
            _outputIncludes(
                translationUnit->getSourceFiles(),
                getSink()->getSourceManager(),
                getSink());
        }

        if (optionSet.getBoolOption(CompilerOptionName::PreprocessorOutput))
        {
            if (m_writers)
            {
                _outputPreprocessorTokens(
                    tokens,
                    m_writers->getWriter(SLANG_WRITER_CHANNEL_STD_OUTPUT));
            }
            // If we output the preprocessor output then we are done doing anything else
            return;
        }

        parseSourceFile(
            astBuilder,
            translationUnit,
            sourceLanguage,
            tokens,
            getSink(),
            languageScope,
            translationUnitSyntax);

        // Let's try dumping

        if (optionSet.getBoolOption(CompilerOptionName::DumpAst))
        {
            StringBuilder buf;
            SourceWriter writer(linkage->getSourceManager(), LineDirectiveMode::None, nullptr);

            ASTDumpUtil::dump(
                translationUnit->getModuleDecl(),
                ASTDumpUtil::Style::Flat,
                0,
                &writer);

            const String& path = sourceFile->getPathInfo().foundPath;
            if (path.getLength())
            {
                String fileName = Path::getFileNameWithoutExt(path);
                fileName.append(".slang-ast");

                File::writeAllText(fileName, writer.getContent());
            }
        }

#if 0
            // Test serialization
            {
                ASTSerialTestUtil::testSerialize(translationUnit->getModuleDecl(), getSession()->getRootNamePool(), getLinkage()->getASTBuilder()->getSharedASTBuilder(), getSourceManager());
            }
#endif
    }
}

RefPtr<ComponentType> createUnspecializedGlobalComponentType(
    FrontEndCompileRequest* compileRequest);

RefPtr<ComponentType> createUnspecializedGlobalAndEntryPointsComponentType(
    FrontEndCompileRequest* compileRequest,
    List<RefPtr<ComponentType>>& outUnspecializedEntryPoints);

RefPtr<ComponentType> createSpecializedGlobalComponentType(EndToEndCompileRequest* endToEndReq);

RefPtr<ComponentType> createSpecializedGlobalAndEntryPointsComponentType(
    EndToEndCompileRequest* endToEndReq,
    List<RefPtr<ComponentType>>& outSpecializedEntryPoints);

void FrontEndCompileRequest::checkAllTranslationUnits()
{
    SLANG_PROFILE;

    LoadedModuleDictionary loadedModules;
    if (additionalLoadedModules)
        loadedModules = *additionalLoadedModules;

    // Iterate over all translation units and
    // apply the semantic checking logic.
    for (auto& translationUnit : translationUnits)
    {
        if (translationUnit->isChecked)
            continue;

        checkTranslationUnit(translationUnit.Ptr(), loadedModules);

        // Add the checked module to list of loadedModules so that they can be
        // discovered by `findOrImportModule` when processing future `import` decls.
        // TODO: this does not handle the case where a translation unit to discover
        // another translation unit added later to the compilation request.
        // We should output an error message when we detect such a case, or support
        // this scenario with a recursive style checking.
        loadedModules.add(translationUnit->moduleName, translationUnit->getModule());
    }
    checkEntryPoints();
}

void FrontEndCompileRequest::generateIR()
{
    SLANG_PROFILE;
    SLANG_AST_BUILDER_RAII(getLinkage()->getASTBuilder());

    // Our task in this function is to generate IR code
    // for all of the declarations in the translation
    // units that were loaded.

    // Each translation unit is its own little world
    // for code generation (we are not trying to
    // replicate the GLSL linkage model), and so
    // we will generate IR for each (if needed)
    // in isolation.
    for (auto& translationUnit : translationUnits)
    {
        // Skip if the module is precompiled.
        if (translationUnit->getModule()->getIRModule())
            continue;

        // We want to only run generateIRForTranslationUnit once here. This is for two side effects:
        // * it can dump ir
        // * it can generate diagnostics

        /// Generate IR for translation unit.
        RefPtr<IRModule> irModule(
            generateIRForTranslationUnit(getLinkage()->getASTBuilder(), translationUnit));

        if (verifyDebugSerialization)
        {
            SerialContainerUtil::WriteOptions options;

            options.sourceManager = getSourceManager();
            options.optionFlags |= SerialOptionFlag::SourceLocation;

            // Verify debug information
            if (SLANG_FAILED(
                    SerialContainerUtil::verifyIRSerialize(irModule, getSession(), options)))
            {
                getSink()->diagnose(
                    irModule->getModuleInst()->sourceLoc,
                    Diagnostics::serialDebugVerificationFailed);
            }
        }

        if (useSerialIRBottleneck)
        {
            // Keep the obfuscated source map (if there is one)
            ComPtr<IBoxValue<SourceMap>> obfuscatedSourceMap(irModule->getObfuscatedSourceMap());

            IRSerialData serialData;
            {
                // Write IR out to serialData - copying over SourceLoc information directly
                IRSerialWriter writer;
                writer.write(irModule, nullptr, SerialOptionFlag::RawSourceLocation, &serialData);

                // Destroy irModule such that memory can be used for newly constructed read
                // irReadModule
                irModule = nullptr;
            }
            RefPtr<IRModule> irReadModule;
            {
                // Read IR back from serialData
                IRSerialReader reader;
                reader.read(serialData, getSession(), nullptr, irReadModule);
            }

            // Set irModule to the read module
            irModule = irReadModule;
            irModule->setObfuscatedSourceMap(obfuscatedSourceMap);
        }

        // Set the module on the translation unit
        translationUnit->getModule()->setIRModule(irModule);
    }
}

// Try to infer a single common source language for a request
static SourceLanguage inferSourceLanguage(FrontEndCompileRequest* request)
{
    SourceLanguage language = SourceLanguage::Unknown;
    for (auto& translationUnit : request->translationUnits)
    {
        // Allow any other language to overide Slang as a choice
        if (language == SourceLanguage::Unknown || language == SourceLanguage::Slang)
        {
            language = translationUnit->sourceLanguage;
        }
        else if (language == translationUnit->sourceLanguage)
        {
            // same language as we currently have, so keep going
        }
        else
        {
            // we found a mismatch, so inference fails
            return SourceLanguage::Unknown;
        }
    }
    return language;
}

SlangResult FrontEndCompileRequest::executeActionsInner()
{
    SLANG_PROFILE_SECTION(frontEndExecute);
    SLANG_AST_BUILDER_RAII(getLinkage()->getASTBuilder());

    for (TranslationUnitRequest* translationUnit : translationUnits)
    {
        // Make sure SourceFile representation is available for all translationUnits
        SLANG_RETURN_ON_FAIL(translationUnit->requireSourceFiles());
    }


    // Parse everything from the input files requested
    for (TranslationUnitRequest* translationUnit : translationUnits)
    {
        parseTranslationUnit(translationUnit);
    }

    if (optionSet.getBoolOption(CompilerOptionName::PreprocessorOutput))
    {
        // If doing pre-processor output, then we are done
        return SLANG_OK;
    }

    if (getSink()->getErrorCount() != 0)
        return SLANG_FAIL;

    // Perform semantic checking on the whole collection
    {
        SLANG_PROFILE_SECTION(SemanticChecking);
        checkAllTranslationUnits();
    }

    if (getSink()->getErrorCount() != 0)
        return SLANG_FAIL;

    // After semantic checking is performed we can try and output doc information for this
    if (optionSet.getBoolOption(CompilerOptionName::Doc))
    {
        // TODO: implement the logic to output generated documents to target directory/zip file.
    }

    // Look up all the entry points that are expected,
    // and use them to populate the `program` member.
    //
    m_globalComponentType = createUnspecializedGlobalComponentType(this);
    if (getSink()->getErrorCount() != 0)
        return SLANG_FAIL;

    m_globalAndEntryPointsComponentType =
        createUnspecializedGlobalAndEntryPointsComponentType(this, m_unspecializedEntryPoints);
    if (getSink()->getErrorCount() != 0)
        return SLANG_FAIL;

    // We always generate IR for all the translation units.
    //
    // TODO: We may eventually have a mode where we skip
    // IR codegen and only produce an AST (e.g., for use when
    // debugging problems in the parser or semantic checking),
    // but for now there are no cases where not having IR
    // makes sense.
    //
    generateIR();
    if (getSink()->getErrorCount() != 0)
        return SLANG_FAIL;

    // Do parameter binding generation, for each compilation target.
    //
    for (auto targetReq : getLinkage()->targets)
    {
        auto targetProgram = m_globalAndEntryPointsComponentType->getTargetProgram(targetReq);
        targetProgram->getOrCreateLayout(getSink());
        targetProgram->getOrCreateIRModuleForLayout(getSink());
    }
    if (getSink()->getErrorCount() != 0)
        return SLANG_FAIL;

    return SLANG_OK;
}

EndToEndCompileRequest::EndToEndCompileRequest(Session* session)
    : m_session(session), m_sink(nullptr, Lexer::sourceLocationLexer)
{
    RefPtr<ASTBuilder> astBuilder(
        new ASTBuilder(session->m_sharedASTBuilder, "EndToEnd::Linkage::astBuilder"));
    m_linkage = new Linkage(session, astBuilder, session->getBuiltinLinkage());
    init();
}

EndToEndCompileRequest::EndToEndCompileRequest(Linkage* linkage)
    : m_session(linkage->getSessionImpl())
    , m_linkage(linkage)
    , m_sink(nullptr, Lexer::sourceLocationLexer)
{
    init();
}

SLANG_NO_THROW SlangResult SLANG_MCALL
EndToEndCompileRequest::queryInterface(SlangUUID const& uuid, void** outObject)
{
    if (uuid == EndToEndCompileRequest::getTypeGuid())
    {
        // Special case to cast directly into internal type
        // NOTE! No addref(!)
        *outObject = this;
        return SLANG_OK;
    }

    if (uuid == ISlangUnknown::getTypeGuid() && uuid == ICompileRequest::getTypeGuid())
    {
        addReference();
        *outObject = static_cast<slang::ICompileRequest*>(this);
        return SLANG_OK;
    }

    return SLANG_E_NO_INTERFACE;
}

void EndToEndCompileRequest::init()
{
    m_sink.setSourceManager(m_linkage->getSourceManager());

    m_writers = new StdWriters;

    // Set all the default writers
    for (int i = 0; i < int(WriterChannel::CountOf); ++i)
    {
        setWriter(WriterChannel(i), nullptr);
    }

    m_frontEndReq = new FrontEndCompileRequest(getLinkage(), m_writers, getSink());
}

SlangResult EndToEndCompileRequest::executeActionsInner()
{
    SLANG_PROFILE_SECTION(endToEndActions);
    // If no code-generation target was specified, then try to infer one from the source language,
    // just to make sure we can do something reasonable when invoked from the command line.
    //
    // TODO: This logic should be moved into `options.cpp` or somewhere else
    // specific to the command-line tool.
    //
    if (getLinkage()->targets.getCount() == 0)
    {
        auto language = inferSourceLanguage(getFrontEndReq());
        switch (language)
        {
        case SourceLanguage::HLSL:
            getLinkage()->addTarget(CodeGenTarget::DXBytecode);
            break;

        case SourceLanguage::GLSL:
            getLinkage()->addTarget(CodeGenTarget::SPIRV);
            break;

        default:
            break;
        }
    }

    // Update compiler settings in target requests.
    for (auto target : getLinkage()->targets)
        target->getOptionSet().inheritFrom(getOptionSet());
    m_frontEndReq->optionSet = getOptionSet();

    // We only do parsing and semantic checking if we *aren't* doing
    // a pass-through compilation.
    //
    if (m_passThrough == PassThroughMode::None)
    {
        SLANG_RETURN_ON_FAIL(getFrontEndReq()->executeActionsInner());
    }

    if (getOptionSet().getBoolOption(CompilerOptionName::PreprocessorOutput))
    {
        return SLANG_OK;
    }

    // If command line specifies to skip codegen, we exit here.
    // Note: this is a debugging option.
    //
    if (getOptionSet().getBoolOption(CompilerOptionName::SkipCodeGen))
    {
        // We will use the program (and matching layout information)
        // that was computed in the front-end for all subsequent
        // reflection queries, etc.
        //
        m_specializedGlobalComponentType = getUnspecializedGlobalComponentType();
        m_specializedGlobalAndEntryPointsComponentType =
            getUnspecializedGlobalAndEntryPointsComponentType();
        m_specializedEntryPoints = getFrontEndReq()->getUnspecializedEntryPoints();

        SLANG_RETURN_ON_FAIL(maybeCreateContainer());

        SLANG_RETURN_ON_FAIL(maybeWriteContainer(m_containerOutputPath));

        return SLANG_OK;
    }

    // If requested, attempt to compile the translation unit all the way down to the target
    // language(s) and stash the result blobs in IR.
    for (auto target : getLinkage()->targets)
    {
        SlangCompileTarget targetEnum = SlangCompileTarget(target->getTarget());
        if (target->getOptionSet().getBoolOption(CompilerOptionName::EmbedDownstreamIR))
        {
            auto frontEndReq = getFrontEndReq();

            for (auto translationUnit : frontEndReq->translationUnits)
            {
                SLANG_RETURN_ON_FAIL(
                    translationUnit->getModule()->precompileForTarget(targetEnum, nullptr));

                if (frontEndReq->optionSet.shouldDumpIR())
                {
                    DiagnosticSinkWriter writer(frontEndReq->getSink());

                    dumpIR(
                        translationUnit->getModule()->getIRModule(),
                        frontEndReq->m_irDumpOptions,
                        "PRECOMPILE_FOR_TARGET_COMPLETE_ALL",
                        frontEndReq->getSourceManager(),
                        &writer);

                    dumpIR(
                        translationUnit->getModule()->getIRModule()->getModuleInst(),
                        frontEndReq->m_irDumpOptions,
                        frontEndReq->getSourceManager(),
                        &writer);
                }
            }
        }
    }

    // If codegen is enabled, we need to move along to
    // apply any generic specialization that the user asked for.
    //
    if (m_passThrough == PassThroughMode::None)
    {
        m_specializedGlobalComponentType = createSpecializedGlobalComponentType(this);
        if (getSink()->getErrorCount() != 0)
            return SLANG_FAIL;

        m_specializedGlobalAndEntryPointsComponentType =
            createSpecializedGlobalAndEntryPointsComponentType(this, m_specializedEntryPoints);
        if (getSink()->getErrorCount() != 0)
            return SLANG_FAIL;

        // For each code generation target, we will generate specialized
        // parameter binding information (taking global generic
        // arguments into account at this time).
        //
        for (auto targetReq : getLinkage()->targets)
        {
            auto targetProgram =
                m_specializedGlobalAndEntryPointsComponentType->getTargetProgram(targetReq);
            targetProgram->getOrCreateLayout(getSink());
        }
        if (getSink()->getErrorCount() != 0)
            return SLANG_FAIL;
    }
    else
    {
        // We need to create dummy `EntryPoint` objects
        // to make sure that the logic in `generateOutput`
        // sees something worth processing.
        //
        List<RefPtr<ComponentType>> dummyEntryPoints;
        for (auto entryPointReq : getFrontEndReq()->getEntryPointReqs())
        {
            RefPtr<EntryPoint> dummyEntryPoint = EntryPoint::createDummyForPassThrough(
                getLinkage(),
                entryPointReq->getName(),
                entryPointReq->getProfile());

            dummyEntryPoints.add(dummyEntryPoint);
        }

        RefPtr<ComponentType> composedProgram =
            CompositeComponentType::create(getLinkage(), dummyEntryPoints);

        m_specializedGlobalComponentType = getUnspecializedGlobalComponentType();
        m_specializedGlobalAndEntryPointsComponentType = composedProgram;
        m_specializedEntryPoints = getFrontEndReq()->getUnspecializedEntryPoints();
    }

    // Generate output code, in whatever format was requested
    generateOutput();
    if (getSink()->getErrorCount() != 0)
        return SLANG_FAIL;

    return SLANG_OK;
}

// Act as expected of the API-based compiler
SlangResult EndToEndCompileRequest::executeActions()
{
    SlangResult res = executeActionsInner();

    m_diagnosticOutput = getSink()->outputBuffer.produceString();
    return res;
}

int FrontEndCompileRequest::addTranslationUnit(SourceLanguage language, Name* moduleName)
{
    RefPtr<TranslationUnitRequest> translationUnit = new TranslationUnitRequest(this);
    translationUnit->compileRequest = this;
    translationUnit->sourceLanguage = SourceLanguage(language);

    translationUnit->setModuleName(moduleName);
    return addTranslationUnit(translationUnit);
}

int FrontEndCompileRequest::addTranslationUnit(TranslationUnitRequest* translationUnit)
{
    Index result = translationUnits.getCount();
    translationUnits.add(translationUnit);
    return (int)result;
}

void FrontEndCompileRequest::addTranslationUnitSourceArtifact(
    int translationUnitIndex,
    IArtifact* sourceArtifact)
{
    auto translationUnit = translationUnits[translationUnitIndex];

    // Add the source file
    translationUnit->addSourceArtifact(sourceArtifact);

    if (!translationUnit->moduleName)
    {
        translationUnit->setModuleName(
            getNamePool()->getName(Path::getFileNameWithoutExt(sourceArtifact->getName())));
    }
    if (translationUnit->module->getFilePath() == nullptr)
        translationUnit->module->setPathInfo(PathInfo::makePath(sourceArtifact->getName()));
}

void FrontEndCompileRequest::addTranslationUnitSourceBlob(
    int translationUnitIndex,
    String const& path,
    ISlangBlob* sourceBlob)
{
    auto translationUnit = translationUnits[translationUnitIndex];
    auto sourceDesc =
        ArtifactDescUtil::makeDescForSourceLanguage(asExternal(translationUnit->sourceLanguage));

    auto artifact = ArtifactUtil::createArtifact(sourceDesc, path.getBuffer());
    artifact->addRepresentationUnknown(sourceBlob);

    addTranslationUnitSourceArtifact(translationUnitIndex, artifact);
}

void FrontEndCompileRequest::addTranslationUnitSourceFile(
    int translationUnitIndex,
    String const& path)
{
    // TODO: We need to consider whether a relative `path` should cause
    // us to look things up using the registered search paths.
    //
    // This behavior wouldn't make sense for command-line invocations
    // of `slangc`, but at least one API user wondered by the search
    // paths were not taken into account by this function.
    //

    auto fileSystemExt = getLinkage()->getFileSystemExt();
    auto translationUnit = getTranslationUnit(translationUnitIndex);

    auto sourceDesc =
        ArtifactDescUtil::makeDescForSourceLanguage(asExternal(translationUnit->sourceLanguage));

    auto sourceArtifact = ArtifactUtil::createArtifact(sourceDesc, path.getBuffer());

    auto extRep = new ExtFileArtifactRepresentation(path.getUnownedSlice(), fileSystemExt);
    sourceArtifact->addRepresentation(extRep);

    SlangResult existsRes = SLANG_OK;

    // If we require caching, we demand it's loaded here.
    //
    // In practice this probably means repro capture is enabled. So we want to
    // load the blob such that it's in the cache, even if it doesn't actually
    // have to be loaded for the compilation.
    if (getLinkage()->m_requireCacheFileSystem)
    {
        ComPtr<ISlangBlob> blob;
        // If we can load the blob, then it exists
        existsRes = sourceArtifact->loadBlob(ArtifactKeep::Yes, blob.writeRef());
    }
    else
    {
        existsRes = sourceArtifact->exists() ? SLANG_OK : SLANG_E_NOT_FOUND;
    }

    if (SLANG_FAILED(existsRes))
    {
        // Emit a diagnostic!
        getSink()->diagnose(SourceLoc(), Diagnostics::cannotOpenFile, path);
        return;
    }

    addTranslationUnitSourceArtifact(translationUnitIndex, sourceArtifact);
}

int FrontEndCompileRequest::addEntryPoint(
    int translationUnitIndex,
    String const& name,
    Profile entryPointProfile)
{
    auto translationUnitReq = translationUnits[translationUnitIndex];

    Index result = m_entryPointReqs.getCount();

    RefPtr<FrontEndEntryPointRequest> entryPointReq = new FrontEndEntryPointRequest(
        this,
        translationUnitIndex,
        getNamePool()->getName(name),
        entryPointProfile);

    m_entryPointReqs.add(entryPointReq);
    //    translationUnitReq->entryPoints.add(entryPointReq);

    return int(result);
}

int EndToEndCompileRequest::addEntryPoint(
    int translationUnitIndex,
    String const& name,
    Profile entryPointProfile,
    List<String> const& genericTypeNames)
{
    getFrontEndReq()->addEntryPoint(translationUnitIndex, name, entryPointProfile);

    EntryPointInfo entryPointInfo;
    for (auto typeName : genericTypeNames)
        entryPointInfo.specializationArgStrings.add(typeName);

    Index result = m_entryPoints.getCount();
    m_entryPoints.add(_Move(entryPointInfo));
    return (int)result;
}

UInt Linkage::addTarget(CodeGenTarget target)
{
    RefPtr<TargetRequest> targetReq = new TargetRequest(this, target);

    Index result = targets.getCount();
    targets.add(targetReq);
    return UInt(result);
}

void Linkage::loadParsedModule(
    RefPtr<FrontEndCompileRequest> compileRequest,
    RefPtr<TranslationUnitRequest> translationUnit,
    Name* name,
    const PathInfo& pathInfo)
{
    // Note: we add the loaded module to our name->module listing
    // before doing semantic checking, so that if it tries to
    // recursively `import` itself, we can detect it.
    //
    RefPtr<Module> loadedModule = translationUnit->getModule();

    // Get a path
    String mostUniqueIdentity = pathInfo.getMostUniqueIdentity();
    SLANG_ASSERT(mostUniqueIdentity.getLength() > 0);

    mapPathToLoadedModule.add(mostUniqueIdentity, loadedModule);
    mapNameToLoadedModules.add(name, loadedModule);

    auto sink = translationUnit->compileRequest->getSink();

    int errorCountBefore = sink->getErrorCount();
    compileRequest->checkAllTranslationUnits();
    int errorCountAfter = sink->getErrorCount();
    if (isInLanguageServer())
    {
        // Don't generate IR as language server.
        // This means that we currently cannot report errors that are detected during IR passes.
        // Ideally we want to run those passes, but that is too risky for what it is worth right
        // now.
    }
    else
    {
        if (errorCountAfter != errorCountBefore)
        {
            // There must have been an error in the loaded module.
        }
        else
        {
            // If we didn't run into any errors, then try to generate
            // IR code for the imported module.
            if (errorCountAfter == 0)
            {
                loadedModule->setIRModule(
                    generateIRForTranslationUnit(getASTBuilder(), translationUnit));
            }
        }
    }
    loadedModulesList.add(loadedModule);
}

RefPtr<Module> Linkage::findOrLoadSerializedModuleForModuleLibrary(
    ModuleChunkRef moduleChunk,
    DiagnosticSink* sink)
{
    RefPtr<Module> resultModule;

    // We will attempt things in a few different steps, trying to
    // decode as little of the serialized module as necessary at
    // each step, so that we don't waste time on the heavyweight
    // stuff when we didn't need to.
    //
    // The first step is to simply decode the module name, and
    // see if we have a already loaded a matching module.

    auto moduleName = getNamePool()->getName(moduleChunk.getName());
    if (mapNameToLoadedModules.tryGetValue(moduleName, resultModule))
        return resultModule;

    // It is possible that the module has been loaded, but somehow
    // under a different name, so next we decode the list of file
    // paths that the module depends on, and then rely on the assumption
    // that the first of those paths represents the file for the module
    // itself to detect if we've already loaded a module from that
    // path.
    //
    // Note: While this is a distasteful assumption to make, it is
    // one that gets made in several parts of the compiler codebase
    // already. It isn't something that can be fixed in just one
    // place at this point.

    auto fileDependenciesChunk = moduleChunk.getFileDependencies();
    auto firstFileDependencyChunk = fileDependenciesChunk.getFirst();
    if (!firstFileDependencyChunk)
        return nullptr;

    auto modulePathInfo = PathInfo::makePath(firstFileDependencyChunk.getValue());
    if (mapPathToLoadedModule.tryGetValue(modulePathInfo.getMostUniqueIdentity(), resultModule))
        return resultModule;

    // If we failed to find a previously-loaded module, then we
    // will go ahead and load the module from the serialized form.
    //
    PathInfo filePathInfo;
    return loadSerializedModule(moduleName, modulePathInfo, moduleChunk, SourceLoc(), sink);
}

RefPtr<Module> Linkage::loadSerializedModule(
    Name* moduleName,
    const PathInfo& moduleFilePathInfo,
    ModuleChunkRef moduleChunk,
    SourceLoc const& requestingLoc,
    DiagnosticSink* sink)
{
    auto astBuilder = getASTBuilder();
    SLANG_AST_BUILDER_RAII(astBuilder);

    auto module = RefPtr(new Module(this, astBuilder));
    module->setName(moduleName);

    // Just as if we were processing an `import` declaration in
    // source code, we will track the fact that this serialized
    // modlue is (effectively) being imported, so that we can
    // diagnose anything troublesome, like an attempt at a
    // recursive import.
    //
    ModuleBeingImportedRAII moduleBeingImported(this, module, moduleName, requestingLoc);

    // We will register the module in our data structures to
    // track loaded modules, and then remove it in the case
    // where there is some kind of failure.
    //
    String mostUniqueIdentity = moduleFilePathInfo.getMostUniqueIdentity();
    SLANG_ASSERT(mostUniqueIdentity.getLength() > 0);

    mapPathToLoadedModule.add(mostUniqueIdentity, module);
    mapNameToLoadedModules.add(moduleName, module);
    try
    {
        if (SLANG_FAILED(
                loadSerializedModuleContents(module, moduleFilePathInfo, moduleChunk, sink)))
        {
            mapPathToLoadedModule.remove(mostUniqueIdentity);
            mapNameToLoadedModules.remove(moduleName);
            return nullptr;
        }

        loadedModulesList.add(module);
        return module;
    }
    catch (...)
    {
        mapPathToLoadedModule.remove(mostUniqueIdentity);
        mapNameToLoadedModules.remove(moduleName);
        throw;
    }
}

RefPtr<Module> Linkage::loadBinaryModuleImpl(
    Name* moduleName,
    const PathInfo& moduleFilePathInfo,
    ISlangBlob* moduleFileContents,
    SourceLoc const& requestingLoc,
    DiagnosticSink* sink)
{
    auto astBuilder = getASTBuilder();
    SLANG_AST_BUILDER_RAII(astBuilder);

    // We start by reading the content of the file into
    // an in-memory RIFF container.
    //
    // TODO(tfoley): this is an unnecessary copy step, since
    // we can simply use the contents of the blob directly
    // and navigate it in-memory.
    //
    RiffContainer riffContainer;
    {
        MemoryStreamBase readStream(
            FileAccess::Read,
            moduleFileContents->getBufferPointer(),
            moduleFileContents->getBufferSize());
        SLANG_RETURN_NULL_ON_FAIL(RiffUtil::read(&readStream, riffContainer));
    }

    auto moduleChunkRef = ModuleChunkRef::find(&riffContainer);
    if (!moduleChunkRef)
    {
        return nullptr;
    }

    // Next, we attempt to check if the binary module is up to
    // date with the compilation options in use as well as
    // the contents of all the files its compilation depended
    // on (as determined by its hash).
    //
    String mostUniqueIdentity = moduleFilePathInfo.getMostUniqueIdentity();
    SLANG_ASSERT(mostUniqueIdentity.getLength() > 0);
    if (m_optionSet.getBoolOption(CompilerOptionName::UseUpToDateBinaryModule))
    {
        if (!isBinaryModuleUpToDate(moduleFilePathInfo.foundPath, moduleChunkRef))
        {
            return nullptr;
        }
    }

    // If everything seems reasonable, then we will go ahead and load
    // the module more completely from that serialized representation.
    //
    RefPtr<Module> module =
        loadSerializedModule(moduleName, moduleFilePathInfo, moduleChunkRef, requestingLoc, sink);

    return module;
}

void Linkage::_diagnoseErrorInImportedModule(DiagnosticSink* sink)
{
    for (auto info = m_modulesBeingImported; info; info = info->next)
    {
        sink->diagnose(info->importLoc, Diagnostics::errorInImportedModule, info->name);
    }
    if (!isInLanguageServer())
    {
        sink->diagnose(SourceLoc(), Diagnostics::complationCeased);
    }
}

RefPtr<Module> Linkage::loadModuleImpl(
    Name* moduleName,
    const PathInfo& modulePathInfo,
    ISlangBlob* moduleBlob,
    SourceLoc const& requestingLoc,
    DiagnosticSink* sink,
    const LoadedModuleDictionary* additionalLoadedModules,
    ModuleBlobType blobType)
{
    switch (blobType)
    {
    case ModuleBlobType::IR:
        return loadBinaryModuleImpl(moduleName, modulePathInfo, moduleBlob, requestingLoc, sink);

    case ModuleBlobType::Source:
        return loadSourceModuleImpl(
            moduleName,
            modulePathInfo,
            moduleBlob,
            requestingLoc,
            sink,
            additionalLoadedModules);

    default:
        SLANG_UNEXPECTED("unknown module blob type");
        UNREACHABLE_RETURN(nullptr);
    }
}

RefPtr<Module> Linkage::loadSourceModuleImpl(
    Name* name,
    const PathInfo& filePathInfo,
    ISlangBlob* sourceBlob,
    SourceLoc const& srcLoc,
    DiagnosticSink* sink,
    const LoadedModuleDictionary* additionalLoadedModules)
{
    RefPtr<FrontEndCompileRequest> frontEndReq = new FrontEndCompileRequest(this, nullptr, sink);

    frontEndReq->additionalLoadedModules = additionalLoadedModules;

    RefPtr<TranslationUnitRequest> translationUnit = new TranslationUnitRequest(frontEndReq);
    translationUnit->compileRequest = frontEndReq;
    translationUnit->setModuleName(name);
    Stage impliedStage;
    translationUnit->sourceLanguage = SourceLanguage::Slang;

    // If we are loading from a file with apparaent glsl extension,
    // set the source language to GLSL to enable GLSL compatibility mode.
    if ((SourceLanguage)findSourceLanguageFromPath(filePathInfo.getName(), impliedStage) ==
        SourceLanguage::GLSL)
    {
        translationUnit->sourceLanguage = SourceLanguage::GLSL;
    }

    frontEndReq->addTranslationUnit(translationUnit);

    auto module = translationUnit->getModule();

    ModuleBeingImportedRAII moduleBeingImported(this, module, name, srcLoc);

    // Create an artifact for the source
    auto sourceArtifact = ArtifactUtil::createArtifact(
        ArtifactDesc::make(ArtifactKind::Source, ArtifactPayload::Slang, ArtifactStyle::Unknown));

    if (sourceBlob)
    {
        // If the user has already provided a source blob, use that.
        sourceArtifact->addRepresentation(
            new SourceBlobWithPathInfoArtifactRepresentation(filePathInfo, sourceBlob));
    }
    else if (
        filePathInfo.type == PathInfo::Type::Normal ||
        filePathInfo.type == PathInfo::Type::FoundPath)
    {
        // Create with the 'friendly' name
        // We create that it was loaded from the file system
        sourceArtifact->addRepresentation(new ExtFileArtifactRepresentation(
            filePathInfo.foundPath.getUnownedSlice(),
            getFileSystemExt()));
    }
    else
    {
        return nullptr;
    }

    translationUnit->addSourceArtifact(sourceArtifact);

    if (SLANG_FAILED(translationUnit->requireSourceFiles()))
    {
        // Some problem accessing source files
        return nullptr;
    }
    int errorCountBefore = sink->getErrorCount();
    frontEndReq->parseTranslationUnit(translationUnit);
    int errorCountAfter = sink->getErrorCount();

    if (errorCountAfter != errorCountBefore && !isInLanguageServer())
    {
        _diagnoseErrorInImportedModule(sink);
    }
    if (errorCountAfter && !isInLanguageServer())
    {
        // Something went wrong during the parsing, so we should bail out.
        return nullptr;
    }

    try
    {
        loadParsedModule(frontEndReq, translationUnit, name, filePathInfo);
    }
    catch (const Slang::AbortCompilationException&)
    {
        // Something is fatally wrong, we should return nullptr.
        module = nullptr;
    }
    errorCountAfter = sink->getErrorCount();

    if (errorCountAfter != errorCountBefore && !isInLanguageServer())
    {
        // If something is fatally wrong, we want to report
        // the diagnostic even if we are in language server
        // and processing a different module.
        _diagnoseErrorInImportedModule(sink);
        // Something went wrong during the parsing, so we should bail out.
        return nullptr;
    }

    if (!module)
        return nullptr;

    module->setPathInfo(filePathInfo);
    return module;
}

bool Linkage::isBeingImported(Module* module)
{
    for (auto ii = m_modulesBeingImported; ii; ii = ii->next)
    {
        if (module == ii->module)
            return true;
    }
    return false;
}

// Derive a file name for the module, by taking the given
// identifier, replacing all occurrences of `_` with `-`,
// and then appending `.slang`.
//
// For example, `foo_bar` becomes `foo-bar.slang`.
String getFileNameFromModuleName(Name* name, bool translateUnderScore)
{
    String fileName;
    if (!getText(name).getUnownedSlice().endsWithCaseInsensitive(".slang"))
    {
        StringBuilder sb;
        for (auto c : getText(name))
        {
            if (translateUnderScore && c == '_')
                c = '-';

            sb.append(c);
        }
        sb.append(".slang");
        fileName = sb.produceString();
    }
    else
    {
        fileName = getText(name);
    }
    return fileName;
}

RefPtr<Module> Linkage::findOrImportModule(
    Name* moduleName,
    SourceLoc const& requestingLoc,
    DiagnosticSink* sink,
    const LoadedModuleDictionary* loadedModules)
{
    // Have we already loaded a module matching this name?
    //
    RefPtr<LoadedModule> previouslyLoadedModule;
    if (mapNameToLoadedModules.tryGetValue(moduleName, previouslyLoadedModule))
    {
        // If the map shows a null module having been loaded,
        // then that means there was a prior load attempt,
        // but it failed, so we won't bother trying again.
        //
        if (!previouslyLoadedModule)
            return nullptr;

        // If state shows us that the module is already being
        // imported deeper on the call stack, then we've
        // hit a recursive case, and that is an error.
        //
        if (isBeingImported(previouslyLoadedModule))
        {
            // We seem to be in the middle of loading this module
            sink->diagnose(requestingLoc, Diagnostics::recursiveModuleImport, moduleName);
            return nullptr;
        }

        return previouslyLoadedModule;
    }

    // If the user is providing an additional list of loaded modules, we find
    // if the module being imported is in that list. This allows a translation
    // unit to use previously checked translation units in the same
    // FrontEndCompileRequest.
    {
        Module* previouslyLoadedLocalModule = nullptr;
        if (loadedModules && loadedModules->tryGetValue(moduleName, previouslyLoadedLocalModule))
        {
            return previouslyLoadedLocalModule;
        }
    }

    // If the name being requested matches the name of a built-in module,
    // then we will special-case the process by loading that builtin
    // module directly.
    //
    // TODO: right now this logic is only considering the built-in `glsl`
    // module, but it should probably be generalized so that we can more
    // easily support having multiple built-in modules rather than just
    // putting everything into `core`.
    //
    if (moduleName == getSessionImpl()->glslModuleName)
    {
        // This is a builtin glsl module, just load it from embedded definition.
        auto glslModule = getSessionImpl()->getBuiltinModule(slang::BuiltinModuleName::GLSL);
        if (!glslModule)
        {
            // Note: the way this logic is currently written, if the built-in
            // `glsl` module fails to load, then we will *not* fall back to
            // searching for a user-defined module in a file like `glsl.slang`.
            //
            // It is unclear if this should be the default behavior or not.
            // Should built-in modules be prioritized over user modules?
            // Should built-in modules shadow user modules, even when the
            // built-in module fails to load, for some reason?
            //
            sink->diagnose(requestingLoc, Diagnostics::glslModuleNotAvailable, moduleName);
        }
        return glslModule;
    }

    // We are going to use a loop to search for a suitable file to
    // load the module from, to account for a few key choices:
    //
    // * We can both load modules from a source `.slang` file,
    //   or from a binary `.slang-module` file.
    //
    // * For a variety of reasons, the `import` logic has historically
    //   translated underscores in a module name into dashes (so that
    //   `import my_module` will look for `my-module.slang`), and we
    //   try to support both that convention as well as a convention
    //   that preserves underscores.
    //
    // To try to keep this logic as orthogonal as possible, we first
    // construct lists of the options we want to iterate over, and
    // then do the actual loop later.

    ShortList<ModuleBlobType, 2> typesToTry;
    if (isInLanguageServer())
    {
        // When in language server, we always prefer to use source module if it is available.
        typesToTry.add(ModuleBlobType::Source);
        typesToTry.add(ModuleBlobType::IR);
    }
    else
    {
        // Look for a precompiled module first, if not exist, load from source.
        typesToTry.add(ModuleBlobType::IR);
        typesToTry.add(ModuleBlobType::Source);
    }

    // We will always search for a file name that directly matches the
    // module name as written first, and then search for one with
    // underscores replaced by dashes. The latter is the original
    // behavior that `import` provided, but it seems safest to prefer
    // the exact name spelled in the user's code when there might
    // actually be ambiguity.
    //
    auto defaultSourceFileName = getFileNameFromModuleName(moduleName, false);
    auto alternativeSourceFileName = getFileNameFromModuleName(moduleName, true);
    String sourceFileNamesToTry[] = {defaultSourceFileName, alternativeSourceFileName};

    // We are going to look for the candidate file using the same
    // logic that would be used for a preprocessor `#include`,
    // so we set up the necessary state.
    //
    IncludeSystem includeSystem(&getSearchDirectories(), getFileSystemExt(), getSourceManager());

    // Just like with a `#include`, the search will take into
    // account the path to the file where the request to import
    // this module came from (e.g. the source file with the
    // `import` declaration), if such a path is available.
    //
    PathInfo requestingPathInfo =
        getSourceManager()->getPathInfo(requestingLoc, SourceLocType::Actual);

    for (auto type : typesToTry)
    {
        for (auto sourceFileName : sourceFileNamesToTry)
        {
            // The `sourceFileName` will have the `.slang` extension,
            // so if we are looking for a binary module, we need
            // to change the extension we will look for.
            //
            String fileName;
            switch (type)
            {
            case ModuleBlobType::Source:
                fileName = sourceFileName;
                break;

            case ModuleBlobType::IR:
                fileName = Path::replaceExt(sourceFileName, "slang-module");
                break;
            }

            // We now search for a file matching the desired name,
            // using the same logic as for a `#include`.
            //
            // TODO: We might want to consider how to handle the case
            // of an `import` with a relative path a little specially,
            // since it could in theory be possible for two `.slang`
            // files with the same base name to exist in different
            // directories in a project, and we'd want file-relative
            // `import`s to work for each, without having either one
            // be able to "claim" the bare identifier of the base
            // name for itself.
            //
            PathInfo filePathInfo;
            if (SLANG_FAILED(
                    includeSystem.findFile(fileName, requestingPathInfo.foundPath, filePathInfo)))
            {
                // If we failed to find the file at this step, we
                // will continue the search for our other options.
                //
                continue;
            }

            // We will *again* search for a previously loaded module.
            //
            // It is possible that the same file will have been loaded
            // as a module under two different module names. The easiest
            // way for this to happen is if there are `import` declarations
            // using both the underscore and dash conventions (e.g., both
            // `import "my-module.slang"` and `import my_module`).
            //
            // This case may also arise if one file `import`s a module using
            // just an identifier for its name, but another `import`s it
            // using a path (e.g., `import "subdir/file.slang"`).
            //
            // No matter how the situation arises, we only want to have one
            // copy of the "same" module loaded at a given time, so we
            // will re-use the existing module if we find one here.
            //
            if (mapPathToLoadedModule.tryGetValue(
                    filePathInfo.getMostUniqueIdentity(),
                    previouslyLoadedModule))
            {
                // TODO: If we find a previously-loaded module at this step,
                // then we should probably register that module under the
                // given `moduleName` in the map of loaded modules, so
                // that subsequent `import`s using the same form will find it.
                //
                return previouslyLoadedModule;
            }

            // Now we try to load the content of the file.
            //
            // If for some reason we could find a file at the
            // given path, but for some reason couldn't *open*
            // and *read* it, then we continue the search
            // using whatever other candidate file names are left.
            //
            ComPtr<ISlangBlob> fileContents;
            if (SLANG_FAILED(includeSystem.loadFile(filePathInfo, fileContents)))
            {
                continue;
            }

            // If we found a real file and were able to load its contents,
            // then we'll go ahead and try to load a module from it,
            // whether by compiling it or decoding the binary.
            //
            auto module = loadModuleImpl(
                moduleName,
                filePathInfo,
                fileContents,
                requestingLoc,
                sink,
                loadedModules,
                type);

            // If the attempt to load the module from the given path
            // was successful, we go ahead and use it, without trying
            // out any other options.
            //
            if (module)
                return module;
        }
    }

    // If we tried out all of our candidate file names
    // and failed with each of them, then we diagnose
    // an error based on the original *source* file
    // name.
    //
    // TODO: this should really be an error message
    // that clearly states something like "no file
    // suitable for module `whatever` was found
    // and loaded.
    //
    // Ideally that error message would include whatever
    // of the candidate file names from the loop above
    // got furthest along in the process (or just a
    // list of the file names that were tried, if
    // nothing was even found via the include system).
    //
    sink->diagnose(requestingLoc, Diagnostics::cannotOpenFile, defaultSourceFileName);

    // If the attempt to import the module failed, then
    // we will stick a null pointer into the map of loaded
    // modules, so that subsequent attempts to load a module
    // with this name will return null without having to
    // go through all the above steps yet again.
    //
    mapNameToLoadedModules[moduleName] = nullptr;
    return nullptr;
}

SourceFile* Linkage::loadSourceFile(String pathFrom, String path)
{
    IncludeSystem includeSystem(&getSearchDirectories(), getFileSystemExt(), getSourceManager());
    ComPtr<slang::IBlob> blob;
    PathInfo pathInfo;
    SLANG_RETURN_NULL_ON_FAIL(includeSystem.findFile(path, pathFrom, pathInfo));
    SourceFile* sourceFile = nullptr;
    SLANG_RETURN_NULL_ON_FAIL(includeSystem.loadFile(pathInfo, blob, sourceFile));
    return sourceFile;
}

// Check if a serialized module is up-to-date with current compiler options and source files.
bool Linkage::isBinaryModuleUpToDate(String fromPath, RiffContainer* riffContainer)
{
    auto moduleChunk = ModuleChunkRef::find(riffContainer);
    if (!moduleChunk)
        return false;

    return isBinaryModuleUpToDate(fromPath, moduleChunk);
}

bool Linkage::isBinaryModuleUpToDate(String fromPath, ModuleChunkRef moduleChunk)
{
    SHA1::Digest existingDigest = moduleChunk.getDigest();

    DigestBuilder<SHA1> digestBuilder;
    auto version = String(getBuildTagString());
    digestBuilder.append(version);
    m_optionSet.buildHash(digestBuilder);

    // Find the canonical path of the directory containing the module source file.
    String moduleSrcPath = "";

    auto dependencyChunks = moduleChunk.getFileDependencies();
    if (auto firstDependencyChunk = dependencyChunks.getFirst())
    {
        moduleSrcPath = firstDependencyChunk.getValue();

        IncludeSystem includeSystem(
            &getSearchDirectories(),
            getFileSystemExt(),
            getSourceManager());
        PathInfo modulePathInfo;
        if (SLANG_SUCCEEDED(includeSystem.findFile(moduleSrcPath, fromPath, modulePathInfo)))
        {
            moduleSrcPath = modulePathInfo.foundPath;
            Path::getCanonical(moduleSrcPath, moduleSrcPath);
        }
    }

    for (auto dependencyChunk : dependencyChunks)
    {
        auto file = dependencyChunk.getValue();
        auto sourceFile = loadSourceFile(fromPath, file);
        if (!sourceFile)
        {
            // If we cannot find the source file from `fromPath`,
            // try again from the module's source file path.
            if (dependencyChunks.getFirst())
                sourceFile = loadSourceFile(moduleSrcPath, file);
        }
        if (!sourceFile)
            return false;
        digestBuilder.append(sourceFile->getDigest());
    }
    return digestBuilder.finalize() == existingDigest;
}

SLANG_NO_THROW bool SLANG_MCALL
Linkage::isBinaryModuleUpToDate(const char* modulePath, slang::IBlob* binaryModuleBlob)
{
    RiffContainer container;
    MemoryStreamBase readStream(
        FileAccess::Read,
        binaryModuleBlob->getBufferPointer(),
        binaryModuleBlob->getBufferSize());
    if (SLANG_FAILED(RiffUtil::read(&readStream, container)))
        return false;
    return isBinaryModuleUpToDate(modulePath, &container);
}

SourceFile* Linkage::findFile(Name* name, SourceLoc loc, IncludeSystem& outIncludeSystem)
{
    auto impl = [&](bool translateUnderScore) -> SourceFile*
    {
        auto fileName = getFileNameFromModuleName(name, translateUnderScore);

        // Next, try to find the file of the given name,
        // using our ordinary include-handling logic.

        auto& searchDirs = getSearchDirectories();
        outIncludeSystem = IncludeSystem(&searchDirs, getFileSystemExt(), getSourceManager());

        // Get the original path info
        PathInfo pathIncludedFromInfo = getSourceManager()->getPathInfo(loc, SourceLocType::Actual);
        PathInfo filePathInfo;

        ComPtr<ISlangBlob> fileContents;

        // We have to load via the found path - as that is how file was originally loaded
        if (SLANG_FAILED(
                outIncludeSystem.findFile(fileName, pathIncludedFromInfo.foundPath, filePathInfo)))
        {
            return nullptr;
        }
        // Otherwise, try to load it.
        SourceFile* sourceFile;
        if (SLANG_FAILED(outIncludeSystem.loadFile(filePathInfo, fileContents, sourceFile)))
        {
            return nullptr;
        }
        return sourceFile;
    };
    if (auto rs = impl(false))
        return rs;
    return impl(true);
}

Linkage::IncludeResult Linkage::findAndIncludeFile(
    Module* module,
    TranslationUnitRequest* translationUnit,
    Name* name,
    SourceLoc const& loc,
    DiagnosticSink* sink)
{
    IncludeResult result;
    result.fileDecl = nullptr;
    result.isNew = false;

    IncludeSystem includeSystem;
    auto sourceFile = findFile(name, loc, includeSystem);
    if (!sourceFile)
    {
        sink->diagnose(loc, Diagnostics::cannotOpenFile, getText(name));
        return result;
    }

    // If the file has already been included, don't need to do anything further.
    if (auto existingFileDecl = module->getIncludedSourceFileMap().tryGetValue(sourceFile))
    {
        result.fileDecl = *existingFileDecl;
        result.isNew = false;
        return result;
    }

    if (isInLanguageServer())
    {
        // HACK: When in language server mode, we will always load the currently opend file as a
        // fresh module even if some previously opened file already references the current file via
        // `import` or `include`. see comments in `WorkspaceVersion::getOrLoadModule()` for the
        // reason behind this. An undesired outcome of this decision is that we could endup
        // including the currently opened file itself via chain of `__include`s because the
        // currently opened file will not have a true unique file system identity that allows it to
        // be deduplicated correct. Therefore we insert a hack logic here to detect re-inclusion by
        // just the file path. We can clean up this hack by making the language server truly support
        // incremental checking so we can reuse the previously loaded module instead of needing to
        // always start with a fresh copy.
        //
        for (auto file : translationUnit->getSourceFiles())
        {
            if (file->getPathInfo().hasFoundPath() &&
                Path::equals(file->getPathInfo().foundPath, sourceFile->getPathInfo().foundPath))
                return result;
        }
    }

    module->addFileDependency(sourceFile);

    // Create a transparent FileDecl to hold all children from the included file.
    auto fileDecl = module->getASTBuilder()->create<FileDecl>();
    fileDecl->nameAndLoc.name = name;
    module->getIncludedSourceFileMap().add(sourceFile, fileDecl);

    FrontEndPreprocessorHandler preprocessorHandler(module, module->getASTBuilder(), sink);
    auto combinedPreprocessorDefinitions = translationUnit->getCombinedPreprocessorDefinitions();
    SourceLanguage sourceLanguage = SourceLanguage::Unknown;
    auto tokens = preprocessSource(
        sourceFile,
        sink,
        &includeSystem,
        combinedPreprocessorDefinitions,
        this,
        sourceLanguage,
        &preprocessorHandler);

    if (sourceLanguage == SourceLanguage::Unknown)
        sourceLanguage = translationUnit->sourceLanguage;

    auto outerScope = module->getModuleDecl()->ownedScope;
    parseSourceFile(
        module->getASTBuilder(),
        translationUnit,
        sourceLanguage,
        tokens,
        sink,
        outerScope,
        fileDecl);

    module->getModuleDecl()->addMember(fileDecl);

    result.fileDecl = fileDecl;
    result.isNew = true;
    return result;
}

//
// ModuleDependencyList
//

void ModuleDependencyList::addDependency(Module* module)
{
    // If we depend on a module, then we depend on everything it depends on.
    //
    // Note: We are processing these sub-depenencies before adding the
    // `module` itself, so that in the common case a module will always
    // appear *after* everything it depends on.
    //
    // However, this rule is being violated in the compiler right now because
    // the modules for hte top-level translation units of a compile request
    // will be added to the list first (using `addLeafDependency`) to
    // maintain compatibility with old behavior. This may be fixed later.
    //
    for (auto subDependency : module->getModuleDependencyList())
    {
        _addDependency(subDependency);
    }
    _addDependency(module);
}

void ModuleDependencyList::addLeafDependency(Module* module)
{
    _addDependency(module);
}

void ModuleDependencyList::_addDependency(Module* module)
{
    if (m_moduleSet.contains(module))
        return;

    m_moduleList.add(module);
    m_moduleSet.add(module);
}

//
// FileDependencyList
//

void FileDependencyList::addDependency(SourceFile* sourceFile)
{
    if (m_fileSet.contains(sourceFile))
        return;

    m_fileList.add(sourceFile);
    m_fileSet.add(sourceFile);
}

void FileDependencyList::addDependency(Module* module)
{
    for (SourceFile* sourceFile : module->getFileDependencyList())
    {
        addDependency(sourceFile);
    }
}

//
// Module
//

Module::Module(Linkage* linkage, ASTBuilder* astBuilder)
    : ComponentType(linkage), m_mangledExportPool(StringSlicePool::Style::Empty)
{
    if (astBuilder)
    {
        m_astBuilder = astBuilder;
    }
    else
    {
        m_astBuilder = linkage->getASTBuilder();
    }
    getOptionSet() = linkage->m_optionSet;
    addModuleDependency(this);
}

ISlangUnknown* Module::getInterface(const Guid& guid)
{
    if (guid == IModule::getTypeGuid())
        return asExternal(this);
    if (guid == IModulePrecompileService_Experimental::getTypeGuid())
        return static_cast<slang::IModulePrecompileService_Experimental*>(this);
    return Super::getInterface(guid);
}

void Module::buildHash(DigestBuilder<SHA1>& builder)
{
    builder.append(computeDigest());
}

slang::DeclReflection* Module::getModuleReflection()
{
    return (slang::DeclReflection*)m_moduleDecl;
}

SHA1::Digest Module::computeDigest()
{
    if (m_digest == SHA1::Digest())
    {
        DigestBuilder<SHA1> digestBuilder;
        auto version = String(getBuildTagString());
        digestBuilder.append(version);
        getOptionSet().buildHash(digestBuilder);

        auto fileDependencies = getFileDependencies();

        for (auto file : fileDependencies)
        {
            digestBuilder.append(file->getDigest());
        }
        m_digest = digestBuilder.finalize();
    }
    return m_digest;
}

void Module::addModuleDependency(Module* module)
{
    m_moduleDependencyList.addDependency(module);
    m_fileDependencyList.addDependency(module);
}

void Module::addFileDependency(SourceFile* sourceFile)
{
    m_fileDependencyList.addDependency(sourceFile);
}

void Module::setModuleDecl(ModuleDecl* moduleDecl)
{
    m_moduleDecl = moduleDecl;
    moduleDecl->module = this;
}

void Module::setName(String name)
{
    m_name = getLinkage()->getNamePool()->getName(name);
}


RefPtr<EntryPoint> Module::findEntryPointByName(UnownedStringSlice const& name)
{
    for (auto entryPoint : m_entryPoints)
    {
        if (entryPoint->getName()->text.getUnownedSlice() == name)
            return entryPoint;
    }

    return nullptr;
}

RefPtr<EntryPoint> Module::findAndCheckEntryPoint(
    UnownedStringSlice const& name,
    SlangStage stage,
    ISlangBlob** outDiagnostics)
{
    // If there is already an entrypoint marked with the [shader] attribute,
    // we should just return that.
    //
    if (auto existingEntryPoint = findEntryPointByName(name))
        return existingEntryPoint;

    SLANG_AST_BUILDER_RAII(m_astBuilder);

    // If the function hasn't been marked as [shader], then it won't be discovered
    // by findEntryPointByName. We need to route this to the `findAndValidateEntryPoint`
    // function. To do that we need to setup a FrontEndCompileRequest and a
    // FrontEndEntryPointRequest.
    //
    DiagnosticSink sink(getLinkage()->getSourceManager(), DiagnosticSink::SourceLocationLexer());
    FrontEndCompileRequest frontEndRequest(getLinkage(), StdWriters::getSingleton(), &sink);
    RefPtr<TranslationUnitRequest> tuRequest = new TranslationUnitRequest(&frontEndRequest);
    tuRequest->module = this;
    tuRequest->moduleName = m_name;
    frontEndRequest.translationUnits.add(tuRequest);
    FrontEndEntryPointRequest entryPointRequest(
        &frontEndRequest,
        0,
        getLinkage()->getNamePool()->getName(name),
        Profile((Stage)stage));
    auto result = findAndValidateEntryPoint(&entryPointRequest);
    if (outDiagnostics)
    {
        sink.getBlobIfNeeded(outDiagnostics);
    }
    return result;
}

void Module::_addEntryPoint(EntryPoint* entryPoint)
{
    m_entryPoints.add(entryPoint);
}

static bool _canExportDeclSymbol(ASTNodeType type)
{
    switch (type)
    {
    case ASTNodeType::EmptyDecl:
        {
            return false;
        }
    default:
        break;
    }

    return true;
}

static bool _canRecurseExportSymbol(Decl* decl)
{
    if (as<FunctionDeclBase>(decl) || as<ScopeDecl>(decl))
    {
        return false;
    }
    return true;
}

void Module::_processFindDeclsExportSymbolsRec(Decl* decl)
{
    if (_canExportDeclSymbol(decl->astNodeType))
    {
        // It's a reference to a declaration in another module, so first get the symbol name.
        String mangledName = getMangledName(getCurrentASTBuilder(), decl);

        Index index = Index(m_mangledExportPool.add(mangledName));

        // TODO(JS): It appears that more than one entity might have the same mangled name.
        // So for now we ignore and just take the first one.
        if (index == m_mangledExportSymbols.getCount())
        {
            m_mangledExportSymbols.add(decl);
        }
    }

    if (!_canRecurseExportSymbol(decl))
    {
        // We don't need to recurse any further into this
        return;
    }

    // If it's a container process it's children
    if (auto containerDecl = as<ContainerDecl>(decl))
    {
        for (auto child : containerDecl->members)
        {
            _processFindDeclsExportSymbolsRec(child);
        }
    }

    // GenericDecl is also a container, so do subsequent test
    if (auto genericDecl = as<GenericDecl>(decl))
    {
        _processFindDeclsExportSymbolsRec(genericDecl->inner);
    }
}

NodeBase* Module::findExportFromMangledName(const UnownedStringSlice& slice)
{
    // Will be non zero if has been previously attempted
    if (m_mangledExportSymbols.getCount() == 0)
    {
        // Build up the exported mangled name list
        _processFindDeclsExportSymbolsRec(getModuleDecl());

        // If nothing found, mark that we have tried looking by making
        // m_mangledExportSymbols.getCount() != 0
        if (m_mangledExportSymbols.getCount() == 0)
        {
            m_mangledExportSymbols.add(nullptr);
        }
    }

    const Index index = m_mangledExportPool.findIndex(slice);
    return (index >= 0) ? m_mangledExportSymbols[index] : nullptr;
}

// ComponentType

ComponentType::ComponentType(Linkage* linkage)
    : m_linkage(linkage)
{
}

ComponentType* asInternal(slang::IComponentType* inComponentType)
{
    // Note: we use a `queryInterface` here instead of just a `static_cast`
    // to ensure that the `IComponentType` we get is the preferred/canonical
    // one, which shares its address with the `ComponentType`.
    //
    // TODO: An alternative choice here would be to have a "magic" IID that
    // we pass into `queryInterface` that returns the `ComponentType` directly
    // (without even `addRef`-ing it).
    //
    ComPtr<slang::IComponentType> componentType;
    inComponentType->queryInterface(SLANG_IID_PPV_ARGS(componentType.writeRef()));
    return static_cast<ComponentType*>(componentType.get());
}

ISlangUnknown* ComponentType::getInterface(Guid const& guid)
{
    if (guid == ISlangUnknown::getTypeGuid() || guid == slang::IComponentType::getTypeGuid())
    {
        return static_cast<slang::IComponentType*>(this);
    }
    if (guid == IModulePrecompileService_Experimental::getTypeGuid())
        return static_cast<slang::IModulePrecompileService_Experimental*>(this);
    return nullptr;
}

SLANG_NO_THROW slang::ISession* SLANG_MCALL ComponentType::getSession()
{
    return m_linkage;
}

SLANG_NO_THROW slang::ProgramLayout* SLANG_MCALL
ComponentType::getLayout(Int targetIndex, slang::IBlob** outDiagnostics)
{
    auto linkage = getLinkage();
    if (targetIndex < 0 || targetIndex >= linkage->targets.getCount())
        return nullptr;
    auto target = linkage->targets[targetIndex];

    DiagnosticSink sink(linkage->getSourceManager(), Lexer::sourceLocationLexer);
    auto programLayout = getTargetProgram(target)->getOrCreateLayout(&sink);
    sink.getBlobIfNeeded(outDiagnostics);

    return asExternal(programLayout);
}

static ICastable* _findDiagnosticRepresentation(IArtifact* artifact)
{
    if (auto rep = findAssociatedRepresentation<IArtifactDiagnostics>(artifact))
    {
        return rep;
    }

    for (auto associated : artifact->getAssociated())
    {
        if (isDerivedFrom(associated->getDesc().payload, ArtifactPayload::Diagnostics))
        {
            return associated;
        }
    }
    return nullptr;
}

static IArtifact* _findObfuscatedSourceMap(IArtifact* artifact)
{
    // If we find any obfuscated source maps, we are done
    for (auto associated : artifact->getAssociated())
    {
        const auto desc = associated->getDesc();

        if (isDerivedFrom(desc.payload, ArtifactPayload::SourceMap) &&
            isDerivedFrom(desc.style, ArtifactStyle::Obfuscated))
        {
            return associated;
        }
    }
    return nullptr;
}

SLANG_NO_THROW SlangResult SLANG_MCALL ComponentType::getResultAsFileSystem(
    SlangInt entryPointIndex,
    Int targetIndex,
    ISlangMutableFileSystem** outFileSystem)
{
    ComPtr<ISlangBlob> diagnostics;
    ComPtr<ISlangBlob> code;

    SLANG_RETURN_ON_FAIL(
        getEntryPointCode(entryPointIndex, targetIndex, diagnostics.writeRef(), code.writeRef()));

    auto linkage = getLinkage();

    auto target = linkage->targets[targetIndex];

    auto targetProgram = getTargetProgram(target);

    IArtifact* artifact = targetProgram->getExistingEntryPointResult(entryPointIndex);

    // Add diagnostics id needs be...
    if (diagnostics && !_findDiagnosticRepresentation(artifact))
    {
        // Add as an associated

        auto diagnosticsArtifact = Artifact::create(
            ArtifactDesc::make(Artifact::Kind::HumanText, ArtifactPayload::Diagnostics));
        diagnosticsArtifact->addRepresentationUnknown(diagnostics);

        artifact->addAssociated(diagnosticsArtifact);

        SLANG_ASSERT(diagnosticsArtifact == _findDiagnosticRepresentation(artifact));
    }

    // Add obfuscated source maps
    if (!_findObfuscatedSourceMap(artifact))
    {
        List<IRModule*> irModules;
        enumerateIRModules([&](IRModule* irModule) -> void { irModules.add(irModule); });

        for (auto irModule : irModules)
        {
            if (auto obfuscatedSourceMap = irModule->getObfuscatedSourceMap())
            {
                auto artifactDesc = ArtifactDesc::make(
                    ArtifactKind::Json,
                    ArtifactPayload::SourceMap,
                    ArtifactStyle::Obfuscated);

                // Create the source map artifact
                auto sourceMapArtifact = Artifact::create(
                    artifactDesc,
                    obfuscatedSourceMap->get().m_file.getUnownedSlice());

                sourceMapArtifact->addRepresentation(obfuscatedSourceMap);

                // associate with the artifact
                artifact->addAssociated(sourceMapArtifact);
            }
        }
    }

    // Turn into a file system and return
    ComPtr<ISlangMutableFileSystem> fileSystem(new MemoryFileSystem);

    // Filter the containerArtifact into things that can be written
    ComPtr<IArtifact> writeArtifact;
    SLANG_RETURN_ON_FAIL(ArtifactContainerUtil::filter(artifact, writeArtifact));
    SLANG_RETURN_ON_FAIL(ArtifactContainerUtil::writeContainer(writeArtifact, "", fileSystem));

    *outFileSystem = fileSystem.detach();

    return SLANG_OK;
}

SLANG_NO_THROW SlangResult SLANG_MCALL ComponentType::getEntryPointCode(
    SlangInt entryPointIndex,
    Int targetIndex,
    slang::IBlob** outCode,
    slang::IBlob** outDiagnostics)
{
    auto linkage = getLinkage();
    if (targetIndex < 0 || targetIndex >= linkage->targets.getCount())
        return SLANG_E_INVALID_ARG;
    auto target = linkage->targets[targetIndex];

    auto targetProgram = getTargetProgram(target);

    DiagnosticSink sink(linkage->getSourceManager(), Lexer::sourceLocationLexer);
    applySettingsToDiagnosticSink(&sink, &sink, linkage->m_optionSet);
    applySettingsToDiagnosticSink(&sink, &sink, m_optionSet);

    IArtifact* artifact = targetProgram->getOrCreateEntryPointResult(entryPointIndex, &sink);
    sink.getBlobIfNeeded(outDiagnostics);

    if (artifact == nullptr)
        return SLANG_FAIL;

    return artifact->loadBlob(ArtifactKeep::Yes, outCode);
}

SLANG_NO_THROW void SLANG_MCALL ComponentType::getEntryPointHash(
    SlangInt entryPointIndex,
    SlangInt targetIndex,
    slang::IBlob** outHash)
{
    DigestBuilder<SHA1> builder;

    // A note on enums that may be hashed in as part of the following two function calls:
    //
    // While enums are not guaranteed to be encoded the same way across all versions of
    // the compiler, part of hashing the linkage is hashing in the compiler version.
    // Consequently, any encoding differences as a result of different compiler versions
    // will already be reflected in the resulting hash.
    getLinkage()->buildHash(builder, targetIndex);

    buildHash(builder);

    // Add the name and name override for the specified entry point to the hash.
    auto entryPointName = getEntryPoint(entryPointIndex)->getName()->text;
    builder.append(entryPointName);
    auto entryPointMangledName = getEntryPointMangledName(entryPointIndex);
    builder.append(entryPointMangledName);
    auto entryPointNameOverride = getEntryPointNameOverride(entryPointIndex);
    builder.append(entryPointNameOverride);

    auto hash = builder.finalize().toBlob();
    *outHash = hash.detach();
}

SLANG_NO_THROW SlangResult SLANG_MCALL ComponentType::getEntryPointHostCallable(
    int entryPointIndex,
    int targetIndex,
    ISlangSharedLibrary** outSharedLibrary,
    slang::IBlob** outDiagnostics)
{
    auto linkage = getLinkage();
    if (targetIndex < 0 || targetIndex >= linkage->targets.getCount())
        return SLANG_E_INVALID_ARG;
    auto target = linkage->targets[targetIndex];

    auto targetProgram = getTargetProgram(target);

    DiagnosticSink sink(linkage->getSourceManager(), Lexer::sourceLocationLexer);
    applySettingsToDiagnosticSink(&sink, &sink, m_optionSet);

    IArtifact* artifact = targetProgram->getOrCreateEntryPointResult(entryPointIndex, &sink);
    sink.getBlobIfNeeded(outDiagnostics);

    if (artifact == nullptr)
        return SLANG_FAIL;

    return artifact->loadSharedLibrary(ArtifactKeep::Yes, outSharedLibrary);
}

SLANG_NO_THROW SlangResult SLANG_MCALL ComponentType::getEntryPointMetadata(
    SlangInt entryPointIndex,
    Int targetIndex,
    slang::IMetadata** outMetadata,
    slang::IBlob** outDiagnostics)
{
    auto linkage = getLinkage();
    if (targetIndex < 0 || targetIndex >= linkage->targets.getCount())
        return SLANG_E_INVALID_ARG;
    auto target = linkage->targets[targetIndex];

    auto targetProgram = getTargetProgram(target);

    DiagnosticSink sink(linkage->getSourceManager(), Lexer::sourceLocationLexer);
    applySettingsToDiagnosticSink(&sink, &sink, linkage->m_optionSet);
    applySettingsToDiagnosticSink(&sink, &sink, m_optionSet);

    IArtifact* artifact = targetProgram->getOrCreateEntryPointResult(entryPointIndex, &sink);
    sink.getBlobIfNeeded(outDiagnostics);

    if (artifact == nullptr)
        return SLANG_E_NOT_AVAILABLE;

    auto metadata = findAssociatedRepresentation<IArtifactPostEmitMetadata>(artifact);
    if (!metadata)
        return SLANG_E_NOT_AVAILABLE;

    *outMetadata = static_cast<slang::IMetadata*>(metadata);
    (*outMetadata)->addRef();
    return SLANG_OK;
}

RefPtr<ComponentType> ComponentType::specialize(
    SpecializationArg const* inSpecializationArgs,
    SlangInt specializationArgCount,
    DiagnosticSink* sink)
{
    if (specializationArgCount == 0)
    {
        return this;
    }

    List<SpecializationArg> specializationArgs;
    specializationArgs.addRange(inSpecializationArgs, specializationArgCount);

    // We next need to validate that the specialization arguments
    // make sense, and also expand them to include any derived data
    // (e.g., interface conformance witnesses) that doesn't get
    // passed explicitly through the API interface.
    //
    RefPtr<SpecializationInfo> specializationInfo =
        _validateSpecializationArgs(specializationArgs.getBuffer(), specializationArgCount, sink);

    return new SpecializedComponentType(this, specializationInfo, specializationArgs, sink);
}

SLANG_NO_THROW SlangResult SLANG_MCALL ComponentType::specialize(
    slang::SpecializationArg const* specializationArgs,
    SlangInt specializationArgCount,
    slang::IComponentType** outSpecializedComponentType,
    ISlangBlob** outDiagnostics)
{
    DiagnosticSink sink(getLinkage()->getSourceManager(), Lexer::sourceLocationLexer);

    // First let's check if the number of arguments given matches
    // the number of parameters that are present on this component type.
    //
    auto specializationParamCount = getSpecializationParamCount();
    if (specializationArgCount != specializationParamCount)
    {
        sink.diagnose(
            SourceLoc(),
            Diagnostics::mismatchSpecializationArguments,
            specializationParamCount,
            specializationArgCount);
        sink.getBlobIfNeeded(outDiagnostics);
        return SLANG_FAIL;
    }

    List<SpecializationArg> expandedArgs;
    for (Int aa = 0; aa < specializationArgCount; ++aa)
    {
        auto apiArg = specializationArgs[aa];

        SpecializationArg expandedArg;
        switch (apiArg.kind)
        {
        case slang::SpecializationArg::Kind::Type:
            expandedArg.val = asInternal(apiArg.type);
            break;

        default:
            sink.getBlobIfNeeded(outDiagnostics);
            return SLANG_FAIL;
        }
        expandedArgs.add(expandedArg);
    }

    auto specializedComponentType =
        specialize(expandedArgs.getBuffer(), expandedArgs.getCount(), &sink);

    sink.getBlobIfNeeded(outDiagnostics);

    *outSpecializedComponentType = specializedComponentType.detach();

    return SLANG_OK;
}

SLANG_NO_THROW SlangResult SLANG_MCALL
ComponentType::renameEntryPoint(const char* newName, IComponentType** outEntryPoint)
{
    RefPtr<RenamedEntryPointComponentType> result =
        new RenamedEntryPointComponentType(this, newName);
    *outEntryPoint = result.detach();
    return SLANG_OK;
}

RefPtr<ComponentType> fillRequirements(ComponentType* inComponentType);

SLANG_NO_THROW SlangResult SLANG_MCALL
ComponentType::link(slang::IComponentType** outLinkedComponentType, ISlangBlob** outDiagnostics)
{
    // TODO: It should be possible for `fillRequirements` to fail,
    // in cases where we have a dependency that can't be automatically
    // resolved.
    //
    SLANG_UNUSED(outDiagnostics);

    DiagnosticSink sink(getLinkage()->getSourceManager(), Lexer::sourceLocationLexer);

    try
    {
        auto linked = fillRequirements(this);
        if (!linked)
            return SLANG_FAIL;

        *outLinkedComponentType = ComPtr<slang::IComponentType>(linked).detach();
        return SLANG_OK;
    }
    catch (const AbortCompilationException& e)
    {
        outputExceptionDiagnostic(e, sink, outDiagnostics);
        return SLANG_FAIL;
    }
    catch (const Exception& e)
    {
        outputExceptionDiagnostic(e, sink, outDiagnostics);
        return SLANG_FAIL;
    }
    catch (...)
    {
        outputExceptionDiagnostic(sink, outDiagnostics);
        return SLANG_FAIL;
    }
}

SLANG_NO_THROW SlangResult SLANG_MCALL ComponentType::linkWithOptions(
    slang::IComponentType** outLinkedComponentType,
    uint32_t count,
    slang::CompilerOptionEntry* entries,
    ISlangBlob** outDiagnostics)
{
    SLANG_RETURN_ON_FAIL(link(outLinkedComponentType, outDiagnostics));

    auto linked = *outLinkedComponentType;

    if (linked)
    {
        static_cast<ComponentType*>(linked)->getOptionSet().load(count, entries);
    }

    return SLANG_OK;
}

/// Visitor used by `ComponentType::enumerateModules`
struct EnumerateModulesVisitor : ComponentTypeVisitor
{
    EnumerateModulesVisitor(ComponentType::EnumerateModulesCallback callback, void* userData)
        : m_callback(callback), m_userData(userData)
    {
    }

    ComponentType::EnumerateModulesCallback m_callback;
    void* m_userData;

    void visitEntryPoint(EntryPoint*, EntryPoint::EntryPointSpecializationInfo*) SLANG_OVERRIDE {}

    void visitRenamedEntryPoint(
        RenamedEntryPointComponentType* entryPoint,
        EntryPoint::EntryPointSpecializationInfo* specializationInfo) SLANG_OVERRIDE
    {
        entryPoint->getBase()->acceptVisitor(this, specializationInfo);
    }

    void visitModule(Module* module, Module::ModuleSpecializationInfo*) SLANG_OVERRIDE
    {
        m_callback(module, m_userData);
    }

    void visitComposite(
        CompositeComponentType* composite,
        CompositeComponentType::CompositeSpecializationInfo* specializationInfo) SLANG_OVERRIDE
    {
        visitChildren(composite, specializationInfo);
    }

    void visitSpecialized(SpecializedComponentType* specialized) SLANG_OVERRIDE
    {
        visitChildren(specialized);
    }

    void visitTypeConformance(TypeConformance* conformance) SLANG_OVERRIDE
    {
        SLANG_UNUSED(conformance);
    }
};


void ComponentType::enumerateModules(EnumerateModulesCallback callback, void* userData)
{
    EnumerateModulesVisitor visitor(callback, userData);
    acceptVisitor(&visitor, nullptr);
}

/// Visitor used by `ComponentType::enumerateIRModules`
struct EnumerateIRModulesVisitor : ComponentTypeVisitor
{
    EnumerateIRModulesVisitor(ComponentType::EnumerateIRModulesCallback callback, void* userData)
        : m_callback(callback), m_userData(userData)
    {
    }

    ComponentType::EnumerateIRModulesCallback m_callback;
    void* m_userData;

    void visitEntryPoint(EntryPoint*, EntryPoint::EntryPointSpecializationInfo*) SLANG_OVERRIDE {}

    void visitRenamedEntryPoint(
        RenamedEntryPointComponentType* entryPoint,
        EntryPoint::EntryPointSpecializationInfo* specializationInfo) SLANG_OVERRIDE
    {
        entryPoint->getBase()->acceptVisitor(this, specializationInfo);
    }

    void visitModule(Module* module, Module::ModuleSpecializationInfo*) SLANG_OVERRIDE
    {
        m_callback(module->getIRModule(), m_userData);
    }

    void visitComposite(
        CompositeComponentType* composite,
        CompositeComponentType::CompositeSpecializationInfo* specializationInfo) SLANG_OVERRIDE
    {
        visitChildren(composite, specializationInfo);
    }

    void visitSpecialized(SpecializedComponentType* specialized) SLANG_OVERRIDE
    {
        visitChildren(specialized);

        m_callback(specialized->getIRModule(), m_userData);
    }

    void visitTypeConformance(TypeConformance* conformance) SLANG_OVERRIDE
    {
        m_callback(conformance->getIRModule(), m_userData);
    }
};

void ComponentType::enumerateIRModules(EnumerateIRModulesCallback callback, void* userData)
{
    EnumerateIRModulesVisitor visitor(callback, userData);
    acceptVisitor(&visitor, nullptr);
}

IArtifact* ComponentType::getTargetArtifact(Int targetIndex, slang::IBlob** outDiagnostics)
{
    auto linkage = getLinkage();
    if (targetIndex < 0 || targetIndex >= linkage->targets.getCount())
        return nullptr;
    ComPtr<IArtifact> artifact;
    if (m_targetArtifacts.tryGetValue(targetIndex, artifact))
    {
        return artifact.get();
    }

    // If the user hasn't specified any entry points, then we should
    // discover all entrypoints that are defined in linked modules, and
    // include all of them in the compile.
    //
    if (getEntryPointCount() == 0)
    {
        List<Module*> modules;
        this->enumerateModules([&](Module* module) { modules.add(module); });
        List<RefPtr<ComponentType>> components;
        components.add(this);
        bool entryPointsDiscovered = false;
        for (auto module : modules)
        {
            for (auto entryPoint : module->getEntryPoints())
            {
                components.add(entryPoint);
                entryPointsDiscovered = true;
            }
        }

        // If any entry points were discovered, then we should emit the program with entrypoints
        // linked.
        if (entryPointsDiscovered)
        {
            RefPtr<CompositeComponentType> composite =
                new CompositeComponentType(linkage, components);
            ComPtr<IComponentType> linkedComponentType;
            SLANG_RETURN_NULL_ON_FAIL(
                composite->link(linkedComponentType.writeRef(), outDiagnostics));
            auto targetArtifact = static_cast<ComponentType*>(linkedComponentType.get())
                                      ->getTargetArtifact(targetIndex, outDiagnostics);
            if (targetArtifact)
            {
                m_targetArtifacts[targetIndex] = targetArtifact;
            }
            return targetArtifact;
        }
    }

    auto target = linkage->targets[targetIndex];
    auto targetProgram = getTargetProgram(target);

    DiagnosticSink sink(linkage->getSourceManager(), Lexer::sourceLocationLexer);
    applySettingsToDiagnosticSink(&sink, &sink, linkage->m_optionSet);
    applySettingsToDiagnosticSink(&sink, &sink, m_optionSet);

    IArtifact* targetArtifact = targetProgram->getOrCreateWholeProgramResult(&sink);
    sink.getBlobIfNeeded(outDiagnostics);
    m_targetArtifacts[targetIndex] = ComPtr<IArtifact>(targetArtifact);
    return targetArtifact;
}

SLANG_NO_THROW SlangResult SLANG_MCALL
ComponentType::getTargetCode(Int targetIndex, slang::IBlob** outCode, slang::IBlob** outDiagnostics)
{
    IArtifact* artifact = getTargetArtifact(targetIndex, outDiagnostics);

    if (artifact == nullptr)
        return SLANG_FAIL;

    return artifact->loadBlob(ArtifactKeep::Yes, outCode);
}

SLANG_NO_THROW SlangResult SLANG_MCALL ComponentType::getTargetMetadata(
    Int targetIndex,
    slang::IMetadata** outMetadata,
    slang::IBlob** outDiagnostics)
{
    IArtifact* artifact = getTargetArtifact(targetIndex, outDiagnostics);

    if (artifact == nullptr)
        return SLANG_FAIL;

    auto metadata = findAssociatedRepresentation<IArtifactPostEmitMetadata>(artifact);
    if (!metadata)
        return SLANG_E_NOT_AVAILABLE;
    *outMetadata = static_cast<slang::IMetadata*>(metadata);
    (*outMetadata)->addRef();
    return SLANG_OK;
}

//
// CompositeComponentType
//

RefPtr<ComponentType> CompositeComponentType::create(
    Linkage* linkage,
    List<RefPtr<ComponentType>> const& childComponents)
{
    // TODO: We should ideally be caching the results of
    // composition on the `linkage`, so that if we get
    // asked for the same composite again later we re-use
    // it rather than re-create it.
    //
    // Similarly, we might want to do some amount of
    // work to "canonicalize" the input for composition.
    // E.g., if the user does:
    //
    //    X = compose(A,B);
    //    Y = compose(C,D);
    //    Z = compose(X,Y);
    //
    //    W = compose(A, B, C, D);
    //
    // Then there is no observable difference between
    // Z and W, so we might prefer to have them be identical.

    // If there is only a single child, then we should
    // just return that child rather than create a dummy composite.
    //
    if (childComponents.getCount() == 1)
    {
        return childComponents[0];
    }

    return new CompositeComponentType(linkage, childComponents);
}


CompositeComponentType::CompositeComponentType(
    Linkage* linkage,
    List<RefPtr<ComponentType>> const& childComponents)
    : ComponentType(linkage), m_childComponents(childComponents)
{
    HashSet<ComponentType*> requirementsSet;
    for (auto child : childComponents)
    {
        child->enumerateModules([&](Module* module) { requirementsSet.add(module); });
    }

    for (auto child : childComponents)
    {
        auto childEntryPointCount = child->getEntryPointCount();
        for (Index cc = 0; cc < childEntryPointCount; ++cc)
        {
            m_entryPoints.add(child->getEntryPoint(cc));
            m_entryPointMangledNames.add(child->getEntryPointMangledName(cc));
            m_entryPointNameOverrides.add(child->getEntryPointNameOverride(cc));
        }

        auto childShaderParamCount = child->getShaderParamCount();
        for (Index pp = 0; pp < childShaderParamCount; ++pp)
        {
            m_shaderParams.add(child->getShaderParam(pp));
        }

        auto childSpecializationParamCount = child->getSpecializationParamCount();
        for (Index pp = 0; pp < childSpecializationParamCount; ++pp)
        {
            m_specializationParams.add(child->getSpecializationParam(pp));
        }

        for (auto module : child->getModuleDependencies())
        {
            m_moduleDependencyList.addDependency(module);
        }
        for (auto sourceFile : child->getFileDependencies())
        {
            m_fileDependencyList.addDependency(sourceFile);
        }

        auto childRequirementCount = child->getRequirementCount();
        for (Index rr = 0; rr < childRequirementCount; ++rr)
        {
            auto childRequirement = child->getRequirement(rr);
            if (!requirementsSet.contains(childRequirement))
            {
                requirementsSet.add(childRequirement);
                m_requirements.add(childRequirement);
            }
        }
    }
}

void CompositeComponentType::buildHash(DigestBuilder<SHA1>& builder)
{
    auto componentCount = getChildComponentCount();

    for (Index i = 0; i < componentCount; ++i)
    {
        getChildComponent(i)->buildHash(builder);
    }
}

Index CompositeComponentType::getEntryPointCount()
{
    return m_entryPoints.getCount();
}

RefPtr<EntryPoint> CompositeComponentType::getEntryPoint(Index index)
{
    return m_entryPoints[index];
}

String CompositeComponentType::getEntryPointMangledName(Index index)
{
    return m_entryPointMangledNames[index];
}

String CompositeComponentType::getEntryPointNameOverride(Index index)
{
    return m_entryPointNameOverrides[index];
}

Index CompositeComponentType::getShaderParamCount()
{
    return m_shaderParams.getCount();
}

ShaderParamInfo CompositeComponentType::getShaderParam(Index index)
{
    return m_shaderParams[index];
}

Index CompositeComponentType::getSpecializationParamCount()
{
    return m_specializationParams.getCount();
}

SpecializationParam const& CompositeComponentType::getSpecializationParam(Index index)
{
    return m_specializationParams[index];
}

Index CompositeComponentType::getRequirementCount()
{
    return m_requirements.getCount();
}

RefPtr<ComponentType> CompositeComponentType::getRequirement(Index index)
{
    return m_requirements[index];
}

List<Module*> const& CompositeComponentType::getModuleDependencies()
{
    return m_moduleDependencyList.getModuleList();
}

List<SourceFile*> const& CompositeComponentType::getFileDependencies()
{
    return m_fileDependencyList.getFileList();
}

void CompositeComponentType::acceptVisitor(
    ComponentTypeVisitor* visitor,
    SpecializationInfo* specializationInfo)
{
    visitor->visitComposite(this, as<CompositeSpecializationInfo>(specializationInfo));
}

RefPtr<ComponentType::SpecializationInfo> CompositeComponentType::_validateSpecializationArgsImpl(
    SpecializationArg const* args,
    Index argCount,
    DiagnosticSink* sink)
{
    SLANG_UNUSED(argCount);

    RefPtr<CompositeSpecializationInfo> specializationInfo = new CompositeSpecializationInfo();

    Index offset = 0;
    for (auto child : m_childComponents)
    {
        auto childParamCount = child->getSpecializationParamCount();
        SLANG_ASSERT(offset + childParamCount <= argCount);

        auto childInfo = child->_validateSpecializationArgs(args + offset, childParamCount, sink);

        specializationInfo->childInfos.add(childInfo);

        offset += childParamCount;
    }
    return specializationInfo;
}

//
// SpecializedComponentType
//

/// Utility type for collecting modules references by types/declarations
struct SpecializationArgModuleCollector : ComponentTypeVisitor
{
    HashSet<Module*> m_modulesSet;
    List<Module*> m_modulesList;

    void addModule(Module* module)
    {
        m_modulesList.add(module);
        m_modulesSet.add(module);
    }

    void maybeAddModule(Module* module)
    {
        if (!module)
            return;
        if (m_modulesSet.contains(module))
            return;

        addModule(module);
    }

    void collectReferencedModules(Decl* decl)
    {
        auto module = getModule(decl);
        maybeAddModule(module);
    }

    void collectReferencedModules(SubstitutionSet substitutions)
    {
        substitutions.forEachGenericSubstitution(
            [this](GenericDecl*, Val::OperandView<Val> args)
            {
                for (auto arg : args)
                {
                    collectReferencedModules(arg);
                }
            });
    }

    void collectReferencedModules(DeclRefBase* declRef)
    {
        collectReferencedModules(declRef->getDecl());
        collectReferencedModules(SubstitutionSet(declRef));
    }

    void collectReferencedModules(Type* type)
    {
        if (auto declRefType = as<DeclRefType>(type))
        {
            collectReferencedModules(declRefType->getDeclRef());
        }

        // TODO: Handle non-decl-ref composite type cases
        // (e.g., function types).
    }

    void collectReferencedModules(Val* val)
    {
        if (auto type = as<Type>(val))
        {
            collectReferencedModules(type);
        }
        else if (auto declRefVal = as<GenericParamIntVal>(val))
        {
            collectReferencedModules(declRefVal->getDeclRef());
        }

        // TODO: other cases of values that could reference
        // a declaration.
    }

    void collectReferencedModules(List<ExpandedSpecializationArg> const& args)
    {
        for (auto arg : args)
        {
            collectReferencedModules(arg.val);
            collectReferencedModules(arg.witness);
        }
    }

    //
    // ComponentTypeVisitor methods
    //

    void visitEntryPoint(
        EntryPoint* entryPoint,
        EntryPoint::EntryPointSpecializationInfo* specializationInfo) SLANG_OVERRIDE
    {
        SLANG_UNUSED(entryPoint);

        if (!specializationInfo)
            return;

        collectReferencedModules(specializationInfo->specializedFuncDeclRef);
        collectReferencedModules(specializationInfo->existentialSpecializationArgs);
    }

    void visitRenamedEntryPoint(
        RenamedEntryPointComponentType* entryPoint,
        EntryPoint::EntryPointSpecializationInfo* specializationInfo) SLANG_OVERRIDE
    {
        entryPoint->getBase()->acceptVisitor(this, specializationInfo);
    }

    void visitModule(Module* module, Module::ModuleSpecializationInfo* specializationInfo)
        SLANG_OVERRIDE
    {
        SLANG_UNUSED(module);

        if (!specializationInfo)
            return;

        for (auto arg : specializationInfo->genericArgs)
        {
            collectReferencedModules(arg.argVal);
        }
        collectReferencedModules(specializationInfo->existentialArgs);
    }

    void visitComposite(
        CompositeComponentType* composite,
        CompositeComponentType::CompositeSpecializationInfo* specializationInfo) SLANG_OVERRIDE
    {
        visitChildren(composite, specializationInfo);
    }

    void visitSpecialized(SpecializedComponentType* specialized) SLANG_OVERRIDE
    {
        visitChildren(specialized);
    }

    void visitTypeConformance(TypeConformance* conformance) SLANG_OVERRIDE
    {
        SLANG_UNUSED(conformance);
    }
};

SpecializedComponentType::SpecializedComponentType(
    ComponentType* base,
    ComponentType::SpecializationInfo* specializationInfo,
    List<SpecializationArg> const& specializationArgs,
    DiagnosticSink* sink)
    : ComponentType(base->getLinkage())
    , m_base(base)
    , m_specializationInfo(specializationInfo)
    , m_specializationArgs(specializationArgs)
{
    m_optionSet.overrideWith(base->getOptionSet());

    m_irModule = generateIRForSpecializedComponentType(this, sink);

    // We need to account for the fact that a specialized
    // entity like `myShader<SomeType>` needs to not only
    // depend on the module(s) that `myShader` depends on,
    // but also on any modules that `SomeType` depends on.
    //
    // We will set up a "collector" type that will be
    // used to build a list of these additional modules.
    //
    SpecializationArgModuleCollector moduleCollector;

    // We don't want to go adding additional requirements for
    // modules that the base component type already includes,
    // so we will add those to the set of modules in
    // the collector before we starting trying to add others.
    //
    base->enumerateModules([&](Module* module) { moduleCollector.m_modulesSet.add(module); });

    // In order to collect the additional modules, we need
    // to inspect the specialization arguments and see what
    // they depend on.
    //
    // Naively, it seems like we'd just want to iterate
    // over `specializationArgs`, which gives the specialization
    // arguments as the user supplied them. However, such
    // an approach would have a subtle problem.
    //
    // If we have a generic entry point like:
    //
    //      // In module A
    //      myShader<T : IThing>
    //
    //
    // And the type `SomeType` that is being used as an argument doesn't
    // directly conform to `IThing`:
    //
    //      // In module B
    //      struct SomeType { ... }
    //
    // and the conformance of `SomeType` to `IThing` is
    // coming from yet another module:
    //
    //      // In module C
    //      import B;
    //      extension SomeType : IThing { ... }
    //
    // In this case, the specialized component for `myShader<SomeType>`
    // needs to depend on all of:
    //
    // * Module A, because it defines `myShader`
    // * Module B, because it defines `SomeType`
    // * Module C, because it defines the conformance `SomeType : IThing`
    //
    // We thus need to iterate over a form of the specialization
    // arguments that includes the "expanded" arguments like
    // interface conformance witnesses that got added during
    // semantic checking.
    //
    // The expanded arguments are being stored in the `specializationInfo`
    // today (for use by downstream code generation), and the easiest
    // way to walk that information and get to the leaf nodes where
    // the expanded arguments are stored is to apply a visitor to
    // the specialized component type we are in the middle of constructing.
    //
    moduleCollector.visitSpecialized(this);

    // Now that we've collected our additional information, we can
    // start to build up the final lists for the specialized component type.
    //
    // The starting point for our lists comes from the base component type.
    //
    m_moduleDependencies = base->getModuleDependencies();
    m_fileDependencies = base->getFileDependencies();

    Index baseRequirementCount = base->getRequirementCount();
    for (Index r = 0; r < baseRequirementCount; r++)
    {
        m_requirements.add(base->getRequirement(r));
    }

    // The specialized component type will need to have additional
    // dependencies and requirements based on the modules that
    // were collected when looking at the specialization arguments.

    // We want to avoid adding the same file dependency more than once.
    //
    HashSet<SourceFile*> fileDependencySet;
    for (SourceFile* sourceFile : m_fileDependencies)
        fileDependencySet.add(sourceFile);

    for (auto module : moduleCollector.m_modulesList)
    {
        // The specialized component type will have an open (unsatisfied)
        // requirement for each of the modules that its specialization
        // arguments need.
        //
        // Note: what this means in practice is that the component type
        // records that the given module(s) will need to be linked in
        // before final code can be generated, but it importantly
        // does not dictate the final placement of the parameters from
        // those modules in the layout.
        //
        m_requirements.add(module);

        // The speciialized component type will also have a dependency
        // on all the files that any of the modules involved in
        // it depend on (including those that are required but not
        // yet linked in).
        //
        // The file path information is what a client would need to
        // use to decide if kernel code is out of date compared to
        // source files, so we want to include anything that could
        // affect the validity of generated code.
        //
        for (SourceFile* sourceFile : module->getFileDependencies())
        {
            if (fileDependencySet.contains(sourceFile))
                continue;
            fileDependencySet.add(sourceFile);
            m_fileDependencies.add(sourceFile);
        }

        // Finalyl we also add the module for the specialization arguments
        // to the list of modules that would be used for legacy lookup
        // operations where we need an implicit/default scope to use
        // and want it to be expansive.
        //
        // TODO: This stuff really isn't worth keeping around long
        // term, and we should ditch the entire "legacy lookup" idea.
        //
        m_moduleDependencies.add(module);
    }

    // Because we are specializing shader code, the mangled entry
    // point names for this component type may be different than
    // for the base component type (e.g., the mangled name for `f<int>`
    // is different than that that of the generic `f` function
    // itself).
    //
    // We will compute the mangled names of all the entry points and
    // store them here, so that we don't have to do it on the fly.
    // Because the `ComponentType` structure is hierarchical, we
    // need to use a recursive visitor to compute the names,
    // and we will define that visitor locally:
    //
    struct EntryPointMangledNameCollector : ComponentTypeVisitor
    {
        List<String>* mangledEntryPointNames;
        List<String>* entryPointNameOverrides;

        void visitEntryPoint(
            EntryPoint* entryPoint,
            EntryPoint::EntryPointSpecializationInfo* specializationInfo) SLANG_OVERRIDE
        {
            auto funcDeclRef = entryPoint->getFuncDeclRef();
            if (specializationInfo)
                funcDeclRef = specializationInfo->specializedFuncDeclRef;

            (*mangledEntryPointNames).add(getMangledName(m_astBuilder, funcDeclRef));
            (*entryPointNameOverrides).add(entryPoint->getEntryPointNameOverride(0));
        }

        void visitRenamedEntryPoint(
            RenamedEntryPointComponentType* entryPoint,
            EntryPoint::EntryPointSpecializationInfo* specializationInfo) SLANG_OVERRIDE
        {
            entryPoint->getBase()->acceptVisitor(this, specializationInfo);
            (*entryPointNameOverrides).getLast() = entryPoint->getEntryPointNameOverride(0);
        }

        void visitModule(Module*, Module::ModuleSpecializationInfo*) SLANG_OVERRIDE {}
        void visitComposite(
            CompositeComponentType* composite,
            CompositeComponentType::CompositeSpecializationInfo* specializationInfo) SLANG_OVERRIDE
        {
            visitChildren(composite, specializationInfo);
        }
        void visitSpecialized(SpecializedComponentType* specialized) SLANG_OVERRIDE
        {
            visitChildren(specialized);
        }
        void visitTypeConformance(TypeConformance* conformance) SLANG_OVERRIDE
        {
            SLANG_UNUSED(conformance);
        }
        EntryPointMangledNameCollector(ASTBuilder* astBuilder)
            : m_astBuilder(astBuilder)
        {
        }
        ASTBuilder* m_astBuilder;
    };

    // With the visitor defined, we apply it to ourself to compute
    // and collect the mangled entry point names.
    //
    EntryPointMangledNameCollector collector(getLinkage()->getASTBuilder());
    collector.mangledEntryPointNames = &m_entryPointMangledNames;
    collector.entryPointNameOverrides = &m_entryPointNameOverrides;
    collector.visitSpecialized(this);
}

void SpecializedComponentType::buildHash(DigestBuilder<SHA1>& builder)
{
    auto specializationArgCount = getSpecializationArgCount();
    for (Index i = 0; i < specializationArgCount; ++i)
    {
        auto specializationArg = getSpecializationArg(i);
        auto argString = specializationArg.val->toString();
        builder.append(argString);
    }

    getBaseComponentType()->buildHash(builder);
}

void SpecializedComponentType::acceptVisitor(
    ComponentTypeVisitor* visitor,
    SpecializationInfo* specializationInfo)
{
    SLANG_ASSERT(specializationInfo == nullptr);
    SLANG_UNUSED(specializationInfo);
    visitor->visitSpecialized(this);
}

Index SpecializedComponentType::getRequirementCount()
{
    return m_requirements.getCount();
}

RefPtr<ComponentType> SpecializedComponentType::getRequirement(Index index)
{
    return m_requirements[index];
}

String SpecializedComponentType::getEntryPointMangledName(Index index)
{
    return m_entryPointMangledNames[index];
}

String SpecializedComponentType::getEntryPointNameOverride(Index index)
{
    return m_entryPointNameOverrides[index];
}

// RenamedEntryPointComponentType

RenamedEntryPointComponentType::RenamedEntryPointComponentType(ComponentType* base, String newName)
    : ComponentType(base->getLinkage()), m_base(base), m_entryPointNameOverride(newName)
{
}

void RenamedEntryPointComponentType::acceptVisitor(
    ComponentTypeVisitor* visitor,
    SpecializationInfo* specializationInfo)
{
    visitor->visitRenamedEntryPoint(
        this,
        as<EntryPoint::EntryPointSpecializationInfo>(specializationInfo));
}

void RenamedEntryPointComponentType::buildHash(DigestBuilder<SHA1>& builder)
{
    SLANG_UNUSED(builder);
}

void ComponentTypeVisitor::visitChildren(
    CompositeComponentType* composite,
    CompositeComponentType::CompositeSpecializationInfo* specializationInfo)
{
    auto childCount = composite->getChildComponentCount();
    for (Index ii = 0; ii < childCount; ++ii)
    {
        auto child = composite->getChildComponent(ii);
        auto childSpecializationInfo =
            specializationInfo ? specializationInfo->childInfos[ii] : nullptr;

        child->acceptVisitor(this, childSpecializationInfo);
    }
}

void ComponentTypeVisitor::visitChildren(SpecializedComponentType* specialized)
{
    specialized->getBaseComponentType()->acceptVisitor(this, specialized->getSpecializationInfo());
}

TargetProgram* ComponentType::getTargetProgram(TargetRequest* target)
{
    RefPtr<TargetProgram> targetProgram;
    if (!m_targetPrograms.tryGetValue(target, targetProgram))
    {
        targetProgram = new TargetProgram(this, target);
        m_targetPrograms[target] = targetProgram;
    }
    return targetProgram;
}

//
// TargetProgram
//

TargetProgram::TargetProgram(ComponentType* componentType, TargetRequest* targetReq)
    : m_program(componentType), m_targetReq(targetReq)
{
    m_entryPointResults.setCount(componentType->getEntryPointCount());
    m_optionSet.overrideWith(m_program->getOptionSet());
    m_optionSet.inheritFrom(targetReq->getOptionSet());
}

//

Session* CompileRequestBase::getSession()
{
    return getLinkage()->getSessionImpl();
}

void Linkage::setFileSystem(ISlangFileSystem* inFileSystem)
{
    // Set the fileSystem
    m_fileSystem = inFileSystem;

    // Release what's there
    m_fileSystemExt.setNull();

    // If nullptr passed in set up default
    if (inFileSystem == nullptr)
    {
        m_fileSystemExt = new Slang::CacheFileSystem(Slang::OSFileSystem::getExtSingleton());
    }
    else
    {
        if (auto cacheFileSystem = as<CacheFileSystem>(inFileSystem))
        {
            m_fileSystemExt = cacheFileSystem;
        }
        else
        {
            if (m_requireCacheFileSystem)
            {
                m_fileSystemExt = new Slang::CacheFileSystem(inFileSystem);
            }
            else
            {
                // See if we have the full ISlangFileSystemExt interface, if we do just use it
                inFileSystem->queryInterface(SLANG_IID_PPV_ARGS(m_fileSystemExt.writeRef()));

                // If not wrap with CacheFileSystem that emulates ISlangFileSystemExt from the
                // ISlangFileSystem interface
                if (!m_fileSystemExt)
                {
                    // Construct a wrapper to emulate the extended interface behavior
                    m_fileSystemExt = new Slang::CacheFileSystem(m_fileSystem);
                }
            }
        }
    }

    // If requires a cache file system, check that it does have one
    SLANG_ASSERT(m_requireCacheFileSystem == false || as<CacheFileSystem>(m_fileSystemExt));

    // Set the file system used on the source manager
    getSourceManager()->setFileSystemExt(m_fileSystemExt);
}

SlangResult Linkage::loadSerializedModuleContents(
    Module* module,
    const PathInfo& moduleFilePathInfo,
    ModuleChunkRef moduleChunk,
    DiagnosticSink* sink)
{
    // At this point we've dealt with basically all of
    // the formalities, and we just need to get down
    // to the real work of decoding the information
    // in the `moduleChunk`.

    auto sourceManager = getSourceManager();
    RefPtr<SerialSourceLocReader> sourceLocReader;
    if (auto debugChunk = findDebugChunk(moduleChunk.ptr()))
    {
        SLANG_RETURN_ON_FAIL(
            readSourceLocationsFromDebugChunk(debugChunk, sourceManager, sourceLocReader));
    }

    auto astChunk = moduleChunk.findAST();
    if (!astChunk)
        return SLANG_FAIL;

    auto irChunk = moduleChunk.findIR();
    if (!irChunk)
        return SLANG_FAIL;

    auto astBuilder = getASTBuilder();
    auto session = getSessionImpl();

    // For the purposes of any modules referenced
    // by the module we're about to decode, we will
    // construct a source location that represents
    // the module itself (if possible).
    //
    // TODO(tfoley): This logic seems like overkill, given
    // that many (most? all?) control-flow paths that can
    // reach this routine will have already found a `SourceFile`
    // to represent the module, as part of even getting the
    // `moduleFilePathInfo` to pass in
    //
    // The approach here is more or less exactly copied
    // from what the old `SerialContainerUtil::read` function
    // used to do, with the hopes that it will as many tests
    // passing as possible.
    //
    // Down the line somebody should scrutinize all of this
    // kind of logic in the compiler codebase, because there
    // is something that feels unclean about how paths are being handled.
    //
    SourceLoc serializedModuleLoc;
    {
        auto sourceFile =
            sourceManager->findSourceFileByPathRecursively(moduleFilePathInfo.foundPath);
        if (!sourceFile)
        {
            sourceFile = sourceManager->createSourceFileWithString(moduleFilePathInfo, String());
            sourceManager->addSourceFile(moduleFilePathInfo.getMostUniqueIdentity(), sourceFile);
        }
        auto sourceView =
            sourceManager->createSourceView(sourceFile, &moduleFilePathInfo, SourceLoc());
        serializedModuleLoc = sourceView->getRange().begin;
    }

    auto moduleDecl = readSerializedModuleAST(
        this,
        astBuilder,
        sink,
        astChunk,
        sourceLocReader,
        serializedModuleLoc);
    if (!moduleDecl)
        return SLANG_FAIL;
    module->setModuleDecl(moduleDecl);

    RefPtr<IRModule> irModule;
    SLANG_RETURN_ON_FAIL(decodeModuleIR(irModule, irChunk, session, sourceLocReader));
    module->setIRModule(irModule);

    // The handling of file dependencies is complicated, because of
    // the way that the encoding logic tried to make all of the
    // paths be relative to the primary source file for the module.
    //
    // We end up needing to undo some amount of that work here.
    //

    module->clearFileDependency();
    String moduleSourcePath = moduleFilePathInfo.foundPath;
    bool isFirst = true;
    for (auto depenencyFileChunk : moduleChunk.getFileDependencies())
    {
        auto encodedDependencyFilePath = depenencyFileChunk.getValue();

        auto sourceFile = loadSourceFile(moduleFilePathInfo.foundPath, encodedDependencyFilePath);
        if (isFirst)
        {
            // The first file is the source for the main module file.
            // We store the module path as the basis for finding the remaining
            // dependent files.
            if (sourceFile)
                moduleSourcePath = sourceFile->getPathInfo().foundPath;
            isFirst = false;
        }
        // If we cannot find the dependent file directly, try to find
        // it relative to the module source path.
        if (!sourceFile)
        {
            sourceFile = loadSourceFile(moduleSourcePath, encodedDependencyFilePath);
        }
        if (sourceFile)
        {
            module->addFileDependency(sourceFile);
        }
    }
    module->setPathInfo(moduleFilePathInfo);
    module->setDigest(moduleChunk.getDigest());
    module->_collectShaderParams();
    module->_discoverEntryPoints(sink, targets);

    // Hook up fileDecl's scope to module's scope.
    for (auto globalDecl : moduleDecl->members)
    {
        if (auto fileDecl = as<FileDecl>(globalDecl))
        {
            addSiblingScopeForContainerDecl(m_astBuilder, moduleDecl->ownedScope, fileDecl);
        }
    }

    return SLANG_OK;
}

void Linkage::setRequireCacheFileSystem(bool requireCacheFileSystem)
{
    if (requireCacheFileSystem == m_requireCacheFileSystem)
    {
        return;
    }

    ComPtr<ISlangFileSystem> scopeFileSystem(m_fileSystem);
    m_requireCacheFileSystem = requireCacheFileSystem;

    setFileSystem(scopeFileSystem);
}

RefPtr<Module> findOrImportModule(
    Linkage* linkage,
    Name* name,
    SourceLoc const& loc,
    DiagnosticSink* sink,
    const LoadedModuleDictionary* loadedModules)
{
    return linkage->findOrImportModule(name, loc, sink, loadedModules);
}

void Session::addBuiltinSource(
    Scope* scope,
    String const& path,
    ISlangBlob* sourceBlob,
    Module*& outModule)
{
    SourceManager* sourceManager = getBuiltinSourceManager();

    DiagnosticSink sink(sourceManager, Lexer::sourceLocationLexer);

    RefPtr<FrontEndCompileRequest> compileRequest =
        new FrontEndCompileRequest(m_builtinLinkage, nullptr, &sink);
    compileRequest->m_isCoreModuleCode = true;

    // Set the source manager on the sink
    sink.setSourceManager(sourceManager);
    // Make the linkage use the builtin source manager
    Linkage* linkage = compileRequest->getLinkage();
    linkage->setSourceManager(sourceManager);

    Name* moduleName = getNamePool()->getName(path);
    auto translationUnitIndex =
        compileRequest->addTranslationUnit(SourceLanguage::Slang, moduleName);

    compileRequest->addTranslationUnitSourceBlob(translationUnitIndex, path, sourceBlob);

    SlangResult res = compileRequest->executeActionsInner();
    if (SLANG_FAILED(res))
    {
        char const* diagnostics = sink.outputBuffer.getBuffer();
        fprintf(stderr, "%s", diagnostics);

        PlatformUtil::outputDebugMessage(diagnostics);

        SLANG_UNEXPECTED("error in Slang core module");
    }

    // Compiling the core module should not yield any warnings.
    SLANG_ASSERT(sink.outputBuffer.getLength() == 0);

    // Extract the AST for the code we just parsed
    auto module = compileRequest->translationUnits[translationUnitIndex]->getModule();
    auto moduleDecl = module->getModuleDecl();

    // Extact documentation markup.
    ASTMarkup markup;
    ASTMarkupUtil::extract(moduleDecl, sourceManager, &sink, &markup);
    markup.attachToAST();

    // Put in the loaded module map
    linkage->mapNameToLoadedModules.add(moduleName, module);

    // Add the resulting code to the appropriate scope
    if (!scope->containerDecl)
    {
        // We are the first chunk of code to be loaded for this scope
        scope->containerDecl = moduleDecl;
    }
    else
    {
        // We need to create a new scope to link into the whole thing
        auto subScope = module->getASTBuilder()->create<Scope>();
        subScope->containerDecl = moduleDecl;
        subScope->nextSibling = scope->nextSibling;
        scope->nextSibling = subScope;
    }

    outModule = module;
}

Session::~Session()
{
    // This is necessary because this ASTBuilder uses the SharedASTBuilder also owned by the
    // session. If the SharedASTBuilder gets dtored before the globalASTBuilder it has a dangling
    // pointer, which is referenced in the ASTBuilder dtor (likely) causing a crash.
    //
    // By destroying first we know it is destroyed, before the SharedASTBuilder.
    globalAstBuilder.setNull();

    // destroy modules next
    coreModules = decltype(coreModules)();
}

} // namespace Slang


/* !!!!!!!!!!!!!!!!!! EndToEndCompileRequestImpl !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

namespace Slang
{

void EndToEndCompileRequest::setFileSystem(ISlangFileSystem* fileSystem)
{
    getLinkage()->setFileSystem(fileSystem);
}

void EndToEndCompileRequest::setCompileFlags(SlangCompileFlags flags)
{
    if (flags & SLANG_COMPILE_FLAG_NO_MANGLING)
        getOptionSet().set(CompilerOptionName::NoMangle, true);
    if (flags & SLANG_COMPILE_FLAG_NO_CODEGEN)
        getOptionSet().set(CompilerOptionName::SkipCodeGen, true);
    if (flags & SLANG_COMPILE_FLAG_OBFUSCATE)
        getOptionSet().set(CompilerOptionName::Obfuscate, true);
}

SlangCompileFlags EndToEndCompileRequest::getCompileFlags()
{
    SlangCompileFlags result = 0;
    if (getOptionSet().getBoolOption(CompilerOptionName::NoMangle))
        result |= SLANG_COMPILE_FLAG_NO_MANGLING;
    if (getOptionSet().getBoolOption(CompilerOptionName::SkipCodeGen))
        result |= SLANG_COMPILE_FLAG_NO_CODEGEN;
    if (getOptionSet().getBoolOption(CompilerOptionName::Obfuscate))
        result |= SLANG_COMPILE_FLAG_OBFUSCATE;
    return result;
}

void EndToEndCompileRequest::setDumpIntermediates(int enable)
{
    getOptionSet().set(CompilerOptionName::DumpIntermediates, enable);
}

void EndToEndCompileRequest::setTrackLiveness(bool v)
{
    getOptionSet().set(CompilerOptionName::TrackLiveness, v);
}

void EndToEndCompileRequest::setDumpIntermediatePrefix(const char* prefix)
{
    getOptionSet().set(CompilerOptionName::DumpIntermediatePrefix, String(prefix));
}

void EndToEndCompileRequest::setLineDirectiveMode(SlangLineDirectiveMode mode)
{
    getOptionSet().set(CompilerOptionName::LineDirectiveMode, mode);
}

void EndToEndCompileRequest::setCommandLineCompilerMode()
{
    m_isCommandLineCompile = true;

    // legacy slangc tool defaults to column major layout.
    if (!getOptionSet().hasOption(CompilerOptionName::MatrixLayoutRow))
        getOptionSet().setMatrixLayoutMode(kMatrixLayoutMode_ColumnMajor);
}

void EndToEndCompileRequest::_completeTargetRequest(UInt targetIndex)
{
    auto linkage = getLinkage();

    TargetRequest* targetRequest = linkage->targets[Index(targetIndex)];

    targetRequest->getOptionSet().inheritFrom(getLinkage()->m_optionSet);
    targetRequest->getOptionSet().inheritFrom(m_optionSetForDefaultTarget);
}

void EndToEndCompileRequest::setCodeGenTarget(SlangCompileTarget target)
{
    auto linkage = getLinkage();
    linkage->targets.clear();
    const auto targetIndex = linkage->addTarget(CodeGenTarget(target));
    SLANG_ASSERT(targetIndex == 0);
    _completeTargetRequest(0);
}

int EndToEndCompileRequest::addCodeGenTarget(SlangCompileTarget target)
{
    const auto targetIndex = getLinkage()->addTarget(CodeGenTarget(target));
    _completeTargetRequest(targetIndex);
    return int(targetIndex);
}

void EndToEndCompileRequest::setTargetProfile(int targetIndex, SlangProfileID profile)
{
    getTargetOptionSet(targetIndex).setProfile(Profile(profile));
}

void EndToEndCompileRequest::setTargetFlags(int targetIndex, SlangTargetFlags flags)
{
    getTargetOptionSet(targetIndex).setTargetFlags(flags);
}

void EndToEndCompileRequest::setTargetForceGLSLScalarBufferLayout(int targetIndex, bool value)
{
    getTargetOptionSet(targetIndex).set(CompilerOptionName::GLSLForceScalarLayout, value);
}

void EndToEndCompileRequest::setTargetForceDXLayout(int targetIndex, bool value)
{
    getTargetOptionSet(targetIndex).set(CompilerOptionName::ForceDXLayout, value);
}

void EndToEndCompileRequest::setTargetFloatingPointMode(
    int targetIndex,
    SlangFloatingPointMode mode)
{
    getTargetOptionSet(targetIndex)
        .set(CompilerOptionName::FloatingPointMode, FloatingPointMode(mode));
}

void EndToEndCompileRequest::setMatrixLayoutMode(SlangMatrixLayoutMode mode)
{
    getOptionSet().setMatrixLayoutMode((MatrixLayoutMode)mode);
}

void EndToEndCompileRequest::setTargetMatrixLayoutMode(int targetIndex, SlangMatrixLayoutMode mode)
{
    getTargetOptionSet(targetIndex).setMatrixLayoutMode(MatrixLayoutMode(mode));
}

void EndToEndCompileRequest::setTargetGenerateWholeProgram(int targetIndex, bool value)
{
    getTargetOptionSet(targetIndex).set(CompilerOptionName::GenerateWholeProgram, value);
}

void EndToEndCompileRequest::setTargetEmbedDownstreamIR(int targetIndex, bool value)
{
    getTargetOptionSet(targetIndex).set(CompilerOptionName::EmbedDownstreamIR, value);
}

void EndToEndCompileRequest::setTargetLineDirectiveMode(
    SlangInt targetIndex,
    SlangLineDirectiveMode mode)
{
    getTargetOptionSet(targetIndex)
        .set(CompilerOptionName::LineDirectiveMode, LineDirectiveMode(mode));
}

void EndToEndCompileRequest::overrideDiagnosticSeverity(
    SlangInt messageID,
    SlangSeverity overrideSeverity)
{
    getSink()->overrideDiagnosticSeverity(int(messageID), Severity(overrideSeverity));
}

SlangDiagnosticFlags EndToEndCompileRequest::getDiagnosticFlags()
{
    DiagnosticSink::Flags sinkFlags = getSink()->getFlags();

    SlangDiagnosticFlags flags = 0;

    if (sinkFlags & DiagnosticSink::Flag::VerbosePath)
        flags |= SLANG_DIAGNOSTIC_FLAG_VERBOSE_PATHS;

    if (sinkFlags & DiagnosticSink::Flag::TreatWarningsAsErrors)
        flags |= SLANG_DIAGNOSTIC_FLAG_TREAT_WARNINGS_AS_ERRORS;

    return flags;
}

void EndToEndCompileRequest::setDiagnosticFlags(SlangDiagnosticFlags flags)
{
    DiagnosticSink::Flags sinkFlags = getSink()->getFlags();

    if (flags & SLANG_DIAGNOSTIC_FLAG_VERBOSE_PATHS)
        sinkFlags |= DiagnosticSink::Flag::VerbosePath;
    else
        sinkFlags &= ~DiagnosticSink::Flag::VerbosePath;

    if (flags & SLANG_DIAGNOSTIC_FLAG_TREAT_WARNINGS_AS_ERRORS)
        sinkFlags |= DiagnosticSink::Flag::TreatWarningsAsErrors;
    else
        sinkFlags &= ~DiagnosticSink::Flag::TreatWarningsAsErrors;

    getSink()->setFlags(sinkFlags);
}

SlangResult EndToEndCompileRequest::addTargetCapability(
    SlangInt targetIndex,
    SlangCapabilityID capability)
{
    auto& targets = getLinkage()->targets;
    if (targetIndex < 0 || targetIndex >= targets.getCount())
        return SLANG_E_INVALID_ARG;
    getTargetOptionSet(targetIndex).addCapabilityAtom(CapabilityName(capability));
    return SLANG_OK;
}

void EndToEndCompileRequest::setDebugInfoLevel(SlangDebugInfoLevel level)
{
    getOptionSet().set(CompilerOptionName::DebugInformation, DebugInfoLevel(level));
}

void EndToEndCompileRequest::setDebugInfoFormat(SlangDebugInfoFormat format)
{
    getOptionSet().set(CompilerOptionName::DebugInformationFormat, DebugInfoFormat(format));
}

void EndToEndCompileRequest::setOptimizationLevel(SlangOptimizationLevel level)
{
    getOptionSet().set(CompilerOptionName::Optimization, OptimizationLevel(level));
}

void EndToEndCompileRequest::setOutputContainerFormat(SlangContainerFormat format)
{
    m_containerFormat = ContainerFormat(format);
}

void EndToEndCompileRequest::setPassThrough(SlangPassThrough inPassThrough)
{
    m_passThrough = PassThroughMode(inPassThrough);
}

void EndToEndCompileRequest::setReportDownstreamTime(bool value)
{
    getOptionSet().set(CompilerOptionName::ReportDownstreamTime, value);
}

void EndToEndCompileRequest::setReportPerfBenchmark(bool value)
{
    getOptionSet().set(CompilerOptionName::ReportPerfBenchmark, value);
}

void EndToEndCompileRequest::setSkipSPIRVValidation(bool value)
{
    getOptionSet().set(CompilerOptionName::SkipSPIRVValidation, value);
}

void EndToEndCompileRequest::setTargetUseMinimumSlangOptimization(int targetIndex, bool value)
{
    getTargetOptionSet(targetIndex).set(CompilerOptionName::MinimumSlangOptimization, value);
}

void EndToEndCompileRequest::setIgnoreCapabilityCheck(bool value)
{
    getOptionSet().set(CompilerOptionName::IgnoreCapabilities, value);
}

void EndToEndCompileRequest::setDiagnosticCallback(
    SlangDiagnosticCallback callback,
    void const* userData)
{
    ComPtr<ISlangWriter> writer(new CallbackWriter(callback, userData, WriterFlag::IsConsole));
    setWriter(WriterChannel::Diagnostic, writer);
}

void EndToEndCompileRequest::setWriter(SlangWriterChannel chan, ISlangWriter* writer)
{
    setWriter(WriterChannel(chan), writer);
}

ISlangWriter* EndToEndCompileRequest::getWriter(SlangWriterChannel chan)
{
    return getWriter(WriterChannel(chan));
}

void EndToEndCompileRequest::addSearchPath(const char* path)
{
    getOptionSet().addSearchPath(path);
}

void EndToEndCompileRequest::addPreprocessorDefine(const char* key, const char* value)
{
    getOptionSet().addPreprocessorDefine(key, value);
}

void EndToEndCompileRequest::setEnableEffectAnnotations(bool value)
{
    getOptionSet().set(CompilerOptionName::EnableEffectAnnotations, value);
}

char const* EndToEndCompileRequest::getDiagnosticOutput()
{
    return m_diagnosticOutput.begin();
}

SlangResult EndToEndCompileRequest::getDiagnosticOutputBlob(ISlangBlob** outBlob)
{
    if (!outBlob)
        return SLANG_E_INVALID_ARG;

    if (!m_diagnosticOutputBlob)
    {
        m_diagnosticOutputBlob = StringUtil::createStringBlob(m_diagnosticOutput);
    }

    ComPtr<ISlangBlob> resultBlob = m_diagnosticOutputBlob;
    *outBlob = resultBlob.detach();
    return SLANG_OK;
}

int EndToEndCompileRequest::addTranslationUnit(SlangSourceLanguage language, char const* inName)
{
    auto frontEndReq = getFrontEndReq();
    NamePool* namePool = frontEndReq->getNamePool();

    // Work out a module name. Can be nullptr if so will generate a name
    Name* moduleName = inName ? namePool->getName(inName) : frontEndReq->m_defaultModuleName;

    // If moduleName is nullptr a name will be generated
    return frontEndReq->addTranslationUnit(Slang::SourceLanguage(language), moduleName);
}

void EndToEndCompileRequest::setDefaultModuleName(const char* defaultModuleName)
{
    auto frontEndReq = getFrontEndReq();
    NamePool* namePool = frontEndReq->getNamePool();
    frontEndReq->m_defaultModuleName = namePool->getName(defaultModuleName);
}

SlangResult _addLibraryReference(
    EndToEndCompileRequest* req,
    ModuleLibrary* moduleLibrary,
    bool includeEntryPoint)
{
    FrontEndCompileRequest* frontEndRequest = req->getFrontEndReq();

    if (includeEntryPoint)
    {
        frontEndRequest->m_extraEntryPoints.addRange(
            moduleLibrary->m_entryPoints.getBuffer(),
            moduleLibrary->m_entryPoints.getCount());
    }

    for (auto m : moduleLibrary->m_modules)
    {
        RefPtr<TranslationUnitRequest> tu = new TranslationUnitRequest(frontEndRequest, m);
        frontEndRequest->translationUnits.add(tu);
        // For modules loaded for EndToEndCompileRequest,
        // we don't need the automatically discovered entrypoints.
        if (!includeEntryPoint)
            m->getEntryPoints().clear();
    }
    return SLANG_OK;
}

SlangResult _addLibraryReference(
    EndToEndCompileRequest* req,
    String path,
    IArtifact* artifact,
    bool includeEntryPoint)
{
    auto desc = artifact->getDesc();

    // TODO(JS):
    // This isn't perhaps the best way to handle this scenario, as IArtifact can
    // support lazy evaluation, with suitable hander.
    // For now we just read in and strip out the bits we want.
    if (isDerivedFrom(desc.kind, ArtifactKind::Container) &&
        isDerivedFrom(desc.payload, ArtifactPayload::CompileResults))
    {
        // We want to read as a file system
        ComPtr<IArtifact> container;

        SLANG_RETURN_ON_FAIL(ArtifactContainerUtil::readContainer(artifact, container));

        // Find the payload... It should be linkable
        if (!ArtifactDescUtil::isLinkable(container->getDesc()))
        {
            return SLANG_FAIL;
        }

        ComPtr<IModuleLibrary> libraryIntf;
        SLANG_RETURN_ON_FAIL(
            loadModuleLibrary(ArtifactKeep::Yes, container, path, req, libraryIntf));

        auto library = as<ModuleLibrary>(libraryIntf);

        // Look for source maps
        for (auto associated : container->getAssociated())
        {
            auto assocDesc = associated->getDesc();

            // If we find an obfuscated source map load it and associate
            if (isDerivedFrom(assocDesc.kind, ArtifactKind::Json) &&
                isDerivedFrom(assocDesc.payload, ArtifactPayload::SourceMap) &&
                isDerivedFrom(assocDesc.style, ArtifactStyle::Obfuscated))
            {
                ComPtr<ICastable> castable;
                SLANG_RETURN_ON_FAIL(associated->getOrCreateRepresentation(
                    SourceMap::getTypeGuid(),
                    ArtifactKeep::Yes,
                    castable.writeRef()));
                auto sourceMap = asBoxValue<SourceMap>(castable);
                SLANG_ASSERT(sourceMap);

                // TODO(JS):
                // There is perhaps (?) a risk here that we might copy the obfuscated map
                // into some output container. Currently that only happens for source maps
                // that are from translation units.
                //
                // On the other hand using "import" is a way that such source maps *would* be
                // copied into the output, and that is something that could be a vector
                // for leaking.
                //
                // That isn't a risk from -r though because, it doesn't create a translation
                // unit(s).
                for (auto module : library->m_modules)
                {
                    module->getIRModule()->setObfuscatedSourceMap(sourceMap);
                }

                // Look up the source file
                auto sourceManager = req->getSink()->getSourceManager();

                auto name = Path::getFileNameWithoutExt(associated->getName());

                if (name.getLength())
                {
                    auto sourceFile = sourceManager->findSourceFileByPathRecursively(name);
                    sourceFile->setSourceMap(sourceMap, SourceMapKind::Obfuscated);
                }
            }
        }

        SLANG_RETURN_ON_FAIL(_addLibraryReference(req, library, includeEntryPoint));
        return SLANG_OK;
    }

    if (desc.kind == ArtifactKind::Library && desc.payload == ArtifactPayload::SlangIR)
    {
        ComPtr<IModuleLibrary> libraryIntf;

        SLANG_RETURN_ON_FAIL(
            loadModuleLibrary(ArtifactKeep::Yes, artifact, path, req, libraryIntf));

        auto library = as<ModuleLibrary>(libraryIntf);
        if (!library)
        {
            return SLANG_FAIL;
        }

        SLANG_RETURN_ON_FAIL(_addLibraryReference(req, library, includeEntryPoint));
        return SLANG_OK;
    }

    // TODO(JS):
    // Do we want to check the path exists?

    // Add to the m_libModules
    auto linkage = req->getLinkage();
    linkage->m_libModules.add(ComPtr<IArtifact>(artifact));

    return SLANG_OK;
}

SlangResult EndToEndCompileRequest::addLibraryReference(
    const char* basePath,
    const void* libData,
    size_t libDataSize)
{
    // We need to deserialize and add the modules
    ComPtr<IModuleLibrary> library;

    SLANG_RETURN_ON_FAIL(
        loadModuleLibrary((const Byte*)libData, libDataSize, basePath, this, library));

    // Create an artifact without any name (as one is not provided)
    auto artifact =
        Artifact::create(ArtifactDesc::make(ArtifactKind::Library, ArtifactPayload::SlangIR));
    artifact->addRepresentation(library);

    return _addLibraryReference(this, basePath, artifact, true);
}

void EndToEndCompileRequest::addTranslationUnitPreprocessorDefine(
    int translationUnitIndex,
    const char* key,
    const char* value)
{
    getFrontEndReq()->translationUnits[translationUnitIndex]->preprocessorDefinitions[key] = value;
}

void EndToEndCompileRequest::addTranslationUnitSourceFile(
    int translationUnitIndex,
    char const* path)
{
    auto frontEndReq = getFrontEndReq();
    if (!path)
        return;
    if (translationUnitIndex < 0)
        return;
    if (Index(translationUnitIndex) >= frontEndReq->translationUnits.getCount())
        return;

    frontEndReq->addTranslationUnitSourceFile(translationUnitIndex, path);
}

void EndToEndCompileRequest::addTranslationUnitSourceString(
    int translationUnitIndex,
    char const* path,
    char const* source)
{
    if (!source)
        return;
    addTranslationUnitSourceStringSpan(translationUnitIndex, path, source, source + strlen(source));
}

void EndToEndCompileRequest::addTranslationUnitSourceStringSpan(
    int translationUnitIndex,
    char const* path,
    char const* sourceBegin,
    char const* sourceEnd)
{
    auto frontEndReq = getFrontEndReq();
    if (!sourceBegin)
        return;
    if (translationUnitIndex < 0)
        return;
    if (Index(translationUnitIndex) >= frontEndReq->translationUnits.getCount())
        return;

    if (!path)
        path = "";

    const auto slice = UnownedStringSlice(sourceBegin, sourceEnd);

    auto blob = RawBlob::create(slice.begin(), slice.getLength());

    frontEndReq->addTranslationUnitSourceBlob(translationUnitIndex, path, blob);
}

void EndToEndCompileRequest::addTranslationUnitSourceBlob(
    int translationUnitIndex,
    char const* path,
    ISlangBlob* sourceBlob)
{
    auto frontEndReq = getFrontEndReq();
    if (!sourceBlob)
        return;
    if (translationUnitIndex < 0)
        return;
    if (Slang::Index(translationUnitIndex) >= frontEndReq->translationUnits.getCount())
        return;

    if (!path)
        path = "";

    frontEndReq->addTranslationUnitSourceBlob(translationUnitIndex, path, sourceBlob);
}


int EndToEndCompileRequest::addEntryPoint(
    int translationUnitIndex,
    char const* name,
    SlangStage stage)
{
    return addEntryPointEx(translationUnitIndex, name, stage, 0, nullptr);
}

int EndToEndCompileRequest::addEntryPointEx(
    int translationUnitIndex,
    char const* name,
    SlangStage stage,
    int genericParamTypeNameCount,
    char const** genericParamTypeNames)
{
    auto frontEndReq = getFrontEndReq();
    if (!name)
        return -1;
    if (translationUnitIndex < 0)
        return -1;
    if (Index(translationUnitIndex) >= frontEndReq->translationUnits.getCount())
        return -1;

    List<String> typeNames;
    for (int i = 0; i < genericParamTypeNameCount; i++)
        typeNames.add(genericParamTypeNames[i]);

    return addEntryPoint(translationUnitIndex, name, Profile(Stage(stage)), typeNames);
}

SlangResult EndToEndCompileRequest::setGlobalGenericArgs(
    int genericArgCount,
    char const** genericArgs)
{
    auto& argStrings = m_globalSpecializationArgStrings;
    argStrings.clear();
    for (int i = 0; i < genericArgCount; i++)
        argStrings.add(genericArgs[i]);

    return SLANG_OK;
}

SlangResult EndToEndCompileRequest::setTypeNameForGlobalExistentialTypeParam(
    int slotIndex,
    char const* typeName)
{
    if (slotIndex < 0)
        return SLANG_FAIL;
    if (!typeName)
        return SLANG_FAIL;

    auto& typeArgStrings = m_globalSpecializationArgStrings;
    if (Index(slotIndex) >= typeArgStrings.getCount())
        typeArgStrings.setCount(slotIndex + 1);
    typeArgStrings[slotIndex] = String(typeName);
    return SLANG_OK;
}

SlangResult EndToEndCompileRequest::setTypeNameForEntryPointExistentialTypeParam(
    int entryPointIndex,
    int slotIndex,
    char const* typeName)
{
    if (entryPointIndex < 0)
        return SLANG_FAIL;
    if (slotIndex < 0)
        return SLANG_FAIL;
    if (!typeName)
        return SLANG_FAIL;

    if (Index(entryPointIndex) >= m_entryPoints.getCount())
        return SLANG_FAIL;

    auto& entryPointInfo = m_entryPoints[entryPointIndex];
    auto& typeArgStrings = entryPointInfo.specializationArgStrings;
    if (Index(slotIndex) >= typeArgStrings.getCount())
        typeArgStrings.setCount(slotIndex + 1);
    typeArgStrings[slotIndex] = String(typeName);
    return SLANG_OK;
}

void EndToEndCompileRequest::setAllowGLSLInput(bool value)
{
    getOptionSet().set(CompilerOptionName::AllowGLSL, value);
}

SlangResult EndToEndCompileRequest::compile()
{
    SlangResult res = SLANG_FAIL;
    double downstreamStartTime = 0.0;
    double totalStartTime = 0.0;

    if (getOptionSet().getBoolOption(CompilerOptionName::ReportDownstreamTime))
    {
        getSession()->getCompilerElapsedTime(&totalStartTime, &downstreamStartTime);
        PerformanceProfiler::getProfiler()->clear();
    }
#if !defined(SLANG_DEBUG_INTERNAL_ERROR)
    // By default we'd like to catch as many internal errors as possible,
    // and report them to the user nicely (rather than just crash their
    // application). Internally Slang currently uses exceptions for this.
    //
    // TODO: Consider using `setjmp()`-style escape so that we can work
    // with applications that disable exceptions.
    //
    // TODO: Consider supporting Windows "Structured Exception Handling"
    // so that we can also recover from a wider class of crashes.

    try
    {
        SLANG_PROFILE_SECTION(compileInner);
        res = executeActions();
    }
    catch (const AbortCompilationException& e)
    {
        // This situation indicates a fatal (but not necessarily internal) error
        // that forced compilation to terminate. There should already have been
        // a diagnostic produced, so we don't need to add one here.
        if (getSink()->getErrorCount() == 0)
        {
            // If for some reason we didn't output any diagnostic, something is
            // going wrong, but we want to make sure we at least output something.
            getSink()->diagnose(
                SourceLoc(),
                Diagnostics::compilationAbortedDueToException,
                typeid(e).name(),
                e.Message);
        }
    }
    catch (const Exception& e)
    {
        // The compiler failed due to an internal error that was detected.
        // We will print out information on the exception to help out the user
        // in either filing a bug, or locating what in their code created
        // a problem.
        getSink()->diagnose(
            SourceLoc(),
            Diagnostics::compilationAbortedDueToException,
            typeid(e).name(),
            e.Message);
    }
    catch (...)
    {
        // The compiler failed due to some exception that wasn't a sublass of
        // `Exception`, so something really fishy is going on. We want to
        // let the user know that we messed up, so they know to blame Slang
        // and not some other component in their system.
        getSink()->diagnose(SourceLoc(), Diagnostics::compilationAborted);
    }
    m_diagnosticOutput = getSink()->outputBuffer.produceString();

#else
    // When debugging, we probably don't want to filter out any errors, since
    // we are probably trying to root-cause and *fix* those errors.
    {
        res = req->executeActions();
    }
#endif

    if (getOptionSet().getBoolOption(CompilerOptionName::ReportDownstreamTime))
    {
        double downstreamEndTime = 0;
        double totalEndTime = 0;
        getSession()->getCompilerElapsedTime(&totalEndTime, &downstreamEndTime);
        double downstreamTime = downstreamEndTime - downstreamStartTime;
        String downstreamTimeStr = String(downstreamTime, "%.2f");
        getSink()->diagnose(SourceLoc(), Diagnostics::downstreamCompileTime, downstreamTimeStr);
    }
    if (getOptionSet().getBoolOption(CompilerOptionName::ReportPerfBenchmark))
    {
        StringBuilder perfResult;
        PerformanceProfiler::getProfiler()->getResult(perfResult);
        perfResult << "\nType Dictionary Size: " << getSession()->m_typeDictionarySize << "\n";
        getSink()->diagnose(
            SourceLoc(),
            Diagnostics::performanceBenchmarkResult,
            perfResult.produceString());
    }

    // Repro dump handling
    {
        auto dumpRepro = getOptionSet().getStringOption(CompilerOptionName::DumpRepro);
        auto dumpReproOnError = getOptionSet().getBoolOption(CompilerOptionName::DumpReproOnError);

        if (dumpRepro.getLength())
        {
            SlangResult saveRes = ReproUtil::saveState(this, dumpRepro);
            if (SLANG_FAILED(saveRes))
            {
                getSink()->diagnose(SourceLoc(), Diagnostics::unableToWriteReproFile, dumpRepro);
                return saveRes;
            }
        }
        else if (dumpReproOnError && SLANG_FAILED(res))
        {
            String reproFileName;
            SlangResult saveRes = SLANG_FAIL;

            RefPtr<Stream> stream;
            if (SLANG_SUCCEEDED(ReproUtil::findUniqueReproDumpStream(this, reproFileName, stream)))
            {
                saveRes = ReproUtil::saveState(this, stream);
            }

            if (SLANG_FAILED(saveRes))
            {
                getSink()->diagnose(
                    SourceLoc(),
                    Diagnostics::unableToWriteReproFile,
                    reproFileName);
            }
        }
    }

    auto reflectionPath = getOptionSet().getStringOption(CompilerOptionName::EmitReflectionJSON);
    if (reflectionPath.getLength() != 0)
    {
        auto bufferWriter = PrettyWriter();
        emitReflectionJSON(this, this->getReflection(), bufferWriter);
        if (reflectionPath == "-")
        {
            auto builder = bufferWriter.getBuilder();
            StdWriters::getOut().write(builder.getBuffer(), builder.getLength());
        }
        else if (SLANG_FAILED(File::writeAllText(reflectionPath, bufferWriter.getBuilder())))
        {
            getSink()->diagnose(SourceLoc(), Diagnostics::unableToWriteFile, reflectionPath);
        }
    }

    return res;
}

int EndToEndCompileRequest::getDependencyFileCount()
{
    auto frontEndReq = getFrontEndReq();
    auto program = frontEndReq->getGlobalAndEntryPointsComponentType();
    return (int)program->getFileDependencies().getCount();
}

char const* EndToEndCompileRequest::getDependencyFilePath(int index)
{
    auto frontEndReq = getFrontEndReq();
    auto program = frontEndReq->getGlobalAndEntryPointsComponentType();
    SourceFile* sourceFile = program->getFileDependencies()[index];
    return sourceFile->getPathInfo().hasFoundPath()
               ? sourceFile->getPathInfo().getMostUniqueIdentity().getBuffer()
               : "unknown";
}

int EndToEndCompileRequest::getTranslationUnitCount()
{
    return (int)getFrontEndReq()->translationUnits.getCount();
}

void const* EndToEndCompileRequest::getEntryPointCode(int entryPointIndex, size_t* outSize)
{
    // Zero the size initially, in case need to return nullptr for error.
    if (outSize)
    {
        *outSize = 0;
    }

    auto linkage = getLinkage();
    auto program = getSpecializedGlobalAndEntryPointsComponentType();

    // TODO: We should really accept a target index in this API
    Index targetIndex = 0;
    auto targetCount = linkage->targets.getCount();
    if (targetIndex >= targetCount)
        return nullptr;
    auto targetReq = linkage->targets[targetIndex];


    if (entryPointIndex < 0)
        return nullptr;
    if (Index(entryPointIndex) >= program->getEntryPointCount())
        return nullptr;
    auto entryPoint = program->getEntryPoint(entryPointIndex);

    auto targetProgram = program->getTargetProgram(targetReq);
    if (!targetProgram)
        return nullptr;
    IArtifact* artifact = targetProgram->getExistingEntryPointResult(entryPointIndex);
    if (!artifact)
    {
        return nullptr;
    }

    ComPtr<ISlangBlob> blob;
    SLANG_RETURN_NULL_ON_FAIL(artifact->loadBlob(ArtifactKeep::Yes, blob.writeRef()));

    if (outSize)
    {
        *outSize = blob->getBufferSize();
    }

    return (void*)blob->getBufferPointer();
}

SlangResult EndToEndCompileRequest::getCompileTimeProfile(
    ISlangProfiler** compileTimeProfile,
    bool shouldClear)
{
    if (compileTimeProfile == nullptr)
    {
        return SLANG_E_INVALID_ARG;
    }

    SlangProfiler* profiler = new SlangProfiler(PerformanceProfiler::getProfiler());

    if (shouldClear)
    {
        PerformanceProfiler::getProfiler()->clear();
    }

    ComPtr<ISlangProfiler> result(profiler);
    *compileTimeProfile = result.detach();
    return SLANG_OK;
}

static SlangResult _getEntryPointResult(
    EndToEndCompileRequest* req,
    int entryPointIndex,
    int targetIndex,
    ComPtr<IArtifact>& outArtifact)
{
    auto linkage = req->getLinkage();
    auto program = req->getSpecializedGlobalAndEntryPointsComponentType();

    Index targetCount = linkage->targets.getCount();
    if ((targetIndex < 0) || (targetIndex >= targetCount))
    {
        return SLANG_E_INVALID_ARG;
    }
    auto targetReq = linkage->targets[targetIndex];

    // Get the entry point count on the program, rather than (say) req->m_entryPoints.getCount()
    // because
    // 1) The entry point is fetched from the program anyway so must be consistent
    // 2) The req may not have all entry points (for example when an entry point is in a module)
    const Index entryPointCount = program->getEntryPointCount();

    if ((entryPointIndex < 0) || (entryPointIndex >= entryPointCount))
    {
        return SLANG_E_INVALID_ARG;
    }
    auto entryPointReq = program->getEntryPoint(entryPointIndex);

    auto targetProgram = program->getTargetProgram(targetReq);
    if (!targetProgram)
        return SLANG_FAIL;

    outArtifact = targetProgram->getExistingEntryPointResult(entryPointIndex);
    return SLANG_OK;
}

static SlangResult _getWholeProgramResult(
    EndToEndCompileRequest* req,
    int targetIndex,
    ComPtr<IArtifact>& outArtifact)
{
    auto linkage = req->getLinkage();
    auto program = req->getSpecializedGlobalAndEntryPointsComponentType();

    if (!program)
    {
        return SLANG_FAIL;
    }

    Index targetCount = linkage->targets.getCount();
    if ((targetIndex < 0) || (targetIndex >= targetCount))
    {
        return SLANG_E_INVALID_ARG;
    }
    auto targetReq = linkage->targets[targetIndex];

    auto targetProgram = program->getTargetProgram(targetReq);
    if (!targetProgram)
        return SLANG_FAIL;
    outArtifact = targetProgram->getExistingWholeProgramResult();
    return SLANG_OK;
}

SlangResult EndToEndCompileRequest::getEntryPointCodeBlob(
    int entryPointIndex,
    int targetIndex,
    ISlangBlob** outBlob)
{
    if (!outBlob)
        return SLANG_E_INVALID_ARG;
    ComPtr<IArtifact> artifact;
    SLANG_RETURN_ON_FAIL(_getEntryPointResult(this, entryPointIndex, targetIndex, artifact));
    SLANG_RETURN_ON_FAIL(artifact->loadBlob(ArtifactKeep::Yes, outBlob));

    return SLANG_OK;
}

SlangResult EndToEndCompileRequest::getEntryPointHostCallable(
    int entryPointIndex,
    int targetIndex,
    ISlangSharedLibrary** outSharedLibrary)
{
    if (!outSharedLibrary)
        return SLANG_E_INVALID_ARG;
    ComPtr<IArtifact> artifact;
    SLANG_RETURN_ON_FAIL(_getEntryPointResult(this, entryPointIndex, targetIndex, artifact));
    SLANG_RETURN_ON_FAIL(artifact->loadSharedLibrary(ArtifactKeep::Yes, outSharedLibrary));
    return SLANG_OK;
}

SlangResult EndToEndCompileRequest::getTargetCodeBlob(int targetIndex, ISlangBlob** outBlob)
{
    if (!outBlob)
        return SLANG_E_INVALID_ARG;

    ComPtr<IArtifact> artifact;
    SLANG_RETURN_ON_FAIL(_getWholeProgramResult(this, targetIndex, artifact));
    SLANG_RETURN_ON_FAIL(artifact->loadBlob(ArtifactKeep::Yes, outBlob));
    return SLANG_OK;
}

SlangResult EndToEndCompileRequest::getTargetHostCallable(
    int targetIndex,
    ISlangSharedLibrary** outSharedLibrary)
{
    if (!outSharedLibrary)
        return SLANG_E_INVALID_ARG;

    ComPtr<IArtifact> artifact;
    SLANG_RETURN_ON_FAIL(_getWholeProgramResult(this, targetIndex, artifact));
    SLANG_RETURN_ON_FAIL(artifact->loadSharedLibrary(ArtifactKeep::Yes, outSharedLibrary));
    return SLANG_OK;
}

char const* EndToEndCompileRequest::getEntryPointSource(int entryPointIndex)
{
    return (char const*)getEntryPointCode(entryPointIndex, nullptr);
}

ISlangMutableFileSystem* EndToEndCompileRequest::getCompileRequestResultAsFileSystem()
{
    if (!m_containerFileSystem)
    {
        if (m_containerArtifact)
        {
            ComPtr<ISlangMutableFileSystem> fileSystem(new MemoryFileSystem);

            // Filter the containerArtifact into things that can be written
            ComPtr<IArtifact> writeArtifact;
            if (SLANG_SUCCEEDED(
                    ArtifactContainerUtil::filter(m_containerArtifact, writeArtifact)) &&
                writeArtifact)
            {
                if (SLANG_SUCCEEDED(
                        ArtifactContainerUtil::writeContainer(writeArtifact, "", fileSystem)))
                {
                    m_containerFileSystem.swap(fileSystem);
                }
            }
        }
    }

    return m_containerFileSystem;
}

void const* EndToEndCompileRequest::getCompileRequestCode(size_t* outSize)
{
    if (m_containerArtifact)
    {
        ComPtr<ISlangBlob> containerBlob;
        if (SLANG_SUCCEEDED(
                m_containerArtifact->loadBlob(ArtifactKeep::Yes, containerBlob.writeRef())))
        {
            *outSize = containerBlob->getBufferSize();
            return containerBlob->getBufferPointer();
        }
    }

    // Container blob does not have any contents
    *outSize = 0;
    return nullptr;
}

SlangResult EndToEndCompileRequest::getContainerCode(ISlangBlob** outBlob)
{
    if (m_containerArtifact)
    {
        ComPtr<ISlangBlob> containerBlob;
        if (SLANG_SUCCEEDED(
                m_containerArtifact->loadBlob(ArtifactKeep::Yes, containerBlob.writeRef())))
        {
            *outBlob = containerBlob.detach();
            return SLANG_OK;
        }
    }
    return SLANG_FAIL;
}

SlangResult EndToEndCompileRequest::loadRepro(
    ISlangFileSystem* fileSystem,
    const void* data,
    size_t size)
{
    List<uint8_t> buffer;
    SLANG_RETURN_ON_FAIL(ReproUtil::loadState((const uint8_t*)data, size, getSink(), buffer));

    MemoryOffsetBase base;
    base.set(buffer.getBuffer(), buffer.getCount());

    ReproUtil::RequestState* requestState = ReproUtil::getRequest(buffer);

    SLANG_RETURN_ON_FAIL(ReproUtil::load(base, requestState, fileSystem, this));
    return SLANG_OK;
}

SlangResult EndToEndCompileRequest::saveRepro(ISlangBlob** outBlob)
{
    OwnedMemoryStream stream(FileAccess::Write);

    SLANG_RETURN_ON_FAIL(ReproUtil::saveState(this, &stream));

    // Put the content of the stream in the blob

    List<uint8_t> data;
    stream.swapContents(data);

    *outBlob = ListBlob::moveCreate(data).detach();
    return SLANG_OK;
}

SlangResult EndToEndCompileRequest::enableReproCapture()
{
    getLinkage()->setRequireCacheFileSystem(true);
    return SLANG_OK;
}

SlangResult EndToEndCompileRequest::processCommandLineArguments(
    char const* const* args,
    int argCount)
{
    return parseOptions(this, argCount, args);
}

SlangReflection* EndToEndCompileRequest::getReflection()
{
    auto linkage = getLinkage();
    auto program = getSpecializedGlobalAndEntryPointsComponentType();

    // Note(tfoley): The API signature doesn't let the client
    // specify which target they want to access reflection
    // information for, so for now we default to the first one.
    //
    // TODO: Add a new `spGetReflectionForTarget(req, targetIndex)`
    // so that we can do this better, and make it clear that
    // `spGetReflection()` is shorthand for `targetIndex == 0`.
    //
    Slang::Index targetIndex = 0;
    auto targetCount = linkage->targets.getCount();
    if (targetIndex >= targetCount)
        return nullptr;

    auto targetReq = linkage->targets[targetIndex];
    auto targetProgram = program->getTargetProgram(targetReq);


    DiagnosticSink sink(linkage->getSourceManager(), Lexer::sourceLocationLexer);
    auto programLayout = targetProgram->getOrCreateLayout(&sink);

    return (SlangReflection*)programLayout;
}

SlangResult EndToEndCompileRequest::getProgram(slang::IComponentType** outProgram)
{
    auto program = getSpecializedGlobalComponentType();
    *outProgram = Slang::ComPtr<slang::IComponentType>(program).detach();
    return SLANG_OK;
}

SlangResult EndToEndCompileRequest::getProgramWithEntryPoints(slang::IComponentType** outProgram)
{
    auto program = getSpecializedGlobalAndEntryPointsComponentType();
    *outProgram = Slang::ComPtr<slang::IComponentType>(program).detach();
    return SLANG_OK;
}

SlangResult EndToEndCompileRequest::getModule(
    SlangInt translationUnitIndex,
    slang::IModule** outModule)
{
    auto module = getFrontEndReq()->getTranslationUnit(translationUnitIndex)->getModule();

    *outModule = Slang::ComPtr<slang::IModule>(module).detach();
    return SLANG_OK;
}

SlangResult EndToEndCompileRequest::getSession(slang::ISession** outSession)
{
    auto session = getLinkage();
    *outSession = Slang::ComPtr<slang::ISession>(session).detach();
    return SLANG_OK;
}

SlangResult EndToEndCompileRequest::getEntryPoint(
    SlangInt entryPointIndex,
    slang::IComponentType** outEntryPoint)
{
    auto entryPoint = getSpecializedEntryPointComponentType(entryPointIndex);
    *outEntryPoint = Slang::ComPtr<slang::IComponentType>(entryPoint).detach();
    return SLANG_OK;
}

SlangResult EndToEndCompileRequest::isParameterLocationUsed(
    Int entryPointIndex,
    Int targetIndex,
    SlangParameterCategory category,
    UInt spaceIndex,
    UInt registerIndex,
    bool& outUsed)
{
    if (!ShaderBindingRange::isUsageTracked((slang::ParameterCategory)category))
        return SLANG_E_NOT_AVAILABLE;

    ComPtr<IArtifact> artifact;
    if (SLANG_FAILED(_getEntryPointResult(
            this,
            static_cast<int>(entryPointIndex),
            static_cast<int>(targetIndex),
            artifact)))
        return SLANG_E_INVALID_ARG;

    if (!artifact)
        return SLANG_E_NOT_AVAILABLE;

    // Find a rep
    auto metadata = findAssociatedRepresentation<IArtifactPostEmitMetadata>(artifact);
    if (!metadata)
        return SLANG_E_NOT_AVAILABLE;

    return metadata->isParameterLocationUsed(category, spaceIndex, registerIndex, outUsed);
}

} // namespace Slang
