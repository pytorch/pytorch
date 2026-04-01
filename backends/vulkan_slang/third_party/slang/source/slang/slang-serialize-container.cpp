// slang-serialize-container.cpp
#include "slang-serialize-container.h"

#include "../core/slang-byte-encode-util.h"
#include "../core/slang-math.h"
#include "../core/slang-stream.h"
#include "../core/slang-text-io.h"
#include "slang-check-impl.h"
#include "slang-compiler.h"
#include "slang-mangled-lexer.h"
#include "slang-parser.h"
#include "slang-serialize-ast.h"
#include "slang-serialize-ir.h"
#include "slang-serialize-source-loc.h"

namespace Slang
{
struct ModuleEncodingContext
{
public:
    ModuleEncodingContext(SerialContainerUtil::WriteOptions const& options, Stream* stream)
        : options(options), encoder(stream), containerStringPool(StringSlicePool::Style::Default)
    {
        if (options.optionFlags & SerialOptionFlag::SourceLocation)
        {
            sourceLocWriter = new SerialSourceLocWriter(options.sourceManager);
        }
    }

    ~ModuleEncodingContext()
    {
        encoder.setRIFFChunk(encoder.getRIFF()->getRoot());
        encodeFinalPieces();
    }

    SlangResult encodeModuleList(FrontEndCompileRequest* frontEndReq)
    {
        // Encoding a front-end compile request into a RIFF
        // is simply a matter of encoding the module for each
        // of the translation units that got compiled.
        //
        Encoder::WithKeyValuePair withArray(&encoder, SerialBinary::kModuleListFourCc);
        for (TranslationUnitRequest* translationUnit : frontEndReq->translationUnits)
        {
            SLANG_RETURN_ON_FAIL(encode(translationUnit->module));
        }
        return SLANG_OK;
    }

    SlangResult encode(FrontEndCompileRequest* frontEndReq)
    {
        Encoder::WithObject withObject(&encoder, SerialBinary::kContainerFourCc);
        SLANG_RETURN_ON_FAIL(encodeModuleList(frontEndReq));
        return SLANG_OK;
    }

    SlangResult encode(EndToEndCompileRequest* request)
    {
        Encoder::WithObject withObject(&encoder, SerialBinary::kContainerFourCc);

        // Encoding an end-to-end compile request starts with the same
        // work as for a front-end request: we encode each of
        // the modules for the translation units.
        //
        SLANG_RETURN_ON_FAIL(encodeModuleList(request->getFrontEndReq()));
        //
        // If code generation is disabled, then we can skip all further
        // steps, and the encoding process is no different
        // than for a front-end request.
        //
        if (request->getOptionSet().getBoolOption(CompilerOptionName::SkipCodeGen))
        {
            return SLANG_OK;
        }

        // If code generation is enabled, then we need to encode
        // information on each of the code generation targets, as well
        // as the entry points.
        //
        // We start with the targets, each of which will have a Slang IR
        // representation of the layout information for the program
        // on that target.
        //
        auto linkage = request->getLinkage();
        auto sink = request->getSink();
        auto program = request->getSpecializedGlobalAndEntryPointsComponentType();
        {
            Encoder::WithArray withArray(&encoder); // kContainerFourCc

            for (auto target : linkage->targets)
            {
                auto targetProgram = program->getTargetProgram(target);
                encode(targetProgram, sink);
            }
        }

        // The compiled `program` may also have zero or more entry points,
        // and we need to encode information about each of them.
        //
        {
            Encoder::WithArray withArray(&encoder, SerialBinary::kEntryPointListFourCc);

            auto entryPointCount = program->getEntryPointCount();
            for (Index ii = 0; ii < entryPointCount; ++ii)
            {
                auto entryPoint = program->getEntryPoint(ii);
                auto entryPointMangledName = program->getEntryPointMangledName(ii);
                encode(entryPoint, entryPointMangledName);
            }
        }

        return SLANG_OK;
    }

    SlangResult encode(TargetProgram* targetProgram, DiagnosticSink* sink)
    {
        // TODO:
        // Serialization of target component IR is causing the embedded precompiled binary
        // feature to fail. The resulting data modules contain both TU IR and TC IR, with only
        // one module header. Yong suggested to ignore the TC IR for now, though also that
        // OV was using the feature, so disabling this might cause problems.

        IRModule* irModule = targetProgram->getOrCreateIRModuleForLayout(sink);

        // Okay, we need to serialize this target program and its IR too...
        IRSerialData serialData;
        IRSerialWriter writer;

        SLANG_RETURN_ON_FAIL(
            writer.write(irModule, sourceLocWriter, options.optionFlags, &serialData));
        SLANG_RETURN_ON_FAIL(IRSerialWriter::writeContainer(serialData, encoder.getRIFF()));

        return SLANG_OK;
    }

    void encode(Name* name) { encoder.encode(name->text); }

    void encode(String const& value) { encoder.encode(value); }

    void encode(uint32_t value) { encoder.encode(UInt(value)); }

    void encodeData(void const* data, size_t size) { encoder.encodeData(data, size); }

    SlangResult encode(EntryPoint* entryPoint, String const& entryPointMangledName)
    {
        Encoder::WithObject withObject(&encoder, SerialBinary::kEntryPointFourCc);

        {
            Encoder::WithObject withProperty(&encoder, SerialBinary::kNameFourCC);
            encode(entryPoint->getName());
        }
        {
            Encoder::WithObject withProperty(&encoder, SerialBinary::kProfileFourCC);
            encode(entryPoint->getProfile().raw);
        }
        {
            Encoder::WithObject withProperty(&encoder, SerialBinary::kMangledNameFourCC);
            encode(entryPointMangledName);
        }

        return SLANG_OK;
    }


    SlangResult encode(Module* module)
    {
        if (!(options.optionFlags & (SerialOptionFlag::IRModule | SerialOptionFlag::ASTModule)))
            return SLANG_OK;

        Encoder::WithObject withModule(&encoder, SerialBinary::kModuleFourCC);

        // The first piece that we write for a module is its header.
        // The header is intended to provide information that can be
        // used to determine if a precompiled module is up-to-date.
        //
        // Update(tfoley): Okay, let's skip the whole header idea and just
        // serialize these things as properties of the module itself...
        {
            // So many things need the module name, that it makes
            // sense to serialize it separately from all the rest.
            //
            {
                Encoder::WithObject withProperty(&encoder, SerialBinary::kNameFourCC);
                encoder.encodeString(module->getNameObj()->text);
            }

            // The header includes a digest of all the compile options and
            // the files that the compiled result depended on.
            //
            auto digest = module->computeDigest();
            encoder.encodeData(PropertyKeys<Module>::Digest, digest.data, sizeof(digest.data));

            // The header includes an array of the paths of all of the
            // files that the compiled result depended on.
            //
            encodeModuleDependencyPaths(module);
        }

        // If serialization of Slang IR modules is enabled, and there
        // is IR available for this module, then we we encode it.
        //
        if ((options.optionFlags & SerialOptionFlag::IRModule))
        {
            if (auto irModule = module->getIRModule())
            {
                Encoder::WithKeyValuePair withKey(&encoder, PropertyKeys<Module>::IRModule);

                IRSerialData serialData;
                IRSerialWriter writer;
                SLANG_RETURN_ON_FAIL(
                    writer.write(irModule, sourceLocWriter, options.optionFlags, &serialData));
                SLANG_RETURN_ON_FAIL(IRSerialWriter::writeContainer(serialData, encoder.getRIFF()));
            }
        }

        // If serialization of AST information is enabled, and we have AST
        // information available, then we serialize it here.
        //
        if (options.optionFlags & SerialOptionFlag::ASTModule)
        {
            if (auto moduleDecl = module->getModuleDecl())
            {
                Encoder::WithKeyValuePair withKey(&encoder, PropertyKeys<Module>::ASTModule);

                writeSerializedModuleAST(&encoder, moduleDecl, sourceLocWriter);
            }
        }

        return SLANG_OK;
    }

    SlangResult encodeModuleDependencyPaths(Module* module)
    {
        Encoder::WithObject withProperty(&encoder, PropertyKeys<Module>::FileDependencies);

        // TODO(tfoley): This is some of the most complicated logic
        // in the encoding system, because it tries to translate
        // the file dependency paths into something that isn't
        // specific to the machine on which a module was built.
        //
        // The comments that follow are from the original implementation
        // of this logic, because I cannot state with confidence
        // that I know what's happening in all of this.


        // Here we assume that the first file in the file dependencies is the module's file path.
        // We store the module's file path as a relative path with respect to the first search
        // directory that contains the module, and store the paths of dependent files as relative
        // paths with respect to the module's path.

        // First, we try to extract the module's main file path from the file dependency list.
        auto fileDependencies = module->getFileDependencies();
        String canonicalModulePath, moduleDir;
        if (fileDependencies.getCount() != 0)
        {
            IncludeSystem includeSystem(
                &module->getLinkage()->getSearchDirectories(),
                module->getLinkage()->getFileSystemExt(),
                module->getLinkage()->getSourceManager());
            PathInfo outFoundSourcePath;
            // If we can find the first file, use that as the module's path.
            if (SLANG_SUCCEEDED(includeSystem.findFile(
                    fileDependencies[0]->getPathInfo().foundPath,
                    "",
                    outFoundSourcePath)))
            {
                canonicalModulePath = outFoundSourcePath.foundPath;
                Path::getCanonical(canonicalModulePath, canonicalModulePath);
                moduleDir = Path::getParentDirectory(canonicalModulePath);
            }
        }

        // If we can't find the module's path from the file dependencies list above,
        // use the file path stored on the module as a fallback.
        // Note that if the module is loaded from a binary precompiled module,
        // this path will be pointing to that binary file instead of the original source file.
        if (!canonicalModulePath.getLength())
        {
            if (auto modulePath = module->getFilePath())
            {
                canonicalModulePath = modulePath;
                Path::getCanonical(canonicalModulePath, canonicalModulePath);
                moduleDir = Path::getParentDirectory(canonicalModulePath);
            }
        }

        // Find the first search directory that contains the module's main file path,
        // so we can store the module's path (the first entry in the dependent files list)
        // as a relative path with respect to that directory.

        String linkageRoot = ".";
        if (auto linkage = module->getLinkage())
        {
            for (const auto& searchDir : linkage->getSearchDirectories().searchDirectories)
            {
                String fullSearchDir;
                Path::getCanonical(searchDir.path, fullSearchDir);
                String relativePath = Path::getRelativePath(fullSearchDir, canonicalModulePath);
                if (!Path::hasRelativeElement(relativePath))
                {
                    linkageRoot = searchDir.path;
                    break;
                }
            }
        }
        Path::getCanonical(linkageRoot, linkageRoot);

        Encoder::WithArray withArray(&encoder);
        for (auto file : fileDependencies)
        {
            if (file->getPathInfo().hasFoundPath())
            {
                String canonicalFilePath = file->getPathInfo().foundPath;
                Path::getCanonical(file->getPathInfo().foundPath, canonicalFilePath);

                // If the dependnet file is the same as the module's file path, store it as a
                // relative path with respect to the search directory discovered above.
                if (file->getPathInfo().hasFileFoundPath())
                {
                    if (canonicalFilePath == canonicalModulePath)
                    {
                        auto relativeModulePath =
                            Path::getRelativePath(linkageRoot, canonicalModulePath);

                        encoder.encodeString(relativeModulePath);
                    }
                    else
                    {
                        // For all other dependnet files, store them as relative paths with respect
                        // to the module's path.
                        canonicalFilePath = Path::getRelativePath(moduleDir, canonicalFilePath);
                        encoder.encodeString(canonicalFilePath);
                    }
                }
                else
                {
                    // If the module is coming from string instead of an actual file, store it as
                    // is.
                    encoder.encodeString(canonicalModulePath);
                }
            }
            else
            {
                encoder.encodeString(file->getPathInfo().getMostUniqueIdentity());
            }
        }

        return SLANG_OK;
    }

    SlangResult encodeFinalPieces()
    {
        // We can now output the debug information. This is for all IR and AST
        if (sourceLocWriter)
        {
            // Write out the debug info
            SerialSourceLocData debugData;
            sourceLocWriter->write(&debugData);

            debugData.writeContainer(encoder.getRIFF());
        }

        // Write the container string table
        if (containerStringPool.getAdded().getCount() > 0)
        {
            Encoder::WithKeyValuePair withKey(&encoder, SerialBinary::kStringTableFourCc);

            List<char> encodedTable;
            SerialStringTableUtil::encodeStringTable(containerStringPool, encodedTable);

            encoder.encodeData(encodedTable.getBuffer(), encodedTable.getCount());
        }

        return SLANG_OK;
    }


private:
    SerialContainerUtil::WriteOptions const& options;
    RefPtr<SerialSourceLocWriter> sourceLocWriter;

    // The string pool used across the whole of the container
    StringSlicePool containerStringPool;

    Encoder encoder;
};

//
// To serialize a module (or compile request) to a stream, we first
// construct a RIFF container from it, and then serialize that
// container out to a byte stream.
//

/* static */ SlangResult SerialContainerUtil::write(
    Module* module,
    const WriteOptions& options,
    Stream* stream)
{
    ModuleEncodingContext context(options, stream);
    SLANG_RETURN_ON_FAIL(context.encode(module));
    return SLANG_OK;
}

/* static */ SlangResult SerialContainerUtil::write(
    FrontEndCompileRequest* request,
    const WriteOptions& options,
    Stream* stream)
{
    ModuleEncodingContext context(options, stream);
    SLANG_RETURN_ON_FAIL(context.encode(request));
    return SLANG_OK;
}

/* static */ SlangResult SerialContainerUtil::write(
    EndToEndCompileRequest* request,
    const WriteOptions& options,
    Stream* stream)
{
    ModuleEncodingContext context(options, stream);
    SLANG_RETURN_ON_FAIL(context.encode(request));
    return SLANG_OK;
}

String StringChunkRef::getValue()
{
    return Decoder(ptr()).decodeString();
}

ChunkRefList<StringChunkRef> ModuleChunkRef::getFileDependencies()
{
    Decoder decoder(ptr());
    Decoder::WithProperty withProperty(decoder, PropertyKeys<Module>::FileDependencies);
    return ChunkRefList<StringChunkRef>(as<RiffContainer::ListChunk>(decoder.getCursor()));
}

ModuleChunkRef ModuleChunkRef::find(RiffContainer* container)
{
    auto found = container->getRoot()->findListRec(SerialBinary::kModuleFourCC);
    return ModuleChunkRef(found);
}

SHA1::Digest ModuleChunkRef::getDigest()
{
    auto foundChunk =
        static_cast<RiffContainer::DataChunk*>(ptr()->findContained(PropertyKeys<Module>::Digest));
    if (!foundChunk)
    {
        SLANG_UNEXPECTED("module chunk had no digest");
    }
    if (foundChunk->calcPayloadSize() != sizeof(SHA1::Digest))
    {
        SLANG_UNEXPECTED("module digest chunk had wrong size");
    }

    SHA1::Digest digest;
    foundChunk->getPayload(&digest);
    return digest;
}

String ModuleChunkRef::getName()
{
    // TODO(tfoley): This kind of logic needs a way
    // to be greatly simplified, so that we don't
    // have to express such complicated logic for
    // simply extracting a single string property...
    //
    Decoder decoder(ptr());
    Decoder::WithProperty withProperty(decoder, SerialBinary::kNameFourCC);
    return decoder.decodeString();
}


IRModuleChunkRef ModuleChunkRef::findIR()
{
    auto foundProperty = ptr()->findContainedList(PropertyKeys<Module>::IRModule);
    if (!foundProperty)
        return IRModuleChunkRef(nullptr);
    return IRModuleChunkRef(
        static_cast<RiffContainer::ListChunk*>(foundProperty->getFirstContainedChunk()));
}

ASTModuleChunkRef ModuleChunkRef::findAST()
{
    auto foundProperty = ptr()->findContainedList(PropertyKeys<Module>::ASTModule);
    if (!foundProperty)
        return ASTModuleChunkRef(nullptr);
    return ASTModuleChunkRef(
        static_cast<RiffContainer::ListChunk*>(foundProperty->getFirstContainedChunk()));
}

ContainerChunkRef ContainerChunkRef::find(RiffContainer* container)
{
    auto found = container->getRoot()->findListRec(SerialBinary::kContainerFourCc);
    return ContainerChunkRef(found);
}

ChunkRefList<ModuleChunkRef> ContainerChunkRef::getModules()
{
    auto found = ptr()->findContainedList(SerialBinary::kModuleListFourCc);
    return ChunkRefList<ModuleChunkRef>(found);
}

ChunkRefList<EntryPointChunkRef> ContainerChunkRef::getEntryPoints()
{
    auto found = ptr()->findContainedList(SerialBinary::kEntryPointListFourCc);
    return ChunkRefList<EntryPointChunkRef>(found);
}

String EntryPointChunkRef::getMangledName() const
{
    // TODO(tfoley): This kind of logic needs a way
    // to be greatly simplified, so that we don't
    // have to express such complicated logic for
    // simply extracting a single string property...
    //
    Decoder decoder(ptr());
    Decoder::WithProperty withProperty(decoder, SerialBinary::kMangledNameFourCC);
    return decoder.decodeString();
}

String EntryPointChunkRef::getName() const
{
    // TODO(tfoley): This kind of logic needs a way
    // to be greatly simplified, so that we don't
    // have to express such complicated logic for
    // simply extracting a single string property...
    //
    Decoder decoder(ptr());
    Decoder::WithProperty withProperty(decoder, SerialBinary::kNameFourCC);
    return decoder.decodeString();
}

Profile EntryPointChunkRef::getProfile() const
{
    // TODO(tfoley): This kind of logic needs a way
    // to be greatly simplified, so that we don't
    // have to express such complicated logic for
    // simply extracting a single string property...
    //
    Decoder decoder(ptr());
    Decoder::WithProperty withProperty(decoder, SerialBinary::kProfileFourCC);

    Profile::RawVal rawVal;
    decoder.decode(rawVal);

    return Profile(rawVal);
}


RiffContainer::ListChunk* findDebugChunk(RiffContainer::Chunk* startingChunk)
{
    if (!startingChunk)
        return nullptr;

    RiffContainer::ListChunk* container = as<RiffContainer::ListChunk>(startingChunk);
    if (!container)
        container = startingChunk->m_parent;

    for (; container; container = container->m_parent)
    {
        if (auto debugChunk = container->findContainedList(SerialSourceLocData::kDebugFourCc))
        {
            return debugChunk;
        }
    }

    return nullptr;
}

SlangResult readSourceLocationsFromDebugChunk(
    RiffContainer::ListChunk* debugChunk,
    SourceManager* sourceManager,
    RefPtr<SerialSourceLocReader>& outReader)
{
    if (!debugChunk)
        return SLANG_FAIL;

    // Source location serialization uses the old approach where
    // there is an intermediate in-memory data structure that the
    // raw data from the RIFF gets deserialized into, before that
    // intermediate representation gets transformed into something
    // more directly usable.
    //
    // Thus we start with a first step where we simply read the data
    // from the RIFF into the intermediate structure.
    //
    SerialSourceLocData intermediateData;
    SLANG_RETURN_ON_FAIL(intermediateData.readContainer(debugChunk));

    // After reading the data into the intermediate representation,
    // we turn it into a `SerialSourceLocReader`, which vends source
    // location information to other deserialization tasks (both IR
    // and AST deserialization).
    //
    auto reader = RefPtr(new SerialSourceLocReader());
    SLANG_RETURN_ON_FAIL(reader->read(&intermediateData, sourceManager));

    outReader = reader;
    return SLANG_OK;
}

SlangResult decodeModuleIR(
    RefPtr<IRModule>& outIRModule,
    RiffContainer::Chunk* chunk,
    Session* session,
    SerialSourceLocReader* sourceLocReader)
{
    // IR serialization still uses the older approach, where
    // data gets deserialized from the RIFF into an intermediate
    // data structure (`IRSerialData`), and then the actual
    // in-memory structures are created based on the intermediate.
    //
    // Thus we start by running the `IRSerialReader::readContainer`
    // logic to get the `IRSerialData` representation.
    //
    // TODO(tfoley): This should all get streamlined so that we
    // are deserializing IR nodes directly from the format written
    // into the RIFF.
    //
    auto listChunk = as<RiffContainer::ListChunk>(chunk);
    if (!listChunk)
        return SLANG_FAIL;
    IRSerialData serialData;
    SLANG_RETURN_ON_FAIL(IRSerialReader::readContainer(listChunk, &serialData));

    // Next we read the actual IR representation out from the
    // `serialData`. This is the step that may pull source-location
    // information from the provided `sourceLocReader`.
    //
    IRSerialReader reader;
    SLANG_RETURN_ON_FAIL(reader.read(serialData, session, sourceLocReader, outIRModule));

    return SLANG_OK;
}

/* static */ SlangResult SerialContainerUtil::verifyIRSerialize(
    IRModule* module,
    Session* session,
    const WriteOptions& options)
{
    // Verify if we can stream out with raw source locs

    List<IRInst*> originalInsts;
    IRSerialWriter::calcInstructionList(module, originalInsts);

    IRSerialData irData;

    OwnedMemoryStream memoryStream(FileAccess::ReadWrite);

    {
        RiffContainer riffContainer;

        // Need to put all of this in a container
        RiffContainer::ScopeChunk containerScope(
            &riffContainer,
            RiffContainer::Chunk::Kind::List,
            SerialBinary::kContainerFourCc);

        RefPtr<SerialSourceLocWriter> sourceLocWriter;

        if (options.optionFlags & SerialOptionFlag::SourceLocation)
        {
            sourceLocWriter = new SerialSourceLocWriter(options.sourceManager);
        }

        {
            // Write IR out to serialData - copying over SourceLoc information directly
            IRSerialWriter writer;
            SLANG_RETURN_ON_FAIL(
                writer.write(module, sourceLocWriter, options.optionFlags, &irData));
        }
        SLANG_RETURN_ON_FAIL(IRSerialWriter::writeContainer(irData, &riffContainer));

        // Write the debug info Riff container
        if (sourceLocWriter)
        {
            SerialSourceLocData serialData;
            sourceLocWriter->write(&serialData);

            SLANG_RETURN_ON_FAIL(serialData.writeContainer(&riffContainer));
        }

        SLANG_RETURN_ON_FAIL(RiffUtil::write(&riffContainer, &memoryStream));
    }

    // Reset stream
    memoryStream.seek(SeekOrigin::Start, 0);

    SourceManager workSourceManager;
    workSourceManager.initialize(options.sourceManager, nullptr);

    // The read ir module
    RefPtr<IRModule> irReadModule;
    {
        RiffContainer riffContainer;
        SLANG_RETURN_ON_FAIL(RiffUtil::read(&memoryStream, riffContainer));

        RiffContainer::ListChunk* rootList = riffContainer.getRoot();

        RefPtr<SerialSourceLocReader> sourceLocReader;

        // If we have debug info then find and read it
        if (options.optionFlags & SerialOptionFlag::SourceLocation)
        {
            RiffContainer::ListChunk* debugList =
                rootList->findContainedList(SerialSourceLocData::kDebugFourCc);
            if (!debugList)
            {
                return SLANG_FAIL;
            }
            SerialSourceLocData sourceLocData;
            SLANG_RETURN_ON_FAIL(sourceLocData.readContainer(debugList));

            sourceLocReader = new SerialSourceLocReader;
            SLANG_RETURN_ON_FAIL(sourceLocReader->read(&sourceLocData, &workSourceManager));
        }

        {
            RiffContainer::ListChunk* irList =
                rootList->findContainedList(IRSerialBinary::kIRModuleFourCc);
            if (!irList)
            {
                return SLANG_FAIL;
            }

            {
                IRSerialData irReadData;
                IRSerialReader reader;
                SLANG_RETURN_ON_FAIL(reader.readContainer(irList, &irReadData));

                // Check the stream read data is the same
                if (irData != irReadData)
                {
                    SLANG_ASSERT(!"Streamed in data doesn't match");
                    return SLANG_FAIL;
                }

                SLANG_RETURN_ON_FAIL(reader.read(irData, session, sourceLocReader, irReadModule));
            }
        }
    }

    List<IRInst*> readInsts;
    IRSerialWriter::calcInstructionList(irReadModule, readInsts);

    if (readInsts.getCount() != originalInsts.getCount())
    {
        SLANG_ASSERT(!"Instruction counts don't match");
        return SLANG_FAIL;
    }

    if (options.optionFlags & SerialOptionFlag::RawSourceLocation)
    {
        SLANG_ASSERT(readInsts[0] == originalInsts[0]);
        // All the source locs should be identical
        for (Index i = 1; i < readInsts.getCount(); ++i)
        {
            IRInst* origInst = originalInsts[i];
            IRInst* readInst = readInsts[i];

            if (origInst->sourceLoc.getRaw() != readInst->sourceLoc.getRaw())
            {
                SLANG_ASSERT(!"Source locs don't match");
                return SLANG_FAIL;
            }
        }
    }
    else if (options.optionFlags & SerialOptionFlag::SourceLocation)
    {
        // They should be on the same line nos
        for (Index i = 1; i < readInsts.getCount(); ++i)
        {
            IRInst* origInst = originalInsts[i];
            IRInst* readInst = readInsts[i];

            if (origInst->sourceLoc.getRaw() == readInst->sourceLoc.getRaw())
            {
                continue;
            }

            // Work out the
            SourceView* origSourceView = options.sourceManager->findSourceView(origInst->sourceLoc);
            SourceView* readSourceView = workSourceManager.findSourceView(readInst->sourceLoc);

            // if both are null we are done
            if (origSourceView == nullptr && origSourceView == readSourceView)
            {
                continue;
            }
            SLANG_ASSERT(origSourceView && readSourceView);

            // The offset should be the same
            Index origOffset =
                origInst->sourceLoc.getRaw() - origSourceView->getRange().begin.getRaw();
            Index readOffset =
                readInst->sourceLoc.getRaw() - readSourceView->getRange().begin.getRaw();

            if (origOffset != readOffset)
            {
                SLANG_ASSERT(!"SourceLoc offset debug data didn't match");
                return SLANG_FAIL;
            }

            {
                auto origInfo =
                    origSourceView->getHumaneLoc(origInst->sourceLoc, SourceLocType::Actual);
                auto readInfo =
                    readSourceView->getHumaneLoc(readInst->sourceLoc, SourceLocType::Actual);

                if (!(origInfo.line == readInfo.line && origInfo.column == readInfo.column &&
                      origInfo.pathInfo.foundPath == readInfo.pathInfo.foundPath))
                {
                    SLANG_ASSERT(!"Debug data didn't match");
                    return SLANG_FAIL;
                }
            }

            // We may have adjusted line numbers -> but they may not match, because we only
            // reconstruct one view So for now disable this test

            if (false)
            {
                auto origInfo =
                    origSourceView->getHumaneLoc(origInst->sourceLoc, SourceLocType::Nominal);
                auto readInfo =
                    readSourceView->getHumaneLoc(readInst->sourceLoc, SourceLocType::Nominal);

                if (!(origInfo.line == readInfo.line && origInfo.column == readInfo.column &&
                      origInfo.pathInfo.foundPath == readInfo.pathInfo.foundPath))
                {
                    SLANG_ASSERT(!"Debug data didn't match");
                    return SLANG_FAIL;
                }
            }
        }
    }

    return SLANG_OK;
}

} // namespace Slang
