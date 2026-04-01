// slang-repro.cpp
#include "slang-repro.h"

#include "../compiler-core/slang-artifact-desc-util.h"
#include "../compiler-core/slang-artifact-util.h"
#include "../compiler-core/slang-source-loc.h"
#include "../core/slang-castable.h"
#include "../core/slang-math.h"
#include "../core/slang-stream.h"
#include "../core/slang-text-io.h"
#include "../core/slang-type-text-util.h"
#include "slang-options.h"

namespace Slang
{

/* static */ const RiffSemanticVersion ReproUtil::g_semanticVersion = RiffSemanticVersion::make(
    ReproUtil::kMajorVersion,
    ReproUtil::kMinorVersion,
    ReproUtil::kPatchVersion);

// We can't just use sizeof for the sizes of these types, because the hash will be dependent on the
// ptr size, which isn't an issue for serialization (we turn all pointers into Offset32Ptr ->
// uint32_t). So we use an x macro to set up the thing to hash.
//
// Note that bool is in the list because size of bool can change between compilers.
// clang-format off
#define SLANG_STATE_TYPES(x) \
    x(Util::FileState) \
    x(Util::PathInfoState) \
        x(Util::PathInfoState::CompressedResult) \
        x(SlangPathType) \
    x(Util::PathAndPathInfo) \
    x(Util::TargetRequestState) \
        x(Profile) \
        x(CodeGenTarget) \
        x(SlangTargetFlags) \
        x(FloatingPointMode) \
    x(Util::StringPair) \
    x(Util::SourceFileState) \
        x(PathInfo::Type) \
    x(Util::TranslationUnitRequestState) \
        x(SourceLanguage) \
    x(Util::EntryPointState) \
        x(Profile) \
    x(Util::RequestState) \
        x(SlangCompileFlags) \
        x(bool) \
        x(LineDirectiveMode) \
        x(DebugInfoLevel) \
        x(OptimizationLevel) \
        x(ContainerFormat) \
        x(PassThroughMode) \
        x(SlangMatrixLayoutMode)
// clang-format on

#define SLANG_STATE_TYPE_SIZE(x) uint32_t(sizeof(x)),

// A function to calculate the hash related in list in part to how the types used are sized. Can
// catch crude breaking binary differences.
static StableHashCode32 _calcTypeHash()
{
    typedef ReproUtil Util;
    const uint32_t sizes[] = {SLANG_STATE_TYPES(SLANG_STATE_TYPE_SIZE)};
    return getStableHashCode32((const char*)&sizes, sizeof(sizes));
}

static StableHashCode32 _getTypeHash()
{
    static StableHashCode32 s_hash = _calcTypeHash();
    return s_hash;
}


namespace
{ // anonymous

struct StoreContext
{
    typedef ReproUtil::FileState FileState;
    typedef ReproUtil::SourceFileState SourceFileState;
    typedef ReproUtil::PathInfoState PathInfoState;

    StoreContext(OffsetContainer* container) { m_container = container; }

    Offset32Ptr<FileState> findFile(const String& uniqueIdentity)
    {
        Offset32Ptr<FileState> file;
        m_uniqueToFileMap.tryGetValue(uniqueIdentity, file);
        return file;
    }

    Offset32Ptr<FileState> addFile(const String& uniqueIdentity, const UnownedStringSlice* content)
    {
        OffsetBase& base = m_container->asBase();

        Offset32Ptr<FileState> file;

        // Get the file, if it has an identity
        if (uniqueIdentity.getLength())
        {
            if (!m_uniqueToFileMap.tryGetValue(uniqueIdentity, file))
            {
                // If file was not found create it
                // Create the file
                file = m_container->newObject<FileState>();
                // Add it
                m_uniqueToFileMap.add(uniqueIdentity, file);

                // Set the identity
                auto offsetUniqueIdentity =
                    m_container->newString(uniqueIdentity.getUnownedSlice());
                base[file]->uniqueIdentity = offsetUniqueIdentity;

                // Add the file
                m_files.add(file);
            }
        }
        else
        {
            // Create a file, but we know it can't have unique identity
            file = m_container->newObject<FileState>();
            // Add the file
            m_files.add(file);
        }

        // If the contents is not set add it
        if (!base[file]->contents && content)
        {
            auto offsetContent = m_container->newString(*content);
            base[file]->contents = offsetContent;
        }

        return file;
    }

    Offset32Ptr<SourceFileState> addSourceFile(SourceFile* sourceFile)
    {
        if (!sourceFile)
        {
            return Offset32Ptr<SourceFileState>();
        }

        auto& base = m_container->asBase();

        Offset32Ptr<ReproUtil::SourceFileState> sourceFileState;
        if (m_sourceFileMap.tryGetValue(sourceFile, sourceFileState))
        {
            return sourceFileState;
        }

        const PathInfo& pathInfo = sourceFile->getPathInfo();

        UnownedStringSlice content = sourceFile->getContent();
        Offset32Ptr<FileState> file = addFile(pathInfo.uniqueIdentity, &content);

        Offset32Ptr<OffsetString> foundPath;

        if (pathInfo.foundPath.getLength() && base[file]->foundPath.isNull())
        {
            foundPath = fromString(pathInfo.foundPath.getUnownedSlice());
        }
        // Set on the file
        base[file]->foundPath = foundPath;

        // Create the source file
        sourceFileState = m_container->newObject<SourceFileState>();

        {
            auto dst = base[sourceFileState];
            dst->file = file;
            dst->foundPath = foundPath;
            dst->type = pathInfo.type;
        }

        m_sourceFileMap.add(sourceFile, sourceFileState);

        return sourceFileState;
    }

    Offset32Ptr<OffsetString> fromString(const String& in)
    {
        Offset32Ptr<OffsetString> value;

        if (m_stringMap.tryGetValue(in, value))
        {
            return value;
        }
        value = m_container->newString(in.getUnownedSlice());
        m_stringMap.add(in, value);
        return value;
    }
    Offset32Ptr<OffsetString> fromName(Name* name)
    {
        if (name)
        {
            return fromString(name->text);
        }
        return Offset32Ptr<OffsetString>();
    }

    Offset32Ptr<PathInfoState> addPathInfo(const CacheFileSystem::PathInfo* srcPathInfo)
    {
        if (!srcPathInfo)
        {
            return Offset32Ptr<PathInfoState>();
        }

        OffsetBase& base = m_container->asBase();

        Offset32Ptr<PathInfoState> pathInfo;
        if (!m_pathInfoMap.tryGetValue(srcPathInfo, pathInfo))
        {
            // Get the associated file
            Offset32Ptr<FileState> fileState;

            // Only store as file if we have the contents
            if (ISlangBlob* fileBlob = srcPathInfo->m_fileBlob)
            {
                UnownedStringSlice content(
                    (const char*)fileBlob->getBufferPointer(),
                    fileBlob->getBufferSize());

                fileState = addFile(srcPathInfo->getUniqueIdentity(), &content);
            }

            // Save the rest of the state
            pathInfo = m_container->newObject<PathInfoState>();
            PathInfoState& dst = base[*pathInfo];

            dst.file = fileState;

            // Save any other info
            dst.getCanonicalPathResult = srcPathInfo->m_getCanonicalPathResult;
            dst.getPathTypeResult = srcPathInfo->m_getPathTypeResult;
            dst.loadFileResult = srcPathInfo->m_loadFileResult;
            dst.pathType = srcPathInfo->m_pathType;

            m_pathInfoMap.add(srcPathInfo, pathInfo);
        }

        // Fill in info on the file
        auto fileState(base[pathInfo]->file);

        // If have fileState add any missing element
        if (fileState)
        {
            if (srcPathInfo->m_fileBlob && base[fileState]->contents.isNull())
            {
                UnownedStringSlice contents(
                    (const char*)srcPathInfo->m_fileBlob->getBufferPointer(),
                    srcPathInfo->m_fileBlob->getBufferSize());
                const auto offsetContents = m_container->newString(contents);
                base[fileState]->contents = offsetContents;
            }

            if (srcPathInfo->m_canonicalPath.getLength() && base[fileState]->canonicalPath.isNull())
            {
                const auto offsetCanonicalPath = fromString(srcPathInfo->m_canonicalPath);
                base[fileState]->canonicalPath = offsetCanonicalPath;
            }

            if (srcPathInfo->m_uniqueIdentity.getLength() &&
                base[fileState]->uniqueIdentity.isNull())
            {
                const auto offsetUniqueIdentity = fromString(srcPathInfo->m_uniqueIdentity);
                base[fileState]->uniqueIdentity = offsetUniqueIdentity;
            }
        }

        return pathInfo;
    }

    const Offset32Array<ReproUtil::StringPair> calcDefines(
        const Dictionary<String, String>& srcDefines)
    {
        typedef ReproUtil::StringPair StringPair;

        Offset32Array<StringPair> dstDefines =
            m_container->newArray<StringPair>(srcDefines.getCount());

        OffsetBase& base = m_container->asBase();

        Index index = 0;
        for (const auto& [srcDefineName, srcDefineVal] : srcDefines)
        {
            auto& dstDefine = base[dstDefines[index]];
            dstDefine.first = fromString(srcDefineName);
            dstDefine.second = fromString(srcDefineVal);

            index++;
        }

        return dstDefines;
    }

    const Offset32Array<Offset32Ptr<OffsetString>> fromList(const List<String>& src)
    {
        Offset32Array<Offset32Ptr<OffsetString>> dst =
            m_container->newArray<Offset32Ptr<OffsetString>>(src.getCount());
        OffsetBase& base = m_container->asBase();

        for (Index j = 0; j < src.getCount(); ++j)
        {
            const auto offsetSrc = fromString(src[j]);

            base[dst[j]] = offsetSrc;
        }
        return dst;
    }

    Dictionary<String, Offset32Ptr<OffsetString>> m_stringMap;

    Dictionary<SourceFile*, Offset32Ptr<ReproUtil::SourceFileState>> m_sourceFileMap;

    Dictionary<String, Offset32Ptr<ReproUtil::FileState>> m_uniqueToFileMap;

    Dictionary<const CacheFileSystem::PathInfo*, Offset32Ptr<PathInfoState>> m_pathInfoMap;

    List<Offset32Ptr<ReproUtil::FileState>> m_files;

    OffsetContainer* m_container;
};

} // namespace

static bool _isStorable(const PathInfo::Type type)
{
    switch (type)
    {
    case PathInfo::Type::Unknown:
    case PathInfo::Type::Normal:
    case PathInfo::Type::FoundPath:
    case PathInfo::Type::FromString:
        {
            return true;
        }
    default:
        return false;
    }
}

static String _scrubName(const String& in)
{
    StringBuilder builder;
    for (auto c : in)
    {
        switch (c)
        {
        case ':':
            c = '_';
            break;
        default:
            break;
        }
        builder.appendChar(c);
    }

    return builder.produceString();
}

/* static */ SlangResult ReproUtil::store(
    EndToEndCompileRequest* request,
    OffsetContainer& inOutContainer,
    Offset32Ptr<RequestState>& outRequest)
{
    StoreContext context(&inOutContainer);

    OffsetBase& base = inOutContainer.asBase();

    auto linkage = request->getLinkage();

    Offset32Ptr<RequestState> requestState = inOutContainer.newObject<RequestState>();

    {
        RequestState* dst = base[requestState];

        dst->compileFlags = 0;
        dst->shouldDumpIntermediates =
            linkage->m_optionSet.getBoolOption(CompilerOptionName::DumpIntermediates);
        dst->debugInfoLevel = linkage->m_optionSet.getEnumOption<DebugInfoLevel>(
            CompilerOptionName::DebugInformation);
        dst->optimizationLevel =
            linkage->m_optionSet.getEnumOption<OptimizationLevel>(CompilerOptionName::Optimization);
        dst->containerFormat = request->m_containerFormat;
        dst->passThroughMode = request->m_passThrough;

        dst->useUnknownImageFormatAsDefault =
            linkage->m_optionSet.getBoolOption(CompilerOptionName::DefaultImageFormatUnknown);
        ;
        dst->obfuscateCode = linkage->m_optionSet.getBoolOption(CompilerOptionName::Obfuscate);

        dst->defaultMatrixLayoutMode =
            (SlangMatrixLayoutMode)linkage->m_optionSet.getMatrixLayoutMode();
    }

    // Entry points
    {
        const auto& srcEntryPoints = request->getFrontEndReq()->m_entryPointReqs;
        const auto& srcEndToEndEntryPoints = request->m_entryPoints;

        SLANG_ASSERT(srcEntryPoints.getCount() == srcEndToEndEntryPoints.getCount());

        Offset32Array<EntryPointState> dstEntryPoints =
            inOutContainer.newArray<EntryPointState>(srcEntryPoints.getCount());

        for (Index i = 0; i < srcEntryPoints.getCount(); ++i)
        {
            FrontEndEntryPointRequest* srcEntryPoint = srcEntryPoints[i];
            const auto& srcEndToEndEntryPoint = srcEndToEndEntryPoints[i];

            auto dstSpecializationArgStrings =
                context.fromList(srcEndToEndEntryPoint.specializationArgStrings);
            Offset32Ptr<OffsetString> dstName = context.fromName(srcEntryPoint->getName());

            EntryPointState& dst = base[dstEntryPoints[i]];

            dst.profile = srcEntryPoint->getProfile();
            dst.translationUnitIndex = uint32_t(srcEntryPoint->getTranslationUnitIndex());
            dst.specializationArgStrings = dstSpecializationArgStrings;
            dst.name = dstName;
        }

        base[requestState]->entryPoints = dstEntryPoints;
    }


    // Add all of the source files
    {
        SourceManager* sourceManager = request->getFrontEndReq()->getSourceManager();
        const List<SourceFile*>& sourceFiles = sourceManager->getSourceFiles();

        for (SourceFile* sourceFile : sourceFiles)
        {
            const PathInfo& pathInfo = sourceFile->getPathInfo();
            if (_isStorable(pathInfo.type))
            {
                context.addSourceFile(sourceFile);
            }
        }
    }

    // Add all the target requests
    {
        Offset32Array<TargetRequestState> dstTargets =
            inOutContainer.newArray<TargetRequestState>(linkage->targets.getCount());

        for (Index i = 0; i < linkage->targets.getCount(); ++i)
        {
            TargetRequest* srcTargetRequest = linkage->targets[i];

            // Copy the simple stuff
            {
                auto& dst = base[dstTargets[i]];
                dst.target = srcTargetRequest->getTarget();
                dst.profile = srcTargetRequest->getOptionSet().getProfile();
                dst.targetFlags = srcTargetRequest->getOptionSet().getTargetFlags();
                dst.floatingPointMode =
                    srcTargetRequest->getOptionSet().getEnumOption<FloatingPointMode>(
                        CompilerOptionName::FloatingPointMode);
            }

            // Copy the entry point/target output names
            {
                const auto& srcTargetInfos = request->m_targetInfos;

                if (const RefPtr<EndToEndCompileRequest::TargetInfo>* infosPtr =
                        srcTargetInfos.tryGetValue(srcTargetRequest))
                {
                    EndToEndCompileRequest::TargetInfo* infos = *infosPtr;

                    const auto& entryPointOutputPaths = infos->entryPointOutputPaths;

                    Offset32Array<OutputState> dstOutputStates =
                        inOutContainer.newArray<OutputState>(entryPointOutputPaths.getCount());

                    Index index = 0;
                    for (const auto& [key, value] : entryPointOutputPaths)
                    {
                        Offset32Ptr<OffsetString> outputPath =
                            inOutContainer.newString(value.getUnownedSlice());

                        auto& dstOutputState = base[dstOutputStates[index]];

                        dstOutputState.entryPointIndex = int32_t(key);
                        dstOutputState.outputPath = outputPath;

                        index++;
                    }

                    base[dstTargets[i]].outputStates = dstOutputStates;
                }
            }
        }

        // Save the result
        base[requestState]->targetRequests = dstTargets;
    }

    // Add the search paths
    {
        auto srcPaths = linkage->getSearchDirectories();
        Offset32Array<Offset32Ptr<OffsetString>> dstPaths =
            inOutContainer.newArray<Offset32Ptr<OffsetString>>(
                srcPaths.searchDirectories.getCount());

        // We don't handle parents here
        SLANG_ASSERT(srcPaths.parent == nullptr);
        for (Index i = 0; i < srcPaths.searchDirectories.getCount(); ++i)
        {
            const auto srcPath = context.fromString(srcPaths.searchDirectories[i].path);
            base[dstPaths[i]] = srcPath;
        }
        base[requestState]->searchPaths = dstPaths;
    }

    // Add preprocessor definitions
    Dictionary<String, String> preprocessorDefines;
    for (auto def : linkage->m_optionSet.getArray(CompilerOptionName::MacroDefine))
    {
        preprocessorDefines[def.stringValue] = def.stringValue2;
    }
    base[requestState]->preprocessorDefinitions = context.calcDefines(preprocessorDefines);

    {
        const auto& srcTranslationUnits = request->getFrontEndReq()->translationUnits;
        Offset32Array<TranslationUnitRequestState> dstTranslationUnits =
            inOutContainer.newArray<TranslationUnitRequestState>(srcTranslationUnits.getCount());

        for (Index i = 0; i < srcTranslationUnits.getCount(); ++i)
        {
            TranslationUnitRequest* srcTranslationUnit = srcTranslationUnits[i];

            // Do before setting, because this can allocate, and therefore break, the following
            // section
            auto defines = context.calcDefines(srcTranslationUnit->preprocessorDefinitions);
            auto moduleName = context.fromName(srcTranslationUnit->moduleName);

            Offset32Array<Offset32Ptr<SourceFileState>> dstSourceFiles;
            {
                const auto& srcFiles = srcTranslationUnit->getSourceFiles();
                dstSourceFiles =
                    inOutContainer.newArray<Offset32Ptr<SourceFileState>>(srcFiles.getCount());

                for (Index j = 0; j < srcFiles.getCount(); ++j)
                {
                    const auto srcFile = context.addSourceFile(srcFiles[j]);
                    base[dstSourceFiles[j]] = srcFile;
                }
            }

            TranslationUnitRequestState& dstTranslationUnit = base[dstTranslationUnits[i]];

            dstTranslationUnit.language = srcTranslationUnit->sourceLanguage;
            dstTranslationUnit.moduleName = moduleName;
            dstTranslationUnit.sourceFiles = dstSourceFiles;
            dstTranslationUnit.preprocessorDefinitions = defines;
        }

        base[requestState]->translationUnits = dstTranslationUnits;
    }

    // Find files from the file system, and mapping paths to files
    {
        CacheFileSystem* cacheFileSystem = as<CacheFileSystem>(linkage->getFileSystemExt());
        if (!cacheFileSystem)
        {
            return SLANG_FAIL;
        }

        // Traverse the references (in process we will construct the map from PathInfo)
        {
            const auto& srcFiles = cacheFileSystem->getPathMap();

            Offset32Array<PathAndPathInfo> pathMap =
                inOutContainer.newArray<PathAndPathInfo>(srcFiles.getCount());

            Index index = 0;
            for (const auto& [key, value] : srcFiles)
            {
                const auto path = context.fromString(key);
                const auto pathInfo = context.addPathInfo(value);

                PathAndPathInfo& dstInfo = base[pathMap[index]];
                dstInfo.path = path;
                dstInfo.pathInfo = pathInfo;

                index++;
            }

            base[requestState]->pathInfoMap = pathMap;
        }
    }

    // Save all of the files
    {
        Dictionary<String, int> uniqueNameMap;

        auto files = inOutContainer.newArray<Offset32Ptr<FileState>>(context.m_files.getCount());
        for (Index i = 0; i < context.m_files.getCount(); ++i)
        {
            Offset32Ptr<FileState> file = context.m_files[i];

            // Need to come up with unique names
            String path;

            if (auto canonicalPath = base[file]->canonicalPath)
            {
                path = base[canonicalPath]->getSlice();
            }
            else if (auto foundPath = base[file]->foundPath)
            {
                path = base[foundPath]->getSlice();
            }
            else if (auto uniqueIdentity = base[file]->uniqueIdentity)
            {
                path = base[uniqueIdentity]->getSlice();
            }

            if (path.getLength() == 0)
            {
                StringBuilder builder;
                builder << "unnamed" << i;
                path = builder;
            }

            String filename = _scrubName(Path::getFileNameWithoutExt(path));
            String ext = _scrubName(Path::getPathExt(path));

            StringBuilder uniqueName;
            for (Index j = 0; j < 0x10000; j++)
            {
                uniqueName.clear();
                uniqueName << filename;

                if (j > 0)
                {
                    uniqueName << "-" << j;
                }

                if (ext.getLength())
                {
                    uniqueName << "." << ext;
                }

                int dummy = 0;
                if (!uniqueNameMap.tryGetValueOrAdd(uniqueName, dummy))
                {
                    // It was added so we are done
                    break;
                }
            }

            // Save the unique generated name
            const auto offsetUniqueName = inOutContainer.newString(uniqueName.getUnownedSlice());

            base[file]->uniqueName = offsetUniqueName;

            base[files[i]] = file;
        }

        base[requestState]->files = files;
    }

    // Save all the SourceFile state
    {
        const auto& srcSourceFiles = context.m_sourceFileMap;
        auto dstSourceFiles =
            inOutContainer.newArray<Offset32Ptr<SourceFileState>>(srcSourceFiles.getCount());

        Index index = 0;
        for (const auto& [_, value] : srcSourceFiles)
        {
            base[dstSourceFiles[index]] = value;
            index++;
        }
        base[requestState]->sourceFiles = dstSourceFiles;
    }

    outRequest = requestState;
    return SLANG_OK;
}

namespace
{ // anonymous

struct LoadContext
{
    typedef ReproUtil::SourceFileState SourceFileState;
    typedef ReproUtil::FileState FileState;
    typedef ReproUtil::PathInfoState PathInfoState;

    CacheFileSystem::PathInfo* getPathInfoFromFile(FileState* file)
    {
        if (!file)
        {
            return nullptr;
        }

        CacheFileSystem::PathInfo* dstInfo = nullptr;
        if (!m_fileToPathInfoMap.tryGetValue(file, dstInfo))
        {
            ComPtr<ISlangBlob> blob;

            if (m_fileSystem && file->uniqueName)
            {
                // Try loading from the file system
                m_fileSystem->loadFile(m_base->asRaw(file->uniqueName)->getCstr(), blob.writeRef());
            }

            // If wasn't loaded, and has contents, use that
            if (!blob && file->contents)
            {
                blob = StringBlob::create(m_base->asRaw(file->contents)->getSlice());
            }

            dstInfo = new CacheFileSystem::PathInfo(String());

            if (file->uniqueIdentity)
            {
                dstInfo->m_uniqueIdentity = m_base->asRaw(file->uniqueIdentity)->getSlice();
            }

            if (file->canonicalPath)
            {
                dstInfo->m_canonicalPath = m_base->asRaw(file->canonicalPath)->getSlice();
            }

            if (blob)
            {
                dstInfo->m_loadFileResult = CacheFileSystem::CompressedResult::Ok;
                dstInfo->m_getPathTypeResult = CacheFileSystem::CompressedResult::Ok;
                dstInfo->m_pathType = SLANG_PATH_TYPE_FILE;
            }

            dstInfo->m_fileBlob = blob;

            // Add to map, even if the blob is nullptr (say from a failed read)
            m_fileToPathInfoMap.add(file, dstInfo);
        }

        return dstInfo;
    }

    ISlangBlob* getFileBlobFromFile(FileState* file)
    {
        CacheFileSystem::PathInfo* pathInfo = getPathInfoFromFile(file);
        return pathInfo ? pathInfo->m_fileBlob.get() : nullptr;
    }

    SourceFile* getSourceFile(SourceFileState* sourceFile)
    {
        if (sourceFile == nullptr)
        {
            return nullptr;
        }

        SourceFile* dstFile;
        if (!m_sourceFileMap.tryGetValue(sourceFile, dstFile))
        {
            FileState* file = m_base->asRaw(sourceFile->file);
            ISlangBlob* blob = getFileBlobFromFile(file);

            PathInfo pathInfo;

            pathInfo.type = sourceFile->type;

            if (sourceFile->foundPath)
            {
                pathInfo.foundPath = m_base->asRaw(sourceFile->foundPath)->getSlice();
            }
            else if (file->foundPath)
            {
                pathInfo.foundPath = m_base->asRaw(file->foundPath)->getSlice();
            }

            if (file->uniqueIdentity)
            {
                pathInfo.uniqueIdentity = m_base->asRaw(file->uniqueIdentity)->getSlice();
            }

            dstFile = new SourceFile(m_sourceManager, pathInfo, blob->getBufferSize());
            dstFile->setContents(blob);

            // Add to map
            m_sourceFileMap.add(sourceFile, dstFile);

            // Add to manager
            m_sourceManager->addSourceFile(pathInfo.uniqueIdentity, dstFile);
        }
        return dstFile;
    }

    CacheFileSystem::PathInfo* addPathInfo(const PathInfoState* srcInfo)
    {
        if (!srcInfo)
        {
            return nullptr;
        }

        CacheFileSystem::PathInfo* pathInfo;
        if (m_pathInfoMap.tryGetValue(srcInfo, pathInfo))
        {
            return pathInfo;
        }

        FileState* file = m_base->asRaw(srcInfo->file);
        CacheFileSystem::PathInfo* dstInfo;

        if (file)
        {
            dstInfo = getPathInfoFromFile(file);
        }
        else
        {
            // TODO(JS): Hmmm... this could end up not being cleared up
            // Because it is not added to the unique set (as unique set is for files and this isn't
            // a file)
            dstInfo = new CacheFileSystem::PathInfo(String());
        }

        dstInfo->m_getCanonicalPathResult = srcInfo->getCanonicalPathResult;
        dstInfo->m_getPathTypeResult = srcInfo->getPathTypeResult;
        dstInfo->m_loadFileResult = srcInfo->loadFileResult;
        dstInfo->m_pathType = srcInfo->pathType;

        m_pathInfoMap.add(srcInfo, dstInfo);
        return dstInfo;
    }

    List<const char*> toList(const Offset32Array<Offset32Ptr<OffsetString>>& src)
    {
        List<const char*> dst;
        dst.setCount(src.getCount());
        for (Index i = 0; i < src.getCount(); ++i)
        {
            OffsetString* srcString = m_base->asRaw(m_base->asRaw(src[i]));
            dst[i] = srcString ? srcString->getCstr() : nullptr;
        }
        return dst;
    }


    void loadDefines(
        const Offset32Array<ReproUtil::StringPair>& in,
        Dictionary<String, String>& out)
    {
        out.clear();

        for (const auto& define : in)
        {
            out.add(
                m_base->asRaw(m_base->asRaw(define).first)->getSlice(),
                m_base->asRaw(m_base->asRaw(define).second)->getSlice());
        }
    }

    LoadContext(SourceManager* sourceManger, ISlangFileSystem* fileSystem, OffsetBase* base)
        : m_sourceManager(sourceManger), m_fileSystem(fileSystem), m_base(base)
    {
    }

    ISlangFileSystem* m_fileSystem;

    OffsetBase* m_base;

    SourceManager* m_sourceManager;

    Dictionary<SourceFileState*, SourceFile*> m_sourceFileMap;
    Dictionary<FileState*, CacheFileSystem::PathInfo*> m_fileToPathInfoMap;
    Dictionary<const PathInfoState*, CacheFileSystem::PathInfo*> m_pathInfoMap;
};

} // namespace


/* static */ SlangResult ReproUtil::loadFileSystem(
    OffsetBase& base,
    RequestState* requestState,
    ISlangFileSystem* replaceFileSystem,
    ComPtr<ISlangFileSystemExt>& outFileSystem)
{
    LoadContext context(nullptr, replaceFileSystem, &base);

    CacheFileSystem* cacheFileSystem = new CacheFileSystem(nullptr);
    ComPtr<ISlangFileSystemExt> scopeCacheFileSystem(cacheFileSystem);

    auto& dstUniqueMap = cacheFileSystem->getUniqueMap();
    auto& dstPathMap = cacheFileSystem->getPathMap();

    for (auto fileOffset : requestState->files)
    {
        // add the file
        FileState* fileState = base.asRaw(base.asRaw(fileOffset));
        CacheFileSystem::PathInfo* pathInfo = context.getPathInfoFromFile(fileState);

        if (fileState->foundPath)
        {
            String foundPath = base.asRaw(fileState->foundPath)->getSlice();
            dstPathMap.addIfNotExists(foundPath, pathInfo);
        }
    }

    // Put all the paths to path info
    {
        for (const auto& pairOffset : requestState->pathInfoMap)
        {
            const auto& pair = base.asRaw(pairOffset);
            CacheFileSystem::PathInfo* pathInfo = context.addPathInfo(base.asRaw(pair.pathInfo));
            dstPathMap.addIfNotExists(base.asRaw(pair.path)->getSlice(), pathInfo);
        }
    }

    // Put all the path infos in the cache system
    {
        for (const auto& [_, pathInfo] : context.m_fileToPathInfoMap)
        {
            SLANG_ASSERT(pathInfo->m_uniqueIdentity.getLength());
            dstUniqueMap.add(pathInfo->m_uniqueIdentity, pathInfo);

            // Add canonical paths too..
            if (pathInfo->m_canonicalPath.getLength())
            {
                String canonicalPath = pathInfo->m_canonicalPath;

                dstPathMap.addIfNotExists(canonicalPath, pathInfo);
            }
        }
    }

    outFileSystem.swap(scopeCacheFileSystem);
    return SLANG_OK;
}

/* static */ SlangResult ReproUtil::load(
    OffsetBase& base,
    RequestState* requestState,
    ISlangFileSystem* optionalFileSystem,
    EndToEndCompileRequest* request)
{
    auto externalRequest = asExternal(request);

    auto linkage = request->getLinkage();

    // TODO(JS): Really should be more exhaustive here, and set up to initial state ideally
    // Reset state
    {
        request->m_targetInfos.clear();
        // Remove any requests
        linkage->targets.clear();
    }

    LoadContext context(linkage->getSourceManager(), optionalFileSystem, &base);

    // Try to set state through API - as doing so means if state stored in multiple places it will
    // be ok

    {
        externalRequest->setCompileFlags((SlangCompileFlags)requestState->compileFlags);
        externalRequest->setDumpIntermediates(int(requestState->shouldDumpIntermediates));
        externalRequest->setLineDirectiveMode(
            SlangLineDirectiveMode(requestState->lineDirectiveMode));
        externalRequest->setDebugInfoLevel(SlangDebugInfoLevel(requestState->debugInfoLevel));
        externalRequest->setOptimizationLevel(
            SlangOptimizationLevel(requestState->optimizationLevel));
        externalRequest->setOutputContainerFormat(
            SlangContainerFormat(requestState->containerFormat));
        externalRequest->setPassThrough(SlangPassThrough(request->m_passThrough));

        linkage->m_optionSet.set(
            CompilerOptionName::DefaultImageFormatUnknown,
            requestState->useUnknownImageFormatAsDefault);
        linkage->m_optionSet.set(CompilerOptionName::Obfuscate, requestState->obfuscateCode);

        linkage->setMatrixLayoutMode(requestState->defaultMatrixLayoutMode);
    }

    {
        const auto& srcPaths = requestState->searchPaths;
        for (Index i = 0; i < srcPaths.getCount(); ++i)
        {
            linkage->m_optionSet.add(
                CompilerOptionName::Include,
                base.asRaw(base.asRaw(srcPaths[i]))->getSlice());
        }
    }
    Dictionary<String, String> preprocessorDefines;
    context.loadDefines(requestState->preprocessorDefinitions, preprocessorDefines);
    for (auto def : preprocessorDefines)
    {
        linkage->m_optionSet.addPreprocessorDefine(def.first, def.second);
    }

    // Add the target requests
    {
        for (Index i = 0; i < requestState->targetRequests.getCount(); ++i)
        {
            TargetRequestState& src = base.asRaw(requestState->targetRequests[i]);
            int index = externalRequest->addCodeGenTarget(SlangCompileTarget(src.target));
            SLANG_ASSERT(index == i);

            auto dstTarget = linkage->targets[index];

            SLANG_ASSERT(dstTarget->getTarget() == src.target);
            dstTarget->getOptionSet().setProfile(src.profile);
            dstTarget->getOptionSet().addTargetFlags(src.targetFlags);
            dstTarget->getOptionSet().set(
                CompilerOptionName::FloatingPointMode,
                src.floatingPointMode);

            // If there is output state (like output filenames) add here
            if (src.outputStates.getCount())
            {
                RefPtr<EndToEndCompileRequest::TargetInfo> dstTargetInfo(
                    new EndToEndCompileRequest::TargetInfo);
                request->m_targetInfos[dstTarget] = dstTargetInfo;

                for (const auto& srcOutputStateOffset : src.outputStates)
                {
                    const auto& srcOutputState = base.asRaw(srcOutputStateOffset);

                    SLANG_ASSERT(
                        srcOutputState.entryPointIndex < requestState->entryPoints.getCount());

                    String entryPointPath;
                    if (srcOutputState.outputPath)
                    {
                        entryPointPath = base.asRaw(srcOutputState.outputPath)->getSlice();
                    }

                    dstTargetInfo->entryPointOutputPaths.add(
                        srcOutputState.entryPointIndex,
                        entryPointPath);
                }
            }
        }
    }


    {
        auto frontEndReq = request->getFrontEndReq();

        const auto& srcTranslationUnits = requestState->translationUnits;
        auto& dstTranslationUnits = frontEndReq->translationUnits;

        dstTranslationUnits.clear();

        for (Index i = 0; i < srcTranslationUnits.getCount(); ++i)
        {
            const auto& srcTranslationUnit = base.asRaw(srcTranslationUnits[i]);

            // TODO(JS): We should probably serialize off the module name
            // Passing in nullptr will just generate the module name
            int index = frontEndReq->addTranslationUnit(srcTranslationUnit.language, nullptr);
            SLANG_UNUSED(index);
            SLANG_ASSERT(index == i);

            TranslationUnitRequest* dstTranslationUnit = dstTranslationUnits[i];

            context.loadDefines(
                srcTranslationUnit.preprocessorDefinitions,
                dstTranslationUnit->preprocessorDefinitions);

            Name* moduleName = nullptr;
            if (srcTranslationUnit.moduleName)
            {
                moduleName = request->getNamePool()->getName(
                    base.asRaw(srcTranslationUnit.moduleName)->getSlice());
            }

            dstTranslationUnit->moduleName = moduleName;

            const auto& srcSourceFiles = srcTranslationUnit.sourceFiles;

            dstTranslationUnit->clearSource();

            const auto sourceDesc = ArtifactDescUtil::makeDescForSourceLanguage(
                asExternal(dstTranslationUnit->sourceLanguage));

            for (Index j = 0; j < srcSourceFiles.getCount(); ++j)
            {
                // Create the source file
                SourceFile* sourceFile =
                    context.getSourceFile(base.asRaw(base.asRaw(srcSourceFiles[i])));

                // Create the artifact
                auto sourceArtifact = ArtifactUtil::createArtifact(
                    sourceDesc,
                    sourceFile->getPathInfo().getName().getBuffer());
                if (sourceFile->getContentBlob())
                {
                    sourceArtifact->addRepresentationUnknown(sourceFile->getContentBlob());
                }

                // Add to translation unit
                dstTranslationUnit->addSource(sourceArtifact, sourceFile);
            }
        }
    }

    // Entry points
    {
        // Check there aren't any set entry point
        SLANG_ASSERT(request->getFrontEndReq()->m_entryPointReqs.getCount() == 0);

        for (const auto& srcEntryPointOffset : requestState->entryPoints)
        {
            const auto srcEntryPoint = base.asRaw(srcEntryPointOffset);

            const char* name =
                srcEntryPoint.name ? base.asRaw(srcEntryPoint.name)->getCstr() : nullptr;

            Stage stage = srcEntryPoint.profile.getStage();

            List<const char*> args = context.toList(srcEntryPoint.specializationArgStrings);

            externalRequest->addEntryPointEx(
                int(srcEntryPoint.translationUnitIndex),
                name,
                SlangStage(stage),
                int(args.getCount()),
                args.getBuffer());
        }
    }

    {
        auto cacheFileSystem = new CacheFileSystem(request->m_reproFallbackFileSystem);
        ComPtr<ISlangFileSystemExt> fileSystemExt(cacheFileSystem);
        auto& dstUniqueMap = cacheFileSystem->getUniqueMap();
        auto& dstPathMap = cacheFileSystem->getPathMap();

        // Put all the paths to path info
        {
            for (const auto& pairOffset : requestState->pathInfoMap)
            {
                const auto& pair = base.asRaw(pairOffset);
                auto srcPathInfo = base.asRaw(pair.pathInfo);

                CacheFileSystem::PathInfo* pathInfo = context.addPathInfo(srcPathInfo);
                dstPathMap.add(base.asRaw(pair.path)->getSlice(), pathInfo);
            }
        }
        // Put all the path infos in the cache system
        {
            for (const auto& [_, pathInfo] : context.m_fileToPathInfoMap)
            {
                // TODO(JS): It's not 100% clear why we are ending up
                // with entries that don't have a unique identity.
                // For now we ignore adding to the unique map, because
                // if we do we'll have multiple entries with the same key
                if (pathInfo->m_uniqueIdentity.getLength() == 0)
                {
                    continue;
                }
                SLANG_ASSERT(pathInfo->m_uniqueIdentity.getLength());
                dstUniqueMap.add(pathInfo->m_uniqueIdentity, pathInfo);
            }
        }

        // This is a bit of a hack, we are going to replace the file system, with our one which is
        // filled in with what was read from the file.

        linkage->m_fileSystemExt.swap(fileSystemExt);
    }

    return SLANG_OK;
}


/* static */ SlangResult ReproUtil::saveState(EndToEndCompileRequest* request, Stream* stream)
{
    OffsetContainer container;
    Offset32Ptr<RequestState> requestState;
    SLANG_RETURN_ON_FAIL(store(request, container, requestState));

    Header header;
    header.m_chunk.type = kSlangStateFourCC;
    header.m_semanticVersion = g_semanticVersion;
    header.m_typeHash = _getTypeHash();

    return RiffUtil::writeData(
        &header.m_chunk,
        sizeof(header),
        container.getData(),
        container.getDataCount(),
        stream);
}

/* static */ SlangResult ReproUtil::saveState(
    EndToEndCompileRequest* request,
    const String& filename)
{
    RefPtr<FileStream> stream(new FileStream);
    SLANG_RETURN_ON_FAIL(
        stream->init(filename, FileMode::Create, FileAccess::Write, FileShare::ReadWrite));
    return saveState(request, stream);
}

/* static */ SlangResult ReproUtil::loadState(
    const String& filename,
    DiagnosticSink* sink,
    List<uint8_t>& outBuffer)
{
    RefPtr<FileStream> stream = new FileStream;
    SLANG_RETURN_ON_FAIL(
        stream->init(filename, FileMode::Open, FileAccess::Read, FileShare::ReadWrite));
    return loadState(stream, sink, outBuffer);
}

/* static */ SlangResult ReproUtil::loadState(
    Stream* stream,
    DiagnosticSink* sink,
    List<uint8_t>& buffer)
{
    Header header;

    {
        Result res = RiffUtil::readData(stream, &header.m_chunk, sizeof(header), buffer);
        if (SLANG_FAILED(res))
        {
            sink->diagnose(SourceLoc(), Diagnostics::unableToReadRiff);
            return res;
        }
    }
    if (header.m_chunk.type != kSlangStateFourCC)
    {
        sink->diagnose(SourceLoc(), Diagnostics::expectingSlangRiffContainer);
        return SLANG_FAIL;
    }

    if (!RiffSemanticVersion::areCompatible(g_semanticVersion, header.m_semanticVersion))
    {
        StringBuilder headerBuf, currentBuf;
        header.m_semanticVersion.asSemanticVersion().append(headerBuf);
        g_semanticVersion.asSemanticVersion().append(currentBuf);

        sink->diagnose(
            SourceLoc(),
            Diagnostics::incompatibleRiffSemanticVersion,
            headerBuf,
            currentBuf);
        return SLANG_FAIL;
    }

    if (header.m_typeHash != _getTypeHash())
    {
        sink->diagnose(SourceLoc(), Diagnostics::riffHashMismatch);
        return SLANG_FAIL;
    }

    return SLANG_OK;
}

/* static */ SlangResult ReproUtil::loadState(
    const uint8_t* data,
    size_t size,
    DiagnosticSink* sink,
    List<uint8_t>& outBuffer)
{
    MemoryStreamBase stream(FileAccess::Read, data, size);
    return loadState(&stream, sink, outBuffer);
}

/* static */ ReproUtil::RequestState* ReproUtil::getRequest(const List<uint8_t>& buffer)
{
    return (ReproUtil::RequestState*)(buffer.getBuffer() + kStartOffset);
}

/* static */ SlangResult ReproUtil::calcDirectoryPathFromFilename(
    const String& filename,
    String& outPath)
{
    String absPath;
    SLANG_RETURN_ON_FAIL(Path::getCanonical(filename, absPath));

    String parentDir = Path::getParentDirectory(absPath);

    String baseName = Path::getFileNameWithoutExt(filename);
    String ext = Path::getPathExt(filename);

    if (ext.getLength() == 0)
    {
        StringBuilder builder;
        builder << baseName << "-files";
        baseName = builder;
    }

    outPath = Path::combine(parentDir, baseName);
    return SLANG_OK;
}

/* static */ SlangResult ReproUtil::extractFilesToDirectory(
    const String& filename,
    DiagnosticSink* sink)
{
    List<uint8_t> buffer;
    SLANG_RETURN_ON_FAIL(ReproUtil::loadState(filename, sink, buffer));

    MemoryOffsetBase base;
    base.set(buffer.getBuffer(), buffer.getCount());

    RequestState* requestState = ReproUtil::getRequest(buffer);

    String dirPath;
    SLANG_RETURN_ON_FAIL(ReproUtil::calcDirectoryPathFromFilename(filename, dirPath));

    if (!Path::createDirectory(dirPath))
    {
        sink->diagnose(SourceLoc(), Diagnostics::unableToCreateDirectory, dirPath);
        return SLANG_FAIL;
    }

    // Set up a file system to write into this directory
    RelativeFileSystem relFileSystem(OSFileSystem::getMutableSingleton(), dirPath);

    return extractFiles(base, requestState, &relFileSystem);
}

static void _calcPreprocessorDefines(
    OffsetBase& base,
    const Offset32Array<ReproUtil::StringPair>& srcDefines,
    CommandLine& cmd)
{
    for (const auto& define : srcDefines)
    {
        StringBuilder builder;
        builder << "-D" << base.asRaw(base.asRaw(define).first)->getSlice();
        if (base.asRaw(define).second)
        {
            builder << "=" << base.asRaw(base.asRaw(define).second)->getSlice();
        }

        cmd.addArg(builder);
    }
}

static SlangResult _calcCommandLine(
    OffsetBase& base,
    ReproUtil::RequestState* requestState,
    CommandLine& cmd)
{
    typedef ReproUtil::TargetRequestState TargetRequestState;
    typedef ReproUtil::SourceFileState SourceFileState;

    {
        SlangCompileFlags flags = (SlangCompileFlags)requestState->compileFlags;
        while (flags)
        {
            // Extract a bit
            const SlangCompileFlags isolatedBit = flags & SlangCompileFlags(-int(flags));

            switch (isolatedBit)
            {
            case SLANG_COMPILE_FLAG_NO_MANGLING:
                cmd.addArg("-no-mangle");
                break;
            case SLANG_COMPILE_FLAG_NO_CODEGEN:
                cmd.addArg("-no-codegen");
                break;
            default:
                break;
            }

            // Remove the bit
            flags &= ~isolatedBit;
        }
        // spSetDumpIntermediates(externalRequest, int(requestState->shouldDumpIntermediates));

        switch (SlangLineDirectiveMode(requestState->lineDirectiveMode))
        {
        case SLANG_LINE_DIRECTIVE_MODE_DEFAULT:
            break;
        case SLANG_LINE_DIRECTIVE_MODE_NONE:
            {
                cmd.addArg("-line-directive-mode");
                cmd.addArg("none");
                break;
            }
        default:
            break;
        }

        switch (SlangDebugInfoLevel(requestState->debugInfoLevel))
        {
        case SLANG_DEBUG_INFO_LEVEL_STANDARD:
            cmd.addArg("-g");
            break;
        case SLANG_DEBUG_INFO_LEVEL_NONE:
            cmd.addArg("-g0");
            break;
        case SLANG_DEBUG_INFO_LEVEL_MINIMAL:
            cmd.addArg("-g1");
            break;
        case SLANG_DEBUG_INFO_LEVEL_MAXIMAL:
            cmd.addArg("-g3");
            break;
        default:
            break;
        }

        switch (SlangOptimizationLevel(requestState->optimizationLevel))
        {
        case SLANG_OPTIMIZATION_LEVEL_NONE:
            cmd.addArg("-O0");
            break;
        case SLANG_OPTIMIZATION_LEVEL_DEFAULT:
            cmd.addArg("-O");
            break;
        case SLANG_OPTIMIZATION_LEVEL_HIGH:
            cmd.addArg("-O2");
            break;
        case SLANG_OPTIMIZATION_LEVEL_MAXIMAL:
            cmd.addArg("-O3");
            break;
        default:
            break;
        }

        // spSetOutputContainerFormat(externalRequest,
        // SlangContainerFormat(requestState->containerFormat));

        switch (SlangPassThrough(requestState->passThroughMode))
        {
        case SLANG_PASS_THROUGH_NONE:
            break;
        default:
            {
                cmd.addArg("-pass-through");
                cmd.addArg(TypeTextUtil::getPassThroughName(
                    SlangPassThrough(requestState->passThroughMode)));
                break;
            }
        }

        // request->getBackEndReq()->useUnknownImageFormatAsDefault =
        // requestState->useUnknownImageFormatAsDefault; request->getBackEndReq()->obfuscateCode =
        // requestState->obfuscateCode; request->getFrontEndReq()->obfuscateCode =
        // requestState->obfuscateCode;

        switch (requestState->defaultMatrixLayoutMode)
        {
        case SLANG_MATRIX_LAYOUT_ROW_MAJOR:
            cmd.addArg("-matrix-layout-row-major");
            break;
        case SLANG_MATRIX_LAYOUT_COLUMN_MAJOR:
            cmd.addArg("-matrix-layout-column-major");
            break;
        default:
            break;
        }
    }

    // Add the target requests
    {
        for (Index i = 0; i < requestState->targetRequests.getCount(); ++i)
        {
            TargetRequestState& src = base.asRaw(requestState->targetRequests[i]);

            cmd.addArg("-target");
            cmd.addArg(TypeTextUtil::getCompileTargetName(SlangCompileTarget(src.target)));

            if (src.profile != Profile::Unknown)
            {
                cmd.addArg("-profile");
                cmd.addArg(Profile(src.profile).getName());
            }

            if (src.targetFlags & SLANG_TARGET_FLAG_PARAMETER_BLOCKS_USE_REGISTER_SPACES)
            {
                cmd.addArg("-parameter-blocks-use-register-spaces");
            }

            switch (src.floatingPointMode)
            {
            case FloatingPointMode::Fast:
                {
                    cmd.addArg("-fp-mode");
                    cmd.addArg("fast");
                    break;
                }
            case FloatingPointMode::Precise:
                {
                    cmd.addArg("-fp-mode");
                    cmd.addArg("precise");
                    break;
                }
            default:
                break;
            }

#if 0
            // If there is output state (like output filenames) add here
            if (src.outputStates.getCount())
            {
                RefPtr<EndToEndCompileRequest::TargetInfo> dstTargetInfo(new EndToEndCompileRequest::TargetInfo);
                request->targetInfos[dstTarget] = dstTargetInfo;

                for (const auto& srcOutputStateOffset : src.outputStates)
                {
                    const auto& srcOutputState = base.asRaw(srcOutputStateOffset);

                    SLANG_ASSERT(srcOutputState.entryPointIndex < requestState->entryPoints.getCount());

                    String entryPointPath;
                    if (srcOutputState.outputPath)
                    {
                        entryPointPath = base.asRaw(srcOutputState.outputPath)->getSlice();
                    }

                    dstTargetInfo->entryPointOutputPaths.add(srcOutputState.entryPointIndex, entryPointPath);
                }
            }
#endif
        }
    }

    {
        const auto& srcPaths = requestState->searchPaths;
        for (Index i = 0; i < srcPaths.getCount(); ++i)
        {
            cmd.addArg("-I");
            cmd.addArg(base.asRaw(base.asRaw(srcPaths[i]))->getSlice());
        }
    }

    _calcPreprocessorDefines(base, requestState->preprocessorDefinitions, cmd);

    {
        const auto& srcTranslationUnits = requestState->translationUnits;

        for (Index i = 0; i < srcTranslationUnits.getCount(); ++i)
        {
            const auto& srcTranslationUnit = base.asRaw(srcTranslationUnits[i]);

            _calcPreprocessorDefines(base, srcTranslationUnit.preprocessorDefinitions, cmd);


#if 0
            if (srcTranslationUnit.moduleName)
            {
                moduleName = base[srcTranslationUnit].moduleName->getSlice());
            }
#endif

            const auto& srcSourceFiles = srcTranslationUnit.sourceFiles;

            for (Index j = 0; j < srcSourceFiles.getCount(); ++j)
            {
                SourceFileState* sourceFile = base.asRaw(base.asRaw(srcSourceFiles[i]));
                OffsetString* path = base[sourceFile->foundPath];

                if (path)
                {
                    cmd.addArg(path->getSlice());
                }
            }
        }
    }

    // Entry points
    {
        for (const auto& srcEntryPointOffset : requestState->entryPoints)
        {
            const auto srcEntryPoint = base.asRaw(srcEntryPointOffset);

            const char* name =
                srcEntryPoint.name ? base.asRaw(srcEntryPoint.name)->getCstr() : nullptr;

            cmd.addArg("-entry");
            cmd.addArg(name);

            cmd.addArg("-stage");
            UnownedStringSlice stageText = getStageText(srcEntryPoint.profile.getStage());
            cmd.addArg(stageText);

            // cmd.addArg("-profile");
            // cmd.addArg(Profile(srcEntryPoint.profile).getName());

            // List<const char*> args = context.toList(srcEntryPoint.specializationArgStrings);

            // externalRequest->addEntryPointEx(int(srcEntryPoint.translationUnitIndex), name,
            // SlangStage(stage), int(args.getCount()), args.getBuffer());
        }
    }

    return SLANG_OK;
}

/* static */ SlangResult ReproUtil::extractFiles(
    OffsetBase& base,
    RequestState* requestState,
    ISlangMutableFileSystem* fileSystem)
{
    StringBuilder builder;

    builder << "[command-line]\n";

    {
        CommandLine cmdLine;
        _calcCommandLine(base, requestState, cmdLine);
        String text = cmdLine.toString();
        builder << text << "\n";
    }

    builder << "[files]\n";

    for (auto fileOffset : requestState->files)
    {
        auto file = base.asRaw(base.asRaw(fileOffset));

        if (file->contents)
        {
            UnownedStringSlice contents = base.asRaw(file->contents)->getSlice();

            SLANG_RETURN_ON_FAIL(fileSystem->saveFile(
                base.asRaw(file->uniqueName)->getCstr(),
                contents.begin(),
                contents.getLength()));

            OffsetString* originalName = nullptr;
            if (file->canonicalPath)
            {
                originalName = base.asRaw(file->canonicalPath);
            }
            else if (file->foundPath)
            {
                originalName = base.asRaw(file->foundPath);
            }
            else if (file->uniqueIdentity)
            {
                originalName = base.asRaw(file->uniqueIdentity);
            }

            builder << base.asRaw(file->uniqueName)->getSlice() << " -> ";
            if (originalName)
            {
                builder << originalName->getSlice();
            }

            if (builder.getLength() == 0)
            {
                builder << "?";
            }

            builder << "\n";
        }
    }

    builder << "[paths]\n";
    for (const auto pathOffset : requestState->pathInfoMap)
    {
        const auto& path = base.asRaw(pathOffset);

        builder << base.asRaw(path.path)->getSlice() << " -> ";

        const auto pathInfo = base.asRaw(path.pathInfo);

        if (pathInfo)
        {
            if (pathInfo->file)
            {
                builder << base.asRaw(base.asRaw(pathInfo->file)->uniqueName)->getSlice();
            }
            else
            {
                typedef CacheFileSystem::CompressedResult CompressedResult;
                if (pathInfo->getPathTypeResult == CompressedResult::Ok)
                {
                    switch (pathInfo->pathType)
                    {
                    case SLANG_PATH_TYPE_FILE:
                        builder << "file ";
                        break;
                    case SLANG_PATH_TYPE_DIRECTORY:
                        builder << "directory ";
                        break;
                    default:
                        builder << "?";
                        break;
                    }
                }

                CompressedResult curRes = pathInfo->getCanonicalPathResult;
                CompressedResult results[] = {
                    pathInfo->getPathTypeResult,
                    pathInfo->loadFileResult,
                };

                for (auto compRes : results)
                {
                    if (int(compRes) > int(curRes))
                    {
                        curRes = compRes;
                    }
                }

                switch (curRes)
                {
                default:
                case CompressedResult::Uninitialized:
                    break;
                case CompressedResult::Ok:
                    break;

                case CompressedResult::NotFound:
                    builder << "[not found]";
                    break;
                case CompressedResult::CannotOpen:
                    builder << "[cannot open]";
                    break;
                case CompressedResult::Fail:
                    builder << "[fail]";
                    break;
                }
            }
        }
        else
        {
            builder << "[not loaded]";
        }

        builder << "\n";
    }

    SLANG_RETURN_ON_FAIL(
        fileSystem->saveFile("manifest.txt", builder.getBuffer(), builder.getLength()));
    return SLANG_OK;
}

static SlangResult _findFirstSourcePath(EndToEndCompileRequest* request, String& outFilename)
{
    // We are going to look through all of the srcTranlationUnits, looking for the first filename

    auto frontEndReq = request->getFrontEndReq();
    const auto& srcTranslationUnits = frontEndReq->translationUnits;

    for (Index i = 0; i < srcTranslationUnits.getCount(); ++i)
    {
        TranslationUnitRequest* srcTranslationUnit = srcTranslationUnits[i];
        const auto& srcSourceFiles = srcTranslationUnit->getSourceFiles();

        for (Index j = 0; j < srcSourceFiles.getCount(); ++j)
        {
            SourceFile* sourceFile = srcSourceFiles[j];

            const PathInfo& pathInfo = sourceFile->getPathInfo();

            if (pathInfo.foundPath.getLength())
            {
                outFilename = pathInfo.foundPath;
                return SLANG_OK;
            }
        }
    }
    return SLANG_FAIL;
}

/* static */ SlangResult ReproUtil::findUniqueReproDumpStream(
    EndToEndCompileRequest* request,
    String& outFileName,
    RefPtr<Stream>& outStream)
{
    String sourcePath;

    if (SLANG_FAILED(_findFirstSourcePath(request, sourcePath)))
    {
        sourcePath = "unknown.slang";
    }

    String sourceFileName = Path::getFileName(sourcePath);
    String sourceBaseName = Path::getFileNameWithoutExt(sourceFileName);

    RefPtr<FileStream> stream = new FileStream;

    // Okay we need a unique number to make sure the name is unique
    const int maxTries = 100;
    for (int triesCount = 0; triesCount < maxTries; ++triesCount)
    {
        // We could include the count in some way perhaps, but for now let's just go with ticks
        auto tick = Process::getClockTick();

        StringBuilder builder;
        builder << sourceBaseName << "-" << tick << ".slang-repro";

        // We write out the file name tried even if it fails, as might be useful in reporting
        outFileName = builder;

        // We could have clashes, as we use ticks, we should get to a point where the clashes stop
        if (SLANG_SUCCEEDED(
                stream
                    ->init(builder, FileMode::CreateNew, FileAccess::Write, FileShare::WriteOnly)))
        {
            outStream = stream;
            return SLANG_OK;
        }

        // TODO(JS):
        // Might make sense to sleep here - but don't seem to have cross platform func for that yet.
    }

    return SLANG_FAIL;
}

} // namespace Slang
