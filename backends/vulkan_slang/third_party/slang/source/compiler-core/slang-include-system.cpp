// slang-include-system.cpp
#include "slang-include-system.h"

#include "../core/slang-file-system.h"
#include "../core/slang-io.h"
#include "../core/slang-string-util.h"
#include "slang-artifact-impl.h"
#include "slang-artifact-representation-impl.h"
#include "slang-slice-allocator.h"

namespace Slang
{

IncludeSystem::IncludeSystem(
    SearchDirectoryList* searchDirectories,
    ISlangFileSystemExt* fileSystemExt,
    SourceManager* sourceManager)
    : m_searchDirectories(searchDirectories)
    , m_fileSystemExt(fileSystemExt)
    , m_sourceManager(sourceManager)
{
}

SlangResult IncludeSystem::findFile(
    SlangPathType fromPathType,
    const String& fromPath,
    const String& path,
    PathInfo& outPathInfo)
{
    String combinedPath;

    if (fromPath.getLength() == 0 || Path::isAbsolute(path))
    {
        // If the path is absolute or the fromPath is empty, the combined path is just the path
        combinedPath = path;
    }
    else
    {
        // Get relative path
        ComPtr<ISlangBlob> combinedPathBlob;
        SLANG_RETURN_ON_FAIL(m_fileSystemExt->calcCombinedPath(
            fromPathType,
            fromPath.begin(),
            path.begin(),
            combinedPathBlob.writeRef()));
        combinedPath = StringUtil::getString(combinedPathBlob);
        if (combinedPath.getLength() <= 0)
        {
            return SLANG_FAIL;
        }
    }

    // This checks the path exists
    SlangPathType pathType;
    SLANG_RETURN_ON_FAIL(m_fileSystemExt->getPathType(combinedPath.begin(), &pathType));
    if (pathType != SLANG_PATH_TYPE_FILE)
    {
        return SLANG_E_NOT_FOUND;
    }

    // Get the uniqueIdentity
    ComPtr<ISlangBlob> uniqueIdentityBlob;
    SLANG_RETURN_ON_FAIL(m_fileSystemExt->getFileUniqueIdentity(
        combinedPath.begin(),
        uniqueIdentityBlob.writeRef()));

    // If the rel path exists -> a uniqueIdentity MUST exists too
    String uniqueIdentity(StringUtil::getString(uniqueIdentityBlob));
    if (uniqueIdentity.getLength() <= 0)
    {
        // Unique identity can't be empty
        return SLANG_FAIL;
    }

    outPathInfo = PathInfo::makeNormal(combinedPath, uniqueIdentity);
    return SLANG_OK;
}

String IncludeSystem::simplifyPath(const String& path)
{
    ComPtr<ISlangBlob> simplifiedPath;
    if (SLANG_FAILED(m_fileSystemExt->getPath(
            PathKind::Simplified,
            path.getBuffer(),
            simplifiedPath.writeRef())))
    {
        return path;
    }
    return StringUtil::getString(simplifiedPath);
}

SlangResult IncludeSystem::findFile(
    String const& pathToInclude,
    String const& pathIncludedFrom,
    PathInfo& outPathInfo)
{
    outPathInfo.type = PathInfo::Type::Unknown;

    // If it's absolute we only have to try and find if it's there - no need to look at search paths
    if (Path::isAbsolute(pathToInclude))
    {
        // We pass in "" as the from path, so ensure no from path is taken into account
        // and to allow easy identification that this is in effect absolute
        return findFile(
            SLANG_PATH_TYPE_DIRECTORY,
            UnownedStringSlice::fromLiteral(""),
            pathToInclude,
            outPathInfo);
    }

    // Try just relative to current path
    {
        SlangResult res =
            findFile(SLANG_PATH_TYPE_FILE, pathIncludedFrom, pathToInclude, outPathInfo);
        // It either succeeded or wasn't found, anything else is a failure passed back
        if (SLANG_SUCCEEDED(res) || res != SLANG_E_NOT_FOUND)
        {
            return res;
        }
    }

    // Search all the searchDirectories
    for (auto sd = m_searchDirectories; sd; sd = sd->parent)
    {
        for (auto& dir : sd->searchDirectories)
        {
            SlangResult res =
                findFile(SLANG_PATH_TYPE_DIRECTORY, dir.path, pathToInclude, outPathInfo);
            if (SLANG_SUCCEEDED(res) || res != SLANG_E_NOT_FOUND)
            {
                return res;
            }
        }
    }

    return SLANG_E_NOT_FOUND;
}

SlangResult IncludeSystem::loadFile(
    const PathInfo& pathInfo,
    ComPtr<ISlangBlob>& outBlob,
    SourceFile*& outSourceFile)
{
    if (m_sourceManager)
    {
        // See if this an already loaded source file
        outSourceFile = m_sourceManager->findSourceFileRecursively(pathInfo.uniqueIdentity);

        // If not create a new one, and add to the list of known source files
        if (!outSourceFile)
        {
            ComPtr<ISlangBlob> foundSourceBlob;
            if (SLANG_FAILED(m_fileSystemExt->loadFile(
                    pathInfo.foundPath.getBuffer(),
                    foundSourceBlob.writeRef())))
            {
                return SLANG_E_CANNOT_OPEN;
            }

            outSourceFile = m_sourceManager->createSourceFileWithBlob(pathInfo, foundSourceBlob);
            m_sourceManager->addSourceFile(pathInfo.uniqueIdentity, outSourceFile);

            outBlob = foundSourceBlob;
            return SLANG_OK;
        }
        else
        {
            if (outSourceFile->getContentBlob())
            {
                outBlob = outSourceFile->getContentBlob();
                return SLANG_OK;
            }

            ComPtr<ISlangBlob> foundSourceBlob;
            if (SLANG_FAILED(m_fileSystemExt->loadFile(
                    pathInfo.foundPath.getBuffer(),
                    foundSourceBlob.writeRef())))
            {
                return SLANG_E_CANNOT_OPEN;
            }

            outSourceFile->setContents(foundSourceBlob);

            outBlob = foundSourceBlob;
            return SLANG_OK;
        }
    }
    else
    {
        // If we don't have the source manager, just load
        outSourceFile = nullptr;
        return m_fileSystemExt->loadFile(pathInfo.foundPath.getBuffer(), outBlob.writeRef());
    }
}

SlangResult IncludeSystem::findAndLoadFile(
    const String& pathToInclude,
    const String& pathIncludedFrom,
    PathInfo& outPathInfo,
    ComPtr<ISlangBlob>& outBlob)
{
    SLANG_RETURN_ON_FAIL(findFile(pathToInclude, pathIncludedFrom, outPathInfo));
    SLANG_RETURN_ON_FAIL(loadFile(outPathInfo, outBlob));
    return SLANG_OK;
}

} // namespace Slang
