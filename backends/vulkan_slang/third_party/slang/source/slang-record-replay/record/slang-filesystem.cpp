#include "slang-filesystem.h"

#include "../../core/slang-io.h"
#include "../util/record-utility.h"
#include "output-stream.h"

#include <stdlib.h>

namespace SlangRecord
{
// We don't actually need to record the methods of ISlangFileSystemExt, we just want to record the
// file content and save them into disk.
FileSystemRecorder::FileSystemRecorder(
    ISlangFileSystemExt* fileSystem,
    RecordManager* recordManager)
    : m_actualFileSystem(fileSystem), m_recordManager(recordManager)
{
    SLANG_RECORD_ASSERT(m_actualFileSystem);
    SLANG_RECORD_ASSERT(m_recordManager);
    slangRecordLog(LogLevel::Verbose, "%s: %p\n", __PRETTY_FUNCTION__, m_actualFileSystem.get());
}

void* FileSystemRecorder::castAs(const Slang::Guid& guid)
{
    return getInterface(guid);
}

ISlangUnknown* FileSystemRecorder::getInterface(const Slang::Guid& guid)
{
    if (guid == ISlangUnknown::getTypeGuid() || guid == ISlangFileSystem::getTypeGuid())
        return static_cast<ISlangFileSystem*>(this);
    return nullptr;
}

// TODO: There could be a potential issue that could not be able to dump the generated file content
// correctly. Details: https://github.com/shader-slang/slang/issues/4423.
SLANG_NO_THROW SlangResult FileSystemRecorder::loadFile(char const* path, ISlangBlob** outBlob)
{
    slangRecordLog(
        LogLevel::Verbose,
        "%p: %s, :%s\n",
        m_actualFileSystem.get(),
        __PRETTY_FUNCTION__,
        path);
    SlangResult res = m_actualFileSystem->loadFile(path, outBlob);

    // Since the loadFile method could be implemented by client, we can't guarantee the result is
    // always as expected, we will check every thing to make sure we won't crash at writing file.
    //
    // We can only dump the file content after this 'loadFile' call, no matter this call crashes or
    // file is not found, we can't save the file anyway, so we don't need to pay special care to the
    // crash recovery. We will know something wrong with the loadFile call if we can't find the file
    // in the record directory.
    if ((res == SLANG_OK) && (*outBlob != nullptr) && ((*outBlob)->getBufferSize() != 0))
    {
        Slang::String filePath =
            Slang::Path::combine(m_recordManager->getRecordFileDirectory(), path);
        Slang::String dirPath = Slang::Path::getParentDirectory(filePath);
        if (!File::exists(dirPath))
        {
            slangRecordLog(
                LogLevel::Debug,
                "Create directory: %s to save captured shader file: %s\n",
                dirPath.getBuffer(),
                filePath.getBuffer());

            if (!Path::createDirectoryRecursive(dirPath))
            {
                slangRecordLog(
                    LogLevel::Error,
                    "Fail to create directory: %s\n",
                    dirPath.getBuffer());
                return SLANG_FAIL;
            }
        }

        FileOutputStream fileStream(filePath);

        fileStream.write((*outBlob)->getBufferPointer(), (*outBlob)->getBufferSize());
        fileStream.flush();
    }
    return res;
}

SLANG_NO_THROW SlangResult
FileSystemRecorder::getFileUniqueIdentity(const char* path, ISlangBlob** outUniqueIdentity)
{
    slangRecordLog(
        LogLevel::Verbose,
        "%p: %s :\"%s\"\n",
        m_actualFileSystem.get(),
        __PRETTY_FUNCTION__,
        path);
    SlangResult res = m_actualFileSystem->getFileUniqueIdentity(path, outUniqueIdentity);
    return res;
}

SLANG_NO_THROW SlangResult FileSystemRecorder::calcCombinedPath(
    SlangPathType fromPathType,
    const char* fromPath,
    const char* path,
    ISlangBlob** pathOut)
{
    slangRecordLog(
        LogLevel::Verbose,
        "%p: %s, :%s\n",
        m_actualFileSystem.get(),
        __PRETTY_FUNCTION__,
        path);
    SlangResult res = m_actualFileSystem->calcCombinedPath(fromPathType, fromPath, path, pathOut);
    return res;
}

SLANG_NO_THROW SlangResult
FileSystemRecorder::getPathType(const char* path, SlangPathType* pathTypeOut)
{
    slangRecordLog(
        LogLevel::Verbose,
        "%p: %s, :%s\n",
        m_actualFileSystem.get(),
        __PRETTY_FUNCTION__,
        path);
    SlangResult res = m_actualFileSystem->getPathType(path, pathTypeOut);
    return res;
}

SLANG_NO_THROW SlangResult
FileSystemRecorder::getPath(PathKind kind, const char* path, ISlangBlob** outPath)
{
    slangRecordLog(
        LogLevel::Verbose,
        "%p: %s, :%s\n",
        m_actualFileSystem.get(),
        __PRETTY_FUNCTION__,
        path);
    SlangResult res = m_actualFileSystem->getPath(kind, path, outPath);
    return res;
}

SLANG_NO_THROW void FileSystemRecorder::clearCache()
{
    slangRecordLog(LogLevel::Verbose, "%p: %s\n", m_actualFileSystem.get(), __PRETTY_FUNCTION__);
    m_actualFileSystem->clearCache();
}

SLANG_NO_THROW SlangResult FileSystemRecorder::enumeratePathContents(
    const char* path,
    FileSystemContentsCallBack callback,
    void* userData)
{
    slangRecordLog(
        LogLevel::Verbose,
        "%p: %s, :%s\n",
        m_actualFileSystem.get(),
        __PRETTY_FUNCTION__,
        path);
    SlangResult res = m_actualFileSystem->enumeratePathContents(path, callback, userData);
    return res;
}

SLANG_NO_THROW OSPathKind FileSystemRecorder::getOSPathKind()
{
    slangRecordLog(LogLevel::Verbose, "%p: %s\n", m_actualFileSystem.get(), __PRETTY_FUNCTION__);
    OSPathKind pathKind = m_actualFileSystem->getOSPathKind();
    return pathKind;
}
} // namespace SlangRecord
