#ifndef SLANG_FILE_SYSTEM_H
#define SLANG_FILE_SYSTEM_H

#include "../../core/slang-com-object.h"
#include "record-manager.h"
#include "slang-com-helper.h"
#include "slang-com-ptr.h"

namespace SlangRecord
{

using namespace Slang;

// slang always requires ISlangFileSystemExt interface, even if user only provides ISlangFileSystem,
// slang will still wrap it with ISlangFileSystemExt. So we have to record ISlangFileSystemExt, even
// though we only need to record loadFile() function.
class FileSystemRecorder : public RefObject, public ISlangFileSystemExt
{
public:
    explicit FileSystemRecorder(ISlangFileSystemExt* fileSystem, RecordManager* recordManager);

    // ISlangUnknown
    SLANG_REF_OBJECT_IUNKNOWN_ALL

    ISlangUnknown* getInterface(const Slang::Guid& guid);

    // ISlangCastable
    virtual SLANG_NO_THROW void* SLANG_MCALL castAs(const Slang::Guid& guid) override;

    // ISlangFileSystem
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    loadFile(char const* path, ISlangBlob** outBlob) override;

    // ISlangFileSystemExt
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    getFileUniqueIdentity(const char* path, ISlangBlob** outUniqueIdentity) override;

    virtual SLANG_NO_THROW SlangResult SLANG_MCALL calcCombinedPath(
        SlangPathType fromPathType,
        const char* fromPath,
        const char* path,
        ISlangBlob** pathOut) override;

    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    getPathType(const char* path, SlangPathType* pathTypeOut) override;

    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    getPath(PathKind kind, const char* path, ISlangBlob** outPath) override;

    virtual SLANG_NO_THROW void SLANG_MCALL clearCache() override;

    virtual SLANG_NO_THROW SlangResult SLANG_MCALL enumeratePathContents(
        const char* path,
        FileSystemContentsCallBack callback,
        void* userData) override;

    virtual SLANG_NO_THROW OSPathKind SLANG_MCALL getOSPathKind() override;

private:
    Slang::ComPtr<ISlangFileSystemExt> m_actualFileSystem;
    RecordManager* m_recordManager = nullptr;
};

} // namespace SlangRecord
#endif
