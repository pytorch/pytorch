// slang-artifact-representation.h
#ifndef SLANG_ARTIFACT_REPRESENTATION_H
#define SLANG_ARTIFACT_REPRESENTATION_H

#include "slang-artifact.h"

namespace Slang
{

/* Base interface for types that have a path. */
class IPathArtifactRepresentation : public IArtifactRepresentation
{
    SLANG_COM_INTERFACE(
        0xcb1c188c,
        0x7e48,
        0x43eb,
        {0xb0, 0x9a, 0xa1, 0x6e, 0xef, 0xd4, 0x9b, 0xef});

    /// The path
    virtual SLANG_NO_THROW const char* SLANG_MCALL getPath() = 0;
    /// Get type
    virtual SLANG_NO_THROW SlangPathType SLANG_MCALL getPathType() = 0;
    /// Returns the unique identity. If a unique identity is not supported
    /// or available will return nullptr.
    virtual SLANG_NO_THROW const char* SLANG_MCALL getUniqueIdentity() = 0;
};

/* Represents a path to a file  held on an ISlangFileSystem. */
class IExtFileArtifactRepresentation : public IPathArtifactRepresentation
{
    SLANG_COM_INTERFACE(
        0xacd65576,
        0xb09d,
        0x4ac9,
        {0xa5, 0x93, 0xeb, 0xf8, 0x9b, 0xd7, 0x11, 0xfd});

    /// File system that holds the item along the path.
    virtual SLANG_NO_THROW ISlangFileSystemExt* SLANG_MCALL getFileSystem() = 0;
};

/*
A representation as a file on the OS file system.  */
class IOSFileArtifactRepresentation : public IPathArtifactRepresentation
{
public:
    SLANG_COM_INTERFACE(
        0xc7d7d3a4,
        0x8683,
        0x44b5,
        {0x87, 0x96, 0xdf, 0xba, 0x9b, 0xc3, 0xf1, 0x7b});

    /* Determines ownership and other characteristics of the OS 'file' */
    enum class Kind
    {
        Reference, ///< References a file on the file system
        NameOnly,  ///< Typically used for items that can be found by the 'system'. The path is just
                   ///< a name, and cannot typically be loaded as a blob.
        Owned,     ///< File is *owned* by this instance and will be deleted when goes out of scope
        Lock, ///< An owned type, indicates potentially in part may only exist to 'lock' a path for
              ///< a temporary file. Other files might exists based on the 'lock' path.
        CountOf,
    };

    /// The the kind of file.
    virtual SLANG_NO_THROW Kind SLANG_MCALL getKind() = 0;
    /// Makes the file no longer owned. Only applicable for Owned/Lock and they will become
    /// 'Reference'
    virtual SLANG_NO_THROW void SLANG_MCALL disown() = 0;
    /// Gets the 'lock file' if any associated with this file. Returns nullptr if there isn't one.
    /// If this file is based on a 'lock file', the lock file must stay in scope at least as long as
    /// this does.
    virtual SLANG_NO_THROW IOSFileArtifactRepresentation* SLANG_MCALL getLockFile() = 0;
};

} // namespace Slang

#endif
