#ifndef SLANG_CORE_IO_H
#define SLANG_CORE_IO_H

#include "slang-blob.h"
#include "slang-secure-crt.h"
#include "slang-stream.h"
#include "slang-string.h"
#include "slang-text-io.h"

namespace Slang
{
class File
{
public:
    static bool exists(const String& fileName);

    static SlangResult readAllText(const String& fileName, String& outString);

    static SlangResult readAllBytes(const String& fileName, List<unsigned char>& out);
    static SlangResult readAllBytes(const String& fileName, ScopedAllocation& out);

    static SlangResult writeAllText(const String& fileName, const String& text);

    static SlangResult writeAllTextIfChanged(const String& fileName, UnownedStringSlice text);

    /// Write as text in native form for the target (so typically may change line endings )
    static SlangResult writeNativeText(const String& filename, const void* data, size_t size);

    static SlangResult writeAllBytes(const String& fileName, const void* data, size_t size);

    static SlangResult remove(const String& fileName);

    static SlangResult makeExecutable(const String& fileName);

    /// Creates a temporary file typically in some way based on the prefix
    /// The file will be *created* with the outFileName, on success.
    /// It's creation in necessary to lock that particular name.
    static SlangResult generateTemporary(const UnownedStringSlice& prefix, String& outFileName);
};

class Path
{
public:
    typedef uint32_t SimplifyIntegral;

    struct SimplifyFlag
    {
        enum Enum : SimplifyIntegral
        {
            /// Can only simplify to an absolute path. Will return an error if not possible.
            /// Useful to constrain a path, such as when wanting something like 'chroot'.
            AbsoluteOnly = 0x1,
            /// If the simplified path is a root path, remove the root.
            /// Will mean that for example
            /// "/" -> "."
            /// "/a/.." -> "."
            /// "/a" -> "a"
            /// Its worth noting that a path prefixed "/" will never be returned and if *just* the
            /// root it specified it will return as ".".
            NoRoot = 0x2,
        };
    };

    // A more convenient typesafe way to specify the SimplifyFlag combinations
    enum SimplifyStyle : SimplifyIntegral
    {
        Normal = 0,
        AbsoluteOnly = SimplifyFlag::AbsoluteOnly,
        NoRoot = SimplifyFlag::NoRoot,
        AbsoluteOnlyAndNoRoot = SimplifyFlag::AbsoluteOnly | SimplifyFlag::NoRoot,
    };

    enum class Type
    {
        Unknown,
        File,
        Directory,
    };

    typedef uint32_t TypeFlags;
    struct TypeFlag
    {
        enum Enum : TypeFlags
        {
            Unknown = TypeFlags(1) << int(Type::Unknown),
            File = TypeFlags(1) << int(Type::File),
            Directory = TypeFlags(1) << int(Type::Directory),
        };
    };

    class Visitor
    {
    public:
        virtual void accept(Type type, const UnownedStringSlice& filename) = 0;
    };

    static const char kPathDelimiter = '/';

#if SLANG_WINDOWS_FAMILY
    static const char kOSCanonicalPathDelimiter = '\\';
    static const char kOSAlternativePathDelimiter = '/';

#else
    static const char kOSCanonicalPathDelimiter = '/';
    static const char kOSAlternativePathDelimiter = '/';
#endif

    /// Finds all all the items in the specified directory, that matches the pattern.
    ///
    /// @param directoryPath The directory to do the search in. If the directory is not found,
    /// SLANG_E_NOT_FOUND is returned
    /// @param pattern. The pattern to match against. The pattern matching is targtet specific (ie
    /// window matching is different to linux/unix). Passing nullptr means no matching.
    /// @return is SLANG_E_NOT_FOUND if the directoryPath is not found
    static SlangResult find(const String& directoryPath, const char* pattern, Visitor* visitor);

    /// Returns -1 if no separator is found
    static Index findLastSeparatorIndex(String const& path)
    {
        return findLastSeparatorIndex(path.getUnownedSlice());
    }
    static Index findLastSeparatorIndex(UnownedStringSlice const& path);
    /// Finds the index of the last dot in a path, else returns -1
    static Index findExtIndex(String const& path) { return findExtIndex(path.getUnownedSlice()); }
    static Index findExtIndex(UnownedStringSlice const& path);

    /// True if isn't just a name (ie has any path separator)
    /// Note this is no the same as having a 'parent' as '/thing' 'has a path', but it doesn't have
    /// a parent.
    static bool hasPath(const UnownedStringSlice& path)
    {
        return findLastSeparatorIndex(path) >= 0;
    }
    static bool hasPath(const String& path) { return findLastSeparatorIndex(path) >= 0; }

    static String replaceExt(const String& path, const char* newExt);
    static String getFileName(const String& path);
    static String getPathWithoutExt(const String& path);

    static String getPathExt(const String& path) { return getPathExt(path.getUnownedSlice()); }
    static UnownedStringSlice getPathExt(const UnownedStringSlice& path);

    static String getParentDirectory(const String& path);

    static String getFileNameWithoutExt(const String& path);

    static String combine(const String& path1, const String& path2);
    static String combine(const String& path1, const String& path2, const String& path3);

    /// Combine path sections and store the result in outBuilder
    static void combineIntoBuilder(
        const UnownedStringSlice& path1,
        const UnownedStringSlice& path2,
        StringBuilder& outBuilder);

    /// Append a path, taking into account path separators onto the end of ioBuilder
    static void append(StringBuilder& ioBuilder, const UnownedStringSlice& path);

    static bool createDirectory(const String& path);
    static bool createDirectoryRecursive(const String& path);

    /// Accept either style of delimiter
    SLANG_FORCE_INLINE static bool isDelimiter(char c) { return c == '/' || c == '\\'; }

    /// True if the element appears to be a drive specification (where element is the prefix to a
    /// path that isn't a directory)
    /// @param pathPrefix The path prefix to test if it's a drive specification
    static bool isDriveSpecification(const UnownedStringSlice& pathPrefix);

    /// Splits the path into it's individual bits
    /// Absolute paths of the form "/" will become [""]
    /// Absolute paths of the form "a:/" will become ["a:", ""]
    /// A drive specification of the form "a:" will become ["a:"]
    /// Relative paths that are in effect "." will become []
    static void split(const UnownedStringSlice& path, List<UnownedStringSlice>& splitOut);

    /// Strips .. and . as much as it can
    static String simplify(const UnownedStringSlice& path);
    static String simplify(const String& path) { return simplify(path.getUnownedSlice()); }

    /// Given a path simplifies it such the the resultant path is absolute (ie contains no . or ..)
    /// Same behavior as simplify around the root
    static SlangResult simplify(
        const UnownedStringSlice& path,
        SimplifyStyle style,
        StringBuilder& outPath);
    static SlangResult simplify(const String& path, SimplifyStyle style, StringBuilder& outPath)
    {
        return simplify(path.getUnownedSlice(), style, outPath);
    }
    static SlangResult simplify(const char* path, SimplifyStyle style, StringBuilder& outPath)
    {
        return simplify(UnownedStringSlice(path), style, outPath);
    }

    /// Simplifies the path split up
    static void simplify(List<UnownedStringSlice>& ioSplit);

    /// Join the parts of the path to produce an output path
    static void join(const UnownedStringSlice* slices, Index count, StringBuilder& out);

    /// Returns true if the path is absolute
    static bool isAbsolute(const UnownedStringSlice& path);
    static bool isAbsolute(const String& path) { return isAbsolute(path.getUnownedSlice()); }

    /// Returns true if path contains contains an element of . or ..
    static bool hasRelativeElement(const UnownedStringSlice& path);
    static bool hasRelativeElement(const String& path)
    {
        return hasRelativeElement(path.getUnownedSlice());
    }

    /// Determines the type of file at the path
    /// @param path The path to test
    /// @param outPathType Holds the object type at the path on success
    /// @return SLANG_OK on success
    static SlangResult getPathType(const String& path, SlangPathType* outPathType);

    /// Determines the canonical equivalent path to path.
    /// The path returned should reference the identical object - and two different references to
    /// the same path should return the same canonical path
    /// @param path Path to get the canonical path for
    /// @param outCanonicalPath The canonical path for 'path' is call is successful
    /// @return SLANG_OK on success
    static SlangResult getCanonical(const String& path, String& outCanonicalPath);

    /// Returns the current working directory
    /// @return The path in platform native format. Returns empty string if failed.
    static String getCurrentPath();

    /// Returns the executable path
    /// @return The path in platform native format. Returns empty string if failed.
    static String getExecutablePath();

    /// Returns the first element of the path or an empty slice if there is none
    /// This broadly equivalent to returning the first element of split
    /// @param path Path to extract first element from
    /// @return The first element of the path, or empty
    static UnownedStringSlice getFirstElement(const UnownedStringSlice& path);

    /// Remove a file or directory at specified path. The directory must be empty for it to be
    /// removed
    /// @param path
    /// @return SLANG_OK if file or directory is removed
    static SlangResult remove(const String& path);

    /// Remove a file or directory at specified path. The directory can be non-empty.
    /// @param path
    /// @return SLANG_OK if file or directory is removed
    static SlangResult removeNonEmpty(const String& path);

    static bool equals(String path1, String path2);

    /// Turn `path` into a relative path from base.
    static String getRelativePath(String base, String path);
};

struct URI
{
    String uri;
    bool operator==(const URI& other) const { return uri == other.uri; }
    bool operator!=(const URI& other) const { return uri != other.uri; }

    HashCode getHashCode() const { return uri.getHashCode(); }

    bool isLocalFile() { return uri.startsWith("file://"); };
    String getPath() const;
    StringSlice getProtocol() const;

    static URI fromLocalFilePath(UnownedStringSlice path);
    static URI fromString(UnownedStringSlice uriString);
    static bool isSafeURIChar(char ch);
};

/// Helper class abstracting lock files.
/// Uses LockFileEx() on windows systems and flock() on POSIX systems.
class LockFile
{
public:
    enum class LockType
    {
        Exclusive,
        Shared,
    };

    /// Open the lock file. This will create the file if it doesn't exist yet.
    /// @param fileName File name to open.
    /// @return SLANG_OK on success.
    SlangResult open(const String& fileName);

    /// Closes the lock file.
    void close();

    /// Returns true if the lock file is open.
    bool isOpen() const { return m_isOpen; }

    /// Acquire the lock in non-blocking mode.
    /// @param lockType Lock type (Exclusive or Shared).
    /// @return SLANG_OK on success. SLANG_E_TIME_OUT if the lock is already held.
    SlangResult tryLock(LockType lockType = LockType::Exclusive);

    /// Acquire the lock in blocking mode.
    /// @param lockType Lock type (Exclusive or Shared).
    /// @return SLANG_OK on success.
    SlangResult lock(LockType lockType = LockType::Exclusive);

    /// Release the lock.
    /// @return SLANG_OK on success.
    SlangResult unlock();

    LockFile();
    ~LockFile();

private:
    LockFile(const LockFile&) = delete;
    LockFile(LockFile&&) = delete;
    LockFile& operator=(const LockFile&) = delete;
    LockFile& operator=(LockFile&&) = delete;

#if SLANG_WINDOWS_FAMILY
    void* m_fileHandle;
#else
    int m_fileHandle;
#endif
    bool m_isOpen;
};

class LockFileGuard
{
public:
    LockFileGuard(LockFile& lockFile, LockFile::LockType lockType = LockFile::LockType::Exclusive)
        : m_lockFile(lockFile)
    {
        m_lockFile.lock(lockType);
    }

    ~LockFileGuard() { m_lockFile.unlock(); }

private:
    LockFileGuard(const LockFileGuard&) = delete;
    LockFileGuard(LockFileGuard&&) = delete;
    LockFileGuard& operator=(const LockFileGuard&) = delete;
    LockFileGuard& operator=(LockFileGuard&&) = delete;

    LockFile& m_lockFile;
};

} // namespace Slang

#endif
