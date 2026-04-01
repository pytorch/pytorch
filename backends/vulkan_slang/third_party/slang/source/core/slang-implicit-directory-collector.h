#ifndef SLANG_CORE_IMPLICIT_DIRECTORY_COLLECTOR_H
#define SLANG_CORE_IMPLICIT_DIRECTORY_COLLECTOR_H

#include "slang-basic.h"
#include "slang-string-slice-index-map.h"

namespace Slang
{

/* This class helps to find the contents and/or existence of an implicit directory.This finds the
contents of a directory.

This is achieved by using a path prefix that any contained path must at least match. If the
remainder of the path contains a folder
 - detectable because it's not a leaf and so contains a delimiter - that directory is added. As a
sub folder may contain many files, and the directory itself may also be defined, it is necessary to
dedup. The deduping is handled by the StringSliceIndexMap. */
class ImplicitDirectoryCollector
{
public:
    enum class State
    {
        None,            ///< Neither the directory or content have been found
        DirectoryExists, ///< The directory exists
        HasContent,      ///< If it has content, the directory must exist
    };

    /// Get the current state
    State getState() const
    {
        return (m_map.getCount() > 0) ? State::HasContent
                                      : (m_directoryExists ? State::DirectoryExists : State::None);
    }
    /// True if collector at least has the specified state
    bool hasState(State state) { return Index(getState()) >= Index(state); }

    /// Set that it exists
    void setDirectoryExists(bool directoryExists) { m_directoryExists = directoryExists; }
    /// Get if it exists (implicitly or explicitly)
    bool getDirectoryExists() const { return m_directoryExists || m_map.getCount() > 0; }

    /// True if the path matches the prefix
    bool hasPrefix(const UnownedStringSlice& path) const
    {
        return path.startsWith(m_prefix.getUnownedSlice());
    }

    /// True if the directory has content
    bool hasContent() const { return m_map.getCount() > 0; }

    /// Gets the remainder or path after the prefix
    UnownedStringSlice getRemainder(const UnownedStringSlice& path) const
    {
        SLANG_ASSERT(hasPrefix(path));
        return UnownedStringSlice(path.begin() + m_prefix.getLength(), path.end());
    }

    /// Add a remaining path
    void addRemainingPath(SlangPathType pathType, const UnownedStringSlice& inPathRemainder);
    /// Add a path
    void addPath(SlangPathType pathType, const UnownedStringSlice& canonicalPath);
    /// Enumerate the contents
    SlangResult enumerate(FileSystemContentsCallBack callback, void* userData);

    /// Ctor
    ImplicitDirectoryCollector(const String& canonicalPath, bool directoryExists = false);

    static bool isRootPath(const UnownedStringSlice& path);

protected:
    StringSliceIndexMap m_map;
    String m_prefix;
    bool m_directoryExists;
};

} // namespace Slang

#endif
