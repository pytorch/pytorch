// slang-command-line.h
#ifndef SLANG_COMMAND_LINE_H
#define SLANG_COMMAND_LINE_H

#include "slang-list.h"
#include "slang-string.h"

namespace Slang
{

struct ExecutableLocation
{
    typedef ExecutableLocation ThisType;

    enum Type
    {
        Unknown, ///< Not specified
        Path,    ///< The executable is set as a path (ie won't be searched for)
        Name,    ///< The executable is passed as a name which will be searched for
    };

    /// Set the executable path.
    /// NOTE! On some targets the executable path *must* include an extension to be able to start as
    /// a process
    void setPath(const String& path)
    {
        m_type = Type::Path;
        m_pathOrName = path;
    }

    /// Set a filename (such that the path will be looked up)
    void setName(const String& filename)
    {
        m_type = Type::Name;
        m_pathOrName = filename;
    }

    void set(Type type, const String& pathOrName)
    {
        m_type = type;
        m_pathOrName = pathOrName;
    }

    /// Set the executable path from a base directory and an executable name (no suffix such as
    /// '.exe' needed)
    void set(const String& dir, const String& name);

    /// Determines if it's a name or a path when it sets
    void set(const String& nameOrPath);

    /// Append as text to out.
    void append(StringBuilder& out) const;

    /// Reset state to be same as ctor
    void reset()
    {
        m_type = Type::Unknown;
        m_pathOrName = String();
    }

    /// Equality means exactly the same definition.
    /// *NOT* that exactly the same executable is specified
    bool operator==(const ThisType& rhs) const
    {
        return m_type == rhs.m_type && m_pathOrName == rhs.m_pathOrName;
    }
    bool operator!=(const ThisType& rhs) const { return !(*this == rhs); }

    ExecutableLocation() {}
    ExecutableLocation(const String& dir, const String& name) { set(dir, name); }
    ExecutableLocation(Type type, const String& pathOrName)
        : m_type(type), m_pathOrName(pathOrName)
    {
    }

    explicit ExecutableLocation(const String& nameOrPath) { set(nameOrPath); }

    Type m_type = Type::Unknown;
    String m_pathOrName;
};

struct CommandLine
{
    typedef CommandLine ThisType;

    /// Add args - assumed unescaped
    void addArg(const String& in) { m_args.add(in); }
    void addArgs(const String* args, Int argsCount)
    {
        for (Int i = 0; i < argsCount; ++i)
            addArg(args[i]);
    }

    void addArgIfNotFound(const String& in);

    /// Find the index of an arg which is exact match for slice
    SLANG_INLINE Index findArgIndex(const UnownedStringSlice& slice) const
    {
        return m_args.indexOf(slice);
    }

    /// For handling args where the switch is placed directly in front of the path
    void addPrefixPathArg(
        const char* prefix,
        const String& path,
        const char* pathPostfix = nullptr);

    /// Get the total number of args
    SLANG_FORCE_INLINE Index getArgCount() const { return m_args.getCount(); }

    /// Reset to the initial state
    void reset() { *this = CommandLine(); }

    /// Append the args
    void appendArgs(StringBuilder& out) const;

    /// Append the command line to out
    void append(StringBuilder& out) const;
    /// convert into a string
    String toString() const;

    /// Convert just the args to string
    String toStringArgs() const;

    /// Set an executable location
    void setExecutableLocation(const ExecutableLocation& loc) { m_executableLocation = loc; }

    ExecutableLocation m_executableLocation; ///< The executable location
    List<String> m_args;                     ///< The arguments (Stored *unescaped*)
};

} // namespace Slang

#endif // SLANG_COMMAND_LINE_H
