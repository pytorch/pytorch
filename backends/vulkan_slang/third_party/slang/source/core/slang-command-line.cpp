// slang-command-line.cpp
#include "slang-command-line.h"

#include "slang-com-helper.h"
#include "slang-process.h"
#include "slang-string-escape-util.h"
#include "slang-string-util.h"
#include "slang-string.h"

namespace Slang
{

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ExecutableLocation !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 */

void ExecutableLocation::set(const String& dir, const String& name)
{
    if (dir.getLength() == 0)
    {
        set(name);
    }
    else
    {
        set(Path::combine(dir, name));
    }
}

void ExecutableLocation::set(const String& nameOrPath)
{
    // See if input looks like a path
    if (Path::hasPath(nameOrPath))
    {
        // If it is a path we may want to add a suffix
        const auto suffix = Process::getExecutableSuffix();

        if (suffix.getLength() == 0 || nameOrPath.endsWith(suffix))
        {
            setPath(nameOrPath);
        }
        else
        {
            // If on target that has suffix make sure name has the suffix
            StringBuilder builder;
            builder << nameOrPath;
            builder << suffix;
            setPath(builder.produceString());
        }
    }
    else
    {
        // If we don't have a parent, we assume it is just a naem
        setName(nameOrPath);
    }
}

void ExecutableLocation::append(StringBuilder& out) const
{
    if (m_type == Type::Unknown)
    {
        out << "(unknown)";
    }
    else
    {
        auto escapeHandler = Process::getEscapeHandler();
        StringEscapeUtil::appendMaybeQuoted(escapeHandler, m_pathOrName.getUnownedSlice(), out);
    }
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! CommandLine !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

void CommandLine::addPrefixPathArg(const char* prefix, const String& path, const char* pathPostfix)
{
    StringBuilder builder;
    builder << prefix;

    // TODO(JS): The assumption here is that quoting will be added as necessary and
    // -prefixSomething Else
    // is okay as
    // "-prefixSomething Else" rather than
    // -prefix"Something Else"

    builder << path;

    if (pathPostfix)
    {
        // Work out the path with the postfix
        builder << pathPostfix;
    }
    addArg(builder.produceString());
}

void CommandLine::append(StringBuilder& out) const
{
    m_executableLocation.append(out);

    if (m_args.getCount())
    {
        out << " ";
        appendArgs(out);
    }
}

void CommandLine::appendArgs(StringBuilder& out) const
{
    auto escapeHandler = Process::getEscapeHandler();

    const Int argCount = m_args.getCount();
    for (Index i = 0; i < argCount; ++i)
    {
        const auto& arg = m_args[i];
        if (i > 0)
        {
            out << " ";
        }
        StringEscapeUtil::appendMaybeQuoted(escapeHandler, arg.getUnownedSlice(), out);
    }
}

void CommandLine::addArgIfNotFound(const String& in)
{
    if (m_args.indexOf(in) < 0)
    {
        addArg(in);
    }
}

String CommandLine::toString() const
{
    StringBuilder buf;
    append(buf);
    return buf.produceString();
}

String CommandLine::toStringArgs() const
{
    StringBuilder buf;
    appendArgs(buf);
    return buf.produceString();
}

} // namespace Slang
