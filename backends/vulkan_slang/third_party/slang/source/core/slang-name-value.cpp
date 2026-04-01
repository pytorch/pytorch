// slang-name-value.cpp

#include "slang-name-value.h"

#include "slang-char-util.h"
#include "slang-string-util.h"

namespace Slang
{

/* static */ ValueInt NameValueUtil::findValue(
    const ConstArrayView<NameValue>& opts,
    const UnownedStringSlice& slice,
    ValueInt defaultValue)
{
    for (const auto& opt : opts)
    {
        if (UnownedStringSlice(opt.name) == slice)
        {
            return opt.value;
        }
    }
    return defaultValue;
}

/* static */ ValueInt NameValueUtil::findValue(
    const ConstArrayView<NamesValue>& opts,
    const UnownedStringSlice& slice,
    ValueInt defaultValue)
{
    for (const auto& opt : opts)
    {
        UnownedStringSlice names(opt.names);

        if (StringUtil::indexOfInSplit(names, ',', slice) >= 0)
        {
            return opt.value;
        }
    }
    return defaultValue;
}

/* static */ ValueInt NameValueUtil::findValue(
    const ConstArrayView<NamesDescriptionValue>& opts,
    const UnownedStringSlice& slice,
    ValueInt defaultValue)
{
    for (const auto& opt : opts)
    {
        UnownedStringSlice names(opt.names);
        if (StringUtil::indexOfInSplit(names, ',', slice) >= 0)
        {
            return opt.value;
        }
    }
    return defaultValue;
}

/* static */ UnownedStringSlice NameValueUtil::findName(
    const ConstArrayView<NameValue>& opts,
    ValueInt value,
    const UnownedStringSlice& defaultName)
{
    for (const auto& opt : opts)
    {
        if (opt.value == value)
        {
            return UnownedStringSlice(opt.name);
        }
    }
    return defaultName;
}

/* static */ UnownedStringSlice NameValueUtil::findName(
    const ConstArrayView<NamesValue>& opts,
    ValueInt value,
    const UnownedStringSlice& defaultName)
{
    for (const auto& opt : opts)
    {
        if (opt.value == value)
        {
            // Get the first name
            return StringUtil::getAtInSplit(UnownedStringSlice(opt.names), ',', 0);
        }
    }

    return defaultName;
}

/* static */ UnownedStringSlice NameValueUtil::findName(
    const ConstArrayView<NamesDescriptionValue>& opts,
    ValueInt value,
    const UnownedStringSlice& defaultName)
{
    for (const auto& opt : opts)
    {
        if (opt.value == value)
        {
            // Get the first name
            return StringUtil::getAtInSplit(UnownedStringSlice(opt.names), ',', 0);
        }
    }

    return defaultName;
}


/* static */ UnownedStringSlice NameValueUtil::findDescription(
    const ConstArrayView<NamesDescriptionValue>& opts,
    ValueInt value,
    const UnownedStringSlice& defaultDescription)
{
    for (const auto& opt : opts)
    {
        if (opt.value == value && opt.description)
        {
            return UnownedStringSlice(opt.description);
        }
    }

    return defaultDescription;
}

/* static */ void NameValueUtil::appendNames(
    NameKind kind,
    const ConstArrayView<NameValue>& opts,
    List<UnownedStringSlice>& out)
{
    SLANG_UNUSED(kind);
    for (auto& opt : opts)
    {
        out.add(UnownedStringSlice(opt.name));
    }
}

static void _appendNames(
    NameValueUtil::NameKind kind,
    const char* names,
    List<UnownedStringSlice>& out)
{
    if (kind == NameValueUtil::NameKind::All)
    {
        StringUtil::appendSplit(UnownedStringSlice(names), ',', out);
    }
    else
    {
        out.add(StringUtil::getAtInSplit(UnownedStringSlice(names), ',', 0));
    }
}

/* static */ void NameValueUtil::appendNames(
    NameKind kind,
    const ConstArrayView<NamesValue>& opts,
    List<UnownedStringSlice>& out)
{
    for (auto& opt : opts)
    {
        _appendNames(kind, opt.names, out);
    }
}

/* static */ void NameValueUtil::appendNames(
    NameKind kind,
    const ConstArrayView<NamesDescriptionValue>& opts,
    List<UnownedStringSlice>& out)
{
    for (auto& opt : opts)
    {
        _appendNames(kind, opt.names, out);
    }
}

/* static */ List<UnownedStringSlice> NameValueUtil::getNames(
    NameKind kind,
    const ConstArrayView<NameValue>& opts)
{
    List<UnownedStringSlice> names;
    appendNames(kind, opts, names);
    return names;
}

/* static */ List<UnownedStringSlice> NameValueUtil::getNames(
    NameKind kind,
    const ConstArrayView<NamesValue>& opts)
{
    List<UnownedStringSlice> names;
    appendNames(kind, opts, names);
    return names;
}

/* static */ List<UnownedStringSlice> NameValueUtil::getNames(
    NameKind kind,
    const ConstArrayView<NamesDescriptionValue>& opts)
{
    List<UnownedStringSlice> names;
    appendNames(kind, opts, names);
    return names;
}

} // namespace Slang
