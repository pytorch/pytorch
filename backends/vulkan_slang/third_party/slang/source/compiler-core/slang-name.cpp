// slang-name.cpp
#include "slang-name.h"

namespace Slang
{

String getText(Name* name)
{
    if (!name)
        return String();
    return name->text;
}

UnownedStringSlice getUnownedStringSliceText(Name* name)
{
    return name ? name->text.getUnownedSlice() : UnownedStringSlice();
}

const char* getCstr(Name* name)
{
    return name ? name->text.getBuffer() : nullptr;
}

Name* NamePool::getName(UnownedStringSlice text)
{
    RefPtr<Name> name;
    if (rootPool->names.tryGetValue(text, name))
        return name;

    name = new Name();
    name->text = text;
    rootPool->names.add(text, name);
    return name;
}

Name* NamePool::getName(String const& text)
{
    return getName(text.getUnownedSlice());
}

Name* NamePool::tryGetName(String const& text)
{
    RefPtr<Name> name;
    if (rootPool->names.tryGetValue(text, name))
        return name;
    return nullptr;
}

} // namespace Slang
