// slang-command-options.cpp

#include "slang-command-options.h"

#include "slang-byte-encode-util.h"
#include "slang-char-util.h"
#include "slang-string-util.h"

namespace Slang
{

UnownedStringSlice CommandOptions::getFirstNameForOption(Index optionIndex)
{
    const auto& opt = m_options[optionIndex];
    return StringUtil::getAtInSplit(opt.names, ',', 0);
}

UnownedStringSlice CommandOptions::getFirstNameForCategory(Index categoryIndex)
{
    const auto& cat = m_categories[categoryIndex];
    return cat.name;
}

CommandOptions::NameKey CommandOptions::getNameKeyForOption(Index optionIndex)
{
    const auto& opt = m_options[optionIndex];
    const auto& cat = m_categories[opt.categoryIndex];
    NameKey key;
    key.nameIndex = m_pool.findIndex(getFirstNameForOption(optionIndex));
    key.kind =
        (cat.kind == CategoryKind::Option) ? LookupKind::Option : makeLookupKind(opt.categoryIndex);
    return key;
}

CommandOptions::NameKey CommandOptions::getNameKeyForCategory(Index categoryIndex)
{
    NameKey key;
    key.nameIndex = m_pool.findIndex(getFirstNameForCategory(categoryIndex));
    key.kind = LookupKind::Category;
    return key;
}

SlangResult CommandOptions::_addName(
    LookupKind kind,
    const UnownedStringSlice& name,
    Index targetIndex)
{
    NameKey nameKey;
    nameKey.kind = kind;
    nameKey.nameIndex = (Index)m_pool.add(name);

    if (m_nameMap.tryGetValueOrAdd(nameKey, targetIndex))
    {
        SLANG_ASSERT(!"Option is already added!");
        return SLANG_FAIL;
    }
    return SLANG_OK;
}

SlangResult CommandOptions::_addOptionName(
    const UnownedStringSlice& name,
    Flags flags,
    Index targetIndex)
{
    SLANG_RETURN_ON_FAIL(_addName(LookupKind::Option, name, targetIndex));

    // Add to prefix flags
    if (flags & (Flag::CanPrefix | Flag::IsPrefix))
    {
        const auto length = name.getLength();
        SLANG_ASSERT(length < 32);
        m_prefixSizes |= uint32_t(1) << length;
    }

    return SLANG_OK;
}

SlangResult CommandOptions::_addValueName(
    const UnownedStringSlice& name,
    Index categoryIndex,
    Index optionIndex)
{
    return _addName(LookupKind(categoryIndex), name, optionIndex);
}

SlangResult CommandOptions::_addUserValue(LookupKind kind, UserValue userValue, Index targetIndex)
{
    // If it's invalid we don't need to add it
    if (userValue == kInvalidUserValue)
    {
        return SLANG_OK;
    }

    UserValueKey userValueKey;
    userValueKey.kind = kind;
    userValueKey.userValue = userValue;

    if (m_userValueMap.tryGetValueOrAdd(userValueKey, targetIndex))
    {
        SLANG_ASSERT(!"UserValue is already used for this kind!");
        return SLANG_FAIL;
    }
    return SLANG_OK;
}

UnownedStringSlice CommandOptions::_addString(const char* text)
{
    if (text == nullptr)
    {
        return UnownedStringSlice();
    }
    return _addString(UnownedStringSlice(text));
}

UnownedStringSlice CommandOptions::_addString(const UnownedStringSlice& slice)
{
    const auto length = slice.getLength();
    const char* dst = m_arena.allocateString(slice.begin(), length);
    return UnownedStringSlice(dst, length);
}

Index CommandOptions::_addOption(const UnownedStringSlice& name, const Option& inOption)
{
    if (name.indexOf(',') < 0)
    {
        return _addOption(&name, 1, inOption);
    }
    else
    {
        List<UnownedStringSlice> names;
        StringUtil::split(name, ',', names);
        return _addOption(names.getBuffer(), names.getCount(), inOption);
    }
}

Index CommandOptions::_addOption(
    const UnownedStringSlice* names,
    Count namesCount,
    const Option& inOption)
{
    SLANG_ASSERT(namesCount > 0);
    SLANG_ASSERT(inOption.categoryIndex >= 0);

    if (namesCount <= 0 || inOption.categoryIndex < 0)
    {
        return -1;
    }

    auto& cat = m_categories[inOption.categoryIndex];

    // If there are already options associated with this category, we have to be in the run of the
    // last ones added
    if (cat.optionStartIndex != cat.optionEndIndex)
    {
        // If we aren't at the end then this is an error
        if (cat.optionEndIndex != m_options.getCount())
        {
            return -1;
        }
    }
    else
    {
        // Move to the end of the option list
        cat.optionStartIndex = m_options.getCount();
        cat.optionEndIndex = cat.optionStartIndex;
    }

    Option option(inOption);

    const Index optionIndex = m_options.getCount();

    if (cat.kind == CategoryKind::Option)
    {
        for (Index i = 0; i < namesCount; ++i)
        {
            if (SLANG_FAILED(_addOptionName(names[i], inOption.flags, optionIndex)))
            {
                return -1;
            }
        }
        if (SLANG_FAILED(_addUserValue(LookupKind::Option, inOption.userValue, optionIndex)))
        {
            return -1;
        }
    }
    else
    {
        for (Index i = 0; i < namesCount; ++i)
        {
            _addValueName(names[i], inOption.categoryIndex, optionIndex);
        }
        if (SLANG_FAILED(
                _addUserValue(LookupKind(inOption.categoryIndex), inOption.userValue, optionIndex)))
        {
            return -1;
        }
    }

    if (namesCount == 1)
    {
        // We already have storage on the slice
        option.names = m_pool.addAndGetSlice(names[0]);
    }
    else
    {
        // Put all of the names in the list
        StringBuilder buf;
        StringUtil::join(names, namesCount, ',', buf);
        // Allocate storage no in the pool
        option.names = _addString(buf.getUnownedSlice());
    }

    m_options.add(option);

    // Set the end index
    cat.optionEndIndex = optionIndex + 1;

    return optionIndex;
}

static void _handlePostFix(UnownedStringSlice& ioSlice, CommandOptions::Flags& ioFlags)
{
    if (ioSlice.endsWith(toSlice("...")))
    {
        if (ioSlice.endsWith(toSlice("?...")))
        {
            ioFlags |= CommandOptions::Flag::CanPrefix;
            ioSlice = ioSlice.head(ioSlice.getLength() - 4);
        }
        else
        {
            ioFlags |= CommandOptions::Flag::IsPrefix;
            ioSlice = ioSlice.head(ioSlice.getLength() - 3);
        }
    }
}

void CommandOptions::add(
    const char* inName,
    const char* usage,
    const char* description,
    UserValue userValue)
{
    UnownedStringSlice nameSlice(inName);

    Option option;
    option.categoryIndex = m_currentCategoryIndex;
    option.usage = _addString(usage);
    option.description = _addString(UnownedStringSlice(description));
    option.userValue = userValue;
    option.flags = 0;

    if (nameSlice.indexOf(',') >= 0)
    {
        List<UnownedStringSlice> names;
        StringUtil::split(nameSlice, ',', names);

        for (auto& name : names)
        {
            _handlePostFix(name, option.flags);
        }

        _addOption(names.getBuffer(), names.getCount(), option);
    }
    else
    {
        _handlePostFix(nameSlice, option.flags);

        _addOption(&nameSlice, 1, option);
    }
}

void CommandOptions::add(
    const UnownedStringSlice* names,
    Count namesCount,
    const char* usage,
    const char* description,
    UserValue userValue,
    Flags flags)
{
    Option option;
    option.categoryIndex = m_currentCategoryIndex;
    option.usage = _addString(usage);
    option.description = _addString(UnownedStringSlice(description));
    option.flags = flags;
    option.userValue = userValue;

    _addOption(names, namesCount, option);
}

Index CommandOptions::_addValue(const UnownedStringSlice& name, const Option& inOption)
{
    SLANG_ASSERT(m_currentCategoryIndex >= 0);
    SLANG_ASSERT(m_categories[m_currentCategoryIndex].kind == CategoryKind::Value);

    return _addOption(name, inOption);
}

void CommandOptions::addValues(const ValuePair* pairs, Count pairsCount)
{
    for (auto& pair : makeConstArrayView(pairs, pairsCount))
    {
        addValue(pair.name, pair.description);
    }
}

void CommandOptions::addValues(const ConstArrayView<NameValue>& values)
{
    for (const auto& value : values)
    {
        addValue(value.name, UserValue(value.value));
    }
}

void CommandOptions::addValues(const ConstArrayView<NamesValue>& values)
{
    for (const auto& value : values)
    {
        addValue(value.names, UserValue(value.value));
    }
}

void CommandOptions::addValues(const ConstArrayView<NamesDescriptionValue>& values)
{
    for (const auto& value : values)
    {
        addValue(value.names, value.description, UserValue(value.value));
    }
}

void CommandOptions::addValuesWithAliases(const ConstArrayView<NameValue>& inValues)
{
    List<NameValue> values;
    values.addRange(inValues.getBuffer(), inValues.getCount());

    values.sort([](const NameValue& a, const NameValue& b) -> bool { return a.value < b.value; });

    List<UnownedStringSlice> names;

    const Count count = values.getCount();
    Index i = 0;
    while (i < count)
    {
        names.clear();

        const auto value = values[i].value;
        names.add(UnownedStringSlice(values[i++].name));

        for (; i < count && values[i].value == value; ++i)
        {
            names.add(UnownedStringSlice(values[i].name));
        }

        addValue(names.getBuffer(), names.getCount(), UserValue(value));
    }
}

void CommandOptions::addValue(const UnownedStringSlice& name, UserValue userValue)
{
    Option option;
    option.categoryIndex = m_currentCategoryIndex;
    option.userValue = userValue;
    _addValue(name, option);
}

void CommandOptions::addValue(
    const UnownedStringSlice& name,
    const UnownedStringSlice& description,
    UserValue userValue)
{
    Option option;
    option.categoryIndex = m_currentCategoryIndex;
    option.description = _addString(description);
    option.userValue = userValue;
    _addValue(name, option);
}

void CommandOptions::addValue(
    const UnownedStringSlice* names,
    Count namesCount,
    UserValue userValue)
{
    Option option;
    option.categoryIndex = m_currentCategoryIndex;
    option.userValue = userValue;

    SLANG_ASSERT(m_currentCategoryIndex >= 0);
    SLANG_ASSERT(m_categories[m_currentCategoryIndex].kind == CategoryKind::Value);

    _addOption(names, namesCount, option);
}

void CommandOptions::addValue(const char* inName, const char* description, UserValue userValue)
{
    const UnownedStringSlice name(inName);

    if (description)
    {
        addValue(name, UnownedStringSlice(description), userValue);
    }
    else
    {
        addValue(name, userValue);
    }
}

void CommandOptions::addValue(const char* name, UserValue userValue)
{
    addValue(UnownedStringSlice(name), userValue);
}

Index CommandOptions::addCategory(
    CategoryKind kind,
    const char* name,
    const char* description,
    UserValue userValue)
{
    const UnownedStringSlice nameSlice(name);

    const auto categoryIndex = m_categories.getCount();

    if (SLANG_FAILED(_addName(LookupKind::Category, nameSlice, categoryIndex)))
    {
        return -1;
    }

    if (userValue != kInvalidUserValue)
    {
        _addUserValue(LookupKind::Category, userValue, categoryIndex);
    }

    Category cat;
    cat.kind = kind;
    cat.name = _addString(nameSlice);
    cat.description = _addString(description);
    cat.userValue = userValue;

    m_currentCategoryIndex = categoryIndex;

    m_categories.add(cat);

    return categoryIndex;
}

void CommandOptions::setCategory(const char* name)
{
    const UnownedStringSlice nameSlice(name);

    for (Index i = 0; i < m_categories.getCount(); ++i)
    {
        auto& cat = m_categories[i];
        if (cat.name == nameSlice)
        {
            m_currentCategoryIndex = i;
            return;
        }
    }

    SLANG_ASSERT(!"Category not found");

    m_currentCategoryIndex = -1;
}

Index CommandOptions::findTargetIndexByName(
    LookupKind kind,
    const UnownedStringSlice& name,
    NameKey* outNameKey) const
{
    // Look up directly
    {
        auto index = _findTargetIndexByName(kind, name, outNameKey);
        if (index >= 0)
        {
            return index;
        }
    }

    // Special case options, which can have prefix styles
    if (kind == LookupKind::Option)
    {
        auto prefixSizes = m_prefixSizes;

        while (prefixSizes)
        {
            auto prefixSize = ByteEncodeUtil::calcMsb32(prefixSizes);

            if (prefixSize < name.getLength())
            {
                // Look it up
                const auto index = _findTargetIndexByName(kind, name.head(prefixSize), outNameKey);
                if (index >= 0)
                {
                    auto& option = m_options[index];

                    // If the option accepts prefixes, we return the index
                    if (option.flags & (Flag::CanPrefix | Flag::IsPrefix))
                    {
                        return index;
                    }
                }
            }

            // Remove the bit
            prefixSizes &= ~(uint32_t(1) << prefixSize);
        }
    }

    // Was not found
    return -1;
}

Index CommandOptions::_findTargetIndexByName(
    LookupKind kind,
    const UnownedStringSlice& name,
    NameKey* outNameKey) const
{
    const auto nameIndex = m_pool.findIndex(name);
    // If the name isn't in the pool then there isn't a category with this name
    if (nameIndex < 0)
    {
        return -1;
    }

    NameKey key;
    key.kind = kind;
    key.nameIndex = nameIndex;

    if (auto ptr = m_nameMap.tryGetValue(key))
    {
        if (outNameKey)
        {
            *outNameKey = key;
        }
        return *ptr;
    }

    return -1;
}

Index CommandOptions::findTargetIndexByUserValue(LookupKind kind, UserValue userValue) const
{
    UserValueKey key;
    key.kind = kind;
    key.userValue = userValue;

    if (auto ptr = m_userValueMap.tryGetValue(key))
    {
        return *ptr;
    }

    return -1;
}

Index CommandOptions::findCategoryByCaseInsensitiveName(const UnownedStringSlice& slice) const
{
    const Count count = m_categories.getCount();
    for (Index i = 0; i < count; ++i)
    {
        const auto& cat = m_categories[i];

        if (cat.name.caseInsensitiveEquals(slice))
        {
            return i;
        }
    }
    return -1;
}

Index CommandOptions::findOptionByCategoryUserValue(
    UserValue categoryUserValue,
    const UnownedStringSlice& name) const
{
    Index categoryIndex = findTargetIndexByUserValue(LookupKind::Category, categoryUserValue);
    if (categoryIndex < 0)
    {
        return -1;
    }

    return findValueByName(categoryIndex, name);
}

ConstArrayView<CommandOptions::Option> CommandOptions::getOptionsForCategory(
    Index categoryIndex) const
{
    const auto& cat = m_categories[categoryIndex];
    return makeConstArrayView(
        m_options.getBuffer() + cat.optionStartIndex,
        cat.optionEndIndex - cat.optionStartIndex);
}


void CommandOptions::appendCategoryOptionNames(
    Index categoryIndex,
    List<UnownedStringSlice>& outNames) const
{
    for (const auto& option : getOptionsForCategory(categoryIndex))
    {
        StringUtil::appendSplit(option.names, ',', outNames);
    }
}

void CommandOptions::getCategoryOptionNames(Index categoryIndex, List<UnownedStringSlice>& outNames)
    const
{
    outNames.clear();
    appendCategoryOptionNames(categoryIndex, outNames);
}

void CommandOptions::splitUsage(
    const UnownedStringSlice& usageSlice,
    List<UnownedStringSlice>& outSlices) const
{
    const auto* cur = usageSlice.begin();
    const auto* end = usageSlice.end();

    while (cur < end)
    {
        // Find <
        while (cur < end && *cur != '<')
            cur++;

        // If we found it look for the end
        if (cur < end && *cur == '<')
        {
            ++cur;
            auto start = cur;
            while (cur < end && (CharUtil::isAlphaOrDigit(*cur) || *cur == '-' || *cur == '_') &&
                   *cur != '>')
            {
                cur++;
            }

            // If we hit closing > we want to lookup
            if (cur < end && *cur == '>')
            {
                const UnownedStringSlice categoryName(start, cur);

                Index categoryIndex = findCategoryByName(categoryName);
                if (categoryIndex >= 0)
                {
                    outSlices.add(categoryName);
                }
            }

            cur++;
        }
    }
}


void CommandOptions::findCategoryIndicesFromUsage(
    const UnownedStringSlice& slice,
    List<Index>& outCategories) const
{
    List<UnownedStringSlice> categoryNames;
    splitUsage(slice, categoryNames);

    for (auto name : categoryNames)
    {
        Index categoryIndex = findCategoryByName(name);
        if (categoryIndex >= 0 && outCategories.indexOf(categoryIndex) < 0)
        {
            outCategories.add(categoryIndex);
        }
    }
}

Count CommandOptions::getOptionCountInRange(
    Index categoryIndex,
    UserValue start,
    UserValue nonInclEnd) const
{
    const UserIndex startIndex = UserIndex(start);
    const UserIndex endIndex = UserIndex(nonInclEnd);

    Count count = 0;

    for (auto& opt : getOptionsForCategory(categoryIndex))
    {
        const auto val = opt.userValue;
        if (val == kInvalidUserValue)
        {
            continue;
        }

        const auto valIndex = UserIndex(val);
        count += Index(valIndex >= startIndex && valIndex < endIndex);
    }

    return count;
}

Count CommandOptions::getOptionCountInRange(LookupKind kind, UserValue start, UserValue nonInclEnd)
    const
{
    Index count = 0;

    if (kind == LookupKind::Category)
    {
        const UserIndex startIndex = UserIndex(start);
        const UserIndex endIndex = UserIndex(nonInclEnd);

        for (auto& cat : m_categories)
        {
            if (cat.userValue != kInvalidUserValue)
            {
                const auto valIndex = UserIndex(cat.userValue);
                count += Index(valIndex >= startIndex && valIndex < endIndex);
            }
        }
    }
    if (kind == LookupKind::Option)
    {
        // If we are lookup up options, then we iterate over all option categories
        const auto catCount = m_categories.getCount();
        for (Index categoryIndex = 0; categoryIndex < catCount; ++categoryIndex)
        {
            if (m_categories[categoryIndex].kind == CategoryKind::Option)
            {
                count += getOptionCountInRange(categoryIndex, start, nonInclEnd);
            }
        }
    }
    else if (Index(kind) >= 0)
    {
        // It's a regular category
        count = getOptionCountInRange(Index(kind), start, nonInclEnd);
    }

    return count;
}


bool CommandOptions::hasContiguousUserValueRange(
    LookupKind kind,
    UserValue start,
    UserValue nonInclEnd) const
{
    const Count rangeCount = Count(nonInclEnd) - Count(start);
    SLANG_ASSERT(rangeCount >= 0);

    if (rangeCount <= 0)
    {
        return true;
    }

    const Count count = getOptionCountInRange(kind, start, nonInclEnd);
    return rangeCount == count;
}

} // namespace Slang
