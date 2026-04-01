#ifndef SLANG_CORE_COMMAND_OPTIONS_H
#define SLANG_CORE_COMMAND_OPTIONS_H

#include "slang-basic.h"
#include "slang-name-value.h"
#include "slang-string-slice-pool.h"

namespace Slang
{

/* For convenience we encode within "names" flags.
"-D..." means that -D *must* be followed by the value
"-D?..." means that -D *can* be a prefix, or it might be followed with the arg
*/

struct CommandOptions
{
    typedef uint32_t Flags;

    typedef int32_t UserIndex;
    enum class UserValue : UserIndex;
    static const UserValue kInvalidUserValue = UserValue(0x80000000);

    enum class LookupKind : int32_t
    {
        Category = -2, ///< Lookup a category name
        Option = -1,   ///< Lookup an option name (all options use the same lookup index even if in
                       ///< different categories)
        Base = 0,      ///< Lookup via category index
    };

    /// A key type that uses the combination of the lookup kind and a name index.
    /// Maps to a target index that could be a category or an option index.
    struct NameKey
    {
        typedef NameKey ThisType;

        SLANG_FORCE_INLINE bool operator==(const ThisType& rhs) const
        {
            return kind == rhs.kind && nameIndex == rhs.nameIndex;
        }
        SLANG_FORCE_INLINE bool operator!=(const ThisType& rhs) const { return !(*this == rhs); }
        HashCode getHashCode() const
        {
            return combineHash(Slang::getHashCode(kind), Slang::getHashCode(nameIndex));
        }

        LookupKind kind; ///< The kind of lookup
        Index nameIndex; ///< The name index in the pool
    };

    enum class CategoryKind
    {
        Option, ///< Command line option (like "-D")
        Value,  ///< One of a set of values (such as an enum or some other kind of list of values)
    };

    struct ValuePair
    {
        const char* name;
        const char* description;
    };

    struct Category
    {
        UserValue userValue = kInvalidUserValue;

        CategoryKind kind;
        UnownedStringSlice name;
        UnownedStringSlice description;

        // Holds the span that defines all of the options associated with the category
        Index optionStartIndex = 0;
        Index optionEndIndex = 0;
    };

    struct Flag
    {
        enum Enum : Flags
        {
            CanPrefix = 0x1, /// Allows -Dfsggf or -D fdsfsd
            IsPrefix = 0x2,  /// Is an option that can only be a prefix
        };
    };

    struct Option
    {
        UnownedStringSlice names; ///< Comma delimited list of names, first name is the default
        UnownedStringSlice usage; ///< Describes usage, can be empty
        UnownedStringSlice description; ///< A description of usage

        UserValue userValue = kInvalidUserValue;

        Index categoryIndex = -1; ///< Category this option belongs to
        Flags flags = 0;          ///< Flags about this option
    };

    /// Get the first name
    UnownedStringSlice getFirstNameForOption(Index optionIndex);
    /// Get the first name for the category
    UnownedStringSlice getFirstNameForCategory(Index categoryIndex);

    /// Get a name key for an opton
    NameKey getNameKeyForOption(Index optionIndex);
    /// Get a name key for a category
    NameKey getNameKeyForCategory(Index optionIndex);

    /// Add a category
    Index addCategory(
        CategoryKind kind,
        const char* name,
        const char* description,
        UserValue userValue = kInvalidUserValue);
    /// Use an already known category. It's an error if the category isn't found
    void setCategory(const char* name);

    void add(
        const char* name,
        const char* usage,
        const char* description,
        UserValue userValue = kInvalidUserValue);
    void add(
        const UnownedStringSlice* names,
        Count namesCount,
        const char* usage,
        const char* description,
        UserValue userValue = kInvalidUserValue,
        Flags flags = 0);

    void addValue(const UnownedStringSlice& name, UserValue userValue = kInvalidUserValue);
    void addValue(
        const UnownedStringSlice& name,
        const UnownedStringSlice& description,
        UserValue userValue = kInvalidUserValue);
    void addValue(
        const char* name,
        const char* description,
        UserValue userValue = kInvalidUserValue);
    void addValue(const char* name, UserValue userValue = kInvalidUserValue);
    void addValue(
        const UnownedStringSlice* names,
        Count namesCount,
        UserValue userValue = kInvalidUserValue);

    /// Add values (without UserValue association)
    void addValues(const ValuePair* pairs, Count pairsCount);

    /// Add values
    void addValues(const ConstArrayView<NameValue>& values);
    void addValues(const ConstArrayView<NamesValue>& values);
    void addValues(const ConstArrayView<NamesDescriptionValue>& values);

    /// Sometimes values are listed with *names* per value. This method will take into account the
    /// aliases
    void addValuesWithAliases(const ConstArrayView<NameValue>& values);

    /// Get the target index based off the name and the kind
    Index findTargetIndexByName(
        LookupKind kind,
        const UnownedStringSlice& name,
        NameKey* outNameKey = nullptr) const;
    /// Given a kind and a user value lookup the target index
    Index findTargetIndexByUserValue(LookupKind kind, UserValue userValue) const;

    /// Finds the category by name or -1 if not found
    Index findCategoryByName(const UnownedStringSlice& name) const
    {
        return findTargetIndexByName(LookupKind::Category, name);
    }
    /// Finds the option index by name or -1 if not found
    Index findOptionByName(const UnownedStringSlice& name) const
    {
        return findTargetIndexByName(LookupKind::Option, name);
    }
    /// Find the option index of a value, using it's category index and the name
    Index findValueByName(Index categoryIndex, const UnownedStringSlice& name) const
    {
        return findTargetIndexByName(LookupKind(categoryIndex), name);
    }

    /// Get the category index from a user value
    Index findCategoryByUserValue(UserValue userValue) const
    {
        return findTargetIndexByUserValue(LookupKind::Category, userValue);
    }
    /// Can only get options
    Index findOptionByUserValue(UserValue userValue) const
    {
        return findTargetIndexByUserValue(LookupKind::Option, userValue);
    }
    /// Get a value associated with a category
    Index findValueByUserValue(Index categoryIndex, UserValue userValue) const
    {
        return findTargetIndexByUserValue(LookupKind(categoryIndex), userValue);
    }

    /// Given a category user value, find the associated name
    /// Returns -1 if not found
    Index findOptionByCategoryUserValue(UserValue categoryUserValue, const UnownedStringSlice& name)
        const;

    /// Find a category by case insensitive name. Returns -1 if not found
    Index findCategoryByCaseInsensitiveName(const UnownedStringSlice& slice) const;

    /// Given a category index returns all the options associated.
    ConstArrayView<Option> getOptionsForCategory(Index categoryIndex) const;

    /// Get the categories
    const List<Category>& getCategories() const { return m_categories; }

    /// Get all the options
    const List<Option>& getOptions() const { return m_options; }

    /// Get the option at the specified index
    const Option& getOptionAt(Index index) const { return m_options[index]; }

    /// Find all of the categories in the usage slice
    void findCategoryIndicesFromUsage(
        const UnownedStringSlice& usageSlice,
        List<Index>& outCategories) const;

    /// Splits usage into category slices
    void splitUsage(const UnownedStringSlice& usageSlice, List<UnownedStringSlice>& outSlices)
        const;

    /// Get all the option names associated with a category index
    void getCategoryOptionNames(Index categoryIndex, List<UnownedStringSlice>& outNames) const;
    void appendCategoryOptionNames(Index categoryIndex, List<UnownedStringSlice>& outNames) const;

    /// Set up a lookup kind from a category index
    static LookupKind makeLookupKind(Index categoryIndex) { return LookupKind(categoryIndex); }

    /// Returns true, if all values from [start, end) are found for the kind
    bool hasContiguousUserValueRange(LookupKind kind, UserValue start, UserValue nonInclEnd) const;

    /// Returns the number of options in the range
    Count getOptionCountInRange(Index categoryIndex, UserValue start, UserValue nonInclEnd) const;
    Count getOptionCountInRange(LookupKind kind, UserValue start, UserValue nonInclEnd) const;


    /// Ctor
    CommandOptions()
        : m_pool(StringSlicePool::Style::Default), m_arena(1024 * 2)
    {
    }

protected:
    /// Returns name in the m_optionPool or -1 on error
    SlangResult _addOptionName(const UnownedStringSlice& name, Flags flags, Index targetIndex);
    SlangResult _addValueName(
        const UnownedStringSlice& name,
        Index categoryIndex,
        Index targetIndex);
    SlangResult _addName(LookupKind kind, const UnownedStringSlice& name, Index targetIndex);

    SlangResult _addUserValue(LookupKind kind, UserValue userValue, Index targetIndex);

    Index _addOption(const UnownedStringSlice& name, const Option& inOption);
    Index _addOption(const UnownedStringSlice* names, Count namesCount, const Option& option);

    Index _addValue(const UnownedStringSlice& name, const Option& inOption);

    UnownedStringSlice _addString(const char* text);
    UnownedStringSlice _addString(const UnownedStringSlice& slice);

    Index _findTargetIndexByName(
        LookupKind kind,
        const UnownedStringSlice& name,
        NameKey* outNameKey) const;

    struct UserValueKey
    {
        typedef UserValueKey ThisType;

        SLANG_FORCE_INLINE bool operator==(const ThisType& rhs) const
        {
            return kind == rhs.kind && userValue == rhs.userValue;
        }
        SLANG_FORCE_INLINE bool operator!=(const ThisType& rhs) const { return !(*this == rhs); }
        HashCode getHashCode() const
        {
            return combineHash(Slang::getHashCode(kind), Slang::getHashCode(userValue));
        }

        LookupKind kind;     ///< The kind of lookup
        UserValue userValue; ///< The user value
    };

    Index m_currentCategoryIndex = -1;

    List<Category> m_categories;

    // Holds a bit for all valid prefix sizes. Max prefix size is therefore 32 chars
    uint32_t m_prefixSizes = 0;

    List<Option> m_options; ///< All of the entries describing each of the options
    StringSlicePool m_pool; ///< Only holds options, and handle therefore matches up to m_entries
    Dictionary<NameKey, Index> m_nameMap;           ///< Maps a name to an option index
    Dictionary<UserValueKey, Index> m_userValueMap; ///< Maps a user value (for a kind) to an index

    MemoryArena m_arena; ///< For other misc storage
};

} // namespace Slang

#endif
