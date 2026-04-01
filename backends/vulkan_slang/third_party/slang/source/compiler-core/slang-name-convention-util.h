#ifndef SLANG_COMPILER_CORE_NAME_CONVENTION_UTIL_H
#define SLANG_COMPILER_CORE_NAME_CONVENTION_UTIL_H

#include "../core/slang-list.h"
#include "../core/slang-string.h"

namespace Slang
{

typedef uint8_t NameConventionBackingType;

enum class NameStyle : NameConventionBackingType
{
    Unknown, /// Unknown style
    Kabab,   /// Words are separated with -. WORDS-ARE-SEPARATED, words-are-separted
    Snake,   /// Words are separated with _. WORDS_ARE_SEPARATED, words_are_separated
    Camel,   /// Words start with a capital. (Upper will make first words character capitalized, aka
             /// PascalCase)
};

struct NameConventionFlag
{
    enum Enum : NameConventionBackingType
    {
        UpperCase = 0x80,
    };
};

struct NameConventionMask
{
    enum Enum : NameConventionBackingType
    {
        Style = 0x7,
    };
};

enum class NameConvention : NameConventionBackingType
{
    Invalid = NameConventionBackingType(NameStyle::Unknown),

    LowerKabab = NameConventionBackingType(NameStyle::Kabab),
    LowerSnake = NameConventionBackingType(NameStyle::Snake),
    LowerCamel = NameConventionBackingType(NameStyle::Camel),

    UpperKabab = NameConventionBackingType(NameStyle::Kabab) | NameConventionFlag::UpperCase,
    UpperSnake = NameConventionBackingType(NameStyle::Snake) | NameConventionFlag::UpperCase,
    UpperCamel = NameConventionBackingType(NameStyle::Camel) | NameConventionFlag::UpperCase,
};

SLANG_FORCE_INLINE NameConvention makeUpper(NameStyle style)
{
    return NameConvention(NameConventionBackingType(style) | NameConventionFlag::UpperCase);
}
SLANG_FORCE_INLINE NameConvention makeLower(NameStyle style)
{
    return NameConvention(style);
}

SLANG_FORCE_INLINE bool isUpper(NameConvention convention)
{
    return (NameConventionBackingType(convention) & NameConventionFlag::UpperCase) != 0;
}
SLANG_FORCE_INLINE bool isLower(NameConvention convention)
{
    return (NameConventionBackingType(convention) & NameConventionFlag::UpperCase) == 0;
}
SLANG_FORCE_INLINE NameStyle getNameStyle(NameConvention convention)
{
    return NameStyle(NameConventionBackingType(convention) & NameConventionMask::Style);
}

/* This utility is to enable easy conversion and interpretation of names that use standard
conventions, typically in programming languages. The conventions are largely how to represent
multiple words together.

Split is used to split up a name into it's constituent 'words' based on a convention.
Join is used to combine words based on a convention/character case

Convert uses split and join to allow easy conversion between conventions.
*/
struct NameConventionUtil
{
    /// Given a slice tries to determine the convention used.
    /// If no separators are found, will assume Camel
    /// Doesn't exhaustively test the string slice, or determine invalid scenarios
    /// Use 'getConvention' to get error checking
    static NameStyle inferStyleFromText(const UnownedStringSlice& slice);

    /// Gets the naming convention based on the slice.
    /// Will return invalid convention if cannot be determined.
    ///
    /// TODO(JS):
    /// Does not handle leading `_` styles: "_a" and "_1" will be invalid.
    /// We may want to make it do so, but requires changes in infer, split and join.
    static NameConvention inferConventionFromText(const UnownedStringSlice& slice);

    /// Given a slice and a naming convention, split into it's constituent parts. If convention
    /// isn't specified, will infer from slice using getConvention.
    static void split(
        NameStyle nameStyle,
        const UnownedStringSlice& slice,
        List<UnownedStringSlice>& out);
    static void split(const UnownedStringSlice& slice, List<UnownedStringSlice>& out);

    /// Given slices, join together with the specified convention into out
    static void join(
        const UnownedStringSlice* slices,
        Index slicesCount,
        NameConvention convention,
        StringBuilder& out);

    /// Join with a join char, and potentially changing case of input slices
    static void join(
        const UnownedStringSlice* slices,
        Index slicesCount,
        NameConvention convention,
        char joinChar,
        StringBuilder& out);

    /// Convert from one convention to another. If fromConvention isn't specified, will infer from
    /// slice using getConvention.
    static void convert(
        NameStyle fromStyle,
        const UnownedStringSlice& slice,
        NameConvention toConvention,
        StringBuilder& out);
    static void convert(
        const UnownedStringSlice& slice,
        NameConvention toConvention,
        StringBuilder& out);
};

} // namespace Slang

#endif // SLANG_COMPILER_CORE_NAME_CONVENTION_UTIL_H
