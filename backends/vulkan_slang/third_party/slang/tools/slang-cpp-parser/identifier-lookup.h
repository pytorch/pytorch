#pragma once

#include "diagnostics.h"

namespace CppParse
{
using namespace Slang;

enum class IdentifierStyle
{
    None, ///< It's not an identifier

    Identifier, ///< Just an identifier

    PreDeclare, ///< Declare a type (not visible in C++ code)
    TypeSet,    ///< TypeSet

    TypeModifier, ///< const, volatile etc
    Keyword,      ///< A keyword C/C++ keyword that is not another type

    Class,     ///< class
    Struct,    ///< struct
    Namespace, ///< namespace
    Enum,      ///< enum

    TypeDef, ///< typedef

    Access, ///< public, protected, private

    Reflected,
    Unreflected,

    CallingConvention, ///< Used on a method
    Virtual,           ///<

    Template,

    Static,

    IntegerModifier,

    Extern,

    CallableMisc, ///< For SLANG_NO_THROW etc

    IntegerType, ///< Built in integer type

    Default, /// default

    CountOf,
};

typedef uint32_t IdentifierFlags;
struct IdentifierFlag
{
    enum Enum : IdentifierFlags
    {
        StartScope = 0x1, ///< namespace, struct or class
        ClassLike = 0x2,  ///< Struct or class
        Keyword = 0x4,
        Reflection = 0x8,
    };
};


class IdentifierLookup
{
public:
    struct Pair
    {
        const char* name;
        IdentifierStyle style;
    };

    IdentifierStyle get(const UnownedStringSlice& slice) const
    {
        Index index = m_pool.findIndex(slice);
        return (index >= 0) ? m_styles[index] : IdentifierStyle::None;
    }

    void set(const char* name, IdentifierStyle style) { set(UnownedStringSlice(name), style); }

    void set(const UnownedStringSlice& name, IdentifierStyle style);

    void set(const char* const* names, size_t namesCount, IdentifierStyle style);

    void set(const Pair* pairs, Index pairsCount);

    void reset()
    {
        m_styles.clear();
        m_pool.clear();
    }

    void initDefault(const UnownedStringSlice& markPrefix);

    IdentifierLookup()
        : m_pool(StringSlicePool::Style::Empty)
    {
        SLANG_ASSERT(m_pool.getSlicesCount() == 0);
    }

    static const IdentifierFlags kIdentifierFlags[Index(IdentifierStyle::CountOf)];

protected:
    List<IdentifierStyle> m_styles;
    StringSlicePool m_pool;
};


SLANG_FORCE_INLINE IdentifierFlags getFlags(IdentifierStyle style)
{
    return IdentifierLookup::kIdentifierFlags[Index(style)];
}

SLANG_FORCE_INLINE bool hasFlag(IdentifierStyle style, IdentifierFlag::Enum flag)
{
    return (getFlags(style) & flag) != 0;
}

} // namespace CppParse
