#include "identifier-lookup.h"

namespace CppParse
{
using namespace Slang;

/* static */ const IdentifierFlags
    IdentifierLookup::kIdentifierFlags[Index(IdentifierStyle::CountOf)] = {
        0,                       /// None
        0,                       /// Identifier
        0,                       /// Declare type
        0,                       /// Type set
        IdentifierFlag::Keyword, /// TypeModifier
        IdentifierFlag::Keyword, /// Keyword

        IdentifierFlag::Keyword | IdentifierFlag::StartScope | IdentifierFlag::ClassLike, /// Class
        IdentifierFlag::Keyword | IdentifierFlag::StartScope | IdentifierFlag::ClassLike, /// Struct
        IdentifierFlag::Keyword | IdentifierFlag::StartScope, /// Namespace
        IdentifierFlag::Keyword | IdentifierFlag::StartScope, /// Enum

        IdentifierFlag::Keyword, /// Typedef

        IdentifierFlag::Keyword,    /// Access
        IdentifierFlag::Reflection, /// Reflected
        IdentifierFlag::Reflection, /// Unreflected

        IdentifierFlag::Keyword, /// virtual
        0,                       /// Calling convention
        IdentifierFlag::Keyword, /// template
        IdentifierFlag::Keyword, /// static

        IdentifierFlag::Keyword, /// unsigned/signed

        IdentifierFlag::Keyword, /// extern

        0, /// Callable misc
        0, /// IntegerType int, short, char, long

        IdentifierFlag::Keyword, /// default
};

void IdentifierLookup::set(const UnownedStringSlice& name, IdentifierStyle style)
{
    StringSlicePool::Handle handle;
    if (m_pool.findOrAdd(name, handle))
    {
        // Add the extra flags
        m_styles[Index(handle)] = style;
    }
    else
    {
        Index index = Index(handle);
        SLANG_ASSERT(index == m_styles.getCount());
        m_styles.add(style);
    }
}

void IdentifierLookup::set(const char* const* names, size_t namesCount, IdentifierStyle style)
{
    for (size_t i = 0; i < namesCount; ++i)
    {
        set(UnownedStringSlice(names[i]), style);
    }
}

void IdentifierLookup::set(const Pair* pairs, Index pairsCount)
{
    for (Index i = 0; i < pairsCount; ++i)
    {
        const auto& pair = pairs[i];
        set(UnownedStringSlice(pair.name), pair.style);
    }
}

void IdentifierLookup::initDefault(const UnownedStringSlice& markPrefix)
{
    reset();

    // Some keywords
    {
        const char* names[] = {
            "continue",
            "if",
            "case",
            "break",
            "catch",
            "delete",
            "do",
            "else",
            "for",
            "new",
            "goto",
            "return",
            "switch",
            "throw",
            "using",
            "while",
            "operator",
            "explicit"};
        set(names, SLANG_COUNT_OF(names), IdentifierStyle::Keyword);
    }

    // Type modifier keywords
    {
        const char* names[] = {"const", "volatile"};
        set(names, SLANG_COUNT_OF(names), IdentifierStyle::TypeModifier);
    }

    // Special markers
    {
        const char* names[] = {"PRE_DECLARE", "TYPE_SET", "REFLECTED", "UNREFLECTED"};
        const IdentifierStyle styles[] = {
            IdentifierStyle::PreDeclare,
            IdentifierStyle::TypeSet,
            IdentifierStyle::Reflected,
            IdentifierStyle::Unreflected};
        SLANG_COMPILE_TIME_ASSERT(SLANG_COUNT_OF(names) == SLANG_COUNT_OF(styles));

        StringBuilder buf;
        for (Index i = 0; i < SLANG_COUNT_OF(names); ++i)
        {
            buf.clear();
            buf << markPrefix << names[i];
            set(buf.getUnownedSlice(), styles[i]);
        }
    }

    {
        set("virtual", IdentifierStyle::Virtual);

        set("template", IdentifierStyle::Template);
        set("static", IdentifierStyle::Static);
        set("extern", IdentifierStyle::Extern);
        set("default", IdentifierStyle::Default);
    }

    {
        const char* names[] = {"char", "short", "int", "long"};
        set(names, SLANG_COUNT_OF(names), IdentifierStyle::IntegerType);
    }

    {
        const char* names[] = {"SLANG_MCALL"};
        set(names, SLANG_COUNT_OF(names), IdentifierStyle::CallingConvention);
    }

    {
        const char* names[] = {"SLANG_NO_THROW", "inline"};
        set(names, SLANG_COUNT_OF(names), IdentifierStyle::CallableMisc);
    }

    // Keywords which introduce types/scopes
    {
        const Pair pairs[] = {
            {"struct", IdentifierStyle::Struct},
            {"class", IdentifierStyle::Class},
            {"namespace", IdentifierStyle::Namespace},
            {"enum", IdentifierStyle::Enum},
            {"typedef", IdentifierStyle::TypeDef},
        };

        set(pairs, SLANG_COUNT_OF(pairs));
    }

    // Keywords that control access
    {
        const char* names[] = {"private", "protected", "public"};
        set(names, SLANG_COUNT_OF(names), IdentifierStyle::Access);
    }
    {
        const char* names[] = {"signed", "unsigned"};

        set(names, SLANG_COUNT_OF(names), IdentifierStyle::IntegerModifier);
    }
}

} // namespace CppParse
