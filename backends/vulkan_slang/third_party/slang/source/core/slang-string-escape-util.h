#ifndef SLANG_CORE_STRING_ESCAPE_UTIL_H
#define SLANG_CORE_STRING_ESCAPE_UTIL_H

#include "slang-list.h"
#include "slang-string.h"

namespace Slang
{

class StringEscapeHandler
{
public:
    /// True if quoting is needed
    virtual bool isQuotingNeeded(const UnownedStringSlice& slice) = 0;
    /// True if any escaping is needed. If not slice can be used (assuming appropriate quoting) as
    /// is
    virtual bool isEscapingNeeded(const UnownedStringSlice& slice) = 0;
    /// True if we need to unescape
    virtual bool isUnescapingNeeeded(const UnownedStringSlice& slice) = 0;

    /// Takes slice and adds any appropriate escaping (for example C++/C type escaping for special
    /// characters like '\', '"' and if not ascii will write out as hex sequence) Does not append
    /// quotes
    virtual SlangResult appendEscaped(const UnownedStringSlice& slice, StringBuilder& out) = 0;
    /// Given a slice append it unescaped
    /// Does not consume surrounding quotes
    virtual SlangResult appendUnescaped(const UnownedStringSlice& slice, StringBuilder& out) = 0;

    /// Lex quoted text.
    /// The first character of cursor should be the quoteCharacter.
    /// cursor points to the string to be lexed - must typically be 0 terminated.
    /// outCursor on successful lex will be at the next character after was processed.
    virtual SlangResult lexQuoted(const char* cursor, const char** outCursor) = 0;

    SLANG_FORCE_INLINE char getQuoteChar() const { return m_quoteChar; }

    StringEscapeHandler(char quoteChar)
        : m_quoteChar(quoteChar)
    {
    }

protected:
    const char m_quoteChar;
};

/* A set of function that can be used for escaping/unescaping quoting/unquoting strings.

The distinction between 'escaping' and 'quoting' here, is just that escaping is the 'payload' of
quotes. In *principal* the Style can determine different styles of escaping that can be used.
*/
struct StringEscapeUtil
{
    typedef StringEscapeHandler Handler;

    enum class Style
    {
        Cpp,   ///< Cpp style quoting and escape handling
        Space, ///< Applies quotes if there are spaces. Does not escape.
        JSON,  ///< Json encoding
        Slang, ///< Slang style string encoding (For now same as Cpp but that may change in the
               ///< future)
        CountOf,
    };

    /// Given a style returns a handler
    static Handler* getHandler(Style style);

    /// Get without quotes. Will assert if not correctly quoted
    static UnownedStringSlice unquote(char quoteChar, const UnownedStringSlice& slice);
    static UnownedStringSlice unquote(Handler* handler, const UnownedStringSlice& slice)
    {
        return unquote(handler->getQuoteChar(), slice);
    }

    /// True is slice is quoted
    static bool isQuoted(char quoteChar, UnownedStringSlice& slice);
    static bool isQuoted(Handler* handler, UnownedStringSlice& slice)
    {
        return isQuoted(handler->getQuoteChar(), slice);
    }

    /// If quoting is needed appends to out quoted
    static SlangResult appendMaybeQuoted(
        Handler* handler,
        const UnownedStringSlice& slice,
        StringBuilder& out);

    /// If the slice appears to be quoted for the style, unquote it, else just append to out
    static SlangResult appendMaybeUnquoted(
        Handler* handler,
        const UnownedStringSlice& slice,
        StringBuilder& out);

    /// Appends to out slice without quotes
    static SlangResult appendUnquoted(
        Handler* handler,
        const UnownedStringSlice& slice,
        StringBuilder& out);

    /// Append with quotes (even if not needed)
    static SlangResult appendQuoted(
        Handler* handler,
        const UnownedStringSlice& slice,
        StringBuilder& out);


    /// True if requires 'shell-like' unescape. With shell-like, quoting does *not* have to start at
    /// the start of the slice. and there may be multiple quoted section
    static SlangResult isUnescapeShellLikeNeeded(Handler* handler, const UnownedStringSlice& slice);

    /// Shells can have multiple quoted sections. This function makes a string with out quoting
    static SlangResult unescapeShellLike(
        Handler* handler,
        const UnownedStringSlice& slice,
        StringBuilder& out);

    static String escapeString(UnownedStringSlice input, Style style = Style::Slang);
    static String unescapeString(UnownedStringSlice input, Style style = Style::Slang);
};


} // namespace Slang

#endif // SLANG_CORE_STRING_ESCAPE_UTIL_H
