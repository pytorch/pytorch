#ifndef SLANG_CORE_STRING_UTIL_H
#define SLANG_CORE_STRING_UTIL_H

#include "slang-com-helper.h"
#include "slang-com-ptr.h"
#include "slang-list.h"
#include "slang-string.h"

#include <stdarg.h>

namespace Slang
{

struct StringUtil
{
    typedef bool (*EqualFn)(const UnownedStringSlice& a, const UnownedStringSlice& b);

    /// True if the splits of a and b (via splitChar) are all equal as compared with the equalFn
    /// function
    static bool areAllEqualWithSplit(
        const UnownedStringSlice& a,
        const UnownedStringSlice& b,
        char splitChar,
        EqualFn equalFn);

    /// True if all slices in match are all equal as compared with the equalFn function
    static bool areAllEqual(
        const List<UnownedStringSlice>& a,
        const List<UnownedStringSlice>& b,
        EqualFn equalFn);

    /// Split in, by specified splitChar into slices out
    /// Slices contents will directly address into in, so contents will only stay valid as long as
    /// in does.
    static void split(
        const UnownedStringSlice& in,
        char splitChar,
        List<UnownedStringSlice>& slicesOut);
    /// Split in by the specified splitSlice
    /// Slices contents will directly address into in, so contents will only stay valid as long as
    /// in does.
    static void split(
        const UnownedStringSlice& in,
        const UnownedStringSlice& splitSlice,
        List<UnownedStringSlice>& slicesOut);

    /// Splits in into outSlices, up to maxSlices. May not consume all of in (for example if it runs
    /// out of space).
    static Index split(
        const UnownedStringSlice& in,
        char splitChar,
        Index maxSlices,
        UnownedStringSlice* outSlices);
    /// Splits into outSlices up to maxSlices. Returns SLANG_OK if of 'in' consumed.
    static SlangResult split(
        const UnownedStringSlice& in,
        char splitChar,
        Index maxSlices,
        UnownedStringSlice* outSlices,
        Index& outSlicesCount);
    /// Splits on white space
    static void splitOnWhitespace(
        const UnownedStringSlice& in,
        List<UnownedStringSlice>& slicesOut);

    /// Split in, by specified splitChar append into slices out
    /// Slices contents will directly address into in, so contents will only stay valid as long as
    /// in does.
    static void appendSplit(
        const UnownedStringSlice& in,
        char splitChar,
        List<UnownedStringSlice>& slicesOut);
    /// Split in by the specified splitSlice
    /// Slices contents will directly address into in, so contents will only stay valid as long as
    /// in does.
    static void appendSplit(
        const UnownedStringSlice& in,
        const UnownedStringSlice& splitSlice,
        List<UnownedStringSlice>& slicesOut);

    /// appends splits on white space
    static void appendSplitOnWhitespace(
        const UnownedStringSlice& in,
        List<UnownedStringSlice>& slicesOut);

    /// Append the joining of in items, separated by 'separator' onto out
    static void join(const List<String>& in, char separator, StringBuilder& out);
    static void join(
        const List<String>& in,
        const UnownedStringSlice& separator,
        StringBuilder& out);

    static void join(
        const UnownedStringSlice* values,
        Index valueCount,
        char separator,
        StringBuilder& out);
    static void join(
        const UnownedStringSlice* values,
        Index valueCount,
        const UnownedStringSlice& separator,
        StringBuilder& out);

    /// Equivalent to doing a split and then finding the index of 'find' on the array
    /// Returns -1 if not found
    static Index indexOfInSplit(
        const UnownedStringSlice& in,
        char splitChar,
        const UnownedStringSlice& find);

    /// Given the entry at the split index specified.
    /// Will return slice with begin() == nullptr if not found or input has begin() == nullptr)
    static UnownedStringSlice getAtInSplit(
        const UnownedStringSlice& in,
        char splitChar,
        Index index);

    /// Returns the size in bytes needed to hold the formatted string using the specified args, NOT
    /// including a terminating 0 NOTE! The caller *should* assume this will consume the va_list
    /// (use va_copy to make a copy to be consumed)
    static size_t calcFormattedSize(const char* format, va_list args);

    /// Calculate the formatted string using the specified args.
    /// NOTE! The caller *should* assume this will consume the va_list
    /// The buffer should be at least calcFormattedSize + 1 bytes. The +1 is needed because a
    /// terminating 0 is written.
    static void calcFormatted(const char* format, va_list args, size_t numChars, char* dst);

    /// Appends formatted string with args into buf
    static void append(const char* format, va_list args, StringBuilder& buf);

    /// Appends the formatted string with specified trailing args
    static void appendFormat(StringBuilder& buf, const char* format, ...);

    /// Create a string from the format string applying args (like sprintf)
    static String makeStringWithFormat(const char* format, ...);

    /// Create a string from the format string and arguments in a buffer.
    static String makeStringWithFormatFromArgArray(
        const char* format,
        ArrayView<const void*> ptrToArgs);

    /// Given a string held in a blob, returns as a String
    /// Returns an empty string if blob is nullptr
    static String getString(ISlangBlob* blob);
    static UnownedStringSlice getSlice(ISlangBlob* blob);

    /// Given a string or slice, replaces all instances of fromChar with toChar
    static String calcCharReplaced(const UnownedStringSlice& slice, char fromChar, char toChar);
    static String calcCharReplaced(const String& string, char fromChar, char toChar);

    /// Replaces all occurrances of subStr with replacement.
    static String replaceAll(
        UnownedStringSlice text,
        UnownedStringSlice subStr,
        UnownedStringSlice replacement);

    /// Create a blob from a string
    static ComPtr<ISlangBlob> createStringBlob(const String& string);

    /// Given input text outputs with standardized line endings. Ie \n\r -> \n
    static void appendStandardLines(const UnownedStringSlice& text, StringBuilder& out);

    /// Extracts a line and stores the remaining text in ioText, and the line in outLine. Returns
    /// true if has a line.
    ///
    /// As well as indicating end of text with the return value, at the end of all the text a
    /// 'special' null UnownedStringSlice with a null 'begin' pointer is also returned as the
    /// outLine. ioText will be modified to contain the remaining text, starting at the beginning of
    /// the next line. As an empty final line is still a line, the special null UnownedStringSlice
    /// is the last value ioText after the last valid line is returned.
    ///
    /// NOTE! That behavior is as if line terminators (like \n) act as separators. Thus input of
    /// "\n" will return *two* lines - an empty line before and then after the \n.
    static bool extractLine(UnownedStringSlice& ioText, UnownedStringSlice& outLine);

    /// Given text, splits into lines stored in outLines. NOTE! That lines is only valid as long as
    /// textIn remains valid
    static void calcLines(const UnownedStringSlice& textIn, List<UnownedStringSlice>& lines);

    /// Given a line that may contain cr/lf, returns the the a slice that doesn't have trailing
    /// cr/lf
    static UnownedStringSlice trimEndOfLine(const UnownedStringSlice& slice);

    /// Equal if the lines are equal (in effect a way to ignore differences in line breaks)
    static bool areLinesEqual(const UnownedStringSlice& a, const UnownedStringSlice& b);

    /// Convert in to int. Returns SLANG_FAIL on error
    static SlangResult parseInt(const UnownedStringSlice& in, Int& outValue);

    /// Convert ioText into double. Returns SLANG_OK on success.
    static SlangResult parseDouble(const UnownedStringSlice& text, double& out);

    /// Convert into int64_t. Returns SLANG_OK on success.
    static SlangResult parseInt64(const UnownedStringSlice& text, int64_t& out);

    /// Parse an integer from text starting at pos until the end or the first non-digit char.
    /// Modifies pos to the position where parsing ends.
    /// Returns parsed integer.
    static int parseIntAndAdvancePos(UnownedStringSlice text, Index& pos);
};

/* A helper class that allows parsing of lines from text with iteration. Uses
 * StringUtil::extractLine for the actual underlying implementation. */
class LineParser
{
public:
    struct Iterator
    {
        const UnownedStringSlice& operator*() const { return m_line; }
        const UnownedStringSlice* operator->() const { return &m_line; }
        Iterator& operator++()
        {
            StringUtil::extractLine(m_remaining, m_line);
            return *this;
        }
        Iterator operator++(int)
        {
            Iterator rs = *this;
            operator++();
            return rs;
        }

        /// Equal if both are at the same m_line address exactly. Handles termination case correctly
        /// where line.begin() == nullptr.
        bool operator==(const Iterator& rhs) const { return m_line.begin() == rhs.m_line.begin(); }
        bool operator!=(const Iterator& rhs) const { return !(*this == rhs); }

        /// Ctor
        Iterator(const UnownedStringSlice& line, const UnownedStringSlice& remaining)
            : m_line(line), m_remaining(remaining)
        {
        }

    protected:
        UnownedStringSlice m_line;
        UnownedStringSlice m_remaining;
    };

    Iterator begin() const
    {
        UnownedStringSlice remaining(m_text), line;
        StringUtil::extractLine(remaining, line);
        return Iterator(line, remaining);
    }
    Iterator end() const
    {
        UnownedStringSlice term(nullptr, nullptr);
        return Iterator(term, term);
    }

    /// Ctor
    LineParser(const UnownedStringSlice& text)
        : m_text(text)
    {
    }

protected:
    UnownedStringSlice m_text;
};

} // namespace Slang

#endif // SLANG_STRING_UTIL_H
