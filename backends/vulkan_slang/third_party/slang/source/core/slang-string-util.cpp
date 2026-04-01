#include "slang-string-util.h"

#include "slang-blob.h"
#include "slang-char-util.h"
#include "slang-text-io.h"

namespace Slang
{

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! StringUtil !!!!!!!!!!!!!!!!!!!!!!!!!!!

/* static */ bool StringUtil::areAllEqual(
    const List<UnownedStringSlice>& a,
    const List<UnownedStringSlice>& b,
    EqualFn equalFn)
{
    if (a.getCount() != b.getCount())
    {
        return false;
    }

    const Index count = a.getCount();
    for (Index i = 0; i < count; ++i)
    {
        if (!equalFn(a[i], b[i]))
        {
            return false;
        }
    }
    return true;
}

/* static */ bool StringUtil::areAllEqualWithSplit(
    const UnownedStringSlice& a,
    const UnownedStringSlice& b,
    char splitChar,
    EqualFn equalFn)
{
    List<UnownedStringSlice> slicesA, slicesB;
    StringUtil::split(a, splitChar, slicesA);
    StringUtil::split(b, splitChar, slicesB);
    return areAllEqual(slicesA, slicesB, equalFn);
}

/* static */ void StringUtil::appendSplitOnWhitespace(
    const UnownedStringSlice& in,
    List<UnownedStringSlice>& outSlices)
{
    const char* start = in.begin();
    const char* end = in.end();

    // Skip any at the start
    while (start < end && CharUtil::isWhitespace(*start))
        start++;

    while (start < end)
    {
        // Find all the non white space in a run
        const char* cur = start;
        while (cur < end && !CharUtil::isWhitespace(*cur))
        {
            cur++;
        }

        // Add to output
        outSlices.add(UnownedStringSlice(start, cur));

        // Find the next start
        start = cur + 1;

        // Skip the split
        while (start < end && CharUtil::isWhitespace(*start))
            start++;
    }
}

/* static */ void StringUtil::appendSplit(
    const UnownedStringSlice& in,
    char splitChar,
    List<UnownedStringSlice>& outSlices)
{
    const char* start = in.begin();
    const char* end = in.end();

    while (start < end)
    {
        // Move cur so it's either at the end or at next split character
        const char* cur = start;
        while (cur < end && *cur != splitChar)
        {
            cur++;
        }

        // Add to output
        outSlices.add(UnownedStringSlice(start, cur));

        // Skip the split character, if at end we are okay anyway
        start = cur + 1;
    }
}

/* static */ void StringUtil::appendSplit(
    const UnownedStringSlice& in,
    const UnownedStringSlice& splitSlice,
    List<UnownedStringSlice>& outSlices)
{
    const Index splitLen = splitSlice.getLength();

    if (splitLen == 1)
    {
        return appendSplit(in, splitSlice[0], outSlices);
    }

    SLANG_ASSERT(splitLen > 0);
    if (splitLen <= 0)
    {
        return;
    }

    const char* start = in.begin();
    const char* end = in.end();

    const char splitChar = splitSlice[0];

    while (start < end)
    {
        // Move cur so it's either at the end or at next splitSlice
        const char* cur = start;
        while (cur < end)
        {
            if (*cur == splitChar &&
                (cur + splitLen <= end && UnownedStringSlice(cur, splitLen) == splitSlice))
            {
                // We hit a split
                break;
            }

            cur++;
        }

        // Add to output
        outSlices.add(UnownedStringSlice(start, cur));

        // Skip the split, if at end we are okay anyway
        start = cur + splitLen;
    }
}

/* static */ void StringUtil::split(
    const UnownedStringSlice& in,
    char splitChar,
    List<UnownedStringSlice>& outSlices)
{
    outSlices.clear();
    appendSplit(in, splitChar, outSlices);
}

/* static */ void StringUtil::split(
    const UnownedStringSlice& in,
    const UnownedStringSlice& splitSlice,
    List<UnownedStringSlice>& outSlices)
{
    outSlices.clear();
    appendSplit(in, splitSlice, outSlices);
}

/* static */ void StringUtil::splitOnWhitespace(
    const UnownedStringSlice& in,
    List<UnownedStringSlice>& outSlices)
{
    outSlices.clear();
    appendSplitOnWhitespace(in, outSlices);
}

/* static */ Index StringUtil::split(
    const UnownedStringSlice& in,
    char splitChar,
    Index maxSlices,
    UnownedStringSlice* outSlices)
{
    Index index = 0;

    const char* start = in.begin();
    const char* end = in.end();

    while (start < end && index < maxSlices)
    {
        // Move cur so it's either at the end or at next split character
        const char* cur = start;
        while (cur < end && *cur != splitChar)
        {
            cur++;
        }

        // Add to output
        outSlices[index++] = UnownedStringSlice(start, cur);

        // Skip the split character, if at end we are okay anyway
        start = cur + 1;
    }

    return index;
}

/* static */ SlangResult StringUtil::split(
    const UnownedStringSlice& in,
    char splitChar,
    Index maxSlices,
    UnownedStringSlice* outSlices,
    Index& outSlicesCount)
{
    const Index sliceCount = split(in, splitChar, maxSlices, outSlices);
    if (sliceCount == maxSlices && sliceCount > 0)
    {
        // To succeed must have parsed all of the input
        if (in.end() != outSlices[sliceCount - 1].end())
        {
            return SLANG_FAIL;
        }
    }
    outSlicesCount = sliceCount;
    return SLANG_OK;
}

/* static */ void StringUtil::join(const List<String>& values, char separator, StringBuilder& out)
{
    join(values, UnownedStringSlice(&separator, 1), out);
}

/* static */ void StringUtil::join(
    const List<String>& values,
    const UnownedStringSlice& separator,
    StringBuilder& out)
{
    const Index count = values.getCount();
    if (count <= 0)
    {
        return;
    }
    out.append(values[0]);
    for (Index i = 1; i < count; i++)
    {
        out.append(separator);
        out.append(values[i]);
    }
}

/* static */ void StringUtil::join(
    const UnownedStringSlice* values,
    Index valueCount,
    char separator,
    StringBuilder& out)
{
    join(values, valueCount, UnownedStringSlice(&separator, 1), out);
}

/* static */ void StringUtil::join(
    const UnownedStringSlice* values,
    Index valueCount,
    const UnownedStringSlice& separator,
    StringBuilder& out)
{
    if (valueCount <= 0)
    {
        return;
    }
    out.append(values[0]);
    for (Index i = 1; i < valueCount; i++)
    {
        out.append(separator);
        out.append(values[i]);
    }
}

/* static */ Index StringUtil::indexOfInSplit(
    const UnownedStringSlice& in,
    char splitChar,
    const UnownedStringSlice& find)
{
    const char* start = in.begin();
    const char* end = in.end();

    for (Index i = 0; start < end; ++i)
    {
        // Move cur so it's either at the end or at next split character
        const char* cur = start;
        while (cur < end && *cur != splitChar)
        {
            cur++;
        }

        // See if we have a match
        if (UnownedStringSlice(start, cur) == find)
        {
            return i;
        }

        // Skip the split character, if at end we are okay anyway
        start = cur + 1;
    }
    return -1;
}

UnownedStringSlice StringUtil::getAtInSplit(
    const UnownedStringSlice& in,
    char splitChar,
    Index index)
{
    const char* start = in.begin();
    const char* end = in.end();

    for (Index i = 0; start < end; ++i)
    {
        // Move cur so it's either at the end or at next split character
        const char* cur = start;
        while (cur < end && *cur != splitChar)
        {
            cur++;
        }

        if (i == index)
        {
            return UnownedStringSlice(start, cur);
        }

        // Skip the split character, if at end we are okay anyway
        start = cur + 1;
    }

    return UnownedStringSlice();
}

/* static */ size_t StringUtil::calcFormattedSize(const char* format, va_list args)
{
#if SLANG_WINDOWS_FAMILY
    return _vscprintf(format, args);
#else
    return vsnprintf(nullptr, 0, format, args);
#endif
}

/* static */ void StringUtil::calcFormatted(
    const char* format,
    va_list args,
    size_t numChars,
    char* dst)
{
#if SLANG_WINDOWS_FAMILY
    vsnprintf_s(dst, numChars + 1, _TRUNCATE, format, args);
#else
    vsnprintf(dst, numChars + 1, format, args);
#endif
}

/* static */ void StringUtil::append(const char* format, va_list args, StringBuilder& buf)
{
    // Calculate the size required (not including terminating 0)
    size_t numChars;
    {
        // Create a copy of args, as will be consumed by calcFormattedSize
        va_list argsCopy;
        va_copy(argsCopy, args);
        numChars = calcFormattedSize(format, argsCopy);
        va_end(argsCopy);
    }

    // Requires + 1 , because calcFormatted appends a terminating 0
    char* dst = buf.prepareForAppend(numChars + 1);
    calcFormatted(format, args, numChars, dst);
    buf.appendInPlace(dst, numChars);
}

/* static */ void StringUtil::appendFormat(StringBuilder& buf, const char* format, ...)
{
    va_list args;
    va_start(args, format);
    append(format, args, buf);
    va_end(args);
}

/* static */ String StringUtil::makeStringWithFormat(const char* format, ...)
{
    StringBuilder builder;

    va_list args;
    va_start(args, format);
    append(format, args, builder);
    va_end(args);

    return builder;
}

template<typename T>
static T readValue(ArrayView<const void*> ptrToArgs, Count& argIndex)
{
    if (argIndex < ptrToArgs.getCount())
    {
        T value;
        memcpy(&value, ptrToArgs[argIndex], sizeof(T));
        argIndex++;
        return value;
    }
    return T();
}

String StringUtil::makeStringWithFormatFromArgArray(
    const char* format,
    ArrayView<const void*> ptrToArgs)
{
    if (!format)
    {
        return String();
    }
    StringBuilder builder;
    const char* ptr = format;
    Count argIndex = 0;
    auto consumeString = [&]()
    {
        if (argIndex < ptrToArgs.getCount())
        {
            const char* strPtr = *(const char**)ptrToArgs[argIndex];
            argIndex++;
            if (strPtr)
            {
                // Append the string to the builder
                builder.append(strPtr);
            }
        }
    };
#define ADVANCE_PTR                     \
    ptr++;                              \
    if (!*ptr)                          \
    {                                   \
        return builder.produceString(); \
    }

    while (*ptr)
    {
        if (*ptr == '%')
        {
            const char* formatStart = ptr;
            ADVANCE_PTR;
            if (*ptr == 's')
            {
                // If we have a %s, then we want to append the data
                consumeString();
                // Move past the 's'
                ADVANCE_PTR;
                continue;
            }
            if (*ptr == '-')
            {
                // If we have a %- then we want to continue parsing format string.
                ADVANCE_PTR;
            }
            while (CharUtil::isDigit(*ptr))
            {
                // Skip the digits after the '.'
                ADVANCE_PTR;
            }
            if (*ptr == '.')
            {
                ADVANCE_PTR;
                while (CharUtil::isDigit(*ptr))
                {
                    // Skip the digits after the '.'
                    ADVANCE_PTR;
                }
            }
            int isLong = 0;
            if (*ptr == 'l' || *ptr == 'L')
            {
                // If we have a 'l' or 'L', then we want to skip it.
                ADVANCE_PTR;
                isLong = 1;
                if (*ptr == 'l' || *ptr == 'L')
                {
                    // If we have another 'l' or 'L', then we want to skip it too.
                    ADVANCE_PTR;
                    isLong = 2;
                }
            }
            const char typeChar = *ptr;
            ADVANCE_PTR;
            String formatStr = UnownedStringSlice(formatStart, ptr);
            switch (CharUtil::toLower(typeChar))
            {
            case 'd':
            case 'x':
            case 'i':
            case 'u':
            case 'o':
            case 'c':
                if (isLong == 2)
                {
                    StringUtil::appendFormat(
                        builder,
                        formatStr.getBuffer(),
                        readValue<int64_t>(ptrToArgs, argIndex));
                }
                else
                {
                    StringUtil::appendFormat(
                        builder,
                        formatStr.getBuffer(),
                        readValue<int>(ptrToArgs, argIndex));
                }
                break;
            case 'e':
            case 'f':
            case 'g':
                if (isLong != 0)
                {
                    StringUtil::appendFormat(
                        builder,
                        formatStr.getBuffer(),
                        readValue<double>(ptrToArgs, argIndex));
                }
                else
                {
                    StringUtil::appendFormat(
                        builder,
                        formatStr.getBuffer(),
                        readValue<float>(ptrToArgs, argIndex));
                }
                break;
            case 'n':
                break;
            case '%':
                // If we have a '%%' then we want to append a single '%'
                builder.appendChar('%');
                continue;
            }
        }
        else
        {
            // Just append the character
            builder.appendChar(*ptr);
            ptr++;
        }
    }
    return builder.produceString();
}


/* static */ UnownedStringSlice StringUtil::getSlice(ISlangBlob* blob)
{
    if (blob)
    {
        size_t size = blob->getBufferSize();
        if (size > 0)
        {
            const char* contents = (const char*)blob->getBufferPointer();
            // Check it has terminating 0, if it has we skip it, because slices do not need zero
            // termination
            if (contents[size - 1] == 0)
            {
                size--;
            }
            return UnownedStringSlice(contents, contents + size);
        }
    }
    return UnownedStringSlice();
}

/* static */ String StringUtil::getString(ISlangBlob* blob)
{
    return getSlice(blob);
}

ComPtr<ISlangBlob> StringUtil::createStringBlob(const String& string)
{
    return StringBlob::create(string);
}

/* static */ String StringUtil::calcCharReplaced(
    const UnownedStringSlice& slice,
    char fromChar,
    char toChar)
{
    if (fromChar == toChar)
    {
        return slice;
    }

    const Index numChars = slice.getLength();
    const char* srcChars = slice.begin();

    StringBuilder builder;
    char* dstChars = builder.prepareForAppend(numChars);

    for (Index i = 0; i < numChars; ++i)
    {
        char c = srcChars[i];
        dstChars[i] = (c == fromChar) ? toChar : c;
    }

    builder.appendInPlace(dstChars, numChars);
    return builder;
}

/* static */ String StringUtil::calcCharReplaced(const String& string, char fromChar, char toChar)
{
    return (fromChar == toChar || string.indexOf(fromChar) == Index(-1))
               ? string
               : calcCharReplaced(string.getUnownedSlice(), fromChar, toChar);
}

String StringUtil::replaceAll(
    UnownedStringSlice text,
    UnownedStringSlice subStr,
    UnownedStringSlice replacement)
{
    StringBuilder builder;
    for (Index i = 0; i < text.getLength();)
    {
        if (i + subStr.getLength() > text.getLength())
        {
            builder.append(text.subString(i, text.getLength() - i));
            break;
        }
        if (text.subString(i, subStr.getLength()) == subStr)
        {
            builder.append(replacement);
            i += subStr.getLength();
        }
        else
        {
            builder.append(text[i]);
            i++;
        }
    }
    return builder.produceString();
}


/* static */ void StringUtil::appendStandardLines(
    const UnownedStringSlice& text,
    StringBuilder& out)
{
    const char* cur = text.begin();
    const char* start = cur;
    const char* const end = text.end();

    while (cur < end)
    {
        const char c = *cur;
        switch (c)
        {
        case '\n':
            {
                ++cur;
                if (cur < end && *cur == '\r')
                {
                    // If we have following \r, we should append with \n
                    // Append (including \n)
                    out.append(start, cur);
                    // Skip the \r
                    start = ++cur;
                }
                else
                {
                    // If not, we don't need to append because just \n is 'standard', and everything
                    // remaining is appended at the end
                }
                break;
            }
        case '\r':
            {
                out.append(start, cur);
                out.appendChar('\n');

                ++cur;
                // If next is \n, we want to skip that
                cur += Index(cur < end && *cur == '\n');
                start = cur;
                break;
            }
        default:
            {
                cur++;
                break;
            }
        }
    }

    if (start < end)
    {
        out.append(start, end);
    }
}

/* static */ bool StringUtil::extractLine(UnownedStringSlice& ioText, UnownedStringSlice& outLine)
{
    char const* const begin = ioText.begin();
    char const* const end = ioText.end();

    // If we have hit the end then return the 'special' terminator
    if (begin == nullptr)
    {
        outLine = UnownedStringSlice(nullptr, nullptr);
        return false;
    }

    char const* cursor = begin;
    while (cursor < end)
    {
        int c = *cursor++;
        switch (c)
        {
        case '\r':
        case '\n':
            {
                // Remember the end of the line
                const char* const lineEnd = cursor - 1;

                // When we see a line-break character we need
                // to record the line break, but we also need
                // to deal with the annoying issue of encodings,
                // where a multi-byte sequence might encode
                // the line break.
                if (cursor < end)
                {
                    int d = *cursor;
                    if ((c ^ d) == ('\r' ^ '\n'))
                        cursor++;
                }

                ioText = UnownedStringSlice(cursor, end);
                outLine = UnownedStringSlice(begin, lineEnd);
                return true;
            }
        default:
            break;
        }
    }

    // There is nothing remaining
    ioText = UnownedStringSlice(nullptr, nullptr);

    // Could be empty, or the remaining line (without line end terminators of)
    SLANG_ASSERT(begin <= cursor);

    outLine = UnownedStringSlice(begin, cursor);
    return true;
}

/* static */ void StringUtil::calcLines(
    const UnownedStringSlice& textIn,
    List<UnownedStringSlice>& outLines)
{
    outLines.clear();
    UnownedStringSlice text(textIn), line;
    while (extractLine(text, line))
    {
        outLines.add(line);
    }
}

/* static */ UnownedStringSlice StringUtil::trimEndOfLine(const UnownedStringSlice& line)
{
    // Strip CR/LF from end of line if present

    const char* begin = line.begin();
    const char* end = line.end();

    if (end > begin)
    {
        const char c = end[-1];
        // If last char is CR/LF move back a char
        if (c == '\n' || c == '\r')
        {
            --end;
            // If next char is a match for the CR/LF pair move back an extra char.
            end -= Index((end > begin) && (c ^ end[-1]) == ('\r' ^ '\n'));
        }
    }

    return line.head(Index(end - begin));
}

/* static */ bool StringUtil::areLinesEqual(
    const UnownedStringSlice& inA,
    const UnownedStringSlice& inB)
{
    UnownedStringSlice a(inA), b(inB), lineA, lineB;

    while (true)
    {
        const auto hasLineA = extractLine(a, lineA);
        const auto hasLineB = extractLine(b, lineB);

        if (!(hasLineA && hasLineB))
        {
            return hasLineA == hasLineB;
        }

        // The lines must be equal
        if (lineA != lineB)
        {
            return false;
        }
    }
}

/* static */ SlangResult StringUtil::parseDouble(const UnownedStringSlice& text, double& out)
{
    const Index bufSize = 32;

    const auto len = text.getLength();

    if (len > bufSize - 1)
    {
        List<char> work;
        work.setCount(len + 1);
        char* dst = work.getBuffer();

        ::memcpy(dst, text.begin(), len * sizeof(char));
        dst[len] = 0;

        out = atof(dst);
    }
    else
    {
        char buf[bufSize];
        ::memcpy(buf, text.begin(), len * sizeof(char));
        buf[len] = 0;
        out = atof(buf);
    }
    return SLANG_OK;
}

/* static */ SlangResult StringUtil::parseInt(const UnownedStringSlice& in, Int& outValue)
{
    const char* cur = in.begin();
    const char* end = in.end();

    bool negate = false;
    if (cur < end && *cur == '-')
    {
        negate = true;
        cur++;
    }

    int radix = 10;
    auto getDigit = CharUtil::getDecimalDigitValue;
    if (cur + 1 < end && *cur == '0' && (*(cur + 1) == 'x' || *(cur + 1) == 'X'))
    {
        radix = 16;
        getDigit = CharUtil::getHexDigitValue;
        cur += 2;
    }

    // We need at least one digit
    if (cur >= end || !CharUtil::isDigit(*cur))
    {
        return SLANG_FAIL;
    }

    Int value = 0;
    // Do the digits
    for (; cur < end; ++cur)
    {
        const auto d = getDigit(*cur);
        if (d == -1)
            return SLANG_FAIL;
        value = value * radix + d;
    }

    value = negate ? -value : value;

    outValue = value;
    return SLANG_OK;
}

/* static */ SlangResult StringUtil::parseInt64(const UnownedStringSlice& text, int64_t& out)
{
    bool negate = false;

    const char* cur = text.begin();
    const char* end = text.end();

    if (cur < end)
    {
        if (*cur == '-')
        {
            negate = true;
            cur++;
        }
        else if (*cur == '+')
        {
            cur++;
        }
    }

    // Must have at least one digit
    if (cur >= end || !CharUtil::isDigit(*cur))
    {
        return SLANG_FAIL;
    }

    uint64_t value = 0;
    // We can have 20 digits, but the last digit can cause overflow.
    // Lets do the easy first digits first
    Index numSimple = 19;
    for (; cur < end && CharUtil::isDigit(*cur) && numSimple > 0; ++cur, --numSimple)
    {
        value = value * 10 + (*cur - '0');
    }

    if (cur < end && CharUtil::isDigit(*cur))
    {
        const auto prevValue = value;
        value = value * 10 + (*cur - '0');
        cur++;

        if (value < prevValue)
        {
            // We have overflow
            return SLANG_FAIL;
        }
    }

    if (negate)
    {
        if (value > ~((~uint64_t(0)) >> 1))
        {
            // Overflow
            return SLANG_FAIL;
        }
        out = -int64_t(value);
    }
    else
    {
        if (value > ((~uint64_t(0)) >> 1))
        {
            // Overflow
            return SLANG_FAIL;
        }
        out = value;
    }

    return (cur == end) ? SLANG_OK : SLANG_FAIL;
}

int StringUtil::parseIntAndAdvancePos(UnownedStringSlice text, Index& pos)
{
    int result = 0;
    while (text[pos] == ' ' && pos < text.getLength())
    {
        pos++;
        continue;
    }
    bool isNeg = false;
    if (pos < text.getLength() && text[pos] == '-')
    {
        pos++;
        isNeg = true;
    }
    while (pos < text.getLength())
    {
        if (text[pos] >= '0' && text[pos] <= '9')
        {
            result *= 10;
            result += text[pos] - '0';
            pos++;
        }
        else
        {
            break;
        }
    }
    if (isNeg)
        result = -result;
    return result;
}

} // namespace Slang
