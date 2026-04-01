#include "slang-string.h"

#include "slang-char-util.h"
#include "slang-text-io.h"

namespace Slang
{
// HACK!
// JS: Many of the inlined functions of CharUtil just access a global map. That referencing this
// global is *NOT* enough to link correctly with CharUtil on linux for a shared library. The
// following call exists to try and force linkage of CharUtil for anything that uses core
static const auto s_charUtilLink = CharUtil::_ensureLink();


// StringRepresentation

void StringRepresentation::setContents(const UnownedStringSlice& slice)
{
    const auto sliceLength = slice.getLength();
    SLANG_ASSERT(sliceLength <= capacity);

    char* chars = getData();

    // Use move (rather than memcpy), because the slice *could* be contained in the
    // StringRepresentation
    ::memmove(chars, slice.begin(), sliceLength * sizeof(char));
    // Zero terminate.
    chars[sliceLength] = 0;
    // Set the length
    length = sliceLength;
}


/* static */ StringRepresentation* StringRepresentation::create(const UnownedStringSlice& slice)
{
    const auto sliceLength = slice.getLength();

    if (sliceLength)
    {
        StringRepresentation* rep = StringRepresentation::createWithLength(sliceLength);

        char* chars = rep->getData();
        ::memcpy(chars, slice.begin(), sizeof(char) * sliceLength);
        chars[sliceLength] = 0;

        return rep;
    }
    else
    {
        return nullptr;
    }
}

/* static */ StringRepresentation* StringRepresentation::createWithReference(
    const UnownedStringSlice& slice)
{
    const auto sliceLength = slice.getLength();

    if (sliceLength)
    {
        StringRepresentation* rep = StringRepresentation::createWithLength(sliceLength);
        rep->addReference();

        char* chars = rep->getData();
        ::memcpy(chars, slice.begin(), sizeof(char) * sliceLength);
        chars[sliceLength] = 0;

        return rep;
    }
    else
    {
        return nullptr;
    }
}

// OSString

OSString::OSString()
    : m_begin(nullptr), m_end(nullptr)
{
}

OSString::OSString(wchar_t* begin, wchar_t* end)
    : m_begin(begin), m_end(end)
{
}

void OSString::_releaseBuffer()
{
    if (m_begin)
    {
        delete[] m_begin;
    }
}

void OSString::set(const wchar_t* begin, const wchar_t* end)
{
    if (m_begin)
    {
        delete[] m_begin;
        m_begin = nullptr;
        m_end = nullptr;
    }
    const size_t len = end - begin;
    if (len > 0)
    {
        // TODO(JS): The allocation is only done this way to be compatible with the buffer being
        // detached from an array This is unfortunate, because it means that the allocation stores
        // the size (and alignment fix), which is a shame because we know the size
        m_begin = new wchar_t[len + 1];
        memcpy(m_begin, begin, len * sizeof(wchar_t));
        // Zero terminate
        m_begin[len] = 0;
        m_end = m_begin + len;
    }
}

static const wchar_t kEmptyOSString[] = {0};

wchar_t const* OSString::begin() const
{
    return m_begin ? m_begin : kEmptyOSString;
}

wchar_t const* OSString::end() const
{
    return m_end ? m_end : kEmptyOSString;
}

// UnownedStringSlice

bool UnownedStringSlice::startsWith(UnownedStringSlice const& other) const
{
    UInt thisSize = getLength();
    UInt otherSize = other.getLength();

    if (otherSize > thisSize)
        return false;

    return head(otherSize) == other;
}

bool UnownedStringSlice::startsWith(char const* str) const
{
    return startsWith(UnownedTerminatedStringSlice(str));
}

bool UnownedStringSlice::startsWithCaseInsensitive(UnownedStringSlice const& other) const
{
    UInt thisSize = getLength();
    UInt otherSize = other.getLength();

    if (otherSize > thisSize)
        return false;

    return head(otherSize).caseInsensitiveEquals(other);
}


bool UnownedStringSlice::endsWith(UnownedStringSlice const& other) const
{
    UInt thisSize = getLength();
    UInt otherSize = other.getLength();

    if (otherSize > thisSize)
        return false;

    return UnownedStringSlice(end() - otherSize, end()) == other;
}

bool UnownedStringSlice::endsWithCaseInsensitive(UnownedStringSlice const& other) const
{
    UInt thisSize = getLength();
    UInt otherSize = other.getLength();

    if (otherSize > thisSize)
        return false;

    return UnownedStringSlice(end() - otherSize, end()).caseInsensitiveEquals(other);
}

bool UnownedStringSlice::endsWith(char const* str) const
{
    return endsWith(UnownedTerminatedStringSlice(str));
}

bool UnownedStringSlice::endsWithCaseInsensitive(char const* str) const
{
    return endsWithCaseInsensitive(UnownedTerminatedStringSlice(str));
}

UnownedStringSlice UnownedStringSlice::trim() const
{
    const char* start = m_begin;
    const char* end = m_end;

    while (start < end && CharUtil::isHorizontalWhitespace(*start))
        start++;
    while (end > start && CharUtil::isHorizontalWhitespace(end[-1]))
        end--;
    return UnownedStringSlice(start, end);
}

UnownedStringSlice UnownedStringSlice::trimStart() const
{
    const char* start = m_begin;

    while (start < m_end && CharUtil::isHorizontalWhitespace(*start))
        start++;
    return UnownedStringSlice(start, m_end);
}

UnownedStringSlice UnownedStringSlice::trim(char c) const
{
    const char* start = m_begin;
    const char* end = m_end;

    while (start < end && *start == c)
        start++;
    while (end > start && end[-1] == c)
        end--;
    return UnownedStringSlice(start, end);
}

// StringSlice

StringSlice::StringSlice()
    : representation(0), beginIndex(0), endIndex(0)
{
}

StringSlice::StringSlice(String const& str)
    : representation(str.m_buffer), beginIndex(0), endIndex(str.getLength())
{
}

StringSlice::StringSlice(String const& str, UInt beginIndex, UInt endIndex)
    : representation(str.m_buffer), beginIndex(beginIndex), endIndex(endIndex)
{
}


//

_EndLine EndLine;

String operator+(const char* op1, const String& op2)
{
    String result(op1);
    result.append(op2);
    return result;
}

String operator+(const String& op1, const char* op2)
{
    String result(op1);
    result.append(op2);
    return result;
}

String operator+(const String& op1, const String& op2)
{
    String result(op1);
    result.append(op2);
    return result;
}

int stringToInt(const String& str, int radix)
{
    if (str.startsWith("0x"))
        return (int)strtoll(str.getBuffer(), NULL, 16);
    else
        return (int)strtoll(str.getBuffer(), NULL, radix);
}
unsigned int stringToUInt(const String& str, int radix)
{
    if (str.startsWith("0x"))
        return (unsigned int)strtoull(str.getBuffer(), NULL, 16);
    else
        return (unsigned int)strtoull(str.getBuffer(), NULL, radix);
}
double stringToDouble(const String& str)
{
    return (double)strtod(str.getBuffer(), NULL);
}
float stringToFloat(const String& str)
{
    return strtof(str.getBuffer(), NULL);
}

#if 0
    String String::ReplaceAll(String src, String dst) const
    {
        String rs = *this;
        int index = 0;
        int srcLen = src.length;
        int len = rs.length;
        while ((index = rs.IndexOf(src, index)) != -1)
        {
            rs = rs.SubString(0, index) + dst + rs.SubString(index + srcLen, len - index - srcLen);
            len = rs.length;
        }
        return rs;
    }
#endif

String String::fromWString(const wchar_t* wstr)
{
    List<char> buf;
#ifdef _WIN32
    Slang::CharEncoding::UTF16->decode(
        (const Byte*)wstr,
        (int)(wcslen(wstr) * sizeof(wchar_t)),
        buf);
#else
    Slang::CharEncoding::UTF32->decode(
        (const Byte*)wstr,
        (int)(wcslen(wstr) * sizeof(wchar_t)),
        buf);
#endif
    return String(buf.begin(), buf.end());
}

String String::fromWString(const wchar_t* wstr, const wchar_t* wend)
{
    List<char> buf;
#ifdef _WIN32
    Slang::CharEncoding::UTF16->decode(
        (const Byte*)wstr,
        (int)((wend - wstr) * sizeof(wchar_t)),
        buf);
#else
    Slang::CharEncoding::UTF32->decode(
        (const Byte*)wstr,
        (int)((wend - wstr) * sizeof(wchar_t)),
        buf);
#endif
    return String(buf.begin(), buf.end());
}

String String::fromWChar(const wchar_t ch)
{
    List<char> buf;
#ifdef _WIN32
    Slang::CharEncoding::UTF16->decode((const Byte*)&ch, (int)(sizeof(wchar_t)), buf);
#else
    Slang::CharEncoding::UTF32->decode((const Byte*)&ch, (int)(sizeof(wchar_t)), buf);
#endif
    return String(buf.begin(), buf.end());
}

/* static */ String String::fromUnicodePoint(Char32 codePoint)
{
    char buf[6];
    int len = Slang::encodeUnicodePointToUTF8(codePoint, buf);
    return String(buf, buf + len);
}

OSString String::toWString(Index* outLength) const
{
    if (!m_buffer)
    {
        return OSString();
    }
    else
    {
        List<Byte> buf;
        switch (sizeof(wchar_t))
        {
        case 2:
            Slang::CharEncoding::UTF16->encode(getUnownedSlice(), buf);
            break;

        case 4:
            Slang::CharEncoding::UTF32->encode(getUnownedSlice(), buf);
            break;

        default:
            break;
        }

        auto length = Index(buf.getCount() / sizeof(wchar_t));
        if (outLength)
            *outLength = length;

        for (size_t ii = 0; ii < sizeof(wchar_t); ++ii)
            buf.add(0);

        wchar_t* beginData = (wchar_t*)buf.getBuffer();
        wchar_t* endData = beginData + length;

        OSString ret;
        ret.set(beginData, endData);
        return ret;
    }
}

//

void String::ensureUniqueStorageWithCapacity(Index requiredCapacity)
{
    if (m_buffer && m_buffer->isUniquelyReferenced() && m_buffer->capacity >= requiredCapacity)
        return;

    Index newCapacity = m_buffer ? 2 * m_buffer->capacity : 16;
    if (newCapacity < requiredCapacity)
    {
        newCapacity = requiredCapacity;
    }

    Index length = getLength();
    StringRepresentation* newRepresentation =
        StringRepresentation::createWithCapacityAndLength(newCapacity, length);

    if (m_buffer)
    {
        memcpy(newRepresentation->getData(), m_buffer->getData(), length + 1);
    }

    m_buffer = newRepresentation;
}

char* String::prepareForAppend(Index count)
{
    auto oldLength = getLength();
    auto newLength = oldLength + count;
    ensureUniqueStorageWithCapacity(newLength);
    return getData() + oldLength;
}
void String::appendInPlace(const char* chars, Index count)
{
    SLANG_UNUSED(chars);

    if (count > 0)
    {
        SLANG_ASSERT(m_buffer && m_buffer->isUniquelyReferenced());

        auto oldLength = getLength();
        auto newLength = oldLength + count;

        char* dst = m_buffer->getData();

        // Make sure the input buffer is the same one returned from prepareForAppend
        SLANG_ASSERT(chars == dst + oldLength);
        // It has to fit within the capacity
        SLANG_ASSERT(newLength <= m_buffer->capacity);

        // We just need to modify the length
        m_buffer->length = newLength;

        // And mark with a terminating 0
        dst[newLength] = 0;
    }
}

void String::reduceLength(Index newLength)
{
    Index oldLength = getLength();
    SLANG_ASSERT(newLength <= oldLength);
    if (oldLength == newLength)
    {
        return;
    }

    // It must have a buffer, because only 0 length allows for nullptr
    // and being 0 sized is already covered
    SLANG_ASSERT(m_buffer);

    if (m_buffer->isUniquelyReferenced())
    {
        m_buffer->length = newLength;
        m_buffer->getData()[newLength] = 0;
    }
    else
    {
        // If 0 length is wanted we can just free
        if (newLength == 0)
        {
            m_buffer.setNull();
        }
        else
        {
            // We need to make a new copy, that we will shrink

            // We'll just go with capacity enough for the new length
            const Index newCapacity = newLength;
            StringRepresentation* newRepresentation =
                StringRepresentation::createWithCapacityAndLength(newCapacity, newLength);

            // Copy
            char* dst = newRepresentation->getData();
            memcpy(dst, m_buffer->getData(), sizeof(char) * newLength);
            // Zero terminate
            dst[newLength] = 0;

            // Set the new rep
            m_buffer = newRepresentation;
        }
    }
}

void String::append(char const* str, size_t len)
{
    append(str, str + len);
}

void String::append(const char* textBegin, char const* textEnd)
{
    auto oldLength = getLength();
    auto textLength = textEnd - textBegin;
    if (textLength <= 0)
        return;

    auto newLength = oldLength + textLength;

    ensureUniqueStorageWithCapacity(newLength);

    memcpy(getData() + oldLength, textBegin, textLength);
    getData()[newLength] = 0;
    m_buffer->length = newLength;
}

void String::append(char const* str)
{
    if (str)
    {
        append(str, str + strlen(str));
    }
}

void String::appendRepeatedChar(char chr, Index count)
{
    SLANG_ASSERT(count >= 0);
    if (count > 0)
    {
        char* chars = prepareForAppend(count);
        // Set all space to repeated chr.
        ::memset(chars, chr, sizeof(char) * count);
        appendInPlace(chars, count);
    }
}

void String::appendChar(char c)
{
    const auto oldLength = getLength();
    const auto newLength = oldLength + 1;

    ensureUniqueStorageWithCapacity(newLength);

    // Since there must be space for at least one character, m_buffer cannot be nullptr
    SLANG_ASSERT(m_buffer);
    char* data = m_buffer->getData();
    data[oldLength] = c;
    data[newLength] = 0;

    m_buffer->length = newLength;
}

void String::append(char chr)
{
    appendChar(chr);
}

void String::append(String const& str)
{
    if (!m_buffer)
    {
        m_buffer = str.m_buffer;
        return;
    }

    append(str.begin(), str.end());
}

void String::append(StringSlice const& slice)
{
    append(slice.begin(), slice.end());
}

void String::append(UnownedStringSlice const& slice)
{
    append(slice.begin(), slice.end());
}

void String::append(int32_t value, int radix)
{
    enum
    {
        kCount = 33
    };
    char* data = prepareForAppend(kCount);
    const auto count = intToAscii(data, value, radix);
    m_buffer->length += count;
}

void String::append(uint32_t value, int radix)
{
    enum
    {
        kCount = 33
    };
    char* data = prepareForAppend(kCount);
    const auto count = intToAscii(data, value, radix);
    m_buffer->length += count;
}

void String::append(int64_t value, int radix)
{
    enum
    {
        kCount = 65
    };
    char* data = prepareForAppend(kCount);
    auto count = intToAscii(data, value, radix);
    m_buffer->length += count;
}

void String::append(uint64_t value, int radix)
{
    enum
    {
        kCount = 65
    };
    char* data = prepareForAppend(kCount);
    auto count = intToAscii(data, value, radix);
    m_buffer->length += count;
}

void String::append(float val, const char* format)
{
    enum
    {
        kCount = 128
    };
    char* data = prepareForAppend(kCount);
    sprintf_s(data, kCount, format, val);
    m_buffer->length += strnlen_s(data, kCount);
}

void String::append(double val, const char* format)
{
    enum
    {
        kCount = 128
    };
    char* data = prepareForAppend(kCount);
    sprintf_s(data, kCount, format, val);
    m_buffer->length += strnlen_s(data, kCount);
}

void String::append(StableHashCode32 value)
{
    const Index digits = 8;
    // + null terminator
    char* data = prepareForAppend(digits + 1);
    auto count = intToAscii(data, value.hash, 16, digits);
    m_buffer->length += count;
}

void String::append(StableHashCode64 value)
{
    const Index digits = 16;
    // + null terminator
    char* data = prepareForAppend(digits + 1);
    auto count = intToAscii(data, value.hash, 16, digits);
    m_buffer->length += count;
}

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! UnownedStringSlice !!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Index UnownedStringSlice::indexOf(char c) const
{
    const Index size = Index(m_end - m_begin);
    for (Index i = 0; i < size; ++i)
    {
        if (m_begin[i] == c)
        {
            return i;
        }
    }
    return -1;
}

Index UnownedStringSlice::indexOf(const UnownedStringSlice& in) const
{
    const Index len = getLength();
    const Index inLen = in.getLength();
    if (inLen > len)
    {
        return -1;
    }

    const char* inChars = in.m_begin;
    switch (inLen)
    {
    case 0:
        return 0;
    case 1:
        return indexOf(inChars[0]);
    default:
        break;
    }

    const char* chars = m_begin;
    const char firstChar = inChars[0];

    for (Int i = 0; i <= len - inLen; ++i)
    {
        if (chars[i] == firstChar && in == UnownedStringSlice(chars + i, inLen))
        {
            return i;
        }
    }

    return -1;
}

UnownedStringSlice UnownedStringSlice::subString(Index idx, Index len) const
{
    const Index totalLen = getLength();
    SLANG_ASSERT(idx >= 0 && len >= 0 && idx <= totalLen);

    // If too large, we truncate
    len = (idx + len > totalLen) ? (totalLen - idx) : len;

    // Return the substring
    return UnownedStringSlice(m_begin + idx, m_begin + idx + len);
}

bool UnownedStringSlice::operator==(ThisType const& other) const
{
    // Note that memcmp is undefined when passed in null ptrs, so if we want to handle
    // we need to cover that case.
    // Can only be nullptr if size is 0.
    auto thisSize = getLength();
    auto otherSize = other.getLength();

    if (thisSize != otherSize)
    {
        return false;
    }

    const char* const thisChars = begin();
    const char* const otherChars = other.begin();
    if (thisChars == otherChars || thisSize == 0)
    {
        return true;
    }
    SLANG_ASSERT(thisChars && otherChars);
    return memcmp(thisChars, otherChars, thisSize) == 0;
}

bool UnownedStringSlice::caseInsensitiveEquals(const ThisType& rhs) const
{
    const auto length = getLength();
    if (length != rhs.getLength())
    {
        return false;
    }

    const char* a = m_begin;
    const char* b = rhs.m_begin;

    // Assuming this is a faster test
    if (memcmp(a, b, length) != 0)
    {
        // They aren't identical so compare character by character
        for (Index i = 0; i < length; ++i)
        {
            if (CharUtil::toLower(a[i]) != CharUtil::toLower(b[i]))
            {
                return false;
            }
        }
    }

    return true;
}
} // namespace Slang

std::ostream& operator<<(std::ostream& stream, const Slang::String& s)
{
    stream << s.getBuffer();
    return stream;
}
