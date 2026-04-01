#ifndef SLANG_CORE_STRING_H
#define SLANG_CORE_STRING_H

#include "slang-common.h"
#include "slang-hash.h"
#include "slang-secure-crt.h"
#include "slang-smart-pointer.h"
#include "slang-stable-hash.h"

#include <cstdlib>
#include <iostream>
#include <new>
#include <stdio.h>
#include <string.h>
#include <type_traits>

namespace Slang
{
class _EndLine
{
};
extern _EndLine EndLine;

// in-place reversion, works only for ascii string
inline void reverseInplaceAscii(char* buffer, int length)
{
    int i, j;
    char c;
    for (i = 0, j = length - 1; i < j; i++, j--)
    {
        c = buffer[i];
        buffer[i] = buffer[j];
        buffer[j] = c;
    }
}
template<typename IntType>
inline int intToAscii(char* buffer, IntType val, int radix, int padTo = 0)
{
    static_assert(std::is_integral_v<IntType>);

    int i = 0;
    IntType sign;

    sign = val;
    if (sign < 0)
    {
        val = (IntType)(0 - val);
    }

    do
    {
        int digit = (val % radix);
        if (digit <= 9)
            buffer[i++] = (char)(digit + '0');
        else
            buffer[i++] = (char)(digit - 10 + 'A');
    } while ((val /= radix) > 0);

    SLANG_ASSERT(i >= 0);
    while (i < padTo)
        buffer[i++] = '0';

    if (sign < 0)
        buffer[i++] = '-';

    // Put in normal character order
    reverseInplaceAscii(buffer, i);

    buffer[i] = '\0';
    return i;
}

SLANG_FORCE_INLINE bool isUtf8LeadingByte(char ch)
{
    return (((unsigned char)ch) & 0xC0) == 0xC0;
}

SLANG_FORCE_INLINE bool isUtf8ContinuationByte(char ch)
{
    return (((unsigned char)ch) & 0xC0) == 0x80;
}

/* A string slice that doesn't own the contained characters.
It is the responsibility of code using the type to keep the memory backing
the slice in scope.
A slice is generally *not* zero terminated. */
struct SLANG_RT_API UnownedStringSlice
{
public:
    typedef UnownedStringSlice ThisType;

    // Type to indicate that a ctor is with a length to disabmiguate 0/nullptr
    // causing ambiguity.
    struct WithLength
    {
    };

    UnownedStringSlice()
        : m_begin(nullptr), m_end(nullptr)
    {
    }

    explicit UnownedStringSlice(char const* a)
        : m_begin(a), m_end(a ? a + strlen(a) : nullptr)
    {
    }
    UnownedStringSlice(char const* b, char const* e)
        : m_begin(b), m_end(e)
    {
    }
    UnownedStringSlice(char const* b, size_t len)
        : m_begin(b), m_end(b + len)
    {
    }
    UnownedStringSlice(WithLength, char const* b, size_t len)
        : m_begin(b), m_end(b + len)
    {
    }

    SLANG_FORCE_INLINE char const* begin() const { return m_begin; }

    SLANG_FORCE_INLINE char const* end() const { return m_end; }

    /// True if slice is strictly contained in memory.
    bool isMemoryContained(const UnownedStringSlice& slice) const
    {
        return slice.m_begin >= m_begin && slice.m_end <= m_end;
    }
    bool isMemoryContained(const char* pos) const { return pos >= m_begin && pos <= m_end; }

    /// Get the length in *bytes*
    Count getLength() const { return Index(m_end - m_begin); }

    /// Finds first index of char 'c'. If not found returns -1.
    Index indexOf(char c) const;
    /// Find first index of slice. If not found returns -1
    Index indexOf(const UnownedStringSlice& slice) const;

    /// Returns a substring. idx is the start index, and len
    /// is the amount of characters.
    /// The returned length might be truncated, if len extends beyond slice.
    UnownedStringSlice subString(Index idx, Index len) const;

    /// Return a head of the slice - everything up to the index
    SLANG_FORCE_INLINE UnownedStringSlice head(Index idx) const
    {
        SLANG_ASSERT(idx >= 0 && idx <= getLength());
        return UnownedStringSlice(m_begin, idx);
    }
    /// Return a tail of the slice - everything from the index to the end of the slice
    SLANG_FORCE_INLINE UnownedStringSlice tail(Index idx) const
    {
        SLANG_ASSERT(idx >= 0 && idx <= getLength());
        return UnownedStringSlice(m_begin + idx, m_end);
    }

    /// True if rhs and this are equal without having to take into account case
    /// Note 'case' here is *not* locale specific - it is only A-Z and a-z
    bool caseInsensitiveEquals(const ThisType& rhs) const;

    Index lastIndexOf(char c) const
    {
        const Index size = Index(m_end - m_begin);
        for (Index i = size - 1; i >= 0; --i)
        {
            if (m_begin[i] == c)
            {
                return i;
            }
        }
        return -1;
    }

    const char& operator[](Index i) const
    {
        assert(i >= 0 && i < Index(m_end - m_begin));
        return m_begin[i];
    }

    bool operator==(ThisType const& other) const;
    bool operator!=(UnownedStringSlice const& other) const { return !(*this == other); }

    bool operator==(char const* str) const { return (*this) == UnownedStringSlice(str); }
    bool operator!=(char const* str) const { return !(*this == str); }

    /// True if contents is a single char of c
    SLANG_FORCE_INLINE bool isChar(char c) const { return getLength() == 1 && m_begin[0] == c; }

    bool startsWithCaseInsensitive(UnownedStringSlice const& other) const;
    bool startsWith(UnownedStringSlice const& other) const;
    bool startsWith(char const* str) const;

    bool endsWithCaseInsensitive(UnownedStringSlice const& other) const;
    bool endsWithCaseInsensitive(char const* str) const;

    bool endsWith(UnownedStringSlice const& other) const;
    bool endsWith(char const* str) const;

    /// Trims any horizontal whitespace from the start and end and returns as a substring
    UnownedStringSlice trim() const;
    /// Trims any 'c' from the start or the end, and returns as a substring
    UnownedStringSlice trim(char c) const;

    /// Trims any horizonatl whitespace from start and returns as a substring
    UnownedStringSlice trimStart() const;

    static constexpr bool kHasUniformHash = true;
    HashCode64 getHashCode() const { return Slang::getHashCode(m_begin, size_t(m_end - m_begin)); }

    template<size_t SIZE>
    SLANG_FORCE_INLINE static UnownedStringSlice fromLiteral(const char (&in)[SIZE])
    {
        return UnownedStringSlice(in, SIZE - 1);
    }

protected:
    char const* m_begin;
    char const* m_end;
};

// A more convenient way to make slices from *string literals*
template<size_t SIZE>
SLANG_FORCE_INLINE UnownedStringSlice toSlice(const char (&in)[SIZE])
{
    return UnownedStringSlice(in, SIZE - 1);
}

/// Same as UnownedStringSlice, but must be zero terminated.
/// Zero termination is *not* included in the length.
struct SLANG_RT_API UnownedTerminatedStringSlice : public UnownedStringSlice
{
public:
    typedef UnownedStringSlice Super;
    typedef UnownedTerminatedStringSlice ThisType;

    /// We can turn into a regular zero terminated string
    SLANG_FORCE_INLINE operator const char*() const { return m_begin; }

    /// Exists to match the equivalent function in String.
    SLANG_FORCE_INLINE char const* getBuffer() const { return m_begin; }

    /// Construct from a literal directly.
    template<size_t SIZE>
    SLANG_FORCE_INLINE static ThisType fromLiteral(const char (&in)[SIZE])
    {
        return ThisType(in, SIZE - 1);
    }

    /// Default constructor
    UnownedTerminatedStringSlice()
        : Super(Super::WithLength(), "", 0)
    {
    }

    /// Note, b cannot be null because if it were then the string would not be null terminated
    UnownedTerminatedStringSlice(char const* b)
        : Super(b, b + strlen(b))
    {
    }
    UnownedTerminatedStringSlice(char const* b, size_t len)
        : Super(b, len)
    {
        // b must be valid and it must be null terminated
        SLANG_ASSERT(b && b[len] == 0);
    }
};

// A more convenient way to make terminated slices from *string literals*
template<size_t SIZE>
SLANG_FORCE_INLINE UnownedTerminatedStringSlice toTerminatedSlice(const char (&in)[SIZE])
{
    return UnownedTerminatedStringSlice(in, SIZE - 1);
}

// A `StringRepresentation` provides the backing storage for
// all reference-counted string-related types.
class SLANG_RT_API StringRepresentation : public RefObject
{
public:
    Index length;
    Index capacity;

    SLANG_FORCE_INLINE Index getLength() const { return length; }

    SLANG_FORCE_INLINE char* getData() { return (char*)(this + 1); }
    SLANG_FORCE_INLINE const char* getData() const { return (const char*)(this + 1); }

    /// Set the contents to be the slice. Must be enough capacity to hold the slice.
    void setContents(const UnownedStringSlice& slice);

    static const char* getData(const StringRepresentation* stringRep)
    {
        return stringRep ? stringRep->getData() : "";
    }

    static UnownedStringSlice asSlice(const StringRepresentation* rep)
    {
        return rep ? UnownedStringSlice(rep->getData(), rep->getLength()) : UnownedStringSlice();
    }

    static bool equal(const StringRepresentation* a, const StringRepresentation* b)
    {
        return (a == b) || asSlice(a) == asSlice(b);
    }

    static StringRepresentation* createWithCapacityAndLength(Index capacity, Index length)
    {
        SLANG_ASSERT(capacity >= length);
        void* allocation = operator new(sizeof(StringRepresentation) + capacity + 1);
        StringRepresentation* obj = new (allocation) StringRepresentation();
        obj->capacity = capacity;
        obj->length = length;
        obj->getData()[length] = 0;
        return obj;
    }

    static StringRepresentation* createWithCapacity(Index capacity)
    {
        return createWithCapacityAndLength(capacity, 0);
    }

    static StringRepresentation* createWithLength(Index length)
    {
        return createWithCapacityAndLength(length, length);
    }

    /// Create a representation from the slice. If slice is empty will return nullptr.
    static StringRepresentation* create(const UnownedStringSlice& slice);
    /// Same as create, but representation will have refcount of 1 (if not nullptr)
    static StringRepresentation* createWithReference(const UnownedStringSlice& slice);

    StringRepresentation* cloneWithCapacity(Index newCapacity)
    {
        StringRepresentation* newObj = createWithCapacityAndLength(newCapacity, length);
        memcpy(getData(), newObj->getData(), length + 1);
        return newObj;
    }

    StringRepresentation* clone() { return cloneWithCapacity(length); }

    StringRepresentation* ensureCapacity(Index required)
    {
        if (capacity >= required)
            return this;

        Index newCapacity = capacity;
        if (!newCapacity)
            newCapacity = 16; // TODO: figure out good value for minimum capacity

        while (newCapacity < required)
        {
            newCapacity = 2 * newCapacity;
        }

        return cloneWithCapacity(newCapacity);
    }

    /// Overload delete to silence ASAN new-delete-type-mismatch errors.
    /// These occur because the allocation size of StringRepresentation
    /// does not match deallocation size (due variable sized string payload).
    void operator delete(void* p)
    {
        StringRepresentation* str = (StringRepresentation*)p;
        ::operator delete(str);
    }
};

class String;

struct SLANG_RT_API StringSlice
{
public:
    StringSlice();

    StringSlice(String const& str);

    StringSlice(String const& str, UInt beginIndex, UInt endIndex);

    UInt getLength() const { return endIndex - beginIndex; }

    char const* begin() const
    {
        return representation ? representation->getData() + beginIndex : "";
    }

    char const* end() const { return begin() + getLength(); }

private:
    RefPtr<StringRepresentation> representation;
    UInt beginIndex;
    UInt endIndex;

    friend class String;

    StringSlice(RefPtr<StringRepresentation> const& representation, UInt beginIndex, UInt endIndex)
        : representation(representation), beginIndex(beginIndex), endIndex(endIndex)
    {
    }
};

/// String as expected by underlying platform APIs
class SLANG_RT_API OSString
{
public:
    /// Default
    OSString();
    /// NOTE! This assumes that begin is a new wchar_t[] buffer, and it will
    /// now be owned by the OSString
    OSString(wchar_t* begin, wchar_t* end);
    /// Move Ctor
    OSString(OSString&& rhs)
        : m_begin(rhs.m_begin), m_end(rhs.m_end)
    {
        rhs.m_begin = nullptr;
        rhs.m_end = nullptr;
    }
    // Copy Ctor
    OSString(const OSString& rhs)
        : m_begin(nullptr), m_end(nullptr)
    {
        set(rhs.m_begin, rhs.m_end);
    }

    /// =
    void operator=(const OSString& rhs) { set(rhs.m_begin, rhs.m_end); }
    void operator=(OSString&& rhs)
    {
        auto begin = m_begin;
        auto end = m_end;
        m_begin = rhs.m_begin;
        m_end = rhs.m_end;
        rhs.m_begin = begin;
        rhs.m_end = end;
    }

    ~OSString() { _releaseBuffer(); }

    size_t getLength() const { return (m_end - m_begin); }
    void set(const wchar_t* begin, const wchar_t* end);

    operator wchar_t const*() const { return begin(); }

    wchar_t const* begin() const;
    wchar_t const* end() const;

private:
    void _releaseBuffer();

    wchar_t* m_begin; ///< First character. This is a new wchar_t[] buffer
    wchar_t* m_end;   ///< Points to terminating 0
};

/*!
@brief Represents a UTF-8 encoded string.
*/

class SLANG_RT_API String
{
    friend struct StringSlice;
    friend class StringBuilder;

private:
    char* getData() const { return m_buffer ? m_buffer->getData() : (char*)""; }


    void ensureUniqueStorageWithCapacity(Index capacity);

    RefPtr<StringRepresentation> m_buffer;

public:
    explicit String(StringRepresentation* buffer)
        : m_buffer(buffer)
    {
    }

    static String fromWString(const wchar_t* wstr);
    static String fromWString(const wchar_t* wstr, const wchar_t* wend);
    static String fromWChar(const wchar_t ch);
    static String fromUnicodePoint(Char32 codePoint);

    String() {}

    /// Returns a buffer which can hold at least count chars
    char* prepareForAppend(Index count);
    /// Append data written to buffer output via 'prepareForAppend' directly written 'inplace'
    void appendInPlace(const char* chars, Index count);

    /// Get the internal string represenation
    SLANG_FORCE_INLINE StringRepresentation* getStringRepresentation() const { return m_buffer; }

    /// Detach the representation (will leave string as empty). Rep ref count will remain unchanged.
    SLANG_FORCE_INLINE StringRepresentation* detachStringRepresentation()
    {
        return m_buffer.detach();
    }

    const char* begin() const { return getData(); }
    const char* end() const { return getData() + getLength(); }

    void append(int32_t value, int radix = 10);
    void append(uint32_t value, int radix = 10);
    void append(int64_t value, int radix = 10);
    void append(uint64_t value, int radix = 10);
    void append(float val, const char* format = "%g");
    void append(double val, const char* format = "%g");

    // Padded hex representations
    void append(StableHashCode32 val);
    void append(StableHashCode64 val);

    void append(char const* str);
    void append(char const* str, size_t len);
    void append(const char* textBegin, char const* textEnd);
    void append(char chr);
    void append(String const& str);
    void append(StringSlice const& slice);
    void append(UnownedStringSlice const& slice);

    /// Append a character (to remove ambiguity with other integral types)
    void appendChar(char chr);

    /// Append the specified char count times
    void appendRepeatedChar(char chr, Index count);

    String(const char* str) { append(str); }
    String(const char* textBegin, char const* textEnd) { append(textBegin, textEnd); }

    // Make all String ctors from a numeric explicit, to avoid unexpected/unnecessary conversions
    explicit String(int32_t val, int radix = 10) { append(val, radix); }
    explicit String(uint32_t val, int radix = 10) { append(val, radix); }
    explicit String(int64_t val, int radix = 10) { append(val, radix); }
    explicit String(uint64_t val, int radix = 10) { append(val, radix); }
    explicit String(StableHashCode32 val) { append(val); }
    explicit String(StableHashCode64 val) { append(val); }
    explicit String(float val, const char* format = "%g") { append(val, format); }
    explicit String(double val, const char* format = "%g") { append(val, format); }

    explicit String(char chr) { appendChar(chr); }
    String(String const& str) { m_buffer = str.m_buffer; }
    String(String&& other) { m_buffer = _Move(other.m_buffer); }

    String(StringSlice const& slice) { append(slice); }

    String(UnownedStringSlice const& slice) { append(slice); }

    ~String() { m_buffer.setNull(); }

    String& operator=(const String& str)
    {
        m_buffer = str.m_buffer;
        return *this;
    }
    String& operator=(String&& other)
    {
        m_buffer = _Move(other.m_buffer);
        return *this;
    }
    char operator[](Index id) const
    {
        SLANG_ASSERT(id >= 0 && id < getLength());
        // Silence a pedantic warning on GCC
#if __GNUC__
        if (id < 0)
            __builtin_unreachable();
#endif
        return begin()[id];
    }

    Index getLength() const { return m_buffer ? m_buffer->getLength() : 0; }
    /// Make the length of the string the amount specified. Must be less than current size
    void reduceLength(Index length);

    friend String operator+(const char* op1, const String& op2);
    friend String operator+(const String& op1, const char* op2);
    friend String operator+(const String& op1, const String& op2);

    StringSlice trimStart() const
    {
        if (!m_buffer)
            return StringSlice();
        Index startIndex = 0;
        const char* const data = getData();
        while (startIndex < getLength() && (data[startIndex] == ' ' || data[startIndex] == '\t' ||
                                            data[startIndex] == '\r' || data[startIndex] == '\n'))
            startIndex++;
        return StringSlice(m_buffer, startIndex, getLength());
    }

    StringSlice trimEnd() const
    {
        if (!m_buffer)
            return StringSlice();

        Index endIndex = getLength();
        const char* const data = getData();
        while (endIndex > 0 && (data[endIndex - 1] == ' ' || data[endIndex - 1] == '\t' ||
                                data[endIndex - 1] == '\r' || data[endIndex - 1] == '\n'))
            endIndex--;

        return StringSlice(m_buffer, 0, endIndex);
    }

    StringSlice trim() const
    {
        if (!m_buffer)
            return StringSlice();

        Index startIndex = 0;
        const char* const data = getData();
        while (startIndex < getLength() && (data[startIndex] == ' ' || data[startIndex] == '\t' ||
                                            data[startIndex] == '\r' || data[startIndex] == '\n'))
            startIndex++;
        Index endIndex = getLength();
        while (endIndex > startIndex && (data[endIndex - 1] == ' ' || data[endIndex - 1] == '\t' ||
                                         data[endIndex - 1] == '\r' || data[endIndex - 1] == '\n'))
            endIndex--;

        return StringSlice(m_buffer, startIndex, endIndex);
    }

    StringSlice subString(Index id, Index len) const
    {
        if (len == 0)
            return StringSlice();

        if (id + len > getLength())
            len = getLength() - id;
#if _DEBUG
        if (id < 0 || id >= getLength() || (id + len) > getLength())
            SLANG_ASSERT_FAILURE("SubString: index out of range.");
        if (len < 0)
            SLANG_ASSERT_FAILURE("SubString: length less than zero.");
#endif
        return StringSlice(m_buffer, id, id + len);
    }

    char const* getBuffer() const { return getData(); }

    OSString toWString(Index* len = 0) const;

    bool equals(const String& str, bool caseSensitive = true)
    {
        if (caseSensitive)
            return (strcmp(begin(), str.begin()) == 0);
        else
        {
#ifdef _MSC_VER
            return (_stricmp(begin(), str.begin()) == 0);
#else
            return (strcasecmp(begin(), str.begin()) == 0);
#endif
        }
    }
    bool operator==(const char* strbuffer) const { return (strcmp(begin(), strbuffer) == 0); }

    bool operator==(const String& str) const { return (strcmp(begin(), str.begin()) == 0); }
    bool operator!=(const char* strbuffer) const { return (strcmp(begin(), strbuffer) != 0); }
    bool operator!=(const String& str) const { return (strcmp(begin(), str.begin()) != 0); }
    bool operator>(const String& str) const { return (strcmp(begin(), str.begin()) > 0); }
    bool operator<(const String& str) const { return (strcmp(begin(), str.begin()) < 0); }
    bool operator>=(const String& str) const { return (strcmp(begin(), str.begin()) >= 0); }
    bool operator<=(const String& str) const { return (strcmp(begin(), str.begin()) <= 0); }

    SLANG_FORCE_INLINE bool operator==(const UnownedStringSlice& slice) const
    {
        return getUnownedSlice() == slice;
    }
    SLANG_FORCE_INLINE bool operator!=(const UnownedStringSlice& slice) const
    {
        return getUnownedSlice() != slice;
    }

    String toUpper() const
    {
        String result;
        for (auto c : *this)
        {
            char d = (c >= 'a' && c <= 'z') ? (c - ('a' - 'A')) : c;
            result.append(d);
        }
        return result;
    }

    String toLower() const
    {
        String result;
        for (auto c : *this)
        {
            char d = (c >= 'A' && c <= 'Z') ? (c - ('A' - 'a')) : c;
            result.append(d);
        }
        return result;
    }

    Index indexOf(const char* str, Index id) const // String str
    {
        if (id >= getLength())
            return Index(-1);
        auto findRs = strstr(begin() + id, str);
        Index res = findRs ? findRs - begin() : Index(-1);
        return res;
    }

    Index indexOf(const String& str, Index id) const { return indexOf(str.begin(), id); }

    Index indexOf(const char* str) const { return indexOf(str, 0); }

    Index indexOf(const String& str) const { return indexOf(str.begin(), 0); }

    void swapWith(String& other) { m_buffer.swapWith(other.m_buffer); }

    Index indexOf(char ch, Index id) const
    {
        const Index length = getLength();
        SLANG_ASSERT(id >= 0 && id <= length);

        if (!m_buffer)
            return Index(-1);

        const char* data = getData();
        for (Index i = id; i < length; i++)
            if (data[i] == ch)
                return i;
        return Index(-1);
    }

    Index indexOf(char ch) const { return indexOf(ch, 0); }

    Index lastIndexOf(char ch) const
    {
        const Index length = getLength();
        const char* data = getData();

        for (Index i = length - 1; i >= 0; --i)
            if (data[i] == ch)
                return i;
        return Index(-1);
    }

    bool startsWith(const char* str) const
    {
        if (!m_buffer)
            return false;
        Index strLen = Index(::strlen(str));
        if (strLen > getLength())
            return false;

        const char* const data = getData();

        for (Index i = 0; i < strLen; i++)
            if (str[i] != data[i])
                return false;
        return true;
    }

    bool startsWith(const String& str) const { return startsWith(str.begin()); }

    bool endsWith(char const* str) const // String str
    {
        if (!m_buffer)
            return false;

        const Index strLen = Index(::strlen(str));
        const Index len = getLength();

        if (strLen > len)
            return false;
        const char* data = getData();
        for (Index i = strLen; i > 0; i--)
            if (str[i - 1] != data[len - strLen + i - 1])
                return false;
        return true;
    }

    bool endsWith(const String& str) const { return endsWith(str.begin()); }

    bool contains(const char* str) const // String str
    {
        return m_buffer && indexOf(str) != Index(-1);
    }

    bool contains(const String& str) const { return contains(str.begin()); }

    static constexpr bool kHasUniformHash = true;
    HashCode64 getHashCode() const
    {
        return Slang::getHashCode(StringRepresentation::asSlice(m_buffer));
    }

    UnownedStringSlice getUnownedSlice() const { return StringRepresentation::asSlice(m_buffer); }
};

class ImmutableHashedString
{
public:
    String slice;
    HashCode64 hashCode;
    ImmutableHashedString()
        : hashCode(0)
    {
    }
    ImmutableHashedString(const UnownedStringSlice& slice)
        : slice(slice), hashCode(slice.getHashCode())
    {
    }
    ImmutableHashedString(const char* begin, const char* end)
        : slice(begin, end), hashCode(slice.getHashCode())
    {
    }
    ImmutableHashedString(const char* begin, size_t len)
        : slice(UnownedStringSlice(begin, len)), hashCode(slice.getHashCode())
    {
    }
    ImmutableHashedString(const char* begin)
        : slice(begin), hashCode(slice.getHashCode())
    {
    }
    ImmutableHashedString(const String& str)
        : slice(str), hashCode(str.getHashCode())
    {
    }
    ImmutableHashedString(String&& str)
        : slice(_Move(str)), hashCode(str.getHashCode())
    {
    }
    ImmutableHashedString(const ImmutableHashedString& other) = default;
    ImmutableHashedString& operator=(const ImmutableHashedString& other) = default;
    bool operator==(const ImmutableHashedString& other) const
    {
        return hashCode == other.hashCode && slice == other.slice;
    }
    bool operator!=(const ImmutableHashedString& other) const
    {
        return hashCode != other.hashCode || slice != other.slice;
    }
    bool operator==(const UnownedStringSlice& other) const { return slice == other; }
    bool operator!=(const UnownedStringSlice& other) const { return slice != other; }
    bool operator==(const String& other) const { return slice == other.getUnownedSlice(); }
    bool operator!=(const String& other) const { return slice != other.getUnownedSlice(); }
    bool operator==(const char* other) const { return slice == UnownedStringSlice(other); }
    HashCode64 getHashCode() const { return hashCode; }
};

class SLANG_RT_API StringBuilder : public String
{
private:
    enum
    {
        InitialSize = 1024
    };

public:
    typedef String Super;
    using Super::append;

    explicit StringBuilder(UInt bufferSize = InitialSize)
    {
        ensureUniqueStorageWithCapacity(bufferSize);
    }

    void ensureCapacity(UInt size) { ensureUniqueStorageWithCapacity(size); }
    StringBuilder& operator<<(char ch)
    {
        appendChar(ch);
        return *this;
    }
    StringBuilder& operator<<(Int32 val)
    {
        append(val);
        return *this;
    }
    StringBuilder& operator<<(UInt32 val)
    {
        append(val);
        return *this;
    }
    StringBuilder& operator<<(Int64 val)
    {
        append(val);
        return *this;
    }
    StringBuilder& operator<<(UInt64 val)
    {
        append(val);
        return *this;
    }
    StringBuilder& operator<<(float val)
    {
        append(val);
        return *this;
    }
    StringBuilder& operator<<(double val)
    {
        append(val);
        return *this;
    }
    StringBuilder& operator<<(const char* str)
    {
        append(str, strlen(str));
        return *this;
    }
    StringBuilder& operator<<(const String& str)
    {
        append(str);
        return *this;
    }
    StringBuilder& operator<<(UnownedStringSlice const& str)
    {
        append(str);
        return *this;
    }
    StringBuilder& operator<<(const _EndLine)
    {
        appendChar('\n');
        return *this;
    }

    String toString() { return *this; }

    String produceString() { return *this; }

#if 0
        void Remove(int id, int len)
        {
#if _DEBUG
            if (id >= length || id < 0)
                SLANG_ASSERT_FAILURE("Remove: Index out of range.");
            if (len < 0)
                SLANG_ASSERT_FAILURE("Remove: remove length smaller than zero.");
#endif
            int actualDelLength = ((id + len) >= length) ? (length - id) : len;
            for (int i = id + actualDelLength; i <= length; i++)
                buffer[i - actualDelLength] = buffer[i];
            length -= actualDelLength;
        }
#endif
    friend std::ostream& operator<<(std::ostream& stream, const String& s);

    void clear() { m_buffer.setNull(); }
};

int stringToInt(const String& str, int radix = 10);
unsigned int stringToUInt(const String& str, int radix = 10);
double stringToDouble(const String& str);
float stringToFloat(const String& str);
} // namespace Slang

std::ostream& operator<<(std::ostream& stream, const Slang::String& s);

#endif
