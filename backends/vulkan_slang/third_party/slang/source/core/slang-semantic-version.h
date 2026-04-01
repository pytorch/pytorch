// slang-semantic-version.h
#ifndef SLANG_SEMANTIC_VERSION_H
#define SLANG_SEMANTIC_VERSION_H

#include "../core/slang-basic.h"
#include "../core/slang-hash.h"

namespace Slang
{

struct SemanticVersion
{
    typedef SemanticVersion ThisType;

    typedef uint64_t IntegerType;

    SemanticVersion()
        : m_major(0), m_minor(0), m_patch(0)
    {
    }

    SemanticVersion(int inMajor, int inMinor = 0, int inPatch = 0)
        : m_major(uint16_t(inMajor)), m_minor(uint16_t(inMinor)), m_patch(uint32_t(inPatch))
    {
    }

    void reset()
    {
        m_major = 0;
        m_minor = 0;
        m_patch = 0;
    }

    /// All zeros means nothing is set
    bool isSet() const { return m_major || m_minor || m_patch; }

    IntegerType toInteger() const
    {
        return (IntegerType(m_major) << 48) | (IntegerType(m_minor) << 32) | m_patch;
    }
    void setFromInteger(IntegerType v)
    {
        set(int(v >> 48), int((v >> 32) & 0xffff), int(v & 0xffffffff));
    }
    void set(int major, int minor, int patch = 0)
    {
        SLANG_ASSERT(major >= 0 && minor >= 0 && patch >= 0);

        m_major = uint16_t(major);
        m_minor = uint16_t(minor);
        m_patch = uint32_t(patch);
    }

    /// Get hash value
    HashCode getHashCode() const { return Slang::getHashCode(toInteger()); }

    static SlangResult parse(const UnownedStringSlice& value, SemanticVersion& outVersion);
    static SlangResult parse(
        const UnownedStringSlice& value,
        char separatorChar,
        SemanticVersion& outVersion);

    static ThisType getEarliest(const ThisType* versions, Count count);
    static ThisType getLatest(const ThisType* versions, Count count);

    void append(StringBuilder& buf) const;

    bool operator>(const ThisType& rhs) const { return toInteger() > rhs.toInteger(); }
    bool operator>=(const ThisType& rhs) const { return toInteger() >= rhs.toInteger(); }

    bool operator<(const ThisType& rhs) const { return toInteger() < rhs.toInteger(); }
    bool operator<=(const ThisType& rhs) const { return toInteger() <= rhs.toInteger(); }

    bool operator==(const ThisType& rhs) const { return toInteger() == rhs.toInteger(); }
    bool operator!=(const ThisType& rhs) const { return toInteger() != rhs.toInteger(); }

    uint16_t m_major;
    uint16_t m_minor;
    uint32_t m_patch; ///< Patch number. Can actually be quite large for some code bases.
};

/* Adds to the semantic versioning information for an incomplete version that can be matched */
struct MatchSemanticVersion
{
    typedef MatchSemanticVersion ThisType;

    enum class Kind
    {
        Unknown,         ///< Not known
        Past,            ///< Some unknown past version
        Future,          ///< Some future unknown version
        Major,           ///< Major version is defined (minor is in effect undefined)
        MajorMinor,      ///< Major and minor version are defined
        MajorMinorPatch, ///< All elements of semantic version are defined
    };

    /// True if has a complete version
    bool hasCompleteVersion() const { return m_kind == Kind::MajorMinorPatch; }
    /// True if has some version information
    bool hasVersion() const { return Index(m_kind) >= Index(Kind::Major); }

    void set(Index major)
    {
        m_kind = Kind::Major;
        m_version = SemanticVersion(int(major), 0, 0);
    }
    void set(Index major, Index minor)
    {
        m_kind = Kind::MajorMinor;
        m_version = SemanticVersion(int(major), int(minor), 0);
    }
    void set(Index major, Index minor, Index patch)
    {
        m_kind = Kind::MajorMinorPatch;
        m_version = SemanticVersion(int(major), int(minor), int(patch));
    }

    void append(StringBuilder& buf) const;

    static MatchSemanticVersion makeFuture()
    {
        MatchSemanticVersion version;
        version.m_kind = Kind::Future;
        return version;
    }

    /// Finds the 'best' version based on the versions passed.
    /// Doesn't follow strict semantic rules as will attempt to return the closest 'any' in past or
    /// future If none can be found, returns an empty semantic version
    static SemanticVersion findAnyBest(
        const SemanticVersion* versions,
        Count count,
        const ThisType& matchVersion);

    MatchSemanticVersion()
        : m_kind(Kind::Unknown)
    {
    }
    MatchSemanticVersion(Kind kind, const SemanticVersion& version)
        : m_kind(kind), m_version(version)
    {
    }

    Kind m_kind;
    SemanticVersion m_version;
};

} // namespace Slang
#endif
