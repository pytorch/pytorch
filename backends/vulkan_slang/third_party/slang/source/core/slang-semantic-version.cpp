// slang-semantic-version.cpp
#include "slang-semantic-version.h"

#include "../core/slang-string-util.h"
#include "slang-com-helper.h"

namespace Slang
{

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! SemanticVersion !!!!!!!!!!!!!!!!!!!!!!!!!!!!!

SlangResult SemanticVersion::parse(
    const UnownedStringSlice& value,
    char separatorChar,
    SemanticVersion& outVersion)
{
    outVersion.reset();

    UnownedStringSlice slices[3];
    Index splitCount;
    SLANG_RETURN_ON_FAIL(StringUtil::split(value, separatorChar, 3, slices, splitCount));
    if (splitCount <= 0)
    {
        return SLANG_FAIL;
    }

    Int ints[3] = {0, 0, 0};
    for (Index i = 0; i < splitCount; i++)
    {
        SLANG_RETURN_ON_FAIL(StringUtil::parseInt(slices[i], ints[i]));

        const Int max = (i == 2) ? 0x7fffffff : 0xffff;
        if (ints[i] < 0 || ints[i] > max)
        {
            return SLANG_FAIL;
        }
    }

    outVersion.m_major = uint16_t(ints[0]);
    outVersion.m_minor = uint16_t(ints[1]);
    outVersion.m_patch = uint32_t(ints[2]);

    return SLANG_OK;
}

SlangResult SemanticVersion::parse(const UnownedStringSlice& value, SemanticVersion& outVersion)
{
    return parse(value, '.', outVersion);
}

void SemanticVersion::append(StringBuilder& buf) const
{
    buf << Int32(m_major) << "." << Int32(m_minor);
    if (m_patch != 0)
    {
        buf << "." << UInt32(m_patch);
    }
}

/* static */ SemanticVersion SemanticVersion::getEarliest(const ThisType* versions, Count count)
{
    if (count <= 0)
    {
        return SemanticVersion();
    }

    SemanticVersion bestVersion = versions[0];
    for (const auto version : makeConstArrayView(versions + 1, count - 1))
    {
        if (version < bestVersion)
        {
            bestVersion = version;
        }
    }
    return bestVersion;
}

/* static */ SemanticVersion SemanticVersion::getLatest(const ThisType* versions, Count count)
{
    if (count <= 0)
    {
        return SemanticVersion();
    }

    SemanticVersion bestVersion = versions[0];
    for (const auto version : makeConstArrayView(versions + 1, count - 1))
    {
        if (version > bestVersion)
        {
            bestVersion = version;
        }
    }
    return bestVersion;
}

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! MatchSemanticVersion !!!!!!!!!!!!!!!!!!!!!!!!!!!!!

/* static */ SemanticVersion MatchSemanticVersion::findAnyBest(
    const SemanticVersion* versions,
    Count count,
    const ThisType& matchVersion)
{
    // If there aren't any we are done
    if (count <= 0)
    {
        return SemanticVersion();
    }

    // If there is only one it must be the best
    if (count == 1)
    {
        return versions[0];
    }

    // Define a version range [start, end)
    SemanticVersion start, end;

    switch (matchVersion.m_kind)
    {
    case Kind::Past:
        {
            return SemanticVersion::getEarliest(versions, count);
        }
    case Kind::Unknown:
    case Kind::Future:
        {
            // If it's unknown, we just get the latest
            return SemanticVersion::getLatest(versions, count);
        }
    case Kind::Major:
        {
            start = SemanticVersion(matchVersion.m_version.m_major, 0, 0);
            end = SemanticVersion(matchVersion.m_version.m_major + 1, 0, 0);
            break;
        }
    case Kind::MajorMinor:
        {
            start =
                SemanticVersion(matchVersion.m_version.m_major, matchVersion.m_version.m_minor, 0);
            end = SemanticVersion(
                matchVersion.m_version.m_major,
                matchVersion.m_version.m_minor + 1,
                0);
            break;
        }
    case Kind::MajorMinorPatch:
        {
            start = SemanticVersion(matchVersion.m_version);
            end = SemanticVersion(
                matchVersion.m_version.m_major,
                matchVersion.m_version.m_minor,
                matchVersion.m_version.m_patch + 1);
            break;
        }
    default:
        break;
    }

    List<SemanticVersion> sortedVersions;
    sortedVersions.addRange(versions, count);

    // Sort into increasing values
    sortedVersions.sort(
        [&](const SemanticVersion& a, const SemanticVersion& b) -> bool { return a < b; });

    Index startIndex = 0;
    for (; startIndex < count && sortedVersions[startIndex] < start; ++startIndex)
        ;

    Index endIndex = startIndex;
    for (; endIndex < count && sortedVersions[endIndex] < end; ++endIndex)
        ;

    // If we have a span of versions, get the last in the span
    if (startIndex < endIndex)
    {
        // Get the last one
        return sortedVersions[endIndex - 1];
    }

    // Get the next greatest if there is one
    if (endIndex < count)
    {
        return sortedVersions[endIndex];
    }

    // Get the prior prior to the start
    if (startIndex > 0)
    {
        return sortedVersions[startIndex - 1];
    }

    // All cases should be covered, but return the last one
    return sortedVersions[count - 1];
}

void MatchSemanticVersion::append(StringBuilder& buf) const
{
    switch (m_kind)
    {
    default:
    case Kind::Unknown:
        buf << "unknown";
        break;
    case Kind::Past:
        buf << "past";
        break;
    case Kind::Future:
        buf << "future";
        break;
    case Kind::Major:
        {
            buf << m_version.m_major;
            break;
        }
    case Kind::MajorMinor:
        {
            buf << m_version.m_major << "." << m_version.m_minor;
            break;
        }
    case Kind::MajorMinorPatch:
        {
            m_version.append(buf);
            break;
        }
    }
}

} // namespace Slang
