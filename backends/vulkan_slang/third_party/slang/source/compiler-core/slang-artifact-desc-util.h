// slang-artifact-desc.h

#ifndef SLANG_ARTIFACT_DESC_UTIL_H
#define SLANG_ARTIFACT_DESC_UTIL_H

#include "slang-artifact.h"

namespace Slang
{

/// Get the parent kind
ArtifactKind getParent(ArtifactKind kind);
/// Returns true if kind is derived from base
bool isDerivedFrom(ArtifactKind kind, ArtifactKind base);
/// Get the name for the kind
UnownedStringSlice getName(ArtifactKind kind);

/// Get the parent payload
ArtifactPayload getParent(ArtifactPayload payload);
/// Returns true if payload is derived from base
bool isDerivedFrom(ArtifactPayload payload, ArtifactPayload base);
/// Get the name for the payload
UnownedStringSlice getName(ArtifactPayload payload);

/// Get the parent style
ArtifactStyle getParent(ArtifactStyle style);
/// Returns true if style is derived from base
bool isDerivedFrom(ArtifactStyle style, ArtifactStyle base);
/// Get the name for the style
UnownedStringSlice getName(ArtifactStyle style);

struct ArtifactDescUtil
{
    typedef ArtifactPayload Payload;
    typedef ArtifactKind Kind;
    typedef ArtifactStyle Style;
    typedef ArtifactDesc Desc;

    /// Returns true if the kind is binary linkable
    static bool isKindBinaryLinkable(Kind kind);

    /// True if is a CPU target - either
    static bool isCpuLikeTarget(const ArtifactDesc& desc);

    /// True if is a CPU binary
    static bool isCpuBinary(const ArtifactDesc& desc);
    /// True if is a GPU usable (can be passed to a driver/API and be used)
    static bool isGpuUsable(const ArtifactDesc& desc);

    /// True if the desc holds textual information
    static bool isText(const ArtifactDesc& desc);

    /// True if artifact  appears to be linkable
    static bool isLinkable(const ArtifactDesc& desc);

    /// Try to determine the desc from just a file extension (passed without .)
    static ArtifactDesc getDescFromExtension(const UnownedStringSlice& slice);

    /// Try to determine the desc from a path
    static ArtifactDesc getDescFromPath(const UnownedStringSlice& slice);

    /// Appends the default file extension for the artifact type.
    static SlangResult appendDefaultExtension(const ArtifactDesc& desc, StringBuilder& out);

    /// Get the extension for CPU/Host for a kind
    static SlangResult appendCpuExtensionForKind(Kind kind, StringBuilder& out);

    /// Given a desc and a path returns the base name (stripped of prefix and extension)
    static String getBaseNameFromPath(const ArtifactDesc& desc, const UnownedStringSlice& path);

    /// Given a desc and a name returns the base name (stripped of prefix and extension)
    static String getBaseNameFromName(const ArtifactDesc& desc, const UnownedStringSlice& path);

    /// Get the base name of the fileRep
    /// If no base name is found will return an empty slice
    static String getBaseName(const ArtifactDesc& desc, IPathArtifactRepresentation* fileRep);

    /// Given a desc and a basePath returns a suitable path for a entity of specified desc
    static SlangResult calcPathForDesc(
        const ArtifactDesc& desc,
        const UnownedStringSlice& basePath,
        StringBuilder& outPath);

    /// Given a desc and a baseName works out the the output file name
    static SlangResult calcNameForDesc(
        const ArtifactDesc& desc,
        const UnownedStringSlice& baseName,
        StringBuilder& outName);

    /// Returns true if there is a defined name extension/type for this desc
    static SlangResult hasDefinedNameForDesc(const ArtifactDesc& desc);

    /// Given a target returns the ArtifactDesc
    static ArtifactDesc makeDescForCompileTarget(SlangCompileTarget target);

    /// Get the payload for the specified language
    static ArtifactPayload getPayloadForSourceLanaguage(SlangSourceLanguage language);
    /// Given a source language return a desc
    static ArtifactDesc makeDescForSourceLanguage(SlangSourceLanguage language);

    /// Returns the closest compile target for desc. Will return
    /// SLANG_TARGET_UNKNOWN if not known
    static SlangCompileTarget getCompileTargetFromDesc(const ArtifactDesc& desc);

    /// Make ArtifactDesc from target
    static bool isDescDerivedFrom(const ArtifactDesc& desc, const ArtifactDesc& from);

    /// True if `to` is disassembly of `from`
    static bool isDisassembly(const ArtifactDesc& from, const ArtifactDesc& to);

    /// Append the desc as text to out
    static void appendText(const ArtifactDesc& desc, StringBuilder& out);

    /// Given an artifact desc return a description as a string
    static String getText(const ArtifactDesc& desc);
};

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
inline /* static */ bool ArtifactDescUtil::isDescDerivedFrom(
    const ArtifactDesc& desc,
    const ArtifactDesc& from)
{
    // TODO(JS): Currently this ignores flags in desc. That may or may not be right
    // long term.
    return isDerivedFrom(desc.kind, from.kind) && isDerivedFrom(desc.payload, from.payload) &&
           isDerivedFrom(desc.style, from.style);
}

} // namespace Slang

#endif
