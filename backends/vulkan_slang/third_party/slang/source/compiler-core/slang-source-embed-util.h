#ifndef SLANG_SOURCE_EMBED_UTIL_H
#define SLANG_SOURCE_EMBED_UTIL_H

#include "../core/slang-basic.h"
#include "../core/slang-name-value.h"
#include "slang-artifact.h"
#include "slang-com-ptr.h"
#include "slang-diagnostic-sink.h"

namespace Slang
{


struct SourceEmbedUtil
{
    enum class Style : uint32_t
    {
        None,    ///< No embedding
        Default, ///< Default embedding for the type
        Text, ///< Embed as text. May change line endings. If output isn't text will use 'default'.
              ///< Size will *not* contain terminating 0
        BinaryText, ///< Embed as text assuming contents is binary.
        U8,         ///< Embed as unsigned bytes
        U16,        ///< Embed as uint16_t
        U32,        ///< Embed as uint32_t
        U64,        ///< Embed as uint64_t
        CountOf,
    };

    struct Options
    {
        Style style = Style::Default; ///< Style of embedding
        Count lineLength = 120; ///< The line length, lines can be larger for some styles, but will
                                ///< aim to keep within range
        SlangSourceLanguage language = SLANG_SOURCE_LANGUAGE_C; ///< The language to output for
        String variableName;                                    ///< The name to give the variable
        String indent = "    ";                                 ///< Indenting
    };

    /// Get the style infos
    static ConstArrayView<NamesDescriptionValue> getStyleInfos();

    /// Given an artifact and
    static SlangResult createEmbedded(
        IArtifact* artifact,
        const Options& options,
        ComPtr<IArtifact>& outArtifact);

    /// Returns the default style for the desc
    static Style getDefaultStyle(const ArtifactDesc& desc);

    /// Returns true if supports the specified language for embedding
    static bool isSupported(SlangSourceLanguage lang);

    /// Given the path return the output path. If no path is available return the empty string
    static String getPath(const String& path, const Options& options);
};

} // namespace Slang

#endif
