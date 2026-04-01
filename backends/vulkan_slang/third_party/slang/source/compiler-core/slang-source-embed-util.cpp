#include "slang-source-embed-util.h"

// Artifact
#include "../compiler-core/slang-artifact-desc-util.h"
#include "../compiler-core/slang-artifact-util.h"
#include "../core/slang-blob.h"
#include "../core/slang-char-util.h"
#include "../core/slang-io.h"
#include "../core/slang-string-escape-util.h"
#include "../core/slang-string-util.h"

namespace Slang
{

namespace
{ // anonymous
typedef SourceEmbedUtil::Style Style;
} // namespace

static const NamesDescriptionValue kSourceEmbedStyleInfos[] = {
    {ValueInt(Style::None), "none", "No source level embedding"},
    {ValueInt(Style::Default), "default", "The default embedding for the type to be embedded"},
    {ValueInt(Style::Text),
     "text",
     "Embed as text. May change line endings. If output isn't text will use 'default'. Size will "
     "*not* contain terminating 0."},
    {ValueInt(Style::BinaryText), "binary-text", "Embed as text assuming contents is binary. "},
    {ValueInt(Style::U8), "u8", "Embed as unsigned bytes."},
    {ValueInt(Style::U16), "u16", "Embed as uint16_t."},
    {ValueInt(Style::U32), "u32", "Embed as uint32_t."},
    {ValueInt(Style::U64), "u64", "Embed as uint64_t."},
};

/* static */ ConstArrayView<NamesDescriptionValue> SourceEmbedUtil::getStyleInfos()
{
    return makeConstArrayView(kSourceEmbedStyleInfos);
}

/* static */ bool SourceEmbedUtil::isSupported(SlangSourceLanguage lang)
{
    return lang == SLANG_SOURCE_LANGUAGE_CPP || lang == SLANG_SOURCE_LANGUAGE_C;
}

static bool _isHeaderExtension(const UnownedStringSlice& in)
{
    // Some "typical" header extensions
    return in == toSlice("h") || in == toSlice("hpp") || in == toSlice("hxx") ||
           in == toSlice("h++") || in == toSlice("hh");
}

/* static */ String SourceEmbedUtil::getPath(const String& path, const Options& options)
{
    if (!isSupported(options.language))
    {
        return String();
    }

    if (!path.getLength())
    {
        return path;
    }

    const auto ext = Path::getPathExt(path);

    if (_isHeaderExtension(ext.getUnownedSlice()))
    {
        return path;
    }

    // Assume it's a header, and just use the .h extension
    StringBuilder buf;
    buf << path << toSlice(".h");
    return buf;
}

/* static */ SourceEmbedUtil::Style SourceEmbedUtil::getDefaultStyle(const ArtifactDesc& desc)
{
    if (ArtifactDescUtil::isText(desc))
    {
        return Style::Text;
    }

    if (isDerivedFrom(desc.kind, ArtifactKind::CompileBinary))
    {
        // SPIR-V is encoded as U32
        if (isDerivedFrom(desc.payload, ArtifactPayload::SPIRV))
        {
            return Style::U32;
        }
    }

    // When in doube encode as U8 bytes.
    // The problem is on some compilers there are limits on how long a U8 based binary can be.
    return Style::U8;
}

// True if we need to copy into a buffer. Necessary if there is an alignement
// issue or if there is a partial entry
static bool _needsCopy(const uint8_t* cur, Count bytesPerElement, Count bytesPerLine)
{
    return ((size_t(bytesPerLine) | size_t(cur)) & size_t(bytesPerElement - 1)) != 0;
}

// NOTE! Assumes T is an unsigned type. Behavior will be incorrect if it is not.
template<typename T>
static void _appendHex(
    const T* in,
    ArrayView<char> elementWork,
    char* dst,
    size_t bytesForLine,
    StringBuilder& out)
{
    // Check that T is unsigned
    SLANG_COMPILE_TIME_ASSERT((T(~T(0))) > T(0));

    // Make sure dst seems plausible
    SLANG_ASSERT(dst >= elementWork.begin() && dst <= elementWork.end());
    // Check the alignment
    SLANG_ASSERT((size_t(in) & (sizeof(T) - 1)) == 0);

    // Calculate the amount of elements for this line.
    const size_t elementsCount = (bytesForLine + sizeof(T) - 1) / sizeof(T);

    // The amount of hex digits needed, is 2 per byte
    const Count numHexDigits = sizeof(T) * 2;

    // Shift to get top nybble
    const Index shift = (numHexDigits - 1) * 4;

    for (size_t i = 0; i < elementsCount; ++i)
    {
        T value = in[i];

        for (Index j = 0; j < numHexDigits; j++, value <<= 4)
        {
            dst[j] = CharUtil::getHexChar(Index(value >> shift) & 0xf);
        }

        out.append(elementWork.getBuffer(), elementWork.getCount());
    }
}

static SlangResult _append(
    const SourceEmbedUtil::Options& options,
    ConstArrayView<uint8_t> data,
    StringBuilder& buf)
{
    const uint8_t* cur = data.begin();

    const auto prefix = toSlice("0x");
    const auto suffix = toSlice(", ");
    UnownedStringSlice literalSuffix;

    UnownedStringSlice elementType;

    Count bytesPerElement;

    switch (options.style)
    {
    case Style::U8:
        {
            elementType = toSlice("unsigned char");
            bytesPerElement = 1;
            break;
        }
    case Style::U16:
        {
            elementType = toSlice("uint16_t");
            bytesPerElement = 2;
            break;
        }
    case Style::U32:
        {
            elementType = toSlice("uint32_t");
            bytesPerElement = 4;
            break;
        }
    case Style::U64:
        {
            elementType = toSlice("uint64_t");
            bytesPerElement = 8;
            // On testing on GCC/CLANG/Recent VS, there is no warning/error without suffix, so
            // will leave off for now.
            // literalSuffix = toSlice("ULL");
            break;
        }
    default:
        return SLANG_FAIL;
    }

    // Output the variable

    buf << "const " << elementType << " " << options.variableName << "[] = \n";
    buf << "{\n";

    // Work out the element work
    char work[80];
    Count elementSizeInChars;
    {
        StringBuilder workBuf;
        workBuf << prefix;
        workBuf.appendRepeatedChar('N', 2 * bytesPerElement);
        workBuf << literalSuffix;
        workBuf << suffix;

        elementSizeInChars = workBuf.getLength();
        ::memcpy(work, workBuf.getBuffer(), elementSizeInChars);
    }

    auto workView = makeArrayView(work, elementSizeInChars);
    char* dstChars = work + prefix.getLength();

    Count elementsPerLine = (options.lineLength - options.indent.getLength()) / elementSizeInChars;
    elementsPerLine = (elementsPerLine <= 0) ? 1 : elementsPerLine;

    // Maximum bytes output per line
    const size_t bytesPerLine = elementsPerLine * bytesPerElement;

    List<uint64_t> alignedElements;
    alignedElements.setCount(Count((bytesPerLine / sizeof(uint64_t)) + 2));
    uint8_t* alignedDst = (uint8_t*)alignedElements.getBuffer();

    size_t bytesRemaining = data.getCount();

    while (bytesRemaining > 0)
    {
        const size_t bytesForLine = bytesRemaining > bytesPerLine ? bytesPerLine : bytesRemaining;
        bytesRemaining -= bytesForLine;

        const uint8_t* lineBytes = cur;
        cur += bytesForLine;

        // We copy if we want alignment of if we hit a partial at the end
        if (_needsCopy(lineBytes, bytesPerElement, bytesForLine))
        {
            // Make sure the last element is zeroed, before copying
            // Needed if the last element is partial.
            alignedElements[Index(bytesForLine / sizeof(uint64_t))] = 0;

            // Copy the bytes over
            ::memcpy(alignedDst, lineBytes, bytesForLine);

            // Use the aligned buffer for the line
            lineBytes = alignedDst;
        }

        buf << options.indent;

        switch (bytesPerElement)
        {
        case 1:
            _appendHex<uint8_t>(lineBytes, workView, dstChars, bytesForLine, buf);
            break;
        case 2:
            _appendHex<uint16_t>((const uint16_t*)lineBytes, workView, dstChars, bytesForLine, buf);
            break;
        case 4:
            _appendHex<uint32_t>((const uint32_t*)lineBytes, workView, dstChars, bytesForLine, buf);
            break;
        case 8:
            _appendHex<uint64_t>((const uint64_t*)lineBytes, workView, dstChars, bytesForLine, buf);
            break;
        }

        buf << "\n";
    }

    buf << "};\n\n";

    return SLANG_OK;
}

/* static */ SlangResult SourceEmbedUtil::createEmbedded(
    IArtifact* artifact,
    const Options& inOptions,
    ComPtr<IArtifact>& outArtifact)
{
    if (!isSupported(inOptions.language))
    {
        return SLANG_E_NOT_IMPLEMENTED;
    }

    ComPtr<ISlangBlob> blob;
    SLANG_RETURN_ON_FAIL(artifact->loadBlob(ArtifactKeep::No, blob.writeRef()));

    const auto desc = artifact->getDesc();

    Options options(inOptions);

    // If the style is text, but the artifact *isn't* a text type, we'll
    // use 'default' for the type
    if (options.style == Style::Text && !ArtifactDescUtil::isText(desc))
    {
        options.style = Style::Default;
    }

    if (options.style == Style::Default)
    {
        options.style = getDefaultStyle(desc);
    }

    // If there is no style there is nothing to do
    if (options.style == Style::None)
    {
        return SLANG_OK;
    }

    if (options.variableName.getLength() <= 0)
    {
        options.variableName = "data";
    }

    StringBuilder buf;

    ConstArrayView<uint8_t> data((const uint8_t*)blob->getBufferPointer(), blob->getBufferSize());

    size_t totalSizeInBytes = data.getCount();

    switch (options.style)
    {
    case Style::Text:
        {
            totalSizeInBytes = 0;

            auto handler = StringEscapeUtil::getHandler(StringEscapeUtil::Style::Cpp);

            buf << "const char " << options.variableName << "[] = \n";

            // Split into lines
            // We dont worry about splitting lines in this impl...
            UnownedStringSlice text((const char*)data.begin(), data.getCount());

            for (auto line : LineParser(text))
            {
                buf << options.indent;
                buf << "\"";

                handler->appendEscaped(line, buf);

                // Work out the total size, taking into account we may encode line endings and \0
                // differently The +1 is for \n
                totalSizeInBytes += line.getLength() + 1;

                buf << "\\n\"\n";
            }

            buf << ";\n";
            break;
        }
    case Style::BinaryText:
        {
            auto handler = StringEscapeUtil::getHandler(StringEscapeUtil::Style::Cpp);

            buf << "const char " << options.variableName << "[] = \n";

            // We could encode everything and then split
            // but if we do that we probably want to not split across an escaped character,
            // although that may be handled correctly.

            // The other way to this is incrementally, so that's what we will do here
            UnownedStringSlice text((const char*)data.begin(), data.getCount());

            auto cur = text.begin();
            auto end = text.end();

            while (cur < end)
            {
                const auto startOffset = buf.getLength();

                buf << options.indent;
                buf << "\"";

                do
                {
                    handler->appendEscaped(UnownedStringSlice(cur, 1), buf);
                    cur++;
                } while (buf.getLength() - startOffset < options.lineLength - 1);

                buf << "\"\n";
            }

            buf << ";\n";
            break;
        }
    case Style::U8:
    case Style::U16:
    case Style::U32:
    case Style::U64:
        {
            SLANG_RETURN_ON_FAIL(_append(options, data, buf));
            break;
        }
    default:
        {
            return SLANG_E_NOT_IMPLEMENTED;
        }
    }

    buf << "const size_t " << options.variableName
        << "_sizeInBytes = " << uint64_t(totalSizeInBytes) << ";\n\n";

    // Make into an artifact
    ArtifactPayload payload =
        options.language == SLANG_SOURCE_LANGUAGE_C ? ArtifactPayload::C : ArtifactPayload::Cpp;
    auto dstDesc = ArtifactDesc::make(ArtifactKind::Source, payload);

    auto dstArtifact = ArtifactUtil::createArtifact(dstDesc);

    auto dstBlob = StringBlob::moveCreate(buf);
    dstArtifact->addRepresentationUnknown(dstBlob);

    outArtifact = dstArtifact;
    return SLANG_OK;
}

} // namespace Slang
