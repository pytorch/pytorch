// slang-hex-dump-util.cpp
#include "slang-hex-dump-util.h"

#include "slang-char-util.h"
#include "slang-com-helper.h"
#include "slang-common.h"
#include "slang-hash.h"
#include "slang-string-util.h"
#include "slang-writer.h"

namespace Slang
{

static const UnownedStringSlice s_start = UnownedStringSlice::fromLiteral("--START--");
static const UnownedStringSlice s_end = UnownedStringSlice::fromLiteral("--END--");
static const char s_hex[] = "0123456789abcdef";

/* static */ SlangResult HexDumpUtil::dumpWithMarkers(
    const List<uint8_t>& data,
    int maxBytesPerLine,
    ISlangWriter* writer)
{
    return dumpWithMarkers(data.getBuffer(), data.getCount(), maxBytesPerLine, writer);
}

/* static */ SlangResult HexDumpUtil::dumpWithMarkers(
    const uint8_t* data,
    size_t dataCount,
    int maxBytesPerLine,
    ISlangWriter* writer)
{
    WriterHelper helper(writer);
    SLANG_RETURN_ON_FAIL(helper.write(s_start.begin(), s_start.getLength()));
    SLANG_RETURN_ON_FAIL(helper.print(" %zu", dataCount));

    const StableHashCode32 hash = getStableHashCode32((const char*)data, dataCount);
    SLANG_RETURN_ON_FAIL(helper.print(" %d\n", hash.hash));

    SLANG_RETURN_ON_FAIL(dump(data, dataCount, maxBytesPerLine, writer));

    SLANG_RETURN_ON_FAIL(helper.write(s_end.begin(), s_end.getLength()));
    SLANG_RETURN_ON_FAIL(helper.put("\n"));
    return SLANG_OK;
}

/* static */ void HexDumpUtil::dump(uint32_t value, ISlangWriter* writer)
{
    char c[9];
    for (int i = 0; i < 8; ++i)
    {
        c[i] = s_hex[value >> 28];
        value <<= 4;
    }
    writer->write(c, 8);
}


/* static */ SlangResult HexDumpUtil::dump(
    const List<uint8_t>& data,
    int maxBytesPerLine,
    ISlangWriter* writer)
{
    return dump(data.getBuffer(), data.getCount(), maxBytesPerLine, writer);
}

SlangResult HexDumpUtil::dumpSourceBytes(
    const uint8_t* data,
    size_t dataCount,
    int maxBytesPerLine,
    ISlangWriter* writer)
{
    const uint8_t* cur = data;
    const uint8_t* end = data + dataCount;

    while (cur < end)
    {
        size_t count = size_t(end - cur);
        count = (count > size_t(maxBytesPerLine)) ? size_t(maxBytesPerLine) : count;

        // each byte is output as "0xAA, "
        // Ends with '\n"
        const size_t lineBytes = count * 6 + 1;

        char* startDst = writer->beginAppendBuffer(lineBytes);
        char* dst = startDst;

        for (size_t i = 0; i < count; ++i)
        {
            uint8_t byte = cur[i];
            dst[0] = '0';
            dst[1] = 'x';
            dst[2] = s_hex[byte >> 4];
            dst[3] = s_hex[byte & 0xf];
            dst[4] = ',';
            dst[5] = ' ';

            dst += 6;
        }

        *dst++ = '\n';

        SLANG_RETURN_ON_FAIL(writer->endAppendBuffer(startDst, size_t(dst - startDst)));

        cur += count;
    }

    return SLANG_OK;
}

/* static */ SlangResult HexDumpUtil::dump(
    const uint8_t* data,
    size_t dataCount,
    int maxBytesPerLine,
    ISlangWriter* writer)
{
    int maxCharsPerLine = 2 * maxBytesPerLine + 1 + maxBytesPerLine + 1;

    const uint8_t* cur = data;
    const uint8_t* end = data + dataCount;

    while (cur < end)
    {
        size_t count = size_t(end - cur);
        count = (count > size_t(maxBytesPerLine)) ? size_t(maxBytesPerLine) : count;

        char* startDst = writer->beginAppendBuffer(maxCharsPerLine);
        char* dst = startDst;

        for (size_t i = 0; i < count; ++i)
        {
            uint8_t byte = cur[i];
            *dst++ = s_hex[byte >> 4];
            *dst++ = s_hex[byte & 0xf];
        }

        // If not a complete line write spaces
        for (size_t i = count; i < size_t(maxBytesPerLine); ++i)
        {
            *dst++ = ' ';
            *dst++ = ' ';
        }

        *dst++ = ' ';

        for (size_t i = 0; i < count; ++i)
        {
            char c = char(cur[i]);

            if ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                (c >= 32 && (c & 0x80) == 0))
            {
            }
            else
            {
                c = '.';
            }
            *dst++ = c;
        }

        *dst++ = '\n';
        SLANG_ASSERT(dst <= startDst + maxCharsPerLine);

        SLANG_RETURN_ON_FAIL(writer->endAppendBuffer(startDst, size_t(dst - startDst)));

        cur += count;
    }

    return SLANG_OK;
}

/* static */ SlangResult HexDumpUtil::parse(
    const UnownedStringSlice& lines,
    List<uint8_t>& outBytes)
{
    outBytes.clear();

    LineParser lineParser(lines);
    for (const auto& line : lineParser)
    {
        const char* cur = line.begin();
        const char* end = line.end();

        while (cur + 2 <= end)
        {
            const char c = cur[0];
            if (c == ' ' || c == '\n' || c == '\r' || c == '\t')
            {
                // Skip to next line
                break;
            }

            const int hi = CharUtil::getHexDigitValue(c);
            const int lo = CharUtil::getHexDigitValue(cur[1]);
            cur += 2;

            if (hi < 0 || lo < 0)
            {
                return SLANG_FAIL;
            }
            outBytes.add(uint8_t((hi << 4) | lo));
        }
    }

    return SLANG_OK;
}

static SlangResult _findLine(
    const UnownedStringSlice& find,
    UnownedStringSlice& ioRemaining,
    UnownedStringSlice& outLine)
{
    // Find the start line
    UnownedStringSlice line;
    while (StringUtil::extractLine(ioRemaining, line))
    {
        if (line.startsWith(find))
        {
            outLine = line;
            return SLANG_OK;
        }
    }
    return SLANG_FAIL;
}

/* static */ SlangResult HexDumpUtil::findStartAndEndLines(
    const UnownedStringSlice& lines,
    UnownedStringSlice& outStart,
    UnownedStringSlice& outEnd)
{
    UnownedStringSlice remaining(lines);
    SLANG_RETURN_ON_FAIL(_findLine(s_start, remaining, outStart));
    SLANG_RETURN_ON_FAIL(_findLine(s_end, remaining, outEnd));
    return SLANG_OK;
}

/* static */ SlangResult HexDumpUtil::parseWithMarkers(
    const UnownedStringSlice& lines,
    List<uint8_t>& outBytes)
{
    UnownedStringSlice startLine, endLine;
    SLANG_RETURN_ON_FAIL(findStartAndEndLines(lines, startLine, endLine));

    StableHashCode32 hash;
    size_t size;
    {
        // Get the size and the hash
        List<UnownedStringSlice> slices;
        StringUtil::split(startLine, ' ', slices);
        if (slices.getCount() != 3)
        {
            return SLANG_FAIL;
        }
        // Extract the size
        size = stringToInt(String(slices[1]));
        hash = StableHashCode32{stringToUInt(String(slices[2]))};
    }

    SLANG_RETURN_ON_FAIL(parse(UnownedStringSlice(startLine.end(), endLine.begin()), outBytes));

    // Calc the hash
    const StableHashCode32 readHash =
        getStableHashCode32((const char*)outBytes.begin(), outBytes.getCount());

    if (readHash != hash || size_t(outBytes.getCount()) != size)
    {
        return SLANG_FAIL;
    }
    return SLANG_OK;
}

} // namespace Slang
