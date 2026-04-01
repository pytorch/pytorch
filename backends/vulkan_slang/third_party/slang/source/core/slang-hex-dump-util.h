#ifndef SLANG_HEX_DUMP_UTIL_H
#define SLANG_HEX_DUMP_UTIL_H

#include "slang-common.h"
#include "slang-list.h"
#include "slang-string.h"
#include "slang.h"

namespace Slang
{

struct HexDumpUtil
{
    /// Dump out bytes in source format - as in
    /// 0x10, 0xab,
    static SlangResult dumpSourceBytes(
        const uint8_t* data,
        size_t dataCount,
        int maxBytesPerLine,
        ISlangWriter* writer);

    /// Dump data to writer, with lines starting with hex data
    static SlangResult dump(const List<uint8_t>& data, int numBytesPerLine, ISlangWriter* writer);

    static SlangResult dump(
        const uint8_t* data,
        size_t dataCount,
        int numBytesPerLine,
        ISlangWriter* writer);

    /// Dump a single value
    static void dump(uint32_t value, ISlangWriter* writer);

    static SlangResult dumpWithMarkers(
        const List<uint8_t>& data,
        int numBytesPerLine,
        ISlangWriter* writer);

    static SlangResult dumpWithMarkers(
        const uint8_t* data,
        size_t dataSize,
        int numBytesPerLine,
        ISlangWriter* writer);

    /// Parses lines formatted by dump, back into bytes
    static SlangResult parse(const UnownedStringSlice& lines, List<uint8_t>& outBytes);

    static SlangResult parseWithMarkers(const UnownedStringSlice& lines, List<uint8_t>& outBytes);

    static SlangResult findStartAndEndLines(
        const UnownedStringSlice& lines,
        UnownedStringSlice& outStart,
        UnownedStringSlice& outEnd);
};

} // namespace Slang

#endif
