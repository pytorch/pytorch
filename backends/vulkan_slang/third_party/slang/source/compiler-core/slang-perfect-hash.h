#pragma once

#include "../core/slang-list.h"
#include "../core/slang-string.h"

namespace Slang
{

struct HashParams
{
    List<UInt32> saltTable;
    List<String> destTable;
};

enum class HashFindResult
{
    Success,
    NonUniqueKeys,
    UnavoidableHashCollision,
};

// Calculate a minimal perfect hash of a list of input strings
HashFindResult minimalPerfectHash(const List<String>& ss, HashParams& hashParams);

String perfectHashToEmbeddableCpp(
    const HashParams& hashParams,
    const UnownedStringSlice& valueType,
    const UnownedStringSlice& funcName,
    const List<String>& values);

} // namespace Slang
