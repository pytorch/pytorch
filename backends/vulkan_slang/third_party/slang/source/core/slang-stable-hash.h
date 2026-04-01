#pragma once

#include <cstdint>
#include <cstring>
#include <type_traits>

namespace Slang
{
//
// Types
//

struct StableHashCode64
{
    uint64_t hash;
    explicit operator uint64_t() const { return hash; }
    bool operator==(StableHashCode64 other) const { return other.hash == hash; };
    bool operator!=(StableHashCode64 other) const { return other.hash != hash; };
};

struct StableHashCode32
{
    uint32_t hash;
    explicit operator uint32_t() const { return hash; }
    bool operator==(StableHashCode32 other) const { return other.hash == hash; };
    bool operator!=(StableHashCode32 other) const { return other.hash != hash; };
};

/* The 'Stable' hash code functions produce hashes that must be

* The same result for the same inputs on all targets
* Rarely change - as their values can change the output of the Slang API/Serialization

Hash value used from the 'Stable' functions can also be used as part of serialization -
so it is in effect part of the API.

In effect this means changing a 'Stable' algorithm will typically require doing a new release.
*/
inline StableHashCode64 getStableHashCode64(const char* buffer, size_t numChars)
{
    uint64_t hash = 0;
    for (size_t i = 0; i < numChars; ++i)
    {
        hash = uint64_t(buffer[i]) + (hash << 6) + (hash << 16) - hash;
    }
    return StableHashCode64{hash};
}

template<typename T>
inline StableHashCode64 getStableHashCode64(const T& t)
{
    static_assert(std::has_unique_object_representations_v<T>);
    return getStableHashCode64(reinterpret_cast<const char*>(&t), sizeof(T));
}

inline StableHashCode32 getStableHashCode32(const char* buffer, size_t numChars)
{
    uint32_t hash = 0;
    for (size_t i = 0; i < numChars; ++i)
    {
        hash = uint32_t(buffer[i]) + (hash << 6) + (hash << 16) - hash;
    }
    return StableHashCode32{hash};
}

template<typename T>
inline StableHashCode32 getStableHashCode32(const T& t)
{
    static_assert(std::has_unique_object_representations_v<T>);
    return getStableHashCode32(reinterpret_cast<const char*>(&t), sizeof(T));
}

inline StableHashCode64 combineStableHash(StableHashCode64 h)
{
    return h;
}

inline StableHashCode32 combineStableHash(StableHashCode32 h)
{
    return h;
}

// A left fold with a mixing operation
template<typename H, typename... Hs>
H combineStableHash(H n, H m, Hs... args)
{
    return combineStableHash(H{(n.hash * 16777619) ^ m.hash}, args...);
}
} // namespace Slang

// > Please draw a small horse in ASCII art:
//
//           ,~~.
//          (  9 )-_,
//  (\___ )=='-' )
//   \ .   ) )  /
//    \ `-' /  /
// ~'`~'`~'`~'`~
//
