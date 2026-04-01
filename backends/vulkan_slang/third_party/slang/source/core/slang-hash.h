#ifndef SLANG_CORE_HASH_H
#define SLANG_CORE_HASH_H

#include "slang-math.h"
#include "slang.h"

#include <ankerl/unordered_dense.h>
#include <cstring>
#include <type_traits>

namespace Slang
{
//
// Types
//

// A fixed 64bit wide hash on all targets.
typedef uint64_t HashCode64;
typedef HashCode64 HashCode;
// A fixed 32bit wide hash on all targets.
typedef uint32_t HashCode32;

//
// Some helpers to determine which hash to use for a type
//

// Forward declare Hash
template<typename T>
struct Hash;

template<typename T, typename = void>
constexpr static bool HasSlangHash = false;
template<typename T>
constexpr static bool HasSlangHash<
    T,
    std::enable_if_t<
        std::is_convertible_v<decltype((std::declval<const T&>()).getHashCode()), HashCode64>>> =
    true;

// Does the hashmap implementation provide a uniform hash for this type.
template<typename T, typename = void>
constexpr static bool HasWyhash = false;
template<typename T>
constexpr static bool HasWyhash<T, typename ankerl::unordered_dense::hash<T>::is_avalanching> =
    true;

// We want to have an associated type 'is_avalanching = void' iff we have a
// hash with good uniformity, the two specializations here add that member
// when appropriate (since we can't declare an associated type with
// constexpr if or something terse like that)
template<typename T, typename = void>
struct DetectAvalanchingHash
{
};
template<typename T>
struct DetectAvalanchingHash<T, std::enable_if_t<HasWyhash<T>>>
{
    using is_avalanching = void;
};
// Have we marked 'getHashCode' as having good uniformity properties.
template<typename T>
struct DetectAvalanchingHash<T, std::enable_if_t<T::kHasUniformHash>>
{
    using is_avalanching = void;
};

// A helper for hashing according to the bit representation
template<typename T, typename U>
struct BitCastHash : DetectAvalanchingHash<U>
{
    auto operator()(const T& t) const
    {
        // Doesn't discard or invent bits
        static_assert(sizeof(T) == sizeof(U));
        // Can we copy bytes to and fro
        static_assert(std::is_trivially_copyable_v<T>);
        static_assert(std::is_trivially_copyable_v<U>);
        // Because we construct a U to memcpy into
        static_assert(std::is_trivially_constructible_v<U>);

        U u;
        memcpy(&u, &t, sizeof(T));
        return Hash<U>{}(u);
    }
};

//
// Our hashing functor which disptaches to the most appropriate hashing
// function for the type
//

template<typename T>
struct Hash : DetectAvalanchingHash<T>
{
    auto operator()(const T& t) const
    {
        // Our preference is for any hash we've defined ourselves
        if constexpr (HasSlangHash<T>)
            return t.getHashCode();
        // Otherwise fall back to any good hash provided by the hashmap
        // library
        else if constexpr (HasWyhash<T>)
            return ankerl::unordered_dense::hash<T>{}(t);
        // Otherwise fail
        else
        {
            // !sizeof(T*) is a 'false' which is dependent on T (pending P2593R0)
            static_assert(!sizeof(T*), "No hash implementation found for this type");
            // This is to avoid the return type being deduced as 'void' and creating further errors.
            return HashCode64(0);
        }
    }
};

// Specializations for float and double which hash 0 and -0 to distinct values
template<>
struct Hash<float> : BitCastHash<float, uint32_t>
{
};
template<>
struct Hash<double> : BitCastHash<double, uint64_t>
{
};

//
// Utility functions for using hashes
//

// A wrapper for Hash<TKey>
template<typename TKey>
auto getHashCode(const TKey& key)
{
    return Hash<TKey>{}(key);
}

inline HashCode64 getHashCode(const char* buffer, std::size_t len)
{
    return ankerl::unordered_dense::detail::wyhash::hash(buffer, len);
}

template<typename T>
HashCode64 hashObjectBytes(const T& t)
{
    static_assert(
        std::has_unique_object_representations_v<T>,
        "This type must have a unique object representation to use hashObjectBytes");
    return getHashCode(reinterpret_cast<const char*>(&t), sizeof(t));
}

// Use in a struct to declare a uniform hash which doens't care about the
// structure of the members.
#define SLANG_BYTEWISE_HASHABLE                   \
    static constexpr bool kHasUniformHash = true; \
    ::Slang::HashCode64 getHashCode() const       \
    {                                             \
        return ::Slang::hashObjectBytes(*this);   \
    }

#define SLANG_COMPONENTWISE_HASHABLE_1 \
    auto getHashCode() const           \
    {                                  \
        const auto& [m1] = *this;      \
        return Slang::getHashCode(m1); \
    }

#define SLANG_COMPONENTWISE_HASHABLE_2                                          \
    auto getHashCode() const                                                    \
    {                                                                           \
        const auto& [m1, m2] = *this;                                           \
        return combineHash(::Slang::getHashCode(m1), ::Slang::getHashCode(m2)); \
    }

inline HashCode64 combineHash(HashCode64 h)
{
    return h;
}

inline HashCode32 combineHash(HashCode32 h)
{
    return h;
}

// A left fold of a mixing operation
template<typename H1, typename H2, typename... Hs>
auto combineHash(H1 n, H2 m, Hs... args)
{
    // TODO: restrict the types here more, currently we tend to throw
    // unhashed integers in here along with proper hashes of objects.
    static_assert(std::is_convertible_v<H1, HashCode64> || std::is_convertible_v<H1, HashCode32>);
    static_assert(std::is_convertible_v<H2, HashCode64> || std::is_convertible_v<H2, HashCode32>);
    return combineHash((n * 16777619) ^ m, args...);
}

struct Hasher
{
public:
    Hasher() {}

    /// Hash the given `value` and combine it into this hash state
    template<typename T>
    void hashValue(T const& value)
    {
        // TODO: Eventually, we should replace `getHashCode`
        // with a "hash into" operation that takes the value
        // and a `Hasher`.

        m_hashCode = combineHash(m_hashCode, getHashCode(value));
    }

    /// Combine the given `hash` code into the hash state.
    ///
    /// Note: users should prefer to use `hashValue` or `hashObject`
    /// when possible, as they may be able to ensure a higher-quality
    /// hash result (e.g., by using more bits to represent the state
    /// during hashing than are used for the final hash code).
    ///
    void addHash(HashCode hash) { m_hashCode = combineHash(m_hashCode, hash); }

    HashCode getResult() const { return m_hashCode; }

private:
    HashCode m_hashCode = 0;
};
} // namespace Slang

#endif
