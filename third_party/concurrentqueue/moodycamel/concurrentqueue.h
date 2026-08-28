// Provides a C++11 implementation of a multi-producer, multi-consumer lock-free queue.
// An overview, including benchmark results, is provided here:
//     http://moodycamel.com/blog/2014/a-fast-general-purpose-lock-free-queue-for-c++
// The full design is also described in excruciating detail at:
//    http://moodycamel.com/blog/2014/detailed-design-of-a-lock-free-queue

// Simplified BSD license:
// Copyright (c) 2013-2020, Cameron Desrochers.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// - Redistributions of source code must retain the above copyright notice, this list of
// conditions and the following disclaimer.
// - Redistributions in binary form must reproduce the above copyright notice, this list of
// conditions and the following disclaimer in the documentation and/or other materials
// provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
// THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
// OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
// TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
// EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Also dual-licensed under the Boost Software License (see LICENSE.md)

#pragma once

#if defined(__GNUC__) && !defined(__INTEL_COMPILER)
// Disable -Wconversion warnings (spuriously triggered when Traits::size_t and
// Traits::index_t are set to < 32 bits, causing integer promotion, causing warnings
// upon assigning any computed values)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"

#ifdef MCDBGQ_USE_RELACY
#pragma GCC diagnostic ignored "-Wint-to-pointer-cast"
#endif
#endif

#if defined(_MSC_VER) && (!defined(_HAS_CXX17) || !_HAS_CXX17)
// VS2019 with /W4 warns about constant conditional expressions but unless /std=c++17 or higher
// does not support `if constexpr`, so we have no choice but to simply disable the warning
#pragma warning(push)
#pragma warning(disable: 4127)  // conditional expression is constant
#endif

#if defined(__APPLE__)
#include "TargetConditionals.h"
#endif

#ifdef MCDBGQ_USE_RELACY
#include "relacy/relacy_std.hpp"
#include "relacy_shims.h"
// We only use malloc/free anyway, and the delete macro messes up `= delete` method declarations.
// We'll override the default trait malloc ourselves without a macro.
#undef new
#undef delete
#undef malloc
#undef free
#else
#include <atomic>		// Requires C++11. Sorry VS2010.
#include <cassert>
#endif
#include <cstddef>              // for max_align_t
#include <cstdint>
#include <cstdlib>
#include <type_traits>
#include <algorithm>
#include <utility>
#include <limits>
#include <climits>		// for CHAR_BIT
#include <array>
#include <thread>		// partly for __WINPTHREADS_VERSION if on MinGW-w64 w/ POSIX threading
#include <mutex>        // used for thread exit synchronization

// Platform-specific definitions of a numeric thread ID type and an invalid value
namespace moodycamel { namespace details {
	template<typename thread_id_t> struct thread_id_converter {
		typedef thread_id_t thread_id_numeric_size_t;
		typedef thread_id_t thread_id_hash_t;
		static thread_id_hash_t prehash(thread_id_t const& x) { return x; }
	};
} }
#if defined(MCDBGQ_USE_RELACY)
namespace moodycamel { namespace details {
	typedef std::uint32_t thread_id_t;
	static const thread_id_t invalid_thread_id  = 0xFFFFFFFFU;
	static const thread_id_t invalid_thread_id2 = 0xFFFFFFFEU;
	static inline thread_id_t thread_id() { return rl::thread_index(); }
} }
#elif defined(_WIN32) || defined(__WINDOWS__) || defined(__WIN32__)
// No sense pulling in windows.h in a header, we'll manually declare the function
// we use and rely on backwards-compatibility for this not to break
extern "C" __declspec(dllimport) unsigned long __stdcall GetCurrentThreadId(void);
namespace moodycamel { namespace details {
	static_assert(sizeof(unsigned long) == sizeof(std::uint32_t), "Expected size of unsigned long to be 32 bits on Windows");
	typedef std::uint32_t thread_id_t;
	static const thread_id_t invalid_thread_id  = 0;			// See http://blogs.msdn.com/b/oldnewthing/archive/2004/02/23/78395.aspx
	static const thread_id_t invalid_thread_id2 = 0xFFFFFFFFU;	// Not technically guaranteed to be invalid, but is never used in practice. Note that all Win32 thread IDs are presently multiples of 4.
	static inline thread_id_t thread_id() { return static_cast<thread_id_t>(::GetCurrentThreadId()); }
} }
#elif defined(__arm__) || defined(_M_ARM) || defined(__aarch64__) || (defined(__APPLE__) && TARGET_OS_IPHONE) || defined(__MVS__) || defined(MOODYCAMEL_NO_THREAD_LOCAL)
namespace moodycamel { namespace details {
	static_assert(sizeof(std::thread::id) == 4 || sizeof(std::thread::id) == 8, "std::thread::id is expected to be either 4 or 8 bytes");
	
	typedef std::thread::id thread_id_t;
	static const thread_id_t invalid_thread_id;         // Default ctor creates invalid ID

	// Note we don't define a invalid_thread_id2 since std::thread::id doesn't have one; it's
	// only used if MOODYCAMEL_CPP11_THREAD_LOCAL_SUPPORTED is defined anyway, which it won't
	// be.
	static inline thread_id_t thread_id() { return std::this_thread::get_id(); }

	template<std::size_t> struct thread_id_size { };
	template<> struct thread_id_size<4> { typedef std::uint32_t numeric_t; };
	template<> struct thread_id_size<8> { typedef std::uint64_t numeric_t; };

	template<> struct thread_id_converter<thread_id_t> {
		typedef thread_id_size<sizeof(thread_id_t)>::numeric_t thread_id_numeric_size_t;
#ifndef __APPLE__
		typedef std::size_t thread_id_hash_t;
#else
		typedef thread_id_numeric_size_t thread_id_hash_t;
#endif

		static thread_id_hash_t prehash(thread_id_t const& x)
		{
#ifndef __APPLE__
			return std::hash<std::thread::id>()(x);
#else
			return *reinterpret_cast<thread_id_hash_t const*>(&x);
#endif
		}
	};
} }
#else
// Use a nice trick from this answer: http://stackoverflow.com/a/8438730/21475
// In order to get a numeric thread ID in a platform-independent way, we use a thread-local
// static variable's address as a thread identifier :-)
#if defined(__GNUC__) || defined(__INTEL_COMPILER)
#define MOODYCAMEL_THREADLOCAL __thread
#elif defined(_MSC_VER)
#define MOODYCAMEL_THREADLOCAL __declspec(thread)
#else
// Assume C++11 compliant compiler
#define MOODYCAMEL_THREADLOCAL thread_local
#endif
namespace moodycamel { namespace details {
	typedef std::uintptr_t thread_id_t;
	static const thread_id_t invalid_thread_id  = 0;		// Address can't be nullptr
	static const thread_id_t invalid_thread_id2 = 1;		// Member accesses off a null pointer are also generally invalid. Plus it's not aligned.
	inline thread_id_t thread_id() { static MOODYCAMEL_THREADLOCAL int x; return reinterpret_cast<thread_id_t>(&x); }
} }
#endif

// Constexpr if
#ifndef MOODYCAMEL_CONSTEXPR_IF
#if (defined(_MSC_VER) && defined(_HAS_CXX17) && _HAS_CXX17) || __cplusplus > 201402L
#define MOODYCAMEL_CONSTEXPR_IF if constexpr
#define MOODYCAMEL_MAYBE_UNUSED [[maybe_unused]]
#else
#define MOODYCAMEL_CONSTEXPR_IF if
#define MOODYCAMEL_MAYBE_UNUSED
#endif
#endif

// Exceptions
#ifndef MOODYCAMEL_EXCEPTIONS_ENABLED
#if (defined(_MSC_VER) && defined(_CPPUNWIND)) || (defined(__GNUC__) && defined(__EXCEPTIONS)) || (!defined(_MSC_VER) && !defined(__GNUC__))
#define MOODYCAMEL_EXCEPTIONS_ENABLED
#endif
#endif
#ifdef MOODYCAMEL_EXCEPTIONS_ENABLED
#define MOODYCAMEL_TRY try
#define MOODYCAMEL_CATCH(...) catch(__VA_ARGS__)
#define MOODYCAMEL_RETHROW throw
#define MOODYCAMEL_THROW(expr) throw (expr)
#else
#define MOODYCAMEL_TRY MOODYCAMEL_CONSTEXPR_IF (true)
#define MOODYCAMEL_CATCH(...) else MOODYCAMEL_CONSTEXPR_IF (false)
#define MOODYCAMEL_RETHROW
#define MOODYCAMEL_THROW(expr)
#endif

#ifndef MOODYCAMEL_NOEXCEPT
#if !defined(MOODYCAMEL_EXCEPTIONS_ENABLED)
#define MOODYCAMEL_NOEXCEPT
#define MOODYCAMEL_NOEXCEPT_CTOR(type, valueType, expr) true
#define MOODYCAMEL_NOEXCEPT_ASSIGN(type, valueType, expr) true
#elif defined(_MSC_VER) && defined(_NOEXCEPT) && _MSC_VER < 1800
// VS2012's std::is_nothrow_[move_]constructible is broken and returns true when it shouldn't :-(
// We have to assume *all* non-trivial constructors may throw on VS2012!
#define MOODYCAMEL_NOEXCEPT _NOEXCEPT
#define MOODYCAMEL_NOEXCEPT_CTOR(type, valueType, expr) (std::is_rvalue_reference<valueType>::value && std::is_move_constructible<type>::value ? std::is_trivially_move_constructible<type>::value : std::is_trivially_copy_constructible<type>::value)
#define MOODYCAMEL_NOEXCEPT_ASSIGN(type, valueType, expr) ((std::is_rvalue_reference<valueType>::value && std::is_move_assignable<type>::value ? std::is_trivially_move_assignable<type>::value || std::is_nothrow_move_assignable<type>::value : std::is_trivially_copy_assignable<type>::value || std::is_nothrow_copy_assignable<type>::value) && MOODYCAMEL_NOEXCEPT_CTOR(type, valueType, expr))
#elif defined(_MSC_VER) && defined(_NOEXCEPT) && _MSC_VER < 1900
#define MOODYCAMEL_NOEXCEPT _NOEXCEPT
#define MOODYCAMEL_NOEXCEPT_CTOR(type, valueType, expr) (std::is_rvalue_reference<valueType>::value && std::is_move_constructible<type>::value ? std::is_trivially_move_constructible<type>::value || std::is_nothrow_move_constructible<type>::value : std::is_trivially_copy_constructible<type>::value || std::is_nothrow_copy_constructible<type>::value)
#define MOODYCAMEL_NOEXCEPT_ASSIGN(type, valueType, expr) ((std::is_rvalue_reference<valueType>::value && std::is_move_assignable<type>::value ? std::is_trivially_move_assignable<type>::value || std::is_nothrow_move_assignable<type>::value : std::is_trivially_copy_assignable<type>::value || std::is_nothrow_copy_assignable<type>::value) && MOODYCAMEL_NOEXCEPT_CTOR(type, valueType, expr))
#else
#define MOODYCAMEL_NOEXCEPT noexcept
#define MOODYCAMEL_NOEXCEPT_CTOR(type, valueType, expr) noexcept(expr)
#define MOODYCAMEL_NOEXCEPT_ASSIGN(type, valueType, expr) noexcept(expr)
#endif
#endif

#ifndef MOODYCAMEL_CPP11_THREAD_LOCAL_SUPPORTED
#ifdef MCDBGQ_USE_RELACY
#define MOODYCAMEL_CPP11_THREAD_LOCAL_SUPPORTED
#else
// VS2013 doesn't support `thread_local`, and MinGW-w64 w/ POSIX threading has a crippling bug: http://sourceforge.net/p/mingw-w64/bugs/445
// g++ <=4.7 doesn't support thread_local either.
// Finally, iOS/ARM doesn't have support for it either, and g++/ARM allows it to compile but it's unconfirmed to actually work
#if (!defined(_MSC_VER) || _MSC_VER >= 1900) && (!defined(__MINGW32__) && !defined(__MINGW64__) || !defined(__WINPTHREADS_VERSION)) && (!defined(__GNUC__) || __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 8)) && (!defined(__APPLE__) || !TARGET_OS_IPHONE) && !defined(__arm__) && !defined(_M_ARM) && !defined(__aarch64__) && !defined(__MVS__)
// Assume `thread_local` is fully supported in all other C++11 compilers/platforms
#define MOODYCAMEL_CPP11_THREAD_LOCAL_SUPPORTED    // tentatively enabled for now; years ago several users report having problems with it on
#endif
#endif
#endif

// VS2012 doesn't support deleted functions. 
// In this case, we declare the function normally but don't define it. A link error will be generated if the function is called.
#ifndef MOODYCAMEL_DELETE_FUNCTION
#if defined(_MSC_VER) && _MSC_VER < 1800
#define MOODYCAMEL_DELETE_FUNCTION
#else
#define MOODYCAMEL_DELETE_FUNCTION = delete
#endif
#endif

namespace moodycamel { namespace details {
#ifndef MOODYCAMEL_ALIGNAS
// VS2013 doesn't support alignas or alignof, and align() requires a constant literal
#if defined(_MSC_VER) && _MSC_VER <= 1800
#define MOODYCAMEL_ALIGNAS(alignment) __declspec(align(alignment))
#define MOODYCAMEL_ALIGNOF(obj) __alignof(obj)
#define MOODYCAMEL_ALIGNED_TYPE_LIKE(T, obj) typename details::Vs2013Aligned<std::alignment_of<obj>::value, T>::type
	template<int Align, typename T> struct Vs2013Aligned { };  // default, unsupported alignment
	template<typename T> struct Vs2013Aligned<1, T> { typedef __declspec(align(1)) T type; };
	template<typename T> struct Vs2013Aligned<2, T> { typedef __declspec(align(2)) T type; };
	template<typename T> struct Vs2013Aligned<4, T> { typedef __declspec(align(4)) T type; };
	template<typename T> struct Vs2013Aligned<8, T> { typedef __declspec(align(8)) T type; };
	template<typename T> struct Vs2013Aligned<16, T> { typedef __declspec(align(16)) T type; };
	template<typename T> struct Vs2013Aligned<32, T> { typedef __declspec(align(32)) T type; };
	template<typename T> struct Vs2013Aligned<64, T> { typedef __declspec(align(64)) T type; };
	template<typename T> struct Vs2013Aligned<128, T> { typedef __declspec(align(128)) T type; };
	template<typename T> struct Vs2013Aligned<256, T> { typedef __declspec(align(256)) T type; };
#else
	template<typename T> struct identity { typedef T type; };
#define MOODYCAMEL_ALIGNAS(alignment) alignas(alignment)
#define MOODYCAMEL_ALIGNOF(obj) alignof(obj)
#define MOODYCAMEL_ALIGNED_TYPE_LIKE(T, obj) alignas(alignof(obj)) typename details::identity<T>::type
#endif
#endif
} }


// TSAN can false report races in lock-free code.  To enable TSAN to be used from projects that use this one,
// we can apply per-function compile-time suppression.
// See https://clang.llvm.org/docs/ThreadSanitizer.html#has-feature-thread-sanitizer
#define MOODYCAMEL_NO_TSAN
#if defined(__has_feature)
 #if __has_feature(thread_sanitizer)
  #undef MOODYCAMEL_NO_TSAN
  #define MOODYCAMEL_NO_TSAN __attribute__((no_sanitize("thread")))
 #endif // TSAN
#endif // TSAN

// Compiler-specific likely/unlikely hints
namespace moodycamel { namespace details {
#if defined(__GNUC__)
	static inline bool (likely)(bool x) { return __builtin_expect((x), true); }
	static inline bool (unlikely)(bool x) { return __builtin_expect((x), false); }
#else
	static inline bool (likely)(bool x) { return x; }
	static inline bool (unlikely)(bool x) { return x; }
#endif
} }

#ifdef MOODYCAMEL_QUEUE_INTERNAL_DEBUG
#include "internal/concurrentqueue_internal_debug.h"
#endif

namespace moodycamel {
namespace details {
	template<typename T>
	struct const_numeric_max {
		static_assert(std::is_integral<T>::value, "const_numeric_max can only be used with integers");
		static const T value = std::numeric_limits<T>::is_signed
			? (static_cast<T>(1) << (sizeof(T) * CHAR_BIT - 1)) - static_cast<T>(1)
			: static_cast<T>(-1);
	};

#if defined(__GLIBCXX__)
	typedef ::max_align_t std_max_align_t;      // libstdc++ forgot to add it to std:: for a while
#else
	typedef std::max_align_t std_max_align_t;   // Others (e.g. MSVC) insist it can *only* be accessed via std::
#endif

	// Some platforms have incorrectly set max_align_t to a type with <8 bytes alignment even while supporting
	// 8-byte aligned scalar values (*cough* 32-bit iOS). Work around this with our own union. See issue #64.
	typedef union {
		std_max_align_t x;
		long long y;
		void* z;
	} max_align_t;
}

// Default traits for the ConcurrentQueue. To change some of the
// traits without re-implementing all of them, inherit from this
// struct and shadow the declarations you wish to be different;
// since the traits are used as a template type parameter, the
// shadowed declarations will be used where defined, and the defaults
// otherwise.
struct ConcurrentQueueDefaultTraits
{
	// General-purpose size type. std::size_t is strongly recommended.
	typedef std::size_t size_t;
	
	// The type used for the enqueue and dequeue indices. Must be at least as
	// large as size_t. Should be significantly larger than the number of elements
	// you expect to hold at once, especially if you have a high turnover rate;
	// for example, on 32-bit x86, if you expect to have over a hundred million
	// elements or pump several million elements through your queue in a very
	// short space of time, using a 32-bit type *may* trigger a race condition.
	// A 64-bit int type is recommended in that case, and in practice will
	// prevent a race condition no matter the usage of the queue. Note that
	// whether the queue is lock-free with a 64-int type depends on the whether
	// std::atomic<std::uint64_t> is lock-free, which is platform-specific.
	typedef std::size_t index_t;
	
	// Internally, all elements are enqueued and dequeued from multi-element
	// blocks; this is the smallest controllable unit. If you expect few elements
	// but many producers, a smaller block size should be favoured. For few producers
	// and/or many elements, a larger block size is preferred. A sane default
	// is provided. Must be a power of 2.
	static const size_t BLOCK_SIZE = 32;
	
	// For explicit producers (i.e. when using a producer token), the block is
	// checked for being empty by iterating through a list of flags, one per element.
	// For large block sizes, this is too inefficient, and switching to an atomic
	// counter-based approach is faster. The switch is made for block sizes strictly
	// larger than this threshold.
	static const size_t EXPLICIT_BLOCK_EMPTY_COUNTER_THRESHOLD = 32;
	
	// How many full blocks can be expected for a single explicit producer? This should
	// reflect that number's maximum for optimal performance. Must be a power of 2.
	static const size_t EXPLICIT_INITIAL_INDEX_SIZE = 32;
	
	// How many full blocks can be expected for a single implicit producer? This should
	// reflect that number's maximum for optimal performance. Must be a power of 2.
	static const size_t IMPLICIT_INITIAL_INDEX_SIZE = 32;
	
	// The initial size of the hash table mapping thread IDs to implicit producers.
	// Note that the hash is resized every time it becomes half full.
	// Must be a power of two, and either 0 or at least 1. If 0, implicit production
	// (using the enqueue methods without an explicit producer token) is disabled.
	static const size_t INITIAL_IMPLICIT_PRODUCER_HASH_SIZE = 32;
	
	// Controls the number of items that an explicit consumer (i.e. one with a token)
	// must consume before it causes all consumers to rotate and move on to the next
	// internal queue.
	static const std::uint32_t EXPLICIT_CONSUMER_CONSUMPTION_QUOTA_BEFORE_ROTATE = 256;
	
	// The maximum number of elements (inclusive) that can be enqueued to a sub-queue.
	// Enqueue operations that would cause this limit to be surpassed will fail. Note
	// that this limit is enforced at the block level (for performance reasons), i.e.
	// it's rounded up to the nearest block size.
	static const size_t MAX_SUBQUEUE_SIZE = details::const_numeric_max<size_t>::value;

	// The number of times to spin before sleeping when waiting on a semaphore.
	// Recommended values are on the order of 1000-10000 unless the number of
	// consumer threads exceeds the number of idle cores (in which case try 0-100).
	// Only affects instances of the BlockingConcurrentQueue.
	static const int MAX_SEMA_SPINS = 10000;

	// Whether to recycle dynamically-allocated blocks into an internal free list or
	// not. If false, only pre-allocated blocks (controlled by the constructor
	// arguments) will be recycled, and all others will be `free`d back to the heap.
	// Note that blocks consumed by explicit producers are only freed on destruction
	// of the queue (not following destruction of the token) regardless of this trait.
	static const bool RECYCLE_ALLOCATED_BLOCKS = false;

	
#ifndef MCDBGQ_USE_RELACY
	// Memory allocation can be customized if needed.
	// malloc should return nullptr on failure, and handle alignment like std::malloc.
#if defined(malloc) || defined(free)
	// Gah, this is 2015, stop defining macros that break standard code already!
	// Work around malloc/free being special macros:
	static inline void* WORKAROUND_malloc(size_t size) { return malloc(size); }
	static inline void WORKAROUND_free(void* ptr) { return free(ptr); }
	static inline void* (malloc)(size_t size) { return WORKAROUND_malloc(size); }
	static inline void (free)(void* ptr) { return WORKAROUND_free(ptr); }
#else
	static inline void* malloc(size_t size) { return std::malloc(size); }
	static inline void free(void* ptr) { return std::free(ptr); }
#endif
#else
	// Debug versions when running under the Relacy race detector (ignore
	// these in user code)
	static inline void* malloc(size_t size) { return rl::rl_malloc(size, $); }
	static inline void free(void* ptr) { return rl::rl_free(ptr, $); }
#endif
};


// When producing or consuming many elements, the most efficient way is to:
//    1) Use one of the bulk-operation methods of the queue with a token
//    2) Failing that, use the bulk-operation methods without a token
//    3) Failing that, create a token and use that with the single-item methods
//    4) Failing that, use the single-parameter methods of the queue
// Having said that, don't create tokens willy-nilly -- ideally there should be
// a maximum of one token per thread (of each kind).
struct ProducerToken;
struct ConsumerToken;

template<typename T, typename Traits> class ConcurrentQueue;
template<typename T, typename Traits> class BlockingConcurrentQueue;
class ConcurrentQueueTests;


namespace details
{
	struct ConcurrentQueueProducerTypelessBase
	{
		ConcurrentQueueProducerTypelessBase* next;
		std::atomic<bool> inactive;
		ProducerToken* token;
		
		ConcurrentQueueProducerTypelessBase()
			: next(nullptr), inactive(false), token(nullptr)
		{
		}
	};
	
	template<bool use32> struct _hash_32_or_64 {
		static inline std::uint32_t hash(std::uint32_t h)
		{
			// MurmurHash3 finalizer -- see https://code.google.com/p/smhasher/source/browse/trunk/MurmurHash3.cpp
			// Since the thread ID is already unique, all we really want to do is propagate that
			// uniqueness evenly across all the bits, so that we can use a subset of the bits while
			// reducing collisions significantly
			h ^= h >> 16;
			h *= 0x85ebca6b;
			h ^= h >> 13;
			h *= 0xc2b2ae35;
			return h ^ (h >> 16);
		}
	};
	template<> struct _hash_32_or_64<1> {
		static inline std::uint64_t hash(std::uint64_t h)
		{
			h ^= h >> 33;
			h *= 0xff51afd7ed558ccd;
			h ^= h >> 33;
			h *= 0xc4ceb9fe1a85ec53;
			return h ^ (h >> 33);
		}
	};
	template<std::size_t size> struct hash_32_or_64 : public _hash_32_or_64<(size > 4)> {  };
	
	static inline size_t hash_thread_id(thread_id_t id)
	{
		static_assert(sizeof(thread_id_t) <= 8, "Expected a platform where thread IDs are at most 64-bit values");
		return static_cast<size_t>(hash_32_or_64<sizeof(thread_id_converter<thread_id_t>::thread_id_hash_t)>::hash(
			thread_id_converter<thread_id_t>::prehash(id)));
	}
	
	template<typename T>
	static inline bool circular_less_than(T a, T b)
	{
		static_assert(std::is_integral<T>::value && !std::numeric_limits<T>::is_signed, "circular_less_than is intended to be used only with unsigned integer types");
		return static_cast<T>(a - b) > static_cast<T>(static_cast<T>(1) << (static_cast<T>(sizeof(T) * CHAR_BIT - 1)));
		// Note: extra parens around rhs of operator<< is MSVC bug: https://developercommunity2.visualstudio.com/t/C4554-triggers-when-both-lhs-and-rhs-is/10034931
		//       silencing the bug requires #pragma warning(disable: 4554) around the calling code and has no effect when done here.
	}
	
	template<typename U>
	static inline char* align_for(char* ptr)
	{
		const std::size_t alignment = std::alignment_of<U>::value;
		return ptr + (alignment - (reinterpret_cast<std::uintptr_t>(ptr) % alignment)) % alignment;
	}

	template<typename T>
	static inline T ceil_to_pow_2(T x)
	{
		static_assert(std::is_integral<T>::value && !std::numeric_limits<T>::is_signed, "ceil_to_pow_2 is intended to be used only with unsigned integer types");

		// Adapted from http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
		--x;
		x |= x >> 1;
		x |= x >> 2;
		x |= x >> 4;
		for (std::size_t i = 1; i < sizeof(T); i <<= 1) {
			x |= x >> (i << 3);
		}
		++x;
		return x;
	}
	
	template<typename T>
	static inline void swap_relaxed(std::atomic<T>& left, std::atomic<T>& right)
	{
		T temp = left.load(std::memory_order_relaxed);
		left.store(right.load(std::memory_order_relaxed), std::memory_order_relaxed);
		right.store(temp, std::memory_order_relaxed);
	}
	
	template<typename T>
	static inline T const& nomove(T const& x)
	{
		return x;
	}
	
	template<bool Enable>
	struct nomove_if
	{
		template<typename T>
		static inline T const& eval(T const& x)
		{
			return x;
		}
	};
	
	template<>
	struct nomove_if<false>
	{
		template<typename U>
		static inline auto eval(U&& x)
			-> decltype(std::forward<U>(x))
		{
			return std::forward<U>(x);
		}
	};
	
	template<typename It>
	static inline auto deref_noexcept(It& it) MOODYCAMEL_NOEXCEPT -> decltype(*it)
	{
		return *it;
	}
	
#if defined(__clang__) || !defined(__GNUC__) || __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 8)
	template<typename T> struct is_trivially_destructible : std::is_trivially_destructible<T> { };
#else
	template<typename T> struct is_trivially_destructible : std::has_trivial_destructor<T> { };
#endif
	
#ifdef MOODYCAMEL_CPP11_THREAD_LOCAL_SUPPORTED
#ifdef MCDBGQ_USE_RELACY
	typedef RelacyThreadExitListener ThreadExitListener;
	typedef RelacyThreadExitNotifier ThreadExitNotifier;
#else
	class ThreadExitNotifier;

	struct ThreadExitListener
	{
		typedef void (*callback_t)(void*);
		callback_t callback;
		void* userData;
		
		ThreadExitListener* next;		// reserved for use by the ThreadExitNotifier
		ThreadExitNotifier* chain;		// reserved for use by the ThreadExitNotifier
	};

	class ThreadExitNotifier
	{
	public:
		static void subscribe(ThreadExitListener* listener)
		{
			auto& tlsInst = instance();
			std::lock_guard<std::mutex> guard(mutex());
			listener->next = tlsInst.tail;
			listener->chain = &tlsInst;
			tlsInst.tail = listener;
		}
		
		static void unsubscribe(ThreadExitListener* listener)
		{
			std::lock_guard<std::mutex> guard(mutex());
			if (!listener->chain) {
				return;  // race with ~ThreadExitNotifier
			}
			auto& tlsInst = *listener->chain;
			listener->chain = nullptr;
			ThreadExitListener** prev = &tlsInst.tail;
			for (auto ptr = tlsInst.tail; ptr != nullptr; ptr = ptr->next) {
				if (ptr == listener) {
					*prev = ptr->next;
					break;
				}
				prev = &ptr->next;
			}
		}
		
	private:
		ThreadExitNotifier() : tail(nullptr) { }
		ThreadExitNotifier(ThreadExitNotifier const&) MOODYCAMEL_DELETE_FUNCTION;
		ThreadExitNotifier& operator=(ThreadExitNotifier const&) MOODYCAMEL_DELETE_FUNCTION;
		
		~ThreadExitNotifier()
		{
			// This thread is about to exit, let everyone know!
			assert(this == &instance() && "If this assert fails, you likely have a buggy compiler! Change the preprocessor conditions such that MOODYCAMEL_CPP11_THREAD_LOCAL_SUPPORTED is no longer defined.");
			std::lock_guard<std::mutex> guard(mutex());
			for (auto ptr = tail; ptr != nullptr; ptr = ptr->next) {
				ptr->chain = nullptr;
				ptr->callback(ptr->userData);
			}
		}
		
		// Thread-local
		static inline ThreadExitNotifier& instance()
		{
			static thread_local ThreadExitNotifier notifier;
			return notifier;
		}

		static inline std::mutex& mutex()
		{
			// Must be static because the ThreadExitNotifier could be destroyed while unsubscribe is called
			static std::mutex mutex;
			return mutex;
		}
		
	private:
		ThreadExitListener* tail;
	};
#endif
#endif
	
	template<typename T> struct static_is_lock_free_num { enum { value = 0 }; };
	template<> struct static_is_lock_free_num<signed char> { enum { value = ATOMIC_CHAR_LOCK_FREE }; };
	template<> struct static_is_lock_free_num<short> { enum { value = ATOMIC_SHORT_LOCK_FREE }; };
	template<> struct static_is_lock_free_num<int> { enum { value = ATOMIC_INT_LOCK_FREE }; };
	template<> struct static_is_lock_free_num<long> { enum { value = ATOMIC_LONG_LOCK_FREE }; };
	template<> struct static_is_lock_free_num<long long> { enum { value = ATOMIC_LLONG_LOCK_FREE }; };
	template<typename T> struct static_is_lock_free : static_is_lock_free_num<typename std::make_signed<T>::type> {  };
	template<> struct static_is_lock_free<bool> { enum { value = ATOMIC_BOOL_LOCK_FREE }; };
	template<typename U> struct static_is_lock_free<U*> { enum { value = ATOMIC_POINTER_LOCK_FREE }; };
}


struct ProducerToken
{
	template<typename T, typename Traits>
	explicit ProducerToken(ConcurrentQueue<T, Traits>& queue);
	
	template<typename T, typename Traits>
	explicit ProducerToken(BlockingConcurrentQueue<T, Traits>& queue);
	
	ProducerToken(ProducerToken&& other) MOODYCAMEL_NOEXCEPT
		: producer(other.producer)
	{
		other.producer = nullptr;
		if (producer != nullptr) {
			producer->token = this;
		}
	}
	
	inline ProducerToken& operator=(ProducerToken&& other) MOODYCAMEL_NOEXCEPT
	{
		swap(other);
		return *this;
	}
	
	void swap(ProducerToken& other) MOODYCAMEL_NOEXCEPT
	{
		std::swap(producer, other.producer);
		if (producer != nullptr) {
			producer->token = this;
		}
		if (other.producer != nullptr) {
			other.producer->token = &other;
		}
	}
	
	// A token is always valid unless:
	//     1) Memory allocation failed during construction
	//     2) It was moved via the move constructor
	//        (Note: assignment does a swap, leaving both potentially valid)
	//     3) The associated queue was destroyed
	// Note that if valid() returns true, that only indicates
	// that the token is valid for use with a specific queue,
	// but not which one; that's up to the user to track.
	inline bool valid() const { return producer != nullptr; }
	
	~ProducerToken()
	{
		if (producer != nullptr) {
			producer->token = nullptr;
			producer->inactive.store(true, std::memory_order_release);
		}
	}
	
	// Disable copying and assignment
	ProducerToken(ProducerToken const&) MOODYCAMEL_DELETE_FUNCTION;
	ProducerToken& operator=(ProducerToken const&) MOODYCAMEL_DELETE_FUNCTION;
	
private:
	template<typename T, typename Traits> friend class ConcurrentQueue;
	friend class ConcurrentQueueTests;
	
protected:
	details::ConcurrentQueueProducerTypelessBase* producer;
};


struct ConsumerToken
{
	template<typename T, typename Traits>
	explicit ConsumerToken(ConcurrentQueue<T, Traits>& q);
	
	template<typename T, typename Traits>
	explicit ConsumerToken(BlockingConcurrentQueue<T, Traits>& q);
	
	ConsumerToken(ConsumerToken&& other) MOODYCAMEL_NOEXCEPT
		: initialOffset(other.initialOffset), lastKnownGlobalOffset(other.lastKnownGlobalOffset), itemsConsumedFromCurrent(other.itemsConsumedFromCurrent), currentProducer(other.currentProducer), desiredProducer(other.desiredProducer)
	{
	}
	
	inline ConsumerToken& operator=(ConsumerToken&& other) MOODYCAMEL_NOEXCEPT
	{
		swap(other);
		return *this;
	}
	
	void swap(ConsumerToken& other) MOODYCAMEL_NOEXCEPT
	{
		std::swap(initialOffset, other.initialOffset);
		std::swap(lastKnownGlobalOffset, other.lastKnownGlobalOffset);
		std::swap(itemsConsumedFromCurrent, other.itemsConsumedFromCurrent);
		std::swap(currentProducer, other.currentProducer);
		std::swap(desiredProducer, other.desiredProducer);
	}
	
	// Disable copying and assignment
	ConsumerToken(ConsumerToken const&) MOODYCAMEL_DELETE_FUNCTION;
	ConsumerToken& operator=(ConsumerToken const&) MOODYCAMEL_DELETE_FUNCTION;

private:
	template<typename T, typename Traits> friend class ConcurrentQueue;
	friend class ConcurrentQueueTests;
	
private: // but shared with ConcurrentQueue
	std::uint32_t initialOffset;
	std::uint32_t lastKnownGlobalOffset;
	std::uint32_t itemsConsumedFromCurrent;
	details::ConcurrentQueueProducerTypelessBase* currentProducer;
	details::ConcurrentQueueProducerTypelessBase* desiredProducer;
};

// Need to forward-declare this swap because it's in a namespace.
// See http://stackoverflow.com/questions/4492062/why-does-a-c-friend-class-need-a-forward-declaration-only-in-other-namespaces
template<typename T, typename Traits>
inline void swap(typename ConcurrentQueue<T, Traits>::ImplicitProducerKVP& a, typename ConcurrentQueue<T, Traits>::ImplicitProducerKVP& b) MOODYCAMEL_NOEXCEPT;


template<typename T, typename Traits = ConcurrentQueueDefaultTraits>
class ConcurrentQueue
{
public:
	typedef ::moodycamel::ProducerToken producer_token_t;
	typedef ::moodycamel::ConsumerToken consumer_token_t;
	
	typedef typename Traits::index_t index_t;
	typedef typename Traits::size_t size_t;
	
	static const size_t BLOCK_SIZE = static_cast<size_t>(Traits::BLOCK_SIZE);
	static const size_t EXPLICIT_BLOCK_EMPTY_COUNTER_THRESHOLD = static_cast<size_t>(Traits::EXPLICIT_BLOCK_EMPTY_COUNTER_THRESHOLD);
	static const size_t EXPLICIT_INITIAL_INDEX_SIZE = static_cast<size_t>(Traits::EXPLICIT_INITIAL_INDEX_SIZE);
	static const size_t IMPLICIT_INITIAL_INDEX_SIZE = static_cast<size_t>(Traits::IMPLICIT_INITIAL_INDEX_SIZE);
	static const size_t INITIAL_IMPLICIT_PRODUCER_HASH_SIZE = static_cast<size_t>(Traits::INITIAL_IMPLICIT_PRODUCER_HASH_SIZE);
	static const std::uint32_t EXPLICIT_CONSUMER_CONSUMPTION_QUOTA_BEFORE_ROTATE = static_cast<std::uint32_t>(Traits::EXPLICIT_CONSUMER_CONSUMPTION_QUOTA_BEFORE_ROTATE);
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4307)		// + integral constant overflow (that's what the ternary expression is for!)
#pragma warning(disable: 4309)		// static_cast: Truncation of constant value
#endif
	static const size_t MAX_SUBQUEUE_SIZE = (details::const_numeric_max<size_t>::value - static_cast<size_t>(Traits::MAX_SUBQUEUE_SIZE) < BLOCK_SIZE) ? details::const_numeric_max<size_t>::value : ((static_cast<size_t>(Traits::MAX_SUBQUEUE_SIZE) + (BLOCK_SIZE - 1)) / BLOCK_SIZE * BLOCK_SIZE);
#ifdef _MSC_VER
#pragma warning(pop)
#endif

	static_assert(!std::numeric_limits<size_t>::is_signed && std::is_integral<size_t>::value, "Traits::size_t must be an unsigned integral type");
	static_assert(!std::numeric_limits<index_t>::is_signed && std::is_integral<index_t>::value, "Traits::index_t must be an unsigned integral type");
	static_assert(sizeof(index_t) >= sizeof(size_t), "Traits::index_t must be at least as wide as Traits::size_t");
	static_assert((BLOCK_SIZE > 1) && !(BLOCK_SIZE & (BLOCK_SIZE - 1)), "Traits::BLOCK_SIZE must be a power of 2 (and at least 2)");
	static_assert((EXPLICIT_BLOCK_EMPTY_COUNTER_THRESHOLD > 1) && !(EXPLICIT_BLOCK_EMPTY_COUNTER_THRESHOLD & (EXPLICIT_BLOCK_EMPTY_COUNTER_THRESHOLD - 1)), "Traits::EXPLICIT_BLOCK_EMPTY_COUNTER_THRESHOLD must be a power of 2 (and greater than 1)");
	static_assert((EXPLICIT_INITIAL_INDEX_SIZE > 1) && !(EXPLICIT_INITIAL_INDEX_SIZE & (EXPLICIT_INITIAL_INDEX_SIZE - 1)), "Traits::EXPLICIT_INITIAL_INDEX_SIZE must be a power of 2 (and greater than 1)");
	static_assert((IMPLICIT_INITIAL_INDEX_SIZE > 1) && !(IMPLICIT_INITIAL_INDEX_SIZE & (IMPLICIT_INITIAL_INDEX_SIZE - 1)), "Traits::IMPLICIT_INITIAL_INDEX_SIZE must be a power of 2 (and greater than 1)");
	static_assert((INITIAL_IMPLICIT_PRODUCER_HASH_SIZE == 0) || !(INITIAL_IMPLICIT_PRODUCER_HASH_SIZE & (INITIAL_IMPLICIT_PRODUCER_HASH_SIZE - 1)), "Traits::INITIAL_IMPLICIT_PRODUCER_HASH_SIZE must be a power of 2");
	static_assert(INITIAL_IMPLICIT_PRODUCER_HASH_SIZE == 0 || INITIAL_IMPLICIT_PRODUCER_HASH_SIZE >= 1, "Traits::INITIAL_IMPLICIT_PRODUCER_HASH_SIZE must be at least 1 (or 0 to disable implicit enqueueing)");

public:
	// Creates a queue with at least `capacity` element slots; note that the
	// actual number of elements that can be inserted without additional memory
	// allocation depends on the number of producers and the block size (e.g. if
	// the block size is equal to `capacity`, only a single block will be allocated
	// up-front, which means only a single producer will be able to enqueue elements
	// without an extra allocation -- blocks aren't shared between producers).
	// This method is not thread safe -- it is up to the user to ensure that the
	// queue is fully constructed before it starts being used by other threads (this
	// includes making the memory effects of construction visible, possibly with a
	// memory barrier).
	explicit ConcurrentQueue(size_t capacity = 32 * BLOCK_SIZE)
		: producerListTail(nullptr),
		producerCount(0),
		initialBlockPoolIndex(0),
		nextExplicitConsumerId(0),
		globalExplicitConsumerOffset(0)
	{
		implicitProducerHashResizeInProgress.clear(std::memory_order_relaxed);
		populate_initial_implicit_producer_hash();
		populate_initial_block_list(capacity / BLOCK_SIZE + ((capacity & (BLOCK_SIZE - 1)) == 0 ? 0 : 1));
		
#ifdef MOODYCAMEL_QUEUE_INTERNAL_DEBUG
		// Track all the producers using a fully-resolved typed list for
		// each kind; this makes it possible to debug them starting from
		// the root queue object (otherwise wacky casts are needed that
		// don't compile in the debugger's expression evaluator).
		explicitProducers.store(nullptr, std::memory_order_relaxed);
		implicitProducers.store(nullptr, std::memory_order_relaxed);
#endif
	}
	
	// Computes the correct amount of pre-allocated blocks for you based
	// on the minimum number of elements you want available at any given
	// time, and the maximum concurrent number of each type of producer.
	ConcurrentQueue(size_t minCapacity, size_t maxExplicitProducers, size_t maxImplicitProducers)
		: producerListTail(nullptr),
		producerCount(0),
		initialBlockPoolIndex(0),
		nextExplicitConsumerId(0),
		globalExplicitConsumerOffset(0)
	{
		implicitProducerHashResizeInProgress.clear(std::memory_order_relaxed);
		populate_initial_implicit_producer_hash();
		size_t blocks = (((minCapacity + BLOCK_SIZE - 1) / BLOCK_SIZE) - 1) * (maxExplicitProducers + 1) + 2 * (maxExplicitProducers + maxImplicitProducers);
		populate_initial_block_list(blocks);
		
#ifdef MOODYCAMEL_QUEUE_INTERNAL_DEBUG
		explicitProducers.store(nullptr, std::memory_order_relaxed);
		implicitProducers.store(nullptr, std::memory_order_relaxed);
#endif
	}
	
	// Note: The queue should not be accessed concurrently while it's
	// being deleted. It's up to the user to synchronize this.
	// This method is not thread safe.
	~ConcurrentQueue()
	{
		// Destroy producers
		auto ptr = producerListTail.load(std::memory_order_relaxed);
		while (ptr != nullptr) {
			auto next = ptr->next_prod();
			if (ptr->token != nullptr) {
				ptr->token->producer = nullptr;
			}
			destroy(ptr);
			ptr = next;
		}
		
		// Destroy implicit producer hash tables
		MOODYCAMEL_CONSTEXPR_IF (INITIAL_IMPLICIT_PRODUCER_HASH_SIZE != 0) {
			auto hash = implicitProducerHash.load(std::memory_order_relaxed);
			while (hash != nullptr) {
				auto prev = hash->prev;
				if (prev != nullptr) {		// The last hash is part of this object and was not allocated dynamically
					for (size_t i = 0; i != hash->capacity; ++i) {
						hash->entries[i].~ImplicitProducerKVP();
					}
					hash->~ImplicitProducerHash();
					(Traits::free)(hash);
				}
				hash = prev;
			}
		}
		
		// Destroy global free list
		auto block = freeList.head_unsafe();
		while (block != nullptr) {
			auto next = block->freeListNext.load(std::memory_order_relaxed);
			if (block->dynamicallyAllocated) {
				destroy(block);
			}
			block = next;
		}
		
		// Destroy initial free list
		destroy_array(initialBlockPool, initialBlockPoolSize);
	}

	// Disable copying and copy assignment
	ConcurrentQueue(ConcurrentQueue const&) MOODYCAMEL_DELETE_FUNCTION;
	ConcurrentQueue& operator=(ConcurrentQueue const&) MOODYCAMEL_DELETE_FUNCTION;
	
	// Moving is supported, but note that it is *not* a thread-safe operation.
	// Nobody can use the queue while it's being moved, and the memory effects
	// of that move must be propagated to other threads before they can use it.
	// Note: When a queue is moved, its tokens are still valid but can only be
	// used with the destination queue (i.e. semantically they are moved along
	// with the queue itself).
	ConcurrentQueue(ConcurrentQueue&& other) MOODYCAMEL_NOEXCEPT
		: producerListTail(other.producerListTail.load(std::memory_order_relaxed)),
		producerCount(other.producerCount.load(std::memory_order_relaxed)),
		initialBlockPoolIndex(other.initialBlockPoolIndex.load(std::memory_order_relaxed)),
		initialBlockPool(other.initialBlockPool),
		initialBlockPoolSize(other.initialBlockPoolSize),
		freeList(std::move(other.freeList)),
		nextExplicitConsumerId(other.nextExplicitConsumerId.load(std::memory_order_relaxed)),
		globalExplicitConsumerOffset(other.globalExplicitConsumerOffset.load(std::memory_order_relaxed))
	{
		// Move the other one into this, and leave the other one as an empty queue
		implicitProducerHashResizeInProgress.clear(std::memory_order_relaxed);
		populate_initial_implicit_producer_hash();
		swap_implicit_producer_hashes(other);
		
		other.producerListTail.store(nullptr, std::memory_order_relaxed);
		other.producerCount.store(0, std::memory_order_relaxed);
		other.nextExplicitConsumerId.store(0, std::memory_order_relaxed);
		other.globalExplicitConsumerOffset.store(0, std::memory_order_relaxed);
		
#ifdef MOODYCAMEL_QUEUE_INTERNAL_DEBUG
		explicitProducers.store(other.explicitProducers.load(std::memory_order_relaxed), std::memory_order_relaxed);
		other.explicitProducers.store(nullptr, std::memory_order_relaxed);
		implicitProducers.store(other.implicitProducers.load(std::memory_order_relaxed), std::memory_order_relaxed);
		other.implicitProducers.store(nullptr, std::memory_order_relaxed);
#endif
		
		other.initialBlockPoolIndex.store(0, std::memory_order_relaxed);
		other.initialBlockPoolSize = 0;
		other.initialBlockPool = nullptr;
		
		reown_producers();
	}
	
	inline ConcurrentQueue& operator=(ConcurrentQueue&& other) MOODYCAMEL_NOEXCEPT
	{
		return swap_internal(other);
	}
	
	// Swaps this queue's state with the other's. Not thread-safe.
	// Swapping two queues does not invalidate their tokens, however
	// the tokens that were created for one queue must be used with
	// only the swapped queue (i.e. the tokens are tied to the
	// queue's movable state, not the object itself).
	inline void swap(ConcurrentQueue& other) MOODYCAMEL_NOEXCEPT
	{
		swap_internal(other);
	}
	
private:
	ConcurrentQueue& swap_internal(ConcurrentQueue& other)
	{
		if (this == &other) {
			return *this;
		}
		
		details::swap_relaxed(producerListTail, other.producerListTail);
		details::swap_relaxed(producerCount, other.producerCount);
		details::swap_relaxed(initialBlockPoolIndex, other.initialBlockPoolIndex);
		std::swap(initialBlockPool, other.initialBlockPool);
		std::swap(initialBlockPoolSize, other.initialBlockPoolSize);
		freeList.swap(other.freeList);
		details::swap_relaxed(nextExplicitConsumerId, other.nextExplicitConsumerId);
		details::swap_relaxed(globalExplicitConsumerOffset, other.globalExplicitConsumerOffset);
		
		swap_implicit_producer_hashes(other);
		
		reown_producers();
		other.reown_producers();
		
#ifdef MOODYCAMEL_QUEUE_INTERNAL_DEBUG
		details::swap_relaxed(explicitProducers, other.explicitProducers);
		details::swap_relaxed(implicitProducers, other.implicitProducers);
#endif
		
		return *this;
	}
	
public:
	// Enqueues a single item (by copying it).
	// Allocates memory if required. Only fails if memory allocation fails (or implicit
	// production is disabled because Traits::INITIAL_IMPLICIT_PRODUCER_HASH_SIZE is 0,
	// or Traits::MAX_SUBQUEUE_SIZE has been defined and would be surpassed).
	// Thread-safe.
	inline bool enqueue(T const& item)
	{
		MOODYCAMEL_CONSTEXPR_IF (INITIAL_IMPLICIT_PRODUCER_HASH_SIZE == 0) return false;
		else return inner_enqueue<CanAlloc>(item);
	}
	
	// Enqueues a single item (by moving it, if possible).
	// Allocates memory if required. Only fails if memory allocation fails (or implicit
	// production is disabled because Traits::INITIAL_IMPLICIT_PRODUCER_HASH_SIZE is 0,
	// or Traits::MAX_SUBQUEUE_SIZE has been defined and would be surpassed).
	// Thread-safe.
	inline bool enqueue(T&& item)
	{
		MOODYCAMEL_CONSTEXPR_IF (INITIAL_IMPLICIT_PRODUCER_HASH_SIZE == 0) return false;
		else return inner_enqueue<CanAlloc>(std::move(item));
	}
	
	// Enqueues a single item (by copying it) using an explicit producer token.
	// Allocates memory if required. Only fails if memory allocation fails (or
	// Traits::MAX_SUBQUEUE_SIZE has been defined and would be surpassed).
	// Thread-safe.
	inline bool enqueue(producer_token_t const& token, T const& item)
	{
		return inner_enqueue<CanAlloc>(token, item);
	}
	
	// Enqueues a single item (by moving it, if possible) using an explicit producer token.
	// Allocates memory if required. Only fails if memory allocation fails (or
	// Traits::MAX_SUBQUEUE_SIZE has been defined and would be surpassed).
	// Thread-safe.
	inline bool enqueue(producer_token_t const& token, T&& item)
	{
		return inner_enqueue<CanAlloc>(token, std::move(item));
	}
	
	// Enqueues several items.
	// Allocates memory if required. Only fails if memory allocation fails (or
	// implicit production is disabled because Traits::INITIAL_IMPLICIT_PRODUCER_HASH_SIZE
	// is 0, or Traits::MAX_SUBQUEUE_SIZE has been defined and would be surpassed).
	// Note: Use std::make_move_iterator if the elements should be moved instead of copied.
	// Thread-safe.
	template<typename It>
	bool enqueue_bulk(It itemFirst, size_t count)
	{
		MOODYCAMEL_CONSTEXPR_IF (INITIAL_IMPLICIT_PRODUCER_HASH_SIZE == 0) return false;
		else return inner_enqueue_bulk<CanAlloc>(itemFirst, count);
	}
	
	// Enqueues several items using an explicit producer token.
	// Allocates memory if required. Only fails if memory allocation fails
	// (or Traits::MAX_SUBQUEUE_SIZE has been defined and would be surpassed).
	// Note: Use std::make_move_iterator if the elements should be moved
	// instead of copied.
	// Thread-safe.
	template<typename It>
	bool enqueue_bulk(producer_token_t const& token, It itemFirst, size_t count)
	{
		return inner_enqueue_bulk<CanAlloc>(token, itemFirst, count);
	}
	
	// Enqueues a single item (by copying it).
	// Does not allocate memory. Fails if not enough room to enqueue (or implicit
	// production is disabled because Traits::INITIAL_IMPLICIT_PRODUCER_HASH_SIZE
	// is 0).
	// Thread-safe.
	inline bool try_enqueue(T const& item)
	{
		MOODYCAMEL_CONSTEXPR_IF (INITIAL_IMPLICIT_PRODUCER_HASH_SIZE == 0) return false;
		else return inner_enqueue<CannotAlloc>(item);
	}
	
	// Enqueues a single item (by moving it, if possible).
	// Does not allocate memory (except for one-time implicit producer).
	// Fails if not enough room to enqueue (or implicit production is
	// disabled because Traits::INITIAL_IMPLICIT_PRODUCER_HASH_SIZE is 0).
	// Thread-safe.
	inline bool try_enqueue(T&& item)
	{
		MOODYCAMEL_CONSTEXPR_IF (INITIAL_IMPLICIT_PRODUCER_HASH_SIZE == 0) return false;
		else return inner_enqueue<CannotAlloc>(std::move(item));
	}
	
	// Enqueues a single item (by copying it) using an explicit producer token.
	// Does not allocate memory. Fails if not enough room to enqueue.
	// Thread-safe.
	inline bool try_enqueue(producer_token_t const& token, T const& item)
	{
		return inner_enqueue<CannotAlloc>(token, item);
	}
	
	// Enqueues a single item (by moving it, if possible) using an explicit producer token.
	// Does not allocate memory. Fails if not enough room to enqueue.
	// Thread-safe.
	inline bool try_enqueue(producer_token_t const& token, T&& item)
	{
		return inner_enqueue<CannotAlloc>(token, std::move(item));
	}
	
	// Enqueues several items.
	// Does not allocate memory (except for one-time implicit producer).
	// Fails if not enough room to enqueue (or implicit production is
	// disabled because Traits::INITIAL_IMPLICIT_PRODUCER_HASH_SIZE is 0).
	// Note: Use std::make_move_iterator if the elements should be moved
	// instead of copied.
	// Thread-safe.
	template<typename It>
	bool try_enqueue_bulk(It itemFirst, size_t count)
	{
		MOODYCAMEL_CONSTEXPR_IF (INITIAL_IMPLICIT_PRODUCER_HASH_SIZE == 0) return false;
		else return inner_enqueue_bulk<CannotAlloc>(itemFirst, count);
	}
	
	// Enqueues several items using an explicit producer token.
	// Does not allocate memory. Fails if not enough room to enqueue.
	// Note: Use std::make_move_iterator if the elements should be moved
	// instead of copied.
	// Thread-safe.
	template<typename It>
	bool try_enqueue_bulk(producer_token_t const& token, It itemFirst, size_t count)
	{
		return inner_enqueue_bulk<CannotAlloc>(token, itemFirst, count);
	}
	
	
	
	// Attempts to dequeue from the queue.
	// Returns false if all producer streams appeared empty at the time they
	// were checked (so, the queue is likely but not guaranteed to be empty).
	// Never allocates. Thread-safe.
	template<typename U>
	bool try_dequeue(U& item)
	{
		// Instead of simply trying each producer in turn (which could cause needless contention on the first
		// producer), we score them heuristically.
		size_t nonEmptyCount = 0;
		ProducerBase* best = nullptr;
		size_t bestSize = 0;
		for (auto ptr = producerListTail.load(std::memory_order_acquire); nonEmptyCount < 3 && ptr != nullptr; ptr = ptr->next_prod()) {
			auto size = ptr->size_approx();
			if (size > 0) {
				if (size > bestSize) {
					bestSize = size;
					best = ptr;
				}
				++nonEmptyCount;
			}
		}
		
		// If there was at least one non-empty queue but it appears empty at the time
		// we try to dequeue from it, we need to make sure every queue's been tried
		if (nonEmptyCount > 0) {
			if ((details::likely)(best->dequeue(item))) {
				return true;
			}
			for (auto ptr = producerListTail.load(std::memory_order_acquire); ptr != nullptr; ptr = ptr->next_prod()) {
				if (ptr != best && ptr->dequeue(item)) {
					return true;
				}
			}
		}
		return false;
	}
	
	// Attempts to dequeue from the queue.
	// Returns false if all producer streams appeared empty at the time they
	// were checked (so, the queue is likely but not guaranteed to be empty).
	// This differs from the try_dequeue(item) method in that this one does
	// not attempt to reduce contention by interleaving the order that producer
	// streams are dequeued from. So, using this method can reduce overall throughput
	// under contention, but will give more predictable results in single-threaded
	// consumer scenarios. This is mostly only useful for internal unit tests.
	// Never allocates. Thread-safe.
	template<typename U>
	bool try_dequeue_non_interleaved(U& item)
	{
		for (auto ptr = producerListTail.load(std::memory_order_acquire); ptr != nullptr; ptr = ptr->next_prod()) {
			if (ptr->dequeue(item)) {
				return true;
			}
		}
		return false;
	}
	
	// Attempts to dequeue from the queue using an explicit consumer token.
	// Returns false if all producer streams appeared empty at the time they
	// were checked (so, the queue is likely but not guaranteed to be empty).
	// Never allocates. Thread-safe.
	template<typename U>
	bool try_dequeue(consumer_token_t& token, U& item)
	{
		// The idea is roughly as follows:
		// Every 256 items from one producer, make everyone rotate (increase the global offset) -> this means the highest efficiency consumer dictates the rotation speed of everyone else, more or less
		// If you see that the global offset has changed, you must reset your consumption counter and move to your designated place
		// If there's no items where you're supposed to be, keep moving until you find a producer with some items
		// If the global offset has not changed but you've run out of items to consume, move over from your current position until you find an producer with something in it
		
		if (token.desiredProducer == nullptr || token.lastKnownGlobalOffset != globalExplicitConsumerOffset.load(std::memory_order_relaxed)) {
			if (!update_current_producer_after_rotation(token)) {
				return false;
			}
		}
		
		// If there was at least one non-empty queue but it appears empty at the time
		// we try to dequeue from it, we need to make sure every queue's been tried
		if (static_cast<ProducerBase*>(token.currentProducer)->dequeue(item)) {
			if (++token.itemsConsumedFromCurrent == EXPLICIT_CONSUMER_CONSUMPTION_QUOTA_BEFORE_ROTATE) {
				globalExplicitConsumerOffset.fetch_add(1, std::memory_order_relaxed);
			}
			return true;
		}
		
		auto tail = producerListTail.load(std::memory_order_acquire);
		auto ptr = static_cast<ProducerBase*>(token.currentProducer)->next_prod();
		if (ptr == nullptr) {
			ptr = tail;
		}
		while (ptr != static_cast<ProducerBase*>(token.currentProducer)) {
			if (ptr->dequeue(item)) {
				token.currentProducer = ptr;
				token.itemsConsumedFromCurrent = 1;
				return true;
			}
			ptr = ptr->next_prod();
			if (ptr == nullptr) {
				ptr = tail;
			}
		}
		return false;
	}
	
	// Attempts to dequeue several elements from the queue.
	// Returns the number of items actually dequeued.
	// Returns 0 if all producer streams appeared empty at the time they
	// were checked (so, the queue is likely but not guaranteed to be empty).
	// Never allocates. Thread-safe.
	template<typename It>
	size_t try_dequeue_bulk(It itemFirst, size_t max)
	{
		size_t count = 0;
		for (auto ptr = producerListTail.load(std::memory_order_acquire); ptr != nullptr; ptr = ptr->next_prod()) {
			count += ptr->dequeue_bulk(itemFirst, max - count);
			if (count == max) {
				break;
			}
		}
		return count;
	}
	
	// Attempts to dequeue several elements from the queue using an explicit consumer token.
	// Returns the number of items actually dequeued.
	// Returns 0 if all producer streams appeared empty at the time they
	// were checked (so, the queue is likely but not guaranteed to be empty).
	// Never allocates. Thread-safe.
	template<typename It>
	size_t try_dequeue_bulk(consumer_token_t& token, It itemFirst, size_t max)
	{
		if (token.desiredProducer == nullptr || token.lastKnownGlobalOffset != globalExplicitConsumerOffset.load(std::memory_order_relaxed)) {
			if (!update_current_producer_after_rotation(token)) {
				return 0;
			}
		}
		
		size_t count = static_cast<ProducerBase*>(token.currentProducer)->dequeue_bulk(itemFirst, max);
		if (count == max) {
			if ((token.itemsConsumedFromCurrent += static_cast<std::uint32_t>(max)) >= EXPLICIT_CONSUMER_CONSUMPTION_QUOTA_BEFORE_ROTATE) {
				globalExplicitConsumerOffset.fetch_add(1, std::memory_order_relaxed);
			}
			return max;
		}
		token.itemsConsumedFromCurrent += static_cast<std::uint32_t>(count);
		max -= count;
		
		auto tail = producerListTail.load(std::memory_order_acquire);
		auto ptr = static_cast<ProducerBase*>(token.currentProducer)->next_prod();
		if (ptr == nullptr) {
			ptr = tail;
		}
		while (ptr != static_cast<ProducerBase*>(token.currentProducer)) {
			auto dequeued = ptr->dequeue_bulk(itemFirst, max);
			count += dequeued;
			if (dequeued != 0) {
				token.currentProducer = ptr;
				token.itemsConsumedFromCurrent = static_cast<std::uint32_t>(dequeued);
			}
			if (dequeued == max) {
				break;
			}
			max -= dequeued;
			ptr = ptr->next_prod();
			if (ptr == nullptr) {
				ptr = tail;
			}
		}
		return count;
	}
	
	
	
	// Attempts to dequeue from a specific producer's inner queue.
	// If you happen to know which producer you want to dequeue from, this
	// is significantly faster than using the general-case try_dequeue methods.
	// Returns false if the producer's queue appeared empty at the time it
	// was checked (so, the queue is likely but not guaranteed to be empty).
	// Never allocates. Thread-safe.
	template<typename U>
	inline bool try_dequeue_from_producer(producer_token_t const& producer, U& item)
	{
		return static_cast<ExplicitProducer*>(producer.producer)->dequeue(item);
	}
	
	// Attempts to dequeue several elements from a specific producer's inner queue.
	// Returns the number of items actually dequeued.
	// If you happen to know which producer you want to dequeue from, this
	// is significantly faster than using the general-case try_dequeue methods.
	// Returns 0 if the producer's queue appeared empty at the time it
	// was checked (so, the queue is likely but not guaranteed to be empty).
	// Never allocates. Thread-safe.
	template<typename It>
	inline size_t try_dequeue_bulk_from_producer(producer_token_t const& producer, It itemFirst, size_t max)
	{
		return static_cast<ExplicitProducer*>(producer.producer)->dequeue_bulk(itemFirst, max);
	}
	
	
	// Returns an estimate of the total number of elements currently in the queue. This
	// estimate is only accurate if the queue has completely stabilized before it is called
	// (i.e. all enqueue and dequeue operations have completed and their memory effects are
	// visible on the calling thread, and no further operations start while this method is
	// being called).
	// Thread-safe.
	size_t size_approx() const
	{
		size_t size = 0;
		for (auto ptr = producerListTail.load(std::memory_order_acquire); ptr != nullptr; ptr = ptr->next_prod()) {
			size += ptr->size_approx();
		}
		return size;
	}
	
	
	// Returns true if the underlying atomic variables used by
	// the queue are lock-free (they should be on most platforms).
	// Thread-safe.
	static constexpr bool is_lock_free()
	{
		return
			details::static_is_lock_free<bool>::value == 2 &&
			details::static_is_lock_free<size_t>::value == 2 &&
			details::static_is_lock_free<std::uint32_t>::value == 2 &&
			details::static_is_lock_free<index_t>::value == 2 &&
			details::static_is_lock_free<void*>::value == 2 &&
			details::static_is_lock_free<typename details::thread_id_converter<details::thread_id_t>::thread_id_numeric_size_t>::value == 2;
	}


private:
	friend struct ProducerToken;
	friend struct ConsumerToken;
	struct ExplicitProducer;
	friend struct ExplicitProducer;
	struct ImplicitProducer;
	friend struct ImplicitProducer;
	friend class ConcurrentQueueTests;
		
	enum AllocationMode { CanAlloc, CannotAlloc };
	
	
	///////////////////////////////
	// Queue methods
	///////////////////////////////
	
	template<AllocationMode canAlloc, typename U>
	inline bool inner_enqueue(producer_token_t const& token, U&& element)
	{
		return static_cast<ExplicitProducer*>(token.producer)->ConcurrentQueue::ExplicitProducer::template enqueue<canAlloc>(std::forward<U>(element));
	}
	
	template<AllocationMode canAlloc, typename U>
	inline bool inner_enqueue(U&& element)
	{
		auto producer = get_or_add_implicit_producer();
		return producer == nullptr ? false : producer->ConcurrentQueue::ImplicitProducer::template enqueue<canAlloc>(std::forward<U>(element));
	}
	
	template<AllocationMode canAlloc, typename It>
	inline bool inner_enqueue_bulk(producer_token_t const& token, It itemFirst, size_t count)
	{
		return static_cast<ExplicitProducer*>(token.producer)->ConcurrentQueue::ExplicitProducer::template enqueue_bulk<canAlloc>(itemFirst, count);
	}
	
	template<AllocationMode canAlloc, typename It>
	inline bool inner_enqueue_bulk(It itemFirst, size_t count)
	{
		auto producer = get_or_add_implicit_producer();
		return producer == nullptr ? false : producer->ConcurrentQueue::ImplicitProducer::template enqueue_bulk<canAlloc>(itemFirst, count);
	}
	
	inline bool update_current_producer_after_rotation(consumer_token_t& token)
	{
		// Ah, there's been a rotation, figure out where we should be!
		auto tail = producerListTail.load(std::memory_order_acquire);
		if (token.desiredProducer == nullptr && tail == nullptr) {
			return false;
		}
		auto prodCount = producerCount.load(std::memory_order_relaxed);
		auto globalOffset = globalExplicitConsumerOffset.load(std::memory_order_relaxed);
		if ((details::unlikely)(token.desiredProducer == nullptr)) {
			// Aha, first time we're dequeueing anything.
			// Figure out our local position
			// Note: offset is from start, not end, but we're traversing from end -- subtract from count first
			std::uint32_t offset = prodCount - 1 - (token.initialOffset % prodCount);
			token.desiredProducer = tail;
			for (std::uint32_t i = 0; i != offset; ++i) {
				token.desiredProducer = static_cast<ProducerBase*>(token.desiredProducer)->next_prod();
				if (token.desiredProducer == nullptr) {
					token.desiredProducer = tail;
				}
			}
		}
		
		std::uint32_t delta = globalOffset - token.lastKnownGlobalOffset;
		if (delta >= prodCount) {
			delta = delta % prodCount;
		}
		for (std::uint32_t i = 0; i != delta; ++i) {
			token.desiredProducer = static_cast<ProducerBase*>(token.desiredProducer)->next_prod();
			if (token.desiredProducer == nullptr) {
				token.desiredProducer = tail;
			}
		}
		
		token.lastKnownGlobalOffset = globalOffset;
		token.currentProducer = token.desiredProducer;
		token.itemsConsumedFromCurrent = 0;
		return true;
	}
	
	
	///////////////////////////
	// Free list
	///////////////////////////
	
	template <typename N>
	struct FreeListNode
	{
		FreeListNode() : freeListRefs(0), freeListNext(nullptr) { }
		
		std::atomic<std::uint32_t> freeListRefs;
		std::atomic<N*> freeListNext;
	};
	
	// A simple CAS-based lock-free free list. Not the fastest thing in the world under heavy contention, but
	// simple and correct (assuming nodes are never freed until after the free list is destroyed), and fairly
	// speedy under low contention.
	template<typename N>		// N must inherit FreeListNode or have the same fields (and initialization of them)
	struct FreeList
	{
		FreeList() : freeListHead(nullptr) { }
		FreeList(FreeList&& other) : freeListHead(other.freeListHead.load(std::memory_order_relaxed)) { other.freeListHead.store(nullptr, std::memory_order_relaxed); }
		void swap(FreeList& other) { details::swap_relaxed(freeListHead, other.freeListHead); }
		
		FreeList(FreeList const&) MOODYCAMEL_DELETE_FUNCTION;
		FreeList& operator=(FreeList const&) MOODYCAMEL_DELETE_FUNCTION;
		
		inline void add(N* node)
		{
#ifdef MCDBGQ_NOLOCKFREE_FREELIST
			debug::DebugLock lock(mutex);
#endif		
			// We know that the should-be-on-freelist bit is 0 at this point, so it's safe to
			// set it using a fetch_add
			if (node->freeListRefs.fetch_add(SHOULD_BE_ON_FREELIST, std::memory_order_acq_rel) == 0) {
				// Oh look! We were the last ones referencing this node, and we know
				// we want to add it to the free list, so let's do it!
		 		add_knowing_refcount_is_zero(node);
			}
		}
		
		inline N* try_get()
		{
#ifdef MCDBGQ_NOLOCKFREE_FREELIST
			debug::DebugLock lock(mutex);
#endif		
			auto head = freeListHead.load(std::memory_order_acquire);
			while (head != nullptr) {
				auto prevHead = head;
				auto refs = head->freeListRefs.load(std::memory_order_relaxed);
				if ((refs & REFS_MASK) == 0 || !head->freeListRefs.compare_exchange_strong(refs, refs + 1, std::memory_order_acquire)) {
					head = freeListHead.load(std::memory_order_acquire);
					continue;
				}
				
				// Good, reference count has been incremented (it wasn't at zero), which means we can read the
				// next and not worry about it changing between now and the time we do the CAS
				auto next = head->freeListNext.load(std::memory_order_relaxed);
				if (freeListHead.compare_exchange_strong(head, next, std::memory_order_acquire, std::memory_order_relaxed)) {
					// Yay, got the node. This means it was on the list, which means shouldBeOnFreeList must be false no
					// matter the refcount (because nobody else knows it's been taken off yet, it can't have been put back on).
					assert((head->freeListRefs.load(std::memory_order_relaxed) & SHOULD_BE_ON_FREELIST) == 0);
					
					// Decrease refcount twice, once for our ref, and once for the list's ref
					head->freeListRefs.fetch_sub(2, std::memory_order_release);
					return head;
				}
				
				// OK, the head must have changed on us, but we still need to decrease the refcount we increased.
				// Note that we don't need to release any memory effects, but we do need to ensure that the reference
				// count decrement happens-after the CAS on the head.
				refs = prevHead->freeListRefs.fetch_sub(1, std::memory_order_acq_rel);
				if (refs == SHOULD_BE_ON_FREELIST + 1) {
					add_knowing_refcount_is_zero(prevHead);
				}
			}
			
			return nullptr;
		}
		
		// Useful for traversing the list when there's no contention (e.g. to destroy remaining nodes)
		N* head_unsafe() const { return freeListHead.load(std::memory_order_relaxed); }
		
	private:
		inline void add_knowing_refcount_is_zero(N* node)
		{
			// Since the refcount is zero, and nobody can increase it once it's zero (except us, and we run
			// only one copy of this method per node at a time, i.e. the single thread case), then we know
			// we can safely change the next pointer of the node; however, once the refcount is back above
			// zero, then other threads could increase it (happens under heavy contention, when the refcount
			// goes to zero in between a load and a refcount increment of a node in try_get, then back up to
			// something non-zero, then the refcount increment is done by the other thread) -- so, if the CAS
			// to add the node to the actual list fails, decrease the refcount and leave the add operation to
			// the next thread who puts the refcount back at zero (which could be us, hence the loop).
			auto head = freeListHead.load(std::memory_order_relaxed);
			while (true) {
				node->freeListNext.store(head, std::memory_order_relaxed);
				node->freeListRefs.store(1, std::memory_order_release);
				if (!freeListHead.compare_exchange_strong(head, node, std::memory_order_release, std::memory_order_relaxed)) {
					// Hmm, the add failed, but we can only try again when the refcount goes back to zero
					if (node->freeListRefs.fetch_add(SHOULD_BE_ON_FREELIST - 1, std::memory_order_acq_rel) == 1) {
						continue;
					}
				}
				return;
			}
		}
		
	private:
		// Implemented like a stack, but where node order doesn't matter (nodes are inserted out of order under contention)
		std::atomic<N*> freeListHead;
	
	static const std::uint32_t REFS_MASK = 0x7FFFFFFF;
	static const std::uint32_t SHOULD_BE_ON_FREELIST = 0x80000000;
		
#ifdef MCDBGQ_NOLOCKFREE_FREELIST
		debug::DebugMutex mutex;
#endif
	};
	
	
	///////////////////////////
	// Block
	///////////////////////////
	
	enum InnerQueueContext { implicit_context = 0, explicit_context = 1 };
	
	struct Block
	{
		Block()
			: next(nullptr), elementsCompletelyDequeued(0), freeListRefs(0), freeListNext(nullptr), dynamicallyAllocated(true)
		{
#ifdef MCDBGQ_TRACKMEM
			owner = nullptr;
#endif
		}
		
		template<InnerQueueContext context>
		inline bool is_empty() const
		{
			MOODYCAMEL_CONSTEXPR_IF (context == explicit_context && BLOCK_SIZE <= EXPLICIT_BLOCK_EMPTY_COUNTER_THRESHOLD) {
				// Check flags
				for (size_t i = 0; i < BLOCK_SIZE; ++i) {
					if (!emptyFlags[i].load(std::memory_order_relaxed)) {
						return false;
					}
				}
				
				// Aha, empty; make sure we have all other memory effects that happened before the empty flags were set
				std::atomic_thread_fence(std::memory_order_acquire);
				return true;
			}
			else {
				// Check counter
				if (elementsCompletelyDequeued.load(std::memory_order_relaxed) == BLOCK_SIZE) {
					std::atomic_thread_fence(std::memory_order_acquire);
					return true;
				}
				assert(elementsCompletelyDequeued.load(std::memory_order_relaxed) <= BLOCK_SIZE);
				return false;
			}
		}
		
		// Returns true if the block is now empty (does not apply in explicit context)
		template<InnerQueueContext context>
		inline bool set_empty(MOODYCAMEL_MAYBE_UNUSED index_t i)
		{
			MOODYCAMEL_CONSTEXPR_IF (context == explicit_context && BLOCK_SIZE <= EXPLICIT_BLOCK_EMPTY_COUNTER_THRESHOLD) {
				// Set flag
				assert(!emptyFlags[BLOCK_SIZE - 1 - static_cast<size_t>(i & static_cast<index_t>(BLOCK_SIZE - 1))].load(std::memory_order_relaxed));
				emptyFlags[BLOCK_SIZE - 1 - static_cast<size_t>(i & static_cast<index_t>(BLOCK_SIZE - 1))].store(true, std::memory_order_release);
				return false;
			}
			else {
				// Increment counter
				auto prevVal = elementsCompletelyDequeued.fetch_add(1, std::memory_order_acq_rel);
				assert(prevVal < BLOCK_SIZE);
				return prevVal == BLOCK_SIZE - 1;
			}
		}
		
		// Sets multiple contiguous item statuses to 'empty' (assumes no wrapping and count > 0).
		// Returns true if the block is now empty (does not apply in explicit context).
		template<InnerQueueContext context>
		inline bool set_many_empty(MOODYCAMEL_MAYBE_UNUSED index_t i, size_t count)
		{
			MOODYCAMEL_CONSTEXPR_IF (context == explicit_context && BLOCK_SIZE <= EXPLICIT_BLOCK_EMPTY_COUNTER_THRESHOLD) {
				// Set flags
				std::atomic_thread_fence(std::memory_order_release);
				i = BLOCK_SIZE - 1 - static_cast<size_t>(i & static_cast<index_t>(BLOCK_SIZE - 1)) - count + 1;
				for (size_t j = 0; j != count; ++j) {
					assert(!emptyFlags[i + j].load(std::memory_order_relaxed));
					emptyFlags[i + j].store(true, std::memory_order_relaxed);
				}
				return false;
			}
			else {
				// Increment counter
				auto prevVal = elementsCompletelyDequeued.fetch_add(count, std::memory_order_acq_rel);
				assert(prevVal + count <= BLOCK_SIZE);
				return prevVal + count == BLOCK_SIZE;
			}
		}
		
		template<InnerQueueContext context>
		inline void set_all_empty()
		{
			MOODYCAMEL_CONSTEXPR_IF (context == explicit_context && BLOCK_SIZE <= EXPLICIT_BLOCK_EMPTY_COUNTER_THRESHOLD) {
				// Set all flags
				for (size_t i = 0; i != BLOCK_SIZE; ++i) {
					emptyFlags[i].store(true, std::memory_order_relaxed);
				}
			}
			else {
				// Reset counter
				elementsCompletelyDequeued.store(BLOCK_SIZE, std::memory_order_relaxed);
			}
		}
		
		template<InnerQueueContext context>
		inline void reset_empty()
		{
			MOODYCAMEL_CONSTEXPR_IF (context == explicit_context && BLOCK_SIZE <= EXPLICIT_BLOCK_EMPTY_COUNTER_THRESHOLD) {
				// Reset flags
				for (size_t i = 0; i != BLOCK_SIZE; ++i) {
					emptyFlags[i].store(false, std::memory_order_relaxed);
				}
			}
			else {
				// Reset counter
				elementsCompletelyDequeued.store(0, std::memory_order_relaxed);
			}
		}
		
		inline T* operator[](index_t idx) MOODYCAMEL_NOEXCEPT { return static_cast<T*>(static_cast<void*>(elements)) + static_cast<size_t>(idx & static_cast<index_t>(BLOCK_SIZE - 1)); }
		inline T const* operator[](index_t idx) const MOODYCAMEL_NOEXCEPT { return static_cast<T const*>(static_cast<void const*>(elements)) + static_cast<size_t>(idx & static_cast<index_t>(BLOCK_SIZE - 1)); }
		
	private:
		static_assert(std::alignment_of<T>::value <= sizeof(T), "The queue does not support types with an alignment greater than their size at this time");
		MOODYCAMEL_ALIGNED_TYPE_LIKE(char[sizeof(T) * BLOCK_SIZE], T) elements;
	public:
		Block* next;
		std::atomic<size_t> elementsCompletelyDequeued;
		std::atomic<bool> emptyFlags[BLOCK_SIZE <= EXPLICIT_BLOCK_EMPTY_COUNTER_THRESHOLD ? BLOCK_SIZE : 1];
	public:
		std::atomic<std::uint32_t> freeListRefs;
		std::atomic<Block*> freeListNext;
		bool dynamicallyAllocated;		// Perhaps a better name for this would be 'isNotPartOfInitialBlockPool'
		
#ifdef MCDBGQ_TRACKMEM
		void* owner;
#endif
	};
	static_assert(std::alignment_of<Block>::value >= std::alignment_of<T>::value, "Internal error: Blocks must be at least as aligned as the type they are wrapping");


#ifdef MCDBGQ_TRACKMEM
public:
	struct MemStats;
private:
#endif
	
	///////////////////////////
	// Producer base
	///////////////////////////
	
	struct ProducerBase : public details::ConcurrentQueueProducerTypelessBase
	{
		ProducerBase(ConcurrentQueue* parent_, bool isExplicit_) :
			tailIndex(0),
			headIndex(0),
			dequeueOptimisticCount(0),
			dequeueOvercommit(0),
			tailBlock(nullptr),
			isExplicit(isExplicit_),
			parent(parent_)
		{
		}
		
		virtual ~ProducerBase() { }
		
		template<typename U>
		inline bool dequeue(U& element)
		{
			if (isExplicit) {
				return static_cast<ExplicitProducer*>(this)->dequeue(element);
			}
			else {
				return static_cast<ImplicitProducer*>(this)->dequeue(element);
			}
		}
		
		template<typename It>
		inline size_t dequeue_bulk(It& itemFirst, size_t max)
		{
			if (isExplicit) {
				return static_cast<ExplicitProducer*>(this)->dequeue_bulk(itemFirst, max);
			}
			else {
				return static_cast<ImplicitProducer*>(this)->dequeue_bulk(itemFirst, max);
			}
		}
		
		inline ProducerBase* next_prod() const { return static_cast<ProducerBase*>(next); }
		
		inline size_t size_approx() const
		{
			auto tail = tailIndex.load(std::memory_order_relaxed);
			auto head = headIndex.load(std::memory_order_relaxed);
			return details::circular_less_than(head, tail) ? static_cast<size_t>(tail - head) : 0;
		}
		
		inline index_t getTail() const { return tailIndex.load(std::memory_order_relaxed); }
	protected:
		std::atomic<index_t> tailIndex;		// Where to enqueue to next
		std::atomic<index_t> headIndex;		// Where to dequeue from next
		
		std::atomic<index_t> dequeueOptimisticCount;
		std::atomic<index_t> dequeueOvercommit;
		
		Block* tailBlock;
		
	public:
		bool isExplicit;
		ConcurrentQueue* parent;
		
	protected:
#ifdef MCDBGQ_TRACKMEM
		friend struct MemStats;
#endif
	};
	
	
	///////////////////////////
	// Explicit queue
	///////////////////////////
		
	struct ExplicitProducer : public ProducerBase
	{
		explicit ExplicitProducer(ConcurrentQueue* parent_) :
			ProducerBase(parent_, true),
			blockIndex(nullptr),
			pr_blockIndexSlotsUsed(0),
			pr_blockIndexSize(EXPLICIT_INITIAL_INDEX_SIZE >> 1),
			pr_blockIndexFront(0),
			pr_blockIndexEntries(nullptr),
			pr_blockIndexRaw(nullptr)
		{
			size_t poolBasedIndexSize = details::ceil_to_pow_2(parent_->initialBlockPoolSize) >> 1;
			if (poolBasedIndexSize > pr_blockIndexSize) {
				pr_blockIndexSize = poolBasedIndexSize;
			}
			
			new_block_index(0);		// This creates an index with double the number of current entries, i.e. EXPLICIT_INITIAL_INDEX_SIZE
		}
		
		~ExplicitProducer()
		{
			// Destruct any elements not yet dequeued.
			// Since we're in the destructor, we can assume all elements
			// are either completely dequeued or completely not (no halfways).
			if (this->tailBlock != nullptr) {		// Note this means there must be a block index too
				// First find the block that's partially dequeued, if any
				Block* halfDequeuedBlock = nullptr;
				if ((this->headIndex.load(std::memory_order_relaxed) & static_cast<index_t>(BLOCK_SIZE - 1)) != 0) {
					// The head's not on a block boundary, meaning a block somewhere is partially dequeued
					// (or the head block is the tail block and was fully dequeued, but the head/tail are still not on a boundary)
					size_t i = (pr_blockIndexFront - pr_blockIndexSlotsUsed) & (pr_blockIndexSize - 1);
					while (details::circular_less_than<index_t>(pr_blockIndexEntries[i].base + BLOCK_SIZE, this->headIndex.load(std::memory_order_relaxed))) {
						i = (i + 1) & (pr_blockIndexSize - 1);
					}
					assert(details::circular_less_than<index_t>(pr_blockIndexEntries[i].base, this->headIndex.load(std::memory_order_relaxed)));
					halfDequeuedBlock = pr_blockIndexEntries[i].block;
				}
				
				// Start at the head block (note the first line in the loop gives us the head from the tail on the first iteration)
				auto block = this->tailBlock;
				do {
					block = block->next;
					if (block->ConcurrentQueue::Block::template is_empty<explicit_context>()) {
						continue;
					}
					
					size_t i = 0;	// Offset into block
					if (block == halfDequeuedBlock) {
						i = static_cast<size_t>(this->headIndex.load(std::memory_order_relaxed) & static_cast<index_t>(BLOCK_SIZE - 1));
					}
					
					// Walk through all the items in the block; if this is the tail block, we need to stop when we reach the tail index
					auto lastValidIndex = (this->tailIndex.load(std::memory_order_relaxed) & static_cast<index_t>(BLOCK_SIZE - 1)) == 0 ? BLOCK_SIZE : static_cast<size_t>(this->tailIndex.load(std::memory_order_relaxed) & static_cast<index_t>(BLOCK_SIZE - 1));
					while (i != BLOCK_SIZE && (block != this->tailBlock || i != lastValidIndex)) {
						(*block)[i++]->~T();
					}
				} while (block != this->tailBlock);
			}
			
			// Destroy all blocks that we own
			if (this->tailBlock != nullptr) {
				auto block = this->tailBlock;
				do {
					auto nextBlock = block->next;
					this->parent->add_block_to_free_list(block);
					block = nextBlock;
				} while (block != this->tailBlock);
			}
			
			// Destroy the block indices
			auto header = static_cast<BlockIndexHeader*>(pr_blockIndexRaw);
			while (header != nullptr) {
				auto prev = static_cast<BlockIndexHeader*>(header->prev);
				header->~BlockIndexHeader();
				(Traits::free)(header);
				header = prev;
			}
		}
		
		template<AllocationMode allocMode, typename U>
		inline bool enqueue(U&& element)
		{
			index_t currentTailIndex = this->tailIndex.load(std::memory_order_relaxed);
			index_t newTailIndex = 1 + currentTailIndex;
			if ((currentTailIndex & static_cast<index_t>(BLOCK_SIZE - 1)) == 0) {
				// We reached the end of a block, start a new one
				auto startBlock = this->tailBlock;
				auto originalBlockIndexSlotsUsed = pr_blockIndexSlotsUsed;
				if (this->tailBlock != nullptr && this->tailBlock->next->ConcurrentQueue::Block::template is_empty<explicit_context>()) {
					// We can re-use the block ahead of us, it's empty!					
					this->tailBlock = this->tailBlock->next;
					this->tailBlock->ConcurrentQueue::Block::template reset_empty<explicit_context>();
					
					// We'll put the block on the block index (guaranteed to be room since we're conceptually removing the
					// last block from it first -- except instead of removing then adding, we can just overwrite).
					// Note that there must be a valid block index here, since even if allocation failed in the ctor,
					// it would have been re-attempted when adding the first block to the queue; since there is such
					// a block, a block index must have been successfully allocated.
				}
				else {
					// Whatever head value we see here is >= the last value we saw here (relatively),
					// and <= its current value. Since we have the most recent tail, the head must be
					// <= to it.
					auto head = this->headIndex.load(std::memory_order_relaxed);
					assert(!details::circular_less_than<index_t>(currentTailIndex, head));
					if (!details::circular_less_than<index_t>(head, currentTailIndex + BLOCK_SIZE)
						|| (MAX_SUBQUEUE_SIZE != details::const_numeric_max<size_t>::value && (MAX_SUBQUEUE_SIZE == 0 || MAX_SUBQUEUE_SIZE - BLOCK_SIZE < currentTailIndex - head))) {
						// We can't enqueue in another block because there's not enough leeway -- the
						// tail could surpass the head by the time the block fills up! (Or we'll exceed
						// the size limit, if the second part of the condition was true.)
						return false;
					}
					// We're going to need a new block; check that the block index has room
					if (pr_blockIndexRaw == nullptr || pr_blockIndexSlotsUsed == pr_blockIndexSize) {
						// Hmm, the circular block index is already full -- we'll need
						// to allocate a new index. Note pr_blockIndexRaw can only be nullptr if
						// the initial allocation failed in the constructor.
						
						MOODYCAMEL_CONSTEXPR_IF (allocMode == CannotAlloc) {
							return false;
						}
						else if (!new_block_index(pr_blockIndexSlotsUsed)) {
							return false;
						}
					}
					
					// Insert a new block in the circular linked list
					auto newBlock = this->parent->ConcurrentQueue::template requisition_block<allocMode>();
					if (newBlock == nullptr) {
						return false;
					}
#ifdef MCDBGQ_TRACKMEM
					newBlock->owner = this;
#endif
					newBlock->ConcurrentQueue::Block::template reset_empty<explicit_context>();
					if (this->tailBlock == nullptr) {
						newBlock->next = newBlock;
					}
					else {
						newBlock->next = this->tailBlock->next;
						this->tailBlock->next = newBlock;
					}
					this->tailBlock = newBlock;
					++pr_blockIndexSlotsUsed;
				}

				MOODYCAMEL_CONSTEXPR_IF (!MOODYCAMEL_NOEXCEPT_CTOR(T, U, new (static_cast<T*>(nullptr)) T(std::forward<U>(element)))) {
					// The constructor may throw. We want the element not to appear in the queue in
					// that case (without corrupting the queue):
					MOODYCAMEL_TRY {
						new ((*this->tailBlock)[currentTailIndex]) T(std::forward<U>(element));
					}
					MOODYCAMEL_CATCH (...) {
						// Revert change to the current block, but leave the new block available
						// for next time
						pr_blockIndexSlotsUsed = originalBlockIndexSlotsUsed;
						this->tailBlock = startBlock == nullptr ? this->tailBlock : startBlock;
						MOODYCAMEL_RETHROW;
					}
				}
				else {
					(void)startBlock;
					(void)originalBlockIndexSlotsUsed;
				}
				
				// Add block to block index
				auto& entry = blockIndex.load(std::memory_order_relaxed)->entries[pr_blockIndexFront];
				entry.base = currentTailIndex;
				entry.block = this->tailBlock;
				blockIndex.load(std::memory_order_relaxed)->front.store(pr_blockIndexFront, std::memory_order_release);
				pr_blockIndexFront = (pr_blockIndexFront + 1) & (pr_blockIndexSize - 1);
				
				MOODYCAMEL_CONSTEXPR_IF (!MOODYCAMEL_NOEXCEPT_CTOR(T, U, new (static_cast<T*>(nullptr)) T(std::forward<U>(element)))) {
					this->tailIndex.store(newTailIndex, std::memory_order_release);
					return true;
				}
			}
			
			// Enqueue
			new ((*this->tailBlock)[currentTailIndex]) T(std::forward<U>(element));
			
			this->tailIndex.store(newTailIndex, std::memory_order_release);
			return true;
		}
		
		template<typename U>
		bool dequeue(U& element)
		{
			auto tail = this->tailIndex.load(std::memory_order_relaxed);
			auto overcommit = this->dequeueOvercommit.load(std::memory_order_relaxed);
			if (details::circular_less_than<index_t>(this->dequeueOptimisticCount.load(std::memory_order_relaxed) - overcommit, tail)) {
				// Might be something to dequeue, let's give it a try
				
				// Note that this if is purely for performance purposes in the common case when the queue is
				// empty and the values are eventually consistent -- we may enter here spuriously.
				
				// Note that whatever the values of overcommit and tail are, they are not going to change (unless we
				// change them) and must be the same value at this point (inside the if) as when the if condition was
				// evaluated.

				// We insert an acquire fence here to synchronize-with the release upon incrementing dequeueOvercommit below.
				// This ensures that whatever the value we got loaded into overcommit, the load of dequeueOptisticCount in
				// the fetch_add below will result in a value at least as recent as that (and therefore at least as large).
				// Note that I believe a compiler (signal) fence here would be sufficient due to the nature of fetch_add (all
				// read-modify-write operations are guaranteed to work on the latest value in the modification order), but
				// unfortunately that can't be shown to be correct using only the C++11 standard.
				// See http://stackoverflow.com/questions/18223161/what-are-the-c11-memory-ordering-guarantees-in-this-corner-case
				std::atomic_thread_fence(std::memory_order_acquire);
				
				// Increment optimistic counter, then check if it went over the boundary
				auto myDequeueCount = this->dequeueOptimisticCount.fetch_add(1, std::memory_order_relaxed);
				
				// Note that since dequeueOvercommit must be <= dequeueOptimisticCount (because dequeueOvercommit is only ever
				// incremented after dequeueOptimisticCount -- this is enforced in the `else` block below), and since we now
				// have a version of dequeueOptimisticCount that is at least as recent as overcommit (due to the release upon
				// incrementing dequeueOvercommit and the acquire above that synchronizes with it), overcommit <= myDequeueCount.
				// However, we can't assert this since both dequeueOptimisticCount and dequeueOvercommit may (independently)
				// overflow; in such a case, though, the logic still holds since the difference between the two is maintained.
				
				// Note that we reload tail here in case it changed; it will be the same value as before or greater, since
				// this load is sequenced after (happens after) the earlier load above. This is supported by read-read
				// coherency (as defined in the standard), explained here: http://en.cppreference.com/w/cpp/atomic/memory_order
				tail = this->tailIndex.load(std::memory_order_acquire);
				if ((details::likely)(details::circular_less_than<index_t>(myDequeueCount - overcommit, tail))) {
					// Guaranteed to be at least one element to dequeue!
					
					// Get the index. Note that since there's guaranteed to be at least one element, this
					// will never exceed tail. We need to do an acquire-release fence here since it's possible
					// that whatever condition got us to this point was for an earlier enqueued element (that
					// we already see the memory effects for), but that by the time we increment somebody else
					// has incremented it, and we need to see the memory effects for *that* element, which is
					// in such a case is necessarily visible on the thread that incremented it in the first
					// place with the more current condition (they must have acquired a tail that is at least
					// as recent).
					auto index = this->headIndex.fetch_add(1, std::memory_order_acq_rel);
					
					
					// Determine which block the element is in
					
					auto localBlockIndex = blockIndex.load(std::memory_order_acquire);
					auto localBlockIndexHead = localBlockIndex->front.load(std::memory_order_acquire);
					
					// We need to be careful here about subtracting and dividing because of index wrap-around.
					// When an index wraps, we need to preserve the sign of the offset when dividing it by the
					// block size (in order to get a correct signed block count offset in all cases):
					auto headBase = localBlockIndex->entries[localBlockIndexHead].base;
					auto blockBaseIndex = index & ~static_cast<index_t>(BLOCK_SIZE - 1);
					auto offset = static_cast<size_t>(static_cast<typename std::make_signed<index_t>::type>(blockBaseIndex - headBase) / static_cast<typename std::make_signed<index_t>::type>(BLOCK_SIZE));
					auto block = localBlockIndex->entries[(localBlockIndexHead + offset) & (localBlockIndex->size - 1)].block;
					
					// Dequeue
					auto& el = *((*block)[index]);
					if (!MOODYCAMEL_NOEXCEPT_ASSIGN(T, T&&, element = std::move(el))) {
						// Make sure the element is still fully dequeued and destroyed even if the assignment
						// throws
						struct Guard {
							Block* block;
							index_t index;
							
							~Guard()
							{
								(*block)[index]->~T();
								block->ConcurrentQueue::Block::template set_empty<explicit_context>(index);
							}
						} guard = { block, index };

						element = std::move(el); // NOLINT
					}
					else {
						element = std::move(el); // NOLINT
						el.~T(); // NOLINT
						block->ConcurrentQueue::Block::template set_empty<explicit_context>(index);
					}
					
					return true;
				}
				else {
					// Wasn't anything to dequeue after all; make the effective dequeue count eventually consistent
					this->dequeueOvercommit.fetch_add(1, std::memory_order_release);		// Release so that the fetch_add on dequeueOptimisticCount is guaranteed to happen before this write
				}
			}
		
			return false;
		}
		
		template<AllocationMode allocMode, typename It>
		bool MOODYCAMEL_NO_TSAN enqueue_bulk(It itemFirst, size_t count)
		{
			// First, we need to make sure we have enough room to enqueue all of the elements;
			// this means pre-allocating blocks and putting them in the block index (but only if
			// all the allocations succeeded).
			index_t startTailIndex = this->tailIndex.load(std::memory_order_relaxed);
			auto startBlock = this->tailBlock;
			auto originalBlockIndexFront = pr_blockIndexFront;
			auto originalBlockIndexSlotsUsed = pr_blockIndexSlotsUsed;
			
			Block* firstAllocatedBlock = nullptr;
			
			// Figure out how many blocks we'll need to allocate, and do so
			size_t blockBaseDiff = ((startTailIndex + count - 1) & ~static_cast<index_t>(BLOCK_SIZE - 1)) - ((startTailIndex - 1) & ~static_cast<index_t>(BLOCK_SIZE - 1));
			index_t currentTailIndex = (startTailIndex - 1) & ~static_cast<index_t>(BLOCK_SIZE - 1);
			if (blockBaseDiff > 0) {
				// Allocate as many blocks as possible from ahead
				while (blockBaseDiff > 0 && this->tailBlock != nullptr && this->tailBlock->next != firstAllocatedBlock && this->tailBlock->next->ConcurrentQueue::Block::template is_empty<explicit_context>()) {
					blockBaseDiff -= static_cast<index_t>(BLOCK_SIZE);
					currentTailIndex += static_cast<index_t>(BLOCK_SIZE);
					
					this->tailBlock = this->tailBlock->next;
					firstAllocatedBlock = firstAllocatedBlock == nullptr ? this->tailBlock : firstAllocatedBlock;
					
					auto& entry = blockIndex.load(std::memory_order_relaxed)->entries[pr_blockIndexFront];
					entry.base = currentTailIndex;
					entry.block = this->tailBlock;
					pr_blockIndexFront = (pr_blockIndexFront + 1) & (pr_blockIndexSize - 1);
				}
				
				// Now allocate as many blocks as necessary from the block pool
				while (blockBaseDiff > 0) {
					blockBaseDiff -= static_cast<index_t>(BLOCK_SIZE);
					currentTailIndex += static_cast<index_t>(BLOCK_SIZE);
					
					auto head = this->headIndex.load(std::memory_order_relaxed);
					assert(!details::circular_less_than<index_t>(currentTailIndex, head));
					bool full = !details::circular_less_than<index_t>(head, currentTailIndex + BLOCK_SIZE) || (MAX_SUBQUEUE_SIZE != details::const_numeric_max<size_t>::value && (MAX_SUBQUEUE_SIZE == 0 || MAX_SUBQUEUE_SIZE - BLOCK_SIZE < currentTailIndex - head));
					if (pr_blockIndexRaw == nullptr || pr_blockIndexSlotsUsed == pr_blockIndexSize || full) {
						MOODYCAMEL_CONSTEXPR_IF (allocMode == CannotAlloc) {
							// Failed to allocate, undo changes (but keep injected blocks)
							pr_blockIndexFront = originalBlockIndexFront;
							pr_blockIndexSlotsUsed = originalBlockIndexSlotsUsed;
							this->tailBlock = startBlock == nullptr ? firstAllocatedBlock : startBlock;
							return false;
						}
						else if (full || !new_block_index(originalBlockIndexSlotsUsed)) {
							// Failed to allocate, undo changes (but keep injected blocks)
							pr_blockIndexFront = originalBlockIndexFront;
							pr_blockIndexSlotsUsed = originalBlockIndexSlotsUsed;
							this->tailBlock = startBlock == nullptr ? firstAllocatedBlock : startBlock;
							return false;
						}
						
						// pr_blockIndexFront is updated inside new_block_index, so we need to
						// update our fallback value too (since we keep the new index even if we
						// later fail)
						originalBlockIndexFront = originalBlockIndexSlotsUsed;
					}
					
					// Insert a new block in the circular linked list
					auto newBlock = this->parent->ConcurrentQueue::template requisition_block<allocMode>();
					if (newBlock == nullptr) {
						pr_blockIndexFront = originalBlockIndexFront;
						pr_blockIndexSlotsUsed = originalBlockIndexSlotsUsed;
						this->tailBlock = startBlock == nullptr ? firstAllocatedBlock : startBlock;
						return false;
					}
					
#ifdef MCDBGQ_TRACKMEM
					newBlock->owner = this;
#endif
					newBlock->ConcurrentQueue::Block::template set_all_empty<explicit_context>();
					if (this->tailBlock == nullptr) {
						newBlock->next = newBlock;
					}
					else {
						newBlock->next = this->tailBlock->next;
						this->tailBlock->next = newBlock;
					}
					this->tailBlock = newBlock;
					firstAllocatedBlock = firstAllocatedBlock == nullptr ? this->tailBlock : firstAllocatedBlock;
					
					++pr_blockIndexSlotsUsed;
					
					auto& entry = blockIndex.load(std::memory_order_relaxed)->entries[pr_blockIndexFront];
					entry.base = currentTailIndex;
					entry.block = this->tailBlock;
					pr_blockIndexFront = (pr_blockIndexFront + 1) & (pr_blockIndexSize - 1);
				}
				
				// Excellent, all allocations succeeded. Reset each block's emptiness before we fill them up, and
				// publish the new block index front
				auto block = firstAllocatedBlock;
				while (true) {
					block->ConcurrentQueue::Block::template reset_empty<explicit_context>();
					if (block == this->tailBlock) {
						break;
					}
					block = block->next;
				}
				
				MOODYCAMEL_CONSTEXPR_IF (MOODYCAMEL_NOEXCEPT_CTOR(T, decltype(*itemFirst), new (static_cast<T*>(nullptr)) T(details::deref_noexcept(itemFirst)))) {
					blockIndex.load(std::memory_order_relaxed)->front.store((pr_blockIndexFront - 1) & (pr_blockIndexSize - 1), std::memory_order_release);
				}
			}
			
			// Enqueue, one block at a time
			index_t newTailIndex = startTailIndex + static_cast<index_t>(count);
			currentTailIndex = startTailIndex;
			auto endBlock = this->tailBlock;
			this->tailBlock = startBlock;
			assert((startTailIndex & static_cast<index_t>(BLOCK_SIZE - 1)) != 0 || firstAllocatedBlock != nullptr || count == 0);
			if ((startTailIndex & static_cast<index_t>(BLOCK_SIZE - 1)) == 0 && firstAllocatedBlock != nullptr) {
				this->tailBlock = firstAllocatedBlock;
			}
			while (true) {
				index_t stopIndex = (currentTailIndex & ~static_cast<index_t>(BLOCK_SIZE - 1)) + static_cast<index_t>(BLOCK_SIZE);
				if (details::circular_less_than<index_t>(newTailIndex, stopIndex)) {
					stopIndex = newTailIndex;
				}
				MOODYCAMEL_CONSTEXPR_IF (MOODYCAMEL_NOEXCEPT_CTOR(T, decltype(*itemFirst), new (static_cast<T*>(nullptr)) T(details::deref_noexcept(itemFirst)))) {
					while (currentTailIndex != stopIndex) {
						new ((*this->tailBlock)[currentTailIndex++]) T(*itemFirst++);
					}
				}
				else {
					MOODYCAMEL_TRY {
						while (currentTailIndex != stopIndex) {
							// Must use copy constructor even if move constructor is available
							// because we may have to revert if there's an exception.
							// Sorry about the horrible templated next line, but it was the only way
							// to disable moving *at compile time*, which is important because a type
							// may only define a (noexcept) move constructor, and so calls to the
							// cctor will not compile, even if they are in an if branch that will never
							// be executed
							new ((*this->tailBlock)[currentTailIndex]) T(details::nomove_if<!MOODYCAMEL_NOEXCEPT_CTOR(T, decltype(*itemFirst), new (static_cast<T*>(nullptr)) T(details::deref_noexcept(itemFirst)))>::eval(*itemFirst));
							++currentTailIndex;
							++itemFirst;
						}
					}
					MOODYCAMEL_CATCH (...) {
						// Oh dear, an exception's been thrown -- destroy the elements that
						// were enqueued so far and revert the entire bulk operation (we'll keep
						// any allocated blocks in our linked list for later, though).
						auto constructedStopIndex = currentTailIndex;
						auto lastBlockEnqueued = this->tailBlock;
						
						pr_blockIndexFront = originalBlockIndexFront;
						pr_blockIndexSlotsUsed = originalBlockIndexSlotsUsed;
						this->tailBlock = startBlock == nullptr ? firstAllocatedBlock : startBlock;
						
						if (!details::is_trivially_destructible<T>::value) {
							auto block = startBlock;
							if ((startTailIndex & static_cast<index_t>(BLOCK_SIZE - 1)) == 0) {
								block = firstAllocatedBlock;
							}
							currentTailIndex = startTailIndex;
							while (true) {
								stopIndex = (currentTailIndex & ~static_cast<index_t>(BLOCK_SIZE - 1)) + static_cast<index_t>(BLOCK_SIZE);
								if (details::circular_less_than<index_t>(constructedStopIndex, stopIndex)) {
									stopIndex = constructedStopIndex;
								}
								while (currentTailIndex != stopIndex) {
									(*block)[currentTailIndex++]->~T();
								}
								if (block == lastBlockEnqueued) {
									break;
								}
								block = block->next;
							}
						}
						MOODYCAMEL_RETHROW;
					}
				}
				
				if (this->tailBlock == endBlock) {
					assert(currentTailIndex == newTailIndex);
					break;
				}
				this->tailBlock = this->tailBlock->next;
			}
			
			MOODYCAMEL_CONSTEXPR_IF (!MOODYCAMEL_NOEXCEPT_CTOR(T, decltype(*itemFirst), new (static_cast<T*>(nullptr)) T(details::deref_noexcept(itemFirst)))) {
				if (firstAllocatedBlock != nullptr)
					blockIndex.load(std::memory_order_relaxed)->front.store((pr_blockIndexFront - 1) & (pr_blockIndexSize - 1), std::memory_order_release);
			}
			
			this->tailIndex.store(newTailIndex, std::memory_order_release);
			return true;
		}
		
		template<typename It>
		size_t dequeue_bulk(It& itemFirst, size_t max)
		{
			auto tail = this->tailIndex.load(std::memory_order_relaxed);
			auto overcommit = this->dequeueOvercommit.load(std::memory_order_relaxed);
			auto desiredCount = static_cast<size_t>(tail - (this->dequeueOptimisticCount.load(std::memory_order_relaxed) - overcommit));
			if (details::circular_less_than<size_t>(0, desiredCount)) {
				desiredCount = desiredCount < max ? desiredCount : max;
				std::atomic_thread_fence(std::memory_order_acquire);
				
				auto myDequeueCount = this->dequeueOptimisticCount.fetch_add(desiredCount, std::memory_order_relaxed);
				
				tail = this->tailIndex.load(std::memory_order_acquire);
				auto actualCount = static_cast<size_t>(tail - (myDequeueCount - overcommit));
				if (details::circular_less_than<size_t>(0, actualCount)) {
					actualCount = desiredCount < actualCount ? desiredCount : actualCount;
					if (actualCount < desiredCount) {
						this->dequeueOvercommit.fetch_add(desiredCount - actualCount, std::memory_order_release);
					}
					
					// Get the first index. Note that since there's guaranteed to be at least actualCount elements, this
					// will never exceed tail.
					auto firstIndex = this->headIndex.fetch_add(actualCount, std::memory_order_acq_rel);
					
					// Determine which block the first element is in
					auto localBlockIndex = blockIndex.load(std::memory_order_acquire);
					auto localBlockIndexHead = localBlockIndex->front.load(std::memory_order_acquire);
					
					auto headBase = localBlockIndex->entries[localBlockIndexHead].base;
					auto firstBlockBaseIndex = firstIndex & ~static_cast<index_t>(BLOCK_SIZE - 1);
					auto offset = static_cast<size_t>(static_cast<typename std::make_signed<index_t>::type>(firstBlockBaseIndex - headBase) / static_cast<typename std::make_signed<index_t>::type>(BLOCK_SIZE));
					auto indexIndex = (localBlockIndexHead + offset) & (localBlockIndex->size - 1);
					
					// Iterate the blocks and dequeue
					auto index = firstIndex;
					do {
						auto firstIndexInBlock = index;
						index_t endIndex = (index & ~static_cast<index_t>(BLOCK_SIZE - 1)) + static_cast<index_t>(BLOCK_SIZE);
						endIndex = details::circular_less_than<index_t>(firstIndex + static_cast<index_t>(actualCount), endIndex) ? firstIndex + static_cast<index_t>(actualCount) : endIndex;
						auto block = localBlockIndex->entries[indexIndex].block;
						if (MOODYCAMEL_NOEXCEPT_ASSIGN(T, T&&, details::deref_noexcept(itemFirst) = std::move((*(*block)[index])))) {
							while (index != endIndex) {
								auto& el = *((*block)[index]);
								*itemFirst++ = std::move(el);
								el.~T();
								++index;
							}
						}
						else {
							MOODYCAMEL_TRY {
								while (index != endIndex) {
									auto& el = *((*block)[index]);
									*itemFirst = std::move(el);
									++itemFirst;
									el.~T();
									++index;
								}
							}
							MOODYCAMEL_CATCH (...) {
								// It's too late to revert the dequeue, but we can make sure that all
								// the dequeued objects are properly destroyed and the block index
								// (and empty count) are properly updated before we propagate the exception
								do {
									block = localBlockIndex->entries[indexIndex].block;
									while (index != endIndex) {
										(*block)[index++]->~T();
									}
									block->ConcurrentQueue::Block::template set_many_empty<explicit_context>(firstIndexInBlock, static_cast<size_t>(endIndex - firstIndexInBlock));
									indexIndex = (indexIndex + 1) & (localBlockIndex->size - 1);
									
									firstIndexInBlock = index;
									endIndex = (index & ~static_cast<index_t>(BLOCK_SIZE - 1)) + static_cast<index_t>(BLOCK_SIZE);
									endIndex = details::circular_less_than<index_t>(firstIndex + static_cast<index_t>(actualCount), endIndex) ? firstIndex + static_cast<index_t>(actualCount) : endIndex;
								} while (index != firstIndex + actualCount);
								
								MOODYCAMEL_RETHROW;
							}
						}
						block->ConcurrentQueue::Block::template set_many_empty<explicit_context>(firstIndexInBlock, static_cast<size_t>(endIndex - firstIndexInBlock));
						indexIndex = (indexIndex + 1) & (localBlockIndex->size - 1);
					} while (index != firstIndex + actualCount);
					
					return actualCount;
				}
				else {
					// Wasn't anything to dequeue after all; make the effective dequeue count eventually consistent
					this->dequeueOvercommit.fetch_add(desiredCount, std::memory_order_release);
				}
			}
			
			return 0;
		}
		
	private:
		struct BlockIndexEntry
		{
			index_t base;
			Block* block;
		};
		
		struct BlockIndexHeader
		{
			size_t size;
			std::atomic<size_t> front;		// Current slot (not next, like pr_blockIndexFront)
			BlockIndexEntry* entries;
			void* prev;
		};
		
		
		bool new_block_index(size_t numberOfFilledSlotsToExpose)
		{
			auto prevBlockSizeMask = pr_blockIndexSize - 1;
			
			// Create the new block
			pr_blockIndexSize <<= 1;
			auto newRawPtr = static_cast<char*>((Traits::malloc)(sizeof(BlockIndexHeader) + std::alignment_of<BlockIndexEntry>::value - 1 + sizeof(BlockIndexEntry) * pr_blockIndexSize));
			if (newRawPtr == nullptr) {
				pr_blockIndexSize >>= 1;		// Reset to allow graceful retry
				return false;
			}
			
			auto newBlockIndexEntries = reinterpret_cast<BlockIndexEntry*>(details::align_for<BlockIndexEntry>(newRawPtr + sizeof(BlockIndexHeader)));
			
			// Copy in all the old indices, if any
			size_t j = 0;
			if (pr_blockIndexSlotsUsed != 0) {
				auto i = (pr_blockIndexFront - pr_blockIndexSlotsUsed) & prevBlockSizeMask;
				do {
					newBlockIndexEntries[j++] = pr_blockIndexEntries[i];
					i = (i + 1) & prevBlockSizeMask;
				} while (i != pr_blockIndexFront);
			}
			
			// Update everything
			auto header = new (newRawPtr) BlockIndexHeader;
			header->size = pr_blockIndexSize;
			header->front.store(numberOfFilledSlotsToExpose - 1, std::memory_order_relaxed);
			header->entries = newBlockIndexEntries;
			header->prev = pr_blockIndexRaw;		// we link the new block to the old one so we can free it later
			
			pr_blockIndexFront = j;
			pr_blockIndexEntries = newBlockIndexEntries;
			pr_blockIndexRaw = newRawPtr;
			blockIndex.store(header, std::memory_order_release);
			
			return true;
		}
		
	private:
		std::atomic<BlockIndexHeader*> blockIndex;
		
		// To be used by producer only -- consumer must use the ones in referenced by blockIndex
		size_t pr_blockIndexSlotsUsed;
		size_t pr_blockIndexSize;
		size_t pr_blockIndexFront;		// Next slot (not current)
		BlockIndexEntry* pr_blockIndexEntries;
		void* pr_blockIndexRaw;
		
#ifdef MOODYCAMEL_QUEUE_INTERNAL_DEBUG
	public:
		ExplicitProducer* nextExplicitProducer;
	private:
#endif
		
#ifdef MCDBGQ_TRACKMEM
		friend struct MemStats;
#endif
	};
	
	
	//////////////////////////////////
	// Implicit queue
	//////////////////////////////////
	
	struct ImplicitProducer : public ProducerBase
	{			
		ImplicitProducer(ConcurrentQueue* parent_) :
			ProducerBase(parent_, false),
			nextBlockIndexCapacity(IMPLICIT_INITIAL_INDEX_SIZE),
			blockIndex(nullptr)
		{
			new_block_index();
		}
		
		~ImplicitProducer()
		{
			// Note that since we're in the destructor we can assume that all enqueue/dequeue operations
			// completed already; this means that all undequeued elements are placed contiguously across
			// contiguous blocks, and that only the first and last remaining blocks can be only partially
			// empty (all other remaining blocks must be completely full).
			
#ifdef MOODYCAMEL_CPP11_THREAD_LOCAL_SUPPORTED
			// Unregister ourselves for thread termination notification
			if (!this->inactive.load(std::memory_order_relaxed)) {
				details::ThreadExitNotifier::unsubscribe(&threadExitListener);
			}
#endif
			
			// Destroy all remaining elements!
			auto tail = this->tailIndex.load(std::memory_order_relaxed);
			auto index = this->headIndex.load(std::memory_order_relaxed);
			Block* block = nullptr;
			assert(index == tail || details::circular_less_than(index, tail));
			bool forceFreeLastBlock = index != tail;		// If we enter the loop, then the last (tail) block will not be freed
			while (index != tail) {
				if ((index & static_cast<index_t>(BLOCK_SIZE - 1)) == 0 || block == nullptr) {
					if (block != nullptr) {
						// Free the old block
						this->parent->add_block_to_free_list(block);
					}
					
					block = get_block_index_entry_for_index(index)->value.load(std::memory_order_relaxed);
				}
				
				((*block)[index])->~T();
				++index;
			}
			// Even if the queue is empty, there's still one block that's not on the free list
			// (unless the head index reached the end of it, in which case the tail will be poised
			// to create a new block).
			if (this->tailBlock != nullptr && (forceFreeLastBlock || (tail & static_cast<index_t>(BLOCK_SIZE - 1)) != 0)) {
				this->parent->add_block_to_free_list(this->tailBlock);
			}
			
			// Destroy block index
			auto localBlockIndex = blockIndex.load(std::memory_order_relaxed);
			if (localBlockIndex != nullptr) {
				for (size_t i = 0; i != localBlockIndex->capacity; ++i) {
					localBlockIndex->index[i]->~BlockIndexEntry();
				}
				do {
					auto prev = localBlockIndex->prev;
					localBlockIndex->~BlockIndexHeader();
					(Traits::free)(localBlockIndex);
					localBlockIndex = prev;
				} while (localBlockIndex != nullptr);
			}
		}
		
		template<AllocationMode allocMode, typename U>
		inline bool enqueue(U&& element)
		{
			index_t currentTailIndex = this->tailIndex.load(std::memory_order_relaxed);
			index_t newTailIndex = 1 + currentTailIndex;
			if ((currentTailIndex & static_cast<index_t>(BLOCK_SIZE - 1)) == 0) {
				// We reached the end of a block, start a new one
				auto head = this->headIndex.load(std::memory_order_relaxed);
				assert(!details::circular_less_than<index_t>(currentTailIndex, head));
				if (!details::circular_less_than<index_t>(head, currentTailIndex + BLOCK_SIZE) || (MAX_SUBQUEUE_SIZE != details::const_numeric_max<size_t>::value && (MAX_SUBQUEUE_SIZE == 0 || MAX_SUBQUEUE_SIZE - BLOCK_SIZE < currentTailIndex - head))) {
					return false;
				}
#ifdef MCDBGQ_NOLOCKFREE_IMPLICITPRODBLOCKINDEX
				debug::DebugLock lock(mutex);
#endif
				// Find out where we'll be inserting this block in the block index
				BlockIndexEntry* idxEntry;
				if (!insert_block_index_entry<allocMode>(idxEntry, currentTailIndex)) {
					return false;
				}
				
				// Get ahold of a new block
				auto newBlock = this->parent->ConcurrentQueue::template requisition_block<allocMode>();
				if (newBlock == nullptr) {
					rewind_block_index_tail();
					idxEntry->value.store(nullptr, std::memory_order_relaxed);
					return false;
				}
#ifdef MCDBGQ_TRACKMEM
				newBlock->owner = this;
#endif
				newBlock->ConcurrentQueue::Block::template reset_empty<implicit_context>();

				MOODYCAMEL_CONSTEXPR_IF (!MOODYCAMEL_NOEXCEPT_CTOR(T, U, new (static_cast<T*>(nullptr)) T(std::forward<U>(element)))) {
					// May throw, try to insert now before we publish the fact that we have this new block
					MOODYCAMEL_TRY {
						new ((*newBlock)[currentTailIndex]) T(std::forward<U>(element));
					}
					MOODYCAMEL_CATCH (...) {
						rewind_block_index_tail();
						idxEntry->value.store(nullptr, std::memory_order_relaxed);
						this->parent->add_block_to_free_list(newBlock);
						MOODYCAMEL_RETHROW;
					}
				}
				
				// Insert the new block into the index
				idxEntry->value.store(newBlock, std::memory_order_relaxed);
				
				this->tailBlock = newBlock;
				
				MOODYCAMEL_CONSTEXPR_IF (!MOODYCAMEL_NOEXCEPT_CTOR(T, U, new (static_cast<T*>(nullptr)) T(std::forward<U>(element)))) {
					this->tailIndex.store(newTailIndex, std::memory_order_release);
					return true;
				}
			}
			
			// Enqueue
			new ((*this->tailBlock)[currentTailIndex]) T(std::forward<U>(element));
			
			this->tailIndex.store(newTailIndex, std::memory_order_release);
			return true;
		}
		
		template<typename U>
		bool dequeue(U& element)
		{
			// See ExplicitProducer::dequeue for rationale and explanation
			index_t tail = this->tailIndex.load(std::memory_order_relaxed);
			index_t overcommit = this->dequeueOvercommit.load(std::memory_order_relaxed);
			if (details::circular_less_than<index_t>(this->dequeueOptimisticCount.load(std::memory_order_relaxed) - overcommit, tail)) {
				std::atomic_thread_fence(std::memory_order_acquire);
				
				index_t myDequeueCount = this->dequeueOptimisticCount.fetch_add(1, std::memory_order_relaxed);
				tail = this->tailIndex.load(std::memory_order_acquire);
				if ((details::likely)(details::circular_less_than<index_t>(myDequeueCount - overcommit, tail))) {
					index_t index = this->headIndex.fetch_add(1, std::memory_order_acq_rel);
					
					// Determine which block the element is in
					auto entry = get_block_index_entry_for_index(index);
					
					// Dequeue
					auto block = entry->value.load(std::memory_order_relaxed);
					auto& el = *((*block)[index]);
					
					if (!MOODYCAMEL_NOEXCEPT_ASSIGN(T, T&&, element = std::move(el))) {
#ifdef MCDBGQ_NOLOCKFREE_IMPLICITPRODBLOCKINDEX
						// Note: Acquiring the mutex with every dequeue instead of only when a block
						// is released is very sub-optimal, but it is, after all, purely debug code.
						debug::DebugLock lock(producer->mutex);
#endif
						struct Guard {
							Block* block;
							index_t index;
							BlockIndexEntry* entry;
							ConcurrentQueue* parent;
							
							~Guard()
							{
								(*block)[index]->~T();
								if (block->ConcurrentQueue::Block::template set_empty<implicit_context>(index)) {
									entry->value.store(nullptr, std::memory_order_relaxed);
									parent->add_block_to_free_list(block);
								}
							}
						} guard = { block, index, entry, this->parent };

						element = std::move(el); // NOLINT
					}
					else {
						element = std::move(el); // NOLINT
						el.~T(); // NOLINT

						if (block->ConcurrentQueue::Block::template set_empty<implicit_context>(index)) {
							{
#ifdef MCDBGQ_NOLOCKFREE_IMPLICITPRODBLOCKINDEX
								debug::DebugLock lock(mutex);
#endif
								// Add the block back into the global free pool (and remove from block index)
								entry->value.store(nullptr, std::memory_order_relaxed);
							}
							this->parent->add_block_to_free_list(block);		// releases the above store
						}
					}
					
					return true;
				}
				else {
					this->dequeueOvercommit.fetch_add(1, std::memory_order_release);
				}
			}
		
			return false;
		}
		
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4706)  // assignment within conditional expression
#endif
		template<AllocationMode allocMode, typename It>
		bool enqueue_bulk(It itemFirst, size_t count)
		{
			// First, we need to make sure we have enough room to enqueue all of the elements;
			// this means pre-allocating blocks and putting them in the block index (but only if
			// all the allocations succeeded).
			
			// Note that the tailBlock we start off with may not be owned by us any more;
			// this happens if it was filled up exactly to the top (setting tailIndex to
			// the first index of the next block which is not yet allocated), then dequeued
			// completely (putting it on the free list) before we enqueue again.
			
			index_t startTailIndex = this->tailIndex.load(std::memory_order_relaxed);
			auto startBlock = this->tailBlock;
			Block* firstAllocatedBlock = nullptr;
			auto endBlock = this->tailBlock;
			
			// Figure out how many blocks we'll need to allocate, and do so
			size_t blockBaseDiff = ((startTailIndex + count - 1) & ~static_cast<index_t>(BLOCK_SIZE - 1)) - ((startTailIndex - 1) & ~static_cast<index_t>(BLOCK_SIZE - 1));
			index_t currentTailIndex = (startTailIndex - 1) & ~static_cast<index_t>(BLOCK_SIZE - 1);
			if (blockBaseDiff > 0) {
#ifdef MCDBGQ_NOLOCKFREE_IMPLICITPRODBLOCKINDEX
				debug::DebugLock lock(mutex);
#endif
				do {
					blockBaseDiff -= static_cast<index_t>(BLOCK_SIZE);
					currentTailIndex += static_cast<index_t>(BLOCK_SIZE);
					
					// Find out where we'll be inserting this block in the block index
					BlockIndexEntry* idxEntry = nullptr;  // initialization here unnecessary but compiler can't always tell
					Block* newBlock;
					bool indexInserted = false;
					auto head = this->headIndex.load(std::memory_order_relaxed);
					assert(!details::circular_less_than<index_t>(currentTailIndex, head));
					bool full = !details::circular_less_than<index_t>(head, currentTailIndex + BLOCK_SIZE) || (MAX_SUBQUEUE_SIZE != details::const_numeric_max<size_t>::value && (MAX_SUBQUEUE_SIZE == 0 || MAX_SUBQUEUE_SIZE - BLOCK_SIZE < currentTailIndex - head));

					if (full || !(indexInserted = insert_block_index_entry<allocMode>(idxEntry, currentTailIndex)) || (newBlock = this->parent->ConcurrentQueue::template requisition_block<allocMode>()) == nullptr) {
						// Index allocation or block allocation failed; revert any other allocations
						// and index insertions done so far for this operation
						if (indexInserted) {
							rewind_block_index_tail();
							idxEntry->value.store(nullptr, std::memory_order_relaxed);
						}
						currentTailIndex = (startTailIndex - 1) & ~static_cast<index_t>(BLOCK_SIZE - 1);
						for (auto block = firstAllocatedBlock; block != nullptr; block = block->next) {
							currentTailIndex += static_cast<index_t>(BLOCK_SIZE);
							idxEntry = get_block_index_entry_for_index(currentTailIndex);
							idxEntry->value.store(nullptr, std::memory_order_relaxed);
							rewind_block_index_tail();
						}
						this->parent->add_blocks_to_free_list(firstAllocatedBlock);
						this->tailBlock = startBlock;
						
						return false;
					}
					
#ifdef MCDBGQ_TRACKMEM
					newBlock->owner = this;
#endif
					newBlock->ConcurrentQueue::Block::template reset_empty<implicit_context>();
					newBlock->next = nullptr;
					
					// Insert the new block into the index
					idxEntry->value.store(newBlock, std::memory_order_relaxed);
					
					// Store the chain of blocks so that we can undo if later allocations fail,
					// and so that we can find the blocks when we do the actual enqueueing
					if ((startTailIndex & static_cast<index_t>(BLOCK_SIZE - 1)) != 0 || firstAllocatedBlock != nullptr) {
						assert(this->tailBlock != nullptr);
						this->tailBlock->next = newBlock;
					}
					this->tailBlock = newBlock;
					endBlock = newBlock;
					firstAllocatedBlock = firstAllocatedBlock == nullptr ? newBlock : firstAllocatedBlock;
				} while (blockBaseDiff > 0);
			}
			
			// Enqueue, one block at a time
			index_t newTailIndex = startTailIndex + static_cast<index_t>(count);
			currentTailIndex = startTailIndex;
			this->tailBlock = startBlock;
			assert((startTailIndex & static_cast<index_t>(BLOCK_SIZE - 1)) != 0 || firstAllocatedBlock != nullptr || count == 0);
			if ((startTailIndex & static_cast<index_t>(BLOCK_SIZE - 1)) == 0 && firstAllocatedBlock != nullptr) {
				this->tailBlock = firstAllocatedBlock;
			}
			while (true) {
				index_t stopIndex = (currentTailIndex & ~static_cast<index_t>(BLOCK_SIZE - 1)) + static_cast<index_t>(BLOCK_SIZE);
				if (details::circular_less_than<index_t>(newTailIndex, stopIndex)) {
					stopIndex = newTailIndex;
				}
				MOODYCAMEL_CONSTEXPR_IF (MOODYCAMEL_NOEXCEPT_CTOR(T, decltype(*itemFirst), new (static_cast<T*>(nullptr)) T(details::deref_noexcept(itemFirst)))) {
					while (currentTailIndex != stopIndex) {
						new ((*this->tailBlock)[currentTailIndex++]) T(*itemFirst++);
					}
				}
				else {
					MOODYCAMEL_TRY {
						while (currentTailIndex != stopIndex) {
							new ((*this->tailBlock)[currentTailIndex]) T(details::nomove_if<!MOODYCAMEL_NOEXCEPT_CTOR(T, decltype(*itemFirst), new (static_cast<T*>(nullptr)) T(details::deref_noexcept(itemFirst)))>::eval(*itemFirst));
							++currentTailIndex;
							++itemFirst;
						}
					}
					MOODYCAMEL_CATCH (...) {
						auto constructedStopIndex = currentTailIndex;
						auto lastBlockEnqueued = this->tailBlock;
						
						if (!details::is_trivially_destructible<T>::value) {
							auto block = startBlock;
							if ((startTailIndex & static_cast<index_t>(BLOCK_SIZE - 1)) == 0) {
								block = firstAllocatedBlock;
							}
							currentTailIndex = startTailIndex;
							while (true) {
								stopIndex = (currentTailIndex & ~static_cast<index_t>(BLOCK_SIZE - 1)) + static_cast<index_t>(BLOCK_SIZE);
								if (details::circular_less_than<index_t>(constructedStopIndex, stopIndex)) {
									stopIndex = constructedStopIndex;
								}
								while (currentTailIndex != stopIndex) {
									(*block)[currentTailIndex++]->~T();
								}
								if (block == lastBlockEnqueued) {
									break;
								}
								block = block->next;
							}
						}
						
						currentTailIndex = (startTailIndex - 1) & ~static_cast<index_t>(BLOCK_SIZE - 1);
						for (auto block = firstAllocatedBlock; block != nullptr; block = block->next) {
							currentTailIndex += static_cast<index_t>(BLOCK_SIZE);
							auto idxEntry = get_block_index_entry_for_index(currentTailIndex);
							idxEntry->value.store(nullptr, std::memory_order_relaxed);
							rewind_block_index_tail();
						}
						this->parent->add_blocks_to_free_list(firstAllocatedBlock);
						this->tailBlock = startBlock;
						MOODYCAMEL_RETHROW;
					}
				}
				
				if (this->tailBlock == endBlock) {
					assert(currentTailIndex == newTailIndex);
					break;
				}
				this->tailBlock = this->tailBlock->next;
			}
			this->tailIndex.store(newTailIndex, std::memory_order_release);
			return true;
		}
#ifdef _MSC_VER
#pragma warning(pop)
#endif
		
		template<typename It>
		size_t dequeue_bulk(It& itemFirst, size_t max)
		{
			auto tail = this->tailIndex.load(std::memory_order_relaxed);
			auto overcommit = this->dequeueOvercommit.load(std::memory_order_relaxed);
			auto desiredCount = static_cast<size_t>(tail - (this->dequeueOptimisticCount.load(std::memory_order_relaxed) - overcommit));
			if (details::circular_less_than<size_t>(0, desiredCount)) {
				desiredCount = desiredCount < max ? desiredCount : max;
				std::atomic_thread_fence(std::memory_order_acquire);
				
				auto myDequeueCount = this->dequeueOptimisticCount.fetch_add(desiredCount, std::memory_order_relaxed);
				
				tail = this->tailIndex.load(std::memory_order_acquire);
				auto actualCount = static_cast<size_t>(tail - (myDequeueCount - overcommit));
				if (details::circular_less_than<size_t>(0, actualCount)) {
					actualCount = desiredCount < actualCount ? desiredCount : actualCount;
					if (actualCount < desiredCount) {
						this->dequeueOvercommit.fetch_add(desiredCount - actualCount, std::memory_order_release);
					}
					
					// Get the first index. Note that since there's guaranteed to be at least actualCount elements, this
					// will never exceed tail.
					auto firstIndex = this->headIndex.fetch_add(actualCount, std::memory_order_acq_rel);
					
					// Iterate the blocks and dequeue
					auto index = firstIndex;
					BlockIndexHeader* localBlockIndex;
					auto indexIndex = get_block_index_index_for_index(index, localBlockIndex);
					do {
						auto blockStartIndex = index;
						index_t endIndex = (index & ~static_cast<index_t>(BLOCK_SIZE - 1)) + static_cast<index_t>(BLOCK_SIZE);
						endIndex = details::circular_less_than<index_t>(firstIndex + static_cast<index_t>(actualCount), endIndex) ? firstIndex + static_cast<index_t>(actualCount) : endIndex;
						
						auto entry = localBlockIndex->index[indexIndex];
						auto block = entry->value.load(std::memory_order_relaxed);
						if (MOODYCAMEL_NOEXCEPT_ASSIGN(T, T&&, details::deref_noexcept(itemFirst) = std::move((*(*block)[index])))) {
							while (index != endIndex) {
								auto& el = *((*block)[index]);
								*itemFirst++ = std::move(el);
								el.~T();
								++index;
							}
						}
						else {
							MOODYCAMEL_TRY {
								while (index != endIndex) {
									auto& el = *((*block)[index]);
									*itemFirst = std::move(el);
									++itemFirst;
									el.~T();
									++index;
								}
							}
							MOODYCAMEL_CATCH (...) {
								do {
									entry = localBlockIndex->index[indexIndex];
									block = entry->value.load(std::memory_order_relaxed);
									while (index != endIndex) {
										(*block)[index++]->~T();
									}
									
									if (block->ConcurrentQueue::Block::template set_many_empty<implicit_context>(blockStartIndex, static_cast<size_t>(endIndex - blockStartIndex))) {
#ifdef MCDBGQ_NOLOCKFREE_IMPLICITPRODBLOCKINDEX
										debug::DebugLock lock(mutex);
#endif
										entry->value.store(nullptr, std::memory_order_relaxed);
										this->parent->add_block_to_free_list(block);
									}
									indexIndex = (indexIndex + 1) & (localBlockIndex->capacity - 1);
									
									blockStartIndex = index;
									endIndex = (index & ~static_cast<index_t>(BLOCK_SIZE - 1)) + static_cast<index_t>(BLOCK_SIZE);
									endIndex = details::circular_less_than<index_t>(firstIndex + static_cast<index_t>(actualCount), endIndex) ? firstIndex + static_cast<index_t>(actualCount) : endIndex;
								} while (index != firstIndex + actualCount);
								
								MOODYCAMEL_RETHROW;
							}
						}
						if (block->ConcurrentQueue::Block::template set_many_empty<implicit_context>(blockStartIndex, static_cast<size_t>(endIndex - blockStartIndex))) {
							{
#ifdef MCDBGQ_NOLOCKFREE_IMPLICITPRODBLOCKINDEX
								debug::DebugLock lock(mutex);
#endif
								// Note that the set_many_empty above did a release, meaning that anybody who acquires the block
								// we're about to free can use it safely since our writes (and reads!) will have happened-before then.
								entry->value.store(nullptr, std::memory_order_relaxed);
							}
							this->parent->add_block_to_free_list(block);		// releases the above store
						}
						indexIndex = (indexIndex + 1) & (localBlockIndex->capacity - 1);
					} while (index != firstIndex + actualCount);
					
					return actualCount;
				}
				else {
					this->dequeueOvercommit.fetch_add(desiredCount, std::memory_order_release);
				}
			}
			
			return 0;
		}
		
	private:
		// The block size must be > 1, so any number with the low bit set is an invalid block base index
		static const index_t INVALID_BLOCK_BASE = 1;
		
		struct BlockIndexEntry
		{
			std::atomic<index_t> key;
			std::atomic<Block*> value;
		};
		
		struct BlockIndexHeader
		{
			size_t capacity;
			std::atomic<size_t> tail;
			BlockIndexEntry* entries;
			BlockIndexEntry** index;
			BlockIndexHeader* prev;
		};
		
		template<AllocationMode allocMode>
		inline bool insert_block_index_entry(BlockIndexEntry*& idxEntry, index_t blockStartIndex)
		{
			auto localBlockIndex = blockIndex.load(std::memory_order_relaxed);		// We're the only writer thread, relaxed is OK
			if (localBlockIndex == nullptr) {
				return false;  // this can happen if new_block_index failed in the constructor
			}
			size_t newTail = (localBlockIndex->tail.load(std::memory_order_relaxed) + 1) & (localBlockIndex->capacity - 1);
			idxEntry = localBlockIndex->index[newTail];
			if (idxEntry->key.load(std::memory_order_relaxed) == INVALID_BLOCK_BASE ||
				idxEntry->value.load(std::memory_order_relaxed) == nullptr) {
				
				idxEntry->key.store(blockStartIndex, std::memory_order_relaxed);
				localBlockIndex->tail.store(newTail, std::memory_order_release);
				return true;
			}
			
			// No room in the old block index, try to allocate another one!
			MOODYCAMEL_CONSTEXPR_IF (allocMode == CannotAlloc) {
				return false;
			}
			else if (!new_block_index()) {
				return false;
			}
			else {
				localBlockIndex = blockIndex.load(std::memory_order_relaxed);
				newTail = (localBlockIndex->tail.load(std::memory_order_relaxed) + 1) & (localBlockIndex->capacity - 1);
				idxEntry = localBlockIndex->index[newTail];
				assert(idxEntry->key.load(std::memory_order_relaxed) == INVALID_BLOCK_BASE);
				idxEntry->key.store(blockStartIndex, std::memory_order_relaxed);
				localBlockIndex->tail.store(newTail, std::memory_order_release);
				return true;
			}
		}
		
		inline void rewind_block_index_tail()
		{
			auto localBlockIndex = blockIndex.load(std::memory_order_relaxed);
			localBlockIndex->tail.store((localBlockIndex->tail.load(std::memory_order_relaxed) - 1) & (localBlockIndex->capacity - 1), std::memory_order_relaxed);
		}
		
		inline BlockIndexEntry* get_block_index_entry_for_index(index_t index) const
		{
			BlockIndexHeader* localBlockIndex;
			auto idx = get_block_index_index_for_index(index, localBlockIndex);
			return localBlockIndex->index[idx];
		}
		
		inline size_t get_block_index_index_for_index(index_t index, BlockIndexHeader*& localBlockIndex) const
		{
#ifdef MCDBGQ_NOLOCKFREE_IMPLICITPRODBLOCKINDEX
			debug::DebugLock lock(mutex);
#endif
			index &= ~static_cast<index_t>(BLOCK_SIZE - 1);
			localBlockIndex = blockIndex.load(std::memory_order_acquire);
			auto tail = localBlockIndex->tail.load(std::memory_order_acquire);
			auto tailBase = localBlockIndex->index[tail]->key.load(std::memory_order_relaxed);
			assert(tailBase != INVALID_BLOCK_BASE);
			// Note: Must use division instead of shift because the index may wrap around, causing a negative
			// offset, whose negativity we want to preserve
			auto offset = static_cast<size_t>(static_cast<typename std::make_signed<index_t>::type>(index - tailBase) / static_cast<typename std::make_signed<index_t>::type>(BLOCK_SIZE));
			size_t idx = (tail + offset) & (localBlockIndex->capacity - 1);
			assert(localBlockIndex->index[idx]->key.load(std::memory_order_relaxed) == index && localBlockIndex->index[idx]->value.load(std::memory_order_relaxed) != nullptr);
			return idx;
		}
		
		bool new_block_index()
		{
			auto prev = blockIndex.load(std::memory_order_relaxed);
			size_t prevCapacity = prev == nullptr ? 0 : prev->capacity;
			auto entryCount = prev == nullptr ? nextBlockIndexCapacity : prevCapacity;
			auto raw = static_cast<char*>((Traits::malloc)(
				sizeof(BlockIndexHeader) +
				std::alignment_of<BlockIndexEntry>::value - 1 + sizeof(BlockIndexEntry) * entryCount +
				std::alignment_of<BlockIndexEntry*>::value - 1 + sizeof(BlockIndexEntry*) * nextBlockIndexCapacity));
			if (raw == nullptr) {
				return false;
			}
			
			auto header = new (raw) BlockIndexHeader;
			auto entries = reinterpret_cast<BlockIndexEntry*>(details::align_for<BlockIndexEntry>(raw + sizeof(BlockIndexHeader)));
			auto index = reinterpret_cast<BlockIndexEntry**>(details::align_for<BlockIndexEntry*>(reinterpret_cast<char*>(entries) + sizeof(BlockIndexEntry) * entryCount));
			if (prev != nullptr) {
				auto prevTail = prev->tail.load(std::memory_order_relaxed);
				auto prevPos = prevTail;
				size_t i = 0;
				do {
					prevPos = (prevPos + 1) & (prev->capacity - 1);
					index[i++] = prev->index[prevPos];
				} while (prevPos != prevTail);
				assert(i == prevCapacity);
			}
			for (size_t i = 0; i != entryCount; ++i) {
				new (entries + i) BlockIndexEntry;
				entries[i].key.store(INVALID_BLOCK_BASE, std::memory_order_relaxed);
				index[prevCapacity + i] = entries + i;
			}
			header->prev = prev;
			header->entries = entries;
			header->index = index;
			header->capacity = nextBlockIndexCapacity;
			header->tail.store((prevCapacity - 1) & (nextBlockIndexCapacity - 1), std::memory_order_relaxed);
			
			blockIndex.store(header, std::memory_order_release);
			
			nextBlockIndexCapacity <<= 1;
			
			return true;
		}
		
	private:
		size_t nextBlockIndexCapacity;
		std::atomic<BlockIndexHeader*> blockIndex;

#ifdef MOODYCAMEL_CPP11_THREAD_LOCAL_SUPPORTED
	public:
		details::ThreadExitListener threadExitListener;
	private:
#endif
		
#ifdef MOODYCAMEL_QUEUE_INTERNAL_DEBUG
	public:
		ImplicitProducer* nextImplicitProducer;
	private:
#endif

#ifdef MCDBGQ_NOLOCKFREE_IMPLICITPRODBLOCKINDEX
		mutable debug::DebugMutex mutex;
#endif
#ifdef MCDBGQ_TRACKMEM
		friend struct MemStats;
#endif
	};
	
	
	//////////////////////////////////
	// Block pool manipulation
	//////////////////////////////////
	
	void populate_initial_block_list(size_t blockCount)
	{
		initialBlockPoolSize = blockCount;
		if (initialBlockPoolSize == 0) {
			initialBlockPool = nullptr;
			return;
		}
		
		initialBlockPool = create_array<Block>(blockCount);
		if (initialBlockPool == nullptr) {
			initialBlockPoolSize = 0;
		}
		for (size_t i = 0; i < initialBlockPoolSize; ++i) {
			initialBlockPool[i].dynamicallyAllocated = false;
		}
	}
	
	inline Block* try_get_block_from_initial_pool()
	{
		if (initialBlockPoolIndex.load(std::memory_order_relaxed) >= initialBlockPoolSize) {
			return nullptr;
		}
		
		auto index = initialBlockPoolIndex.fetch_add(1, std::memory_order_relaxed);
		
		return index < initialBlockPoolSize ? (initialBlockPool + index) : nullptr;
	}
	
	inline void add_block_to_free_list(Block* block)
	{
#ifdef MCDBGQ_TRACKMEM
		block->owner = nullptr;
#endif
		if (!Traits::RECYCLE_ALLOCATED_BLOCKS && block->dynamicallyAllocated) {
			destroy(block);
		}
		else {
			freeList.add(block);
		}
	}
	
	inline void add_blocks_to_free_list(Block* block)
	{
		while (block != nullptr) {
			auto next = block->next;
			add_block_to_free_list(block);
			block = next;
		}
	}
	
	inline Block* try_get_block_from_free_list()
	{
		return freeList.try_get();
	}
	
	// Gets a free block from one of the memory pools, or allocates a new one (if applicable)
	template<AllocationMode canAlloc>
	Block* requisition_block()
	{
		auto block = try_get_block_from_initial_pool();
		if (block != nullptr) {
			return block;
		}
		
		block = try_get_block_from_free_list();
		if (block != nullptr) {
			return block;
		}
		
		MOODYCAMEL_CONSTEXPR_IF (canAlloc == CanAlloc) {
			return create<Block>();
		}
		else {
			return nullptr;
		}
	}
	

#ifdef MCDBGQ_TRACKMEM
	public:
		struct MemStats {
			size_t allocatedBlocks;
			size_t usedBlocks;
			size_t freeBlocks;
			size_t ownedBlocksExplicit;
			size_t ownedBlocksImplicit;
			size_t implicitProducers;
			size_t explicitProducers;
			size_t elementsEnqueued;
			size_t blockClassBytes;
			size_t queueClassBytes;
			size_t implicitBlockIndexBytes;
			size_t explicitBlockIndexBytes;
			
			friend class ConcurrentQueue;
			
		private:
			static MemStats getFor(ConcurrentQueue* q)
			{
				MemStats stats = { 0 };
				
				stats.elementsEnqueued = q->size_approx();
			
				auto block = q->freeList.head_unsafe();
				while (block != nullptr) {
					++stats.allocatedBlocks;
					++stats.freeBlocks;
					block = block->freeListNext.load(std::memory_order_relaxed);
				}
				
				for (auto ptr = q->producerListTail.load(std::memory_order_acquire); ptr != nullptr; ptr = ptr->next_prod()) {
					bool implicit = dynamic_cast<ImplicitProducer*>(ptr) != nullptr;
					stats.implicitProducers += implicit ? 1 : 0;
					stats.explicitProducers += implicit ? 0 : 1;
					
					if (implicit) {
						auto prod = static_cast<ImplicitProducer*>(ptr);
						stats.queueClassBytes += sizeof(ImplicitProducer);
						auto head = prod->headIndex.load(std::memory_order_relaxed);
						auto tail = prod->tailIndex.load(std::memory_order_relaxed);
						auto hash = prod->blockIndex.load(std::memory_order_relaxed);
						if (hash != nullptr) {
							for (size_t i = 0; i != hash->capacity; ++i) {
								if (hash->index[i]->key.load(std::memory_order_relaxed) != ImplicitProducer::INVALID_BLOCK_BASE && hash->index[i]->value.load(std::memory_order_relaxed) != nullptr) {
									++stats.allocatedBlocks;
									++stats.ownedBlocksImplicit;
								}
							}
							stats.implicitBlockIndexBytes += hash->capacity * sizeof(typename ImplicitProducer::BlockIndexEntry);
							for (; hash != nullptr; hash = hash->prev) {
								stats.implicitBlockIndexBytes += sizeof(typename ImplicitProducer::BlockIndexHeader) + hash->capacity * sizeof(typename ImplicitProducer::BlockIndexEntry*);
							}
						}
						for (; details::circular_less_than<index_t>(head, tail); head += BLOCK_SIZE) {
							//auto block = prod->get_block_index_entry_for_index(head);
							++stats.usedBlocks;
						}
					}
					else {
						auto prod = static_cast<ExplicitProducer*>(ptr);
						stats.queueClassBytes += sizeof(ExplicitProducer);
						auto tailBlock = prod->tailBlock;
						bool wasNonEmpty = false;
						if (tailBlock != nullptr) {
							auto block = tailBlock;
							do {
								++stats.allocatedBlocks;
								if (!block->ConcurrentQueue::Block::template is_empty<explicit_context>() || wasNonEmpty) {
									++stats.usedBlocks;
									wasNonEmpty = wasNonEmpty || block != tailBlock;
								}
								++stats.ownedBlocksExplicit;
								block = block->next;
							} while (block != tailBlock);
						}
						auto index = prod->blockIndex.load(std::memory_order_relaxed);
						while (index != nullptr) {
							stats.explicitBlockIndexBytes += sizeof(typename ExplicitProducer::BlockIndexHeader) + index->size * sizeof(typename ExplicitProducer::BlockIndexEntry);
							index = static_cast<typename ExplicitProducer::BlockIndexHeader*>(index->prev);
						}
					}
				}
				
				auto freeOnInitialPool = q->initialBlockPoolIndex.load(std::memory_order_relaxed) >= q->initialBlockPoolSize ? 0 : q->initialBlockPoolSize - q->initialBlockPoolIndex.load(std::memory_order_relaxed);
				stats.allocatedBlocks += freeOnInitialPool;
				stats.freeBlocks += freeOnInitialPool;
				
				stats.blockClassBytes = sizeof(Block) * stats.allocatedBlocks;
				stats.queueClassBytes += sizeof(ConcurrentQueue);
				
				return stats;
			}
		};
		
		// For debugging only. Not thread-safe.
		MemStats getMemStats()
		{
			return MemStats::getFor(this);
		}
	private:
		friend struct MemStats;
#endif
	
	
	//////////////////////////////////
	// Producer list manipulation
	//////////////////////////////////	
	
	ProducerBase* recycle_or_create_producer(bool isExplicit)
	{
#ifdef MCDBGQ_NOLOCKFREE_IMPLICITPRODHASH
		debug::DebugLock lock(implicitProdMutex);
#endif
		// Try to re-use one first
		for (auto ptr = producerListTail.load(std::memory_order_acquire); ptr != nullptr; ptr = ptr->next_prod()) {
			if (ptr->inactive.load(std::memory_order_relaxed) && ptr->isExplicit == isExplicit) {
				bool expected = true;
				if (ptr->inactive.compare_exchange_strong(expected, /* desired */ false, std::memory_order_acquire, std::memory_order_relaxed)) {
					// We caught one! It's been marked as activated, the caller can have it
					return ptr;
				}
			}
		}

		return add_producer(isExplicit ? static_cast<ProducerBase*>(create<ExplicitProducer>(this)) : create<ImplicitProducer>(this));
	}
	
	ProducerBase* add_producer(ProducerBase* producer)
	{
		// Handle failed memory allocation
		if (producer == nullptr) {
			return nullptr;
		}
		
		producerCount.fetch_add(1, std::memory_order_relaxed);
		
		// Add it to the lock-free list
		auto prevTail = producerListTail.load(std::memory_order_relaxed);
		do {
			producer->next = prevTail;
		} while (!producerListTail.compare_exchange_weak(prevTail, producer, std::memory_order_release, std::memory_order_relaxed));
		
#ifdef MOODYCAMEL_QUEUE_INTERNAL_DEBUG
		if (producer->isExplicit) {
			auto prevTailExplicit = explicitProducers.load(std::memory_order_relaxed);
			do {
				static_cast<ExplicitProducer*>(producer)->nextExplicitProducer = prevTailExplicit;
			} while (!explicitProducers.compare_exchange_weak(prevTailExplicit, static_cast<ExplicitProducer*>(producer), std::memory_order_release, std::memory_order_relaxed));
		}
		else {
			auto prevTailImplicit = implicitProducers.load(std::memory_order_relaxed);
			do {
				static_cast<ImplicitProducer*>(producer)->nextImplicitProducer = prevTailImplicit;
			} while (!implicitProducers.compare_exchange_weak(prevTailImplicit, static_cast<ImplicitProducer*>(producer), std::memory_order_release, std::memory_order_relaxed));
		}
#endif
		
		return producer;
	}
	
	void reown_producers()
	{
		// After another instance is moved-into/swapped-with this one, all the
		// producers we stole still think their parents are the other queue.
		// So fix them up!
		for (auto ptr = producerListTail.load(std::memory_order_relaxed); ptr != nullptr; ptr = ptr->next_prod()) {
			ptr->parent = this;
		}
	}
	
	
	//////////////////////////////////
	// Implicit producer hash
	//////////////////////////////////
	
	struct ImplicitProducerKVP
	{
		std::atomic<details::thread_id_t> key;
		ImplicitProducer* value;		// No need for atomicity since it's only read by the thread that sets it in the first place
		
		ImplicitProducerKVP() : value(nullptr) { }
		
		ImplicitProducerKVP(ImplicitProducerKVP&& other) MOODYCAMEL_NOEXCEPT
		{
			key.store(other.key.load(std::memory_order_relaxed), std::memory_order_relaxed);
			value = other.value;
		}
		
		inline ImplicitProducerKVP& operator=(ImplicitProducerKVP&& other) MOODYCAMEL_NOEXCEPT
		{
			swap(other);
			return *this;
		}
		
		inline void swap(ImplicitProducerKVP& other) MOODYCAMEL_NOEXCEPT
		{
			if (this != &other) {
				details::swap_relaxed(key, other.key);
				std::swap(value, other.value);
			}
		}
	};
	
	template<typename XT, typename XTraits>
	friend void moodycamel::swap(typename ConcurrentQueue<XT, XTraits>::ImplicitProducerKVP&, typename ConcurrentQueue<XT, XTraits>::ImplicitProducerKVP&) MOODYCAMEL_NOEXCEPT;
	
	struct ImplicitProducerHash
	{
		size_t capacity;
		ImplicitProducerKVP* entries;
		ImplicitProducerHash* prev;
	};
	
	inline void populate_initial_implicit_producer_hash()
	{
		MOODYCAMEL_CONSTEXPR_IF (INITIAL_IMPLICIT_PRODUCER_HASH_SIZE == 0) {
			return;
		}
		else {
			implicitProducerHashCount.store(0, std::memory_order_relaxed);
			auto hash = &initialImplicitProducerHash;
			hash->capacity = INITIAL_IMPLICIT_PRODUCER_HASH_SIZE;
			hash->entries = &initialImplicitProducerHashEntries[0];
			for (size_t i = 0; i != INITIAL_IMPLICIT_PRODUCER_HASH_SIZE; ++i) {
				initialImplicitProducerHashEntries[i].key.store(details::invalid_thread_id, std::memory_order_relaxed);
			}
			hash->prev = nullptr;
			implicitProducerHash.store(hash, std::memory_order_relaxed);
		}
	}
	
	void swap_implicit_producer_hashes(ConcurrentQueue& other)
	{
		MOODYCAMEL_CONSTEXPR_IF (INITIAL_IMPLICIT_PRODUCER_HASH_SIZE == 0) {
			return;
		}
		else {
			// Swap (assumes our implicit producer hash is initialized)
			initialImplicitProducerHashEntries.swap(other.initialImplicitProducerHashEntries);
			initialImplicitProducerHash.entries = &initialImplicitProducerHashEntries[0];
			other.initialImplicitProducerHash.entries = &other.initialImplicitProducerHashEntries[0];
			
			details::swap_relaxed(implicitProducerHashCount, other.implicitProducerHashCount);
			
			details::swap_relaxed(implicitProducerHash, other.implicitProducerHash);
			if (implicitProducerHash.load(std::memory_order_relaxed) == &other.initialImplicitProducerHash) {
				implicitProducerHash.store(&initialImplicitProducerHash, std::memory_order_relaxed);
			}
			else {
				ImplicitProducerHash* hash;
				for (hash = implicitProducerHash.load(std::memory_order_relaxed); hash->prev != &other.initialImplicitProducerHash; hash = hash->prev) {
					continue;
				}
				hash->prev = &initialImplicitProducerHash;
			}
			if (other.implicitProducerHash.load(std::memory_order_relaxed) == &initialImplicitProducerHash) {
				other.implicitProducerHash.store(&other.initialImplicitProducerHash, std::memory_order_relaxed);
			}
			else {
				ImplicitProducerHash* hash;
				for (hash = other.implicitProducerHash.load(std::memory_order_relaxed); hash->prev != &initialImplicitProducerHash; hash = hash->prev) {
					continue;
				}
				hash->prev = &other.initialImplicitProducerHash;
			}
		}
	}
	
	// Only fails (returns nullptr) if memory allocation fails
	ImplicitProducer* get_or_add_implicit_producer()
	{
		// Note that since the data is essentially thread-local (key is thread ID),
		// there's a reduced need for fences (memory ordering is already consistent
		// for any individual thread), except for the current table itself.
		
		// Start by looking for the thread ID in the current and all previous hash tables.
		// If it's not found, it must not be in there yet, since this same thread would
		// have added it previously to one of the tables that we traversed.
		
		// Code and algorithm adapted from http://preshing.com/20130605/the-worlds-simplest-lock-free-hash-table
		
#ifdef MCDBGQ_NOLOCKFREE_IMPLICITPRODHASH
		debug::DebugLock lock(implicitProdMutex);
#endif
		
		auto id = details::thread_id();
		auto hashedId = details::hash_thread_id(id);
		
		auto mainHash = implicitProducerHash.load(std::memory_order_acquire);
		assert(mainHash != nullptr);  // silence clang-tidy and MSVC warnings (hash cannot be null)
		for (auto hash = mainHash; hash != nullptr; hash = hash->prev) {
			// Look for the id in this hash
			auto index = hashedId;
			while (true) {		// Not an infinite loop because at least one slot is free in the hash table
				index &= hash->capacity - 1u;
				
				auto probedKey = hash->entries[index].key.load(std::memory_order_relaxed);
				if (probedKey == id) {
					// Found it! If we had to search several hashes deep, though, we should lazily add it
					// to the current main hash table to avoid the extended search next time.
					// Note there's guaranteed to be room in the current hash table since every subsequent
					// table implicitly reserves space for all previous tables (there's only one
					// implicitProducerHashCount).
					auto value = hash->entries[index].value;
					if (hash != mainHash) {
						index = hashedId;
						while (true) {
							index &= mainHash->capacity - 1u;
							auto empty = details::invalid_thread_id;
#ifdef MOODYCAMEL_CPP11_THREAD_LOCAL_SUPPORTED
							auto reusable = details::invalid_thread_id2;
							if (mainHash->entries[index].key.compare_exchange_strong(empty,    id, std::memory_order_seq_cst, std::memory_order_relaxed) ||
								mainHash->entries[index].key.compare_exchange_strong(reusable, id, std::memory_order_seq_cst, std::memory_order_relaxed)) {
#else
							if (mainHash->entries[index].key.compare_exchange_strong(empty,    id, std::memory_order_seq_cst, std::memory_order_relaxed)) {
#endif
								mainHash->entries[index].value = value;
								break;
							}
							++index;
						}
					}
					
					return value;
				}
				if (probedKey == details::invalid_thread_id) {
					break;		// Not in this hash table
				}
				++index;
			}
		}
		
		// Insert!
		auto newCount = 1 + implicitProducerHashCount.fetch_add(1, std::memory_order_relaxed);
		while (true) {
			// NOLINTNEXTLINE(clang-analyzer-core.NullDereference)
			if (newCount >= (mainHash->capacity >> 1) && !implicitProducerHashResizeInProgress.test_and_set(std::memory_order_acquire)) {
				// We've acquired the resize lock, try to allocate a bigger hash table.
				// Note the acquire fence synchronizes with the release fence at the end of this block, and hence when
				// we reload implicitProducerHash it must be the most recent version (it only gets changed within this
				// locked block).
				mainHash = implicitProducerHash.load(std::memory_order_acquire);
				if (newCount >= (mainHash->capacity >> 1)) {
					size_t newCapacity = mainHash->capacity << 1;
					while (newCount >= (newCapacity >> 1)) {
						newCapacity <<= 1;
					}
					auto raw = static_cast<char*>((Traits::malloc)(sizeof(ImplicitProducerHash) + std::alignment_of<ImplicitProducerKVP>::value - 1 + sizeof(ImplicitProducerKVP) * newCapacity));
					if (raw == nullptr) {
						// Allocation failed
						implicitProducerHashCount.fetch_sub(1, std::memory_order_relaxed);
						implicitProducerHashResizeInProgress.clear(std::memory_order_relaxed);
						return nullptr;
					}
					
					auto newHash = new (raw) ImplicitProducerHash;
					newHash->capacity = static_cast<size_t>(newCapacity);
					newHash->entries = reinterpret_cast<ImplicitProducerKVP*>(details::align_for<ImplicitProducerKVP>(raw + sizeof(ImplicitProducerHash)));
					for (size_t i = 0; i != newCapacity; ++i) {
						new (newHash->entries + i) ImplicitProducerKVP;
						newHash->entries[i].key.store(details::invalid_thread_id, std::memory_order_relaxed);
					}
					newHash->prev = mainHash;
					implicitProducerHash.store(newHash, std::memory_order_release);
					implicitProducerHashResizeInProgress.clear(std::memory_order_release);
					mainHash = newHash;
				}
				else {
					implicitProducerHashResizeInProgress.clear(std::memory_order_release);
				}
			}
			
			// If it's < three-quarters full, add to the old one anyway so that we don't have to wait for the next table
			// to finish being allocated by another thread (and if we just finished allocating above, the condition will
			// always be true)
			if (newCount < (mainHash->capacity >> 1) + (mainHash->capacity >> 2)) {
				auto producer = static_cast<ImplicitProducer*>(recycle_or_create_producer(false));
				if (producer == nullptr) {
					implicitProducerHashCount.fetch_sub(1, std::memory_order_relaxed);
					return nullptr;
				}
				
#ifdef MOODYCAMEL_CPP11_THREAD_LOCAL_SUPPORTED
				producer->threadExitListener.callback = &ConcurrentQueue::implicit_producer_thread_exited_callback;
				producer->threadExitListener.userData = producer;
				details::ThreadExitNotifier::subscribe(&producer->threadExitListener);
#endif
				
				auto index = hashedId;
				while (true) {
					index &= mainHash->capacity - 1u;
					auto empty = details::invalid_thread_id;
#ifdef MOODYCAMEL_CPP11_THREAD_LOCAL_SUPPORTED
					auto reusable = details::invalid_thread_id2;
					if (mainHash->entries[index].key.compare_exchange_strong(reusable, id, std::memory_order_seq_cst, std::memory_order_relaxed)) {
						implicitProducerHashCount.fetch_sub(1, std::memory_order_relaxed);  // already counted as a used slot
						mainHash->entries[index].value = producer;
						break;
					}
#endif
					if (mainHash->entries[index].key.compare_exchange_strong(empty,    id, std::memory_order_seq_cst, std::memory_order_relaxed)) {
						mainHash->entries[index].value = producer;
						break;
					}
					++index;
				}
				return producer;
			}
			
			// Hmm, the old hash is quite full and somebody else is busy allocating a new one.
			// We need to wait for the allocating thread to finish (if it succeeds, we add, if not,
			// we try to allocate ourselves).
			mainHash = implicitProducerHash.load(std::memory_order_acquire);
		}
	}
	
#ifdef MOODYCAMEL_CPP11_THREAD_LOCAL_SUPPORTED
	void implicit_producer_thread_exited(ImplicitProducer* producer)
	{
		// Remove from hash
#ifdef MCDBGQ_NOLOCKFREE_IMPLICITPRODHASH
		debug::DebugLock lock(implicitProdMutex);
#endif
		auto hash = implicitProducerHash.load(std::memory_order_acquire);
		assert(hash != nullptr);		// The thread exit listener is only registered if we were added to a hash in the first place
		auto id = details::thread_id();
		auto hashedId = details::hash_thread_id(id);
		details::thread_id_t probedKey;
		
		// We need to traverse all the hashes just in case other threads aren't on the current one yet and are
		// trying to add an entry thinking there's a free slot (because they reused a producer)
		for (; hash != nullptr; hash = hash->prev) {
			auto index = hashedId;
			do {
				index &= hash->capacity - 1u;
				probedKey = id;
				if (hash->entries[index].key.compare_exchange_strong(probedKey, details::invalid_thread_id2, std::memory_order_seq_cst, std::memory_order_relaxed)) {
					break;
				}
				++index;
			} while (probedKey != details::invalid_thread_id);		// Can happen if the hash has changed but we weren't put back in it yet, or if we weren't added to this hash in the first place
		}
		
		// Mark the queue as being recyclable
		producer->inactive.store(true, std::memory_order_release);
	}
	
	static void implicit_producer_thread_exited_callback(void* userData)
	{
		auto producer = static_cast<ImplicitProducer*>(userData);
		auto queue = producer->parent;
		queue->implicit_producer_thread_exited(producer);
	}
#endif
	
	//////////////////////////////////
	// Utility functions
	//////////////////////////////////

	template<typename TAlign>
	static inline void* aligned_malloc(size_t size)
	{
		MOODYCAMEL_CONSTEXPR_IF (std::alignment_of<TAlign>::value <= std::alignment_of<details::max_align_t>::value)
			return (Traits::malloc)(size);
		else {
			size_t alignment = std::alignment_of<TAlign>::value;
			void* raw = (Traits::malloc)(size + alignment - 1 + sizeof(void*));
			if (!raw)
				return nullptr;
			char* ptr = details::align_for<TAlign>(reinterpret_cast<char*>(raw) + sizeof(void*));
			*(reinterpret_cast<void**>(ptr) - 1) = raw;
			return ptr;
		}
	}

	template<typename TAlign>
	static inline void aligned_free(void* ptr)
	{
		MOODYCAMEL_CONSTEXPR_IF (std::alignment_of<TAlign>::value <= std::alignment_of<details::max_align_t>::value)
			return (Traits::free)(ptr);
		else
			(Traits::free)(ptr ? *(reinterpret_cast<void**>(ptr) - 1) : nullptr);
	}

	template<typename U>
	static inline U* create_array(size_t count)
	{
		assert(count > 0);
		U* p = static_cast<U*>(aligned_malloc<U>(sizeof(U) * count));
		if (p == nullptr)
			return nullptr;

		for (size_t i = 0; i != count; ++i)
			new (p + i) U();
		return p;
	}

	template<typename U>
	static inline void destroy_array(U* p, size_t count)
	{
		if (p != nullptr) {
			assert(count > 0);
			for (size_t i = count; i != 0; )
				(p + --i)->~U();
		}
		aligned_free<U>(p);
	}

	template<typename U>
	static inline U* create()
	{
		void* p = aligned_malloc<U>(sizeof(U));
		return p != nullptr ? new (p) U : nullptr;
	}

	template<typename U, typename A1>
	static inline U* create(A1&& a1)
	{
		void* p = aligned_malloc<U>(sizeof(U));
		return p != nullptr ? new (p) U(std::forward<A1>(a1)) : nullptr;
	}

	template<typename U>
	static inline void destroy(U* p)
	{
		if (p != nullptr)
			p->~U();
		aligned_free<U>(p);
	}

private:
	std::atomic<ProducerBase*> producerListTail;
	std::atomic<std::uint32_t> producerCount;
	
	std::atomic<size_t> initialBlockPoolIndex;
	Block* initialBlockPool;
	size_t initialBlockPoolSize;
	
#ifndef MCDBGQ_USEDEBUGFREELIST
	FreeList<Block> freeList;
#else
	debug::DebugFreeList<Block> freeList;
#endif
	
	std::atomic<ImplicitProducerHash*> implicitProducerHash;
	std::atomic<size_t> implicitProducerHashCount;		// Number of slots logically used
	ImplicitProducerHash initialImplicitProducerHash;
	std::array<ImplicitProducerKVP, INITIAL_IMPLICIT_PRODUCER_HASH_SIZE> initialImplicitProducerHashEntries;
	std::atomic_flag implicitProducerHashResizeInProgress;
	
	std::atomic<std::uint32_t> nextExplicitConsumerId;
	std::atomic<std::uint32_t> globalExplicitConsumerOffset;
	
#ifdef MCDBGQ_NOLOCKFREE_IMPLICITPRODHASH
	debug::DebugMutex implicitProdMutex;
#endif
	
#ifdef MOODYCAMEL_QUEUE_INTERNAL_DEBUG
	std::atomic<ExplicitProducer*> explicitProducers;
	std::atomic<ImplicitProducer*> implicitProducers;
#endif
};


template<typename T, typename Traits>
ProducerToken::ProducerToken(ConcurrentQueue<T, Traits>& queue)
	: producer(queue.recycle_or_create_producer(true))
{
	if (producer != nullptr) {
		producer->token = this;
	}
}

template<typename T, typename Traits>
ProducerToken::ProducerToken(BlockingConcurrentQueue<T, Traits>& queue)
	: producer(reinterpret_cast<ConcurrentQueue<T, Traits>*>(&queue)->recycle_or_create_producer(true))
{
	if (producer != nullptr) {
		producer->token = this;
	}
}

template<typename T, typename Traits>
ConsumerToken::ConsumerToken(ConcurrentQueue<T, Traits>& queue)
	: itemsConsumedFromCurrent(0), currentProducer(nullptr), desiredProducer(nullptr)
{
	initialOffset = queue.nextExplicitConsumerId.fetch_add(1, std::memory_order_release);
	lastKnownGlobalOffset = static_cast<std::uint32_t>(-1);
}

template<typename T, typename Traits>
ConsumerToken::ConsumerToken(BlockingConcurrentQueue<T, Traits>& queue)
	: itemsConsumedFromCurrent(0), currentProducer(nullptr), desiredProducer(nullptr)
{
	initialOffset = reinterpret_cast<ConcurrentQueue<T, Traits>*>(&queue)->nextExplicitConsumerId.fetch_add(1, std::memory_order_release);
	lastKnownGlobalOffset = static_cast<std::uint32_t>(-1);
}

template<typename T, typename Traits>
inline void swap(ConcurrentQueue<T, Traits>& a, ConcurrentQueue<T, Traits>& b) MOODYCAMEL_NOEXCEPT
{
	a.swap(b);
}

inline void swap(ProducerToken& a, ProducerToken& b) MOODYCAMEL_NOEXCEPT
{
	a.swap(b);
}

inline void swap(ConsumerToken& a, ConsumerToken& b) MOODYCAMEL_NOEXCEPT
{
	a.swap(b);
}

template<typename T, typename Traits>
inline void swap(typename ConcurrentQueue<T, Traits>::ImplicitProducerKVP& a, typename ConcurrentQueue<T, Traits>::ImplicitProducerKVP& b) MOODYCAMEL_NOEXCEPT
{
	a.swap(b);
}

}

#if defined(_MSC_VER) && (!defined(_HAS_CXX17) || !_HAS_CXX17)
#pragma warning(pop)
#endif

#if defined(__GNUC__) && !defined(__INTEL_COMPILER)
#pragma GCC diagnostic pop
#endif
