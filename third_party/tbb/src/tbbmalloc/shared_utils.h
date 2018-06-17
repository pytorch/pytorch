/*
    Copyright (c) 2005-2018 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.




*/

#ifndef __TBB_shared_utils_H
#define __TBB_shared_utils_H

// Include files containing declarations of intptr_t and uintptr_t
#include <stddef.h>  // size_t
#if _MSC_VER
typedef unsigned __int16 uint16_t;
typedef unsigned __int32 uint32_t;
typedef unsigned __int64 uint64_t;
 #if !UINTPTR_MAX
  #define UINTPTR_MAX SIZE_MAX
 #endif
#else // _MSC_VER
#include <stdint.h>
#endif

/*
 * Functions to align an integer down or up to the given power of two,
 * and test for such an alignment, and for power of two.
 */
template<typename T>
static inline T alignDown(T arg, uintptr_t alignment) {
    return T( (uintptr_t)arg                & ~(alignment-1));
}
template<typename T>
static inline T alignUp  (T arg, uintptr_t alignment) {
    return T(((uintptr_t)arg+(alignment-1)) & ~(alignment-1));
    // /*is this better?*/ return (((uintptr_t)arg-1) | (alignment-1)) + 1;
}
template<typename T> // works for not power-of-2 alignments
static inline T alignUpGeneric(T arg, uintptr_t alignment) {
    if (size_t rem = arg % alignment) {
        arg += alignment - rem;
    }
    return arg;
}

template<typename T, size_t N> // generic function to find length of array
inline size_t arrayLength(const T(&)[N]) {
    return N;
}

#if defined(min)
#undef min
#endif

template<typename T>
T min ( const T& val1, const T& val2 ) {
    return val1 < val2 ? val1 : val2;
}

namespace rml {
namespace internal {

/*
 * Best estimate of cache line size, for the purpose of avoiding false sharing.
 * Too high causes memory overhead, too low causes false-sharing overhead.
 * Because, e.g., 32-bit code might run on a 64-bit system with a larger cache line size,
 * it would probably be better to probe at runtime where possible and/or allow for an environment variable override,
 * but currently this is still used for compile-time layout of class Block, so the change is not entirely trivial.
 */
#if __powerpc64__ || __ppc64__ || __bgp__
const uint32_t estimatedCacheLineSize = 128;
#else
const uint32_t estimatedCacheLineSize =  64;
#endif

}} // namespaces
#endif /* __TBB_shared_utils_H */
