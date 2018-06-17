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

#include "tbb/concurrent_hash_map.h"

namespace tbb {

namespace internal {
#if !TBB_NO_LEGACY
struct hash_map_segment_base {
    typedef spin_rw_mutex segment_mutex_t;
    //! Type of a hash code.
    typedef size_t hashcode_t;
    //! Log2 of n_segment
    static const size_t n_segment_bits = 6;
    //! Maximum size of array of chains
    static const size_t max_physical_size = size_t(1)<<(8*sizeof(hashcode_t)-n_segment_bits);
    //! Mutex that protects this segment
    segment_mutex_t my_mutex;
    // Number of nodes
    atomic<size_t> my_logical_size;
    // Size of chains
    /** Always zero or a power of two */
    size_t my_physical_size;
    //! True if my_logical_size>=my_physical_size.
    /** Used to support Intel(R) Thread Checker. */
    bool __TBB_EXPORTED_METHOD internal_grow_predicate() const;
};

bool hash_map_segment_base::internal_grow_predicate() const {
    // Intel(R) Thread Checker considers the following reads to be races, so we hide them in the
    // library so that Intel(R) Thread Checker will ignore them.  The reads are used in a double-check
    // context, so the program is nonetheless correct despite the race.
    return my_logical_size >= my_physical_size && my_physical_size < max_physical_size;
}
#endif//!TBB_NO_LEGACY

} // namespace internal

} // namespace tbb

