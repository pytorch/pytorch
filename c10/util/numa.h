#pragma once

#include <c10/macros/Export.h>
#include <c10/util/Flags.h>
#include <cstddef>

C10_DECLARE_bool(caffe2_cpu_numa_enabled);

namespace c10 {

/**
 * Check whether NUMA is enabled
 */
C10_API bool IsNUMAEnabled();

/**
 * Returns true when the platform exposes a usable NUMA API, regardless of
 * whether FLAGS_caffe2_cpu_numa_enabled has been set.  The queries below only
 * report topology, so gating them on an opt-in "use NUMA" flag would force
 * every caller that merely wants to know where it is running to also opt into
 * NUMA *placement*.  The functions that actually move or bind memory
 * (NUMABind, NUMAMove) keep the flag.
 */
C10_API bool IsNUMAAvailable();

/**
 * Bind to a given NUMA node
 */
C10_API void NUMABind(int numa_node_id);

/**
 * Get the NUMA id for a given pointer `ptr`
 */
C10_API int GetNUMANode(const void* ptr);

/**
 * Get number of NUMA nodes
 */
C10_API int GetNumNUMANodes();

/**
 * One past the highest NUMA node id the system can hand out, for sizing arrays
 * indexed by node id.  Node ids may be sparse, so this is not the same as
 * GetNumNUMANodes().
 */
C10_API int GetNUMANodeIdUpperBound();

/**
 * Move the memory pointed to by `ptr` of a given size to another NUMA node
 */
C10_API void NUMAMove(void* ptr, size_t size, int numa_node_id);

/**
 * Get the current NUMA node id
 */
C10_API int GetCurrentNUMANode();

/**
 * Same as GetCurrentNUMANode(), but returns -1 unless
 * FLAGS_caffe2_cpu_numa_enabled is set.  For callers on hot paths that only
 * want the node in order to act on it, so that they pay nothing when NUMA
 * placement is off.
 */
C10_API int GetCurrentNUMANodeIfEnabled();

} // namespace c10
