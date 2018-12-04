#pragma once

#include <c10/util/Logging.h>
#include <c10/util/Optional.h>

namespace c10 {

/**
 * Check whether NUMA is enabled
 */
C10_API bool IsNUMAEnabled();

/**
 * Bind to a given NUMA node
 * Assumes: numa_node_id will always be valid
 */
C10_API void NUMABind(int numa_node_id);

/**
 * Get the NUMA id for a given pointer `ptr`
 * returns c10::nullopt when NUMA is not enabled
 */
C10_API c10::optional<int> GetNUMANode(const void* ptr);

/**
 * Get number of currently configured NUMA nodes
 * returns c10::nullopt when NUMA is not enabled
 */
C10_API c10::optional<int> GetNumNUMANodes();

/**
 * Move the memory pointed to by `ptr` of a given size to current NUMA node
 */
C10_API void NUMAMoveToCurrent(void* ptr, size_t size);

/**
 * Move the memory pointed to by `ptr` of a given size to another NUMA node
 * Assumes: numa_node_id will always be valid
 */
C10_API void NUMAMove(void* ptr, size_t size, int numa_node_id);

/**
 * Get the NUMA node id that the current cpu belongs to
 * returns c10::nullopt when NUMA is not enabled or the current cpu is invalid
 */
C10_API c10::optional<int> GetCurrentNUMANode();

} // namespace c10
