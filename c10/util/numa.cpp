#include "c10/util/numa.h"

#if defined(__linux__) && !defined(C10_DISABLE_NUMA) && C10_MOBILE == 0
#include <numa.h>
#include <numaif.h>
#include <unistd.h>
#define C10_ENABLE_NUMA
#endif

namespace c10 {

#ifdef C10_ENABLE_NUMA
bool IsNUMAEnabled() {
  return numa_available() >= 0;
}

void NUMABind(int numa_node_id) {
  if (!IsNUMAEnabled()) {
    VLOG(1) << "NUMA is not enabled";
    return;
  }

  AT_CHECK(
      numa_node_id <= numa_max_node(),
      "NUMA node id ", numa_node_id, " is unavailable");

  auto bm = numa_allocate_nodemask();
  numa_bitmask_setbit(bm, numa_node_id);
  numa_bind(bm);
  numa_bitmask_free(bm);
}

c10::optional<int> GetNUMANode(const void* ptr) {
  if (!IsNUMAEnabled()) {
    VLOG(1) << "NUMA is not enabled";
    return c10::nullopt;
  }
  AT_ASSERT(ptr);

  int numa_node = -1;
  AT_CHECK(
      get_mempolicy(
          &numa_node, NULL, 0, const_cast<void*>(ptr), MPOL_F_NODE | MPOL_F_ADDR) == 0,
      "Unable to get memory policy, errno:", errno);
  if (numa_node == -1) {
    return c10::nullopt;
  }
  return numa_node;
}

c10::optional<int> GetNumNUMANodes() {
  if (!IsNUMAEnabled()) {
    VLOG(1) << "NUMA is not enabled";
    return c10::nullopt;
  }

  return numa_num_configured_nodes();
}

void NUMAMoveToCurrent(void* ptr, size_t size) {
  auto numa_node_id_opt = GetCurrentNUMANode();
  if (numa_node_id_opt) {
    NUMAMove(ptr, size, *numa_node_id_opt);
  }
}

void NUMAMove(void* ptr, size_t size, int numa_node_id) {
  if (!IsNUMAEnabled()) {
    VLOG(1) << "NUMA is not enabled";
    return;
  }
  AT_ASSERT(ptr);

  uintptr_t page_start_ptr = ((reinterpret_cast<uintptr_t>(ptr)) & ~(getpagesize() - 1));
  ptrdiff_t offset = reinterpret_cast<uintptr_t>(ptr) - page_start_ptr;
  // Avoid extra dynamic allocation and NUMA api calls
  AT_ASSERT(numa_node_id >= 0 && static_cast<unsigned>(numa_node_id) < sizeof(unsigned long) * 8);
  unsigned long mask = 1UL << numa_node_id;
  AT_CHECK(
      mbind(
          reinterpret_cast<void*>(page_start_ptr),
          size + offset,
          MPOL_BIND,
          &mask,
          sizeof(mask) * 8,
          MPOL_MF_MOVE | MPOL_MF_STRICT) == 0,
      "Could not move memory to a NUMA node");
}

c10::optional<int> GetCurrentNUMANode() {
  if (!IsNUMAEnabled()) {
    VLOG(1) << "NUMA is not enabled";
    return c10::nullopt;
  }

  auto n = numa_node_of_cpu(sched_getcpu());
  if (n == -1) {
    return c10::nullopt;
  }
  return n;
}

#else // C10_ENABLE_NUMA

bool IsNUMAEnabled() {
  return false;
}

void NUMABind(int numa_node_id) {
  VLOG(1) << "NUMA is not enabled";
}

c10::optional<int> GetNUMANode(const void* ptr) {
  VLOG(1) << "NUMA is not enabled";
  return c10::nullopt;
}

c10::optional<int> GetNumNUMANodes() {
  VLOG(1) << "NUMA is not enabled";
  return c10::nullopt;
}

void NUMAMoveToCurrent(void* ptr, size_t size) {
  VLOG(1) << "NUMA is not enabled";
}

void NUMAMove(void* ptr, size_t size, int numa_node_id) {
  VLOG(1) << "NUMA is not enabled";
}

c10::optional<int> GetCurrentNUMANode() {
  VLOG(1) << "NUMA is not enabled";
  return c10::nullopt;
}

#endif // C10_NUMA_ENABLED

} // namespace c10
