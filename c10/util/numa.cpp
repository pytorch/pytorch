#include "c10/util/numa.h"

C10_DEFINE_bool(caffe2_cpu_numa_enabled, false, "Use NUMA whenever possible.");

#if defined(__linux__) && !defined(C10_DISABLE_NUMA) && C10_MOBILE == 0
#include <numa.h>
#include <numaif.h>
#include <unistd.h>
#define C10_ENABLE_NUMA
#endif

// This code used to have a lot of VLOGs. However, because allocation might be
// triggered during static initialization, it's unsafe to invoke VLOG here

namespace c10 {

#ifdef C10_ENABLE_NUMA
bool IsNUMAEnabled() {
  return FLAGS_caffe2_cpu_numa_enabled && numa_available() >= 0;
}

void NUMABind(int numa_node_id) {
  if (numa_node_id < 0) {
    return;
  }
  if (!IsNUMAEnabled()) {
    return;
  }

  AT_CHECK(
      numa_node_id <= numa_max_node(),
      "NUMA node id ",
      numa_node_id,
      " is unavailable");

  auto bm = numa_allocate_nodemask();
  numa_bitmask_setbit(bm, numa_node_id);
  numa_bind(bm);
  numa_bitmask_free(bm);
}

int GetNUMANode(const void* ptr) {
  if (!IsNUMAEnabled()) {
    return -1;
  }
  AT_ASSERT(ptr);

  int numa_node = -1;
  AT_CHECK(
      get_mempolicy(
          &numa_node,
          NULL,
          0,
          const_cast<void*>(ptr),
          MPOL_F_NODE | MPOL_F_ADDR) == 0,
      "Unable to get memory policy, errno:",
      errno);
  return numa_node;
}

int GetNumNUMANodes() {
  if (!IsNUMAEnabled()) {
    return -1;
  }

  return numa_num_configured_nodes();
}

void NUMAMove(void* ptr, size_t size, int numa_node_id) {
  if (numa_node_id < 0) {
    return;
  }
  if (!IsNUMAEnabled()) {
    return;
  }
  AT_ASSERT(ptr);

  uintptr_t page_start_ptr =
      ((reinterpret_cast<uintptr_t>(ptr)) & ~(getpagesize() - 1));
  ptrdiff_t offset = reinterpret_cast<uintptr_t>(ptr) - page_start_ptr;
  // Avoid extra dynamic allocation and NUMA api calls
  AT_ASSERT(
      numa_node_id >= 0 &&
      static_cast<unsigned>(numa_node_id) < sizeof(unsigned long) * 8);
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

int GetCurrentNUMANode() {
  if (!IsNUMAEnabled()) {
    return -1;
  }

  auto n = numa_node_of_cpu(sched_getcpu());
  return n;
}

#else // C10_ENABLE_NUMA

bool IsNUMAEnabled() {
  return false;
}

void NUMABind(int numa_node_id) {
}

int GetNUMANode(const void* ptr) {
  return -1;
}

int GetNumNUMANodes() {
  return -1;
}

void NUMAMove(void* ptr, size_t size, int numa_node_id) {
}

int GetCurrentNUMANode() {
  return -1;
}

#endif // C10_NUMA_ENABLED

} // namespace c10
