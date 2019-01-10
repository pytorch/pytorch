#include "caffe2/core/numa.h"

CAFFE2_DEFINE_bool(
    caffe2_cpu_numa_enabled,
    false,
    "Use NUMA whenever possible.");

#if defined(__linux__) && !defined(CAFFE2_DISABLE_NUMA) && CAFFE2_MOBILE == 0
#include <numa.h>
#include <numaif.h>
#define CAFFE2_NUMA_ENABLED
#endif

namespace caffe2 {

#ifdef CAFFE2_NUMA_ENABLED
bool IsNUMAEnabled() {
  return FLAGS_caffe2_cpu_numa_enabled && numa_available() >= 0;
}

void NUMABind(int numa_node_id) {
  if (numa_node_id < 0) {
    return;
  }
  if (!IsNUMAEnabled()) {
    VLOG(1) << "NUMA is not enabled";
    return;
  }

  CAFFE_ENFORCE(
      numa_node_id <= numa_max_node(),
      "NUMA node id " + caffe2::to_string(numa_node_id) + " is unavailable");

  auto bm = numa_allocate_nodemask();
  numa_bitmask_clearall(bm);
  numa_bitmask_setbit(bm, numa_node_id);
  numa_bind(bm);
  numa_bitmask_free(bm);
}

int GetNUMANode(const void* ptr) {
  if (!IsNUMAEnabled()) {
    VLOG(1) << "NUMA is not enabled";
    return -1;
  }
  CAFFE_ENFORCE(ptr);

  int numa_node = -1;
  CAFFE_ENFORCE(
      get_mempolicy(
          &numa_node, NULL, 0, (void*)ptr, MPOL_F_NODE | MPOL_F_ADDR) == 0,
      "Unable to get memory policy");
  return numa_node;
}

int GetNumNUMANodes() {
  if (!IsNUMAEnabled()) {
    VLOG(1) << "NUMA is not enabled";
    return -1;
  }

  return numa_num_configured_nodes();
}

void NUMAMove(void* ptr, size_t size, int numa_node_id) {
  if (numa_node_id < 0) {
    return;
  }
  if (!IsNUMAEnabled()) {
    VLOG(1) << "NUMA is not enabled";
    return;
  }
  CAFFE_ENFORCE(ptr);

  size_t page_start_ptr = (((size_t)ptr) & ~(getpagesize() - 1));
  size_t offset = ((size_t)ptr) - page_start_ptr;
  // Avoid extra dynamic allocation and NUMA api calls
  CAFFE_ENFORCE(numa_node_id >= 0 && numa_node_id < sizeof(unsigned long) * 8);
  unsigned long mask = 1UL << numa_node_id;
  CAFFE_ENFORCE(
      mbind(
          (void*)page_start_ptr,
          size + offset,
          MPOL_BIND,
          &mask,
          sizeof(mask) * 8,
          MPOL_MF_MOVE | MPOL_MF_STRICT) == 0,
      "Could not move memory to a NUMA node");
}

int GetCurrentNUMANode() {
  if (!IsNUMAEnabled()) {
    VLOG(1) << "NUMA is not enabled";
    return -1;
  }

  return numa_node_of_cpu(sched_getcpu());
}

#else // CAFFE2_NUMA_ENABLED

bool IsNUMAEnabled() {
  return false;
}

void NUMABind(int numa_node_id) {
  if (numa_node_id >= 0) {
    VLOG(1) << "NUMA is not enabled";
  }
}

int GetNUMANode(const void* ptr) {
  VLOG(1) << "NUMA is not enabled";
  return -1;
}

int GetNumNUMANodes() {
  VLOG(1) << "NUMA is not enabled";
  return -1;
}

void NUMAMove(void* ptr, size_t size, int numa_node_id) {
  if (numa_node_id >= 0) {
    VLOG(1) << "NUMA is not enabled";
  }
}

int GetCurrentNUMANode() {
  VLOG(1) << "NUMA is not enabled";
  return -1;
}

#endif // CAFFE2_NUMA_ENABLED

} // namespace caffe2
