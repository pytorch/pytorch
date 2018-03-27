#ifndef CAFFE2_CORE_NUMA_H_
#define CAFFE2_CORE_NUMA_H_

#include "caffe2/core/logging.h"

CAFFE2_DECLARE_bool(caffe2_cpu_numa_enabled);

namespace caffe2 {

bool IsNUMAEnabled();

void NUMABind(int numa_node_id);

int GetNUMANode(const void* ptr);

int GetNumNUMANodes();

void NUMAMove(void* ptr, size_t size, int numa_node_id);

int GetCurrentNUMANode();

} // namespace caffe2

#endif // CAFFE2_CORE_NUMA_H_
