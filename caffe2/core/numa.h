#ifndef CAFFE2_CORE_NUMA_H_
#define CAFFE2_CORE_NUMA_H_

#include "caffe2/core/logging.h"

C10_DECLARE_bool(caffe2_cpu_numa_enabled);

namespace caffe2 {

CAFFE2_API bool IsNUMAEnabled();

CAFFE2_API void NUMABind(int numa_node_id);

CAFFE2_API int GetNUMANode(const void* ptr);

CAFFE2_API int GetNumNUMANodes();

CAFFE2_API void NUMAMove(void* ptr, size_t size, int numa_node_id);

CAFFE2_API int GetCurrentNUMANode();

} // namespace caffe2

#endif // CAFFE2_CORE_NUMA_H_
