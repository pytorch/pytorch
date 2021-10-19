#include "caffe2/distributed/file_store_handler_op.h"

#if !defined(USE_ROCM)
#include <caffe2/core/context_gpu.h>
#else
#include <caffe2/core/hip/context_gpu.h>
#endif

namespace caffe2 {

#if !defined(USE_ROCM)
REGISTER_CUDA_OPERATOR(
    FileStoreHandlerCreate,
    FileStoreHandlerCreateOp<CUDAContext>);
#else
REGISTER_HIP_OPERATOR(
    FileStoreHandlerCreate,
    FileStoreHandlerCreateOp<HIPContext>);
#endif

} // namespace caffe2
