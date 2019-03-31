#include "caffe2/distributed/file_store_handler_op.h"

#ifndef __HIP_PLATFORM_HCC__
#include <caffe2/core/context_gpu.h>
#else
#include <caffe2/core/hip/context_gpu.h>
#endif

namespace caffe2 {

#ifndef __HIP_PLATFORM_HCC__
REGISTER_CUDA_OPERATOR(
    FileStoreHandlerCreate,
    FileStoreHandlerCreateOp<CUDAContext>);
#else
REGISTER_HIP_OPERATOR(
    FileStoreHandlerCreate,
    FileStoreHandlerCreateOp<HIPContext>);
#endif

} // namespace caffe2
