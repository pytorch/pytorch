#include "file_store_handler_op.h"

#include <caffe2/core/context_gpu.h>

namespace caffe2 {

REGISTER_CUDA_OPERATOR(
    FileStoreHandlerCreate,
    FileStoreHandlerCreateOp<CUDAContext>);

} // namespace caffe2
