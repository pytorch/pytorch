#include "caffe2/core/context_gpu.h"
#include "caffe2/db/create_db_op.h"

namespace caffe2 {
REGISTER_CUDA_OPERATOR(CreateDB, CreateDBOp<CUDAContext>);
} // namespace caffe2
