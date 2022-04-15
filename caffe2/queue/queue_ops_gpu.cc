#include "caffe2/queue/queue_ops.h"
#include "caffe2/utils/math.h"

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

REGISTER_CUDA_OPERATOR(CreateBlobsQueue, CreateBlobsQueueOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(EnqueueBlobs, EnqueueBlobsOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(DequeueBlobs, DequeueBlobsOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(CloseBlobsQueue, CloseBlobsQueueOp<CUDAContext>);

REGISTER_CUDA_OPERATOR(SafeEnqueueBlobs, SafeEnqueueBlobsOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(SafeDequeueBlobs, SafeDequeueBlobsOp<CUDAContext>);

} // namespace caffe2
