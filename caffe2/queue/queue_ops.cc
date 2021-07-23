#include "queue_ops.h"
#include <memory>
#include "caffe2/utils/math.h"

namespace caffe2 {

CAFFE_KNOWN_TYPE(std::shared_ptr<BlobsQueue>);

REGISTER_CPU_OPERATOR(CreateBlobsQueue, CreateBlobsQueueOp<CPUContext>);
REGISTER_CPU_OPERATOR(EnqueueBlobs, EnqueueBlobsOp<CPUContext>);
REGISTER_CPU_OPERATOR(DequeueBlobs, DequeueBlobsOp<CPUContext>);
REGISTER_CPU_OPERATOR(CloseBlobsQueue, CloseBlobsQueueOp<CPUContext>);

REGISTER_CPU_OPERATOR(SafeEnqueueBlobs, SafeEnqueueBlobsOp<CPUContext>);
REGISTER_CPU_OPERATOR(SafeDequeueBlobs, SafeDequeueBlobsOp<CPUContext>);
REGISTER_CPU_OPERATOR(
    WeightedSampleDequeueBlobs,
    WeightedSampleDequeueBlobsOp<CPUContext>);

OPERATOR_SCHEMA(CreateBlobsQueue).NumInputs(0).NumOutputs(1);
OPERATOR_SCHEMA(EnqueueBlobs)
    .NumInputsOutputs([](int inputs, int outputs) {
      return inputs >= 2 && outputs >= 1 && inputs == outputs + 1;
    })
    .EnforceInplace([](int input, int output) { return input == output + 1; });
OPERATOR_SCHEMA(DequeueBlobs)
    .NumInputsOutputs([](int inputs, int outputs) {
      return inputs == 1 && outputs >= 1;
    })
    .SetDoc(R"DOC(
  Dequeue the blobs from queue.
  )DOC")
    .Arg("timeout_secs", "Timeout in secs, default: no timeout")
    .Input(0, "queue", "The shared pointer for the BlobsQueue")
    .Output(0, "blob", "The blob to store the dequeued data");

OPERATOR_SCHEMA(CloseBlobsQueue).NumInputs(1).NumOutputs(0);

OPERATOR_SCHEMA(SafeEnqueueBlobs)
    .NumInputsOutputs([](int inputs, int outputs) {
      return inputs >= 2 && outputs >= 2 && inputs == outputs;
    })
    .EnforceInplace([](int input, int output) { return input == output + 1; })
    .SetDoc(R"DOC(
Enqueue the blobs into queue. When the queue is closed and full, the output
status will be set to true which can be used as exit criteria for execution
step.
The 1st input is the queue and the last output is the status. The rest are
data blobs.
)DOC")
    .Input(0, "queue", "The shared pointer for the BlobsQueue");

OPERATOR_SCHEMA(SafeDequeueBlobs)
    .NumInputsOutputs([](int inputs, int outputs) {
      return inputs == 1 && outputs >= 2;
    })
    .SetDoc(R"DOC(
Dequeue the blobs from queue. When the queue is closed and empty, the output
status will be set to true which can be used as exit criteria for execution
step.
The 1st input is the queue and the last output is the status. The rest are
data blobs.
)DOC")
    .Arg(
        "num_records",
        "(default 1) If > 1, multiple records will be dequeued and tensors "
        "for each column will be concatenated. This requires all tensors in "
        "the records to be at least 1D, and to have the same inner dimensions.")
    .Input(0, "queue", "The shared pointer for the BlobsQueue")
    .Output(0, "blob", "The blob to store the dequeued data")
    .Output(1, "status", "Is set to 0/1 depending on the success of dequeue");

OPERATOR_SCHEMA(WeightedSampleDequeueBlobs)
    .NumInputs(1, INT_MAX)
    .NumOutputs(2, INT_MAX)
    .SetDoc(R"DOC(
Dequeue the blobs from multiple queues. When one of queues is closed and empty,
the output status will be set to true which can be used as exit criteria for
execution step.
The 1st input is the queue and the last output is the status. The rest are
data blobs.
)DOC")
    .Arg("weights", "Weights for sampling from multiple queues")
    .Arg(
        "table_idx_blob",
        "The index of the blob (among the output blob list) "
        "that will be used to store the index of the table chosen to read the "
        "current batch.");

NO_GRADIENT(CreateBlobsQueue);
NO_GRADIENT(EnqueueBlobs);
NO_GRADIENT(DequeueBlobs);
NO_GRADIENT(CloseBlobsQueue);

NO_GRADIENT(SafeEnqueueBlobs);
NO_GRADIENT(SafeDequeueBlobs);
NO_GRADIENT(WeightedSampleDequeueBlobs);

}
