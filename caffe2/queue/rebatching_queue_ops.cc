#include "rebatching_queue_ops.h"

namespace caffe2 {

CAFFE_KNOWN_TYPE(RebatchingQueuePtr);

namespace {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(CreateRebatchingQueue, CreateRebatchingQueueOp);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(EnqueueRebatchingQueue, EnqueueRebatchingQueueOp);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(DequeueRebatchingQueue, DequeueRebatchingQueueOp);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(CloseRebatchingQueue, CloseRebatchingQueueOp);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
NO_GRADIENT(CreateRebatchingQueue);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
NO_GRADIENT(EnqueueRebatchingQueue);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
NO_GRADIENT(DequeueRebatchingQueue);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
NO_GRADIENT(CloseRebatchingQueue);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(CreateRebatchingQueue)
    .NumInputs(0)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Creates the Queue.
)DOC")
    .Output(0, "queue", "object representing the queue")
    .Arg("num_blobs", "Number of input tensors the queue will support")
    .Arg(
        "capacity",
        "Maximal number of elements the queue can hold at any given point");

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(CloseRebatchingQueue)
    .NumInputs(1)
    .NumOutputs(0)
    .SetDoc(R"DOC(
Closes the Queue.
)DOC")
    .Input(0, "queue", "object representing the queue");

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(EnqueueRebatchingQueue)
    .NumInputs(2, INT_MAX)
    .NumOutputs(0)
    .SetDoc(R"DOC(
Enqueues Tensors into the queue.
Number of input tensors should be equal to the number of components passed
during creation of the queue.
If the Queue is closed this operation will fail.
If enqueue_batch argument is set. We will split the input tensors by the
first dimension to produce single queue elements.
)DOC")
    .Input(0, "queue", "object representing the queue")
    .Input(1, "tensor", "First tensor to enque. ")
    .Arg(
        "enqueue_batch",
        "Are we enqueuing a batch or just a single element. \
        By default we enqueue single element.");

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(DequeueRebatchingQueue)
    .NumInputs(1)
    .NumOutputs(1, INT_MAX)
    .SetDoc(R"DOC(
Dequeue Tensors from the Queue.
If the Queue is closed this might return less elements than asked.
If num_elements > 1 the returned elements will be concatenated into one
tensor per component.
)DOC")
    .Input(0, "rebatching_queue", "object representing the queue")
    .Input(1, "tensor", "First tensor to enqueue")
    .Arg(
        "num_elements",
        "Number of elements to dequeue. By default we dequeue one element.");
}
}
