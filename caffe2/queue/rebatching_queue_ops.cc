/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "rebatching_queue_ops.h"

namespace caffe2 {

CAFFE_KNOWN_TYPE(RebatchingQueuePtr);

namespace {

REGISTER_CPU_OPERATOR(CreateRebatchingQueue, CreateRebatchingQueueOp);
REGISTER_CPU_OPERATOR(EnqueueRebatchingQueue, EnqueueRebatchingQueueOp);
REGISTER_CPU_OPERATOR(DequeueRebatchingQueue, DequeueRebatchingQueueOp);
REGISTER_CPU_OPERATOR(CloseRebatchingQueue, CloseRebatchingQueueOp);

NO_GRADIENT(CreateRebatchingQueue);
NO_GRADIENT(EnqueueRebatchingQueue);
NO_GRADIENT(DequeueRebatchingQueue);
NO_GRADIENT(CloseRebatchingQueue);

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

OPERATOR_SCHEMA(CloseRebatchingQueue)
    .NumInputs(1)
    .NumOutputs(0)
    .SetDoc(R"DOC(
      Closes the Queue.
)DOC")
    .Input(0, "queue", "object representing the queue");

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
