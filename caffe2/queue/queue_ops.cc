#include "queue_ops.h"
#include <memory>

namespace caffe2 {

namespace {

REGISTER_CPU_OPERATOR(CreateBlobsQueue, CreateBlobsQueueOp<CPUContext>);
REGISTER_CPU_OPERATOR(EnqueueBlobs, EnqueueBlobsOp<CPUContext>);
REGISTER_CPU_OPERATOR(DequeueBlobs, DequeueBlobsOp<CPUContext>);
REGISTER_CPU_OPERATOR(CloseBlobsQueue, CloseBlobsQueueOp<CPUContext>);

OPERATOR_SCHEMA(CreateBlobsQueue).NumInputs(0).NumOutputs(1);
OPERATOR_SCHEMA(EnqueueBlobs)
    .NumInputsOutputs([](int inputs, int outputs) {
      return inputs >= 2 && outputs >= 1 && inputs == outputs + 1;
    })
    .EnforceInplace([](int input, int output) { return input == output + 1; });
OPERATOR_SCHEMA(DequeueBlobs).NumInputsOutputs([](int inputs, int outputs) {
  return inputs == 1 && outputs >= 1;
});

OPERATOR_SCHEMA(CloseBlobsQueue).NumInputs(1).NumOutputs(0);

NO_GRADIENT(CreateBlobsQueue);
NO_GRADIENT(EnqueueBlobs);
NO_GRADIENT(DequeueBlobs);
NO_GRADIENT(CloseBlobsQueue);

}

}
