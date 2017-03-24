#pragma once

#include <memory>
#include "blobs_queue.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename Context>
class CreateBlobsQueueOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  CreateBlobsQueueOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws), ws_(ws) {}

  bool RunOnDevice() override {
    const auto capacity =
        OperatorBase::template GetSingleArgument<int>("capacity", 1);
    const auto numBlobs =
        OperatorBase::template GetSingleArgument<int>("num_blobs", 1);
    const auto enforceUniqueName =
        OperatorBase::template GetSingleArgument<int>(
            "enforce_unique_name", false);
    const auto fieldNames =
        OperatorBase::template GetRepeatedArgument<std::string>("field_names");
    CAFFE_ENFORCE_EQ(def().output().size(), 1);
    const auto name = def().output().Get(0);
    auto queuePtr = Operator<Context>::Outputs()[0]
                        ->template GetMutable<std::shared_ptr<BlobsQueue>>();
    CAFFE_ENFORCE(queuePtr);
    *queuePtr = std::make_shared<BlobsQueue>(
        ws_, name, capacity, numBlobs, enforceUniqueName, fieldNames);
    return true;
  }

 private:
  Workspace* ws_{nullptr};
};

template <typename Context>
class EnqueueBlobsOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  using Operator<Context>::Operator;
  bool RunOnDevice() override {
    CAFFE_ENFORCE(InputSize() > 1);
    auto queue = Operator<Context>::Inputs()[0]
                     ->template Get<std::shared_ptr<BlobsQueue>>();
    CAFFE_ENFORCE(queue && OutputSize() == queue->getNumBlobs());
    return queue->blockingWrite(this->Outputs());
  }

 private:
};

template <typename Context>
class DequeueBlobsOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  using Operator<Context>::Operator;
  bool RunOnDevice() override {
    CAFFE_ENFORCE(InputSize() == 1);
    auto queue =
        OperatorBase::Inputs()[0]->template Get<std::shared_ptr<BlobsQueue>>();
    CAFFE_ENFORCE(queue && OutputSize() == queue->getNumBlobs());
    return queue->blockingRead(this->Outputs());
  }

 private:
};

template <typename Context>
class CloseBlobsQueueOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  using Operator<Context>::Operator;
  bool RunOnDevice() override {
    CAFFE_ENFORCE_EQ(InputSize(), 1);
    auto queue =
        OperatorBase::Inputs()[0]->template Get<std::shared_ptr<BlobsQueue>>();
    CAFFE_ENFORCE(queue);
    queue->close();
    return true;
  }

 private:
};

template <typename Context>
class SafeEnqueueBlobsOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  using Operator<Context>::Operator;
  bool RunOnDevice() override {
    auto queue = Operator<Context>::Inputs()[0]
                     ->template Get<std::shared_ptr<BlobsQueue>>();
    CAFFE_ENFORCE(queue);
    auto size = queue->getNumBlobs();
    CAFFE_ENFORCE(
        OutputSize() == size + 1,
        "Expected " + caffe2::to_string(size + 1) + ", " + " got: " +
            caffe2::to_string(size));
    bool status = queue->blockingWrite(this->Outputs());
    Output(size)->Resize();
    math::Set<bool, Context>(
        1, !status, Output(size)->template mutable_data<bool>(), &context_);
    return true;
  }
};

template <typename Context>
class SafeDequeueBlobsOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  using Operator<Context>::Operator;
  bool RunOnDevice() override {
    CAFFE_ENFORCE(InputSize() == 1);
    auto queue = Operator<Context>::Inputs()[0]
                     ->template Get<std::shared_ptr<BlobsQueue>>();
    CAFFE_ENFORCE(queue);
    auto size = queue->getNumBlobs();
    CAFFE_ENFORCE(
        OutputSize() == size + 1,
        "Expected " + caffe2::to_string(size + 1) + ", " + " got: " +
            caffe2::to_string(size));
    bool status = queue->blockingRead(this->Outputs());
    Output(size)->Resize();
    math::Set<bool, Context>(
        1, !status, Output(size)->template mutable_data<bool>(), &context_);
    return true;
  }

 private:
};
}
