#pragma once

#include <memory>
#include "blobs_queue.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename Context>
class CreateBlobsQueueOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  CreateBlobsQueueOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        ws_(ws),
        name(operator_def.output().Get(0)) {}

  bool RunOnDevice() override {
    const auto capacity = GetSingleArgument("capacity", 1);
    const auto numBlobs = GetSingleArgument("num_blobs", 1);
    const auto enforceUniqueName =
        GetSingleArgument("enforce_unique_name", false);
    const auto fieldNames =
        OperatorBase::template GetRepeatedArgument<std::string>("field_names");
    CAFFE_ENFORCE_EQ(this->OutputSize(), 1);
    auto queuePtr = Operator<Context>::Outputs()[0]
                        ->template GetMutable<std::shared_ptr<BlobsQueue>>();
    CAFFE_ENFORCE(queuePtr);
    *queuePtr = std::make_shared<BlobsQueue>(
        ws_, name, capacity, numBlobs, enforceUniqueName, fieldNames);
    return true;
  }

 private:
  Workspace* ws_{nullptr};
  const std::string name;
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
    // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
    CAFFE_ENFORCE(queue && OutputSize() == queue->getNumBlobs());
    return queue->blockingWrite(this->Outputs());
  }

 private:
};

template <typename Context>
class DequeueBlobsOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  DequeueBlobsOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {
    timeout_secs_ = OperatorBase::GetSingleArgument<float>("timeout_secs", 0);
  }

  bool RunOnDevice() override {
    CAFFE_ENFORCE(InputSize() == 1);
    auto queue =
        OperatorBase::Inputs()[0]->template Get<std::shared_ptr<BlobsQueue>>();
    // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
    CAFFE_ENFORCE(queue && OutputSize() == queue->getNumBlobs());
    return queue->blockingRead(this->Outputs(), timeout_secs_);
  }

 private:
  float timeout_secs_;
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
        // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
        OutputSize() == size + 1,
        "Expected " + c10::to_string(size + 1) + ", " +
            " got: " + c10::to_string(size));
    bool status = queue->blockingWrite(this->Outputs());
    Output(size)->Resize();
    math::Set<bool, Context>(
        1, !status, Output(size)->template mutable_data<bool>(), &context_);
    return true;
  }

  void Cancel() override {
    auto queue = Operator<Context>::Inputs()[0]
                     ->template Get<std::shared_ptr<BlobsQueue>>();
    queue->close();
  }
};

template <typename Context>
class SafeDequeueBlobsOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  using Operator<Context>::Operator;

  SafeDequeueBlobsOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        numRecords_(OperatorBase::GetSingleArgument<int>("num_records", 1)) {
    CAFFE_ENFORCE_GT(numRecords_, 0);
  }

  bool dequeueMany(std::shared_ptr<BlobsQueue>& queue) {
    auto size = queue->getNumBlobs();

    if (blobs_.size() != size) {
      blobs_.resize(size);
      blobPtrs_.resize(size);
      // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
      for (int col = 0; col < size; ++col) {
        blobPtrs_.at(col) = &blobs_.at(col);
      }
    }

    const int kTensorGrowthPct = 40;
    for (int i = 0; i < numRecords_; ++i) {
      if (!queue->blockingRead(blobPtrs_)) {
        // if we read at least one record, status is still true
        return i > 0;
      }
      for (int col = 0; col < size; ++col) {
        auto* out = this->Output(col);
        const auto& in = blobPtrs_.at(col)->template Get<Tensor>();
        if (i == 0) {
          out->CopyFrom(in);
        } else {
          auto oldSize = out->numel();

          CAFFE_ENFORCE(
              in.dim() > 0,
              "Empty tensor to dequeue at column ",
              col,
              " within ",
              size,
              " total columns");

          out->Extend(in.sizes()[0], kTensorGrowthPct);
          auto* dst =
              (char*)out->raw_mutable_data() + oldSize * in.dtype().itemsize();
          context_.template CopyItems<Context, Context>(
              in.meta(), in.numel(), in.raw_data(), dst);
        }
      }
    }
    return true;
  }

  bool dequeueOne(std::shared_ptr<BlobsQueue>& queue) {
    return queue->blockingRead(this->Outputs());
  }

  bool RunOnDevice() override {
    CAFFE_ENFORCE(InputSize() == 1);
    auto queue = Operator<Context>::Inputs()[0]
                     ->template Get<std::shared_ptr<BlobsQueue>>();
    CAFFE_ENFORCE(queue);

    auto size = queue->getNumBlobs();
    CAFFE_ENFORCE_EQ(OutputSize(), size + 1);

    bool status = numRecords_ > 1 ? dequeueMany(queue) : dequeueOne(queue);

    Output(size)->Resize();
    math::Set<bool, Context>(
        1, !status, Output(size)->template mutable_data<bool>(), &context_);
    return true;
  }

  void Cancel() override {
    auto queue = Operator<Context>::Inputs()[0]
                     ->template Get<std::shared_ptr<BlobsQueue>>();
    queue->close();
  }

 private:
  int numRecords_;
  std::vector<Blob> blobs_;
  std::vector<Blob*> blobPtrs_;
};

template <typename Context>
class WeightedSampleDequeueBlobsOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  WeightedSampleDequeueBlobsOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        table_idx_blob_(
            OperatorBase::GetSingleArgument<int>("table_idx_blob", -1)) {
    CAFFE_ENFORCE_LT(table_idx_blob_, OutputSize() - 1);
    vector<float> weights = OperatorBase::GetRepeatedArgument<float>("weights");
    if (weights.empty()) {
      weights.resize(InputSize(), 1.0f);
    }
    CAFFE_ENFORCE_EQ(InputSize(), weights.size());

    float sum = accumulate(weights.begin(), weights.end(), 0.0f);
    CAFFE_ENFORCE(sum > 0.0f, "Sum of weights must be positive");
    cumProbs_.resize(weights.size());
    // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
    for (int i = 0; i < weights.size(); i++) {
      cumProbs_[i] = weights[i] / sum;
      CAFFE_ENFORCE_GE(
          cumProbs_[i], 0.0f, "Each probability must be non-negative");
    }
    std::partial_sum(cumProbs_.begin(), cumProbs_.end(), cumProbs_.begin());
    // Put last value to be 1.0001 to avoid numerical issues.
    cumProbs_.back() = 1.0001f;

    LOG(INFO) << "Dequeue weights: " << weights;
    LOG(INFO) << "cumProbs: " << cumProbs_;
  }

  bool RunOnDevice() override {
    float r;
    math::RandUniform<float, Context>(1, 0.0f, 1.0f, &r, &context_);
    auto lb = lower_bound(cumProbs_.begin(), cumProbs_.end(), r);
    CAFFE_ENFORCE(lb != cumProbs_.end(), "Cannot find ", r, " in cumProbs_.");
    const int32_t idx = lb - cumProbs_.begin();
    auto queue = Operator<Context>::Inputs()[idx]
                     ->template Get<std::shared_ptr<BlobsQueue>>();

    CAFFE_ENFORCE(queue);
    auto size = queue->getNumBlobs();
    CAFFE_ENFORCE_EQ(OutputSize(), size + 1);
    bool status = queue->blockingRead(this->Outputs());
    if (table_idx_blob_ >= 0) {
      auto* table_idx_blob_out =
          Output(table_idx_blob_, {1}, at::dtype<int32_t>());
      int32_t* data = table_idx_blob_out->template mutable_data<int32_t>();
      data[0] = idx;
    }

    Output(size)->Resize();
    math::Set<bool, Context>(
        1, !status, Output(size)->template mutable_data<bool>(), &context_);
    return true;
  }

 private:
  vector<float> cumProbs_;
  int table_idx_blob_;
};
} // namespace caffe2
