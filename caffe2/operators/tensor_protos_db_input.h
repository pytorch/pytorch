#ifndef CAFFE2_OPERATORS_TENSOR_PROTOS_DB_INPUT_H_
#define CAFFE2_OPERATORS_TENSOR_PROTOS_DB_INPUT_H_

#include <iostream>
#include <mutex>

#include "caffe2/core/db.h"
#include "caffe2/operators/prefetch_op.h"

namespace caffe2 {

template <class Context>
class TensorProtosDBInput final : public PrefetchOperator<Context> {
 public:
  using OperatorBase::OutputSize;
  using PrefetchOperator<Context>::prefetch_thread_;
  explicit TensorProtosDBInput(const OperatorDef& operator_def, Workspace* ws);
  ~TensorProtosDBInput() {
    PrefetchOperator<Context>::Finalize();
  }

  bool Prefetch() override;
  bool CopyPrefetched() override;

 private:
  // Prefetch will always just happen on the CPU side.
  vector<Blob> prefetched_blobs_;
  int batch_size_;
  bool shape_inferred_ = false;
  string key_;
  string value_;
};

template <class Context>
TensorProtosDBInput<Context>::TensorProtosDBInput(
    const OperatorDef& operator_def,
    Workspace* ws)
    : PrefetchOperator<Context>(operator_def, ws),
      prefetched_blobs_(operator_def.output_size()),
      batch_size_(
          this->template GetSingleArgument<int>("batch_size", 0)) {}

template <class Context>
bool TensorProtosDBInput<Context>::Prefetch() {
  const db::DBReader& reader = this->template Input<db::DBReader>(0);
  TensorDeserializer deserializer;
  if (batch_size_ == 0) {
    // We do not need to construct a batch. As a result, we will simply
    // deserialize everything into the target prefetched blob.
    reader.Read(&key_, &value_);
    TensorProtos protos;
    CAFFE_ENFORCE(protos.ParseFromString(value_));
    CAFFE_ENFORCE(protos.protos_size() == OutputSize());
    for (int i = 0; i < protos.protos_size(); ++i) {
      if (protos.protos(i).has_device_detail()) {
        protos.mutable_protos(i)->clear_device_detail();
      }
      BlobSetTensor(
          &prefetched_blobs_[i], deserializer.Deserialize(protos.protos(i)));
      // deserializer.Deserialize(
      //     protos.protos(i), BlobGetMutableTensor(&prefetched_blobs_[i],
      //     CPU));
    }
  } else {
    for (int item_id = 0; item_id < batch_size_; ++item_id) {
      reader.Read(&key_, &value_);
      TensorProtos protos;
      CAFFE_ENFORCE(protos.ParseFromString(value_));
      CAFFE_ENFORCE(protos.protos_size() == OutputSize());
      // Note: shape_inferred_ is ignored, we'll always get dimensions from
      // proto
      for (int i = 0; i < protos.protos_size(); ++i) {
        vector<int64_t> dims(
            protos.protos(i).dims().begin(), protos.protos(i).dims().end());
        dims.insert(dims.begin(), batch_size_);
        if (protos.protos(i).has_device_detail()) {
          protos.mutable_protos(i)->clear_device_detail();
        }
        Tensor src = deserializer.Deserialize(protos.protos(i));
        Tensor* dst = BlobGetMutableTensor(
            &prefetched_blobs_[i], dims, at::dtype(src.dtype()).device(CPU));
        DCHECK_EQ(src.numel() * batch_size_, dst->numel());
        this->context_.CopyItemsSameDevice(
            src.dtype(),
            src.numel(),
            src.raw_data(),
            static_cast<char*>(dst->raw_mutable_data(src.dtype())) +
                src.nbytes() * item_id);
      }
    }
  }
  return true;
}

template <class Context>
bool TensorProtosDBInput<Context>::CopyPrefetched() {
  for (int i = 0; i < OutputSize(); ++i) {
    OperatorBase::template Output<Tensor>(i, Context::GetDeviceType())
        ->CopyFrom(
            prefetched_blobs_[i].template Get<TensorCPU>(), /* async */ true);
  }
  return true;
}

} // namespace caffe2

#endif // CAFFE2_OPERATORS_TENSOR_PROTOS_DB_INPUT_H_
