#ifndef CAFFE2_OPERATORS_TENSOR_PROTOS_DB_INPUT_H_
#define CAFFE2_OPERATORS_TENSOR_PROTOS_DB_INPUT_H_

#include <iostream>
#include <mutex>

#include "caffe2/core/db.h"
#include "caffe2/operators/prefetch_op.h"

namespace caffe2 {

// tensor protos db input is the simplest input that basically reads
// things from a db where each key-value pair stores a TensorProtos object.
// These tensorprotos should have the same size, and they will be grouped into
// batches of the given size. The output will simply be tensors of float data.
template <class Context>
class TensorProtosDBInput final
    : public PrefetchOperator<Context> {
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
  void InferDataTypes();
  // Prefetch will always just happen on the CPU side.
  vector<unique_ptr<Blob> > prefetched_blobs_;
  vector<TensorProto::DataType> data_types_;
  bool infer_data_type_called_;
  int batch_size_;
  string key_;
  string value_;
  DISABLE_COPY_AND_ASSIGN(TensorProtosDBInput);
};

template <class Context>
TensorProtosDBInput<Context>::TensorProtosDBInput(
      const OperatorDef& operator_def, Workspace* ws)
      : PrefetchOperator<Context>(operator_def, ws),
        infer_data_type_called_(false),
        batch_size_(
            OperatorBase::template GetSingleArgument<int>("batch_size", 0)) {
  CAFFE_CHECK_GT(batch_size_, 0) << "Batch size should be nonnegative.";
}

template <class Context>
void TensorProtosDBInput<Context>::InferDataTypes() {
  TensorProtos protos;
  protos.ParseFromString(value_);
  CAFFE_CHECK_EQ(protos.protos_size(), OutputSize());
  prefetched_blobs_.resize(protos.protos_size());
  data_types_.resize(protos.protos_size());
  CAFFE_VLOG(1) << "Figuring data types.";
  for (int i = 0; i < protos.protos_size(); ++i) {
    vector<TIndex> dims;
    for (const int dim : protos.protos(i).dims()) {
      dims.push_back(dim);
    }
    dims[0] = batch_size_;
    prefetched_blobs_[i].reset(new Blob());
    Blob* blob = prefetched_blobs_[i].get();
    data_types_[i] = protos.protos(i).data_type();
    blob->GetMutable<TensorCPU>()->Reshape(dims);
    switch (data_types_[i]) {
    case TensorProto::FLOAT:
      CAFFE_VLOG(1) << "Output " << i << ": float";
      break;
    case TensorProto::INT32:
      CAFFE_VLOG(1) << "Output " << i << ": int";
      break;
    case TensorProto::BYTE:
      CAFFE_VLOG(1) << "Output " << i << ": byte -> float";
      break;
    case TensorProto::STRING:
      CAFFE_LOG_FATAL << "Not expecting string.";
    }
  }
}

template <class Context>
bool TensorProtosDBInput<Context>::Prefetch() {
  const db::DBReader& reader = OperatorBase::Input<db::DBReader>(0);
  for (int item_id = 0; item_id < batch_size_; ++item_id) {
    // CAFFE_LOG_INFO << "Prefetching item " << item_id;
    // process data
    reader.Read(&key_, &value_);
    if (!infer_data_type_called_) {
      InferDataTypes();
      infer_data_type_called_ = true;
    }
    TensorProtos protos;
    protos.ParseFromString(value_);
    // TODO(Yangqing): do we want to do anything to sanity check the data?
    for (int i = 0; i < protos.protos_size(); ++i) {
      const TensorProto& proto = protos.protos(i);
      Blob* blob = prefetched_blobs_[i].get();
      CAFFE_DCHECK((blob->IsType<TensorCPU>()));
      auto* tensor = blob->GetMutable<TensorCPU>();
      switch (proto.data_type()) {
      case TensorProto::FLOAT:
      {
        int single_size = proto.float_data_size();
        CAFFE_CHECK_EQ(single_size * batch_size_, tensor->size());
        memcpy(tensor->mutable_data<float>() + single_size * item_id,
               proto.float_data().data(), single_size * sizeof(float));
        break;
      }
      case TensorProto::INT32:
      {
        int single_size = proto.int32_data_size();
        CAFFE_CHECK_EQ(single_size * batch_size_, tensor->size());
        int* dst_pointer = tensor->mutable_data<int>() + single_size * item_id;
        for (int i = 0; i < single_size; ++i) {
          dst_pointer[i] = proto.int32_data(i);
        }
        break;
      }
      case TensorProto::BYTE:
      {
        const string& src_data = proto.byte_data();
        int single_size = src_data.size();
        CAFFE_CHECK_EQ(single_size * batch_size_, tensor->size());
        float* dst_pointer = tensor->mutable_data<float>() + single_size * item_id;
        for (int i = 0; i < single_size; ++i) {
          dst_pointer[i] =
              static_cast<float>(static_cast<uint8_t>(src_data[i])) / 256.f;
        }
        break;
      }
      default:
        CAFFE_LOG_ERROR << "Unknown input data type: " << proto.data_type();
        return false;
      }
    }
  }
  return true;
}

template <class Context>
bool TensorProtosDBInput<Context>::CopyPrefetched() {
  for (int i = 0; i < OutputSize(); ++i) {
    auto* output = OperatorBase::Output<Tensor<Context> >(i);
    auto& input =
        prefetched_blobs_[i]->template Get<TensorCPU>();
    switch (data_types_[i]) {
    case TensorProto::FLOAT:
    case TensorProto::BYTE:
    {
      output->ReshapeLike(input);
      this->context_.template Copy<float, CPUContext, Context>(
          input.size(), input.template data<float>(),
          output->template mutable_data<float>());
      break;
    }
    case TensorProto::INT32:
    {
      output->ReshapeLike(input);
      this->context_.template Copy<int, CPUContext, Context>(
          input.size(), input.template data<int>(),
          output->template mutable_data<int>());
      break;
    }
    case TensorProto::STRING:
      CAFFE_LOG_FATAL << "Not expecting string.";
    }
  }
  return true;
}


}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_TENSOR_PROTOS_DB_INPUT_H_
