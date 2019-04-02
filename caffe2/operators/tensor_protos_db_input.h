#ifndef CAFFE2_OPERATORS_TENSOR_PROTOS_DB_INPUT_H_
#define CAFFE2_OPERATORS_TENSOR_PROTOS_DB_INPUT_H_

#include <iostream>

#include "caffe2/core/db.h"
#include "caffe2/operators/prefetch_op.h"

namespace caffe2 {

// tensor protos db input is the simplest input that basically reads
// things from a db where each key-value pair stores a TensorProtos object.
// These tensorprotos should have the same size, and they will be grouped into
// batches of the given size. The output will simply be tensors of float data.
template <class DeviceContext>
class TensorProtosDBInput final
    : public PrefetchOperator<DeviceContext> {
 public:
  using OperatorBase::OutputSize;
  using PrefetchOperator<DeviceContext>::prefetch_thread_;
  explicit TensorProtosDBInput(const OperatorDef& operator_def, Workspace* ws);
  ~TensorProtosDBInput() {
    if (prefetch_thread_.get() != nullptr) {
      prefetch_thread_->join();
    }
  }

  bool Prefetch() override;
  bool CopyPrefetched() override;

 private:
  unique_ptr<db::DB> db_;
  unique_ptr<db::Cursor> cursor_;
  // Prefetch will always just happen on the CPU side.
  vector<unique_ptr<Blob> > prefetched_blobs_;
  vector<TensorProto::DataType> data_types_;
  int batch_size_;
  string db_name_;
  string db_type_;
  DISABLE_COPY_AND_ASSIGN(TensorProtosDBInput);
};

template <class DeviceContext>
TensorProtosDBInput<DeviceContext>::TensorProtosDBInput(
      const OperatorDef& operator_def, Workspace* ws)
      : PrefetchOperator<DeviceContext>(operator_def, ws),
        batch_size_(
            OperatorBase::template GetSingleArgument<int>("batch_size", 0)),
        db_name_(
            OperatorBase::template GetSingleArgument<string>("db", "")),
        db_type_(OperatorBase::template GetSingleArgument<string>(
            "db_type", "leveldb")) {
  CHECK_GT(batch_size_, 0) << "Batch size should be nonnegative.";
  CHECK_GT(db_name_.size(), 0) << "Must provide a leveldb name.";

  db_.reset(db::CreateDB(db_type_, db_name_, db::READ));
  cursor_.reset(db_->NewCursor());
  cursor_->SeekToFirst();

  // Now, we want to read a data point to initialize the contents.
  TensorProtos protos;
  protos.ParseFromString(cursor_->value());
  CHECK_EQ(protos.protos_size(), OutputSize());
  prefetched_blobs_.resize(protos.protos_size());
  data_types_.resize(protos.protos_size());
  VLOG(1) << "Figuring data types.";
  for (int i = 0; i < protos.protos_size(); ++i) {
    vector<int> dims;
    for (const int dim : protos.protos(i).dims()) {
      dims.push_back(dim);
    }
    dims[0] = batch_size_;
    prefetched_blobs_[i].reset(new Blob());
    Blob* blob = prefetched_blobs_[i].get();
    data_types_[i] = protos.protos(i).data_type();
    switch (data_types_[i]) {
    case TensorProto::FLOAT:
      VLOG(1) << "Output " << i << ": float";
      blob->GetMutable<Tensor<float, CPUContext> >()->Reshape(dims);
      break;
    case TensorProto::INT32:
      VLOG(1) << "Output " << i << ": int";
      blob->GetMutable<Tensor<int, CPUContext> >()->Reshape(dims);
      break;
    case TensorProto::BYTE:
      VLOG(1) << "Output " << i << ": byte -> float";
      // TODO(Yangqing): What type should I use here? Float?
      blob->GetMutable<Tensor<float, CPUContext> >()->Reshape(dims);
      break;
    case TensorProto::STRING:
      LOG(FATAL) << "Not expecting string.";
    }
  }
  cursor_->SeekToFirst();
}

template <class DeviceContext>
bool TensorProtosDBInput<DeviceContext>::Prefetch() {
  for (int item_id = 0; item_id < batch_size_; ++item_id) {
    // LOG(INFO) << "Prefetching item " << item_id;
    // process data
    TensorProtos protos;
    protos.ParseFromString(cursor_->value());
    // TODO(Yangqing): do we want to do anything to sanity check the data?
    for (int i = 0; i < protos.protos_size(); ++i) {
      const TensorProto& proto = protos.protos(i);
      Blob* blob = prefetched_blobs_[i].get();
      switch (proto.data_type()) {
      case TensorProto::FLOAT:
      {
        DCHECK((blob->IsType<Tensor<float, CPUContext> >()));
        auto* tensor = blob->GetMutable<Tensor<float, CPUContext> >();
        int single_size = proto.float_data_size();
        CHECK_EQ(single_size * batch_size_, tensor->size());
        memcpy(tensor->mutable_data() + single_size * item_id,
               proto.float_data().data(), single_size * sizeof(float));
        break;
      }
      case TensorProto::INT32:
      {
        DCHECK((blob->IsType<Tensor<int, CPUContext> >()));
        auto* tensor = blob->GetMutable<Tensor<int, CPUContext> >();
        int single_size = proto.int32_data_size();
        CHECK_EQ(single_size * batch_size_, tensor->size());
        int* dst_pointer = tensor->mutable_data() + single_size * item_id;
        for (int i = 0; i < single_size; ++i) {
          dst_pointer[i] = proto.int32_data(i);
        }
        break;
      }
      case TensorProto::BYTE:
      {
        DCHECK((blob->IsType<Tensor<float, CPUContext> >()));
        auto* tensor = blob->GetMutable<Tensor<float, CPUContext> >();
        const string& src_data = proto.byte_data();
        int single_size = src_data.size();
        CHECK_EQ(single_size * batch_size_, tensor->size());
        float* dst_pointer = tensor->mutable_data() + single_size * item_id;
        for (int i = 0; i < single_size; ++i) {
          dst_pointer[i] =
              static_cast<float>(static_cast<uint8_t>(src_data[i])) / 256.f;
        }
        break;
      }
      default:
        LOG(ERROR) << "Unknown input data type: " << proto.data_type();
        return false;
      }
    }
    cursor_->Next();
    if (!cursor_->Valid()) {
      cursor_->SeekToFirst();
    }
  }
  return true;
}

template <class DeviceContext>
bool TensorProtosDBInput<DeviceContext>::CopyPrefetched() {
  for (int i = 0; i < OutputSize(); ++i) {
    switch (data_types_[i]) {
    case TensorProto::FLOAT:
    case TensorProto::BYTE:
    {
      auto* output = OperatorBase::Output<Tensor<float, DeviceContext> >(i);
      auto& input =
          prefetched_blobs_[i]->template Get<Tensor<float, CPUContext> >();
      output->ReshapeLike(input);
      this->device_context_.template Copy<float, DeviceContext, CPUContext>(
          output->mutable_data(), input.data(), input.size());
      break;
    }
    case TensorProto::INT32:
    {
      auto* output = OperatorBase::Output<Tensor<int, DeviceContext> >(i);
      auto& input =
          prefetched_blobs_[i]->template Get<Tensor<int, CPUContext> >();
      output->ReshapeLike(input);
      this->device_context_.template Copy<int, DeviceContext, CPUContext>(
          output->mutable_data(), input.data(), input.size());
      break;
    }
    case TensorProto::STRING:
      LOG(FATAL) << "Not expecting string.";
    }
  }
  return true;
}


}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_TENSOR_PROTOS_DB_INPUT_H_
