#ifndef CAFFE2_CORE_BLOB_SERIALIZATION_H_
#define CAFFE2_CORE_BLOB_SERIALIZATION_H_

#include "caffe2/core/blob.h"

namespace caffe2 {

template <class DeviceContext>
class TensorSerializerFloat : public BlobSerializerBase {
 public:
  TensorSerializerFloat() : device_context_() {}
  ~TensorSerializerFloat() {}
  string Serialize(const Blob& blob, const string& name) {
    CHECK((blob.IsType<Tensor<float, DeviceContext> >()));
    auto& input = blob.Get<Tensor<float, DeviceContext> >();
    TensorProto proto;
    proto.set_data_type(TensorProto::FLOAT);
    proto.set_name(name);
    for (int dim : input.dims()) {
      proto.add_dims(dim);
    }
    proto.mutable_float_data()->Reserve(input.size());
    for (int i = 0; i < input.size(); ++i) {
      proto.add_float_data(0);
    }
    this->device_context_.template Copy<float, DeviceContext, CPUContext>(
        input.size(), input.data(), proto.mutable_float_data()->mutable_data());
    return proto.SerializeAsString();
  }

 private:
  DeviceContext device_context_;
};

template <class DeviceContext>
class TensorSerializerInt32 : public BlobSerializerBase {
 public:
  TensorSerializerInt32() : device_context_() {}
  ~TensorSerializerInt32() {}
  string Serialize(const Blob& blob, const string& name) {
    static_assert(sizeof(int) == 4,
        "int in this compiler does not equal to 4 bytes.");
    CHECK((blob.IsType<Tensor<int, DeviceContext> >()));
    auto& input = blob.Get<Tensor<int, DeviceContext> >();
    TensorProto proto;
    proto.set_data_type(TensorProto::INT32);
    proto.set_name(name);
    for (int dim : input.dims()) {
      proto.add_dims(dim);
    }
    proto.mutable_int32_data()->Reserve(input.size());
    for (int i = 0; i < input.size(); ++i) {
      proto.add_int32_data(0);
    }
    this->device_context_.template Copy<int, DeviceContext, CPUContext>(
        input.size(), input.data(), proto.mutable_int32_data()->mutable_data());
    return proto.SerializeAsString();
  }

 private:
  DeviceContext device_context_;
};

template <typename dtype, class DeviceContext>
class TensorSerializerBytes : public BlobSerializerBase {
 public:
  TensorSerializerBytes() : device_context_(DeviceContext()) {}
  ~TensorSerializerBytes() {}
  string Serialize(const Blob& blob, const string& name) {
    static_assert(sizeof(dtype) == sizeof(char),
        "dtype in TensorSerializerBytes must be of the same size as char.");
    CHECK((blob.IsType<Tensor<dtype, DeviceContext> >()));
    auto& input = blob.Get<Tensor<dtype, DeviceContext> >();
    TensorProto proto;
    proto.set_data_type(TensorProto::BYTE);
    proto.set_name(name);
    for (int dim : input.dims()) {
      proto.add_dims(dim);
    }
    std::unique_ptr<char[]> buffer(new char[input.size()]);
    this->device_context_.template Copy<char, DeviceContext, CPUContext>(
        input.size(), input.data(), buffer.get());
    proto.set_byte_data(buffer, input.size());
    return proto.SerializeAsString();
  }

 private:
  DeviceContext device_context_;
};

}  // namespace caffe2

#endif  // CAFFE2_CORE_BLOB_SERIALIZATION_H_
