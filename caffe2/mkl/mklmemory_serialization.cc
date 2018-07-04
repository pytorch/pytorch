#include "caffe2/core/blob.h"
#include "caffe2/core/blob_serialization.h"
#include "caffe2/mkl/mkl_utils.h"

#ifdef CAFFE2_HAS_MKL_DNN

namespace caffe2 {
namespace mkl {
/**
 * @brief MKLMemorySerializer is the serializer for MKLMemory.
 *
 * MKLMemorySerializer takes in a blob that contains an MKLMemory, and
 * serializes it into a TensorProto protocol buffer.
 */
class MKLMemorySerializer : public BlobSerializerBase {
 public:
  MKLMemorySerializer() {}
  ~MKLMemorySerializer() {}

  void Serialize(
      const Blob& blob,
      const string& name,
      SerializationAcceptor acceptor) override {
    BlobProto blob_proto;
    blob_proto.set_name(name);
    blob_proto.set_type(kTensorBlobType);
    TensorProto* proto = blob_proto.mutable_tensor();
    auto* device_detail = proto->mutable_device_detail();
    device_detail->set_device_type(MKLDNN);
    proto->set_name(name);
    if (blob.IsType<MKLMemory<float>>()) {
      const MKLMemory<float>& src = blob.Get<MKLMemory<float>>();
      CAFFE_ENFORCE(
          src.buffer(), "Cannot serialize an empty MKLMemory object.");
      size_t total = 1;
      for (int i = 0; i < src.dims().size(); ++i) {
        proto->add_dims(src.dims()[i]);
        total *= src.dims()[i];
      }
      proto->mutable_float_data()->Reserve(total);
      while (total--) {
        proto->add_float_data(0);
      }
      src.CopyTo(proto->mutable_float_data()->mutable_data());
    } else if (blob.IsType<MKLMemory<double>>()) {
      const MKLMemory<double>& src = blob.Get<MKLMemory<double>>();
      CAFFE_ENFORCE(
          src.buffer(), "Cannot serialize an empty MKLMemory object.");
      size_t total = 1;
      for (int i = 0; i < src.dims().size(); ++i) {
        proto->add_dims(src.dims()[i]);
        total *= src.dims()[i];
      }
      proto->mutable_double_data()->Reserve(total);
      while (total--) {
        proto->add_double_data(0);
      }
      src.CopyTo(proto->mutable_double_data()->mutable_data());
    } else {
      CAFFE_THROW(
          "MKLMemory could only be either float or double. "
          "Encountered unsupported type.");
    }
    acceptor(name, blob_proto.SerializeAsString());
  }
};

/**
 * @brief MKLMemoryDeserializer is the deserializer for TensorProto that has
 * MKLDNN as its device.
 *
 * The device that the deserialized Tensor will live under is determined by the
 * device_detail field. If you want to specify the device of the deserialized
 * tensor, change the TensorProto's corresponding fields before calling
 * Deserialize.
 */
class MKLMemoryDeserializer : public BlobDeserializerBase {
 public:
  void Deserialize(const BlobProto& blob_proto, Blob* blob) override {
    const TensorProto& proto = blob_proto.tensor();
    CAFFE_ENFORCE(
        proto.data_type() == TensorProto_DataType_FLOAT ||
            proto.data_type() == TensorProto_DataType_DOUBLE,
        "MKLMemory only supports either float or double formats.");
    CAFFE_ENFORCE(
        !proto.has_segment(), "MKLMemory does not support segment right now.");
    vector<TIndex> dims;
    for (const TIndex d : proto.dims()) {
      dims.push_back(d);
    }
    // TODO: right now, every time we do a deserializer we create a new MKL
    // Memory object. Optionally, we can change that.
    switch (proto.data_type()) {
      case TensorProto_DataType_FLOAT: {
        auto dst = make_unique<MKLMemory<float>>(dims);
        dst->CopyFrom(proto.float_data().data());
        blob->Reset(dst.release());
        break;
      }
      case TensorProto_DataType_DOUBLE: {
        auto dst = make_unique<MKLMemory<double>>(dims);
        dst->CopyFrom(proto.double_data().data());
        blob->Reset(dst.release());
        break;
      }
      default:
        CAFFE_THROW("This should not happen, we guarded things above already.");
    }
  }
};

} // namespace mkl

REGISTER_BLOB_SERIALIZER(
    (TypeMeta::Id<mkl::MKLMemory<float>>()),
    mkl::MKLMemorySerializer);
REGISTER_BLOB_SERIALIZER(
    (TypeMeta::Id<mkl::MKLMemory<double>>()),
    mkl::MKLMemorySerializer);
REGISTER_BLOB_DESERIALIZER(TensorMKLDNN, mkl::MKLMemoryDeserializer);
} // namespace caffe2

#endif // CAFFE2_HAS_MKL_DNN
