#ifndef CAFFE2_CORE_BLOB_SERIALIZATION_H_
#define CAFFE2_CORE_BLOB_SERIALIZATION_H_

#include <limits>
#include <future>

#include <google/protobuf/repeated_field.h>

#include "caffe2/core/blob.h"
#include "caffe2/core/blob_serializer_base.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/typeid.h"
#include "caffe2/core/types.h"
#include "caffe2/utils/simple_queue.h"

CAFFE2_DECLARE_int(caffe2_tensor_chunk_size);
CAFFE2_DECLARE_int(caffe2_max_tensor_serializer_threads);
CAFFE2_DECLARE_bool(caffe2_serialize_fp16_as_bytes);

namespace caffe2 {

constexpr auto kTensorBlobType = "Tensor";
// String used to separate chunk id from the blob name when storing in DB
constexpr auto kChunkIdSeparator = "#%";

/**
 * @brief TensorSerializer is the serializer for Tensors.
 *
 * TensorSerializer takes in a blob that contains a Tensor, and serializes it
 * into a TensorProto protocol buffer.
 */
class CAFFE2_API TensorSerializer : public BlobSerializerBase {
 public:
  TensorSerializer() {}
  ~TensorSerializer() override {}
  /**
   * Serializes a Blob. Note that this blob has to contain Tensor,
   * otherwise this function produces a fatal error.
   */
  void Serialize(
      const Blob& blob,
      const string& name,
      SerializationAcceptor acceptor) override;
  void SerializeWithChunkSize(
      const Blob& blob,
      const string& name,
      SerializationAcceptor acceptor,
      int chunk_size) override;

  void Serialize(
      const Tensor& tensor,
      const string& name,
      TensorProto* proto,
      size_t chunkBegin,
      int32_t chunkSize);

 private:
  // A utility function to store the device context detauls.
  void StoreDeviceDetail(const Tensor& input, TensorProto* proto);
  unique_ptr<BaseContext> context_;
};


/**
 * @brief TensorDeserializer is the deserializer for Tensors.
 *
 * The device that the deserialized Tensor will live under is determined by the
 * device_detail field. If you want to specify the device of the deserialized
 * tensor, change the TensorProto's corresponding fields before calling
 * Deserialize.
 */
class CAFFE2_API TensorDeserializer : public BlobDeserializerBase {
 public:
  void Deserialize(const BlobProto& proto, Blob* blob) override;
  void Deserialize(const TensorProto& proto, Tensor* tensor);
};

////////////////////////////////////////////////////////////////////////////////
// Implementations
////////////////////////////////////////////////////////////////////////////////

namespace detail {
template <typename SrcType, typename DstType>
inline void CopyToProtoAsIs(
    const size_t size,
    const SrcType* src,
    google::protobuf::RepeatedField<DstType>* field,
    BaseContext* context) {
  static_assert(
      sizeof(SrcType) == sizeof(DstType),
      "The source type and dest type cannot be copied as-is. Did "
      "you mean CopyToProtoWithCast?");
  field->Reserve(size);
  for (int i = 0; i < size; ++i) {
    field->Add(0);
  }
  context->template CopyToCPU<SrcType>(
      size, src, reinterpret_cast<SrcType*>(field->mutable_data()));
  // Make sure that we finish the copy into the protobuf.
  context->FinishDeviceComputation();
}

template <typename SrcType, typename DstType>
inline void CopyToProtoWithCast(
    const size_t size,
    const SrcType* src,
    google::protobuf::RepeatedField<DstType>* field,
    BaseContext* context) {
  // TODO: we are having one unnecessary copy here if the context is already
  // CPUContext. Remove it if it is performance critical.
  unique_ptr<SrcType[]> buffer(new SrcType[size]);
  context->template CopyToCPU<SrcType>(size, src, buffer.get());
  context->FinishDeviceComputation();
  field->Reserve(size);
  for (int i = 0; i < size; ++i) {
    field->Add(static_cast<DstType>(buffer[i]));
  }
}

template <typename SrcType, typename DstType>
inline void CopyFromProtoAsIs(
    const size_t size,
    const google::protobuf::RepeatedField<SrcType>& field,
    DstType* dst,
    BaseContext* context) {
  static_assert(
      sizeof(SrcType) == sizeof(DstType),
      "The source type and dest type cannot be copied as-is. Did "
      "you mean CopyFromProtoWithCast?");
  CAFFE_ENFORCE_EQ(size, field.size(), "Incorrect proto field size.");
  context->template CopyFromCPU<DstType>(
      size, reinterpret_cast<const DstType*>(field.data()), dst);
}

template <typename SrcType, typename DstType>
inline void CopyFromProtoWithCast(
    const size_t size,
    const google::protobuf::RepeatedField<SrcType>& field,
    DstType* dst,
    BaseContext* context) {
  CAFFE_ENFORCE_EQ(size, field.size(), "Incorrect proto field size.");
  // TODO: we are having one unnecessary copy here if the context is already
  // CPUContext. Remove it if it is performance critical.
  unique_ptr<DstType[]> buffer(new DstType[size]);
  const SrcType* src = field.data();
  for (int i = 0; i < size; ++i) {
    buffer[i] = static_cast<DstType>(src[i]);
  }
  context->template CopyFromCPU<DstType>(size, buffer.get(), dst);
}

}  // namespace detail
}  // namespace caffe2

#endif  // CAFFE2_CORE_BLOB_SERIALIZATION_H_
