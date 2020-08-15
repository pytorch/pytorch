#ifndef CAFFE2_CORE_BLOB_SERIALIZATION_H_
#define CAFFE2_CORE_BLOB_SERIALIZATION_H_

#include <limits>
#include <future>

#include <google/protobuf/repeated_field.h>

#include "caffe2/core/blob.h"
#include "caffe2/core/blob_serializer_base.h"
#include "caffe2/core/tensor.h"
#include <c10/util/typeid.h>
#include "caffe2/core/types.h"
#include "caffe2/utils/simple_queue.h"

C10_DECLARE_int(caffe2_tensor_chunk_size);
C10_DECLARE_int(caffe2_max_tensor_serializer_threads);
C10_DECLARE_bool(caffe2_serialize_fp16_as_bytes);

namespace caffe2 {

constexpr auto kTensorBlobType = "Tensor";
// String used to separate chunk id from the blob name when storing in DB
constexpr auto kChunkIdSeparator = "#%";

/**
 * Serializes the given blob, if possible. Note that this serialization uses
 * the registration mechanism and one has to implement specific serialization
 * approaches for specific classes. Acceptor should take care of writing data
 * to the actual storage.
 */
CAFFE2_API void SerializeBlob(
    const Blob& blob,
    const string& name,
    BlobSerializerBase::SerializationAcceptor acceptor,
    int chunk_size = kDefaultChunkSize);

/**
 * @brief Convenience function to serialize a blob to a string.
 *
 * This is a convenience function to serialize small Blobs that produce
 * manageable serialized strings. To serialize big blobs such as
 * large sparse tensors, use the fully-functional interface in
 * blob_serializer_base.h.
 *
 * NOTE: this function doesn't do chunking and might break with big tensors.
 */
CAFFE2_API string SerializeBlob(const Blob& blob, const string& name);

/**
 * Deserializes from a string containing either BlobProto or TensorProto. If
 * the deserialization fails, the content in the blob should no longer be
 * trusted.
 */
CAFFE2_API void DeserializeBlob(const string& content, Blob* result);
CAFFE2_API void DeserializeBlob(const BlobProto& proto, Blob* result);

/*
 * Get an empty Tensor from the TensorProto given the meta data in proto (data
 * type and size of the Tensor) without actually filling in the data.
 *
 * We need this function because we want to construct a fully initialized Tensor
 * in the beginning instead of keeping partially initialized Tensor around the
 * process. Consider the case when we have a Tensor that is split into multiple
 * protos during serialization, in deserialization, we have to fill the Tensor
 * in multiple calls to Deserialize, therefore we need to create a new Tensor
 * with the correct size and data type before the call to Deserialize, because
 * otherwise we will have to check whether the function call is the first call
 * to initialize the underlying Tensor, which makes the function stateful and
 * complicated.
 *
 * The legacy code get away with this problem by passing in a partially
 * initialized Tensor and use Resize and mutable_data to set the correct size,
 * data type and allocate memory for the Tensor, so the state is encoded in
 * these function calls. e.g. mutable_data will allocate memory on the first
 * call and it will return a pointer to the allocated memory on later calls.
 */
CAFFE2_API Tensor EmptyTensorFromProto(const TensorProto& proto);

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
      const void* pointer,
      TypeMeta typeMeta,
      const string& name,
      SerializationAcceptor acceptor) override;
  void SerializeWithChunkSize(
      const void* pointer,
      TypeMeta typeMeta,
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

  /* There are cases when a Tensor is split into multiple protos and
   * we have to call Deserialize multiple times to get the complete deserialized
   * Tensor, each call will fill part of the Tensor given the segment begin and
   * end information in proto, therefore we have to pass in the Tensor pointer
   * rather than create a new Tensor every time.
   *
   * Precondition: Tensor must be initialized
   */
  void DeserializeToTensor(const TensorProto& proto, Tensor* tensor);

  /* Deserialize the proto and return a new Tensor
   * This is a utility function that combines EmptyTensorFromProto and
   * Deserialize(const TensorProto&, Tensor*);
   */
  Tensor Deserialize(const TensorProto& proto);
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
  for (size_t i = 0; i < size; ++i) {
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
  for (size_t i = 0; i < size; ++i) {
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
  for (size_t i = 0; i < size; ++i) {
    buffer[i] = static_cast<DstType>(src[i]);
  }
  context->template CopyFromCPU<DstType>(size, buffer.get(), dst);
}

}  // namespace detail

////////////////////////////////////////////////////////////////////////////////
// Serialization Helpers
////////////////////////////////////////////////////////////////////////////////

// Converts MessageLite to string while also checking that SerializeAsString
// succeeds. Pass description of class/function of the call if you'd
// like it appended to the error message.
CAFFE2_API std::string SerializeAsString_EnforceCheck(
    const google::protobuf::MessageLite&,
    const char* error_location = nullptr);

// Convert BlobProto to string with success checks.
inline std::string SerializeBlobProtoAsString_EnforceCheck(
    const BlobProto& blob) {
  return SerializeAsString_EnforceCheck(blob, blob.name().c_str());
}

}  // namespace caffe2

#endif  // CAFFE2_CORE_BLOB_SERIALIZATION_H_
