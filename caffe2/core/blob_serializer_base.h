#pragma once

#include <string>
#include <functional>

#include <c10/util/Registry.h>
#include <c10/util/string_view.h>
#include "caffe2/core/common.h"
#include "caffe2/proto/caffe2_pb.h"

namespace caffe2 {

class Blob;

// Constants for use in the BlobSerializationOptions chunk_size field.
// These should ideally be defined in caffe2.proto so they can be exposed across
// languages, but protobuf does not appear to allow defining constants.
constexpr int kDefaultChunkSize = 0;
constexpr int kNoChunking = -1;

/**
 * @brief BlobSerializerBase is an abstract class that serializes a blob to a
 * string.
 *
 * This class exists purely for the purpose of registering type-specific
 * serialization code. If you need to serialize a specific type, you should
 * write your own Serializer class, and then register it using
 * REGISTER_BLOB_SERIALIZER. For a detailed example, see TensorSerializer for
 * details.
 */
class BlobSerializerBase {
 public:
  virtual ~BlobSerializerBase() {}
  using SerializationAcceptor =
     std::function<void(const std::string& blobName, std::string&& data)>;
  /**
   * @brief The virtual function that returns a serialized string for the input
   * blob.
   * @param blob
   *     the input blob to be serialized.
   * @param name
   *     the blob name to be used in the serialization implementation. It is up
   *     to the implementation whether this name field is going to be used or
   *     not.
   * @param acceptor
   *     a lambda which accepts key value pairs to save them to storage.
   *     serailizer can use it to save blob in several chunks
   *     acceptor should be thread-safe
   */
  virtual void Serialize(
      const void* pointer,
      TypeMeta typeMeta,
      const std::string& name,
      SerializationAcceptor acceptor) = 0;

  virtual void SerializeWithOptions(
      const void* pointer,
      TypeMeta typeMeta,
      const std::string& name,
      SerializationAcceptor acceptor,
      const BlobSerializationOptions& /*options*/) {
    // Base implementation.
    Serialize(pointer, typeMeta, name, acceptor);
  }

  virtual size_t EstimateSerializedBlobSize(
      const void* /*pointer*/,
      TypeMeta /*typeMeta*/,
      c10::string_view /*name*/,
      const BlobSerializationOptions& /*options*/) {
    // Base implementation.
    // This returns 0 just to allow us to roll this out without needing to
    // define an implementation for all serializer types.  Returning a size of 0
    // for less-commonly used blob types is acceptable for now.  Eventually it
    // would be nice to ensure that this method is implemented for all
    // serializers and then make this method virtual.
    return 0;
  }
};

// The Blob serialization registry and serializer creator functions.
C10_DECLARE_TYPED_REGISTRY(
    BlobSerializerRegistry,
    TypeIdentifier,
    BlobSerializerBase,
    std::unique_ptr);
#define REGISTER_BLOB_SERIALIZER(id, ...) \
  C10_REGISTER_TYPED_CLASS(BlobSerializerRegistry, id, __VA_ARGS__)
// Creates an operator with the given operator definition.
inline unique_ptr<BlobSerializerBase> CreateSerializer(TypeIdentifier id) {
  return BlobSerializerRegistry()->Create(id);
}


/**
 * @brief BlobDeserializerBase is an abstract class that deserializes a blob
 * from a BlobProto or a TensorProto.
 */
class TORCH_API BlobDeserializerBase {
 public:
  virtual ~BlobDeserializerBase() {}

  // Deserializes from a BlobProto object.
  virtual void Deserialize(const BlobProto& proto, Blob* blob) = 0;
};

C10_DECLARE_REGISTRY(BlobDeserializerRegistry, BlobDeserializerBase);
#define REGISTER_BLOB_DESERIALIZER(name, ...) \
  C10_REGISTER_CLASS(BlobDeserializerRegistry, name, __VA_ARGS__)
// Creates an operator with the given operator definition.
inline unique_ptr<BlobDeserializerBase> CreateDeserializer(const string& type) {
  return BlobDeserializerRegistry()->Create(type);
}


} // namespace caffe2
