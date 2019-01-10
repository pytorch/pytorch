#pragma once

#include <string>
#include <functional>

namespace caffe2 {

class Blob;

constexpr int kDefaultChunkSize = -1;
constexpr int kNoChunking = 0;

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
     std::function<void(const std::string& blobName, const std::string& data)>;
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
  virtual void Serialize(const Blob& blob, const std::string& name,
                        SerializationAcceptor acceptor) = 0;

  virtual void SerializeWithChunkSize(
      const Blob& blob,
      const std::string& name,
      SerializationAcceptor acceptor,
      int /*chunk_size*/) {
    // Base implementation.
    Serialize(blob, name, acceptor);
  }
};

} // namespace caffe2
