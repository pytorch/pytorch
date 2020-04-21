#ifndef CAFFE2_OPERATORS_DATASET_OPS_H_
#define CAFFE2_OPERATORS_DATASET_OPS_H_

#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include "caffe2/core/blob.h"
#include "caffe2/core/blob_serialization.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {
namespace dataset_ops {

// used for lengths tensors in the dataset
using TLength = int32_t;
// used for all internal dataset operations (offsets, sizes to read, etc.)
using TOffset = int64_t;

/**
 * Provides functionality to iterate across a list of tensors where some
 * of those tensors represent lengths in a hierarchical structure.
 */
class TreeIterator {
 public:
  struct FieldDesc {
    int id;
    int lengthFieldId = -1;
    std::string name;
  };

  explicit TreeIterator(const std::vector<std::string>& fields);

  void advance(
      const std::vector<const TLength*>& lengths,
      std::vector<TOffset>& offsets,
      std::vector<TOffset>& sizes,
      std::vector<TOffset>& limits,
      TOffset num);

  // Corresponds to the number of fields that have "length" as its last name
  int numLengthFields() const {
    return lengthFieldIds_.size();
  }

  // Corresponds to the number of length fields + 1 (for the top-level domain)
  int numOffsetFields() const {
    return numLengthFields() + 1;
  }

  // Get lengthField description for the given field
  const FieldDesc* lengthFieldFor(const FieldDesc& desc) {
    return (desc.lengthFieldId == -1)
        ? nullptr
        : &fields_.at(lengthFieldIds_.at(desc.lengthFieldId));
  }

  // Get lengthField description for the given lengthFieldId, where
  // 0 <= lengthFieldId < numLengthFields()
  const FieldDesc& lengthField(int lengthFieldId) {
    return fields_.at(lengthFieldIds_.at(lengthFieldId));
  }

  // Returns the index into the 'offset' vector for the given field.
  int offsetFieldIdFor(const FieldDesc& fieldDesc) {
    return fieldDesc.lengthFieldId + 1;
  }

  // Returns the field description for all fields.
  const std::vector<FieldDesc>& fields() {
    return fields_;
  }

  const std::vector<int>& lengthFieldIds() const {
    return lengthFieldIds_;
  }

 private:
  // Description of each field
  std::vector<FieldDesc> fields_;
  // Index into fields_ above for the fields that are lengths.
  std::vector<int> lengthFieldIds_;
};

class TreeCursor {
 public:
  explicit TreeCursor(const TreeIterator& iterator) : it(iterator) {}
  std::vector<TOffset> offsets;
  std::mutex mutex_;
  TreeIterator it;
};

/**
 * Simple wrapper class allowing an easy traversal of the tensors representing
 * the hirerarchical structure.
 */
class TreeWalker {
 public:
  TreeWalker(const vector<const Blob*>& inputs, TreeCursor& cursor);

  // Returns the number of records in a dataset
  inline TOffset size() const {
    return limits_.at(0);
  }

  void advance();

 private:
  inline const TensorCPU& input(int32_t idx) const {
    return inputs_[idx]->Get<TensorCPU>();
  }

  // TODO: Change to fieldDesc
  inline const TreeIterator::FieldDesc& field(int idx) const {
    return cursor_.it.fields().at(idx);
  }

  inline int lengthIdx(int fieldId) const {
    return field(fieldId).lengthFieldId + 1;
  }

  inline TOffset offset(int fieldId) const {
    return prevOffsets_[lengthIdx(fieldId)];
  }

  std::vector<int64_t> fieldDim(int fieldId) const;

  void* fieldPtr(int fieldId) const;

 public:
  // Simple Proxy class to expose nicer API for field access
  class Field {
   public:
    Field(TreeWalker& walker, int fieldId)
        : walker_(walker), fieldId_(fieldId) {}

    inline std::vector<int64_t> dim() const {
      return walker_.fieldDim(fieldId_);
    }

    inline int64_t size() const {
      int64_t size = 1;
      for (const auto d : dim()) {
        size *= d;
      }
      return size;
    }

    inline const TypeMeta& meta() const {
      return walker_.input(fieldId_).dtype();
    }

    inline void* ptr() const {
      return walker_.fieldPtr(fieldId_);
    }

    int fieldId() const {
      return fieldId_;
    }

    inline TOffset offset() const {
      return walker_.offset(fieldId_);
    }

   private:
    const TreeWalker& walker_;
    const int fieldId_;
  };

  // Notice that a reference is returned. If advance() is called the fields will
  // be updated to represent the new state.
  inline const std::vector<Field>& fields() const {
    return fields_;
  }

 private:
  void gatherLengthData();

  void gatherSizeLimits();

  const vector<const Blob*>& inputs_;
  TreeCursor& cursor_;
  std::vector<Field> fields_;

  std::vector<const TLength*> lengths_;
  std::vector<TOffset> limits_;
  std::vector<TOffset> sizes_;
  std::vector<TOffset> offsets_;
  std::vector<TOffset> prevOffsets_;
};

using SharedTensorVectorPtr = std::shared_ptr<std::vector<TensorCPU>>;

using Shared2DTensorVectorPtr =
    std::shared_ptr<std::vector<std::vector<caffe2::TensorCPU>>>;

using Tensor2DVector = std::vector<std::vector<caffe2::TensorCPU>>;

using TensorVectorPtr = std::unique_ptr<std::vector<Tensor>>;

class SharedTensorVectorPtrSerializer : public BlobSerializerBase {
 public:
  void Serialize(
      const void* pointer,
      TypeMeta typeMeta,
      const string& name,
      BlobSerializerBase::SerializationAcceptor acceptor) override;
};

class SharedTensorVectorPtrDeserializer : public BlobDeserializerBase {
 public:
  void Deserialize(const BlobProto& proto, Blob* blob) override;
};

} // namespace dataset_ops
} // namespace caffe2

#endif // CAFFE2_OPERATORS_DATASET_OPS_H_
