#pragma once

#include "pickler.h"
#include <ATen/core/ivalue.h>
#include <caffe2/serialize/inline_container.h>

namespace torch {
namespace jit {

using TypeResolver =
    std::function<c10::StrongTypePtr(const c10::QualifiedName&)>;

using ObjLoader =
    std::function<c10::intrusive_ptr<c10::ivalue::Object>(at::StrongTypePtr, IValue)>;

// [unpickler refactor] there is some cruft around PickleOpCode::BUILD,
// PickleOpCode::NEWOBJ, and the last_opcode_ member below that should be deleted at
// some point, the Pickler doesn't produce it and it's only around to support
// models saved before 1.1
class Unpickler {
  TH_DISALLOW_COPY_AND_ASSIGN(Unpickler);

 public:
  // tensors inside the pickle are references to the tensor_table.
  // class_resolver is to resolve strong class type, type_resolver_ is
  // to resolve any JIT type. class_resolver and type_resolver are not merged
  // here because some use cases need to get strong class type that
  // type_resolver_ can not return.
  Unpickler(
      std::function<size_t(char*, size_t)> reader,
      TypeResolver type_resolver,
      const std::vector<at::Tensor>* tensor_table)
      : reader_(reader),
        tensor_table_(tensor_table),
        type_resolver_(std::move(type_resolver)),
        version_(caffe2::serialize::kProducedFileFormatVersion) {}

  // tensors inside the pickle contain meta-data, the raw tensor
  // dead is retrieved by calling `read_record`.
  Unpickler(
      std::function<size_t(char*, size_t)> reader,
      TypeResolver type_resolver,
      ObjLoader obj_loader,
      std::function<at::DataPtr(const std::string&)> read_record,
      c10::optional<at::Device> device)
      : reader_(reader),
        tensor_table_(nullptr),
        type_resolver_(std::move(type_resolver)),
        obj_loader_(std::move(obj_loader)),
        read_record_(std::move(read_record)),
        device_(std::move(device)),
        version_(caffe2::serialize::kProducedFileFormatVersion) {}

  // consume the pickle stream, producing an IValue from the contents.
  // Type Tags: the pickler will restore the type tags on
  // List and Dict objects when possible IValue is an Object.
  // Otherwise, Dict and List objects will end up with Any as their tag.
  // If you know the type of the ivalue, tags can be restored with
  // restoreAccurateTypeTags
  IValue parse_ivalue();

  // [type tag serialization]
  // This is used to determine whether to restore type tags be recursively
  // descending into the returned stack object (if version_number <= 2), or
  // if version_number >= 3, to use the type strings included in the pickle
  // archive for container types. By default this is set to
  // `kProducedFileFormatVersion` so unless you're loading a pickle file
  // from alongside a corresponding `version` file, you don't need to set
  // the version manually.
  void set_version(uint64_t version_number) {
    version_ = version_number;
  }

 private:
  // No arguments ensures that a template argument must be specified
  // so that the number of bytes read / type read is explicit
  template <typename T>
  T read() {
    T item;
    if (sizeof(T) <= buffer_remaining_) {
      // Fast path: entirely from buffer.
      memcpy(&item, buffer_.data() + buffer_pos_, sizeof(T));
      buffer_remaining_ -= sizeof(T);
      buffer_pos_ += sizeof(T);
    } else {
      // Don't over-template the slow path, to avoid code size bloat.
      readSlowWithBuffer(reinterpret_cast<char*>(&item), sizeof(T));
    }
    return item;
  }
  void readSlowWithBuffer(char *dest, size_t sz);
  std::string readBytes(size_t num_bytes);

  double readFloat();
  void readGlobal(
      const std::string& module_name,
      const std::string& class_name);
  void rebuildTensor(bool quantized);
  #ifdef USE_DISTRIBUTED
    void rebuildRRef();
  #endif
  PickleOpCode readInstruction();
  PickleOpCode readOpCode() {
    return static_cast<PickleOpCode>(read<uint8_t>());
  }
  std::string readString();
  void readList(IValue list_ivalue);
  void setInput(size_t memo_id);
  void run();

  // Returns the number of bytes read. This should statefully
  // remember the position. Don't call reader_ directly.
  std::function<size_t(char*, size_t)> reader_;
  // Small buffer to avoid calling reader_ on a per-byte basis.
  std::array<char, 256> buffer_;
  size_t buffer_pos_{0};
  size_t buffer_remaining_{0};

  std::vector<IValue> stack_;

  // globals are represented on the stack as IValue integer indices
  // into this list
  std::vector<std::function<void(void)>> globals_;
  std::vector<IValue> memo_table_;
  std::vector<size_t> marks_;
  const std::vector<at::Tensor>* tensor_table_;

  // When deserializing types on lists and dicts, cache the type here
  // so we don't have to parse the same type multiple times. Strings
  // are already de-duplicated and replaced with BINGETs in the
  // pickler, so we can just use the actual data pointer of each string.
  std::unordered_map<std::string, c10::TypePtr> type_cache_;

  // optionally nullptr, needs to be present for creating classes
  TypeResolver type_resolver_;
  ObjLoader obj_loader_;
  IValue empty_tuple_;


  std::function<at::DataPtr(const std::string&)> read_record_;
  c10::optional<at::Device> device_;

  // See [type tag serialization]
  uint64_t version_;
};

void restoreAccurateTypeTags(const IValue& root, const c10::TypePtr& type_tag);

} // namespace jit
} // namespace torch
