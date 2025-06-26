#pragma once

#include <ATen/core/ivalue.h>
#include <c10/util/ArrayRef.h>
#include <caffe2/serialize/inline_container.h>

#include <torch/csrc/Export.h>
#include <torch/csrc/jit/frontend/script_type_parser.h>
#include <torch/csrc/jit/serialization/pickler_helper.h>

namespace torch::jit {

using TypeResolver =
    std::function<c10::StrongTypePtr(const c10::QualifiedName&)>;

using ObjLoader = std::function<
    c10::intrusive_ptr<c10::ivalue::Object>(const at::StrongTypePtr&, IValue)>;

class DeserializationStorageContext;

// [unpickler refactor] there is some cruft around PickleOpCode::BUILD,
// PickleOpCode::NEWOBJ, and the last_opcode_ member below that should be
// deleted at some point, the Pickler doesn't produce it and it's only around to
// support models saved before 1.1
class TORCH_API Unpickler {
  AT_DISALLOW_COPY_AND_ASSIGN(Unpickler);

  using TypeParserT = c10::TypePtr (*)(const std::string&);

 public:
  // tensors inside the pickle are references to the tensor_table.
  // class_resolver is to resolve strong class type, type_resolver_ is
  // to resolve any JIT type. class_resolver and type_resolver are not merged
  // here because some use cases need to get strong class type that
  // type_resolver_ can not return.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  Unpickler(
      std::function<size_t(char*, size_t)> reader,
      TypeResolver type_resolver,
      c10::ArrayRef<at::Tensor> tensor_table,
      TypeParserT type_parser = defaultTypeParser)
      : reader_(std::move(reader)),
        tensor_table_(tensor_table),
        type_resolver_(std::move(type_resolver)),
        use_storage_device_(false),
        type_parser_(type_parser),
        version_(caffe2::serialize::kProducedFileFormatVersion) {}

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  Unpickler(
      std::function<size_t(char*, size_t)> reader,
      TypeResolver type_resolver,
      c10::ArrayRef<at::Tensor> tensor_table,
      ObjLoader obj_loader,
      TypeParserT type_parser = defaultTypeParser)
      : reader_(std::move(reader)),
        tensor_table_(tensor_table),
        type_resolver_(std::move(type_resolver)),
        obj_loader_(std::move(obj_loader)),
        use_storage_device_(false),
        type_parser_(type_parser),
        version_(caffe2::serialize::kProducedFileFormatVersion) {}

  // tensors inside the pickle contain meta-data, the raw tensor
  // dead is retrieved by calling `read_record`.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  Unpickler(
      std::function<size_t(char*, size_t)> reader,
      TypeResolver type_resolver,
      ObjLoader obj_loader,
      std::function<at::DataPtr(const std::string&)> read_record,
      std::optional<at::Device> device,
      bool use_storage_device = false,
      TypeParserT type_parser = defaultTypeParser,
      std::shared_ptr<DeserializationStorageContext> storage_context = nullptr)
      : reader_(std::move(reader)),
        type_resolver_(std::move(type_resolver)),
        obj_loader_(std::move(obj_loader)),
        read_record_(std::move(read_record)),
        device_(device),
        use_storage_device_(use_storage_device),
        type_parser_(type_parser),
        storage_context_(std::move(storage_context)),
        version_(caffe2::serialize::kProducedFileFormatVersion) {}

  Unpickler(Unpickler&&) = delete;
  Unpickler& operator=(Unpickler&&) = delete;
  ~Unpickler() = default;

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

  static c10::TypePtr defaultTypeParser(const std::string& str) {
    ScriptTypeParser parser;
    return parser.parseType(str);
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
  void readSlowWithBuffer(char* dest, size_t sz);
  std::string readBytes(size_t num_bytes);

  double readFloat();
  void readGlobal(
      const std::string& module_name,
      const std::string& class_name);
  void rebuildTensor(bool quantized);
  void rebuildTensorFromTypeV2();
  void rebuildSparseTensor();
#ifdef USE_DISTRIBUTED
  void rebuildRRef();
#endif
  PickleOpCode readInstruction();
  PickleOpCode readOpCode() {
    return static_cast<PickleOpCode>(read<uint8_t>());
  }
  std::string readString();
  void readList(IValue list_ivalue);
  void readListElements(IValue list_ivalue, size_t start);
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
  c10::ArrayRef<at::Tensor> tensor_table_;

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
  std::optional<at::Device> device_;
  // When set to true, Unpickler will ignore the pickled device and use the
  // device of the DataPtr returned by the read_record_ function. The default
  // value of this flag is false.
  const bool use_storage_device_;

  TypeParserT type_parser_{defaultTypeParser};

  // Used for torch.package to enable sharing of storages across
  // ScriptModules and eager modules
  std::shared_ptr<DeserializationStorageContext> storage_context_;

  // See [type tag serialization]
  uint64_t version_;

  // See [NOTE] skip_next_read_global
  uint8_t skip_next_read_global = 0;
};

void restoreAccurateTypeTags(const IValue& root, const c10::TypePtr& type_tag);

} // namespace torch::jit
