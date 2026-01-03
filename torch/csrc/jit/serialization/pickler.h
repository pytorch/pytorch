#pragma once

#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <ATen/Utils.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/qualified_name.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/FbcodeMaps.h>
#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/jit/serialization/pickler_helper.h>

namespace torch::jit {

using ::c10::IValue;

class TORCH_API Pickler {
  AT_DISALLOW_COPY_AND_ASSIGN(Pickler);

 public:
  Pickler(std::function<void(const char*, size_t)> writer)
      : Pickler(std::move(writer), nullptr, nullptr, nullptr) {}

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  Pickler(
      std::function<void(const char*, size_t)> writer,
      std::vector<at::Tensor>* tensor_table,
      std::function<c10::QualifiedName(const c10::ClassTypePtr&)> type_renamer,
      std::vector<c10::ClassTypePtr>* memoized_class_types,
      std::function<std::string(const at::Tensor&)> get_tensor_id = nullptr,
      bool tag_aggregates = true)
      : writer_(std::move(writer)),
        tensor_table_(tensor_table),
        type_renamer_(std::move(type_renamer)),
        memoized_class_types_(memoized_class_types),
        get_tensor_id_(std::move(get_tensor_id)),
        tag_aggregates_(tag_aggregates) {}
  ~Pickler();

  // Push protocol onto the stack
  void protocol();

  // Push STOP PickleOpCode onto the stack
  void stop();

  void pushIValue(const IValue& ivalue);

  void startTuple();
  void endTuple();

  const std::vector<at::Tensor>& tensorData() {
    return tensor_data_;
  }

  void pushEmptyDict();
  void pushDict(const IValue& ivalue);
  void pushInt(int64_t value);
  void pushLong(const std::string& data);

 private:
  void pushIValueImpl(const IValue& ivalue);
  void startTypeTag();
  void endTypeTag(const IValue& value);
  void pushBool(bool value);
  void pushDouble(double value);
  void pushComplexDouble(const IValue& value);
  void pushGenericList(const IValue& ivalue);
  void pushIntList(const IValue& ivalue);
  void pushList(const IValue& ivalue);
  void pushTensor(const IValue& ivalue);
  void pushTensorReference(const IValue& ivalue);
  void pushLiteralTensor(const IValue& ivalue);
  void pushLiteralSparseTensor(const at::Tensor& tensor);
  void pushTuple(const IValue& ivalue);
  void pushString(const std::string& string);
  void pushDevice(const IValue& ivalue);
#ifdef USE_DISTRIBUTED
  void pushRRef(const IValue& ivalue);
#endif
  // unmemoized version
  void pushStringImpl(const std::string& string);
  void pushStorageOfTensor(const at::Tensor& tensor);

  void pushBinGet(uint32_t memo_id);
  void pushSpecializedList(
      const IValue& ivalue,
      const char* list_name,
      const std::function<void(const IValue&)>& item_pusher);
  void pushGlobal(std::string_view module_name, std::string_view class_name);
  // raw string data is appended directly to the byte stream
  void pushBytes(const std::string& string);
  void pushTensorData(const at::Tensor& tensor);

  // Add a BINPUT op and return the memoization id used
  size_t pushNextBinPut();

  const void* getPointer(const IValue& ivalue);

  // Caller checks that bufferPos_ > 0
  void flushNonEmpty() {
    writer_(buffer_.data(), bufferPos_);
    bufferPos_ = 0;
  }

  void flush() {
    if (bufferPos_ != 0) {
      flushNonEmpty();
    }
  }

  // These convert values to bytes and add them to the stack (NB: since T is to
  // the left of a '::', its type cannot be deduced by the compiler so one must
  // explicitly instantiate the template, i.e. push<int>(int) works, push(int)
  // does not)
  static constexpr size_t kBufferSize = 256;
  template <typename T>
  void push(std::common_type_t<T> value) {
    const char* begin = reinterpret_cast<const char*>(&value);
    if (bufferPos_ + sizeof(T) > buffer_.size()) {
      flushNonEmpty();
    }
    static_assert(sizeof(T) <= kBufferSize, "Buffer size assumption");
    memcpy(buffer_.data() + bufferPos_, begin, sizeof(T));
    bufferPos_ += sizeof(T);
  }

  // Stream to write binary data to
  // Code shouldn't call writer_ directly without first flushing.
  std::function<void(const char*, size_t)> writer_;

  // Buffer to avoid calling a writer_ on a per-byte basis.
  std::array<char, kBufferSize> buffer_;
  size_t bufferPos_{0};

  // Stack of opcodes/data
  std::vector<char> stack_;

  // External table of tensors to serialize. If this is missing, then tensors
  // are serialized directly into the pickle
  std::vector<at::Tensor>* tensor_table_;

  // TODO: only use this if necessary (add a pass to find all shared ivalues,
  // and only memoize those)
  uint32_t memo_id_ = 0;

  // Memoization of IValues that have been written (index in table is used for
  // BINPUT opcodes) to enable shared references
  c10::FastMap<const void*, uint32_t> memoized_ivalue_map_;

  // because we de-dup ivalues based on their raw pointer address in the above
  // map we need to keep all the memoized values alive during the pickle.
  // Otherwise, it is possible that a raw address gets reused for another
  // object, and we will alias it to the old object at that address.
  std::vector<IValue> memoized_ivalues_;

  std::function<c10::QualifiedName(const c10::ClassTypePtr&)> type_renamer_;

  // List of all the types that it wrote, inspect from the IValues it wrote.
  std::vector<c10::ClassTypePtr>* memoized_class_types_;

  // Function to grab next id_name for tensor storage, function is responsible
  // for returning unique ids
  std::function<std::string(const at::Tensor&)> get_tensor_id_;

  // List of tensor storages to serialize in the same binary as the pickle data
  // similar to ivalues, they are memoized using BINPUT
  std::vector<at::Tensor> tensor_data_;
  c10::FastMap<const void*, uint32_t> memoized_storage_map_;

  c10::FastMap<std::string, uint32_t> memoized_globals_map_;
  c10::FastMap<std::string, uint32_t> memoized_strings_map_;
  c10::FastMap<std::string, uint32_t> memoized_devices_map_;
  // when true, List and Dict objects will be wrapped in a
  // torch.jit._pickle.restore_type_tag call to correctly set the dynamic
  // TorchScript type for the object. When true the thing unpickling must have
  // torch installed.
  bool tag_aggregates_;
};

} // namespace torch::jit
