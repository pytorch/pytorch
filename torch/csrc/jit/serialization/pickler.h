#pragma once

#include <ATen/core/qualified_name.h>
#include <string>
#include <utility>
#include <vector>

#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <c10/util/ArrayRef.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/utils/disallow_copy.h>

namespace torch {
namespace jit {

// See Python's pickletools.py for a detailed description of each of these codes
enum class PickleOpCode : char {
  MARK = '(',
  STOP = '.',
  POP = '0',
  POP_MARK = '1',
  DUP = '2',
  FLOAT = 'F',
  INT = 'I',
  BININT = 'J',
  BININT1 = 'K',
  LONG = 'L',
  BININT2 = 'M',
  NONE = 'N',
  PERSID = 'P',
  BINPERSID = 'Q',
  REDUCE = 'R',
  STRING = 'S',
  BINSTRING = 'T',
  SHORT_BINSTRING = 'U',
  // NB: Avoid using UNICODE as it is a macro in the Windows API
  UNICODE_ = 'V',
  BINUNICODE = 'X',
  APPEND = 'a',
  BUILD = 'b',
  GLOBAL = 'c',
  DICT = 'd',
  EMPTY_DICT = '}',
  APPENDS = 'e',
  GET = 'g',
  BINGET = 'h',
  INST = 'i',
  LONG_BINGET = 'j',
  LIST = 'l',
  EMPTY_LIST = ']',
  OBJ = 'o',
  PUT = 'p',
  BINPUT = 'q',
  LONG_BINPUT = 'r',
  SETITEM = 's',
  TUPLE = 't',
  EMPTY_TUPLE = ')',
  SETITEMS = 'u',
  BINFLOAT = 'G',

  // Protocol 2
  PROTO = '\x80',
  NEWOBJ = '\x81',
  EXT1 = '\x82',
  EXT2 = '\x83',
  EXT4 = '\x84',
  TUPLE1 = '\x85',
  TUPLE2 = '\x86',
  TUPLE3 = '\x87',
  NEWTRUE = '\x88',
  NEWFALSE = '\x89',
  LONG1 = '\x8a',
  LONG4 = '\x8b',

  // Protocol 3 (Python 3.x)
  BINBYTES = 'B',
  SHORT_BINBYTES = 'C',

  // Protocol 4
  SHORT_BINUNICODE = '\x8c',
  BINUNICODE8 = '\x8d',
  BINBYTES8 = '\x8e',
  EMPTY_SET = '\x8f',
  ADDITEMS = '\x90',
  FROZENSET = '\x91',
  NEWOBJ_EX = '\x92',
  STACK_GLOBAL = '\x93',
  MEMOIZE = '\x94',
  FRAME = '\x95'
};

using ::c10::IValue;

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct WriteableTensorData {
  const char* data() const {
    return static_cast<const char*>(tensor_.storage().data());
  }
  size_t sizeInBytes() const {
    return size_;
  }
  size_t nbytes() const {
    return tensor_.storage().nbytes();
  }
  bool storageHasDeleter() const {
    return tensor_.storage().data_ptr().get_context() != nullptr;
  }

 private:
  friend TORCH_API WriteableTensorData
  getWriteableTensorData(const at::Tensor& tensor, bool to_cpu);
  at::Tensor tensor_;
  uint64_t size_;
};

void setTypeTags(bool state);
bool getTypeTags();

class TORCH_API Pickler {
  TH_DISALLOW_COPY_AND_ASSIGN(Pickler);

 public:
  Pickler(std::function<void(const char*, size_t)> writer)
      : Pickler(std::move(writer), nullptr, nullptr, nullptr) {}

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  Pickler(
      std::function<void(const char*, size_t)> writer,
      std::vector<at::Tensor>* tensor_table,
      std::function<c10::QualifiedName(const c10::ClassTypePtr&)> type_renamer,
      std::vector<c10::ClassTypePtr>* memoized_class_types,
      std::function<std::string(const at::Tensor&)> get_tensor_id = nullptr)
      : writer_(std::move(writer)),
        tensor_table_(tensor_table),
        type_renamer_(std::move(type_renamer)),
        memoized_class_types_(memoized_class_types),
        get_tensor_id_(std::move(get_tensor_id)) {}
  // NOLINTNEXTLINE(bugprone-exception-escape)
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
  void pushGlobal(
      const std::string& module_name,
      const std::string& class_name);
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
  static CONSTEXPR_EXCEPT_WIN_CUDA size_t kBufferSize = 256;
  template <typename T>
  void push(typename std::common_type<T>::type value) {
    const char* begin = reinterpret_cast<const char*>(&value);
    if (bufferPos_ + sizeof(T) > buffer_.size()) {
      flushNonEmpty();
    }
    static_assert(sizeof(T) <= kBufferSize, "Buffer size assumption");
    memcpy(buffer_.data() + bufferPos_, begin, sizeof(T));
    bufferPos_ += sizeof(T);
  }

  // Stream to write binary data to
  // Code shouldn't call writer_ directly without first flush()ing.
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
  std::unordered_map<const void*, uint32_t> memoized_ivalue_map_;

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
  std::unordered_map<const void*, uint32_t> memoized_storage_map_;

  std::unordered_map<std::string, uint32_t> memoized_globals_map_;
  std::unordered_map<std::string, uint32_t> memoized_strings_map_;
  std::unordered_map<std::string, uint32_t> memoized_devices_map_;
};

// returns a (tensor, record_size) for a tensor, converting it to a CPU tensor
// if it was CUDA and to_cpu is True.
TORCH_API WriteableTensorData
getWriteableTensorData(const at::Tensor& tensor, bool to_cpu = true);

// return the value of the tensor's storage pointer
uint64_t getStorageKey(const at::Tensor& tensor);

// if the cls has __getstate__/__setstate__
// assert they have the right schema and return true,
// otherwise return false
bool checkHasValidSetGetState(const std::shared_ptr<c10::ClassType>& cls);

} // namespace jit
} // namespace torch
