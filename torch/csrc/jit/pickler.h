#pragma once

#include <string>
#include <vector>

#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <c10/util/ArrayRef.h>
#include <torch/csrc/utils/disallow_copy.h>

namespace torch {
namespace jit {

using ClassResolver =
    std::function<c10::StrongTypePtr(const c10::QualifiedName&)>;

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
  UNICODE = 'V',
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

enum PicklerClass : uint8_t {
  // A reference to the tensor table
  TENSOR = 0,
  // List[int]
  INTLIST = 1,
  // List[Tensor]
  TENSORLIST = 2,
  // List[float]
  DOUBLELIST = 3,
  // List[bool]
  BOOLLIST = 4
};

using ::c10::IValue;

struct WriteableTensorData {
  const char* data() const {
    return static_cast<const char*>(tensor_.storage().data());
  }
  size_t sizeInBytes() const {
    return size_;
  }
  size_t numel() const {
    return tensor_.storage().numel();
  }

 private:
  friend WriteableTensorData getWriteableTensorData(const at::Tensor& tensor);
  at::Tensor tensor_;
  uint64_t size_;
};

class Pickler {
  TH_DISALLOW_COPY_AND_ASSIGN(Pickler);

 public:
  Pickler(
      std::function<void(const char*, size_t)> writer,
      std::vector<at::Tensor>* tensor_table)
      : writer_(writer), tensor_table_(tensor_table) {}

  // Push protocol onto the stack
  void protocol();

  // Push STOP PickleOpCode onto the stack
  void stop();

  void pushIValue(const IValue& ivalue);

  void startTuple();
  void endTuple();

  const std::vector<WriteableTensorData>& tensorData() {
    return tensor_data_;
  }
  void pushEmptyDict();
  void pushDict(const IValue& ivalue);
  void pushInt(int64_t value);
  void pushLong(const std::string& data);

 private:
  void pushIValueImpl(const IValue& ivalue);
  void pushDouble(double value);
  void pushGenericList(const IValue& ivalue);
  void pushIntList(const IValue& ivalue);
  void pushList(const IValue& ivalue);
  void pushTensor(const IValue& ivalue);
  void pushTensorReference(const IValue& ivalue);
  void pushLiteralTensor(const IValue& ivalue);
  void pushTuple(const IValue& ivalue);
  void pushString(const std::string& string);
  // unmemoized version
  void pushStringImpl(const std::string& string);
  void pushStorageOfTensor(const at::Tensor& tensor);

  void pushBinGet(uint32_t memo_id);
  void pushClass(PicklerClass cls);
  void pushSpecializedList(
      const IValue& ivalue,
      PicklerClass cls,
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

  // These convert values to bytes and add them to the stack (NB: since T is to
  // the left of a '::', its type cannot be deduced by the compiler so one must
  // explicitly instantiate the template, i.e. push<int>(int) works, push(int)
  // does not)
  template <typename T>
  void push(typename std::common_type<T>::type value) {
    const char* begin = reinterpret_cast<const char*>(&value);
    writer_(begin, sizeof(T));
  }

  // Stream to write binary data to
  std::function<void(const char*, size_t)> writer_;

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

  // List of tensor storages to serialize in the same binary as the pickle data
  // similar to ivalues, they are memoized using BINPUT
  std::vector<WriteableTensorData> tensor_data_;
  std::unordered_map<const void*, uint32_t> memoized_storage_map_;

  std::unordered_map<std::string, uint32_t> memoized_globals_map_;
  std::unordered_map<std::string, uint32_t> memoized_strings_map_;
};

// returns a (tensor, record_size) for a tensor, converting it to a CPU tensor
// if necessary
WriteableTensorData getWriteableTensorData(const at::Tensor& tensor);

// return the value of the tensor's storage pointer
uint64_t getStorageKey(const at::Tensor& tensor);

// if the cls has __getstate__/__setstate__
// assert they have the right schema and return true,
// otherwise return false
bool checkHasValidSetGetState(const std::shared_ptr<c10::ClassType>& cls);

} // namespace jit
} // namespace torch
