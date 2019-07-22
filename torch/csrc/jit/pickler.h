#pragma once

#include <string>
#include <vector>

#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <c10/util/ArrayRef.h>
#include <torch/csrc/utils/disallow_copy.h>

namespace torch {
namespace jit {

// See Python's pickletools.py for a detailed description of each of these codes
enum class OpCode : char {
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

class Pickler {
  TH_DISALLOW_COPY_AND_ASSIGN(Pickler);

 public:
  Pickler(std::vector<at::Tensor>* tensor_table = nullptr)
      : tensor_table_(tensor_table) {}

  const std::vector<char>& stack();

  // Push protocol onto the stack
  void start();

  // Push STOP OpCode onto the stack
  void finish();

  void addIValue(const IValue& ivalue);

  // See torch/serialization.py for details, pushes a magic number, torch
  // serialization version, and system info to the pickle archive all as
  // individual pickle programs
  void pushMetadata();

  void startTuple();
  void endTuple();

 private:
  void addIValueImpl(const IValue& ivalue);
  void pushDict(const IValue& ivalue);
  void pushDouble(const IValue& ivalue);
  void pushGenericList(const IValue& ivalue);
  void pushInt(const IValue& ivalue);
  void pushIntList(const IValue& ivalue);
  void pushList(const IValue& ivalue);
  void pushLiteralTensor(const IValue& ivalue);
  void pushMemoization(const IValue& ivalue);
  void pushTensor(const IValue& ivalue);
  void pushTensorReference(const IValue& ivalue);
  void pushTuple(const IValue& ivalue);
  void pushString(const std::string& string);
  // unmemoized version
  void pushStringImpl(const std::string& string);

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
    stack_.insert(stack_.end(), begin, begin + sizeof(T));
  }

  // Stack of opcodes/data
  std::vector<char> stack_;

  // External table of tensors to serialize. If this is missing, then tensors
  // are serialized directly into the pickle
  std::vector<at::Tensor>* tensor_table_;

  // List of tensors to serialize in the same binary as the pickle data
  std::vector<at::Tensor> literal_tensors_;

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

  std::unordered_map<std::string, uint32_t> memoized_globals_map_;
  std::unordered_map<std::string, uint32_t> memoized_strings_map_;
};

// [unpickler refactor] there is some cruft around OpCode::BUILD,
// OpCode::NEWOBJ, and the last_opcode_ member below that should be deleted at
// some point, the Pickler doesn't produce it and it's only around to support
// models saved before 1.1
class Unpickler {
  TH_DISALLOW_COPY_AND_ASSIGN(Unpickler);

 public:
  Unpickler(
      const void* data,
      size_t size,
      const std::vector<at::Tensor>* tensor_table,
      std::function<c10::StrongTypePtr(const c10::QualifiedName&)>
          class_resolver)
      : bytes_(static_cast<const uint8_t*>(data)),
        end_ptr_(bytes_ + size),
        tensor_table_(tensor_table),
        class_resolver_(class_resolver) {}

  std::vector<IValue> parse_ivalue_list();

 private:
  // No arguments ensures that a template arugment must be specified
  // so that the number of bytes read / type read is explicit
  template <typename T>
  T read() {
    TORCH_CHECK(
        bytes_ + sizeof(T) <= end_ptr_,
        "Unpickler overran buffer while reading a value");
    T item;
    std::memcpy(&item, bytes_, sizeof(T));
    bytes_ += sizeof(T);
    return item;
  }

  double readFloat();
  OpCode readInstruction();
  OpCode readOpCode();
  std::string readString();
  void readList();
  void setInput(size_t memo_id);
  void run();

  std::vector<IValue> stack_;
  // globals are represented on the stack as IValue integer indices
  // into this list
  std::vector<std::function<void(void)>> globals_;
  std::vector<IValue> memo_table_;
  std::vector<size_t> marks_;
  const uint8_t* bytes_;
  const uint8_t* end_ptr_;
  const std::vector<at::Tensor>* tensor_table_;

  // optionally nullptr, needs to be present for creating classes
  std::function<c10::StrongTypePtr(const c10::QualifiedName&)> class_resolver_;
  IValue empty_tuple_;
};

// returns a (tensor, record_size) for a tensor, converting it to a CPU tensor
// if necessary
std::pair<at::Tensor, uint64_t> getWriteableTensor(const at::Tensor& tensor);

// return the value of the tensor's storage pointer
uint64_t getStorageKey(const at::Tensor& tensor);

} // namespace jit
} // namespace torch
