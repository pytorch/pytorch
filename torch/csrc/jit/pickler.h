#pragma once

#include <string>
#include <vector>

#include <ATen/core/ivalue.h>
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

enum PicklerClass : uint8_t { TENSOR = 0, INTLIST = 1 };

using ::c10::IValue;

class Pickler {
  TH_DISALLOW_COPY_AND_ASSIGN(Pickler);

 public:
  Pickler(std::vector<at::Tensor>* tensor_table)
      : tensor_table_(tensor_table) {}

  const std::vector<char>& stack();
  void start();
  void finish();
  void addIValue(const IValue& ivalue);

 private:
  void pushBinGet(uint32_t memo_id);
  void pushMemoizedString(const IValue& ivalue);
  void pushString(const std::string& string);
  void pushTensor(const IValue& ivalue);
  void pushDouble(const IValue& ivalue);
  void pushMemoization(const void* item);
  void pushMemoization(const IValue& ivalue);
  void pushList(const IValue& ivalue);
  void pushIntList(const IValue& ivalue);
  void pushTuple(const IValue& ivalue);
  void pushDict(const IValue& ivalue);
  void pushClass(PicklerClass cls);
  void pushInt(const IValue& ivalue);
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

  // Memoization of IValues that have been written (index in table is used for
  // BINPUT opcodes) to enable shared references
  std::unordered_map<const void*, uint32_t> memo_map_;

  // External table of tensors to serialize
  std::vector<at::Tensor>* tensor_table_;

  // TODO: only use this if necessary (add a pass to find all shared ivalues,
  // and only memoize those)
  uint32_t memo_id = 0;
};

// An item in the unpickler stack. There needs to be a way to differentiate
// between a GLOBAL item (PicklerClass) and a normal value item (IValue)
struct StackItem {
  StackItem(IValue ivalue)
      : pickler_class_(c10::nullopt), ivalue_(std::move(ivalue)) {}
  StackItem(PicklerClass pickler_class)
      : pickler_class_(pickler_class), ivalue_(c10::nullopt) {}

  IValue ivalue() {
    return *ivalue_;
  }

  PicklerClass pickler_class() {
    return *pickler_class_;
  }

  c10::optional<IValue> ivalue_opt() {
    return ivalue_;
  }

  c10::optional<PicklerClass> pickler_class_opt() {
    return pickler_class_;
  }

 private:
  c10::optional<PicklerClass> pickler_class_;
  c10::optional<IValue> ivalue_;
};

// [unpickler refactor] there is some cruft around OpCode::BUILD,
// OpCode::NEWOBJ, and the last_opcode_ member below that should be deleted at
// some point, the Pickler doesn't produce it and it's only around to support
// models saved before 1.1
class Unpickler {
  TH_DISALLOW_COPY_AND_ASSIGN(Unpickler);

 public:
  Unpickler(
      void* data,
      size_t size,
      const std::vector<at::Tensor>* tensor_table)
      : bytes_(static_cast<const uint8_t*>(data)),
        end_ptr_(bytes_ + size),
        tensor_table_(tensor_table),
        last_opcode_(OpCode::STOP) {}

  std::vector<IValue> parse_ivalue_list();

 private:
  // No arguments ensures that a template arugment must be specified
  // so that the number of bytes read / type read is explicit
  template <typename T>
  T read() {
    AT_CHECK(
        bytes_ + sizeof(T) <= end_ptr_,
        "Unpickler overran buffer while reading a value");
    T item;
    std::memcpy(&item, bytes_, sizeof(T));
    bytes_ += sizeof(T);
    return item;
  }

  double readFloat();
  void run();
  OpCode readInstruction();
  std::string readString();
  OpCode readOpCode();
  void readList();

  std::vector<StackItem> stack_;
  std::vector<StackItem> memo_table_;
  std::vector<size_t> marks_;
  const uint8_t* bytes_;
  const uint8_t* end_ptr_;
  const std::vector<at::Tensor>* tensor_table_;

  // [unpickler refactor]
  OpCode last_opcode_;
};

} // namespace jit
} // namespace torch
