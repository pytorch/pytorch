#include <string>
#include <vector>

#include <ATen/core/ivalue.h>

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

using ::c10::IValue;

struct Pickler {
  Pickler(std::vector<at::Tensor>& tensor_table)
      : tensor_table_(tensor_table),
        tensor_class_name_("__main__\n", "TensorID\n") {}

  const std::vector<char>& stack();
  void start();
  void finish();
  void addIValue(const IValue& ivalue);

 private:
  void pushBinGet(uint32_t memo_id);
  void pushString(const IValue& ivalue);
  void pushString(const std::string& string);
  void pushTensor(const IValue& ivalue);
  void pushDouble(const IValue& ivalue);
  void pushMemoization(const void* item);
  void pushMemoization(const IValue& ivalue);
  void pushList(const IValue& ivalue);
  void pushTuple(const IValue& ivalue);
  void pushDict(const IValue& ivalue);
  const void* getPointer(const IValue& ivalue);

  template <typename T>
  struct identity {
    typedef T type;
  };
  // identity is a trick to force the template type to be manually specified
  // (e.g. push<uint8_t>(2) compiles, push(2) does not). We want this so the
  // types pushed on the stack are explicit:
  // https://stackoverflow.com/questions/28171518/is-there-a-way-to-force-the-user-to-explicitly-specify-the-template-argument-typ
  template <typename T>
  void push(typename identity<const T&>::type value) {
    const char* begin = reinterpret_cast<const char*>(&value);
    stack_.insert(stack_.end(), begin, begin + sizeof(T));
  }

  // Stack of opcodes/data
  std::vector<char> stack_;

  // Memoization of IValues that have been written (index in table is used for
  // BINPUT opcodes) to enable shared references
  std::unordered_map<const void*, uint32_t> memo_;

  // External table of tensors to serialize
  std::vector<at::Tensor>& tensor_table_;

  // Module name, class name for fake tensor class
  std::pair<std::string, std::string> tensor_class_name_;

  // TODO: only use this if necessary (add a pass to find all shared ivalues,
  // and only memoize those)
  uint32_t memo_id = 0;
};

struct Unpickler {
  Unpickler(
      void* data,
      size_t size,
      const std::vector<at::Tensor>& tensor_table)
      : bytes_(static_cast<const uint8_t*>(data)),
        end_ptr_(bytes_ + size),
        tensor_table_(tensor_table) {}

  const std::vector<IValue> get_ivalue_list();

 private:
  // No arguments ensures that a template arugment must be specified
  // so that the number of bytes read / type read is explicit
  template <typename T>
  T read() {
    AT_CHECK(
        bytes_ + sizeof(T) <= end_ptr_,
        "Unpickler overran buffer while reading a value");
    const T* data = reinterpret_cast<const T*>(bytes_);
    bytes_ += sizeof(T);
    return *data;
  }

  double readFloat();
  void run();
  OpCode readInstruction();
  std::string readString(char terminator = '\n');
  bool isValidChar(char value);
  OpCode readOpCode();

  std::vector<IValue> stack_;
  std::vector<IValue> memo_;
  std::vector<size_t> marks_;
  const uint8_t* bytes_;
  const uint8_t* end_ptr_;
  const std::vector<at::Tensor>& tensor_table_;
};

} // namespace jit
} // namespace torch
