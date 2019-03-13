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

 public:
  std::vector<char>& stack() {
    return stack_;
  }

  void start() {
    push<OpCode>(OpCode::PROTO);
    push<uint8_t>(2);

    // All attributes get pushed into a list and their indices saved in the
    // module def
    push<OpCode>(OpCode::EMPTY_LIST);
    push<OpCode>(OpCode::MARK);
  }

  void finish() {
    push<OpCode>(OpCode::APPENDS);
    push<OpCode>(OpCode::STOP);
  }

  void addIValue(const IValue& ivalue) {
    // Check if reference ivalue has been saved before
    const void* ivalue_ptr = getPointer(ivalue);
    if (ivalue_ptr) {
      auto memo_entry = memo_.find(ivalue_ptr);
      if (memo_entry != memo_.end()) {
        // This value has already been pushed, just do a BINGET
        pushBinGet(memo_entry->second);
        return;
      }
    }

    if (ivalue.isTensor()) {
      pushTensor(ivalue);
    } else if (ivalue.isTuple()) {
      pushTuple(ivalue);
    } else if (ivalue.isDouble()) {
      pushDouble(ivalue);
    } else if (ivalue.isInt()) {
      // TODO: use BININT1/BININT2/LONG if possible/necessary
      AT_ASSERT(ivalue.toInt() <= std::numeric_limits<int32_t>::max());
      push<OpCode>(OpCode::BININT);
      push<int32_t>(ivalue.toInt());
    } else if (ivalue.isBool()) {
      if (ivalue.toBool()) {
        push<OpCode>(OpCode::NEWTRUE);
      } else {
        push<OpCode>(OpCode::NEWFALSE);
      }
    } else if (ivalue.isString()) {
      pushString(ivalue);
    } else if (ivalue.isGenericList()) {
      pushList(ivalue);
    } else if (ivalue.isGenericDict()) {
      pushDict(ivalue);
    } else if (ivalue.isNone()) {
      push<OpCode>(OpCode::NONE);
    } else {
      AT_ERROR("Unknown IValue type for pickling");
    }
  }

 private:
  const void* getPointer(const IValue& ivalue) {
    if (ivalue.isGenericDict()) {
      return &(ivalue.toGenericDictRef());
    } else if (ivalue.isGenericList()) {
      return &(ivalue.toGenericListRef());
    } else if (ivalue.isTuple()) {
      return &(ivalue.toTuple()->elements());
    } else if (ivalue.isString()) {
      return &(ivalue.toStringRef());
    }

    return nullptr;
  }

  void pushBinGet(uint32_t memo_id) {
    if (memo_id <= std::numeric_limits<uint8_t>::max()) {
      push<OpCode>(OpCode::BINGET);
      push<uint8_t>(memo_id);
    } else {
      // Memoized too many items, issue a LONG_BINGET instead
      push<OpCode>(OpCode::LONG_BINGET);
      push<uint32_t>(memo_id);
    }
  }

  void pushString(const IValue& ivalue) {
    auto string = ivalue.toStringRef();

    push<OpCode>(OpCode::BINUNICODE);
    push<uint32_t>(string.size());
    pushString(string);
    pushMemoization(ivalue);
  }

  void pushString(const std::string& string) {
    stack_.insert(stack_.end(), string.begin(), string.end());
  }

  void pushTensor(const IValue& ivalue) {
    // Write it to the tensor table
    auto memo_entry = memo_.find(&tensor_class_name_);
    if (memo_entry == memo_.end()) {
      push<OpCode>(OpCode::GLOBAL);
      // Module name + "\n"
      pushString(tensor_class_name_.first);
      // Class name + "\n"
      pushString(tensor_class_name_.second);
      pushMemoization((void*)&tensor_class_name_);
    } else {
      pushBinGet(memo_entry->second);
    }
    push<OpCode>(OpCode::EMPTY_TUPLE);
    push<OpCode>(OpCode::NEWOBJ);

    tensor_table_.push_back(ivalue.toTensor());
    auto tensor_id = tensor_table_.size() - 1;
    push<OpCode>(OpCode::BININT);
    push<uint32_t>(tensor_id);

    push<OpCode>(OpCode::BUILD);
  }

  void pushDouble(const IValue& ivalue) {
    double value = ivalue.toDouble();
    AT_ASSERT(sizeof(double) == 8);
    char* bytes = reinterpret_cast<char*>(&value);

    push<OpCode>(OpCode::BINFLOAT);
    for (size_t i = 0; i < 8; ++i) {
      push<char>(bytes[8 - i - 1]);
    }
  }

  using ivalue_pair = std::pair<IValue, IValue>;

  struct IValueComparator {
    bool operator()(const ivalue_pair& lhs, const ivalue_pair& rhs) const {
      if (lhs.first.isString()) {
        return lhs.first.toString() < rhs.first.toString();
      }
      if (lhs.first.isInt()) {
        return lhs.first.toInt() < rhs.first.toInt();
      }
      if (lhs.first.isDouble()) {
        return lhs.first.toDouble() < rhs.first.toDouble();
      }
      AT_ERROR("Uncomparable IValue types");
    }
  };

  void pushDict(const IValue& ivalue) {
    auto dict = ivalue.toGenericDictRef();

    push<OpCode>(OpCode::EMPTY_DICT);
    pushMemoization(ivalue);

    push<OpCode>(OpCode::MARK);

    // Sort the dict for deterministic keys
    std::multiset<std::pair<IValue, IValue>, IValueComparator> dict_items(
        dict.begin(), dict.end());

    for (auto pair : dict_items) {
      addIValue(pair.first);
      addIValue(pair.second);
    }

    push<OpCode>(OpCode::SETITEMS);
  }

  void pushMemoization(const void* item) {
    AT_ASSERT(item != nullptr);
    if (memo_id <= std::numeric_limits<uint8_t>::max()) {
      push<OpCode>(OpCode::BINPUT);
      push<uint8_t>(memo_id);
    } else {
      // Memoized too many items, issue a LONG_BINPUT instead
      push<OpCode>(OpCode::LONG_BINPUT);
      push<uint32_t>(memo_id);
    }
    memo_[item] = memo_id;
    ++memo_id;
  }

  void pushMemoization(const IValue& ivalue) {
    pushMemoization(getPointer(ivalue));
  }

  void pushList(const IValue& ivalue) {
    auto list = ivalue.toGenericListRef();
    push<OpCode>(OpCode::EMPTY_LIST);
    pushMemoization(ivalue);

    push<OpCode>(OpCode::MARK);

    for (auto item : list) {
      addIValue(item);
    }

    push<OpCode>(OpCode::APPENDS);
  }

  void pushTuple(const IValue& ivalue) {
    // TODO: Small tuple unrolling (e.g. TUPLE3)
    push<OpCode>(OpCode::MARK);
    auto tuple = ivalue.toTuple()->elements();

    for (const auto& item : tuple) {
      addIValue(item);
    }

    push<OpCode>(OpCode::TUPLE);
    pushMemoization(ivalue);
  }

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

  const std::vector<IValue> get_ivalue_list() {
    run();
    AT_ASSERT(stack_.size() == 1);
    return stack_[0].toGenericListRef();
  }

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

  double readFloat() {
    AT_ASSERT(sizeof(double) == 8)
    char float_data[8];

    // Pickle floats are big endian, so reverse the bytes
    for (size_t i = 0; i < 8; ++i) {
      float_data[i] = bytes_[8 - 1 - i];
    }
    bytes_ += 8;

    return *(reinterpret_cast<double*>(float_data));
  }

  void run() {
    // Expect a PROTO opcode and protocol number at the start of blob
    AT_ASSERT(readOpCode() == OpCode::PROTO);
    uint8_t protocol = read<uint8_t>();
    AT_CHECK(
        protocol == 2,
        "Only Pickle protocol 2 is supported, found protocol = ",
        protocol);

    while (bytes_ < end_ptr_) {
      OpCode opcode = readInstruction();
      if (opcode == OpCode::STOP) {
        return;
      }
      last_opcode_ = opcode;
    }

    AT_ERROR("Overran buffer while unpickling data, didn't find STOP opcode");
  }

  OpCode readInstruction() {
    auto op = readOpCode();
    switch (op) {
      case OpCode::EMPTY_LIST: {
        // TODO: Use fake classes to mark list specializations
        stack_.push_back(std::vector<IValue>());
        break;
      }
      case OpCode::EMPTY_TUPLE: {
        stack_.push_back(c10::ivalue::Tuple::create({}));
        break;
      }
      case OpCode::BINPUT: {
        size_t memo_id = read<uint8_t>();
        if (memo_.size() <= memo_id) {
          memo_.reserve(1 + 2 * memo_id);
        }
        memo_.push_back(stack_.back());
        break;
      }
      case OpCode::MARK: {
        // Mark location of the container ivalue in the stack
        marks_.push_back(stack_.size());
        break;
      }
      case OpCode::BININT: {
        int32_t value = read<uint32_t>();
        stack_.push_back(int64_t(value));
        break;
      }
      case OpCode::BINUNICODE: {
        int32_t length = read<uint32_t>();
        const char* characters = reinterpret_cast<const char*>(bytes_);
        bytes_ += length;
        stack_.push_back(std::string(characters, /*n=*/length));
        break;
      }
      case OpCode::BINFLOAT:
        stack_.push_back(readFloat());
        break;
      case OpCode::TUPLE: {
        size_t start = marks_.back();
        marks_.pop_back();
        IValue tup = c10::ivalue::Tuple::create(
            std::vector<IValue>(stack_.begin() + start, stack_.end()));
        stack_.resize(start);
        stack_.push_back(tup);
      } break;
      case OpCode::EMPTY_DICT:
        stack_.push_back(c10::ivalue::UnorderedMap());
        break;
      case OpCode::APPENDS: {
        size_t start = marks_.back();
        marks_.pop_back();
        auto list = stack_[start - 1].toGenericList();
        list->elements().insert(
            list->elements().end(), stack_.begin() + start, stack_.end());
        stack_.resize(start);
      } break;
      case OpCode::SETITEMS: {
        size_t start = marks_.back();
        marks_.pop_back();
        auto dict = stack_[start - 1].toGenericDict();
        for (size_t i = start; i < stack_.size(); i += 2) {
          dict->elements()[stack_[i]] = stack_[i + 1];
        }
        stack_.resize(start);
      } break;
      case OpCode::BINGET: {
        stack_.push_back(memo_.at(read<uint8_t>()));
      } break;
      case OpCode::STOP:
        return op;
      case OpCode::GLOBAL: {
        AT_ASSERT(readString() == "__main__");
        // Push class name to stack
        stack_.push_back(readString());
      } break;
      case OpCode::NEWOBJ: {
        // Do nothing, should be followed by BUILD
      } break;
      case OpCode::BUILD: {
        // TODO: this only works for Tensors
        auto tensor_id = stack_.back().toInt();
        stack_.pop_back();
        // pop empty tuple
        stack_.pop_back();
        auto class_name = stack_.back().toStringRef();
        stack_.pop_back();
        AT_ASSERT(class_name == "TensorID");
        stack_.push_back(tensor_table_.at(tensor_id));

      } break;
      default:
        AT_ERROR("Unknown opcode for unpickling");
    }

    return op;
  }

  std::string readString(char terminator = '\n') {
    const char* chars = reinterpret_cast<const char*>(bytes_);
    size_t n = 0;
    while (true) {
      char c = chars[n];
      if (c == '\n') {
        break;
      }
      AT_ASSERT(isValidChar(c));
      // Increment after to exclude newline from string
      ++n;
    }

    // Increment by string length + newline char
    bytes_ += n + 1;
    return std::string(chars, n);
  }

  bool isValidChar(char value) {
    return value >= 'A' && value <= 'z';
  }

  OpCode readOpCode() {
    return static_cast<OpCode>(read<uint8_t>());
  }

  std::vector<IValue> stack_;
  std::vector<IValue> memo_;
  std::vector<size_t> marks_;
  const uint8_t* bytes_;
  const uint8_t* end_ptr_;
  const std::vector<at::Tensor>& tensor_table_;
  OpCode last_opcode_;
};

} // namespace jit
} // namespace torch
