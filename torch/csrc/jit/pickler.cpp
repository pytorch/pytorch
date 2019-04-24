#include <torch/csrc/jit/pickler.h>

namespace torch {
namespace jit {

using ::c10::IValue;

PicklerClass getClass(const std::string& str) {
  if (str == "build_tensor_from_id") {
    return PicklerClass::TENSOR;
  } else if (str == "build_intlist") {
    return PicklerClass::INTLIST;
  }
  AT_ERROR("Unknown class name for unpickler: ", str);
}

const std::string& getClassName(PicklerClass cls) {
  static const std::string tensor_class("build_tensor_from_id\n");
  static const std::string intlist_class("build_intlist\n");
  switch (cls) {
    case PicklerClass::TENSOR:
      return tensor_class;
    case PicklerClass::INTLIST:
      return intlist_class;
    default:
      AT_ERROR("Unknown class for pickler");
  }
}

const std::string& getModuleName() {
  static const std::string module_name("torch.jit._pickle\n");
  return module_name;
}

const std::vector<char>& Pickler::stack() {
  return stack_;
}

void Pickler::start() {
  push<OpCode>(OpCode::PROTO);
  push<uint8_t>(2);

  // All attributes get pushed into a list and their indices saved in the
  // module def
  push<OpCode>(OpCode::EMPTY_LIST);
  push<OpCode>(OpCode::MARK);
}

void Pickler::finish() {
  push<OpCode>(OpCode::APPENDS);
  push<OpCode>(OpCode::STOP);
}

void Pickler::addIValue(const IValue& ivalue) {
  // Check if reference ivalue has been saved before
  const void* ivalue_ptr = getPointer(ivalue);
  if (ivalue_ptr) {
    auto memo_entry = memo_map_.find(ivalue_ptr);
    if (memo_entry != memo_map_.end()) {
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
    pushInt(ivalue);
  } else if (ivalue.isBool()) {
    if (ivalue.toBool()) {
      push<OpCode>(OpCode::NEWTRUE);
    } else {
      push<OpCode>(OpCode::NEWFALSE);
    }
  } else if (ivalue.isString()) {
    pushMemoizedString(ivalue);
  } else if (ivalue.isGenericList()) {
    pushList(ivalue);
  } else if (ivalue.isGenericDict()) {
    pushDict(ivalue);
  } else if (ivalue.isNone()) {
    push<OpCode>(OpCode::NONE);
  } else if (ivalue.isIntList()) {
    pushIntList(ivalue);
  } else {
    AT_ERROR("Unknown IValue type for pickling: ", ivalue.tagKind());
  }
}

/// Returns a void* uniquely identifying this IValue's data. For non-containers,
/// returns nullptr.
const void* Pickler::getPointer(const IValue& ivalue) {
  if (ivalue.isGenericDict()) {
    return ivalue.toGenericDict().get();
  } else if (ivalue.isGenericList()) {
    return ivalue.toGenericList().get();
  } else if (ivalue.isTuple()) {
    return ivalue.toTuple().get();
  } else if (ivalue.isString()) {
    return ivalue.toString().get();
  } else if (ivalue.isIntList()) {
    return ivalue.toIntList().get();
  }

  return nullptr;
}

void Pickler::pushInt(const IValue& ivalue) {
  auto n = ivalue.toInt();
  if (n >= std::numeric_limits<int8_t>::min() &&
      n <= std::numeric_limits<int8_t>::max()) {
    push<OpCode>(OpCode::BININT1);
    push<int8_t>(n);
  } else if (
      n >= std::numeric_limits<int32_t>::min() &&
      n <= std::numeric_limits<int32_t>::max()) {
    push<OpCode>(OpCode::BININT);
    push<int32_t>(n);
  } else {
    // Push 8 byte integer
    push<OpCode>(OpCode::LONG1);
    push<uint8_t>(8);
    push<int64_t>(n);
  }
}

void Pickler::pushBinGet(uint32_t memo_id) {
  if (memo_id <= std::numeric_limits<uint8_t>::max()) {
    push<OpCode>(OpCode::BINGET);
    push<uint8_t>(memo_id);
  } else {
    // Memoized too many items, issue a LONG_BINGET instead
    push<OpCode>(OpCode::LONG_BINGET);
    push<uint32_t>(memo_id);
  }
}

void Pickler::pushMemoizedString(const IValue& ivalue) {
  const auto& string = ivalue.toStringRef();

  push<OpCode>(OpCode::BINUNICODE);
  push<uint32_t>(string.size());
  pushString(string);
  pushMemoization(ivalue);
}

void Pickler::pushString(const std::string& string) {
  stack_.insert(stack_.end(), string.begin(), string.end());
}

void Pickler::pushClass(PicklerClass cls) {
  const auto& name = getClassName(cls);
  // Write it to the tensor table
  auto memo_entry = memo_map_.find(&name);
  if (memo_entry == memo_map_.end()) {
    push<OpCode>(OpCode::GLOBAL);
    // Module name + "\n"
    pushString(getModuleName());
    // Class name + "\n"
    pushString(name);
    pushMemoization((void*)&name);
  } else {
    pushBinGet(memo_entry->second);
  }
}

void Pickler::pushTensor(const IValue& ivalue) {
  pushClass(PicklerClass::TENSOR);

  tensor_table_->push_back(ivalue.toTensor());
  int64_t tensor_id = tensor_table_->size() - 1;
  // Reduce arguments are spread (e.g. `*args`) before calling the global,
  // so wrap in a tuple
  addIValue(c10::ivalue::Tuple::create({tensor_id}));

  push<OpCode>(OpCode::REDUCE);
}

void Pickler::pushIntList(const IValue& ivalue) {
  pushClass(PicklerClass::INTLIST);


  // Reduce arguments are spread (e.g. `*args`) before calling the global,
  // so wrap in a tuple
  push<OpCode>(OpCode::MARK);

  push<OpCode>(OpCode::EMPTY_LIST);
  // Mark list
  push<OpCode>(OpCode::MARK);

  // Add items
  for (const auto& item : ivalue.toIntListRef()) {
    addIValue(item);
  }

  // Finish list
  push<OpCode>(OpCode::APPENDS);

  // Finish tuple
  push<OpCode>(OpCode::TUPLE);

  // Call reduce
  push<OpCode>(OpCode::REDUCE);
  pushMemoization(ivalue);
}

void Pickler::pushDouble(const IValue& ivalue) {
  double value = ivalue.toDouble();
  AT_ASSERT(sizeof(double) == 8);
  char* bytes = reinterpret_cast<char*>(&value);

  push<OpCode>(OpCode::BINFLOAT);
  for (size_t i = 0; i < 8; ++i) {
    push<uint8_t>(bytes[8 - i - 1]);
  }
}

using ivalue_pair = std::pair<IValue, IValue>;

struct IValuePairComparator {
  bool operator()(const ivalue_pair& lhs, const ivalue_pair& rhs) const {
    if (lhs.first.isString()) {
      return lhs.first.toStringRef() < rhs.first.toStringRef();
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

void Pickler::pushDict(const IValue& ivalue) {
  auto dict = ivalue.toGenericDictRef();

  push<OpCode>(OpCode::EMPTY_DICT);
  pushMemoization(ivalue);

  push<OpCode>(OpCode::MARK);

  // Sort the dict for deterministic keys
  std::vector<std::pair<IValue, IValue>> dict_items(dict.begin(), dict.end());
  std::sort(dict_items.begin(), dict_items.end(), IValuePairComparator());

  for (const auto& pair : dict_items) {
    addIValue(pair.first);
    addIValue(pair.second);
  }

  push<OpCode>(OpCode::SETITEMS);
}

void Pickler::pushMemoization(const void* item) {
  AT_CHECK(item != nullptr, "Pickler cannot memoize a nullptr");
  if (memo_id <= std::numeric_limits<uint8_t>::max()) {
    push<OpCode>(OpCode::BINPUT);
    push<uint8_t>(memo_id);
  } else {
    // Memoized too many items, issue a LONG_BINPUT instead
    push<OpCode>(OpCode::LONG_BINPUT);
    push<uint32_t>(memo_id);
  }
  memo_map_[item] = memo_id;
  AT_ASSERT(memo_id <= std::numeric_limits<uint32_t>::max());
  ++memo_id;
}

void Pickler::pushMemoization(const IValue& ivalue) {
  auto ptr = getPointer(ivalue);
  AT_CHECK(
      ptr != nullptr,
      "Pickler cannot memoize ",
      ivalue.tagKind(),
      " IValue ",
      ivalue)
  pushMemoization(getPointer(ivalue));
}

void Pickler::pushList(const IValue& ivalue) {
  auto list = ivalue.toGenericListRef();
  push<OpCode>(OpCode::EMPTY_LIST);
  pushMemoization(ivalue);

  push<OpCode>(OpCode::MARK);

  for (const auto& item : list) {
    addIValue(item);
  }

  push<OpCode>(OpCode::APPENDS);
}

void Pickler::pushTuple(const IValue& ivalue) {
  // TODO: Small tuple unrolling (e.g. TUPLE3)
  push<OpCode>(OpCode::MARK);
  auto tuple = ivalue.toTuple()->elements();

  for (const auto& item : tuple) {
    addIValue(item);
  }

  push<OpCode>(OpCode::TUPLE);
  pushMemoization(ivalue);
}

std::vector<IValue> Unpickler::parse_ivalue_list() {
  run();
  AT_ASSERT(stack_.size() == 1);
  return stack_[0].toGenericListRef();
}

double Unpickler::readFloat() {
  AT_ASSERT(sizeof(double) == 8);
  AT_ASSERT(bytes_ + 8 < end_ptr_);
  double result;

  // Pickle floats are big endian, so reverse the bytes
  std::reverse_copy(
      reinterpret_cast<const char*>(bytes_),
      reinterpret_cast<const char*>(bytes_ + 8),
      reinterpret_cast<char*>(&result));

  bytes_ += 8;
  return result;
}

void Unpickler::run() {
  // Expect a PROTO opcode and protocol number at the start of blob
  AT_CHECK(
      readOpCode() == OpCode::PROTO,
      "Expected PROTO opcode at the start"
      " of pickle archive");
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
  }

  AT_ERROR("Overran buffer while unpickling data, didn't find STOP opcode");
}

OpCode Unpickler::readInstruction() {
  auto opcode = readOpCode();
  switch (opcode) {
    case OpCode::EMPTY_LIST: {
      if (globals_.size() > 0 && globals_.back() == PicklerClass::INTLIST) {
          // Check if we're in a GLOBAL opcode and if so, if it's a list
          // specialization
          stack_.emplace_back(std::vector<int64_t>());
      } else {
        stack_.emplace_back(std::vector<IValue>());
      }
    } break;
    case OpCode::EMPTY_TUPLE: {
      stack_.emplace_back(c10::ivalue::Tuple::create({}));
    } break;
    case OpCode::BINPUT: {
      size_t memo_id = read<uint8_t>();
      if (memo_table_.size() <= memo_id) {
        memo_table_.reserve(1 + 2 * memo_id);
      }
      memo_table_.push_back(stack_.back());
    } break;
    case OpCode::MARK: {
      // Mark location of the container ivalue in the stack
      marks_.push_back(stack_.size());
    } break;
    case OpCode::NEWTRUE: {
      stack_.emplace_back(true);
    } break;
    case OpCode::NEWFALSE: {
      stack_.emplace_back(false);
    } break;
    case OpCode::BININT1: {
      int8_t value = read<int8_t>();
      stack_.emplace_back(int64_t(value));
    } break;
    case OpCode::BININT: {
      int32_t value = read<int32_t>();
      stack_.emplace_back(int64_t(value));
    } break;
    case OpCode::LONG1: {
      // Only read LONG1s with 8 as the length
      uint8_t length = read<uint8_t>();
      AT_ASSERT(length == 8);
      stack_.emplace_back(int64_t(read<int64_t>()));
    } break;
    case OpCode::BINUNICODE: {
      uint32_t length = read<uint32_t>();
      const char* characters = reinterpret_cast<const char*>(bytes_);
      AT_ASSERT(bytes_ + length < end_ptr_);
      bytes_ += length;
      stack_.emplace_back(std::string(characters, /*n=*/length));
    } break;
    case OpCode::BINFLOAT:
      stack_.emplace_back(readFloat());
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
      stack_.emplace_back(c10::ivalue::UnorderedMap());
      break;
    case OpCode::APPENDS: {
      readList();
    } break;
    case OpCode::SETITEMS: {
      size_t start = marks_.back();
      marks_.pop_back();
      auto dict = stack_.at(start - 1).toGenericDict();
      for (size_t i = start; i < stack_.size(); i += 2) {
        dict->elements()[stack_[i]] = stack_[i + 1];
      }
      stack_.resize(start);
    } break;
    case OpCode::BINGET: {
      stack_.push_back(memo_table_.at(read<uint8_t>()));
    } break;
    case OpCode::STOP:
      break;
    case OpCode::GLOBAL: {
      // Module name, it's not needed for anything
      readString();
      // Push class name to stack
      globals_.emplace_back(getClass(readString()));
    } break;
    case OpCode::REDUCE: {
      // Pop reduce arg off the stack
      auto data = stack_.back().toTuple();
      stack_.pop_back();

      // Remove GLOBAL from stack
      auto class_name = globals_.back();
      globals_.pop_back();

      switch (class_name) {
        case PicklerClass::TENSOR:
          stack_.emplace_back(
              tensor_table_->at(data->elements().at(0).toInt()));
          break;
        case PicklerClass::INTLIST:
          stack_.push_back(data->elements().at(0).toIntListRef());
          break;
        default:
          AT_ERROR("Unknown pickler class id");
      }
    } break;
    default:
      AT_ERROR("Unknown opcode for unpickling: ", static_cast<uint8_t>(opcode));
  }
  return opcode;
}

void Unpickler::readList() {
  size_t start = marks_.back();
  marks_.pop_back();
  auto list_ivalue = stack_.at(start - 1);
  if (list_ivalue.isIntList()) {
    auto list = stack_.at(start - 1).toIntList();
    auto num_elements = stack_.size() - start;
    list->elements().reserve(num_elements);
    for (auto it = stack_.begin() + start; it != stack_.end(); ++it) {
      list->elements().emplace_back(it->toInt());
    }
  } else {
    auto list = stack_.at(start - 1).toGenericList();
    list->elements().insert(
        list->elements().end(), stack_.begin() + start, stack_.end());
  }
  stack_.resize(start);
}

inline bool is_valid_python_id_char(char c) {
  return c == '_' || c == '.' || (c >= '0' && c <= '9') ||
      (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

// Read a newline terminated string
std::string Unpickler::readString() {
  const char* chars = reinterpret_cast<const char*>(bytes_);
  const char* char_end_ptr = reinterpret_cast<const char*>(end_ptr_);
  size_t n = 0;
  while (true) {
    char c = chars[n];
    if (c == '\n') {
      break;
    }

    // Simple check just in case there is no terminating '\n'
    AT_CHECK(
        is_valid_python_id_char(c),
        "Found character '",
        uint8_t(c),
        "' in string, "
        "strings must be qualified Python identifiers");

    // Increment after to exclude newline from string
    ++n;
    AT_CHECK(
        chars + n < char_end_ptr,
        "Unpickler overran buffer while reading a string (expected a newline)");
  }

  // Increment by string length + newline char
  bytes_ += n + 1;
  return std::string(chars, n);
}

OpCode Unpickler::readOpCode() {
  return static_cast<OpCode>(read<uint8_t>());
}

} // namespace jit
} // namespace torch
