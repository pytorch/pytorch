#include <torch/csrc/jit/pickler.h>

namespace torch {
namespace jit {

using ::c10::IValue;

PicklerClass getClass(const std::string& str) {
  if (str == "TensorID") {
    return PicklerClass::TENSOR;
  } else if (str == "IntList") {
    return PicklerClass::INTLIST;
  }
  AT_ERROR("Unknown class name for unpickler: ", str);
}

const std::string& getClassName(PicklerClass cls) {
  static const std::string tensor_class("TensorID\n");
  static const std::string intlist_class("IntList\n");
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
  static const std::string module_name("__main__\n");
  return module_name;
}

const std::vector<char>& Pickler::stack() {
  return stack_;
}

void Pickler::start() {
  pushOpCode(OpCode::PROTO);
  pushUint8(2);

  // All attributes get pushed into a list and their indices saved in the
  // module def
  pushOpCode(OpCode::EMPTY_LIST);
  pushOpCode(OpCode::MARK);
}

void Pickler::finish() {
  pushOpCode(OpCode::APPENDS);
  pushOpCode(OpCode::STOP);
}

void Pickler::addIValue(const IValue& ivalue) {
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
    AT_ASSERT(
        ivalue.toInt() <= std::numeric_limits<int32_t>::max() &&
        ivalue.toInt() >= std::numeric_limits<int32_t>::min());
    pushOpCode(OpCode::BININT);
    pushInt32(ivalue.toInt());
  } else if (ivalue.isBool()) {
    if (ivalue.toBool()) {
      pushOpCode(OpCode::NEWTRUE);
    } else {
      pushOpCode(OpCode::NEWFALSE);
    }
  } else if (ivalue.isString()) {
    pushMemoizedString(ivalue);
  } else if (ivalue.isGenericList()) {
    pushList(ivalue);
  } else if (ivalue.isGenericDict()) {
    pushDict(ivalue);
  } else if (ivalue.isNone()) {
    pushOpCode(OpCode::NONE);
  } else if (ivalue.isIntList()) {
    pushIntList(ivalue);
  } else {
    AT_ERROR("Unknown IValue type for pickling: ", ivalue.tagKind());
  }
}

const void* Pickler::getPointer(const IValue& ivalue) {
  if (ivalue.isGenericDict()) {
    return &(ivalue.toGenericDictRef());
  } else if (ivalue.isGenericList()) {
    return &(ivalue.toGenericListRef());
  } else if (ivalue.isTuple()) {
    return &(ivalue.toTuple()->elements());
  } else if (ivalue.isString()) {
    return &(ivalue.toStringRef());
  } else if (ivalue.isIntList()) {
    return &(ivalue.toIntListRef());
  }

  return nullptr;
}

void Pickler::pushBinGet(uint32_t memo_id) {
  if (memo_id <= std::numeric_limits<uint8_t>::max()) {
    pushOpCode(OpCode::BINGET);
    pushUint8(memo_id);
  } else {
    // Memoized too many items, issue a LONG_BINGET instead
    pushOpCode(OpCode::LONG_BINGET);
    pushUint32(memo_id);
  }
}

void Pickler::pushMemoizedString(const IValue& ivalue) {
  const auto& string = ivalue.toStringRef();

  pushOpCode(OpCode::BINUNICODE);
  pushUint32(string.size());
  pushString(string);
  pushMemoization(ivalue);
}

void Pickler::pushString(const std::string& string) {
  stack_.insert(stack_.end(), string.begin(), string.end());
}

void Pickler::pushClass(PicklerClass cls) {
  const auto& name = getClassName(cls);
  // Write it to the tensor table
  auto memo_entry = memo_.find(&name);
  if (memo_entry == memo_.end()) {
    pushOpCode(OpCode::GLOBAL);
    // Module name + "\n"
    pushString(getModuleName());
    // Class name + "\n"
    pushString(name);
    pushMemoization((void*)&name);
  } else {
    pushBinGet(memo_entry->second);
  }

  pushOpCode(OpCode::EMPTY_TUPLE);
  pushOpCode(OpCode::NEWOBJ);
}

void Pickler::pushTensor(const IValue& ivalue) {
  pushClass(PicklerClass::TENSOR);

  tensor_table_->push_back(ivalue.toTensor());
  auto tensor_id = tensor_table_->size() - 1;
  pushOpCode(OpCode::BININT);
  pushUint32(tensor_id);

  pushOpCode(OpCode::BUILD);
}

void Pickler::pushIntList(const IValue& ivalue) {
  pushClass(PicklerClass::INTLIST);

  pushOpCode(OpCode::EMPTY_LIST);
  pushMemoization(ivalue);
  pushOpCode(OpCode::MARK);

  for (const auto& item : ivalue.toIntListRef()) {
    addIValue(item);
  }

  pushOpCode(OpCode::APPENDS);
  pushOpCode(OpCode::BUILD);
}

void Pickler::pushDouble(const IValue& ivalue) {
  double value = ivalue.toDouble();
  AT_ASSERT(sizeof(double) == 8);
  char* bytes = reinterpret_cast<char*>(&value);

  pushOpCode(OpCode::BINFLOAT);
  for (size_t i = 0; i < 8; ++i) {
    pushUint8(bytes[8 - i - 1]);
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

  pushOpCode(OpCode::EMPTY_DICT);
  pushMemoization(ivalue);

  pushOpCode(OpCode::MARK);

  // Sort the dict for deterministic keys
  std::vector<std::pair<IValue, IValue>> dict_items(dict.begin(), dict.end());
  std::sort(dict_items.begin(), dict_items.end(), IValuePairComparator());

  for (const auto& pair : dict_items) {
    addIValue(pair.first);
    addIValue(pair.second);
  }

  pushOpCode(OpCode::SETITEMS);
}

void Pickler::pushMemoization(const void* item) {
  AT_ASSERT(item != nullptr);
  if (memo_id <= std::numeric_limits<uint8_t>::max()) {
    pushOpCode(OpCode::BINPUT);
    pushUint8(memo_id);
  } else {
    // Memoized too many items, issue a LONG_BINPUT instead
    pushOpCode(OpCode::LONG_BINPUT);
    pushUint32(memo_id);
  }
  memo_[item] = memo_id;
  AT_ASSERT(memo_id <= std::numeric_limits<uint32_t>::max());
  ++memo_id;
}

void Pickler::pushMemoization(const IValue& ivalue) {
  pushMemoization(getPointer(ivalue));
}

void Pickler::pushList(const IValue& ivalue) {
  auto list = ivalue.toGenericListRef();
  pushOpCode(OpCode::EMPTY_LIST);
  pushMemoization(ivalue);

  pushOpCode(OpCode::MARK);

  for (const auto& item : list) {
    addIValue(item);
  }

  pushOpCode(OpCode::APPENDS);
}

void Pickler::pushTuple(const IValue& ivalue) {
  // TODO: Small tuple unrolling (e.g. TUPLE3)
  pushOpCode(OpCode::MARK);
  auto tuple = ivalue.toTuple()->elements();

  for (const auto& item : tuple) {
    addIValue(item);
  }

  pushOpCode(OpCode::TUPLE);
  pushMemoization(ivalue);
}

void Pickler::pushUint8(uint8_t value) {
  const char* begin = reinterpret_cast<const char*>(&value);
  stack_.insert(stack_.end(), begin, begin + sizeof(uint8_t));
}

void Pickler::pushOpCode(OpCode value) {
  const char* begin = reinterpret_cast<const char*>(&value);
  stack_.insert(stack_.end(), begin, begin + sizeof(OpCode));
}

void Pickler::pushUint32(uint32_t value) {
  const char* begin = reinterpret_cast<const char*>(&value);
  stack_.insert(stack_.end(), begin, begin + sizeof(uint32_t));
}

void Pickler::pushInt32(int32_t value) {
  const char* begin = reinterpret_cast<const char*>(&value);
  stack_.insert(stack_.end(), begin, begin + sizeof(int32_t));
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

OpCode Unpickler::readInstruction() {
  auto opcode = readOpCode();
  switch (opcode) {
    case OpCode::EMPTY_LIST: {
      // Look back to see if the last opcode was an IntList class
      if (last_opcode_ == OpCode::NEWOBJ) {
        // It's a list specialization, the enum ID of which is on the stack
        PicklerClass cls =
            static_cast<PicklerClass>(uint8_t(stack_.back().toInt()));
        if (cls == PicklerClass::INTLIST) {
          stack_.emplace_back(std::vector<int64_t>());
        }
      } else {
        stack_.emplace_back(std::vector<IValue>());
      }
    } break;
    case OpCode::EMPTY_TUPLE: {
      stack_.emplace_back(c10::ivalue::Tuple::create({}));
    } break;
    case OpCode::BINPUT: {
      size_t memo_id = read<uint8_t>();
      if (memo_.size() <= memo_id) {
        memo_.reserve(1 + 2 * memo_id);
      }
      memo_.push_back(stack_.back());
    } break;
    case OpCode::MARK: {
      // Mark location of the container ivalue in the stack
      marks_.push_back(stack_.size());
    } break;
    case OpCode::BININT: {
      int32_t value = read<int32_t>();
      stack_.emplace_back(int64_t(value));
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
      stack_.push_back(memo_.at(read<uint8_t>()));
    } break;
    case OpCode::STOP:
      break;
    case OpCode::GLOBAL: {
      AT_ASSERT(readString() == "__main__");
      // Push class name to stack
      stack_.emplace_back(static_cast<uint8_t>(getClass(readString())));
    } break;
    case OpCode::NEWOBJ: {
      // pop empty tuple
      stack_.pop_back();
    } break;
    case OpCode::BUILD: {
      auto setitem_data = stack_.back();
      stack_.pop_back();

      auto class_name =
          static_cast<PicklerClass>(uint8_t(stack_.back().toInt()));
      stack_.pop_back();

      switch (class_name) {
        case PicklerClass::TENSOR:
          stack_.emplace_back(tensor_table_->at(setitem_data.toInt()));
          break;
        case PicklerClass::INTLIST:
          stack_.push_back(setitem_data);
          break;
        default:
          AT_ERROR("Unknown pickler class id");
      }
    } break;
    default:
      AT_ERROR("Unknown opcode for unpickling");
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

// Read a newline terminated string
std::string Unpickler::readString() {
  const char* chars = reinterpret_cast<const char*>(bytes_);
  size_t n = 0;
  while (true) {
    char c = chars[n];
    if (c == '\n') {
      break;
    }

    // Simple check just in case there is no terminating '\n'
    AT_ASSERT(c >= '0' && c <= 'z');

    // Increment after to exclude newline from string
    ++n;
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
