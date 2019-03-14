#include <torch/csrc/jit/pickler.h>

namespace torch {
namespace jit {

using ::c10::IValue;

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

const void* Pickler::getPointer(const IValue& ivalue) {
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

void Pickler::pushString(const IValue& ivalue) {
  const auto& string = ivalue.toStringRef();

  push<OpCode>(OpCode::BINUNICODE);
  push<uint32_t>(string.size());
  pushString(string);
  pushMemoization(ivalue);
}

void Pickler::pushString(const std::string& string) {
  stack_.insert(stack_.end(), string.begin(), string.end());
}

void Pickler::pushTensor(const IValue& ivalue) {
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

void Pickler::pushDouble(const IValue& ivalue) {
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
  std::multiset<std::pair<IValue, IValue>, IValueComparator> dict_items(
      dict.begin(), dict.end());

  for (const auto& pair : dict_items) {
    addIValue(pair.first);
    addIValue(pair.second);
  }

  push<OpCode>(OpCode::SETITEMS);
}

void Pickler::pushMemoization(const void* item) {
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

void Pickler::pushMemoization(const IValue& ivalue) {
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

const std::vector<IValue> Unpickler::get_ivalue_list() {
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
  }

  AT_ERROR("Overran buffer while unpickling data, didn't find STOP opcode");
}

OpCode Unpickler::readInstruction() {
  auto op = readOpCode();
  switch (op) {
    case OpCode::EMPTY_LIST: {
      // TODO: Use fake classes to mark list specializations
      stack_.emplace_back(std::vector<IValue>());
      break;
    }
    case OpCode::EMPTY_TUPLE: {
      stack_.emplace_back(c10::ivalue::Tuple::create({}));
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
      stack_.emplace_back(int64_t(value));
      break;
    }
    case OpCode::BINUNICODE: {
      int32_t length = read<uint32_t>();
      const char* characters = reinterpret_cast<const char*>(bytes_);
      bytes_ += length;
      stack_.emplace_back(std::string(characters, /*n=*/length));
      break;
    }
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
      stack_.emplace_back(readString());
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
      stack_.emplace_back(tensor_table_.at(tensor_id));

    } break;
    default:
      AT_ERROR("Unknown opcode for unpickling");
  }

  return op;
}

std::string Unpickler::readString(char terminator) {
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

bool Unpickler::isValidChar(char value) {
  return value >= 'A' && value <= 'z';
}

OpCode Unpickler::readOpCode() {
  return static_cast<OpCode>(read<uint8_t>());
}

} // namespace jit
} // namespace torch
