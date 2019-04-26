#include <torch/csrc/jit/pickler.h>
#include <ATen/ATen.h>
#include <string>

namespace torch {
namespace jit {

using ::c10::IValue;

const static std::unordered_map<std::string, PicklerClass> name_to_class{
  {"TensorID", PicklerClass::TENSOR},
  {"IntList", PicklerClass::INTLIST},
};

const static std::unordered_map<PicklerClass, std::string, std::hash<uint8_t>>
    class_to_name{
        {PicklerClass::TENSOR, "TensorID\n"},
        {PicklerClass::INTLIST, "IntList\n"},
    };

static std::string module_name = "__main__\n";

// Protocol 2 is the highest that can be decoded by Python 2
// See https://docs.python.org/3/library/pickle.html#data-stream-format
constexpr static uint8_t PROTOCOL_VERSION = 2;

const std::vector<char>& Pickler::stack() {
  return stack_;
}

void Pickler::start() {
  push<OpCode>(OpCode::PROTO);
  push<uint8_t>(PROTOCOL_VERSION);
}

void Pickler::pushTuple() {
  push<OpCode>(OpCode::MARK);
}

void Pickler::endTuple() {
  push<OpCode>(OpCode::TUPLE);
}

void Pickler::finish() {
  push<OpCode>(OpCode::STOP);

  // Add the binary data for all the tensors to be included in the same binary
  // TODO: The pickler should be refactored to stream out to a stream directly
  // instead of staging in the stack_ array
  if (literal_tensors_.size() > 0) {
    // As another pickle program in the same binary archive, add a list of
    // keys for each tensor (see torch/serialization.py)
    start();
    pushTuple();
    for (const auto& tensor : literal_tensors_) {
      std::string key = std::to_string(getTensorKey(tensor));
      push<OpCode>(OpCode::BINUNICODE);
      push<uint32_t>(key.size());
      pushString(key);
    }
    endTuple();
    push<OpCode>(OpCode::STOP);

    // Now dump the tensor binary data
    for (const auto& tensor : literal_tensors_) {
      pushTensorData(tensor);
    }
  }
}

void Pickler::pushTensorData(const at::Tensor& tensor) {
  // first dump size
  auto numel = tensor.numel();
  auto numel_ptr = reinterpret_cast<const char*>(&numel);
  stack_.insert(stack_.end(), numel_ptr, numel_ptr + sizeof(numel));

  uint64_t record_size;
  void* storage_ptr;
  std::tie(storage_ptr, record_size) = getWriteableTensor(tensor);
  auto storage_byte_ptr = reinterpret_cast<uint8_t*>(storage_ptr);
  stack_.insert(stack_.end(), storage_byte_ptr, storage_byte_ptr + record_size);
}

void Pickler::pushMetadata() {
  // Output data to match torch.save, see torch/serialization.py for details
  // Magic number (0x1950a86a20f9469cfc6c)
  start();
  push<OpCode>(OpCode::LONG1);
  // LONG1 size
  pushString("\x0a");
  // LONG1 data
  pushString("\x6c\xfc\x9c\x46\xf9\x20\x6a\xa8\x50\x19");
  push<OpCode>(OpCode::STOP);

  // Protocol Version (1001)
  start();
  push<OpCode>(OpCode::BININT2);
  pushString("\xe9\x03");
  push<OpCode>(OpCode::STOP);

  // sys_info, this isn't actually used in de-serialization so we can leave this
  // one empty
  start();
  push<OpCode>(OpCode::EMPTY_DICT);
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

void Pickler::pushGlobal(const std::string& name) {
  auto memo_entry = memo_map_.find(&name);
  if (memo_entry == memo_map_.end()) {
    push<OpCode>(OpCode::GLOBAL);
    pushString(name);
    pushMemoization((void*)&name);
  } else {
    pushBinGet(memo_entry->second);
  }
}

void Pickler::pushClass(PicklerClass cls) {
  pushGlobal(module_name + class_to_name.at(cls));

  push<OpCode>(OpCode::EMPTY_TUPLE);
  push<OpCode>(OpCode::NEWOBJ);
}

void Pickler::pushTensor(const IValue& ivalue) {
  if (tensor_table_ == nullptr) {
    pushLiteralTensor(ivalue);
  } else {
    pushTensorReference(ivalue);
  }
}

void Pickler::pushLiteralTensor(const IValue& ivalue) {
  // In contrast to tensor references, literal tensors are included in the
  // pickle program binary blob. They are written to the file after the STOP
  // opcode. They can't be included in the pickle program itself without a bunch
  // of extra machinery since byte strings are limited to 4 GB.
  //
  // The format here is the same one used by `torch.save()`. The code for the
  // format can be found in `torch/serialization.py`.
  auto tensor = ivalue.toTensor();

  // The arguments to this function are:
  //    storage, storage_offset, size, stride, requires_grad, backward_hooks
  pushGlobal("torch._utils\n_rebuild_tensor_v2\n");
  push<OpCode>(OpCode::MARK);

  // Tuple for persistent_load
  push<OpCode>(OpCode::MARK);
  // typename
  pushMemoizedString(std::string("storage"));
  // data_type
  std::stringstream data_type;
  data_type << "torch\n" << toString(tensor.scalar_type()) << "Storage\n";
  pushGlobal(data_type.str());
  // root_key
  pushMemoizedString(std::to_string(getTensorKey(tensor)));
  // location
  pushMemoizedString(std::string("cpu"));
  // size
  pushInt(tensor.numel());
  // view_metadata
  push<OpCode>(OpCode::NONE);
  push<OpCode>(OpCode::TUPLE);


  push<OpCode>(OpCode::BINPERSID);

  // storage offset
  int64_t storage_offset = 0;
  pushInt(storage_offset);

  // size
  std::vector<IValue> size_ivalues;
  for (auto size : tensor.sizes()) {
    size_ivalues.emplace_back(size);
  }
  auto sizes = c10::ivalue::Tuple::create(size_ivalues);
  addIValue(sizes);

  // stride
  std::vector<IValue> stride_ivalues;
  for (auto stride : tensor.strides()) {
    stride_ivalues.emplace_back(stride);
  }
  auto strides = c10::ivalue::Tuple::create(stride_ivalues);
  addIValue(strides);

  // requires_grad
  addIValue(tensor.requires_grad());

  // backward_hooks
  pushGlobal("collections\nOrderedDict\n");
  push<OpCode>(OpCode::EMPTY_TUPLE);
  // Construct the collections.OrderedDict for the backward_hooks
  push<OpCode>(OpCode::REDUCE);

  push<OpCode>(OpCode::TUPLE);

  // Call torch._utils._rebuild_tensor_v2
  push<OpCode>(OpCode::REDUCE);

  // Store tensor so it can be placed into the binary after the pickle program
  literal_tensors_.push_back(ivalue.toTensor());
}


void Pickler::pushTensorReference(const IValue& ivalue) {
  pushClass(PicklerClass::TENSOR);
  tensor_table_->push_back(ivalue.toTensor());
  auto tensor_id = tensor_table_->size() - 1;
  push<OpCode>(OpCode::BININT);
  push<uint32_t>(tensor_id);

  push<OpCode>(OpCode::BUILD);
}

void Pickler::pushIntList(const IValue& ivalue) {
  pushClass(PicklerClass::INTLIST);

  push<OpCode>(OpCode::EMPTY_LIST);
  pushMemoization(ivalue);
  push<OpCode>(OpCode::MARK);

  for (const auto& item : ivalue.toIntListRef()) {
    addIValue(item);
  }

  push<OpCode>(OpCode::APPENDS);
  push<OpCode>(OpCode::BUILD);
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

void Pickler::pushDict(const IValue& ivalue) {
  push<OpCode>(OpCode::EMPTY_DICT);
  pushMemoization(ivalue);

  push<OpCode>(OpCode::MARK);

  // Sort the dict for deterministic keys
  auto dict_items = ivalue.toGenericDict()->iterationOrder();
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
  memo_map_[item] = memo_id;
  AT_ASSERT(memo_id <= std::numeric_limits<uint32_t>::max());
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

std::vector<IValue> Unpickler::parse_ivalue_list() {
  run();
  AT_CHECK(
      stack_.size() == 1,
      "Unpickler expected 1 element on the stack, but found ",
      stack_.size());
  return stack_.at(0).toTuple()->elements();
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
        AT_CHECK(
            stack_.size() > 0,
            "Unpickler found an empty stack when it expected a value");
        auto value = stack_.back().toInt();
        AT_CHECK(
            value >= 0 && value <= std::numeric_limits<uint8_t>::max(),
            "Unpickler could not decode PicklerClass for ",
            value);
        PicklerClass cls = static_cast<PicklerClass>(uint8_t(value));
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
      if (memo_table_.size() <= memo_id) {
        memo_table_.reserve(1 + 2 * memo_id);
      }
      memo_table_.push_back(stack_.back());
    } break;
    case OpCode::LONG_BINPUT: {
      AT_CHECK(
          std::numeric_limits<size_t>::max() >=
              std::numeric_limits<uint32_t>::max(),
          "Found a LONG_BINPUT opcode, but size_t on this system is "
          "not big enough to decode it");
      size_t memo_id = read<uint32_t>();
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
      auto module = readString();
      auto attr = readString();
      // Push class name to stack
      stack_.emplace_back(static_cast<uint8_t>(name_to_class.at(attr)));
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
        c == '_' || (c >= '0' && c <= 'z'),
        "Found character '", c, "' in string, "
        "strings must be Python identifiers");

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

std::pair<void*, uint64_t> getWriteableTensor(const at::Tensor& tensor) {
  at::Tensor storage_tensor = tensor;
  uint64_t record_size = tensor.element_size() * tensor.storage().size();
  // TODO HIP support
  if (tensor.storage().device_type() == at::DeviceType::CUDA) {
    // NB: This new tensor is created to support cuda tensors.
    // Storages can be mutated when converting tensors from cuda to cpu,
    // and we need a cpu tensor to copy data from.
    storage_tensor = at::empty({0}, tensor.options())
                         .set_(
                             tensor.storage(),
                             /* storage_offset = */ 0,
                             /* size = */
                             {static_cast<int64_t>(tensor.storage().size())},
                             /* stride = */ {1})
                         .cpu();
    AT_CHECK(
        storage_tensor.element_size() * storage_tensor.storage().size() ==
            record_size,
        "Storage tensor size did not match record size");
  }

  return std::make_pair(tensor.storage().data(), record_size);
}

uint64_t getTensorKey(const at::Tensor& tensor) {
  at::StorageImpl* storage_key = tensor.storage().unsafeGetStorageImpl();
  return reinterpret_cast<intptr_t>(storage_key);
}

} // namespace jit
} // namespace torch
