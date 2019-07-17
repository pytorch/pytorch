#include <torch/csrc/jit/pickler.h>
#include <ATen/ATen.h>
#include <string>
#include <ATen/core/Dict.h>

namespace torch {
namespace jit {

using ::c10::IValue;

// Protocol 2 is the highest that can be decoded by Python 2
// See https://docs.python.org/3/library/pickle.html#data-stream-format
constexpr static uint8_t PROTOCOL_VERSION = 2;

PicklerClass getClass(const std::string& str) {
  if (str == "build_tensor_from_id") {
    return PicklerClass::TENSOR;
  } else if (str == "build_intlist") {
    return PicklerClass::INTLIST;
  } else if (str == "build_tensorlist") {
    return PicklerClass::TENSORLIST;
  } else if (str == "build_doublelist") {
    return PicklerClass::DOUBLELIST;
  } else if (str == "build_boollist") {
    return PicklerClass::BOOLLIST;
  }

  // TODO [unpickler refactor]
  if (str == "TensorID") {
    return PicklerClass::TENSOR;
  } else if (str == "IntList") {
    return PicklerClass::INTLIST;
  }
  AT_ERROR("Unknown class name for unpickler: ", str);
}

const char* getClassName(PicklerClass cls) {
  switch (cls) {
    case PicklerClass::TENSOR:
      return "build_tensor_from_id";
    case PicklerClass::INTLIST:
      return "build_intlist";
    case PicklerClass::TENSORLIST:
      return "build_tensorlist";
    case PicklerClass::DOUBLELIST:
      return "build_doublelist";
    case PicklerClass::BOOLLIST:
      return "build_boollist";
    default:
      AT_ERROR("Unknown class for pickler");
  }
}

const std::vector<char>& Pickler::stack() {
  return stack_;
}

void Pickler::start() {
  push<OpCode>(OpCode::PROTO);
  push<uint8_t>(PROTOCOL_VERSION);
}

void Pickler::startTuple() {
  // All attributes get pushed into a tuple and their indices saved in the
  // module def
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
    push<OpCode>(OpCode::MARK);
    for (const auto& tensor : literal_tensors_) {
      std::string key = std::to_string(getStorageKey(tensor));
      push<OpCode>(OpCode::BINUNICODE);
      push<uint32_t>(key.size());
      pushBytes(key);
    }
    push<OpCode>(OpCode::TUPLE);
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
  at::Tensor storage_tensor;
  std::tie(storage_tensor, record_size) = getWriteableTensor(tensor);
  auto storage_byte_ptr = reinterpret_cast<uint8_t*>(storage_tensor.storage().data());
  stack_.insert(stack_.end(), storage_byte_ptr, storage_byte_ptr + record_size);
}

void Pickler::pushMetadata() {
  // Output data to match torch.save, see torch/serialization.py for details
  // Magic number (0x1950a86a20f9469cfc6c)
  start();
  push<OpCode>(OpCode::LONG1);
  // LONG1 size
  pushBytes("\x0a");
  // LONG1 data
  pushBytes("\x6c\xfc\x9c\x46\xf9\x20\x6a\xa8\x50\x19");
  push<OpCode>(OpCode::STOP);

  // Protocol Version (1001)
  start();
  push<OpCode>(OpCode::BININT2);
  pushBytes("\xe9\x03");
  push<OpCode>(OpCode::STOP);

  // sys_info, this isn't actually used in de-serialization so we can leave this
  // one empty
  start();
  push<OpCode>(OpCode::EMPTY_DICT);
  push<OpCode>(OpCode::STOP);
}

// unmemoized version called by addIValue
void Pickler::addIValueImpl(const IValue& ivalue) {
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
    pushStringImpl(ivalue.toStringRef());
  } else if (ivalue.isGenericList()) {
    pushGenericList(ivalue);
  } else if (ivalue.isGenericDict()) {
    pushDict(ivalue);
  } else if (ivalue.isNone()) {
    push<OpCode>(OpCode::NONE);
  } else if (ivalue.isIntList()) {
    pushSpecializedList(
        ivalue, PicklerClass::INTLIST, [=](const IValue& ivalue) {
          for (const int64_t item : ivalue.toIntListRef()) {
            addIValue(item);
          }
        });
  } else if (ivalue.isTensorList()) {
    pushSpecializedList(
        ivalue, PicklerClass::TENSORLIST, [=](const IValue& ivalue) {
          for (const at::Tensor& item : ivalue.toTensorListRef()) {
            addIValue(item);
          }
        });
  } else if (ivalue.isDoubleList()) {
    pushSpecializedList(
        ivalue, PicklerClass::DOUBLELIST, [=](const IValue& ivalue) {
          for (double item : ivalue.toDoubleListRef()) {
            addIValue(item);
          }
        });
  } else if (ivalue.isBoolList()) {
    pushSpecializedList(
        ivalue, PicklerClass::BOOLLIST, [=](const IValue& ivalue) {
          for (bool item : ivalue.toBoolList()) {
            addIValue(item);
          }
        });
  } else if (ivalue.isObject()) {
    auto obj = ivalue.toObject();
    auto type = obj->type();
    pushGlobal(type->qualifier(), type->basename());
    push<OpCode>(OpCode::EMPTY_TUPLE);
    push<OpCode>(OpCode::NEWOBJ);
    push<OpCode>(OpCode::EMPTY_DICT);
    push<OpCode>(OpCode::MARK);
    for (size_t i = 0, n = type->numAttributes(); i < n; ++i) {
      pushString(type->getAttributeName(i));
      addIValue(obj->getSlot(i));
    }
    push<OpCode>(OpCode::SETITEMS);
    push<OpCode>(OpCode::BUILD);
  } else {
    AT_ERROR("Unknown IValue type for pickling: ", ivalue.tagKind());
  }
}

void Pickler::addIValue(const IValue& ivalue) {
  // Check if reference ivalue has been saved before
  if (ivalue.isPtrType()) {
    const void* ptr = ivalue.internalToPointer();
    TORCH_CHECK(
        ptr != nullptr,
        "Pickler cannot memoize ",
        ivalue.tagKind(),
        " IValue ",
        ivalue);
    auto memo_entry = memoized_ivalue_map_.find(ptr);
    if (memo_entry != memoized_ivalue_map_.end()) {
      // This value has already been pushed, just do a BINGET
      pushBinGet(memo_entry->second);
      return;
    }
  }
  addIValueImpl(ivalue);
  if (ivalue.isPtrType()) {
    memoized_ivalues_.push_back(ivalue);
    memoized_ivalue_map_[ivalue.internalToPointer()] = pushNextBinPut();
  }
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

// unmemoized encoding of a string
void Pickler::pushStringImpl(const std::string& string) {
  push<OpCode>(OpCode::BINUNICODE);
  push<uint32_t>(string.size());
  pushBytes(string);
}

void Pickler::pushString(const std::string& string) {
  auto it = memoized_strings_map_.find(string);
  if (it == memoized_strings_map_.end()) {
    pushStringImpl(string);
    memoized_strings_map_[string] = pushNextBinPut();
  } else {
    pushBinGet(it->second);
  }
}

void Pickler::pushBytes(const std::string& string) {
  stack_.insert(stack_.end(), string.begin(), string.end());
}

void Pickler::pushGlobal(
    const std::string& module_name,
    const std::string& class_name) {
  std::stringstream ss;
  ss << module_name << "\n" << class_name << "\n";
  std::string key = ss.str();
  auto memo_entry = memoized_globals_map_.find(key);
  if (memo_entry == memoized_globals_map_.end()) {
    push<OpCode>(OpCode::GLOBAL);
    pushBytes(key);
    // Push BINPUT without adding anything to the memoized_ivalues_
    size_t memo_id = pushNextBinPut();
    memoized_globals_map_.insert({key, memo_id});
  } else {
    pushBinGet(memo_entry->second);
  }
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
  pushGlobal("torch._utils", "_rebuild_tensor_v2");
  push<OpCode>(OpCode::MARK);

  // Tuple for persistent_load
  push<OpCode>(OpCode::MARK);
  // typename
  pushString("storage");
  // data_type
  std::stringstream data_type;
  data_type << toString(tensor.scalar_type()) << "Storage";
  pushGlobal("torch", data_type.str());
  // root_key
  pushString(std::to_string(getStorageKey(tensor)));
  // location
  pushString("cpu");
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
  push<OpCode>(OpCode::MARK);
  for (auto size : tensor.sizes()) {
    pushInt(size);
  }
  push<OpCode>(OpCode::TUPLE);

  // stride
  push<OpCode>(OpCode::MARK);
  for (auto stride : tensor.strides()) {
    pushInt(stride);
  }
  push<OpCode>(OpCode::TUPLE);

  // requires_grad
  addIValue(tensor.requires_grad());

  // backward_hooks
  pushGlobal("collections", "OrderedDict");
  push<OpCode>(OpCode::EMPTY_TUPLE);
  // Construct the collections.OrderedDict for the backward_hooks
  push<OpCode>(OpCode::REDUCE);

  push<OpCode>(OpCode::TUPLE);

  // Call torch._utils._rebuild_tensor_v2
  push<OpCode>(OpCode::REDUCE);

  // Store tensor so it can be placed into the binary after the pickle program
  literal_tensors_.push_back(ivalue.toTensor());
}

void Pickler::pushClass(PicklerClass cls) {
  pushGlobal("torch.jit._pickle", getClassName(cls));
}

void Pickler::pushTensorReference(const IValue& ivalue) {
  pushClass(PicklerClass::TENSOR);
  tensor_table_->push_back(ivalue.toTensor());
  int64_t tensor_id = tensor_table_->size() - 1;
  // Reduce arguments are spread (e.g. `*args`) before calling the global,
  // so wrap in a tuple
  push<OpCode>(OpCode::MARK);
  addIValue(tensor_id);
  push<OpCode>(OpCode::TUPLE);

  push<OpCode>(OpCode::REDUCE);
}

void Pickler::pushSpecializedList(
    const IValue& ivalue,
    PicklerClass cls,
    const std::function<void(const IValue&)>& item_pusher) {
  pushClass(cls);

  // Reduce arguments are spread (e.g. `*args`) before calling the global,
  // so wrap in a tuple
  push<OpCode>(OpCode::MARK);

  push<OpCode>(OpCode::EMPTY_LIST);
  // Mark list
  push<OpCode>(OpCode::MARK);

  // Add all items
  item_pusher(ivalue);

  // Finish list
  push<OpCode>(OpCode::APPENDS);

  // Finish tuple
  push<OpCode>(OpCode::TUPLE);

  // Call reduce
  push<OpCode>(OpCode::REDUCE);
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

  push<OpCode>(OpCode::MARK);

  // Sort the dict for deterministic keys
  auto dict_items = iterationOrder(ivalue.toGenericDict());
  for (const auto& pair : dict_items) {
    addIValue(pair.first);
    addIValue(pair.second);
  }

  push<OpCode>(OpCode::SETITEMS);
}

size_t Pickler::pushNextBinPut() {
  if (memo_id_ <= std::numeric_limits<uint8_t>::max()) {
    push<OpCode>(OpCode::BINPUT);
    push<uint8_t>(memo_id_);
  } else {
    // Memoized too many items, issue a LONG_BINPUT instead
    push<OpCode>(OpCode::LONG_BINPUT);
    push<uint32_t>(memo_id_);
  }
  AT_ASSERT(memo_id_ <= std::numeric_limits<uint32_t>::max());
  ++memo_id_;
  return memo_id_ - 1;
}

void Pickler::pushGenericList(const IValue& ivalue) {
  auto list = ivalue.toGenericListRef();
  push<OpCode>(OpCode::EMPTY_LIST);

  push<OpCode>(OpCode::MARK);

  for (const IValue& item : list) {
    addIValue(item);
  }

  push<OpCode>(OpCode::APPENDS);
}

void Pickler::pushTuple(const IValue& ivalue) {
  // TODO: Small tuple unrolling (e.g. TUPLE3)
  push<OpCode>(OpCode::MARK);
  auto tuple = ivalue.toTuple();

  for (const IValue& item : tuple->elements()) {
    addIValue(item);
  }

  push<OpCode>(OpCode::TUPLE);
}

std::vector<IValue> Unpickler::parse_ivalue_list() {
  run();
  TORCH_CHECK(
      stack_.size() == 1,
      "Unpickler expected 1 element on the stack, but found ",
      stack_.size());

  auto value = stack_[0];
  if (value.isGenericList()) {
    // TODO [unpickler refactor]
    return value.toGenericListRef().vec();
  }
  return value.toTuple()->elements();
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
  TORCH_CHECK(
      readOpCode() == OpCode::PROTO,
      "Expected PROTO opcode at the start"
      " of pickle archive");
  uint8_t protocol = read<uint8_t>();
  TORCH_CHECK(
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
void Unpickler::setInput(size_t memo_id) {
  AT_ASSERT(!stack_.empty());
  if (memo_id >= memo_table_.size()) {
    memo_table_.insert(
        memo_table_.end(), memo_id - memo_table_.size(), IValue());
    memo_table_.push_back(stack_.back());
  } else {
    memo_table_[memo_id] = stack_.back();
  }
}

// emplace_back on bool vectors does not exist on some systems
// avoid it by calling push_back for bool
template <typename T>
inline void append(std::vector<T>& a, T&& e) {
  a.emplace_back(std::move(e));
}
template <>
inline void append<bool>(std::vector<bool>& a, bool&& e) {
  a.push_back(e);
}

template <typename T>
static IValue toSpecializedList(const IValue& generic) {
  auto ivalues = generic.toGenericListRef();
  std::vector<T> specialized;
  specialized.reserve(ivalues.size());
  for (const IValue& iv : ivalues) {
    append(specialized, iv.to<T>());
  }
  return IValue(std::move(specialized));
}

OpCode Unpickler::readInstruction() {
  auto opcode = readOpCode();
  switch (opcode) {
    case OpCode::EMPTY_LIST: {
      stack_.emplace_back(
          c10::impl::GenericList(c10::impl::deprecatedUntypedList()));
    } break;
    case OpCode::EMPTY_TUPLE: {
      if (empty_tuple_.isNone()) {
        // we only need one object, since tuples are not mutable.
        empty_tuple_ = c10::ivalue::Tuple::create({});
      }
      stack_.emplace_back(empty_tuple_);
    } break;
    case OpCode::BINPUT: {
      size_t memo_id = read<uint8_t>();
      setInput(memo_id);
    } break;
    case OpCode::LONG_BINPUT: {
      TORCH_CHECK(
          std::numeric_limits<size_t>::max() >=
              std::numeric_limits<uint32_t>::max(),
          "Found a LONG_BINPUT opcode, but size_t on this system is "
          "not big enough to decode it");
      size_t memo_id = read<uint32_t>();
      setInput(memo_id);
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
    case OpCode::NONE: {
      stack_.emplace_back(IValue());
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
      auto tuple = c10::ivalue::Tuple::create({});
      tuple->elements().reserve(stack_.size() - start);
      auto start_it = stack_.begin() + start;
      for (auto it = start_it; it != stack_.end(); ++it) {
        tuple->elements().emplace_back(*it);
      }
      stack_.erase(start_it, stack_.end());
      stack_.emplace_back(tuple);
    } break;
    case OpCode::EMPTY_DICT:
      stack_.emplace_back(c10::impl::GenericDict(c10::impl::deprecatedUntypedDict()));
      break;
    case OpCode::APPENDS: {
      readList();
    } break;
    case OpCode::SETITEMS: {
      size_t start = marks_.back();
      marks_.pop_back();
      auto dict = stack_.at(start - 1).toGenericDict();
      for (size_t i = start; i < stack_.size(); i += 2) {
        dict.insert_or_assign(stack_[i], stack_[i + 1]);
      }
      stack_.erase(stack_.begin() + start, stack_.end());
    } break;
    case OpCode::BINGET: {
      stack_.push_back(memo_table_.at(read<uint8_t>()));
    } break;
    case OpCode::LONG_BINGET: {
      stack_.push_back(memo_table_.at(read<uint32_t>()));
    } break;
    case OpCode::STOP:
      break;
    case OpCode::GLOBAL: {
      // Module name, it's not needed for anything
      auto module_name = readString();
      auto class_name = readString();
      // TODO [unpickler refactor] __main__ isn't used by the pickler anymore
      if (module_name == "__main__") {
        auto pickler_class = getClass(class_name);
        globals_.emplace_back([this, pickler_class] {
          // TODO: [unpickler refactor]
          auto setitem_data = stack_.back();
          stack_.pop_back();
          switch (pickler_class) {
            case PicklerClass::TENSOR:
              stack_.emplace_back(tensor_table_->at(setitem_data.toInt()));
              break;
            case PicklerClass::INTLIST:
              stack_.emplace_back(toSpecializedList<int64_t>(setitem_data));
              break;
            default:
              AT_ERROR("Unknown pickler class id", pickler_class);
          }
        });
      } else if (module_name == "torch.jit._pickle") {
        auto pickler_class = getClass(class_name);
        globals_.emplace_back([this, pickler_class] {
          // Pop reduce arg off the stack
          auto data = stack_.back().toTuple()->elements().at(0);
          stack_.pop_back();
          switch (pickler_class) {
            case PicklerClass::TENSOR:
              stack_.emplace_back(tensor_table_->at(data.toInt()));
              break;
            case PicklerClass::INTLIST:
              stack_.emplace_back(toSpecializedList<int64_t>(data));
              break;
            case PicklerClass::TENSORLIST:
              stack_.emplace_back(toSpecializedList<at::Tensor>(data));
              break;
            case PicklerClass::DOUBLELIST:
              stack_.emplace_back(toSpecializedList<double>(data));
              break;
            case PicklerClass::BOOLLIST:
              stack_.emplace_back(toSpecializedList<bool>(data));
              break;
            default:
              AT_ERROR("Unknown pickler class id");
          }
        });
      } else {
        AT_ASSERT(class_resolver_);
        at::StrongTypePtr type =
            class_resolver_(c10::QualifiedName(module_name, class_name));
        globals_.emplace_back([this, type] {
          auto dict = stack_.back().toGenericDict();
          stack_.pop_back();
          auto cls = type.type_->expect<at::ClassType>();
          size_t n = cls->numAttributes();
          auto obj = c10::ivalue::Object::create(type, n);
          for (size_t i = 0; i < n; ++i) {
            obj->setSlot(i, dict.at(cls->getAttributeName(i)));
          }
          stack_.emplace_back(std::move(obj));
        });
      }
      stack_.emplace_back(int64_t(globals_.size() - 1));
    } break;
    case OpCode::NEWOBJ: {
      // pop empty tuple, the actual action is stored in the globals_stack_
      stack_.pop_back();
    } break;
    // because we have NEWOBJ do nothing, BUILD and REDUCE end up doing
    // the same thing
    case OpCode::BUILD:
    case OpCode::REDUCE: {
      // stack is: <functor_idx> <functor_arg>
      // extract <functor_idx> and remove from the stack:
      std::swap(*(stack_.end() - 2), *(stack_.end() - 1));
      size_t idx = stack_.back().toInt();
      stack_.pop_back();
      // stack is: <functor_arg>
      globals_.at(idx)();
    } break;
    default:
      AT_ERROR(
          "Unknown opcode for unpickling at ",
          reinterpret_cast<void*>(opcode),
          ": ",
          static_cast<uint8_t>(opcode));
  }
  return opcode;
}

// Pop all the list items off of the stack and append them to the list at the
// corresponding MARK
void Unpickler::readList() {
  size_t start = marks_.back();
  marks_.pop_back();
  auto list_ivalue = stack_.at(start - 1);
  auto num_elements = stack_.size() - start;
  auto elements = at::ArrayRef<IValue>(stack_).slice(start);
  if (list_ivalue.isIntList()) {
    auto list = std::move(list_ivalue).toIntList();
    list.reserve(num_elements);
    for (const auto& elem : elements) {
      list.emplace_back(elem.toInt());
    }
  } else if (list_ivalue.isTensorList()) {
    auto list = std::move(list_ivalue).toTensorList();
    list.reserve(num_elements);
    for (const auto& elem : elements) {
      list.emplace_back(elem.toTensor());
    }
  } else if (list_ivalue.isDoubleList()) {
    auto list = std::move(list_ivalue).toDoubleList();
    list.reserve(num_elements);
    for (const auto& elem : elements) {
      list.emplace_back(elem.toDouble());
    }
  } else if (list_ivalue.isBoolList()) {
    auto list = std::move(list_ivalue).toBoolList();
    list.reserve(num_elements);
    for (const auto& elem : elements) {
      list.push_back(elem.toBool());
    }
  } else if (list_ivalue.isGenericList()) {
    auto list = std::move(list_ivalue).toGenericList();
    list.reserve(num_elements);
    for (const auto& elem : elements) {
      list.emplace_back(elem);
    }
  } else {
    AT_ERROR("Unknown IValue list kind: ", list_ivalue.tagKind());
  }

  stack_.erase(stack_.begin() + start, stack_.end());
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
    TORCH_CHECK(
        is_valid_python_id_char(c),
        "Found character '",
        uint8_t(c),
        "' in string, "
        "strings must be qualified Python identifiers");

    // Increment after to exclude newline from string
    ++n;
    TORCH_CHECK(
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

std::pair<at::Tensor, uint64_t> getWriteableTensor(const at::Tensor& tensor) {
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
    TORCH_CHECK(
        storage_tensor.element_size() * storage_tensor.storage().size() ==
            record_size,
        "Storage tensor size did not match record size");
  }

  return std::make_pair(storage_tensor, record_size);
}

uint64_t getStorageKey(const at::Tensor& tensor) {
  at::StorageImpl* storage_key = tensor.storage().unsafeGetStorageImpl();
  return reinterpret_cast<intptr_t>(storage_key);
}

} // namespace jit
} // namespace torch
