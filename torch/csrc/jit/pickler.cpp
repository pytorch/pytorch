#include <ATen/ATen.h>
#include <ATen/core/Dict.h>
#include <torch/csrc/jit/function.h>
#include <torch/csrc/jit/pickler.h>
#include <string>

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

static void postSetStateValidate(const IValue& v) {
  auto obj = v.toObject();
  const auto& objType = obj->type();
  for (size_t i = 0; i < objType->numAttributes(); i++) {
    const auto& attrType = objType->getAttribute(i);
    const auto& attrName = objType->getAttributeName(i);
    const auto& slot = obj->getSlot(i);
    // const auto attrType = objType->getAttribute(i);
    // Verify that all the non-optional attributes have been initialized
    // TODO: Issue #20497
    if (attrType->kind() != TypeKind::OptionalType) {
      TORCH_CHECK(
          !slot.isNone(),
          "The field '",
          attrName,
          "' was left unitialized after __setstate__, but expected a ",
          "value of type '",
          attrType->python_str(),
          "'");
    }
  }
}

void Pickler::protocol() {
  push<PickleOpCode>(PickleOpCode::PROTO);
  push<uint8_t>(PROTOCOL_VERSION);
}

void Pickler::startTuple() {
  // All attributes get pushed into a tuple and their indices saved in the
  // module def
  push<PickleOpCode>(PickleOpCode::MARK);
}

void Pickler::endTuple() {
  push<PickleOpCode>(PickleOpCode::TUPLE);
}

void Pickler::stop() {
  push<PickleOpCode>(PickleOpCode::STOP);
}

// unmemoized version called by pushIValue
void Pickler::pushIValueImpl(const IValue& ivalue) {
  if (ivalue.isTensor()) {
    pushTensor(ivalue);
  } else if (ivalue.isTuple()) {
    pushTuple(ivalue);
  } else if (ivalue.isDouble()) {
    pushDouble(ivalue.toDouble());
  } else if (ivalue.isInt()) {
    pushInt(ivalue.toInt());
  } else if (ivalue.isBool()) {
    if (ivalue.toBool()) {
      push<PickleOpCode>(PickleOpCode::NEWTRUE);
    } else {
      push<PickleOpCode>(PickleOpCode::NEWFALSE);
    }
  } else if (ivalue.isString()) {
    pushString(ivalue.toStringRef());
  } else if (ivalue.isGenericList()) {
    pushGenericList(ivalue);
  } else if (ivalue.isGenericDict()) {
    pushDict(ivalue);
  } else if (ivalue.isNone()) {
    push<PickleOpCode>(PickleOpCode::NONE);
  } else if (ivalue.isIntList()) {
    pushSpecializedList(
        ivalue, PicklerClass::INTLIST, [=](const IValue& ivalue) {
          for (const int64_t item : ivalue.toIntListRef()) {
            pushIValue(item);
          }
        });
  } else if (ivalue.isTensorList()) {
    pushSpecializedList(
        ivalue, PicklerClass::TENSORLIST, [=](const IValue& ivalue) {
          for (const at::Tensor& item : ivalue.toTensorListRef()) {
            pushIValue(item);
          }
        });
  } else if (ivalue.isDoubleList()) {
    pushSpecializedList(
        ivalue, PicklerClass::DOUBLELIST, [=](const IValue& ivalue) {
          for (double item : ivalue.toDoubleListRef()) {
            pushIValue(item);
          }
        });
  } else if (ivalue.isBoolList()) {
    pushSpecializedList(
        ivalue, PicklerClass::BOOLLIST, [=](const IValue& ivalue) {
          for (bool item : ivalue.toBoolList()) {
            pushIValue(item);
          }
        });
  } else if (ivalue.isObject()) {
    auto obj = ivalue.toObject();
    auto type = obj->type();
    pushGlobal(type->name()->prefix(), type->name()->name());
    push<PickleOpCode>(PickleOpCode::EMPTY_TUPLE);
    push<PickleOpCode>(PickleOpCode::NEWOBJ);
    if (checkHasValidSetGetState(type)) {
      Function* getstate = type->getMethod("__getstate__");
      pushIValue((*getstate)({obj}));
    } else {
      push<PickleOpCode>(PickleOpCode::EMPTY_DICT);
      push<PickleOpCode>(PickleOpCode::MARK);
      for (size_t i = 0, n = type->numAttributes(); i < n; ++i) {
        pushString(type->getAttributeName(i));
        pushIValue(obj->getSlot(i));
      }
      push<PickleOpCode>(PickleOpCode::SETITEMS);
    }
    push<PickleOpCode>(PickleOpCode::BUILD);
  } else {
    AT_ERROR("Unknown IValue type for pickling: ", ivalue.tagKind());
  }
}

void Pickler::pushIValue(const IValue& ivalue) {
  bool shouldMemoizeByPointer =
    ivalue.isPtrType() && !ivalue.isString() && ivalue.use_count() > 1;

  // Mutable ivalues are memoized by pointer equality, which we handle at this outer
  // granularity.  Immutable ivalues are memoized by value equality which is handled in
  // the type-specific handlers inside pushIValueImpl.
  if (shouldMemoizeByPointer) {
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

    pushIValueImpl(ivalue);

    memoized_ivalues_.push_back(ivalue);
    memoized_ivalue_map_[ivalue.internalToPointer()] = pushNextBinPut();
  } else {
    pushIValueImpl(ivalue);
  }
}

void Pickler::pushInt(int64_t n) {
  if (n >= std::numeric_limits<uint8_t>::min() &&
      n <= std::numeric_limits<uint8_t>::max()) {
    push<PickleOpCode>(PickleOpCode::BININT1);
    push<uint8_t>(n);
  } else if (
      n >= std::numeric_limits<uint16_t>::min() &&
      n <= std::numeric_limits<uint16_t>::max()) {
    push<PickleOpCode>(PickleOpCode::BININT2);
    push<uint16_t>(n);
  } else if (
      n >= std::numeric_limits<int32_t>::min() &&
      n <= std::numeric_limits<int32_t>::max()) {
    push<PickleOpCode>(PickleOpCode::BININT);
    push<int32_t>(n);
  } else {
    // Push 8 byte integer
    push<PickleOpCode>(PickleOpCode::LONG1);
    push<uint8_t>(8);
    push<int64_t>(n);
  }
}

void Pickler::pushBinGet(uint32_t memo_id) {
  if (memo_id <= std::numeric_limits<uint8_t>::max()) {
    push<PickleOpCode>(PickleOpCode::BINGET);
    push<uint8_t>(memo_id);
  } else {
    // Memoized too many items, issue a LONG_BINGET instead
    push<PickleOpCode>(PickleOpCode::LONG_BINGET);
    push<uint32_t>(memo_id);
  }
}

// unmemoized encoding of a string
void Pickler::pushStringImpl(const std::string& string) {
  push<PickleOpCode>(PickleOpCode::BINUNICODE);
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

void Pickler::pushStorageOfTensor(const at::Tensor& tensor) {
  const at::Storage& storage = tensor.storage();
  void* addr = storage.unsafeGetStorageImpl();
  auto it = memoized_storage_map_.find(addr);
  if (it != memoized_storage_map_.end()) {
    pushBinGet(it->second);
    return;
  }

  // Tuple for persistent_load
  push<PickleOpCode>(PickleOpCode::MARK);
  // typename
  pushString("storage");
  // data_type
  std::stringstream data_type;
  data_type << toString(tensor.scalar_type()) << "Storage";
  pushGlobal("torch", data_type.str());
  // root_key
  pushString(std::to_string(tensor_data_.size()));
  // location
  std::stringstream ss;
  ss << tensor.device();
  pushString(ss.str());
  // size
  pushInt(tensor.storage().size());
  // view_metadata
  push<PickleOpCode>(PickleOpCode::NONE);
  push<PickleOpCode>(PickleOpCode::TUPLE);
  push<PickleOpCode>(PickleOpCode::BINPERSID);

  // TODO: Skip this if not writing tensors
  memoized_storage_map_[addr] = pushNextBinPut();
  tensor_data_.push_back(getWriteableTensorData(tensor));
}

void Pickler::pushBytes(const std::string& string) {
  writer_(string.data(), string.size());
}

void Pickler::pushGlobal(
    const std::string& module_name,
    const std::string& class_name) {
  std::stringstream ss;
  ss << module_name << "\n" << class_name << "\n";
  std::string key = ss.str();
  auto memo_entry = memoized_globals_map_.find(key);
  if (memo_entry == memoized_globals_map_.end()) {
    push<PickleOpCode>(PickleOpCode::GLOBAL);
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
  bool quantized = tensor.is_quantized();
  // The arguments to this function are:
  //    storage, storage_offset, size, stride, requires_grad, backward_hooks
  pushGlobal(
      "torch._utils", quantized ? "_rebuild_qtensor" : "_rebuild_tensor_v2");

  push<PickleOpCode>(PickleOpCode::MARK);

  pushStorageOfTensor(tensor);

  // storage offset
  pushInt(tensor.storage_offset());

  // size
  push<PickleOpCode>(PickleOpCode::MARK);
  for (auto size : tensor.sizes()) {
    pushInt(size);
  }
  push<PickleOpCode>(PickleOpCode::TUPLE);

  // stride
  push<PickleOpCode>(PickleOpCode::MARK);
  for (auto stride : tensor.strides()) {
    pushInt(stride);
  }
  push<PickleOpCode>(PickleOpCode::TUPLE);

  if (quantized) {
    pushDouble(tensor.q_scale());
    pushInt(tensor.q_zero_point());
  }

  // requires_grad
  pushIValue(tensor.requires_grad());

  // backward_hooks
  pushGlobal("collections", "OrderedDict");
  push<PickleOpCode>(PickleOpCode::EMPTY_TUPLE);
  // Construct the collections.OrderedDict for the backward_hooks
  push<PickleOpCode>(PickleOpCode::REDUCE);

  push<PickleOpCode>(PickleOpCode::TUPLE);

  // Call torch._utils._rebuild_tensor_v2
  push<PickleOpCode>(PickleOpCode::REDUCE);
}

void Pickler::pushClass(PicklerClass cls) {
  pushGlobal("torch.jit._pickle", getClassName(cls));
}

void Pickler::pushSpecializedList(
    const IValue& ivalue,
    PicklerClass cls,
    const std::function<void(const IValue&)>& item_pusher) {
  pushClass(cls);

  // Reduce arguments are spread (e.g. `*args`) before calling the global,
  // so wrap in a tuple
  push<PickleOpCode>(PickleOpCode::MARK);

  push<PickleOpCode>(PickleOpCode::EMPTY_LIST);
  // Mark list
  push<PickleOpCode>(PickleOpCode::MARK);

  // Add all items
  item_pusher(ivalue);

  // Finish list
  push<PickleOpCode>(PickleOpCode::APPENDS);

  // Finish tuple
  push<PickleOpCode>(PickleOpCode::TUPLE);

  // Call reduce
  push<PickleOpCode>(PickleOpCode::REDUCE);
}

void Pickler::pushDouble(double value) {
  AT_ASSERT(sizeof(double) == 8);
  char* bytes = reinterpret_cast<char*>(&value);

  push<PickleOpCode>(PickleOpCode::BINFLOAT);
  for (size_t i = 0; i < 8; ++i) {
    push<uint8_t>(bytes[8 - i - 1]);
  }
}

void Pickler::pushLong(const std::string& data) {
  uint64_t size = data.size();

  if (size <= std::numeric_limits<uint8_t>::max()) {
    push<PickleOpCode>(PickleOpCode::LONG1);
    push<uint8_t>(size);
  } else {
    TORCH_INTERNAL_ASSERT(
        data.size() > std::numeric_limits<uint32_t>::max(),
        "Cannot pickle a long with a size larger than 4 bytes")
    push<PickleOpCode>(PickleOpCode::LONG4);
    push<uint64_t>(size);
  }
  pushBytes(data);
}

void Pickler::pushTensorReference(const IValue& ivalue) {
  pushClass(PicklerClass::TENSOR);
  tensor_table_->push_back(ivalue.toTensor());
  int64_t tensor_id = tensor_table_->size() - 1;
  // Reduce arguments are spread (e.g. `*args`) before calling the global,
  // so wrap in a tuple
  push<PickleOpCode>(PickleOpCode::MARK);
  pushIValue(tensor_id);
  push<PickleOpCode>(PickleOpCode::TUPLE);

  push<PickleOpCode>(PickleOpCode::REDUCE);
}

void Pickler::pushEmptyDict() {
  push<PickleOpCode>(PickleOpCode::EMPTY_DICT);
}
void Pickler::pushDict(const IValue& ivalue) {
  pushEmptyDict();
  auto dict_items = iterationOrder(ivalue.toGenericDict());
  if (dict_items.size() == 0) {
    return;
  }

  push<PickleOpCode>(PickleOpCode::MARK);

  // Sort the dict for deterministic keys
  for (const auto& pair : dict_items) {
    pushIValue(pair.first);
    pushIValue(pair.second);
  }

  push<PickleOpCode>(PickleOpCode::SETITEMS);
}

size_t Pickler::pushNextBinPut() {
  if (memo_id_ <= std::numeric_limits<uint8_t>::max()) {
    push<PickleOpCode>(PickleOpCode::BINPUT);
    push<uint8_t>(memo_id_);
  } else {
    // Memoized too many items, issue a LONG_BINPUT instead
    push<PickleOpCode>(PickleOpCode::LONG_BINPUT);
    push<uint32_t>(memo_id_);
  }
  AT_ASSERT(memo_id_ <= std::numeric_limits<uint32_t>::max());
  ++memo_id_;
  return memo_id_ - 1;
}

void Pickler::pushGenericList(const IValue& ivalue) {
  auto list = ivalue.toGenericListRef();
  push<PickleOpCode>(PickleOpCode::EMPTY_LIST);

  push<PickleOpCode>(PickleOpCode::MARK);

  for (const IValue& item : list) {
    pushIValue(item);
  }

  push<PickleOpCode>(PickleOpCode::APPENDS);
}

void Pickler::pushTuple(const IValue& ivalue) {
  auto tuple = ivalue.toTuple();
  auto tuple_size = tuple->elements().size();

  switch (tuple_size) {
  case 0: {
    push<PickleOpCode>(PickleOpCode::EMPTY_TUPLE);
  } break;
  case 1: {
    pushIValue(tuple->elements()[0]);
    push<PickleOpCode>(PickleOpCode::TUPLE1);
  } break;
  case 2: {
    pushIValue(tuple->elements()[0]);
    pushIValue(tuple->elements()[1]);
    push<PickleOpCode>(PickleOpCode::TUPLE2);
  } break;
  case 3: {
    pushIValue(tuple->elements()[0]);
    pushIValue(tuple->elements()[1]);
    pushIValue(tuple->elements()[2]);
    push<PickleOpCode>(PickleOpCode::TUPLE3);
  } break;
  default: {
    push<PickleOpCode>(PickleOpCode::MARK);
    for (const IValue& item : tuple->elements()) {
      pushIValue(item);
    }
    push<PickleOpCode>(PickleOpCode::TUPLE);
  } break;
  }
}

// Pickled objects are stored in a form compatible with Python pickling.
// In torchscript List[T]/Dict[K, V] are statically typed and contain
// dynamic type tags allow T, K, and V to be recovered. But this info
// is not stored in the Python pickling information. However, we
// can recover this information from the static type of the top-level
// object being unpickled, because we have a record of the type of the
// objects it contains as attributes.
// `IfPossible` - we can only do this recovery when we have an object as
// the top-level unpickled thing (which is guarenteed for Modules, but
// not for torch.load/torch,save). Otherwise we do not know the types
// of the contained objects and cannot restore the tags.
static void restoreAccurateTypeTagsIfPossible(const IValue& root) {
  if (!root.isObject()) {
    return;
  }
  struct Work {
    TypePtr static_type;
    IValue value;
  };
  std::vector<Work> to_process = {{root.type(), root}};
  std::unordered_set<const void*> scanned;
  while (!to_process.empty()) {
    Work w = std::move(to_process.back());
    to_process.pop_back();
    // ensure we only scan each pointer value once, otherwise this
    // can become exponential (and if we allow recursive data in the future,
    // it would not terminiate).
    if (w.value.isPtrType()) {
      const void* key = w.value.internalToPointer();
      auto it = scanned.find(key);
      if (it != scanned.end()) {
        continue;
      }
      scanned.emplace_hint(it, key);
    }
    switch (w.static_type->kind()) {
      case TensorType::Kind:
      case NumberType::Kind:
      case FloatType::Kind:
      case IntType::Kind:
      case NoneType::Kind:
      case GeneratorType::Kind:
      case BoolType::Kind:
      case VarType::Kind:
      case CapsuleType::Kind:
      case StringType::Kind:
      case FunctionType::Kind:
      case DeviceObjType::Kind:
        // no op, there is nothing to tag
        break;
      case AnyType::Kind:
        // if Any type does show up, we no longer have a way to precisely
        // recover the type information since the w.value may be an untagged
        // List/Dict. We should prevent objects being serialized from having the
        // Any type and if we do allow it in functions limit it to non-heap
        // locations.
        TORCH_INTERNAL_ASSERT(
            false, "AnyType should not show up in the static type of objects");
      case TupleType::Kind: {
        auto t = w.value.toTuple();
        auto ttype = w.static_type->expect<TupleType>();
        for (size_t i = 0; i < ttype->containedTypes().size(); ++i) {
          Work elem = {ttype->containedTypes().at(i), t->elements().at(i)};
          to_process.emplace_back(std::move(elem));
        }
      } break;
      case FutureType::Kind: {
        auto f = w.value.toFuture();
        auto t = w.static_type->expect<FutureType>();
        if (f->completed()) {
          Work elem = {t->getElementType(), f->value()};
          to_process.emplace_back(std::move(elem));
        }
      } break;
      case OptionalType::Kind: {
        if (!w.value.isNone()) {
          auto t = w.static_type->expect<OptionalType>();
          Work elem = {t->getElementType(), w.value};
          to_process.emplace_back(std::move(elem));
        }
      } break;
      case ListType::Kind: {
        // specialized lists do not need their type refined, so we can exit
        // early here
        if (!w.value.isGenericList()) {
          break;
        }
        auto elem_type = w.static_type->cast<ListType>()->getElementType();
        auto lst = w.value.toGenericList();
        lst.unsafeSetElementType(elem_type);
        for (const IValue& item : lst) {
          Work elem = {elem_type, item};
          to_process.emplace_back(std::move(elem));
        }
      } break;
      case DictType::Kind: {
        auto dt = w.static_type->cast<DictType>();
        auto d = w.value.toGenericDict();
        d.unsafeSetKeyType(dt->getKeyType());
        d.unsafeSetValueType(dt->getValueType());
        for (const auto& item : d) {
          Work kelem = {dt->getKeyType(), item.key()};
          Work velem = {dt->getValueType(), item.value()};
          to_process.emplace_back(std::move(kelem));
          to_process.emplace_back(std::move(velem));
        }
      } break;
      // in both cases the dynamic type is a class, and we are going to tag with
      // the dynamic type
      case InterfaceType::Kind:
      case ClassType::Kind: {
        auto obj = w.value.toObject();
        auto typ = obj->type(); // note: intentionally using the dynamic type,
                                // the static type is potentially less accurate
        for (size_t i = 0; i < typ->numAttributes(); ++i) {
          Work elem = {typ->getAttribute(i), obj->getSlot(i)};
          to_process.emplace_back(std::move(elem));
        }
      };
    }
  }
}

IValue Unpickler::parse_ivalue() {
  run();
  TORCH_CHECK(
      stack_.size() == 1,
      "Unpickler expected 1 element on the stack, but found ",
      stack_.size());
  restoreAccurateTypeTagsIfPossible(stack_[0]);

  return stack_[0];
}

double Unpickler::readFloat() {
  AT_ASSERT(sizeof(double) == 8);
  double big_endian = read<double>();
  double little_endian;

  // Pickle floats are big endian, so reverse the bytes
  auto big_endian_ptr = reinterpret_cast<const char*>(&big_endian);
  std::reverse_copy(
      big_endian_ptr,
      big_endian_ptr + sizeof(big_endian),
      reinterpret_cast<char*>(&little_endian));

  return little_endian;
}

void Unpickler::run() {
  // Expect a PROTO opcode and protocol number at the start of blob
  auto opcode = readOpCode();
  TORCH_CHECK(
      opcode == PickleOpCode::PROTO,
      "Expected PROTO opcode at the start"
      " of pickle archive, found ", int(static_cast<uint8_t>(opcode)));
  uint8_t protocol = read<uint8_t>();
  TORCH_CHECK(
      protocol == 2,
      "Only Pickle protocol 2 is supported, found protocol = ",
      protocol);

  while (true) {
    PickleOpCode opcode = readInstruction();
    if (opcode == PickleOpCode::STOP) {
      return;
    }
  }
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

static std::vector<int64_t> tupleToIntList(const IValue& v) {
  return fmap(v.toTuple()->elements(), [](const IValue& v) -> int64_t {
    return v.toInt();
  });
}

PickleOpCode Unpickler::readInstruction() {
  auto opcode = readOpCode();
  switch (opcode) {
    case PickleOpCode::EMPTY_LIST: {
      stack_.emplace_back(c10::impl::GenericList(AnyType::get()));
    } break;
    case PickleOpCode::EMPTY_TUPLE: {
      if (empty_tuple_.isNone()) {
        // we only need one object, since tuples are not mutable.
        empty_tuple_ = c10::ivalue::Tuple::create({});
      }
      stack_.emplace_back(empty_tuple_);
    } break;
    case PickleOpCode::BINPUT: {
      size_t memo_id = read<uint8_t>();
      setInput(memo_id);
    } break;
    case PickleOpCode::LONG_BINPUT: {
      TORCH_CHECK(
          std::numeric_limits<size_t>::max() >=
              std::numeric_limits<uint32_t>::max(),
          "Found a LONG_BINPUT opcode, but size_t on this system is "
          "not big enough to decode it");
      size_t memo_id = read<uint32_t>();
      setInput(memo_id);
    } break;
    case PickleOpCode::MARK: {
      // Mark location of the container ivalue in the stack
      marks_.push_back(stack_.size());
    } break;
    case PickleOpCode::NEWTRUE: {
      stack_.emplace_back(true);
    } break;
    case PickleOpCode::NEWFALSE: {
      stack_.emplace_back(false);
    } break;
    case PickleOpCode::NONE: {
      stack_.emplace_back(IValue());
    } break;
    case PickleOpCode::BININT1: {
      uint8_t value = read<uint8_t>();
      stack_.emplace_back(int64_t(value));
    } break;
    case PickleOpCode::BININT2: {
      uint16_t value = read<uint16_t>();
      stack_.emplace_back(int64_t(value));
    } break;
    case PickleOpCode::BININT: {
      int32_t value = read<int32_t>();
      stack_.emplace_back(int64_t(value));
    } break;
    case PickleOpCode::LONG1: {
      // Only read LONG1s with 8 as the length
      uint8_t length = read<uint8_t>();
      TORCH_CHECK(length == 8, "Expected length to be 8, got ", int(length));
      stack_.emplace_back(int64_t(read<int64_t>()));
    } break;
    case PickleOpCode::BINUNICODE: {
      uint32_t length = read<uint32_t>();
      stack_.emplace_back(readBytes(length));
    } break;
    case PickleOpCode::BINFLOAT:
      stack_.emplace_back(readFloat());
      break;
    case PickleOpCode::TUPLE: {
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
    case PickleOpCode::TUPLE1: {
        auto tuple = c10::ivalue::Tuple::create(pop(stack_, 1));
        stack_.emplace_back(tuple);
    } break;
    case PickleOpCode::TUPLE2: {
        auto tuple = c10::ivalue::Tuple::create(pop(stack_, 2));
        stack_.emplace_back(tuple);
    } break;
    case PickleOpCode::TUPLE3: {
        auto tuple = c10::ivalue::Tuple::create(pop(stack_, 3));
        stack_.emplace_back(tuple);
    } break;
    case PickleOpCode::EMPTY_DICT:
      stack_.emplace_back(
          c10::impl::GenericDict(AnyType::get(), AnyType::get()));
      break;
    case PickleOpCode::APPENDS: {
      readList();
    } break;
    case PickleOpCode::SETITEMS: {
      size_t start = marks_.back();
      marks_.pop_back();
      auto dict = stack_.at(start - 1).toGenericDict();
      for (size_t i = start; i < stack_.size(); i += 2) {
        dict.insert_or_assign(stack_[i], stack_[i + 1]);
      }
      stack_.erase(stack_.begin() + start, stack_.end());
    } break;
    case PickleOpCode::BINGET: {
      stack_.push_back(memo_table_.at(read<uint8_t>()));
    } break;
    case PickleOpCode::LONG_BINGET: {
      stack_.push_back(memo_table_.at(read<uint32_t>()));
    } break;
    case PickleOpCode::STOP:
      break;
    case PickleOpCode::GLOBAL: {
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
              TORCH_INTERNAL_ASSERT(
                  tensor_table_,
                  "Pickler tried to write a tensor but had no tensor table to write to");
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
              TORCH_CHECK(
                  tensor_table_,
                  "Found a tensor table reference but Unpickler"
                  " has no tensor table\n");
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
      } else if (
          module_name == "torch._utils" &&
          (class_name == "_rebuild_tensor_v2" ||
           class_name == "_rebuild_qtensor")) {
        bool quantized = class_name == "_rebuild_qtensor";
        globals_.emplace_back([this, quantized] {
          auto tup = pop(stack_).toTuple();
          const auto& elements = tup->elements();
          size_t idx = 0;
          auto storage_tensor = elements.at(idx++).toTensor();
          int64_t storage_offset = elements.at(idx++).toInt();
          std::vector<int64_t> size = tupleToIntList(elements.at(idx++));
          std::vector<int64_t> stride = tupleToIntList(elements.at(idx++));
          double q_scale = 0.;
          int64_t q_zero_point = 0;
          if (quantized) {
            q_scale = elements.at(idx++).toDouble();
            q_zero_point = elements.at(idx++).toInt();
          }
          bool requires_grad = elements.at(idx++).toBool();
          // elements[idx++] is empty backwards hooks
          at::Tensor result = quantized
              ? at::_empty_affine_quantized(
                    {}, storage_tensor.options(), q_scale, q_zero_point)
              : at::empty({0}, storage_tensor.options());
          at::TensorImpl* impl = result.unsafeGetTensorImpl();
          impl->set_storage(storage_tensor.storage());
          impl->set_storage_offset(storage_offset);
          impl->set_sizes_and_strides(size, stride);
          result = autograd::make_variable(result, requires_grad);
          stack_.push_back(std::move(result));
        });
      } else if (module_name == "collections" && class_name == "OrderedDict") {
        globals_.emplace_back([this] {
          // drop the Tuple that was argument to OrderedDict, and replace it
          // with None OrderedDicts only appear in tensor deserialization and
          // their value is never used
          stack_.back() = IValue();
        });
      } else if (module_name == "torch") {
        c10::optional<c10::ScalarType> scalar_type;
#define CHECK_SCALAR(_, name)          \
  if (class_name == #name "Storage") { \
    scalar_type = c10::k##name;        \
  }
        AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(CHECK_SCALAR)
#undef CHECK_SCALAR
        // NOTE: this does not put a global into the global table,
        // like the other branches here because no REDUCE or BUILD will
        // be called on this value. Instead, we just put it on the stack
        // and return early
        AT_ASSERT(
            scalar_type.has_value(),
            "class name not understood: torch.",
            class_name);
        stack_.emplace_back(int64_t(*scalar_type));
        return opcode;
      } else {
        AT_ASSERT(class_resolver_);
        at::StrongTypePtr type =
            class_resolver_(c10::QualifiedName(module_name, class_name));
        auto cls = type.type_->expect<at::ClassType>();
        size_t n = cls->numAttributes();
        if (checkHasValidSetGetState(type.type_)) {
          globals_.emplace_back([this, type, n] {
            auto arg = std::move(stack_.back());
            stack_.pop_back();
            auto obj = c10::ivalue::Object::create(type, n);
            // XXX: Do not optimize __setstate__, so that we don't try to
            // specialize the class before it is initialized.
            setGraphExecutorOptimize(false);
            (*type.type_->getMethod("__setstate__"))({obj, arg});
            setGraphExecutorOptimize(true);
            postSetStateValidate(obj);
            stack_.emplace_back(std::move(obj));
          });
        } else {
          globals_.emplace_back([this, type, cls, n] {
            auto dict = std::move(stack_.back()).toGenericDict();
            stack_.pop_back();
            auto obj = c10::ivalue::Object::create(type, n);
            for (size_t i = 0; i < n; ++i) {
              obj->setSlot(i, dict.at(cls->getAttributeName(i)));
            }
            stack_.emplace_back(std::move(obj));
          });
        }
      }
      stack_.emplace_back(int64_t(globals_.size() - 1));
    } break;
    case PickleOpCode::NEWOBJ: {
      // pop empty tuple, the actual action is stored in the globals_stack_
      stack_.pop_back();
    } break;
    // because we have NEWOBJ do nothing, BUILD and REDUCE end up doing
    // the same thing
    case PickleOpCode::BUILD:
    case PickleOpCode::REDUCE: {
      // stack is: <functor_idx> <functor_arg>
      // extract <functor_idx> and remove from the stack:
      std::swap(*(stack_.end() - 2), *(stack_.end() - 1));
      size_t idx = stack_.back().toInt();
      stack_.pop_back();
      // stack is: <functor_arg>
      globals_.at(idx)();
    } break;
    case PickleOpCode::BINPERSID: {
      auto args = pop(stack_).toTuple()->elements();
      AT_ASSERT(
          args.at(0).toStringRef() == "storage",
          "unknown PERSID key ",
          args.at(0).toStringRef());
      at::ScalarType type = args.at(1).toScalarType();
      const std::string& key = args.at(2).toStringRef();
      at::Device device(args.at(3).toStringRef());
      if (device_) {
        device = *device_;
      }
      at::DataPtr storage_ptr = read_record_(key);
      int64_t numel = args.at(4).toInt();
      at::Storage storage(
          at::CPU(type).typeMeta(),
          numel,
          std::move(storage_ptr),
          /*allocator=*/nullptr,
          /*resizable=*/false); // NB: we didn't set any allocator for the
                                // tensor
      auto options = at::CPU(type).options();
      at::Tensor tensor;
      if (options.backend() == c10::Backend::QuantizedCPU) {
        tensor = at::_empty_affine_quantized({}, options, 0, 0)
                     .set_(storage, 0, {}, {});
      } else {
        tensor = at::empty({0}, options).set_(storage);
      }

      if (device.type() == at::DeviceType::CUDA) {
        tensor = tensor.to(device, tensor.scalar_type());
      } else if (device.type() != at::DeviceType::CPU) {
        AT_ERROR(
            "supported devices include CPU and CUDA, however got ",
            at::DeviceTypeName(device.type(), false));
      }
      stack_.push_back(std::move(tensor));
    } break;
    default: {
      AT_ERROR(
          "Unknown opcode for unpickling at ",
          reinterpret_cast<void*>(opcode),
          ": ",
          int(static_cast<uint8_t>(opcode)));
    } break;
  }
  return opcode;
}

// Read a number of bytes from the input stream
std::string Unpickler::readBytes(size_t length) {
  std::string data(length, 0);
  // This is fine since C++11 has contiguous strings
  if (!reader_(&data[0], length)) {
    AT_ERROR("Unexpected end of pickler archive.");
  }
  return data;
}

// Pop all the list items off of the stack and append them to the list at
// the corresponding MARK
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
  std::stringstream ss;
  while (true) {
    char c = read<char>();
    if (c == '\n') {
      break;
    }

    ss << c;

    // Simple check just in case there is no terminating '\n'
    TORCH_CHECK(
        is_valid_python_id_char(c),
        "Found character '",
        int(uint8_t(c)),
        "' in string, ",
        "strings must be qualified Python identifiers");
  }
  return ss.str();
}

PickleOpCode Unpickler::readOpCode() {
  return static_cast<PickleOpCode>(read<uint8_t>());
}

WriteableTensorData getWriteableTensorData(const at::Tensor& tensor) {
  WriteableTensorData result;
  result.tensor_ = tensor;
  result.size_ = tensor.element_size() * tensor.storage().size();
  // TODO HIP support
  if (tensor.storage().device_type() == at::DeviceType::CUDA) {
    // NB: This new tensor is created to support cuda tensors.
    // Storages can be mutated when converting tensors from cuda to cpu,
    // and we need a cpu tensor to copy data from.
    result.tensor_ = at::empty({0}, tensor.options())
                         .set_(
                             tensor.storage(),
                             /* storage_offset = */ 0,
                             /* size = */
                             {static_cast<int64_t>(tensor.storage().size())},
                             /* stride = */ {1})
                         .cpu();
    TORCH_CHECK(
        result.tensor_.element_size() * result.tensor_.storage().size() ==
            result.size_,
        "Storage tensor size did not match record size");
  }
  return result;
}

bool checkHasValidSetGetState(const std::shared_ptr<c10::ClassType>& cls) {
  // Check that the schemas for __getstate__ and __setstate__ are correct
  auto getstate = cls->getMethod("__getstate__");
  if (getstate == nullptr) {
    return false;
  }
  auto get_schema = getstate->getSchema();

  // Check __getstate__
  //   __getstate__ is expected to be (self) -> T
  TORCH_CHECK(
      get_schema.arguments().size() == 1,
      "'__getstate__' must have 'self' as its only argument, but found ",
      get_schema.arguments().size(),
      " arguments");
  TORCH_CHECK(
      get_schema.returns().size() == 1,
      "'__getstate__' must return 1 value, but found ",
      get_schema.returns().size());

  // Check __setstate__ if the method exists
  //   __setstate__ is expected to be (self, T) -> None
  auto setstate = cls->getMethod("__setstate__");
  if (!setstate) {
    return false;
  }
  auto set_schema = setstate->getSchema();

  TORCH_CHECK(
      set_schema.arguments().size() == 2,
      "'__setstate__' must have 'self' and the state as its "
      "only arguments, but found ",
      set_schema.arguments().size(),
      " arguments");
  TORCH_CHECK(
      set_schema.returns().size() == 1,
      "'__setstate__' must return None, but found ",
      set_schema.returns().size(),
      " return values");
  TORCH_CHECK(
      set_schema.returns().at(0).type()->isSubtypeOf(NoneType::get()),
      "'__setstate__' must return None, but found value of type",
      set_schema.returns().at(0).type()->python_str());

  // Check that the return type of __getstate__ matches the input to
  // __setstate__
  auto get_type = get_schema.returns().at(0).type();
  auto set_type = set_schema.arguments().at(1).type();

  TORCH_CHECK(
      set_type->isSubtypeOf(get_type),
      "'__getstate__'s return type (",
      get_type->python_str(),
      ") does not match '__setstate__'s argument type (",
      set_type->python_str(),
      ")");

  return true;
}

} // namespace jit
} // namespace torch
