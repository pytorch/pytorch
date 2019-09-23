#include <ATen/ATen.h>
#include <ATen/core/Dict.h>
#include <torch/csrc/jit/function.h>
#include <torch/csrc/jit/pickler.h>
#include <aten/src/ATen/quantized/Quantizer.h>
#include <string>

namespace torch {
namespace jit {

using ::c10::IValue;

// Protocol 2 is the highest that can be decoded by Python 2
// See https://docs.python.org/3/library/pickle.html#data-stream-format
constexpr static uint8_t PROTOCOL_VERSION = 2;

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
    push<PickleOpCode>(PickleOpCode::MARK);
    pushGlobal("torch", toString(tensor.qscheme()));
    // tuple of (qscheme, scale, zp) or (qscheme, scales, zps, axis)
    switch (tensor.qscheme()) {
      case at::kPerTensorAffine:
        pushDouble(tensor.q_scale());
        pushInt(tensor.q_zero_point());
        break;
      case at::kPerChannelAffine: {
        const auto* quantizer = static_cast<at::PerChannelAffineQuantizer*>(
            tensor.quantizer().get());
        pushIValue(c10::List<double>(quantizer->scales()));
        pushIValue(c10::List<int64_t>(quantizer->zero_points()));
        pushIValue(c10::List<int64_t>(quantizer->axis()));
      } break;
      default:
        TORCH_CHECK(
            false,
            "Unsupported tensor quantization type in serialization ",
            toString(tensor.qscheme()));
        break;
    }
    push<PickleOpCode>(PickleOpCode::TUPLE);
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
