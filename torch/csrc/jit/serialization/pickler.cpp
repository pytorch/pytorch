#include <ATen/ATen.h>
#include <ATen/core/Dict.h>
#ifdef USE_RPC
#include <torch/csrc/distributed/rpc/rref_context.h>
#endif
#include <ATen/quantized/Quantizer.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/serialization/pickler.h>
#include <string>
#include <type_traits>

namespace torch {
namespace jit {

using ::c10::IValue;

// Protocol 2 is the highest that can be decoded by Python 2
// See https://docs.python.org/3/library/pickle.html#data-stream-format
constexpr static uint8_t PROTOCOL_VERSION = 2;

// NOLINTNEXTLINE(bugprone-exception-escape)
Pickler::~Pickler() {
  flush();
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
  flush();
}

// unmemoized version called by pushIValue
void Pickler::pushIValueImpl(const IValue& ivalue) {
  if (ivalue.isTensor()) {
    pushTensor(ivalue);
  } else if (ivalue.isTuple()) {
    pushTuple(ivalue);
  } else if (ivalue.isDouble()) {
    pushDouble(ivalue.toDouble());
  } else if (ivalue.isComplexDouble()) {
    pushComplexDouble(ivalue);
  } else if (ivalue.isInt()) {
    pushInt(ivalue.toInt());
  } else if (ivalue.isBool()) {
    pushBool(ivalue.toBool());
  } else if (ivalue.isString()) {
    pushString(ivalue.toStringRef());
  } else if (ivalue.isGenericDict()) {
    pushDict(ivalue);
  } else if (ivalue.isNone()) {
    push<PickleOpCode>(PickleOpCode::NONE);
  } else if (ivalue.isIntList()) {
    pushSpecializedList(ivalue, "build_intlist", [=](const IValue& ivalue) {
      for (const int64_t item : ivalue.toIntVector()) {
        pushInt(item);
      }
    });
  } else if (ivalue.isTensorList()) {
    pushSpecializedList(ivalue, "build_tensorlist", [=](const IValue& ivalue) {
      for (const at::Tensor& item : ivalue.toTensorVector()) {
        pushIValue(item);
      }
    });
  } else if (ivalue.isDoubleList()) {
    pushSpecializedList(ivalue, "build_doublelist", [=](const IValue& ivalue) {
      for (double item : ivalue.toDoubleVector()) {
        pushDouble(item);
      }
    });
  } else if (ivalue.isBoolList()) {
    pushSpecializedList(ivalue, "build_boollist", [=](const IValue& ivalue) {
      for (bool item : ivalue.toBoolList()) {
        pushBool(item);
      }
    });
    // note: isList must be after isIntList and friends because
    // isList is true for all lists.
  } else if (ivalue.isList()) {
    pushGenericList(ivalue);
  } else if (ivalue.isObject()) {
    auto obj = ivalue.toObject();
    auto type = obj->type();
    if (memoized_class_types_ != nullptr) {
      // memoize every class type the Pickler encountered
      // This is used to make sure we capture all the run-time types
      // and serialize them properly for class/interface polymorphism
      memoized_class_types_->emplace_back(type);
    }
    auto type_name = type->name().value();
    if (type_renamer_) {
      type_name = type_renamer_(type);
    }
    pushGlobal(type_name.prefix(), type_name.name());
    push<PickleOpCode>(PickleOpCode::EMPTY_TUPLE);
    push<PickleOpCode>(PickleOpCode::NEWOBJ);
    if (checkHasValidSetGetState(type)) {
      Function& getstate = type->getMethod("__getstate__");
      pushIValue(getstate({obj}));
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
  } else if (ivalue.isDevice()) {
    pushDevice(ivalue);
  } else if (ivalue.isCapsule()) {
    std::stringstream err;
    err << "Cannot serialize custom bound C++ class";
    if (memoized_class_types_ && memoized_class_types_->size()) {
      if (auto qualname = memoized_class_types_->back()->name()) {
        err << " " << qualname->qualifiedName();
      }
    }
    err << ". Please define serialization methods via def_pickle() for "
           "this class.";
    AT_ERROR(err.str());
  } else if (ivalue.isRRef()) {
#ifdef USE_RPC
    TORCH_CHECK(
        torch::distributed::rpc::getAllowJitRRefPickle() == true,
        "RRef jit pickling is only allowed inside RPC calls.");
    pushRRef(ivalue);
#else
    TORCH_CHECK(
        false, "RRef pickling is only supported with the distributed package");
#endif
  } else if (ivalue.isEnum()) {
    auto enum_holder = ivalue.toEnumHolder();
    const auto& qualified_class_name =
        enum_holder->type()->qualifiedClassName();
    pushGlobal(qualified_class_name.prefix(), qualified_class_name.name());
    pushIValue(enum_holder->value());
    push<PickleOpCode>(PickleOpCode::REDUCE);
  } else {
    AT_ERROR("Unknown IValue type for pickling: ", ivalue.tagKind());
  }
}

void Pickler::pushDevice(const IValue& ivalue) {
  auto device = ivalue.toDevice();
  auto deviceStr = device.str();
  auto it = memoized_devices_map_.find(deviceStr);
  if (it == memoized_devices_map_.end()) {
    pushGlobal("torch", "device");
    pushString(deviceStr);
    push<PickleOpCode>(PickleOpCode::TUPLE1);
    push<PickleOpCode>(PickleOpCode::REDUCE);
    memoized_devices_map_[deviceStr] = pushNextBinPut();
  } else {
    pushBinGet(it->second);
  }
}

#ifdef USE_RPC
void Pickler::pushRRef(const IValue& ivalue) {
  // It is the same as how rref is pickled in python, see PyRRef::pickle
  auto rrefInterface = ivalue.toRRef();
  auto rref =
      c10::static_intrusive_pointer_cast<distributed::rpc::RRef>(rrefInterface);
  pushGlobal("torch.distributed.rpc", "rref");
  auto& ctx = distributed::rpc::RRefContext::getInstance();
  auto rrefForkData = ctx.prepareChildFork(rref);
  push<PickleOpCode>(PickleOpCode::MARK);
  pushInt(rrefForkData.ownerId_);
  pushInt(rrefForkData.rrefId_.createdOn_);
  pushInt(rrefForkData.rrefId_.localId_);
  pushInt(rrefForkData.forkId_.createdOn_);
  pushInt(rrefForkData.forkId_.localId_);
  pushInt(rrefForkData.parent_);
  pushString(rrefForkData.typeStr_);
  push<PickleOpCode>(PickleOpCode::TUPLE);
  push<PickleOpCode>(PickleOpCode::REDUCE);
}
#endif

void Pickler::pushIValue(const IValue& ivalue) {
  bool shouldMemoizeByPointer =
      ivalue.isPtrType() && !ivalue.isString() && ivalue.use_count() > 1;

  // Mutable ivalues are memoized by pointer equality, which we handle at this
  // outer granularity.  Immutable ivalues are memoized by value equality which
  // is handled in the type-specific handlers inside pushIValueImpl.
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
    memoized_ivalue_map_[ptr] = pushNextBinPut();
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

void Pickler::pushBool(bool value) {
  push<PickleOpCode>(value ? PickleOpCode::NEWTRUE : PickleOpCode::NEWFALSE);
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
  std::string data_type =
      std::string(toString(tensor.scalar_type())).append("Storage");
  pushGlobal("torch", data_type);
  // root_key
  std::string root_key = get_tensor_id_ != nullptr
      ? get_tensor_id_(tensor)
      : c10::to_string(tensor_data_.size());
  pushString(root_key);
  // location
  pushString(tensor.device().str());
  // size
  pushInt(tensor.storage().nbytes() / tensor.element_size());

  push<PickleOpCode>(PickleOpCode::TUPLE);
  push<PickleOpCode>(PickleOpCode::BINPERSID);

  // TODO: Skip this if not writing tensors
  memoized_storage_map_[addr] = pushNextBinPut();
  tensor_data_.push_back(tensor);
}

void Pickler::pushBytes(const std::string& string) {
  static const size_t kSmallStr = 32;
  if (string.size() <= kSmallStr &&
      bufferPos_ + string.size() <= buffer_.size()) {
    // Small string that fits: buffer the data.
    memcpy(buffer_.data() + bufferPos_, string.data(), string.size());
    bufferPos_ += string.size();
  } else {
    // Otherwise, first flush, then write directly.
    flush();
    writer_(string.data(), string.size());
  }
}

void Pickler::pushGlobal(
    const std::string& module_name,
    const std::string& class_name) {
  std::string key;
  key.reserve(module_name.size() + class_name.size() + 2);
  key.append(module_name).append("\n").append(class_name).append("\n");
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

void Pickler::pushLiteralSparseTensor(const at::Tensor& tensor) {
  pushGlobal("torch._utils", "_rebuild_sparse_tensor");
  push<PickleOpCode>(PickleOpCode::MARK);
  // layout
  auto layout = static_cast<int>(tensor.layout());
  pushInt(layout);
  switch (layout) {
    case static_cast<int>(c10::Layout::Sparse):
      // size
      push<PickleOpCode>(PickleOpCode::MARK);
      for (auto size : tensor.sizes()) {
        pushInt(size);
      }
      push<PickleOpCode>(PickleOpCode::TUPLE);
      // requires grad
      pushIValue(tensor.requires_grad());
      // indices
      pushTensor(tensor._indices());
      // values
      pushTensor(tensor._values());
      break;
    case static_cast<int>(c10::Layout::SparseCsr):
      push<PickleOpCode>(PickleOpCode::MARK);
      for (auto size : tensor.sizes()) {
        pushInt(size);
      }
      push<PickleOpCode>(PickleOpCode::TUPLE);

      pushIValue(tensor.requires_grad());
      pushTensor(tensor.crow_indices());
      pushTensor(tensor.col_indices());
      pushTensor(tensor.values());
      break;
    default:
      TORCH_CHECK(
          false,
          "Unsupported sparse tensor layout type in serialization ",
          static_cast<c10::Layout>(layout));
      break;
  }
  // backward_hooks
  pushGlobal("collections", "OrderedDict");
  push<PickleOpCode>(PickleOpCode::EMPTY_TUPLE);
  // Construct the collections.OrderedDict for the backward_hooks
  push<PickleOpCode>(PickleOpCode::REDUCE);
  push<PickleOpCode>(PickleOpCode::TUPLE);
  // Call torch._utils._rebuild_sparse_coo_tensor
  push<PickleOpCode>(PickleOpCode::REDUCE);
}

void Pickler::pushLiteralTensor(const IValue& ivalue) {
  // In contrast to tensor references, literal tensors are included in the
  // pickle program binary blob. They are written to the file after the STOP
  // opcode. They can't be included in the pickle program itself without a bunch
  // of extra machinery since byte strings are limited to 4 GB.
  //
  // The format here is the same one used by `torch.save()`. The code for the
  // format can be found in `torch/serialization.py`.
  auto& tensor = ivalue.toTensor();

  if (tensor.is_sparse() || tensor.is_sparse_csr()) {
    pushLiteralSparseTensor(tensor);
    return;
  }

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
      case at::kPerChannelAffineFloatQParams:
      case at::kPerChannelAffine: {
        pushTensor(tensor.q_per_channel_scales());
        pushTensor(tensor.q_per_channel_zero_points());
        pushInt(tensor.q_per_channel_axis());
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

  if (!quantized) {
    // Only push it for regular tensor.
    auto math_bits = torch::jit::getTensorMathBits(tensor);
    // c10::Dict support only `int64_t` keys.
    c10::Dict<int64_t, bool> math_bits_dict;
    for (auto& pair : math_bits) {
      math_bits_dict.insert_or_assign(
          static_cast<int64_t>(pair.first), pair.second);
    }
    pushDict(math_bits_dict);
  }

  push<PickleOpCode>(PickleOpCode::TUPLE);

  // Call torch._utils._rebuild_tensor_v2
  push<PickleOpCode>(PickleOpCode::REDUCE);
}

void Pickler::pushSpecializedList(
    const IValue& ivalue,
    const char* list_name,
    const std::function<void(const IValue&)>& item_pusher) {
  pushGlobal("torch.jit._pickle", list_name);

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

static inline double swapDouble(double value) {
  const char* bytes = reinterpret_cast<const char*>(&value);
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  double flipped;
  char* out_bytes = reinterpret_cast<char*>(&flipped);
  for (const auto i : c10::irange(sizeof(double))) {
    out_bytes[i] = bytes[sizeof(double) - i - 1];
  }
  return *reinterpret_cast<double*>(out_bytes);
}

void Pickler::pushDouble(double value) {
  push<PickleOpCode>(PickleOpCode::BINFLOAT);
  // Python pickle format is big endian, swap.
  push<double>(swapDouble(value));
}
void Pickler::pushComplexDouble(const IValue& value) {
  c10::complex<double> d = value.toComplexDouble();
  pushGlobal("builtins", "complex");
  pushIValue(d.real());
  pushIValue(d.imag());
  push<PickleOpCode>(PickleOpCode::TUPLE2);
  push<PickleOpCode>(PickleOpCode::REDUCE);
}

void Pickler::pushLong(const std::string& data) {
  uint64_t size = data.size();

  TORCH_INTERNAL_ASSERT(
      size <= std::numeric_limits<uint8_t>::max(),
      "Cannot pickle a long larger than 255 bytes");
  push<PickleOpCode>(PickleOpCode::LONG1);
  push<uint8_t>(size);
  pushBytes(data);
}

void Pickler::pushTensorReference(const IValue& ivalue) {
  pushGlobal("torch.jit._pickle", "build_tensor_from_id");
  tensor_table_->push_back(ivalue.toTensor());
  int64_t tensor_id = tensor_table_->size() - 1;
  // Reduce arguments are spread (e.g. `*args`) before calling the global,
  // so wrap in a tuple
  push<PickleOpCode>(PickleOpCode::MARK);
  pushIValue(tensor_id);
  push<PickleOpCode>(PickleOpCode::TUPLE);

  push<PickleOpCode>(PickleOpCode::REDUCE);
}

// startTypeTag() and endTypeTag() must be called in a pair, with 1 argument
// pushed on the stack in between them. They will add the type of a container
// ivalue to the stack as a string so we can preserve type tags across
// serialization
void Pickler::startTypeTag() {
  if (tag_aggregates_) {
    pushGlobal("torch.jit._pickle", "restore_type_tag");
  }
}
namespace {
c10::optional<std::string> type_printer(const c10::Type& type) {
  if (auto dyn = type.castRaw<c10::DynamicType>()) {
    return dyn->fallback()->annotation_str(type_printer);
  }
  return c10::nullopt;
}
} // namespace

// See startTypeTag
void Pickler::endTypeTag(const IValue& ivalue) {
  if (!tag_aggregates_) {
    return;
  }
  TORCH_INTERNAL_ASSERT(ivalue.isGenericDict() || ivalue.isList());

  // Push the dict type
  TORCH_INTERNAL_ASSERT(ivalue.type());

  auto type = ivalue.type();
  auto annot_str = type->annotation_str(type_printer);
  pushString(annot_str);

  // Pop the dict and type into a tuple
  push<PickleOpCode>(PickleOpCode::TUPLE2);

  // Call function via reduce
  push<PickleOpCode>(PickleOpCode::REDUCE);
}

void Pickler::pushDict(const IValue& ivalue) {
  auto dict = ivalue.toGenericDict();

  startTypeTag();

  push<PickleOpCode>(PickleOpCode::EMPTY_DICT);

  static_assert(
      std::is_unsigned<decltype(dict.size())>::value,
      "Expected size to be non-negative.");
  push<PickleOpCode>(PickleOpCode::MARK);

  // Sort the dict for deterministic keys
  for (const auto& entry : dict) {
    pushIValue(entry.key());
    pushIValue(entry.value());
  }

  push<PickleOpCode>(PickleOpCode::SETITEMS);

  endTypeTag(ivalue);
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
  auto list = ivalue.toListRef();
  startTypeTag();

  // Push the list items
  push<PickleOpCode>(PickleOpCode::EMPTY_LIST);
  push<PickleOpCode>(PickleOpCode::MARK);
  for (const IValue& item : list) {
    pushIValue(item);
  }
  push<PickleOpCode>(PickleOpCode::APPENDS);

  endTypeTag(ivalue);
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

WriteableTensorData getWriteableTensorData(
    const at::Tensor& tensor,
    bool to_cpu) {
  WriteableTensorData result;
  result.tensor_ = tensor;
  result.size_ = tensor.storage().nbytes();
  // TODO HIP support
  if (tensor.storage().device_type() != DeviceType::CPU && to_cpu) {
    // NB: This new tensor is created to support cuda tensors.
    // Storages can be mutated when converting tensors from cuda to cpu,
    // and we need a cpu tensor to copy data from.
    result.tensor_ =
        at::empty({0}, tensor.options())
            .set_(
                tensor.storage(),
                /* storage_offset = */ 0,
                /* size = */
                {static_cast<int64_t>(
                    tensor.storage().nbytes() / tensor.element_size())},
                /* stride = */ {1})
            .cpu();
    TORCH_CHECK(
        result.tensor_.storage().nbytes() == result.size_,
        "Storage tensor size did not match record size");
  }
  return result;
}

bool checkHasValidSetGetState(const std::shared_ptr<c10::ClassType>& cls) {
  // Check that the schemas for __getstate__ and __setstate__ are correct
  auto getstate = cls->findMethod("__getstate__");
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
  auto setstate = cls->findMethod("__setstate__");
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
      set_schema.returns().at(0).type()->isSubtypeOf(*NoneType::get()),
      "'__setstate__' must return None, but found value of type",
      set_schema.returns().at(0).type()->annotation_str());

  // Check that the return type of __getstate__ matches the input to
  // __setstate__
  auto get_type = get_schema.returns().at(0).type();
  auto set_type = set_schema.arguments().at(1).type();

  TORCH_CHECK(
      get_type->isSubtypeOf(*set_type),
      "'__getstate__'s return type (",
      get_type->annotation_str(),
      ") does not match '__setstate__'s argument type (",
      set_type->annotation_str(),
      ")");

  return true;
}

} // namespace jit
} // namespace torch
