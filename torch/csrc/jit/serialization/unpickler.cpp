#include <ATen/ATen.h>
#include <ATen/core/Dict.h>
#ifdef USE_RPC
#include <torch/csrc/distributed/rpc/rref_context.h>
#endif
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/mobile/type_parser.h>
#include <torch/csrc/jit/serialization/pickler.h>
#include <torch/csrc/jit/serialization/storage_context.h>
#include <torch/csrc/jit/serialization/unpickler.h>
#include <string>

namespace torch {
namespace jit {

using ::c10::IValue;

static void restoreAccurateTypeTagsIfPossible(const IValue& root) {
  if (root.isObject()) {
    restoreAccurateTypeTags(root, root.type());
  }
}

// Pickled objects are stored in a form compatible with Python pickling.
// In torchscript List[T]/Dict[K, V] are statically typed and contain
// dynamic type tags that allow T, K, and V to be recovered. But this
// info is not stored in the Python pickling information. However, we
// can recover this information from the static type of the top-level
// object being unpickled, because we have a record of the type of the
// objects it contains as attributes.
// `IfPossible` - we can only do this recovery when we have an object as
// the top-level unpickled thing (which is guaranteed for Modules, but
// not for torch.load/torch.save). Otherwise we do not know the types
// of the contained objects and cannot restore the tags.
void restoreAccurateTypeTags(const IValue& root, const TypePtr& type_tag) {
  struct Work {
    TypePtr type;
    IValue value;
  };
  std::vector<Work> to_process = {{type_tag, root}};
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
    auto kind = w.type->kind();
    if (auto dyn = w.type->castRaw<c10::DynamicType>()) {
      kind = dyn->dynamicKind();
    }
    switch (kind) {
      case TensorType::Kind:
      case StorageType::Kind:
      case NumberType::Kind:
      case FloatType::Kind:
      case ComplexType::Kind:
      case IntType::Kind:
      case NoneType::Kind:
      case GeneratorType::Kind:
      case QuantizerType::Kind:
      case BoolType::Kind:
      case VarType::Kind:
      case CapsuleType::Kind:
      case PyObjectType::Kind:
      case StringType::Kind:
      case FunctionType::Kind:
      case DeviceObjType::Kind:
      case StreamObjType::Kind:
      case QSchemeType::Kind:
      case LayoutType::Kind:
      case MemoryFormatType::Kind:
      case ScalarTypeType::Kind:
      case RRefType::Kind:
      case AnyType::Kind:
      case AnyListType::Kind:
      case AnyTupleType::Kind:
      case AnyClassType::Kind:
      case AnyEnumType::Kind:
        // no op, there is nothing to tag
        break;
      case c10::SymIntType::Kind:
        TORCH_CHECK(!w.value.toSymInt().is_symbolic());
        // no op, there is nothing to tag
        break;
      case c10::SymFloatType::Kind:
        TORCH_CHECK(!w.value.toSymFloat().is_symbolic());
        // no op, there is nothing to tag
        break;
      case DynamicType::Kind:
      case UnionType::Kind:
      case EnumType::Kind:
        // TODO(gmagogsfm): Implement serialization/deserialization of Enum.
        TORCH_INTERNAL_ASSERT(false);
      case TupleType::Kind: {
        auto t = w.value.toTuple();
        for (size_t i = 0; i < w.type->containedTypeSize(); ++i) {
          Work elem = {w.type->containedType(i), t->elements().at(i)};
          to_process.emplace_back(std::move(elem));
        }
      } break;
      case FutureType::Kind: {
        auto f = w.value.toFuture();
        if (f->completed()) {
          Work elem = {w.type->containedType(0), f->value()};
          to_process.emplace_back(std::move(elem));
        }
      } break;
      case OptionalType::Kind: {
        if (!w.value.isNone()) {
          Work elem = {w.type->containedType(0), w.value};
          to_process.emplace_back(std::move(elem));
        }
      } break;
      case ListType::Kind: {
        // specialized lists do not need their type refined, so we can exit
        // early here
        if (!w.value.isList()) {
          break;
        }
        auto elem_type = w.type->containedType(0);
        auto lst = w.value.toList();
        lst.unsafeSetElementType(elem_type);
        for (const IValue& item : lst) {
          Work elem = {elem_type, item};
          to_process.emplace_back(std::move(elem));
        }
      } break;
      case DictType::Kind: {
        auto d = w.value.toGenericDict();
        auto keyType = w.type->containedType(0);
        auto valType = w.type->containedType(1);
        d.unsafeSetKeyType(keyType);
        d.unsafeSetValueType(valType);
        for (const auto& item : d) {
          Work kelem = {keyType, item.key()};
          Work velem = {valType, item.value()};
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

namespace {
template <typename T>
bool is(const Type& type) {
  if (type.kind() == T::Kind) {
    return true;
  }
  if (auto dyn = type.castRaw<c10::DynamicType>()) {
    return dyn->tag() == c10::DynamicTypeTrait<T>::tagValue();
  }
  return false;
}
} // namespace

void restoreContainerTypeTags(const IValue& ivalue, const TypePtr& type) {
  if (is<DictType>(*type)) {
    auto dict = ivalue.toGenericDict();
    dict.unsafeSetKeyType(type->containedType(0));
    dict.unsafeSetValueType(type->containedType(1));
  } else if (is<ListType>(*type)) {
    ivalue.toList().unsafeSetElementType(type->containedType(0));
  } else {
    AT_ERROR("Unknown type for tag restoration: " + type->annotation_str());
  }
}

IValue Unpickler::parse_ivalue() {
  run();
  TORCH_CHECK(
      stack_.size() == 1,
      "Unpickler expected 1 element on the stack, but found ",
      stack_.size());
  if (version_ <= 2) {
    // See [type tag serialization]
    restoreAccurateTypeTagsIfPossible(stack_[0]);
  }
  return stack_[0];
}

double Unpickler::readFloat() {
  AT_ASSERT(sizeof(double) == 8);
  double big_endian = read<double>();
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
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
      " of pickle archive, found ",
      int(static_cast<uint8_t>(opcode)));
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
  a.emplace_back(std::forward<T>(e));
}
template <>
inline void append<bool>(std::vector<bool>& a, bool&& e) {
  a.push_back(e);
}

static std::vector<int64_t> tupleToIntList(const IValue& v) {
  return fmap(v.toTupleRef().elements(), [](const IValue& v) -> int64_t {
    return v.toInt();
  });
}

// note we cannot use toIntList, toDoubleList because during unpickling the
// lists are not yet tagged
template <typename T>
static std::vector<T> convertList(const IValue& v) {
  return fmap(v.toListRef(), [](const IValue& elem) { return elem.to<T>(); });
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
        empty_tuple_ = c10::ivalue::Tuple::create(std::vector<IValue>());
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
      TORCH_CHECK(!marks_.empty(), "Parsing error: marks_ is empty");
      size_t start = marks_.back();
      marks_.pop_back();
      std::vector<IValue> elements;
      const auto tupleSize = stack_.size() - start;
      switch (tupleSize) {
        case 3: {
          auto e3 = pop(stack_);
          auto e2 = pop(stack_);
          auto e1 = pop(stack_);
          stack_.emplace_back(c10::ivalue::Tuple::create(
              std::move(e1), std::move(e2), std::move(e3)));
          break;
        }
        case 2: {
          auto e2 = pop(stack_);
          auto e1 = pop(stack_);
          stack_.emplace_back(
              c10::ivalue::Tuple::create(std::move(e1), std::move(e2)));
          break;
        }
        case 1:
          stack_.emplace_back(c10::ivalue::Tuple::create(pop(stack_)));
          break;
        default: {
          elements.reserve(stack_.size() - start);
          auto start_it = stack_.begin() + start;
          for (auto it = start_it; it != stack_.end(); ++it) {
            elements.emplace_back(std::move(*it));
          }
          stack_.erase(start_it, stack_.end());
          stack_.emplace_back(c10::ivalue::Tuple::create(std::move(elements)));
          break;
        }
      }
    } break;
    case PickleOpCode::TUPLE1: {
      stack_.emplace_back(c10::ivalue::Tuple::create(pop(stack_)));
    } break;
    case PickleOpCode::TUPLE2: {
      auto e2 = pop(stack_);
      auto e1 = pop(stack_);
      stack_.emplace_back(
          c10::ivalue::Tuple::create(std::move(e1), std::move(e2)));
    } break;
    case PickleOpCode::TUPLE3: {
      auto e3 = pop(stack_);
      auto e2 = pop(stack_);
      auto e1 = pop(stack_);
      stack_.emplace_back(c10::ivalue::Tuple::create(
          std::move(e1), std::move(e2), std::move(e3)));
    } break;
    case PickleOpCode::EMPTY_DICT:
      stack_.emplace_back(
          c10::impl::GenericDict(AnyType::get(), AnyType::get()));
      break;
    case PickleOpCode::APPENDS: {
      TORCH_CHECK(!marks_.empty(), "Parsing error: marks_ is empty");
      size_t start = marks_.back();
      TORCH_CHECK(
          start > 0 && start <= stack_.size(),
          "Parsing error: wrong start index for stack_");
      auto list_ivalue = stack_.at(start - 1);
      readList(list_ivalue);
    } break;
    case PickleOpCode::LIST: {
      IValue list_ivalue = c10::impl::GenericList(AnyType::get());
      readList(list_ivalue);
      stack_.push_back(std::move(list_ivalue));
    } break;
    case PickleOpCode::DICT: {
      TORCH_CHECK(!marks_.empty(), "Parsing error: marks_ is empty");
      size_t start = marks_.back();
      marks_.pop_back();
      auto dict = c10::impl::GenericDict(AnyType::get(), AnyType::get());
      for (size_t i = start; i < stack_.size(); i += 2) {
        dict.insert_or_assign(stack_[i], stack_[i + 1]);
      }
      stack_.erase(stack_.begin() + start, stack_.end());
      stack_.emplace_back(std::move(dict));
    } break;
    case PickleOpCode::SETITEMS: {
      TORCH_CHECK(!marks_.empty(), "Parsing error: marks_ is empty");
      size_t start = marks_.back();
      marks_.pop_back();
      TORCH_CHECK(
          start > 0 && start <= stack_.size(),
          "Parsing error: wrong start index for stack_");
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
      readGlobal(module_name, class_name);
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
      TORCH_CHECK(
          idx < globals_.size(),
          "Parsing error: out of bounds access to globals_");
      globals_.at(idx)();
    } break;
    case PickleOpCode::BINPERSID: {
      auto tuple = pop(stack_).toTuple();
      const auto& args = tuple->elements();
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

      at::Storage storage;
      if (storage_context_ != nullptr && storage_context_->hasStorage(key)) {
        // for torch.package logic where storage may be loaded already
        storage = storage_context_->getStorage(key);
      } else {
        int64_t numel = args.at(4).toInt();
        caffe2::TypeMeta dtype = at::CPU(type).typeMeta();

        at::DataPtr storage_ptr;
        if (numel > 0) {
          // If there are no elements in the tensor, there's no point in
          // reading a zero (0) byte file from the input stream and paying
          // that cost.
          storage_ptr = read_record_(key);
        }

        storage = at::Storage(
            c10::Storage::use_byte_size_t(),
            numel * dtype.itemsize(),
            std::move(storage_ptr),
            /*allocator=*/nullptr,
            /*resizable=*/false); // NB: we didn't set any allocator for the
                                  // tensor
        if (storage_context_ != nullptr) {
          storage_context_->addStorage(key, storage);
        }
      }

      auto options = at::CPU(type).options();
      if (use_storage_device_) {
        options = options.device(storage.device());
        device = storage.device();
      }

      at::Tensor tensor;
      if (options.backend() == c10::Backend::QuantizedCPU) {
        tensor = at::_empty_affine_quantized({}, options, 0, 0)
                     .set_(storage, 0, {}, {});
      } else {
        tensor = at::empty({0}, options).set_(storage);
      }

      if (device.is_cuda() || device.is_xpu() || device.is_meta() ||
          device.is_hpu()) {
        tensor = tensor.to(device, tensor.scalar_type());
      } else if (device.type() != DeviceType::CPU) {
        AT_ERROR(
            "supported devices include CPU, CUDA and HPU, however got ",
            DeviceTypeName(device.type(), false));
      }
      stack_.emplace_back(std::move(tensor));
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

void Unpickler::readGlobal(
    const std::string& module_name,
    const std::string& class_name) {
  // TODO [unpickler refactor] __main__ isn't used by the pickler anymore, this
  // is only here for bc-compatibility reasons
  if (module_name == "__main__") {
    if (class_name == "TensorID") {
      globals_.emplace_back([this] {
        auto setitem_data = stack_.back();
        stack_.pop_back();
        TORCH_INTERNAL_ASSERT(
            !tensor_table_.empty(),
            "Pickler tried to write a tensor but had no tensor table to write to");
        stack_.emplace_back(tensor_table_.at(setitem_data.toInt()));
      });
    } else if (class_name == "IntList") {
      globals_.emplace_back([this] {
        stack_.back().toList().unsafeSetElementType(IntType::get());
      });
    } else {
      AT_ERROR("Unknown pickler class id", class_name);
    }
  } else if (module_name == "torch.jit._pickle") {
    if (class_name == "build_tensor_from_id") {
      globals_.emplace_back([this] {
        // Pop reduce arg off the stack
        auto data = stack_.back().toTupleRef().elements().at(0);
        stack_.pop_back();
        TORCH_CHECK(
            !tensor_table_.empty(),
            "Found a tensor table reference but Unpickler"
            " has no tensor table\n");
        stack_.emplace_back(tensor_table_.at(data.toInt()));
      });
    } else if (class_name == "restore_type_tag") {
      globals_.emplace_back([this] {
        auto tuple = stack_.back().toTuple();
        const auto& data = tuple->elements();
        auto type_str = data.at(1).toStringRef();
        stack_.pop_back();
        TypePtr type = nullptr;
        auto entry = type_cache_.find(type_str);
        if (entry != type_cache_.end()) {
          type = entry->second;
        } else {
          if (type_resolver_ == nullptr) {
            // If we haven't injected a custom way of retrieving types from
            // names, use a barebones type parser.
            type = type_parser_(type_str);
          } else {
            type = type_resolver_(type_str).type_;
          }
          type_cache_[type_str] = type;
        }
        // TODO: Use lookahead to avoid creating the tuple and immediately
        // destroying it here
        restoreContainerTypeTags(data.at(0), type);
        stack_.emplace_back(data.at(0));
      });
    } else {
      TypePtr elem_type = nullptr;
      if (class_name == "build_intlist") {
        elem_type = IntType::get();
      } else if (class_name == "build_tensorlist") {
        elem_type = TensorType::get();
      } else if (class_name == "build_doublelist") {
        elem_type = FloatType::get();
      } else if (class_name == "build_boollist") {
        elem_type = BoolType::get();
      } else {
        AT_ERROR("Unknown pickler class id ", class_name);
      }
      // Unpickle a list specialization (e.g. List[Tensor], List[int], ...)
      globals_.emplace_back([this, elem_type] {
        // Pop reduce arg off the stack
        auto data = stack_.back().toTupleRef().elements().at(0).toList();
        stack_.pop_back();
        data.unsafeSetElementType(elem_type);
        stack_.emplace_back(std::move(data));
      });
    }
  } else if (
      module_name == "torch._utils" &&
      (class_name == "_rebuild_tensor_v2" ||
       class_name == "_rebuild_qtensor")) {
    // Unpickle a tensor
    bool quantized = class_name == "_rebuild_qtensor";
    rebuildTensor(quantized);
  } else if (
      module_name == "torch._utils" && class_name == "_rebuild_sparse_tensor") {
    rebuildSparseTensor();
  } else if (module_name == "builtins" && class_name == "complex") {
    globals_.emplace_back([this] {
      auto tuple = pop(stack_).toTuple();
      const auto& elems = tuple->elements();
      AT_ASSERT(elems.size() == 2);
      auto complex =
          c10::complex<double>(elems.at(0).toDouble(), elems.at(1).toDouble());
      stack_.emplace_back(complex);
    });

  } else if (module_name == "collections" && class_name == "OrderedDict") {
    // collections.OrderedDict is used in tensor serialization for a tensor's
    // backward hooks (but they are not actually saved with this Pickler)
    globals_.emplace_back([this] {
      // drop the Tuple that was argument to OrderedDict, and replace it
      // with None OrderedDicts only appear in tensor deserialization and
      // their value is never used
      stack_.back() = IValue();
    });
  } else if (module_name == "torch" && class_name == "device") {
    globals_.emplace_back([this] {
      auto device_string = stack_.back().toTupleRef().elements().at(0);
      stack_.pop_back();
      stack_.emplace_back(c10::Device(device_string.toStringRef()));
    });
    stack_.emplace_back(int64_t(globals_.size() - 1));
    return;
  } else if (module_name == "torch.distributed.rpc" && class_name == "rref") {
#ifdef USE_RPC
    return rebuildRRef();
#else
    TORCH_INTERNAL_ASSERT(
        false,
        "RRef unpickling is only supported with the distributed package");
#endif
  } else if (module_name == "torch") {
    // Try to manually resolve several global enums
    // NOTE: this does not put a global into the global table,
    // like the other branches here because no REDUCE or BUILD will
    // be called on this value. Instead, we just put it on the stack
    // and return early
    c10::optional<c10::ScalarType> scalar_type;
#define CHECK_SCALAR(_, name)          \
  if (class_name == #name "Storage") { \
    scalar_type = c10::k##name;        \
  }
    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(CHECK_SCALAR)
#undef CHECK_SCALAR
    if (scalar_type.has_value()) {
      stack_.emplace_back(int64_t(*scalar_type));
      return;
    }

    c10::optional<at::QScheme> qscheme;
    for (int i = 0; i < at::COMPILE_TIME_NUM_QSCHEMES; ++i) {
      if (class_name == toString(static_cast<at::QScheme>(i))) {
        qscheme = static_cast<at::QScheme>(i);
      }
    }
    if (qscheme.has_value()) {
      stack_.emplace_back(int64_t(*qscheme));
      return;
    }
    TORCH_CHECK(
        false,
        "Unpickler found unknown torch global, 'torch.",
        class_name,
        "'");
  } else {
    TORCH_CHECK(
        type_resolver_,
        "Unpickler found unknown type ",
        module_name,
        ".",
        class_name);
    at::StrongTypePtr type =
        type_resolver_(c10::QualifiedName(module_name, class_name));
    if (auto enum_type = type.type_->cast<c10::EnumType>()) {
      globals_.emplace_back([this, enum_type] {
        auto val = stack_.back();
        stack_.pop_back();
        for (const auto& p : enum_type->enumNamesValues()) {
          if (p.second == val) {
            auto enum_holder = c10::make_intrusive<at::ivalue::EnumHolder>(
                enum_type, p.first, p.second);
            stack_.emplace_back(std::move(enum_holder));
            return;
          }
        }
      });
    } else {
      // Otherwise, global is a class/object type.
      globals_.emplace_back([this, type] {
        auto val = stack_.back();
        stack_.pop_back();
        auto obj = obj_loader_(type, val);
        stack_.emplace_back(std::move(obj));
      });
    }
  }
  stack_.emplace_back(int64_t(globals_.size() - 1));
}

void Unpickler::rebuildSparseTensor() {
  globals_.emplace_back([this] {
    auto tup = pop(stack_).toTuple();
    const auto& elements = tup->elements();
    size_t idx = 0;
    auto layout = elements.at(idx++).toInt();
    at::Tensor result;
    switch (layout) {
      case static_cast<int>(c10::Layout::Sparse): {
        std::vector<int64_t> size = tupleToIntList(elements.at(idx++));
        bool requires_grad = elements.at(idx++).toBool();
        auto& indices_tensor = elements.at(idx++).toTensor();
        auto& values_tensor = elements.at(idx++).toTensor();
        auto options = values_tensor.options()
                           .layout(c10::Layout::Sparse)
                           .requires_grad(requires_grad);
        result = at::_sparse_coo_tensor_unsafe(
            indices_tensor, values_tensor, size, options);
        result = autograd::make_variable(result, options.requires_grad());
        break;
      }
      case static_cast<int>(c10::Layout::SparseCsr): {
        std::vector<int64_t> size = tupleToIntList(elements.at(idx++));
        bool requires_grad = elements.at(idx++).toBool();
        auto& crow_indices = elements.at(idx++).toTensor();
        auto& col_indices = elements.at(idx++).toTensor();
        auto& values_tensor = elements.at(idx++).toTensor();
        auto options = values_tensor.options()
                           .layout(c10::Layout::SparseCsr)
                           .requires_grad(requires_grad);
        result = at::_sparse_csr_tensor_unsafe(
            crow_indices, col_indices, values_tensor, size, options);
        result =
            autograd::make_variable(std::move(result), options.requires_grad());
        break;
      }
      default:
        TORCH_CHECK(
            false,
            "Unsupported sparse tensor layout type in serialization ",
            static_cast<c10::Layout>(layout));
        break;
    }
    stack_.emplace_back(std::move(result));
  });
}

void Unpickler::rebuildTensor(bool quantized) {
  globals_.emplace_back([this, quantized] {
    auto tup = pop(stack_).toTuple();
    const auto& elements = tup->elements();
    size_t idx = 0;
    auto& storage_tensor = elements.at(idx++).toTensor();
    int64_t storage_offset = elements.at(idx++).toInt();
    std::vector<int64_t> size = tupleToIntList(elements.at(idx++));
    std::vector<int64_t> stride = tupleToIntList(elements.at(idx++));
    at::Tensor result;
    if (quantized) {
      auto qparams_tuple = elements.at(idx++).toTuple();
      const auto& qparams = qparams_tuple->elements();
      auto qscheme = static_cast<at::QScheme>(qparams.at(0).toInt());
      switch (qscheme) {
        case at::kPerTensorAffine: {
          double q_scale = qparams.at(1).toDouble();
          int64_t q_zero_point = qparams.at(2).toInt();
          result = at::_empty_affine_quantized(
              {0}, storage_tensor.options(), q_scale, q_zero_point);
        } break;
        case at::kPerChannelAffineFloatQParams:
        case at::kPerChannelAffine: {
          const auto& scales = qparams.at(1).toTensor();
          const auto& zero_points = qparams.at(2).toTensor();
          int64_t axis = qparams.at(3).toInt();
          result = at::_empty_per_channel_affine_quantized(
              {0}, scales, zero_points, axis, storage_tensor.options());
        } break;
        default:
          TORCH_CHECK(
              false,
              "Unsupported tensor quantization type in serialization ",
              toString(qscheme));
          break;
      }
    } else {
      result = at::empty({0}, storage_tensor.options());
    }
    bool requires_grad = elements.at(idx++).toBool();
    idx++; // backwards hooks is empty
    at::TensorImpl* impl = result.unsafeGetTensorImpl();
    impl->set_storage_keep_dtype(storage_tensor.storage());
    impl->set_storage_offset(storage_offset);
    impl->set_sizes_and_strides(size, stride);
    result = autograd::make_variable(result, requires_grad);

    // Handle if math_bits were pickled.
    // See `args` of _reduce_ex_internal
    // for a regular tensor (final else case).
    // Tensors pickled before this patch didn't
    // have this argument for storing MathBits,
    // in that case, we do nothing.
    // NOTE: `math_bits` is the 7th arg.
    // NOTE: This is only meant for regular tensor and not quantized
    //       which also has 7 args serialized.
    if (!quantized && elements.size() == 7) {
      auto math_bits = elements.at(idx++).toGenericDict();
      torch::jit::setTensorMathBits(result, math_bits);
    }

    stack_.emplace_back(std::move(result));
  });
}

#ifdef USE_RPC
void Unpickler::rebuildRRef() {
  globals_.emplace_back([this] {
    // It is the same as how rref is unpickled in python,
    // see PyRRef::unpickle
    auto tuple = std::move(stack_.back()).toTuple();
    const auto& args = tuple->elements();
    stack_.pop_back();
    TORCH_INTERNAL_ASSERT(
        args.size() == distributed::rpc::RFD_TUPLE_SIZE,
        "Pickled RRefForkData must contain 7 numbers.");
    auto ownerId =
        static_cast<int16_t>(args.at(distributed::rpc::OWNER_IDX).toInt());
    // const reference will extend the lifetime of the temporary variable
    const auto& rrefId = distributed::rpc::RRefId(
        static_cast<int16_t>(args.at(distributed::rpc::RREFID_ON_IDX).toInt()),
        static_cast<int64_t>(args.at(distributed::rpc::RREFID_ID_IDX).toInt()));
    const auto& forkId = distributed::rpc::RRefId(
        static_cast<int16_t>(args.at(distributed::rpc::FORKID_ON_IDX).toInt()),
        static_cast<int64_t>(args.at(distributed::rpc::FORKID_ID_IDX).toInt()));
    auto parent =
        static_cast<int16_t>(args.at(distributed::rpc::PARENT_IDX).toInt());
    const auto& typeStr = static_cast<std::string>(
        args.at(distributed::rpc::TYPE_IDX).toStringRef());
    auto rrefForkData = distributed::rpc::RRefForkData(
        ownerId, rrefId, forkId, parent, typeStr);
    auto& ctx = distributed::rpc::RRefContext::getInstance();
    c10::intrusive_ptr<distributed::rpc::RRef> rref;
    TORCH_INTERNAL_ASSERT(
        type_resolver_ != nullptr, "type_resolver_ is nullptr.");
    at::StrongTypePtr type = type_resolver_(c10::QualifiedName(typeStr));
    rref = ctx.getOrCreateRRef(rrefForkData, type.type_);
    ctx.notifyOwnerAndParentOfFork(
        rrefForkData.forkId_, rrefForkData.parent_, rref);
    stack_.emplace_back(
        c10::static_intrusive_pointer_cast<c10::RRefInterface>(rref));
  });
  stack_.emplace_back(int64_t(globals_.size() - 1));
  return;
}
#endif

void Unpickler::readSlowWithBuffer(char* dest, size_t sz) {
  // First, read any partial from buffer (may be 0).
  // We explicitly assume that sz > buffer_remaining_,
  // and that sz is never bigger than buffer_.size().
  AT_ASSERT(sz > buffer_remaining_);
  const size_t from_old_buf = buffer_remaining_;
  if (from_old_buf != 0) {
    memcpy(dest, buffer_.data() + buffer_pos_, from_old_buf);
  }
  const size_t needed = sz - from_old_buf;
  // Full read into the buffer. The calls here all explicitly
  // assume that one buffer will be enough for any sz.
  AT_ASSERT(sz <= buffer_.size());
  buffer_remaining_ = reader_(buffer_.data(), buffer_.size());
  if (buffer_remaining_ < needed) {
    AT_ERROR("Unexpected end of pickler archive.");
  }
  memcpy(dest + from_old_buf, buffer_.data(), needed);
  buffer_pos_ = needed; // assignment (0'ed from read)
  buffer_remaining_ -= needed;
}

// Read a number of bytes from the input stream
std::string Unpickler::readBytes(size_t length) {
  std::string data;
  static const size_t kSmallString = 64;
  if (length <= buffer_remaining_) {
    // Fast-path: entirely in buffer.
    data.assign(buffer_.data() + buffer_pos_, length);
    buffer_pos_ += length;
    buffer_remaining_ -= length;
  } else if (length <= kSmallString) {
    // If the string is smallish, do a full buffer read,
    // and read out of that buffer.
    data.resize(length);
    readSlowWithBuffer(&data[0], length);
  } else {
    // Otherwise, for larger strings, read what we can from
    // the buffer, and then read directly to the destination.
    const size_t from_old_buf = buffer_remaining_;
    if (from_old_buf != 0) {
      data.reserve(length);
      data.append(buffer_.data() + buffer_pos_, from_old_buf);
    }
    data.resize(length);
    const size_t needed = length - from_old_buf;
    size_t nread = reader_(&data[from_old_buf], needed);
    if (nread != needed) {
      AT_ERROR("Unexpected end of pickler archive.");
    }
    buffer_remaining_ = 0;
    // buffer_pos_ has no meaning with buffer_remaining_ == 0.
  }
  return data;
}

// Pop all the list items off of the stack and append them to the list at
// the corresponding MARK
void Unpickler::readList(IValue list_ivalue) {
  TORCH_CHECK(!marks_.empty(), "Parsing error: marks_ is empty");
  size_t start = marks_.back();
  marks_.pop_back();
  auto num_elements = stack_.size() - start;
  auto elements = c10::ArrayRef<IValue>(stack_).slice(start);
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
  } else if (list_ivalue.isList()) {
    auto list = std::move(list_ivalue).toList();
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
  std::string ss;
  while (true) {
    auto* const bufferStart = buffer_.data() + buffer_pos_;
    const auto bufferLeft = buffer_.size() - buffer_pos_;
    char* const newlinePtr =
        static_cast<char*>(memchr(bufferStart, '\n', bufferLeft));
    if (newlinePtr) {
      // read up to newline and we are done.
      auto const charsRead = newlinePtr - bufferStart;
      ss.append(bufferStart, charsRead);
      buffer_remaining_ -= charsRead + 1;
      buffer_pos_ += charsRead + 1;
      break;
    } else {
      // read whole buffer, refill
      for (const char* p = bufferStart; p < bufferStart + bufferLeft; ++p) {
        // Simple check just in case there is no terminating '\n'
        TORCH_CHECK(
            is_valid_python_id_char(*p),
            "Found character '",
            int(uint8_t(*p)),
            "' in string, ",
            "strings must be qualified Python identifiers");
      }
      ss.append(bufferStart, bufferLeft);
      buffer_remaining_ = reader_(buffer_.data(), buffer_.size());
      buffer_pos_ = 0;
    }
  }
  return ss;
}

} // namespace jit
} // namespace torch
