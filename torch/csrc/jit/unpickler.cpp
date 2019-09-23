#include <ATen/ATen.h>
#include <ATen/core/Dict.h>
#include <torch/csrc/jit/function.h>
#include <torch/csrc/jit/pickler.h>
#include "unpickler.h"
#include <string>

namespace torch {
namespace jit {

using ::c10::IValue;

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
      size_t start = marks_.back();
      auto list_ivalue = stack_.at(start - 1);
      readList(list_ivalue);
    } break;
    case PickleOpCode::LIST: {
      IValue list_ivalue = c10::impl::GenericList(AnyType::get());
      readList(list_ivalue);
      stack_.push_back(std::move(list_ivalue));
    } break;
    case PickleOpCode::DICT: {
      size_t start = marks_.back();
      marks_.pop_back();
      auto dict = c10::impl::GenericDict(AnyType::get(), AnyType::get());
      for (size_t i = start; i < stack_.size(); i += 2) {
        dict.insert_or_assign(stack_[i], stack_[i + 1]);
      }
      stack_.erase(stack_.begin() + start, stack_.end());
      stack_.push_back(std::move(dict));
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
              case at::kPerChannelAffine: {
                const auto& scales_list = qparams.at(1).toDoubleList();
                std::vector<double> scales(
                    scales_list.begin(), scales_list.end());
                const auto& zero_points_list = qparams.at(2).toIntList();
                std::vector<int64_t> zero_points(
                    zero_points_list.begin(), zero_points_list.end());
                const auto& axis_list = qparams.at(3).toIntList();
                std::vector<int64_t> axis(axis_list.begin(), axis_list.end());
                result = _empty_per_channel_affine_quantized(
                    {0},
                    at::tensor(scales),
                    at::tensor(zero_points),
                    axis,
                    storage_tensor.options());
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
          // elements[idx++] is empty backwards hooks
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
          return opcode;
        }

        c10::optional<at::QScheme> qscheme;
        for (int i = 0; i < at::COMPILE_TIME_NUM_QSCHEMES; ++i) {
          if (class_name == toString(static_cast<at::QScheme>(i))) {
            qscheme = static_cast<at::QScheme>(i);
          }
        }
        if (qscheme.has_value()) {
          stack_.emplace_back(int64_t(*qscheme));
          return opcode;
        }
        TORCH_CHECK(false, "class name not understood: torch.", class_name);
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
void Unpickler::readList(IValue list_ivalue) {
  size_t start = marks_.back();
  marks_.pop_back();
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

} // namespace jit
} // namespace torch
