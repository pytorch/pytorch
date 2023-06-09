#include <ATen/core/Dict.h>
#include <ATen/core/Formatting.h>
#include <ATen/core/class_type.h>
#include <ATen/core/enum_type.h>
#include <ATen/core/function.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/stack.h>
#include <ATen/core/type_factory.h>
#include <c10/util/StringUtil.h>
#include <c10/util/hash.h>
#include <c10/util/irange.h>
#include <cmath>
#include <iostream>
#include <utility>

namespace c10 {
bool _fastEqualsForContainer(const IValue& lhs, const IValue& rhs) {
  if (lhs.is(rhs)) {
    // Like Python, for containers we consider identity equality to be
    // sufficient but not necessary for value equality
    return true;
  }
  return lhs == rhs;
}

namespace ivalue {

// This is in ivalue.cpp because we need to access Type::annotation_str, which
// is declared in jit_type.h
void checkCustomClassType(const ClassType* expected_type, const Type* actual_type) {
  // NB: doing pointer comparison here
  // If in the future there ever arises a need to call operator== on custom class
  // Type's, this needs to be changed!
  TORCH_CHECK(actual_type == static_cast<const Type*>(expected_type),
              "Tried to convert an IValue of type ",
              actual_type ? actual_type->repr_str() : std::string("*NULL*"),
              " to custom class type ",
              expected_type ? expected_type->repr_str() : std::string("*NULL*"));
}

TORCH_API c10::intrusive_ptr<ConstantString> ConstantString::create(
    std::string str_) {
  return c10::make_intrusive<ConstantString>(std::move(str_));
}

TORCH_API c10::intrusive_ptr<ConstantString> ConstantString::create(
    c10::string_view str_) {
  return c10::make_intrusive<ConstantString>(std::string(str_));
}

TORCH_API c10::intrusive_ptr<ConstantString> ConstantString::create(
    const char* str_) {
  return c10::make_intrusive<ConstantString>(std::string(str_));
}

bool operator==(const ivalue::Tuple& lhs, const ivalue::Tuple& rhs) {
  return lhs.size() == rhs.size() &&
      // see [container equality]
      std::equal(
             lhs.elements().cbegin(),
             lhs.elements().cend(),
             rhs.elements().cbegin(),
             _fastEqualsForContainer);
}

bool operator==(const ivalue::EnumHolder& lhs, const ivalue::EnumHolder& rhs) {
  return lhs.name() == rhs.name() && *rhs.type() == *lhs.type();
}

const std::string ivalue::EnumHolder::qualifiedClassName() const {
  return type_->qualifiedClassName().qualifiedName();
}

const std::string ivalue::EnumHolder::unqualifiedClassName() const {
  return type_->qualifiedClassName().name();
}

} // namespace ivalue

c10::TypePtr IValue::TagType<c10::Type>::get(const IValue& v) {
  switch (v.tag) {
      case Tag::None:
        return NoneType::get();
      case Tag::Tensor:
        return TensorType::create(v.toTensor());
      case Tag::Storage:
        return StorageType::get();
      case Tag::Double:
        return FloatType::get();
      case Tag::ComplexDouble:
        return ComplexType::get();
      case Tag::Int:
        return IntType::get();
      case Tag::SymInt:
        return c10::SymIntType::get();
      case Tag::SymFloat:
        return c10::SymFloatType::get();
      case Tag::SymBool:
        return c10::SymBoolType::get();
      case Tag::Bool:
        return BoolType::get();
      case Tag::String:
        return StringType::get();
      case Tag::Blob:
        return AnyType::get();
      case Tag::GenericDict: {
        auto d = v.toGenericDict();
        return DictType::create(d.keyType(), d.valueType());
      }
      case Tag::GenericList:
        return ListType::create(v.toList().elementType());
      case Tag::Await:
        return AwaitType::create(v.toAwait()->elementType());
      case Tag::Future:
        return FutureType::create(v.toFuture()->elementType());
      case Tag::RRef:
        return RRefType::create(v.toRRef()->type());
      case Tag::Device:
        return DeviceObjType::get();
      case Tag::Stream:
        return StreamObjType::get();
      case Tag::Object:
        return v.toObjectRef().type();
      case Tag::PyObject:
        return PyObjectType::get();
      case Tag::Uninitialized:
        return AnyType::get();
      case Tag::Capsule:
        return CapsuleType::get();
      case Tag::Tuple:
        return v.toTupleRef().type();
      case Tag::Generator:
        return GeneratorType::get();
      case Tag::Quantizer:
        return QuantizerType::get();
      case Tag::Enum:
        return v.toEnumHolder()->type();
  }
  // switch above is complete but this silences compiler warnings
  TORCH_INTERNAL_ASSERT(false, "unhandled case in IValue::type()");
}

void IValue::visit(const std::function<bool (const IValue &)>& visitor) const {
  if (visitor(*this)) {
    // Shortcut
    return;
  }
  switch (this->tag) {
    case Tag::Tuple:
    case Tag::GenericList: {
      c10::ArrayRef<IValue> elems;
      if (isTuple()) {
        elems = this->toTupleRef().elements();
      } else {
        elems = this->toListRef();
      }
      for (auto& elem : elems) {
        elem.visit(visitor);
      }
      break;
    }
    case Tag::GenericDict:
      for (const auto& pair : this->toGenericDict()) {
        pair.value().visit(visitor);
        pair.key().visit(visitor);
      }
      break;
    case Tag::Object: {
      auto obj_type = type()->expect<ClassType>();
      auto obj_value = toObject();
      auto attributes = obj_type->getAttributes();
      for (const auto& attr: attributes) {
        auto attribute = obj_value->getAttr(attr.getName());
        attribute.visit(visitor);
      }
      break;
    }
    case Tag::PyObject: {
      c10::intrusive_ptr<at::ivalue::PyObjectHolder> py_obj = toPyObjectHolder();
      auto match = py_obj->tryToInferType();
      if (match.success()) {
        auto contained_value = py_obj->toIValue(match.type());
        contained_value.visit(visitor);
      }
      break;
    }
    default:
      break;
 }
}

void IValue::getSubValues(HashAliasedIValues& subValues) const {
  switch (this->tag) {
    case Tag::Tensor:
      subValues.insert(*this);
      return;
    case Tag::Tuple:
    case Tag::GenericList: {
      subValues.insert(*this);
      c10::ArrayRef<IValue> elems;
      if (isTuple()) {
        elems = this->toTupleRef().elements();
      } else {
        elems = this->toListRef();
      }
      for (auto& elem : elems) {
        elem.getSubValues(subValues);
      }
      break;
    }
    case Tag::GenericDict:
      subValues.insert(*this);
      for (const auto& pair : this->toGenericDict()) {
        pair.value().getSubValues(subValues);
        pair.key().getSubValues(subValues);
      }
      break;
    case Tag::Object: {
      // Record Object IValue and its attributes.
      subValues.insert(*this);
      auto obj_type = type()->expect<ClassType>();
      auto obj_value = toObject();
      auto attributes = obj_type->getAttributes();
      for (const auto& attr: attributes) {
        auto attribute = obj_value->getAttr(attr.getName());
        attribute.getSubValues(subValues);
      }
      break;
    }
    case Tag::PyObject: {
      subValues.insert(*this);
      c10::intrusive_ptr<at::ivalue::PyObjectHolder> py_obj = toPyObjectHolder();
      auto match = py_obj->tryToInferType();
      TORCH_CHECK_TYPE(match.success(),
            "Cannot infer type of ", py_obj->toStr(), ": ", match.reason());
      auto contained_value = py_obj->toIValue(match.type());
      contained_value.getSubValues(subValues);
      break;
    }
    case Tag::Future:
    case Tag::Await:
    case Tag::Device:
    case Tag::Uninitialized:
    case Tag::Capsule:
      TORCH_CHECK_TYPE(
          false, "Cannot inspect value of type ", this->tagKind());
      [[fallthrough]];
    default:
      // don't record scalars.
      break;
  }
}

bool IValue::overlaps(const IValue& rhs) const {
  HashAliasedIValues rhsSubValues, thisSubValues;
  rhs.getSubValues(rhsSubValues);
  getSubValues(thisSubValues);
  for (auto& sub : thisSubValues) {
    if (rhsSubValues.count(sub)) {
      return true;
    }
  }
  return false;
}

bool operator!=(const IValue& lhs, const IValue& rhs) {
  return !(lhs == rhs);
}

bool operator==(const IValue& lhs, const IValue& rhs) {
  IValue eq = lhs.equals(rhs);
  if (eq.isBool()) {
    return eq.toBool();
  }
  // The only case we don't return bool is for tensor comparison. In Python,
  // `bool()` is called on the return value of `__eq__` if the return value is
  // not a boolean. Mimic that behavior here.
  TORCH_INTERNAL_ASSERT(eq.isTensor());
  return eq.toTensor().is_nonzero();
}

bool IValue::ptrEqual(const IValue& lhs, const IValue& rhs) {
  TORCH_INTERNAL_ASSERT(lhs.isIntrusivePtr());
  TORCH_INTERNAL_ASSERT(rhs.isIntrusivePtr());
  return lhs.tag == rhs.tag &&
      lhs.payload.u.as_intrusive_ptr == rhs.payload.u.as_intrusive_ptr;
}

IValue IValue::equals(const IValue& rhs) const {
  const IValue& lhs = *this;
  switch (lhs.tag) {
    case Tag::None:
      // In Python you're not supposed to do this comparison apparently. Not
      // sure if we should warn here or what
      return rhs.isNone();
    case Tag::Tensor: {
      if (!rhs.isTensor()) {
        return false;
      }
      return lhs.toTensor().eq(rhs.toTensor());
    }
    case Tag::Storage:
      return rhs.isStorage() && lhs.toStorage().unsafeGetStorageImpl() == rhs.toStorage().unsafeGetStorageImpl();
    case Tag::Double:
      return rhs.isDouble() && lhs.toDouble() == rhs.toDouble();
    case Tag::ComplexDouble:
      return rhs.isComplexDouble() && lhs.toComplexDouble() == rhs.toComplexDouble();
    case Tag::Int:
      return rhs.isInt() && lhs.toInt() == rhs.toInt();
    case Tag::SymInt:
      return rhs.isSymInt() && lhs.toSymInt() == rhs.toSymInt();
    case Tag::SymFloat:
      return rhs.isSymFloat() && lhs.toSymFloat() == rhs.toSymFloat();
    case Tag::SymBool:
      return rhs.isSymBool() && lhs.toSymBool() == rhs.toSymBool();
    case Tag::Bool:
      return rhs.isBool() && lhs.toBool() == rhs.toBool();
    case Tag::String:
      return rhs.isString() && lhs.toStringRef() == rhs.toStringRef();
    case Tag::GenericDict:
      return rhs.isGenericDict() && lhs.toGenericDict() == rhs.toGenericDict();
    case Tag::Tuple:
      return rhs.isTuple() && *lhs.toTuple() == *rhs.toTuple();
    case Tag::Stream:
      return rhs.isStream() && lhs.toStream() == rhs.toStream();
    case Tag::Device:
      return rhs.isDevice() && lhs.toDevice() == rhs.toDevice();
    case Tag::GenericList:
      return rhs.isList() && lhs.toList() == rhs.toList();
    case Tag::Blob:
    case Tag::Future:
    case Tag::Await:
    case Tag::RRef:
    case Tag::Object:
    case Tag::PyObject:
    case Tag::Capsule:
    case Tag::Generator:
    case Tag::Quantizer:
      return ptrEqual(lhs, rhs);
    case Tag::Enum:
      return lhs.toEnumHolder()->is(*rhs.toEnumHolder());
    case Tag::Uninitialized:
      // Unitialized ivalues show up in no-ops when the compiler can prove a
      // value will never be used. Just return false on any equality comparison.
      return false;
  }
  // the above switch should be exhaustive
  TORCH_INTERNAL_ASSERT(false, "we should never reach here")
}

size_t IValue::hash(const IValue& v) {
  switch (v.tag) {
    case Tag::None:
      return 0;
    case Tag::Bool:
      return c10::get_hash(v.payload.u.as_bool);
    case Tag::Double:
      return c10::get_hash(v.payload.u.as_double);
    case Tag::Tensor:
      // Tensor __hash__ is equivalent to `id()`, so take the pointer value of
      // the tensor to emulate it
      return c10::get_hash(v.payload.as_tensor.unsafeGetTensorImpl());
    // NOLINTNEXTLINE(bugprone-branch-clone)
    case Tag::Storage:
      return c10::get_hash(v.payload.u.as_int);
    case Tag::Int:
      return c10::get_hash(v.payload.u.as_int);
    // NB: these are technically strict aliasing violations
    case Tag::SymInt:
      return c10::get_hash(v.payload.u.as_int);
    case Tag::SymFloat:
      return c10::get_hash(v.payload.u.as_int);
    case Tag::SymBool:
      return c10::get_hash(v.payload.u.as_int);
    case Tag::String:
      return c10::get_hash(v.toStringRef());
    case Tag::Tuple:
      return c10::get_hash(*v.toTuple());
    case Tag::Device:
      return c10::get_hash(v.toDevice());
    case Tag::GenericDict:
    case Tag::GenericList:
    case Tag::Blob:
    case Tag::Future:
    case Tag::Await:
    case Tag::RRef:
    case Tag::Object:
    case Tag::PyObject:
    case Tag::Capsule:
    case Tag::Generator:
    case Tag::Quantizer:
    case Tag::ComplexDouble:
    case Tag::Enum:
    case Tag::Stream:
    case Tag::Uninitialized:
      throw std::runtime_error(
          "unhashable type: '" + v.type()->repr_str() + "'");
  }
  // the above switch should be exhaustive
  TORCH_INTERNAL_ASSERT(false, "we should never reach here")
}

static bool isUndefinedTensor(const IValue& iv) {
  return iv.isTensor() && !iv.toTensor().defined();
}

bool IValue::is(const IValue& rhs) const {
  const IValue& lhs = *this;
  // Special handling for undefined tensors:
  // 1. Undefined_tensor is None and vice versa.
  if ((isUndefinedTensor(lhs) && rhs.isNone()) ||
      (lhs.isNone() && isUndefinedTensor(rhs))) {
    return true;
  }
  // 2. Undefined_tensor is Undefined_tensor.
  if (isUndefinedTensor(lhs) && isUndefinedTensor(rhs)) {
    return true;
  }

  if (lhs.isTensor()) {
    // Use the standard way of comparing two tensors for identity
    return rhs.isTensor() && lhs.toTensor().is_same(rhs.toTensor());
  }

  if (lhs.isIntrusivePtr()) {
    return rhs.isIntrusivePtr() && ptrEqual(lhs, rhs);
  }
  return lhs == rhs;
}

template <typename T>
inline bool IValue::isListOf() const {
  // note: avoids calling type() to avoid extra referencing counting for the returned type.
  if (!isList()) {
    return false;
  }
  const auto& ty = static_cast<detail::ListImpl*>(payload.u.as_intrusive_ptr)->elementType;
  if (ty->kind() == T::Kind) {
    return true;
  }
  return *ty == *TypeFactory::get<T>();
}

bool IValue::isDoubleList() const {
  return isListOf<c10::FloatType>();
}

bool IValue::isComplexDoubleList() const {
  return isListOf<c10::ComplexType>();
}

bool IValue::isTensorList() const {
  return isListOf<c10::TensorType>();
}

bool IValue::isOptionalTensorList() const {
  if (!isList()) {
    return false;
  }
  const auto& ty = static_cast<detail::ListImpl*>(payload.u.as_intrusive_ptr)->elementType;
  const auto& expected_ty = c10::getTypePtr<c10::optional<at::Tensor>>();
  return expected_ty == ty;
}

bool IValue::isIntList() const {
  return isListOf<c10::IntType>();
}

bool IValue::isSymIntList() const {
  return isListOf<c10::SymIntType>();
}

bool IValue::isBoolList() const {
  return isListOf<c10::BoolType>();
}

namespace {

using IValueFormatter = std::function<void(std::ostream&, const IValue&)>;

template <class T>
std::ostream& printList(
    std::ostream& out,
    const T& list,
    const std::string start,
    const std::string finish,
    IValueFormatter formatter) {
  out << start;
  for (const auto i : c10::irange(list.size())) {
    if (i > 0) {
      out << ", ";
    }
    formatter(out, IValue(list[i]));
  }
  out << finish;
  return out;
}

// Properly disambiguate the type of an empty list
std::ostream& printMaybeAnnotatedList(
    std::ostream& out,
    const IValue& the_list,
    IValueFormatter formatter) {
  auto list_elem_type = the_list.type()->containedType(0);
  if (the_list.toListRef().empty() ||
      !elementTypeCanBeInferredFromMembers(list_elem_type)) {
    out << "annotate(" << the_list.type<c10::Type>()->annotation_str() << ", ";
    printList(out, the_list.toListRef(), "[", "]", std::move(formatter));
    out << ")";
    return out;
  } else {
    return printList(out, the_list.toListRef(), "[", "]", std::move(formatter));
  }
}

template <typename Dict>
std::ostream& printDict(
    std::ostream& out,
    const Dict& v,
    IValueFormatter formatter) {
  out << "{";

  bool first = true;
  for (const auto& pair : v) {
    if (!first) {
      out << ", ";
    }

    formatter(out, pair.key());
    out << ": ";
    formatter(out, pair.value());
    first = false;
  }

  out << "}";
  return out;
}
}

// Properly disambiguate the type of an empty dict
static std::ostream& printMaybeAnnotatedDict(
    std::ostream& out,
    const IValue& the_dict,
    IValueFormatter formatter) {
  auto value_type = the_dict.type()->castRaw<DictType>()->getValueType();
  if (the_dict.toGenericDict().empty() ||
      !elementTypeCanBeInferredFromMembers(value_type)) {
    out << "annotate(" << the_dict.type<c10::Type>()->annotation_str() << ",";
    printDict(out, the_dict.toGenericDict(), std::move(formatter)) << ")";
  } else {
    return printDict(out, the_dict.toGenericDict(), std::move(formatter));
  }
  return out;
}

static std::ostream& printComplex(std::ostream & out, const IValue & v) {
  c10::complex<double> d = v.toComplexDouble();
  IValue real(d.real()), imag(std::abs(d.imag()));
  auto sign = "";
  if (d.imag() >= 0) {
    sign = "+";
  } else {
    sign = "-";
  }
  return out << real << sign << imag << "j";
}

std::ostream& IValue::repr(
    std::ostream& out,
    std::function<bool(std::ostream&, const IValue& v)>
        customFormatter) const {
  // First check if the caller has provided a custom formatter. Use that if possible.
  if (customFormatter(out, *this)) {
    return out;
  }

  const IValue& v = *this;
  // continue to use custom formatter in recursion
  auto formatter = [&](std::ostream& out, const IValue& input) {
    input.repr(out, customFormatter);
  };
  switch (v.tag) {
    case IValue::Tag::None:
      return out << v.toNone();
    case IValue::Tag::Double: {
      double d = v.toDouble();
      int c = std::fpclassify(d);
      if ((c == FP_NORMAL || c == FP_ZERO ) && std::abs(d) < 1e10) {
        int64_t i = int64_t(d);
        if (double(i) == d) {
          // -0.0 (signed zero) needs to be parsed as -0.
          if (i == 0 && std::signbit(d)) {
            return out << "-" << i << ".";
          }
          return out << i << ".";
        }
      }
      auto orig_prec = out.precision();
      return out << std::setprecision(std::numeric_limits<double>::max_digits10)
                 << d << std::setprecision(orig_prec);
    }
    case IValue::Tag::ComplexDouble: {
      return printComplex(out, v);
    }
    case IValue::Tag::Int:
      return out << v.toInt();
    case IValue::Tag::SymInt:
      return out << v.toSymInt();
    case IValue::Tag::SymFloat:
      return out << v.toSymFloat();
    case IValue::Tag::SymBool:
      return out << v.toSymBool();
    case IValue::Tag::Bool:
      return out << (v.toBool() ? "True" : "False");
    case IValue::Tag::Tuple: {
      const auto& elements = v.toTupleRef().elements();
      const auto& finish = elements.size() == 1 ? ",)" : ")";
      return printList(out, elements, "(", finish, formatter);
    }
    case IValue::Tag::String:
      c10::printQuotedString(out, v.toStringRef());
      return out;
    case IValue::Tag::GenericList: {
      return printMaybeAnnotatedList(out, *this, formatter);
    }
    case IValue::Tag::Device: {
      std::stringstream device_stream;
      device_stream << v.toDevice();
      out << "torch.device(";
      c10::printQuotedString(out, device_stream.str());
      return out << ")";
    }
    case IValue::Tag::GenericDict:
      return printMaybeAnnotatedDict(out, v, formatter);
    case IValue::Tag::Enum: {
      auto enum_holder = v.toEnumHolder();
      return out << enum_holder->qualifiedClassName() << "." <<
          enum_holder->name();
    }
    case IValue::Tag::Object: {
      TORCH_INTERNAL_ASSERT(false, "repr() not defined on: ", v.tagKind(), ". Perhaps you've frozen a module with custom classes?");
    }
    default:
      TORCH_INTERNAL_ASSERT(false, "repr() not defined on: ", v.tagKind());
  }
}

static bool simpleClassTypeArg(const Argument& arg, const ClassTypePtr& type) {
  return arg.type() == type && !arg.kwarg_only() && !arg.default_value();
}

torch::jit::Function* checkObjectSortSchema(const c10::ClassTypePtr& t, std::stringstream& why_not) {
  if (auto method = t->findMethod("__lt__")) {
      const auto& lt_schema = method->getSchema();
      const auto& schema_args = lt_schema.arguments();
      bool error =
          (schema_args.size() != 2 ||
           !simpleClassTypeArg(schema_args[0], t) ||
           !simpleClassTypeArg(schema_args[1], t) ||
           lt_schema.returns().size() != 1 ||
           lt_schema.returns()[0].type() != BoolType::get());
      if (!error) {
        return method;
      }
    }

    why_not << "To sort a list of " << t->repr_str()
            << " it must define a "
            << "__lt__ method with two inputs of type "
            << t->repr_str() << " that "
            << "returns a bool";
    return nullptr;
}

IValueComparator getLessThanComparator(const IValue& v) {
  if (v.isTensor()) {
      return [](const IValue& a, const IValue& b) {
        return a.toTensor().lt(b.toTensor()).is_nonzero();
      };
  }

  if (v.isDouble()) {
      return [](const IValue& a, const IValue& b) {
        return a.toDouble() < b.toDouble();
      };
  }

  if (v.isInt()) {
      return [](const IValue& a, const IValue& b) {
        return a.toInt() < b.toInt();
      };
  }

  if (v.isBool()) {
      return [](const IValue& a, const IValue& b) {
        return a.toBool() == false && b.toBool() == true;
      };
  }

  if (v.isString()) {
      return [](const IValue& a, const IValue& b) {
       return a.toStringRef() < b.toStringRef();
      };
  }

  if (v.isTuple()) {
      const auto& elements = v.toTupleRef().elements();
      size_t n = elements.size();

      std::vector<IValueComparator> elements_lts;
      elements_lts.reserve(n);
      for (const auto i : c10::irange(n)) {
        elements_lts.push_back(getLessThanComparator(elements[i]));
      }

      return [elements_lts=std::move(elements_lts), n](const IValue& a, const IValue& b) {
        const auto& a_elements = a.toTupleRef().elements();
        const auto& b_elements = b.toTupleRef().elements();

        for (const auto i : c10::irange(n)) {
          if (elements_lts[i](a_elements[i], b_elements[i])) {
            return true;
          }
          if (a_elements[i] == b_elements[i]) {
            continue;
          }
          return false;
        }
        // Reaching here means two tuples are equal.
        return false;
      };
  }

  if (v.isObject()) {
    std::stringstream why_not;
    torch::jit::Function* lt_func =
        checkObjectSortSchema(v.type()->expect<ClassType>(), why_not);
    if (!lt_func) {
      AT_ERROR(why_not.str());
    }

    return [lt_func](const IValue& a, const IValue& b) {
      // Quick pass to satisfy "strict weak ordering" requirement
      if (a.is(b)) {
        return false;
      }
      torch::jit::Stack sort_stack;
      sort_stack.push_back(a);
      sort_stack.push_back(b);
      lt_func->run(sort_stack);
      return torch::jit::pop(sort_stack).toBool();
    };
  }

  AT_ERROR("IValues of type: ", v.tagKind(), " are not comparable");
}

IValueComparator getGreaterThanComparator(const IValue& v) {
  auto lt = getLessThanComparator(v);
  return [lt = std::move(lt)](const IValue& a, const IValue& b) {
    return lt(b, a);  // gt(a, b) === lt(b, a)
  };
}

std::ostream& operator<<(std::ostream& out, const ivalue::EnumHolder& v) {
  out << v.qualifiedClassName() << "." << v.name();
  return out;
}

std::ostream& operator<<(std::ostream & out, const IValue & v) {
  auto formatter = [&](std::ostream& out, const IValue& v) {
    out << v;
  };
  switch(v.tag) {
    case IValue::Tag::None:
      return out << v.toNone();
    case IValue::Tag::Tensor:
      return out << v.toTensor();
    case IValue::Tag::Storage:
      return out << v.toStorage().unsafeGetStorageImpl();
    case IValue::Tag::Double: {
      double d = v.toDouble();
      int c = std::fpclassify(d);
      if (c == FP_NORMAL || c == FP_ZERO) {
        int64_t i = int64_t(d);
        if (double(i) == d) {
          return out << i << ".";
        }
      }
      auto orig_prec = out.precision();
      return out
        << std::setprecision(std::numeric_limits<double>::max_digits10)
        << v.toDouble()
        << std::setprecision(orig_prec);
    } case IValue::Tag::ComplexDouble: {
      return printComplex(out, v);
    } case IValue::Tag::Int:
      return out << v.toInt();
    case IValue::Tag::SymInt:
      return out << v.toSymInt();
    case IValue::Tag::SymFloat:
      return out << v.toSymFloat();
    case IValue::Tag::SymBool:
      return out << v.toSymBool();
    case IValue::Tag::Bool:
      return out << (v.toBool() ? "True" : "False");
    case IValue::Tag::Tuple: {
      const auto& elements = v.toTupleRef().elements();
      const auto& finish = elements.size() == 1 ? ",)" : ")";
      return printList(out, elements, "(", finish, formatter);
    }
    case IValue::Tag::String:
      return out << v.toStringRef();
    case IValue::Tag::Blob:
      return out << *v.toBlob();
    case IValue::Tag::Capsule:
      return out << "Capsule";
    case IValue::Tag::GenericList:
      return printList(out, v.toList(), "[", "]", formatter);
    case IValue::Tag::RRef:
      return out << "RRef";
    case IValue::Tag::Future:
      return out << "Future";
    case IValue::Tag::Await:
      return out << "Await";
    case IValue::Tag::Uninitialized:
      return out << "Uninitialized";
    case IValue::Tag::Device:
      return out << v.toDevice();
    case IValue::Tag::Stream:
      return out << v.toStream();
    case IValue::Tag::GenericDict:
      return printDict(out, v.toGenericDict(), formatter);
    case IValue::Tag::PyObject: {
      auto py_obj = v.toPyObject();
      return out << "<PyObject at" << py_obj << ">";
    }
    case IValue::Tag::Generator:
      return out << "Generator";
    case IValue::Tag::Quantizer:
      return out << "Quantizer";
    case IValue::Tag::Object: {
      // TODO we should attempt to call __str__ if the object defines it.
      auto obj = v.toObject();
      // print this out the way python would do it
      return out << "<" << obj->name() << " object at " << obj.get() << ">";
    }
    case IValue::Tag::Enum: {
      auto enum_holder = v.toEnumHolder();
      return out << "Enum<" << enum_holder->unqualifiedClassName() << "." <<
          enum_holder->name() << ">";
    }

  }
  AT_ERROR("Tag not found: ", v.tagKind());
}

#undef TORCH_FORALL_TAGS

void IValue::dump() const {
  std::cout << *this << "\n";
}

std::shared_ptr<ClassType> ivalue::Object::type() const {
  return type_.type_->expect<ClassType>();
}

c10::intrusive_ptr<ivalue::Object> ivalue::Object::create(
    ClassTypePtr classType, size_t numSlots) {
  return ivalue::Object::create(
      StrongTypePtr(nullptr, std::move(classType)), numSlots);
}


IValue IValue::deepcopy() const {
  IValue::HashAliasedIValueMap memo;
  return deepcopy(memo);
}

IValue IValue::deepcopy(
    IValue::HashAliasedIValueMap& memo) const {
  if (memo.count(*this)) {
    return memo.at(*this);
  }
  IValue copy;
  switch(tag) {
    case IValue::Tag::Tensor:
      copy = IValue(toTensor().clone());
      break;
    case IValue::Tag::Tuple: {
      std::vector<IValue> copied_tuple;
      for (const auto& e : toTupleRef().elements()) {
        copied_tuple.emplace_back(e.deepcopy(memo));
      }
      copy = IValue(ivalue::Tuple::create(std::move(copied_tuple)));
    }
      break;
    case IValue::Tag::GenericList: {
      auto list = toList();
      auto copied_list = c10::impl::GenericList(list.elementType());
      for (IValue v : list) {
        copied_list.push_back(v.deepcopy(memo));
      }
      copy = IValue(copied_list);
    }
      break;
    case IValue::Tag::GenericDict: {
      auto dict = toGenericDict();
      auto copied_dict = c10::impl::GenericDict(dict.keyType(), dict.valueType());
      for (const auto& entry : dict) {
        copied_dict.insert(entry.key().deepcopy(memo), entry.value().deepcopy(memo));
      }
      copy = IValue(copied_dict);
    }
      break;
    case IValue::Tag::Object: {
      auto class_type = type()->expect<ClassType>();
      if (class_type->hasMethod("__getstate__") &&
          class_type->hasMethod("__setstate__")) {
        copy = ivalue::Object::create(
            c10::StrongTypePtr(class_type->compilation_unit(), type()),
            class_type->numAttributes());
        auto state = class_type->getMethod("__getstate__")({*this});
        class_type->getMethod("__setstate__")({copy, std::move(state)});
      } else {
        copy = IValue(toObject()->deepcopy(memo));
      }
    } break;
    case IValue::Tag::Enum: {
      auto enum_holder = toEnumHolder();
      copy = IValue(c10::make_intrusive<ivalue::EnumHolder>(
          enum_holder->type(),
          enum_holder->name(),
          enum_holder->value().deepcopy(memo)));
    } break;
    case IValue::Tag::String:
    case IValue::Tag::None:
    case IValue::Tag::Double:
    case IValue::Tag::Int:
    case IValue::Tag::SymInt:
    case IValue::Tag::SymFloat:
    case IValue::Tag::SymBool:
    case IValue::Tag::Bool:
    case IValue::Tag::Device:
    case IValue::Tag::Uninitialized: {
      copy = *this;
    } break;
    default: {
      AT_ERROR("Can't deepcopy IValue with tag: ", tagKind());
    }
  }
  // NB: this doesn't work if an object contains itself, and it may
  // come up in the future when we expand the object system, we will
  // have a follow up PR to fix this when it becomes an issue.
  if (!isAliasOf(copy)) {
    memo[*this] = copy;
  }
  return copy;
}

void IValue::reportToTensorTypeError() const {
  TORCH_CHECK(false, "Expected Tensor but got ", tagKind());
}

std::string ivalue::Object::name() const {
  return type()->name()->qualifiedName();
}

IValue ivalue::Object::getAttr(const std::string& name) const {
  const size_t slot = type()->getAttributeSlot(name);
  return getSlot(slot);
}

void ivalue::Object::setAttr(const std::string& name, IValue v) {
  const size_t slot = type()->getAttributeSlot(name);
  setSlot(slot, std::move(v));
}

void ivalue::Object::unsafeRemoveAttr(const std::string& name) {
  const size_t slot = type()->getAttributeSlot(name);
  unsafeRemoveSlot(slot);
}

void ivalue::Object::resizeObject(size_t slot) {
  AT_ASSERT(slot < type()->numAttributes());
  slots_.resize(type()->numAttributes());
}


c10::intrusive_ptr<ivalue::Object> ivalue::Object::copy() const {
  auto object = ivalue::Object::create(type_, type()->numAttributes());
  for (const auto i : c10::irange(slots_.size())) {
    object->setSlot(i, slots_[i]);
  }
  return object;
}

c10::intrusive_ptr<ivalue::Object> ivalue::Object::copy_to_weak_compilation_ref() const {
  auto object = ivalue::Object::create(
      WeakOrStrongTypePtr(type_.asWeakTypePtr()), type()->numAttributes());
  for (const auto i : c10::irange(slots_.size())) {
    object->setSlot(i, slots_[i]);
  }
  return object;
}

c10::intrusive_ptr<ivalue::Object> ivalue::Object::deepcopy() const {
  IValue::HashAliasedIValueMap memo;
  return deepcopy(memo);
}

c10::intrusive_ptr<ivalue::Object> ivalue::Object::deepcopy(IValue::HashAliasedIValueMap& memo) const {
  auto cu = type_.cu_;
  auto object = ivalue::Object::create(WeakOrStrongTypePtr(type_.cu_, type_.type_), type()->numAttributes());
  for (const auto i : c10::irange(slots_.size())) {
    if (*slots_[i].type() == *c10::TypeFactory::get<CapsuleType>()) {
      // If we've gotten here, it means that we have *not* copied this
      // class via __getstate__ and __setstate__. That fact and the
      // fact that we have a Capsule attribute mean that this is a
      // custom C++ class without serialization methods defined.
      std::stringstream err;
      err << "Cannot serialize custom bound C++ class";
      if (auto qualname = type()->name()) {
        err << " " << qualname->qualifiedName();
      }
      err << ". Please define serialization methods via def_pickle() for "
            "this class.";
      AT_ERROR(err.str());
    }
    object->setSlot(i, slots_[i].deepcopy(memo));
  }
  return object;
}

StrongTypePtr::StrongTypePtr(
    std::shared_ptr<torch::jit::CompilationUnit> cu,
    TypePtr type) : cu_(std::move(cu)), type_(std::move(type)) {
  TORCH_INTERNAL_ASSERT(type_);
}

WeakTypePtr::WeakTypePtr(
    std::weak_ptr<torch::jit::CompilationUnit> cu,
    TypePtr type) : cu_(std::move(cu)), type_(std::move(type)) {}

WeakTypePtr WeakOrStrongTypePtr::asWeakTypePtr() const {
  if (!holds_strong_ref()) {
    return WeakTypePtr(cu_.getWeakRefOrThrow(), type_);
  } else {
    std::weak_ptr<torch::jit::CompilationUnit> weak_cu =
        cu_.getStrongRefOrThrow();
    return WeakTypePtr(std::move(weak_cu), type_);
  }
}

// Needs to be in this .cpp file to access the full definition of PyObjectHolder
std::vector<c10::weak_intrusive_ptr<c10::StorageImpl>> ivalue::Future::extractStorages(
    const at::IValue& value) {
  std::vector<c10::weak_intrusive_ptr<c10::StorageImpl>> weakStorageImpls;
  // getSubValues works poorly on Python objects: it only works if they can be
  // converted to a "regular" IValue type hence, for example, it doesn't support
  // custom subclasses. Thus, instead, we extract the tensors through pickling.
  if (value.isPyObject()) {
    std::vector<at::Tensor> tensors =
        value.toPyObjectHolder()->extractTensors();
    size_t num_storages = 0;
    for (const at::Tensor& tensor : tensors) {
      if (tensor.is_sparse()) {
        // Sparse tensor is indices and values. Both are tensors
        // and contain storage. Therefore num_storages needs to be
        // incremented by 2.
        num_storages += 2;
      } else {
        // A dense/strided tensor contains 1 storage.
        num_storages += 1;
      }
    }
    weakStorageImpls.reserve(num_storages);
    for (const at::Tensor& tensor : tensors) {
      if (tensor.is_sparse()) {
        // Sparse tensor is indices and values. Both are tensors
        // and contain storage.
        // TODO (rohan-varma): for tensors created with at::sparse_coo_tensor held
        // in a python object, this might need a coalesce().
        weakStorageImpls.emplace_back(tensor.indices().storage().getWeakStorageImpl());
        weakStorageImpls.emplace_back(tensor.values().storage().getWeakStorageImpl());
      } else {
        // A dense/strided tensor contains 1 storage
        weakStorageImpls.emplace_back(tensor.storage().getWeakStorageImpl());
      }
    }
  } else {
    at::IValue::HashAliasedIValues sub_values;
    // Prefer getSubValues() over visit() as the latter is a silent no-op for
    // some unsupported types, whereas the former at least fails loudly.
    value.getSubValues(sub_values);
    for (const at::IValue& sub_value : sub_values) {
      if (sub_value.isTensor()) {
        auto tens = sub_value.toTensor();
        if (tens.is_sparse()) {
          // sparse tensors have 2 storages! one for indices one for values
          auto coalesced = tens.coalesce();
          weakStorageImpls.emplace_back(coalesced.indices().storage().getWeakStorageImpl());
          weakStorageImpls.emplace_back(coalesced.values().storage().getWeakStorageImpl());
        } else {
          weakStorageImpls.emplace_back(tens.storage().getWeakStorageImpl());
        }
      }
    }
  }
  return weakStorageImpls;
}

TORCH_API intrusive_ptr<ivalue::Future> collectAll(
    List<intrusive_ptr<ivalue::Future>> srcs) {
  struct Ctx {
    explicit Ctx(List<intrusive_ptr<ivalue::Future>> srcs)
        : remaining(srcs.size()),
          srcFutures(std::move(srcs)),
          asIvalue(srcFutures),
          // No need to pass devices, because dstFuture won't directly contain
          // the value, it will contain the srcFutures (which have no DataPtrs).
          dstFuture(make_intrusive<ivalue::Future>(asIvalue.type())) {}
    std::atomic<int32_t> remaining{0};
    List<intrusive_ptr<ivalue::Future>> srcFutures;
    IValue asIvalue;
    intrusive_ptr<ivalue::Future> dstFuture;
  };

  auto ctx = std::make_shared<Ctx>(std::move(srcs));
  if (ctx->srcFutures.empty()) {
    ctx->dstFuture->markCompleted(ctx->asIvalue);
  } else {
    auto typePtr = ctx->srcFutures.get(0)->elementType();
    for (const auto i : c10::irange(ctx->srcFutures.size())) {

      std::function<void(ivalue::Future&)> func = [ctx](ivalue::Future& fut) {
        // Set error and exit early if encountered.
        if (fut.hasError()) {
          ctx->dstFuture->setErrorIfNeeded(fut.exception_ptr());
          return;
        }

        if (--ctx->remaining == 0 && !ctx->dstFuture->completed()) {
          // No need to pass DataPtrs, because dstFuture won't directly contain
          // the value, it will contain the srcFutures (which have no DataPtrs).
          ctx->dstFuture->markCompleted(ctx->asIvalue);
        }
      };
      ctx->srcFutures.get(i)->addCallback(func);
    }
  }
  return ctx->dstFuture;
}

namespace {

std::string formatSetOfDevices(const std::vector<c10::Device>& devices) {
  std::ostringstream oss;
  std::copy(
      devices.begin(),
      devices.end(),
      std::ostream_iterator<c10::Device>(oss, ", "));
  return oss.str();
}

}

TORCH_API intrusive_ptr<ivalue::Future> collectAny(
    List<intrusive_ptr<ivalue::Future>> srcs) {
  if (srcs.empty()) {
    auto res = make_intrusive<ivalue::Future>(NoneType::get());
    res->markCompleted();
    return res;
  }
  TypePtr typePtr = srcs.get(0)->elementType();
  const std::vector<c10::Device>& devices = srcs.get(0)->devices();
  for (const auto i : c10::irange(srcs.size())) {
    if (srcs.get(i)->completed()) {
      return srcs.get(i);
    }
    TORCH_CHECK_TYPE(
        i == 0 || (*typePtr == *srcs.get(i)->elementType()),
        "Expected all futures to have the same type, but found ", *typePtr,
        " in position 0 and ", *srcs.get(i)->elementType(), " in position ", i);
    TORCH_CHECK_VALUE(
        i == 0 || (devices == srcs.get(i)->devices()),
        "Expected all futures to have the same devices, but found ",
        formatSetOfDevices(devices), " in position 0 and ",
        formatSetOfDevices(srcs.get(i)->devices()), " in position ", i);
  }
  struct Ctx {
    explicit Ctx(
        List<intrusive_ptr<ivalue::Future>> srcs,
        TypePtr typePtr,
        std::vector<c10::Device> devices)
        : srcFutures(std::move(srcs)),
          dstFuture(make_intrusive<ivalue::Future>(typePtr, std::move(devices))) {}
    std::atomic<bool> done{false};
    List<intrusive_ptr<ivalue::Future>> srcFutures;
    intrusive_ptr<ivalue::Future> dstFuture;
  };
  auto ctx = std::make_shared<Ctx>(std::move(srcs), typePtr, devices);
  std::function<void(ivalue::Future&)> func = [ctx](ivalue::Future& src) {
    if (!ctx->done.exchange(true)) {
      intrusive_ptr<ivalue::Future> dst = ctx->dstFuture;
      ctx->dstFuture.reset(); // Once future is satisfied, remove refs.
      ctx->srcFutures =
          List<intrusive_ptr<ivalue::Future>>(ctx->srcFutures.elementType());
      if (src.hasError()) {
        dst->setError(src.exception_ptr());
      } else {
        dst->markCompleted(src.constValue(), src.storages());
      }
    }
  };
  for (const auto i : c10::irange(ctx->srcFutures.size())) {
    ctx->srcFutures.get(i)->addCallback(func);
  }
  return ctx->dstFuture;
}
} // namespace c10
