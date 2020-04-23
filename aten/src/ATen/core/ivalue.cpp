#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/Formatting.h>
#include <c10/util/StringUtil.h>
#include <cmath>
#include <ATen/core/Dict.h>

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

// This is in ivalue.cpp because we need to access Type::python_str, which
// is declared in jit_type.h
void checkCustomClassType(TypePtr expected_type, TypePtr actual_type) {
  // NB: doing pointer comparison here
  // If in the future there ever arises a need to call operator== on custom class
  // Type's, this needs to be changed!
  TORCH_CHECK(actual_type == expected_type,
              "Tried to convert an IValue of type ",
              actual_type->python_str(),
              " to custom class type ",
              expected_type->python_str());
}

CAFFE2_API c10::intrusive_ptr<ConstantString> ConstantString::create(
    std::string str_) {
  return c10::make_intrusive<ConstantString>(std::move(str_));
}

bool operator==(const ivalue::Tuple& lhs, const ivalue::Tuple& rhs) {
  return lhs.elements_.size() == rhs.elements_.size() &&
      // see [container equality]
      std::equal(
             lhs.elements_.cbegin(),
             lhs.elements_.cend(),
             rhs.elements_.cbegin(),
             _fastEqualsForContainer);
}

TupleTypePtr Tuple::type() const {
  if (!type_) {
    type_ = TupleType::create(
        fmap(elements_, [&](const IValue& v) { return v.type(); }));
  }
  return type_;
}

} // namespace ivalue

TypePtr IValue::type() const {
  switch (tag) {
    case Tag::None:
      return NoneType::get();
    case Tag::Tensor:
      return TensorType::create(toTensor());
    case Tag::Double:
      return FloatType::get();
    case Tag::Int:
      return IntType::get();
    case Tag::Bool:
      return BoolType::get();
    case Tag::String:
      return StringType::get();
    case Tag::Blob:
      return AnyType::get();
    case Tag::GenericDict: {
      auto d = toGenericDict();
      return DictType::create(d.keyType(), d.valueType());
    }
    case Tag::GenericList:
      return ListType::create(toList().elementType());
    case Tag::Future:
      return toFuture()->type();
    case Tag::RRef:
      return RRefType::create(toRRef()->type());
    case Tag::Device:
      return DeviceObjType::get();
    case Tag::Object:
      return toObjectRef().type();
    case Tag::PyObject:
      return PyObjectType::get();
    case Tag::Uninitialized:
      return AnyType::get();
    case Tag::Capsule:
      return CapsuleType::get();
    case Tag::Tuple:
      return toTuple()->type();
    case Tag::Generator:
      return GeneratorType::get();
  }
  // switch above is complete but this silences compiler warnings
  TORCH_INTERNAL_ASSERT(false, "unhandled case in IValue::type()");
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
        elems = this->toTuple()->elements();
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
      auto attribute_names = obj_type->attributeNames();
      for (const auto& name: attribute_names) {
        auto attribute = obj_value->getAttr(name);
        attribute.getSubValues(subValues);
      }
      break;
    }
    case Tag::Future:
    case Tag::Device:
    case Tag::PyObject:
    case Tag::Uninitialized:
    case Tag::Capsule:
      TORCH_INTERNAL_ASSERT(
          false, "sub ivalue is nat enabled for: ", this->tagKind());
      // Fall through
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
  TORCH_INTERNAL_ASSERT(lhs.is_intrusive_ptr);
  TORCH_INTERNAL_ASSERT(rhs.is_intrusive_ptr);
  return lhs.tag == rhs.tag &&
      lhs.payload.as_intrusive_ptr == rhs.payload.as_intrusive_ptr;
}

IValue IValue::equals(const IValue& rhs) const {
  const IValue& lhs = *this;
  switch (lhs.tag) {
    case Tag::None:
      // In Python you're not supposed to do this comparison apparently. Not
      // sure if we should warn here or what
      return rhs.isNone();
    case Tag::Tensor:
      if (!rhs.isTensor()) {
        return false;
      }
      return lhs.toTensor().eq(rhs.toTensor());
    case Tag::Double:
      return rhs.isDouble() && lhs.toDouble() == rhs.toDouble();
    case Tag::Int:
      return rhs.isInt() && lhs.toInt() == rhs.toInt();
    case Tag::Bool:
      return rhs.isBool() && lhs.toBool() == rhs.toBool();
    case Tag::String:
      return rhs.isString() && lhs.toStringRef() == rhs.toStringRef();
    case Tag::GenericDict:
      return rhs.isGenericDict() && lhs.toGenericDict() == rhs.toGenericDict();
    case Tag::Tuple:
      return rhs.isTuple() && *lhs.toTuple() == *rhs.toTuple();
    case Tag::Device:
      return rhs.isDevice() && lhs.toDevice() == rhs.toDevice();
    case Tag::GenericList:
      return rhs.isList() && lhs.toList() == rhs.toList();
    case Tag::Blob:
    case Tag::Future:
    case Tag::RRef:
    case Tag::Object:
    case Tag::PyObject:
    case Tag::Capsule:
    case Tag::Generator:
      return ptrEqual(lhs, rhs);
    case Tag::Uninitialized:
      // Unitialized ivalues show up in no-ops when the compiler can prove a
      // value will never be used. Just return false on any equality comparison.
      return false;
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

  if (lhs.is_intrusive_ptr) {
    return rhs.is_intrusive_ptr && ptrEqual(lhs, rhs);
  }
  return lhs == rhs;
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
  for (size_t i = 0; i < list.size(); ++i) {
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
  if (the_list.toListRef().size() == 0) {
    out << "annotate(" << the_list.type()->python_str() << ", [])";
  } else {
    return printList(out, the_list.toListRef(), "[", "]", formatter);
  }
  return out;
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
std::ostream& printMaybeAnnotatedDict(
    std::ostream& out,
    const IValue& the_dict,
    IValueFormatter formatter) {
  auto value_type = the_dict.type()->cast<DictType>()->getValueType();
  if (the_dict.toGenericDict().size() == 0 ||
      !elementTypeCanBeInferredFromMembers(value_type)) {
    out << "annotate(" << the_dict.type()->python_str() << ",";
    printDict(out, the_dict.toGenericDict(), formatter) << ")";
  } else {
    return printDict(out, the_dict.toGenericDict(), formatter);
  }
  return out;
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
      if (c == FP_NORMAL || c == FP_ZERO) {
        int64_t i = int64_t(d);
        if (double(i) == d) {
          return out << i << ".";
        }
      }
      auto orig_prec = out.precision();
      return out << std::setprecision(std::numeric_limits<double>::max_digits10)
                 << v.toDouble() << std::setprecision(orig_prec);
    }
    case IValue::Tag::Int:
      return out << v.toInt();
    case IValue::Tag::Bool:
      return out << (v.toBool() ? "True" : "False");
    case IValue::Tag::Tuple: {
      const auto& elements = v.toTuple()->elements();
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
    default:
      TORCH_INTERNAL_ASSERT(false, "repr() not defined on: ", v.tagKind());
  }
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
    } case IValue::Tag::Int:
      return out << v.toInt();
    case IValue::Tag::Bool:
      return out << (v.toBool() ? "True" : "False");
    case IValue::Tag::Tuple: {
      const auto& elements = v.toTuple()->elements();
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
    case IValue::Tag::Uninitialized:
      return out << "Uninitialized";
    case IValue::Tag::Device:
      return out << v.toDevice();
    case IValue::Tag::GenericDict:
      return printDict(out, v.toGenericDict(), formatter);
    case IValue::Tag::PyObject: {
      auto py_obj = v.toPyObject();
      return out << "<PyObject at" << py_obj << ">";
    }
    case IValue::Tag::Generator:
      return out << "Generator";
    case IValue::Tag::Object: {
      // TODO we should attempt to call __str__ if the object defines it.
      auto obj = v.toObject();
      // print this out the way python would do it
      return out << "<" << obj->name() << " object at " << obj.get() << ">";
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

IValue IValue::copy() const {
  IValue copy;
  switch(tag) {
    case IValue::Tag::Tensor:
      copy = IValue(toTensor());
      break;
    case IValue::Tag::Tuple:
      copy = IValue(toTuple());
      break;
    case IValue::Tag::GenericList:
      copy = IValue(toList());
      break;
    case IValue::Tag::GenericDict:
      copy = IValue(toGenericDict());
      break;
    case IValue::Tag::Object:
      copy = IValue(toObject()->copy());
      break;
    case IValue::Tag::String:
    case IValue::Tag::None:
    case IValue::Tag::Double:
    case IValue::Tag::Int:
    case IValue::Tag::Bool:
    case IValue::Tag::Device:
    case IValue::Tag::Uninitialized:
      copy = *this;
      break;
    default:
      AT_ERROR("Can't copy IValue with tag: ", tagKind());
  }
  return copy;
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
      for (const auto& e : toTuple()->elements()) {
        copied_tuple.push_back(e.deepcopy(memo));
      }
      copy = IValue(ivalue::Tuple::create(copied_tuple));
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
      copy = IValue(toObject()->deepcopy(memo));
      break;
    case IValue::Tag::String:
    case IValue::Tag::None:
    case IValue::Tag::Double:
    case IValue::Tag::Int:
    case IValue::Tag::Bool:
    case IValue::Tag::Device:
    case IValue::Tag::Uninitialized:
      copy = *this;
      break;
    default:
      AT_ERROR("Can't deepcopy IValue with tag: ", tagKind());
    }
  }
  if (!isAliasOf(copy)) {
    memo[*this] = copy;
  }
  return copy;
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
  auto object = ivalue::Object::create(c10::StrongTypePtr(type_.cu_, type()), type()->numAttributes());
  for (auto i = 0; i < slots_.size(); ++i) {
    object->setSlot(i, slots_[i].copy());
  }
  return object;
}

c10::intrusive_ptr<ivalue::Object> ivalue::Object::deepcopy() const {
  IValue::HashAliasedIValueMap memo;
  return deepcopy(memo);
}

c10::intrusive_ptr<ivalue::Object> ivalue::Object::deepcopy(IValue::HashAliasedIValueMap& memo) const {
  auto object = ivalue::Object::create(c10::StrongTypePtr(type_.cu_, type()), type()->numAttributes());
  for (auto i = 0; i < slots_.size(); ++i) {
    object->setSlot(i, slots_[i].deepcopy(memo));
  }
  return object;
}

StrongTypePtr::StrongTypePtr(
    std::shared_ptr<torch::jit::CompilationUnit> cu,
    std::shared_ptr<Type> type) {
  cu_ = std::move(cu);
  type_ = type;
  TORCH_INTERNAL_ASSERT(type_);
}

std::unordered_map<std::string, c10::ClassTypePtr>& getCustomClassTypeMap() {
    static std::unordered_map<std::string, c10::ClassTypePtr> tmap;
    return tmap;
}

std::unordered_map<std::string, std::function<PyObject*(void*)>>&
getClassConverter() {
  static std::unordered_map<std::string, std::function<PyObject*(void*)>>
      classConverter;
  return classConverter;
}
} // namespace c10
