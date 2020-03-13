#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/Formatting.h>
#include <c10/util/StringUtil.h>
#include <cmath>
#include <ATen/core/Dict.h>

namespace c10 {
namespace ivalue {

CAFFE2_API c10::intrusive_ptr<ConstantString> ConstantString::create(
    std::string str_) {
  return c10::make_intrusive<ConstantString>(std::move(str_));
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
    case Tag::Future:
    case Tag::Device:
    case Tag::Object:
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


static bool CompareKeys(const std::pair<IValue, IValue>& aWrap,
                        const std::pair<IValue, IValue>& bWrap) {
  const auto a = aWrap.first;
  const auto b = bWrap.first;
  if (a.isString() && b.isString()) {
    return a.toStringRef().compare(b.toStringRef()) < 0;
  } else if (a.isInt() && b.isInt()) {
    return a.toInt() < b.toInt();
  } else if (a.isDouble() && b.isDouble()) {
    return a.toDouble() < b.toDouble();
  } else if (a.isTensor() && b.isTensor()) {
    return a.toTensor().unsafeGetTensorImpl() < b.toTensor().unsafeGetTensorImpl();
  }
  AT_ERROR("Illegal dict key");
}

std::vector<std::pair<IValue, IValue>> iterationOrder(const c10::Dict<IValue, IValue>& dict) {
  std::vector<std::pair<IValue, IValue>> ordered;
  for (auto& element : dict) {
    ordered.emplace_back(element.key(), element.value());
  }
  std::sort(ordered.begin(), ordered.end(), CompareKeys);
  return ordered;
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
