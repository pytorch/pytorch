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

} // namespace ivalue

namespace {

template<class T>
std::ostream& printList(std::ostream & out, const c10::List<T> &v,
  const std::string start, const std::string finish) {
  out << start;
  for(size_t i = 0; i < v.size(); ++i) {
    if(i > 0)
      out << ", ";
    // make sure we use ivalue printing, and not default printing for the element type
    out << IValue(v.get(i));
  }
  out << finish;
  return out;
}

template<class T>
std::ostream& printList(std::ostream & out, const std::vector<T> &v,
  const std::string start, const std::string finish) {
  out << start;
  for(size_t i = 0; i < v.size(); ++i) {
    if(i > 0)
      out << ", ";
    // make sure we use ivalue printing, and not default printing for the element type
    out << IValue(v[i]);
  }
  out << finish;
  return out;
}

template<typename Dict>
std::ostream& printDict(std::ostream& out, const Dict& v) {
  out << "{";

  bool first = true;
  for (const auto& pair : v) {
    if (!first) {
      out << ", ";
    }
    out << pair.key() << ": " << pair.value();
    first = false;
  }

  out << "}";
  return out;
}

} // anonymous namespace

std::ostream& operator<<(std::ostream & out, const IValue & v) {
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
    case IValue::Tag::Tuple:
      return printList(out, v.toTuple()->elements(), "(", ")");
    case IValue::Tag::IntList:
      return printList(out, v.toIntList(), "[", "]");
    case IValue::Tag::DoubleList:
      return printList(out, v.toDoubleList(), "[", "]");
    case IValue::Tag::BoolList:
      return printList(out, v.toBoolList(), "[", "]");
    case IValue::Tag::String:
      return out << v.toStringRef();
    case IValue::Tag::TensorList:
      return printList(out, v.toTensorList(), "[", "]");
    case IValue::Tag::Blob:
      return out << *v.toBlob();
    case IValue::Tag::Capsule:
      return out << "Capsule";
    case IValue::Tag::GenericList:
      return printList(out, v.toGenericList(), "[", "]");
    case IValue::Tag::Future:
      return out << "Future";
    case IValue::Tag::Uninitialized:
      return out << "Uninitialized";
    case IValue::Tag::Device:
      return out << v.toDevice();
    case IValue::Tag::GenericDict:
      return printDict(out, v.toGenericDict());
    case IValue::Tag::Object:
      // TODO we should attempt to call __str__ if the object defines it.
      auto obj = v.toObject();
      // print this out the way python would do it
      return out << "<" << obj->name() << " object at " << obj.get() << ">";
  }
  AT_ERROR("Tag not found: ", v.tagKind());
}

#undef TORCH_FORALL_TAGS

void IValue::dump() const {
  std::cout << *this << "\n";
}


std::string ivalue::Object::name() const {
  return this->type_.type_->name()->qualifiedName();
}

IValue ivalue::Object::getAttr(const std::string& name) const {
  const size_t slot = type_.type_->getAttributeSlot(name);
  return getSlot(slot);
}

void ivalue::Object::setAttr(const std::string& name, IValue v) {
  const size_t slot = type_.type_->getAttributeSlot(name);
  setSlot(slot, std::move(v));
}

void ivalue::Object::resizeObject(size_t slot) {
  AT_ASSERT(slot < type()->numAttributes());
  slots_.resize(type()->numAttributes());
}

static bool CompareIValue(const std::pair<IValue, IValue>& aWrap,
                          const std::pair<IValue, IValue>& bWrap) {
  const auto a = aWrap.first;
  const auto b = bWrap.first;
  if (a.isString() && b.isString()) {
    return a.toStringRef().compare(b.toStringRef()) < 0;
  } else if (a.isInt() && b.isInt()) {
    return a.toInt() < b.toInt();
  } else if (a.isDouble() && b.isDouble()) {
    return a.toDouble() < b.toDouble();
  }
  AT_ERROR("Illegal dict key");
}

std::vector<std::pair<IValue, IValue>> iterationOrder(const c10::Dict<IValue, IValue>& dict) {
  std::vector<std::pair<IValue, IValue>> ordered;
  for (auto& element : dict) {
    ordered.emplace_back(element.key(), element.value());
  }
  std::sort(ordered.begin(), ordered.end(), CompareIValue);
  return ordered;
}

std::unordered_map<std::string, c10::StrongTypePtr>& getCustomClassTypeMap() {
    static std::unordered_map<std::string, c10::StrongTypePtr> tmap;
    return tmap;
}

std::unordered_map<std::string, std::function<PyObject*(void*)>>&
getClassConverter() {
  static std::unordered_map<std::string, std::function<PyObject*(void*)>>
      classConverter;
  return classConverter;
}
} // namespace c10
