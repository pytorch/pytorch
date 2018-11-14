#include <ATen/core/ivalue.h>
#include <ATen/core/Formatting.h>
#include <cmath>

namespace c10 {
namespace ivalue {

CAFFE2_API c10::intrusive_ptr<ConstantString> ConstantString::create(
    std::string str_) {
  return c10::make_intrusive<ConstantString>(std::move(str_));
}

} // namespace ivalue

namespace {

template<typename List>
std::ostream& printList(std::ostream & out, const List &v,
  const std::string start, const std::string finish) {
  out << start;
  for(size_t i = 0; i < v->elements().size(); ++i) {
    if(i > 0)
      out << ", ";
    // make sure we use ivalue printing, and not default printing for the element type
    out << IValue(v->elements()[i]);
  }
  out << finish;
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
      int64_t i = int64_t(d);
      if (std::isnormal(d) && double(i) == d) {
        return out << i << ".";
      }
      return out << v.toDouble();
    } case IValue::Tag::Int:
      return out << v.toInt();
    case IValue::Tag::Bool:
      return out << v.toBool();
    case IValue::Tag::Tuple:
      return printList(out, v.toTuple(), "(", ")");
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
      return out << v.toBlob();
    case IValue::Tag::GenericList:
      return printList(out, v.toGenericList(), "[", "]");
    case IValue::Tag::Future:
      return out << "Future";
  }
  AT_ERROR("Tag not found\n");
}

#undef TORCH_FORALL_TAGS

} // namespace c10
