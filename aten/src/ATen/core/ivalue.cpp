#include <ATen/core/ivalue.h>
#include <ATen/core/Formatting.h>

#define TORCH_FORALL_TAGS(_) \
  _(None) \
  _(Tensor) \
  _(Double) \
  _(Int) \
  _(Bool) \
  _(Tuple) \
  _(IntList) \
  _(DoubleList) \
  _(BoolList) \
  _(String) \
  _(TensorList) \
  _(Blob) \
  _(GenericList) \
  _(World) \

namespace torch { namespace jit {

CAFFE2_API c10::intrusive_ptr<ConstantString> ConstantString::create(
    std::string str_) {
  return c10::make_intrusive<ConstantString>(std::move(str_));
}

namespace {

template<typename Elem>
std::ostream& printList(std::ostream & out, const List<Elem> &v,
  const std::string start, const std::string delim, const std::string finish) {
  out << start;
  for(size_t i = 0; i < v.elements().size(); ++i) {
    if(i > 0)
      out << delim;
    out << v.elements()[i];
  }
  out << finish;
  return out;
}

} // anonymous namespace

template<typename PointerType>
std::ostream& operator<<(std::ostream & out, const Shared<PointerType> & v) {
  return out << *v;
}

std::ostream& operator<<(std::ostream & out, const ConstantString & v) {
  return out << v.string();
}

template<typename Elem>
std::ostream& operator<<(std::ostream & out, const List<Elem> & v) {
  return printList<Elem>(out, v, "[", ", ", "]");
}

// tuple case
template<>
std::ostream& operator<<(std::ostream & out, const List<IValue> & v) {
  return printList<IValue>(out, v, "(", ", ", ")");
}

std::ostream& operator<<(std::ostream & out, const IValue & v) {
  switch(v.tag) {
    #define DEFINE_CASE(x) case IValue::Tag::x: return out << v.to ## x();
    TORCH_FORALL_TAGS(DEFINE_CASE)
    #undef DEFINE_CASE
  }
  AT_ERROR("Tag not found\n");
}

#undef TORCH_FORALL_TAGS

}}
