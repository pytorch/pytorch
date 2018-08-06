#include "torch/csrc/jit/assertions.h"
#include "torch/csrc/jit/ivalue.h"
#include <ATen/ATen.h>

#define TORCH_FORALL_TAGS(_) \
  _(None) _(Tensor) _(Double) _(Int) _(Tuple) _(IntList) _(DoubleList) _(String) _(TensorList)

namespace torch { namespace jit {
std::ostream& operator<<(std::ostream & out, const IValue & v) {
  switch(v.tag) {
    #define DEFINE_CASE(x) case IValue::Tag::x: return v.format ## x(out);
    TORCH_FORALL_TAGS(DEFINE_CASE)
    #undef DEFINE_CASE
  }
  AT_ERROR("Tag not found\n");
}

#undef TORCH_FORALL_TAGS

}}
