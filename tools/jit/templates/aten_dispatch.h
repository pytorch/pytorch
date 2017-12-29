#include "torch/csrc/jit/ir.h"
#include "torch/csrc/autograd/function.h"

#include <functional>

// ${generated_comment}

namespace torch { namespace jit {


// a list of refcounted pointers. this is not called refcounted_list because
// that sounds like a refcounted list of pointers, which is something else.
using list_of_retainable = std::vector<at::Retainable*>;


using Operation = std::function<void(const list_of_retainable &, // inputs
                                   list_of_retainable &)>; // outputs
// An Operation borrows the inputs without changing their refcount
// it returns outputs, a list of Retainable* that the caller is responsible for releasing

// functions for converting between retainables and Tensors

// create a Tensor, sharing ownership of the retainable with its current owners
// 'unsafe' because Retainable might not be a tensor
static inline at::Tensor unsafeToTensorShare(at::Retainable* rc) {
  return at::Tensor(static_cast<at::TensorImpl*>(rc), true);
}

// create a Tensor, stealing ownership of the retainable from the caller
// and resetting the retainable pointer
static inline at::Tensor unsafeToTensorSteal(at::Retainable* && rc_) {
  at::Retainable* rc = rc_;
  auto r = at::Tensor(static_cast<at::TensorImpl*>(rc), false);
  rc_ = nullptr;
  return r;
}

// share the underlying TensorImpl, bumping its refcount
static inline at::Retainable * toRetainableShare(const at::Tensor & t) {
  return at::Tensor(t).detach();
}

// extract the underlying TensorImpl from a Tensor, stealing its ownership from
// the caller and reseting its pointer.
static inline at::Retainable * toRetainableSteal(at::Tensor && t) {
  return t.detach();
}

struct TensorOp {
  TensorOp(Operation op, std::string name, size_t num_inputs)
    : op(op)
    , name(name)
    , num_inputs(num_inputs) {}

  const Operation op;
  const std::string name;
  const size_t num_inputs;
};

TensorOp getTensorOp(jit::Node* n);

}} // namespace torch::jit;
