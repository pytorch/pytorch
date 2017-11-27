#include "torch/csrc/jit/ir.h"
#include "torch/csrc/autograd/function.h"

#include <functional>

// ${generated_comment}

namespace torch { namespace jit {

// a list of refcounted pointers. this is not called refcounted_list because
// that sounds like a refcounted list of pointers, which is something else.
using list_of_refcounted = std::vector<at::RefCounted*>;


using Operation = std::function<void(const list_of_refcounted &, // inputs
                                   list_of_refcounted &)>; // outputs
// An Operation borrows the inputs without changing their refcount
// it returns outputs, a list of RefCounted* that the caller is responsible for releasing

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
