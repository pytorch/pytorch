#include "torch/csrc/jit/ir.h"
#include "torch/csrc/autograd/function.h"

#include <functional>

// ${generated_comment}

namespace torch { namespace jit {

using refcounted_list = std::vector<at::RefCounted*>;
using Operation = std::function<void(const refcounted_list &,
                                   refcounted_list &)>;
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
