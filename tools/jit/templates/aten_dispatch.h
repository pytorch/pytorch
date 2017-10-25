#include "torch/csrc/jit/ir.h"
#include "torch/csrc/autograd/function.h"

#include <functional>

// ${generated_comment}

namespace torch { namespace jit {

struct TensorOp {
  using op_type = std::function<autograd::variable_list(const autograd::variable_list&)>;

  TensorOp(op_type op, std::string name, size_t num_inputs)
    : op(op)
    , name(name)
    , num_inputs(num_inputs) {}

  const op_type op;
  const std::string name;
  const size_t num_inputs;
};

TensorOp getTensorOp(jit::Node* n);

}} // namespace torch::jit;
