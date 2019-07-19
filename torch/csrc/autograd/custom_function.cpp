#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/functions/accumulate_grad.h>

namespace torch { namespace autograd {

variable_list _wrap_outputs(const variable_list &input_vars,
  const std::unordered_set<at::TensorImpl*> &non_differentiable,
  const std::unordered_set<at::TensorImpl*> &dirty_inputs,
  const at::ArrayRef<Variable> raw_outputs,
  const std::shared_ptr<Function> &cdata) {

  std::unordered_set<at::TensorImpl*> inputs;
  inputs.reserve(input_vars.size());
  for (auto& var : input_vars) {
    inputs.emplace(var.unsafeGetTensorImpl());
  }

  // Sets the grad_fn and output_nr of an output Variable.
  auto set_history = [&](Variable& var, uint32_t output_nr, bool is_input, bool is_modified,
                         bool is_differentiable) {
    if (!is_differentiable) {
      if (!var.requires_grad()) {
        return;
      }
      // NB: we don't support returning non-differentiable views that could require grad
      if (var.is_view()) {
        throw std::runtime_error("Returning Variables sharing storage with other Variables "
                                 "that require grad is not supported in Python functions. "
                                 "Please submit a feature request if you hit this error.");
      }
      // Return detached aliases of inputs, instead of changing their requires_grad
      // property.
      if (is_input) {
        var = var.detach();
      } else {
        var.detach_();
      }
    } else if (is_modified) {
      if (var.is_leaf() && var.requires_grad()) {
        throw std::runtime_error("a leaf Variable that requires grad has been used in an in-place operation.");
      }
      // If the input was modified, transplant the grad_fn in the graph:
      // grad_fn <- variable <- self  ==>  grad_fn <- self <- variable
      var.grad().reset();
      var.clear_hooks();
      if (auto grad_acc_fn = var.try_get_grad_accumulator()) {
        auto grad_acc = dynamic_cast<AccumulateGrad*>(grad_acc_fn.get());
        grad_acc->variable.reset();
      }
      if (cdata) {
        var.rebase_history({cdata, output_nr});
      }
    } else if (is_input) {
      // An input has been returned, but it wasn't modified. Return it as a view
      // so that we can attach a new grad_fn to the Variable.
      var = var.view_as(var);
      var.set_gradient_edge({cdata, output_nr});
    } else if (cdata) {
      var.set_gradient_edge({cdata, output_nr});
    }
  };

  int num_outputs = raw_outputs.size();

  std::vector<torch::autograd::Variable> outputs;
  outputs.reserve(num_outputs);

  for (auto i = 0; i < num_outputs; ++i) {
    auto out_tensor_impl = raw_outputs[i].unsafeGetTensorImpl();
    bool is_input = inputs.count(out_tensor_impl) > 0;
    bool is_modified = dirty_inputs.count(out_tensor_impl) > 0;
    bool is_differentiable = cdata && non_differentiable.count(out_tensor_impl) == 0;

    Variable var = raw_outputs[i];

    if (cdata) {
      auto output_nr = cdata->add_input_metadata(var);
      AT_ASSERT(i == (int)output_nr);
    }
    set_history(var, i, is_input, is_modified, is_differentiable);

    outputs.emplace_back(var);
  }

  return outputs;
}




}} // namespace torch::autograd
