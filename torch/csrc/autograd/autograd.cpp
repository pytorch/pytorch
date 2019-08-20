#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/variable.h>

#include <torch/csrc/autograd/edge.h>
#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/autograd/function.h>

namespace torch {
namespace autograd {

variable_list _make_grads(
    const variable_list& outputs,
    const variable_list& grad_outputs) {
  size_t num_tensors = outputs.size();
  size_t num_gradients = grad_outputs.size();
  variable_list new_grads;
  new_grads.reserve(num_tensors);
  if (grad_outputs.empty()) {
    for (const Variable& output : outputs) {
      if (output.requires_grad()) {
        TORCH_CHECK(
            output.numel() == 1,
            "grad can be implicitly created only for scalar outputs");
        new_grads.emplace_back(at::ones_like(output));
      }
    }
  } else {
    TORCH_CHECK(
        num_tensors == num_gradients,
        "got %ld tensors and %ld "
        "gradients",
        num_tensors,
        num_gradients);
    for (size_t i = 0; i < outputs.size(); ++i) {
      const Variable& output = outputs[i];
      const Variable& grad_output = grad_outputs[i];
      if (!grad_output.defined()) {
        if (output.requires_grad()) {
          TORCH_CHECK(
              output.numel() == 1,
              "grad can be implicitly created only for scalar outputs");
          new_grads.emplace_back(at::ones_like(output));
        }
      } else {
        // grad output is defined, just append to the new_grads
        new_grads.emplace_back(grad_output);
      }
    }
  }
  return new_grads;
}
variable_list run_backward(
    const variable_list& outputs,
    const variable_list& grad_outputs,
    bool keep_graph,
    bool create_graph,
    const variable_list& inputs) {
  size_t num_tensors = outputs.size();
  edge_list roots;
  roots.reserve(num_tensors);
  for (size_t i = 0; i < num_tensors; i++) {
    const Variable& output = outputs[i];
    auto gradient_edge = output.gradient_edge();
    TORCH_CHECK(
        gradient_edge.function,
        "element %d of tensors does not require grad and does not have a grad_fn",
        i);
    roots.push_back(std::move(gradient_edge));
  }

  edge_list output_edges;
  if (!inputs.empty()) {
    size_t num_inputs = inputs.size();
    output_edges.reserve(num_inputs);
    for (size_t i = 0; i < num_inputs; ++i) {
      const Variable& input = inputs[i];
      const auto output_nr = input.output_nr();
      auto grad_fn = input.grad_fn();
      if (!grad_fn) {
        grad_fn = input.try_get_grad_accumulator();
      }
      TORCH_CHECK(
          input.requires_grad(),
          "One of the differentiated Tensors does not require grad");
      if (!grad_fn) {
        output_edges.emplace_back();
      } else {
        output_edges.emplace_back(grad_fn, output_nr);
      }
    }
  }

  return Engine::get_default_engine().execute(
      roots, grad_outputs, keep_graph, create_graph, output_edges);
}

void backward(
    const variable_list& tensors,
    const variable_list& grad_tensors,
    c10::optional<bool> retain_graph,
    bool create_graph) {
  variable_list gradients = _make_grads(tensors, grad_tensors);
  if (!retain_graph) {
    retain_graph = create_graph;
  }
  run_backward(tensors, gradients, retain_graph.value(), create_graph, {});
}

variable_list grad(
    const variable_list& outputs,
    const variable_list& inputs,
    const variable_list& grad_outputs,
    c10::optional<bool> retain_graph,
    bool create_graph,
    bool allow_unused) {
  variable_list gradients = _make_grads(outputs, grad_outputs);
  if (!retain_graph) {
    retain_graph = create_graph;
  }
  variable_list grad_inputs = run_backward(
      outputs, gradients, retain_graph.value(), create_graph, inputs);
  // check if grad_inputs contains None or not base on the allow_unused flag
  if (inputs.empty()) {
    size_t num_inputs = inputs.size();
    for (size_t i = 0; i < num_inputs; ++i) {
      TORCH_CHECK(
          allow_unused || grad_inputs[i].defined(),
          "One of the "
          "differentiated Tensors appears to not have been used "
          "in the graph. Set allow_unused=True if this is the "
          "desired behavior.");
    }
  }
  return grad_inputs;
}

} // namespace autograd
} // namespace torch