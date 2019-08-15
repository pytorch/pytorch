#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/variable.h>

#include <torch/csrc/autograd/edge.h>
#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/autograd/function.h>


namespace torch {
namespace autograd {
   variable_list run_backward(const variable_list& outputs,
                              const variable_list& grad_outputs,
                              bool keep_graph,
                              bool create_graph,
                              const variable_list& inputs) {
    size_t num_tensors = outputs.size();
    size_t num_gradients = grad_outputs.size();
    TORCH_CHECK(num_tensors == num_gradients, "got %ld tensors and %ld "
                "gradients", num_tensors, num_gradients);

    edge_list roots;
    roots.reserve(num_tensors);
    variable_list grads;
    grads.reserve(num_tensors);

    for (size_t i = 0; i < num_tensors; i++) {
        const Variable& output = outputs[i];
        auto gradient_edge = output.gradient_edge();
        TORCH_CHECK(gradient_edge.function,
            "element %d of tensors does not require grad and does not have a grad_fn", i);
        roots.push_back(std::move(gradient_edge));

        const Variable& grad = grad_outputs[i];
        if (!grad.defined()) {
            // if grad is undefined
            if (output.requires_grad()) {
                TORCH_CHECK(output.numel() == 1, "grad can be implicitly created only for scalar outputs");
                // create the grad with ones tensor for output that requires grad
                grads.push_back(at::ones_like(output));
            }
            // otherwise we don't put grad into the grads list
        } else if (grad.is_variable()) {
            // if grad is a typical variable
            grads.push_back(grad);
        } else {
            AT_ERROR("element %d of gradients tuple is not a Tensor or None", i);
        }
    }

    edge_list output_edges;
    if (!inputs.empty()) {
        size_t num_inputs = inputs.size();
        output_edges.reserve(num_inputs);
        for (size_t i = 0; i < num_inputs; ++ i) {
            const Variable& input = inputs[i];
            const auto output_nr = input.output_nr();
            auto grad_fn = input.grad_fn();
            if (!grad_fn) {
                grad_fn = input.try_get_grad_accumulator();
            }
            TORCH_CHECK(input.requires_grad(),
                "One of the differentiated Tensors does not require grad");
            if (!grad_fn) {
                output_edges.emplace_back();
            } else {
                output_edges.emplace_back(grad_fn, output_nr);
            }
        }
    }

    return Engine::get_default_engine().execute(roots, grads, keep_graph, create_graph, output_edges);
   }

  void backward(
      const variable_list& tensors,
      variable_list grad_tensors,
      c10::optional<bool> keep_graph,
      bool create_graph) {
    if (!keep_graph) {
        keep_graph = create_graph;
    }
    if (grad_tensors.empty()) {
        for(size_t i = 0; i < tensors.size(); ++ i) {
            grad_tensors.emplace_back(at::Tensor());
        }
    }
    run_backward(tensors, grad_tensors, keep_graph.value(), create_graph, {});
  }

  variable_list grad(
      const variable_list& outputs,
      const variable_list& inputs,
      variable_list grad_outputs,
      c10::optional<bool> keep_graph,
      bool create_graph,
      bool allow_unused) {
    if (!keep_graph) {
        keep_graph = create_graph;
    }
    if (grad_outputs.empty()) {
        for(size_t i = 0; i < outputs.size(); ++ i) {
            grad_outputs.emplace_back(at::Tensor());
        }
    }
    variable_list grad_inputs = run_backward(outputs, grad_outputs, keep_graph.value(), create_graph, inputs);
    // check if grad_inputs contains None or not base on the allow_unused flag
    if (inputs.empty()) {
        size_t num_inputs = inputs.size();
        for (size_t i = 0; i < num_inputs; ++ i) {
            TORCH_CHECK(allow_unused || grad_inputs[i].defined(), "One of the "
                    "differentiated Tensors appears to not have been used "
                    "in the graph. Set allow_unused=True if this is the "
                    "desired behavior.");
        }
    }
    return grad_inputs;
   }

}} // namespace torch::autograd