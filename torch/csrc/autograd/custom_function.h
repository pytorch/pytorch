#pragma once

#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>

namespace torch { namespace autograd {

TORCH_API variable_list _wrap_outputs(
  const variable_list &input_vars,
  const std::unordered_set<at::TensorImpl*> &non_differentiable,
  const std::unordered_set<at::TensorImpl*> &dirty_inputs,
  const at::ArrayRef<Variable> raw_outputs,
  const std::shared_ptr<Node> &cdata);

// To use custom autograd operations implement a CFunction subclass with
// static backward and forward functions
//
// forward() can take as many arguments as you want and should return a
// variable list. Use of any direct Variable arguments will be registered in
// the graph but no vectors/sets or any other data structures will be traversed.
//
// backward() will be given a variable list containing as many Variables as
// there were outputs from forward. It should return as many Variables as there
// were inputs with each of them containing the gradient w.r.t. its
// corresponding input
//
// For example:
// class MyFunction : public Function<MyFunction> {
//   public:
//   static variable_list forward(AutogradContext *ctx, int n, Variable var);
//
//   static variable_list backward(AutogradContext *ctx, variable_list grad_output);
// };
// To use MyFunction
// Variable x;
// MyFunction::apply(6, x);
template <class T>
struct TORCH_API Function {
  template<typename... Args>
  static variable_list apply(Args&&... args);
};

struct AutogradContext {
  std::vector<torch::autograd::SavedVariable> saved_variables_;

  std::unordered_set<at::TensorImpl*> non_differentiable_;
  std::unordered_set<at::TensorImpl*> dirty_inputs_;
};

struct TORCH_API VariableInfo {
  explicit VariableInfo(const Variable& var);

  Variable zeros(at::OptionalDeviceGuard& device_guard) const;

  at::Backend backend = at::Backend::Undefined;
  at::Device device = at::kCPU;
  at::ScalarType scalar_type = at::kFloat;
  std::vector<int64_t> size;
  bool requires_grad;
};

template <class T>
struct CppNode : public Node {

  variable_list apply(variable_list&& inputs) override;
  AutogradContext ctx_;
  std::vector<bool> is_variable_input_;
  std::vector<VariableInfo> input_info_;
  std::vector<VariableInfo> output_info_;

  void release_variables() override;
};

template <typename T>
using enable_if_var_t = typename std::enable_if<std::is_constructible<Variable, T>::value>::type;

template <typename T>
using enable_if_not_var_t = typename std::enable_if<!std::is_constructible<Variable, T>::value>::type;

template <typename T, typename... Args>
enable_if_not_var_t<T> extract_vars(std::vector<bool> &is_var, variable_list& list, T&& cur, Args&& ... args) {
  is_var.push_back(false);
  extract_vars(is_var, list, std::forward<Args>(args)...);
}

template <typename T, typename... Args>
enable_if_var_t<T> extract_vars(std::vector<bool> &is_var, variable_list& list, T&& cur, Args&& ... args) {
  is_var.push_back(true);
  list.emplace_back(cur);
  extract_vars(is_var, list, std::forward<Args>(args)...);
}

template <typename... Args>
void extract_vars(std::vector<bool> &is_var, variable_list& list, Args&& ... args) {
}

template<class T>
template<typename... Args>
variable_list Function<T>::apply(Args&&... args) {
  std::shared_ptr<CppNode<T>> node(new CppNode<T>, deleteNode);
  variable_list input_vars;

  const size_t num_inputs = sizeof...(Args);
  input_vars.reserve(num_inputs);
  node->is_variable_input_.reserve(num_inputs);
  // TODO Add tracing here
  extract_vars(node->is_variable_input_, input_vars, args...);

  bool is_executable =  GradMode::is_enabled() && any_variable_requires_grad(input_vars);
  auto next_edges = collect_next_edges(input_vars);
  set_ctx_grad_fn(node->ctx_, node);
  node->set_next_edges(std::move(next_edges));
  node->clear_input_metadata();

  node->input_info_.reserve(input_vars.size());
  for (auto& var : input_vars) {
      node->input_info_.emplace_back(var);
  }

  variable_list outputs;
  {
    AutoGradMode grad_mode(false);
    outputs = T::forward(&node->ctx_, std::forward<Args>(args)...);
  }

  auto wrapped_outputs = _wrap_outputs(input_vars, node->ctx_.non_differentiable_, node->ctx_.dirty_inputs_, outputs, is_executable ? node : nullptr);

  node->output_info_.reserve(wrapped_outputs.size());
  for (auto& output : wrapped_outputs) {
    if (is_executable) {
      node->output_info_.emplace_back(output);
    }
  }

  return wrapped_outputs;
}

// The logic here is the same as PyNode::apply, so changes to it should be done
// in both the places
template<class T>
variable_list CppNode<T>::apply(variable_list&& inputs) {
  at::OptionalDeviceGuard _device_guard;

  int num_inputs = inputs.size();
  variable_list backward_inputs;
  backward_inputs.reserve(num_inputs);
  for (int i = 0 ; i < num_inputs; ++i) {
    if (inputs[i].defined()) {
      backward_inputs.emplace_back(inputs[i]);
    } else {
      backward_inputs.emplace_back(output_info_[i].zeros(_device_guard));
    }
  }

  auto outputs = T::backward(&ctx_, backward_inputs);

  int num_forward_inputs = is_variable_input_.size();
  int num_outputs = outputs.size();
  // Returning too many results is ok, but only as long as they're all undefined.
  // Truncate the result vector in that case.
  if (num_outputs > num_forward_inputs) {
    bool all_undef = true;
    for (int i = num_forward_inputs; i < num_outputs; ++i) {
      all_undef &= (!outputs[i].defined());
    }
    if (all_undef) {
      outputs.resize(num_forward_inputs);
      num_outputs = num_forward_inputs;
    }
  }

  if (num_outputs != num_forward_inputs) {
    std::string msg("function ");
    msg += name() + " returned an incorrect number of gradients (expected ";
    msg += std::to_string(num_forward_inputs) + ", got " ;
    msg += std::to_string(num_outputs) + ")";
    throw std::runtime_error(msg);
  }

  variable_list results;
  results.reserve(num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    if (!is_variable_input_[i]) {
      if (outputs[i].defined()) {
        std::string msg("function ");
        msg += name() + " returned a gradient different that is defined at position ";
        msg += std::to_string(i + 1) + ", but the corresponding forward input was not a Variable";
        throw std::runtime_error(msg);
      }
      continue;
    }
    if (!outputs[i].defined()) {
      auto& info = input_info_[results.size()];
      if (info.requires_grad) {
        results.emplace_back(info.zeros(_device_guard));
      } else {
        results.emplace_back();
      }
    } else {
      results.emplace_back(outputs[i]);
    }
  }
  return results;
}

template<class T>
void CppNode<T>::release_variables() {
}
}} // namespace torch::autograd
