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
// class MyFunction : public CFunction<MyFunction> {
//   public:
//   static variable_list forward(AutogradContext *ctx, int n, Variable var);
//
//   static variable_list backward(AutogradContext *ctx, variable_list grad_output);
// };
// To use MyFunction
// Variable x;
// MyFunction::apply(6, x);
template <class T>
struct Function {
  template<typename... Args>
  static variable_list apply(Args&&... args);
};

struct AutogradContext {
  std::vector<torch::autograd::SavedVariable> saved_variables;

  std::unordered_set<at::TensorImpl*> non_differentiable;
  std::unordered_set<at::TensorImpl*> dirty_inputs;
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
  AutogradContext ctx;
  std::vector<bool> is_variable_input;
  std::vector<VariableInfo> input_info;

  void release_variables() override;
};

template <typename T, typename... Args>
void extract_vars(std::vector<bool> &is_var, variable_list& list, T&& cur, Args&& ... args) {
  is_var.push_back(false);
  extract_vars(is_var, list, std::forward<Args>(args)...);
}

template <typename... Args>
void extract_vars(std::vector<bool> &is_var, variable_list& list, Variable&& cur, Args&& ... args) {
  list.push_back(cur);
  is_var.push_back(true);
  extract_vars(is_var, list, std::forward<Args>(args)...);
}

template <typename... Args>
void extract_vars(std::vector<bool> &is_var, variable_list& list, Variable& cur, Args&& ... args) {
  list.push_back(cur);
  is_var.push_back(true);
  extract_vars(is_var, list, std::forward<Args>(args)...);
}

template <typename... Args>
void extract_vars(std::vector<bool> &is_var, variable_list& list, const Variable& cur, Args&& ... args) {
  list.push_back(cur);
  is_var.push_back(true);
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
  node->is_variable_input.reserve(num_inputs);

  extract_vars(node->is_variable_input, input_vars, args...);

  bool is_executable =  GradMode::is_enabled() && any_variable_requires_grad(input_vars);
  auto next_edges = collect_next_edges(input_vars);
  set_ctx_grad_fn(node->ctx, node);
  node->set_next_edges(std::move(next_edges));
  node->clear_input_metadata();

  node->input_info.reserve(input_vars.size());
  for (auto& var : input_vars) {
      node->input_info.emplace_back(var);
  }

  variable_list outputs;
  {
    AutoGradMode grad_mode(false);
    outputs = T::forward(&node->ctx, std::forward<Args>(args)...);
  }

  return _wrap_outputs(input_vars, node->ctx.get_non_differentiable(), node->ctx.get_dirty(), outputs, is_executable ? node : nullptr);
}

template<class T>
variable_list CppNode<T>::apply(variable_list&& inputs) {
  auto outputs = T::backward(&ctx, inputs);
  at::OptionalDeviceGuard _device_guard;

  int num_forward_inputs = is_variable_input.size();
  int num_outputs = outputs.size();
  // Returning too many results is ok, but only as long as they're all undefined.
  // Truncate the result vector in that case.
  if (num_outputs > num_forward_inputs) {
    bool all_undef = true;
    for (int i = num_forward_inputs; i < num_outputs; ++i) {
      all_undef &= (!outputs[i].defined());
    }
    if (all_undef) {
      std::cout << "all undef...\n";
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
    if (!is_variable_input[i]) {
      if (outputs[i].defined()) {
        std::string msg("function ");
        msg += name() + " returned a gradient different that is defined at position ";
        msg += std::to_string(i + 1) + ", but the corresponding forward input was not a Variable";
        throw std::runtime_error(msg);
      }
      continue;
    }
    if (!outputs[i].defined()) {
      auto& info = input_info[results.size()];
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
