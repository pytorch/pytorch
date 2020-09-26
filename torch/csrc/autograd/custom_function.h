#pragma once

#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include <ATen/core/ivalue.h>
#include <c10/util/flat_hash_map.h>
#include <vector>

namespace torch { namespace autograd {

TORCH_API variable_list _wrap_outputs(
  const variable_list &input_vars,
  const std::unordered_set<at::TensorImpl*> &non_differentiable,
  const std::unordered_set<at::TensorImpl*> &dirty_inputs,
  const at::ArrayRef<Variable> raw_outputs,
  const std::shared_ptr<Node> &cdata);

TORCH_API void check_variable_result(const Variable& original,
  const Variable& result, std::string hook_name);

// Get the return type of the forward function of the custom Function class X
template<typename X, typename... Args>
using forward_t = decltype(X::forward(nullptr, std::declval<Args>()...));

/// To use custom autograd operations, implement a Function subclass with
/// static forward and backward functions:
///
/// `forward` can take as many arguments as you want and should return either a
/// variable list or a Variable. Use of any direct Variable arguments will be
/// registered in the graph but no vectors/sets or any other data structures
/// will be traversed. You can use c10::optional<Tensor> as one of the arguments
/// and it will be registered as a variable in the graph if the argument has a
/// value. It should take a pointer to `torch::autograd::AutogradContext` as the
/// first argument. Variables can be saved in the `ctx` using
/// `ctx->save_for_backward`
/// (see `torch::autograd::AutogradContext::save_for_backward`) and other data
/// can be saved in the `ctx->saved_data` map
/// (see `torch::autograd::AutogradContext::saved_data`)
/// in the form of `<std::string, at::IValue>` pairs.
///
/// `backward` should take a pointer to `torch::autograd::AutogradContext`
/// and a variable list containing as many Variables as there were outputs from
/// `forward` as arguments. It should return as many Variables as there were
/// inputs with each of them containing the gradient w.r.t. its corresponding
/// input. Variables saved in `forward` can be accessed with
/// `ctx->get_saved_variables` (see
/// `torch::autograd::AutogradContext::get_saved_variables`) and other saved
/// data can be accessed from `ctx->saved_data`.
///
/// For example:
/// ```
/// class MyFunction : public Function<MyFunction> {
///   public:
///   static variable_list forward(AutogradContext *ctx, int n, Variable var) {
///      // Save data for backward in context
///      ctx->saved_data["n"] = n;
///      var.mul_(2);
///      // Mark var as modified by inplace operation
///      ctx->mark_dirty({var});
///      return {var};
///   }
///
///   static variable_list backward(AutogradContext *ctx, variable_list
///   grad_output) {
///      // Use data saved in forward
///      auto n = ctx->saved_data["n"].toInt();
///      return {grad_output[0]*n};
///   }
/// };
/// ```
///
/// To use `MyFunction`:
/// ```
/// Variable x;
/// auto y = MyFunction::apply(6, x);
/// // Example backward call
/// y[0].sum().backward();
/// ```
template <class T>
struct TORCH_API Function {
  // We need to use a different template parameter than T here because T will
  // inherit from Function, and when Function<T> is instantiated, T::forward
  // is not declared yet.
  // The enable_if check is to ensure that the user doesn't explicitly provide
  // the parameter X.
  template<typename X=T, typename... Args>
  static auto apply(Args&&... args) -> std::enable_if_t<std::is_same<X,T>::value, forward_t<X,Args...>>;
};

/// Context to save information during `forward` that can be accessed in `backward`
/// in custom autograd operations (see `torch::autograd::Function` for details).
struct TORCH_API AutogradContext {
  AutogradContext() : materialize_grads_(true) {}
  AutogradContext(const AutogradContext &other) = delete;
  AutogradContext& operator=(const AutogradContext& other) = delete;

  /// Can be used to save non-variable data for `backward`.
  ska::flat_hash_map<std::string, at::IValue> saved_data;

  /// Saves the list of variables for a future call to `backward`. This
  /// should be called at most once from inside of `forward`.
  void save_for_backward(variable_list to_save);
  /// Marks variables in the list as modified in an in-place operation. This
  /// should be called at most once from inside of `forward` and all arguments
  /// should be inputs.
  void mark_dirty(const variable_list &inputs);
  /// Marks outputs in the list as not requiring gradients. This should be called
  /// at most once from inside of `forward` and all arguments should be outputs.
  void mark_non_differentiable(const variable_list &outputs);
  // Sets whether undefined output grad tensors should be expanded to tensors
  // full of zeros before calling backward function. Default value is true.
  void set_materialize_grads(bool value);

  /// Get the list of variables that were saved in `forward` using
  /// `save_for_backward()`. Before returning them to the user, a check is made to
  /// ensure that they were not modified by any in-place operations.
  variable_list get_saved_variables() const;
  const std::unordered_set<at::TensorImpl*>& get_and_bump_dirty() const;
  const std::unordered_set<at::TensorImpl*>& get_non_differentiable() const;

private:
  std::unordered_set<at::TensorImpl*> non_differentiable_;
  std::unordered_set<at::TensorImpl*> dirty_inputs_;
  std::vector<torch::autograd::SavedVariable> saved_variables_;
  variable_list to_save_;
  bool materialize_grads_;

  // The CppNode in the autograd graph that owns this AutogradContext. We need a
  // weak_ptr to avoid a refcycle. Since grad_fn_ owns this AutogradContext, it
  // will always be alive when we want to use it.
  std::weak_ptr<Node> grad_fn_;
  bool has_freed_buffers_;

  void save_variables();

  template <class T> friend struct CppNode;
};

struct TORCH_API VariableInfo {
  explicit VariableInfo(const Variable& var);

  Variable zeros(at::OptionalDeviceGuard& device_guard) const;

  at::Layout layout = at::Layout::Strided;
  at::Device device = at::kCPU;
  at::ScalarType scalar_type = at::kFloat;
  std::vector<int64_t> size;
  bool requires_grad;
};

// CppNode<T> is the Node in the autograd graph that represents the user defined
// backward function for Function<T>. Calls to CppNode::apply are forward to
// T::backward().
template <class T>
struct CppNode : public Node {

  variable_list apply(variable_list&& inputs) override;
  AutogradContext ctx_;
  std::vector<bool> is_variable_input_;
  std::vector<VariableInfo> input_info_;
  std::vector<VariableInfo> output_info_;

  void release_variables() override;

  void set_ctx_grad_fn(const std::shared_ptr<Node> &node);
  void save_variables_to_ctx();
};

struct ExtractVariables : IterArgs<ExtractVariables> {
  std::vector<bool>& is_var_;
  variable_list& list_;
  ExtractVariables(std::vector<bool>& is_var, variable_list& list) : is_var_(is_var), list_(list) {}
  void operator()(const c10::optional<at::Tensor>& x) {
    if (x) {
      is_var_.push_back(true);
      list_.emplace_back(x.value());
    } else {
      is_var_.push_back(false);
    }
  }
  void operator()(const at::Tensor& x) {
    is_var_.push_back(true);
    list_.emplace_back(x);
  }
  template <typename T>
  void operator()(const T& x) {
    is_var_.push_back(false);
  }
};

template <typename... Args>
inline void extract_vars(std::vector<bool> &is_var, variable_list& list, Args&&... args) {
  ExtractVariables(is_var, list).apply(std::forward<Args>(args)...);
}

template <typename T>
typename std::enable_if<std::is_same<T, variable_list>::value, T&>::type to_output_type(variable_list& output_list) { return output_list; }

template <typename T>
typename std::enable_if<std::is_same<T, Variable>::value, T>::type to_output_type(variable_list& output_list) { return output_list[0]; }

template<class T>
template<typename X, typename... Args>
auto Function<T>::apply(Args&&... args) -> std::enable_if_t<std::is_same<X,T>::value, forward_t<X,Args...>> {
  std::shared_ptr<CppNode<T>> node(new CppNode<T>(), deleteNode);
  variable_list input_vars;

  const size_t num_inputs = sizeof...(Args);
  input_vars.reserve(num_inputs);
  node->is_variable_input_.reserve(num_inputs);
  // TODO Add tracing here
  extract_vars(node->is_variable_input_, input_vars, args...);

  bool is_executable =  GradMode::is_enabled() && any_variable_requires_grad(input_vars);
  auto next_edges = collect_next_edges(input_vars);
  node->set_ctx_grad_fn(node);
  node->set_next_edges(std::move(next_edges));
  node->clear_input_metadata();

  node->input_info_.reserve(input_vars.size());
  for (auto& var : input_vars) {
      node->input_info_.emplace_back(var);
  }

  using forward_return_t = forward_t<X, Args...>;
  forward_return_t outputs;
  {
    AutoGradMode grad_mode(false);
    outputs = T::forward(&node->ctx_, std::forward<Args>(args)...);
  }

  auto wrapped_outputs = _wrap_outputs(input_vars, node->ctx_.get_non_differentiable(), node->ctx_.get_and_bump_dirty(), outputs, is_executable ? node : nullptr);

  node->output_info_.reserve(wrapped_outputs.size());
  for (auto& output : wrapped_outputs) {
    if (is_executable) {
      node->output_info_.emplace_back(output);
    }
  }

  if (is_executable) {
    node->save_variables_to_ctx();
  }

  // wrapped_outputs will be a variable_list so, convert it to the correct
  // return type. Only Variable and variable_list are accepted as return types.
 return to_output_type<forward_return_t>(wrapped_outputs);
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
    if (inputs[i].defined() || !ctx_.materialize_grads_) {
      backward_inputs.emplace_back(inputs[i]);
    } else {
      backward_inputs.emplace_back(output_info_[i].zeros(_device_guard));
    }
  }

  // Acquire lock to here protect thread safety on custom C++ Autograd Node
  // This is needed for the custom Autograd Node since we don't know if the
  // user defined Node will write to the shared data during backward.
  // see Note [Thread Safety on Autograd Node]
  std::lock_guard<std::mutex> lock(mutex_);

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
    msg += c10::to_string(num_forward_inputs) + ", got " ;
    msg += c10::to_string(num_outputs) + ")";
    throw std::runtime_error(msg);
  }

  variable_list results;
  results.reserve(num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    if (!is_variable_input_[i]) {
      if (outputs[i].defined()) {
        std::string msg("function ");
        msg += name() + " returned a gradient different that is defined at position ";
        msg += c10::to_string(i + 1) + ", but the corresponding forward input was not a Variable";
        throw std::runtime_error(msg);
      }
      continue;
    }
    results.emplace_back(outputs[i]);
  }
  return results;
}

template<class T>
void CppNode<T>::release_variables() {
  // lock to ensure thread safety, see [Thread Safety on Autograd Node]
  std::lock_guard<std::mutex> lock(mutex_);
  ctx_.saved_variables_.clear();
  ctx_.has_freed_buffers_ = true;
}

template<class T>
void CppNode<T>::save_variables_to_ctx() {
  ctx_.save_variables();
}

template<class T>
void CppNode<T>::set_ctx_grad_fn(const std::shared_ptr<Node> &node) {
  ctx_.grad_fn_ = node;
}

}} // namespace torch::autograd
