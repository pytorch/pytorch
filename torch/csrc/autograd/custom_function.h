#pragma once

#include <ATen/core/ivalue.h>
#include <c10/core/SymInt.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/irange.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/variable_info.h>
#include <torch/csrc/dynamo/compiled_autograd.h>
#include <vector>

namespace torch::autograd {

using optional_variable_list = std::vector<std::optional<Variable>>;
using _jvp_fn_t = std::function<variable_list(variable_list, variable_list)>;
using _view_as_self_fn_t = std::function<at::Tensor(at::Tensor)>;

TORCH_API std::vector<std::optional<Variable>> _wrap_outputs(
    const variable_list& input_vars,
    const std::unordered_set<at::TensorImpl*>& non_differentiable,
    const std::unordered_set<at::TensorImpl*>& dirty_inputs,
    const at::ArrayRef<std::optional<Variable>> raw_outputs,
    const std::shared_ptr<Node>& cdata,
    const _jvp_fn_t& jvp_user_function,
    const std::unordered_set<at::TensorImpl*>& to_save_if_setup_context,
    const _view_as_self_fn_t& view_as_self_fn,
    bool pure_view);

TORCH_API void check_variable_result(
    const at::TensorBase& original,
    const at::TensorBase& result,
    const std::string& hook_name);

// Get the return type of the forward function of the custom Function class X
template <typename X, typename... Args>
using forward_t = decltype(X::forward(nullptr, std::declval<Args>()...));

/// To use custom autograd operations, implement a Function subclass with
/// static forward and backward functions:
///
/// `forward` can take as many arguments as you want and should return either a
/// variable list or a Variable. Use of any direct Variable arguments will be
/// registered in the graph but no vectors/sets or any other data structures
/// will be traversed. You can use std::optional<Tensor> as one of the arguments
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
/// To enable compiled autograd support (torch.compile for backward) for your
/// custom autograd operation, you can set MyFunction::is_traceable
/// (see Function::istraceable notes below).
///
/// For example:
/// ```
/// class MyFunction : public Function<MyFunction> {
///   public:
///   static constexpr bool is_traceable = true;
///
///   static variable_list forward(AutogradContext *ctx, int n, Variable var) {
///      // Save data for backward in context
///      ctx->saved_data["n"] = n;
///      var.mul_(n);
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
  template <typename X = T, typename... Args>
  static auto apply(Args&&... args)
      -> std::enable_if_t<std::is_same_v<X, T>, forward_t<X, Args...>>;

  // This flag is for an experimental feature: compiled autograd. Not all
  // built-in APIs are supported at the moment e.g. mark_dirty and
  // mark_non_differentiable. Before setting this flag to enable tracing for
  // your custom function <T>, you need to ensure that the backward function is
  // traceable i.e. any variables accessed in the backward other than the input
  // arguments must be handled in a similar manner to built-ins in
  // CppNode::compiled_args and CppNode::apply_with_saved.
  static constexpr bool is_traceable = false;
};

/// Context to save information during `forward` that can be accessed in
/// `backward` in custom autograd operations (see `torch::autograd::Function`
/// for details).
struct TORCH_API AutogradContext {
  AutogradContext() = default;
  AutogradContext(const AutogradContext& other) = delete;
  AutogradContext& operator=(const AutogradContext& other) = delete;
  AutogradContext(AutogradContext&& other) = delete;
  AutogradContext& operator=(AutogradContext&& other) = delete;
  ~AutogradContext() = default;

  AutogradContext(PackedArgs& packed_args);

  /// Can be used to save non-variable data for `backward`.
  ska::flat_hash_map<std::string, at::IValue> saved_data;

  /// Saves the list of variables for a future call to `backward`. This
  /// should be called at most once from inside of `forward`.
  void save_for_backward(variable_list to_save);
  /// Marks variables in the list as modified in an in-place operation. This
  /// should be called at most once from inside of `forward` and all arguments
  /// should be inputs.
  void mark_dirty(const variable_list& inputs);
  /// Marks outputs in the list as not requiring gradients. This should be
  /// called at most once from inside of `forward` and all arguments should be
  /// outputs.
  void mark_non_differentiable(const variable_list& outputs);
  // Sets whether undefined output grad tensors should be expanded to tensors
  // full of zeros before calling backward function. Default value is true.
  void set_materialize_grads(bool value);

  /// Get the list of variables that were saved in `forward` using
  /// `save_for_backward()`. Before returning them to the user, a check is made
  /// to ensure that they were not modified by any in-place operations.
  variable_list get_saved_variables() const;
  const std::unordered_set<at::TensorImpl*>& get_and_bump_dirty() const;
  const std::unordered_set<at::TensorImpl*>& get_non_differentiable() const;

  /// Expose the Node's `task_should_compute_output` method to the cpp
  /// custom autograd Function as `needs_input_grad`.
  bool needs_input_grad(size_t output_edge_index) const;
  bool needs_input_grad(std::initializer_list<IndexRange> idxs) const;

 private:
  std::unordered_set<at::TensorImpl*> non_differentiable_;
  std::unordered_set<at::TensorImpl*> dirty_inputs_;
  std::vector<torch::autograd::SavedVariable> saved_variables_;
  variable_list to_save_;
  bool materialize_grads_{true};

  // The CppNode in the autograd graph that owns this AutogradContext. We need a
  // weak_ptr to avoid a refcycle. Since grad_fn_ owns this AutogradContext, it
  // will always be alive when we want to use it.
  std::weak_ptr<Node> grad_fn_;
  bool has_freed_buffers_{false};

  // Compiled autograd overrides saved_variables() and needs_input_grad().
  // We store the values we want to return here.
  std::optional<variable_list> saved_variables_override_;
  std::optional<std::vector<bool>> needs_input_grad_override_;

  void save_variables();

  template <class T>
  friend struct CppNode;
  template <class T>
  friend variable_list CppNode_apply_functional(
      variable_list&& inputs,
      AutogradContext& ctx_,
      const std::vector<bool>& is_variable_input_,
      const std::vector<VariableInfo>& output_info_,
      const std::string& name);
};

template <typename T>
inline variable_list CppNode_apply_functional(
    // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
    variable_list&& inputs,
    AutogradContext& ctx_,
    const std::vector<bool>& is_variable_input_,
    const std::vector<VariableInfo>& output_info_,
    const std::string& name) {
  at::OptionalDeviceGuard _device_guard;

  auto num_inputs = inputs.size();
  variable_list backward_inputs;
  backward_inputs.reserve(num_inputs);
  for (const auto i : c10::irange(num_inputs)) {
    if (inputs[i].defined() || !ctx_.materialize_grads_) {
      backward_inputs.emplace_back(std::move(inputs[i]));
    } else {
      backward_inputs.emplace_back(output_info_[i].zeros(_device_guard));
    }
  }

  auto outputs = T::backward(&ctx_, backward_inputs);

  const auto num_forward_inputs =
      static_cast<int64_t>(is_variable_input_.size());
  auto num_outputs = static_cast<int64_t>(outputs.size());
  // Returning too many results is ok, but only as long as they're all
  // undefined. Truncate the result vector in that case.
  if (num_outputs > num_forward_inputs) {
    bool all_undef = true;
    for (const auto i : c10::irange(num_forward_inputs, num_outputs)) {
      all_undef &= (!outputs[i].defined());
    }
    if (all_undef) {
      outputs.resize(num_forward_inputs);
      num_outputs = num_forward_inputs;
    }
  }

  TORCH_CHECK(
      num_outputs == num_forward_inputs,
      "function ",
      name,
      " returned an incorrect number of gradients (expected ",
      num_forward_inputs,
      ", got ",
      num_outputs,
      ")");

  variable_list results;
  results.reserve(num_outputs);
  for (const auto i : c10::irange(num_outputs)) {
    if (!is_variable_input_[i]) {
      TORCH_CHECK(
          outputs[i].defined() == false,
          "function ",
          name,
          " returned a gradient different that is defined at position ",
          i + 1,
          ", std the corresponding forward input was not a Variable");
      continue;
    }
    results.emplace_back(outputs[i]);
  }

  return results;
}

template <typename T>
inline variable_list CppNode_apply_functional_ivalue(
    const variable_list& inputs,
    const std::vector<c10::IValue>& args) {
  auto packed_args = PackedArgs(args);
  auto ctx = AutogradContext(packed_args);
  auto output_info = packed_args.unpack<std::vector<VariableInfo>>();
  auto is_variable_input = packed_args.unpack<std::vector<bool>>();
  auto name = packed_args.unpack<std::string>();
  return CppNode_apply_functional<T>(
      variable_list(inputs), ctx, is_variable_input, output_info, name);
}

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

  void set_ctx_grad_fn(const std::shared_ptr<Node>& node);
  void save_variables_to_ctx();

  void compiled_args(CompiledNodeArgs& args) const override {
    // although neither of the 2 methods below have uniqueness guarantees
    // it is unlikely for them to collide at the same time
    args.collect(static_cast<uint64_t>(typeid(T).hash_code()));
    args.collect(std::string(typeid(T).name()));

    args.collect(ctx_.saved_data);
    TORCH_INTERNAL_ASSERT(ctx_.non_differentiable_.empty());
    TORCH_INTERNAL_ASSERT(ctx_.dirty_inputs_.empty());
    args.collect(
        ctx_.saved_variables_, true); // always unpacked as output in eager
    TORCH_INTERNAL_ASSERT(ctx_.to_save_.empty());
    args.collect(ctx_.materialize_grads_);
    args.collect(ctx_.has_freed_buffers_);
    args.collect(is_variable_input_);
    args.collect(input_info_);
    args.collect(output_info_);
  }

  variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) override {
    saved.before(ctx_.saved_data);
    TORCH_INTERNAL_ASSERT(ctx_.non_differentiable_.empty());
    TORCH_INTERNAL_ASSERT(ctx_.dirty_inputs_.empty());
    saved.before(ctx_.saved_variables_);
    TORCH_INTERNAL_ASSERT(ctx_.to_save_.empty());
    saved.before(ctx_.materialize_grads_);
    saved.before(ctx_.has_freed_buffers_);
    saved.before(input_info_);
    saved.before(output_info_);

    PackedArgs packed_args;
    packed_args.pack_saved_data(ctx_.saved_data);
    variable_list saved_variables = ctx_.get_saved_variables();
    packed_args.pack(saved_variables);
    packed_args.pack(ctx_.materialize_grads_);
    packed_args.pack(ctx_.has_freed_buffers_);

    std::vector<bool> needs_input_grad;
    {
      auto ptr = ctx_.grad_fn_.lock();
      TORCH_INTERNAL_ASSERT(ptr);
      for (const auto i : c10::irange(ptr->next_edges().size())) {
        needs_input_grad.push_back(ptr->task_should_compute_output(i));
      }
    }
    packed_args.pack(needs_input_grad);

    packed_args.pack(output_info_);
    packed_args.pack(is_variable_input_);
    packed_args.pack(name());
    auto args = std::move(packed_args).vec();

    auto output_metadata = torch::dynamo::autograd::
        IValuePacker<std::vector<std::optional<InputMetadata>>>::pack(
            torch::dynamo::autograd::get_input_metadata(next_edges()));

    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();

    // Each time apply_with_saved is called, we bind a new function to Python.
    // This is because the schema might be different on compiled autograd cache
    // misses. An alternative is to pass the schema to Python so that it can be
    // an input to a function, but the schema can't be put into an FX graph
    // right now.
    std::vector<at::TypePtr> schema;
    schema.reserve(args.size());
    for (const auto& ivalue : args) {
      if (ivalue.isTensor()) {
        schema.emplace_back(at::TensorType::get());
      } else {
        schema.emplace_back(ivalue.type());
      }
    }
    static_assert(
        std::is_same_v<std::remove_cv_t<decltype(T::is_traceable)>, bool>);
    auto fn_name = pyinterface->bind_function(
        saved.get_py_compiler(),
        std::string(typeid(T).name()),
        CppNode_apply_functional_ivalue<T>,
        schema,
        /*is_custom_function*/ true,
        /*is_traceable*/ T::is_traceable);

    auto results = pyinterface->call_function(
        saved.get_py_compiler(),
        "apply_functional",
        fn_name,
        inputs,
        args,
        output_metadata);

    saved.after(ctx_.saved_data);
    TORCH_INTERNAL_ASSERT(ctx_.non_differentiable_.empty());
    TORCH_INTERNAL_ASSERT(ctx_.dirty_inputs_.empty());
    saved.after(ctx_.saved_variables_);
    TORCH_INTERNAL_ASSERT(ctx_.to_save_.empty());
    saved.after(ctx_.materialize_grads_);
    saved.after(ctx_.has_freed_buffers_);
    saved.after(input_info_);
    saved.after(output_info_);
    return results;
  }
};

struct ExtractVariables : IterArgs<ExtractVariables> {
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  std::vector<bool>& is_var_;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  variable_list& list_;
  ExtractVariables(std::vector<bool>& is_var, variable_list& list)
      : is_var_(is_var), list_(list) {}
  void operator()(const std::optional<at::Tensor>& x) {
    if (x.has_value() && x.value().defined()) {
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
  void operator()(const at::TensorList& list) {
    for (const at::Tensor& x : list) {
      is_var_.push_back(true);
      list_.emplace_back(x);
    }
  }
  template <typename T>
  void operator()(const T& x) {
    is_var_.push_back(false);
  }
};

template <typename... Args>
inline void extract_vars(
    std::vector<bool>& is_var,
    variable_list& list,
    Args&&... args) {
  ExtractVariables(is_var, list).apply(std::forward<Args>(args)...);
}

template <typename T>
std::enable_if_t<std::is_same_v<T, variable_list>, T> to_output_type(
    std::vector<std::optional<Variable>>& output_list) {
  variable_list result;
  std::transform(
      output_list.begin(),
      output_list.end(),
      std::back_inserter(result),
      [](const std::optional<Variable>& var) { return *var; });
  return result;
}

template <typename T>
std::enable_if_t<std::is_same_v<T, Variable>, T> to_output_type(
    std::vector<std::optional<Variable>>& output_list) {
  return *output_list[0];
}

inline std::vector<std::optional<Variable>> to_optional(Variable& output) {
  return std::vector<std::optional<Variable>>{output};
}

inline std::vector<std::optional<Variable>> to_optional(variable_list& output) {
  std::vector<std::optional<Variable>> result;
  std::transform(
      output.begin(),
      output.end(),
      std::back_inserter(result),
      [](const Variable& var) { return var; });
  return result;
}

template <class T>
template <typename X, typename... Args>
auto Function<T>::apply(Args&&... args)
    -> std::enable_if_t<std::is_same_v<X, T>, forward_t<X, Args...>> {
  const auto& functorch_tls = at::functorch::functorchTLSAccessor();
  if (functorch_tls) {
    // Function support for functorch is handled in Python.
    // Here we are dealing with a (C++) Function, which is not supported.
    // Let's raise an error instead of being silently incorrect.
    functorch_tls->checkSupportsCppAutogradFunction();
  }

  std::shared_ptr<CppNode<T>> node(new CppNode<T>(), deleteNode);
  variable_list input_vars;

  const size_t num_inputs = sizeof...(Args);
  input_vars.reserve(num_inputs);
  node->is_variable_input_.reserve(num_inputs);
  // TODO Add tracing here
  extract_vars(node->is_variable_input_, input_vars, args...);

  bool is_executable =
      GradMode::is_enabled() && any_variable_requires_grad(input_vars);
  auto next_edges =
      (is_executable ? collect_next_edges(input_vars) : edge_list());
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

  _jvp_fn_t jvp_fn = [](const variable_list& inputs,
                        const variable_list& gI) -> variable_list {
    TORCH_CHECK(
        false,
        "jvp is not implemented for the c++ API of custom Function yet.",
        "Please open a feature request on GitHub if you need this.");
  };

  auto view_as_self_fn = [](const at::Tensor& x) -> at::Tensor {
    return x.view_as(x);
  };

  auto wrapped_outputs = _wrap_outputs(
      input_vars,
      node->ctx_.get_non_differentiable(),
      node->ctx_.get_and_bump_dirty(),
      to_optional(outputs),
      is_executable ? node : nullptr,
      jvp_fn,
      {},
      view_as_self_fn,
      false);

  node->output_info_.reserve(wrapped_outputs.size());
  for (auto& output : wrapped_outputs) {
    if (is_executable && output.has_value()) {
      node->output_info_.emplace_back(output.value());
    } else if (is_executable) {
      node->output_info_.emplace_back();
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
template <class T>
// NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
variable_list CppNode<T>::apply(variable_list&& inputs) {
  // Acquire lock to here protect thread safety on custom C++ Autograd Node
  // This is needed for the custom Autograd Node since we don't know if the
  // user defined Node will write to the shared data during backward.
  // see Note [Thread Safety on Autograd Node]
  std::lock_guard<std::mutex> lock(mutex_);
  return CppNode_apply_functional<T>(
      std::move(inputs), ctx_, is_variable_input_, output_info_, name());
}

template <class T>
void CppNode<T>::release_variables() {
  // lock to ensure thread safety, see [Thread Safety on Autograd Node]
  std::lock_guard<std::mutex> lock(mutex_);
  ctx_.saved_variables_.clear();
  ctx_.has_freed_buffers_ = true;
}

template <class T>
void CppNode<T>::save_variables_to_ctx() {
  ctx_.save_variables();
}

template <class T>
void CppNode<T>::set_ctx_grad_fn(const std::shared_ptr<Node>& node) {
  ctx_.grad_fn_ = node;
}

} // namespace torch::autograd
