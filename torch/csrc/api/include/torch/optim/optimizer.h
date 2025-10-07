#pragma once

#include <ATen/Tensor.h>
#include <c10/util/Exception.h>
#include <c10/util/flat_hash_map.h>

#include <torch/arg.h>
#include <torch/csrc/Export.h>

#include <algorithm>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

// Forward declarations confuse Doxygen
#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace at {
class Tensor;
} // namespace at

namespace torch {
using at::Tensor;
namespace serialize {
class OutputArchive;
class InputArchive;
} // namespace serialize
} // namespace torch
#endif // DOXYGEN_SHOULD_SKIP_THIS

namespace torch::optim {

class TORCH_API OptimizerParamState {
 public:
  OptimizerParamState() = default;
  OptimizerParamState(const OptimizerParamState&) = default;
  OptimizerParamState& operator=(const OptimizerParamState&) = default;
  OptimizerParamState(OptimizerParamState&&) noexcept = default;
  OptimizerParamState& operator=(OptimizerParamState&&) noexcept = default;
  virtual std::unique_ptr<OptimizerParamState> clone() const;
  virtual void serialize(torch::serialize::InputArchive& archive);
  virtual void serialize(torch::serialize::OutputArchive& archive) const;
  virtual ~OptimizerParamState() = default;
};

template <typename Derived>
class OptimizerCloneableParamState : public OptimizerParamState {
  std::unique_ptr<OptimizerParamState> clone() const override {
    return std::make_unique<Derived>(static_cast<const Derived&>(*this));
  }
};

class TORCH_API OptimizerOptions {
 public:
  OptimizerOptions() = default;
  OptimizerOptions(const OptimizerOptions&) = default;
  OptimizerOptions& operator=(const OptimizerOptions&) = default;
  OptimizerOptions(OptimizerOptions&&) noexcept = default;
  OptimizerOptions& operator=(OptimizerOptions&&) noexcept = default;
  virtual std::unique_ptr<OptimizerOptions> clone() const;
  virtual void serialize(torch::serialize::InputArchive& archive);
  virtual void serialize(torch::serialize::OutputArchive& archive) const;
  virtual ~OptimizerOptions() = default;
  virtual double get_lr() const;
  virtual void set_lr(const double lr);
};

// Forward declarations for optimizer option types
struct SGDOptions;
struct AdamOptions;
struct AdamWOptions;
struct AdagradOptions;
struct RMSpropOptions;
struct LBFGSOptions;

/**
 * OptimizerCloneableOptions provides parameter group inheritance functionality
 * for PyTorch C++ optimizer options. When creating parameter groups with
 * partial options (e.g., AdamOptions().weight_decay(0.1)), fields not
 * explicitly set by the user inherit from the optimizer's default values,
 * while explicitly set fields are preserved.
 *
 * This enables Python-like behavior in C++:
 * ```cpp
 * // Python equivalent:
 * // optimizer = Adam([{'params': params1, 'weight_decay': 0.1}], lr=0.01)
 * // Result: weight_decay=0.1 preserved, lr=0.01 inherited
 *
 * AdamOptions defaults;
 * defaults.lr(0.01).weight_decay(0.05);
 *
 * std::vector<OptimizerParamGroup> groups;
 * groups.emplace_back(params1, std::make_unique<AdamOptions>(
 *     AdamOptions().weight_decay(0.1)));  // Only weight_decay specified
 *
 * Adam optimizer(groups, defaults);
 * // Result: group inherits lr=0.01, preserves weight_decay=0.1
 * ```
 *
 * **Implementation**: Uses SFINAE-based field detection and constructor-default
 * comparison to distinguish explicitly set fields from default values.
 * Fields that match constructor defaults are inherited; others are preserved.
 */
template <typename Derived>
class OptimizerCloneableOptions : public OptimizerOptions {
 private:
  std::unique_ptr<OptimizerOptions> clone() const override {
    return std::make_unique<Derived>(static_cast<const Derived&>(*this));
  }

  // SFINAE field detection - detects optimizer fields using public accessor
  // methods
  template <class T, class Enable = void>
  struct _has_lr : std::false_type {};
  template <class T>
  struct _has_lr<T, std::void_t<decltype(std::declval<const T&>().lr())>>
      : std::true_type {};

  template <class T, class Enable = void>
  struct _has_momentum : std::false_type {};
  template <class T>
  struct _has_momentum<
      T,
      std::void_t<decltype(std::declval<const T&>().momentum())>>
      : std::true_type {};

  template <class T, class Enable = void>
  struct _has_weight_decay : std::false_type {};
  template <class T>
  struct _has_weight_decay<
      T,
      std::void_t<decltype(std::declval<const T&>().weight_decay())>>
      : std::true_type {};

  template <class T, class Enable = void>
  struct _has_dampening : std::false_type {};
  template <class T>
  struct _has_dampening<
      T,
      std::void_t<decltype(std::declval<const T&>().dampening())>>
      : std::true_type {};

  template <class T, class Enable = void>
  struct _has_nesterov : std::false_type {};
  template <class T>
  struct _has_nesterov<
      T,
      std::void_t<decltype(std::declval<const T&>().nesterov())>>
      : std::true_type {};

  template <class T, class Enable = void>
  struct _has_betas : std::false_type {};
  template <class T>
  struct _has_betas<T, std::void_t<decltype(std::declval<const T&>().betas())>>
      : std::true_type {};

  template <class T, class Enable = void>
  struct _has_eps : std::false_type {};
  template <class T>
  struct _has_eps<T, std::void_t<decltype(std::declval<const T&>().eps())>>
      : std::true_type {};

  template <class T, class Enable = void>
  struct _has_amsgrad : std::false_type {};
  template <class T>
  struct _has_amsgrad<
      T,
      std::void_t<decltype(std::declval<const T&>().amsgrad())>>
      : std::true_type {};

  // Optimizer-specific field detection
  template <class T, class Enable = void>
  struct _has_lr_decay : std::false_type {};
  template <class T>
  struct _has_lr_decay<
      T,
      std::void_t<decltype(std::declval<const T&>().lr_decay())>>
      : std::true_type {};

  template <class T, class Enable = void>
  struct _has_alpha : std::false_type {};
  template <class T>
  struct _has_alpha<T, std::void_t<decltype(std::declval<const T&>().alpha())>>
      : std::true_type {};

  template <class T, class Enable = void>
  struct _has_centered : std::false_type {};
  template <class T>
  struct _has_centered<
      T,
      std::void_t<decltype(std::declval<const T&>().centered())>>
      : std::true_type {};

  template <class T, class Enable = void>
  struct _has_initial_accumulator_value : std::false_type {};
  template <class T>
  struct _has_initial_accumulator_value<
      T,
      std::void_t<
          decltype(std::declval<const T&>().initial_accumulator_value())>>
      : std::true_type {};

  // LBFGS-specific fields with appropriate types
  template <class T, class Enable = void>
  struct _has_max_iter : std::false_type {};
  template <class T>
  struct _has_max_iter<
      T,
      std::void_t<decltype(std::declval<const T&>().max_iter())>>
      : std::true_type {};

  template <class T, class Enable = void>
  struct _has_max_eval : std::false_type {};
  template <class T>
  struct _has_max_eval<
      T,
      std::void_t<decltype(std::declval<const T&>().max_eval())>>
      : std::true_type {};

  template <class T, class Enable = void>
  struct _has_tolerance_grad : std::false_type {};
  template <class T>
  struct _has_tolerance_grad<
      T,
      std::void_t<decltype(std::declval<const T&>().tolerance_grad())>>
      : std::true_type {};

  template <class T, class Enable = void>
  struct _has_tolerance_change : std::false_type {};
  template <class T>
  struct _has_tolerance_change<
      T,
      std::void_t<decltype(std::declval<const T&>().tolerance_change())>>
      : std::true_type {};

  template <class T, class Enable = void>
  struct _has_history_size : std::false_type {};
  template <class T>
  struct _has_history_size<
      T,
      std::void_t<decltype(std::declval<const T&>().history_size())>>
      : std::true_type {};

  template <class T, class Enable = void>
  struct _has_line_search_fn : std::false_type {};
  template <class T>
  struct _has_line_search_fn<
      T,
      std::void_t<decltype(std::declval<const T&>().line_search_fn())>>
      : std::true_type {};

  /**
   * Merges user-specified options with optimizer defaults using
   * constructor-default comparison to detect explicitly set fields.
   *
   * Algorithm:
   * 1. Start with optimizer defaults as base
   * 2. Create fresh constructor instance for comparison
   * 3. If user_value != constructor_default → user explicitly set it → preserve
   * 4. If user_value == constructor_default → user didn't set it → inherit from
   * defaults
   */
  void _merge_by_comparison(
      const Derived& defaults,
      const Derived& user_options) {
    auto* result = static_cast<Derived*>(this);
    *result = defaults; // Start with optimizer defaults

    // Create constructor defaults instance for comparison
    Derived constructor_defaults = []() {
      if constexpr (std::is_default_constructible_v<Derived>) {
        return Derived{};
      } else {
        // Handle optimizers requiring constructor parameters
        if constexpr (std::is_same_v<Derived, SGDOptions>) {
          return Derived(1e-3);
        } else if constexpr (std::is_same_v<Derived, AdagradOptions>) {
          return Derived(1e-2);
        } else if constexpr (std::is_same_v<Derived, RMSpropOptions>) {
          return Derived(1e-2);
        } else if constexpr (std::is_same_v<Derived, LBFGSOptions>) {
          return Derived(1);
        } else {
          return Derived{};
        }
      }
    }();

    // Merge fields: preserve user-set values, inherit defaults for unset values

    if constexpr (_has_lr<Derived>::value) {
      if (user_options.lr() != constructor_defaults.lr()) {
        result->lr(user_options.lr());
      }
    }
    if constexpr (_has_momentum<Derived>::value) {
      if (user_options.momentum() != constructor_defaults.momentum()) {
        result->momentum(user_options.momentum());
      }
    }
    if constexpr (_has_weight_decay<Derived>::value) {
      if (user_options.weight_decay() != constructor_defaults.weight_decay()) {
        result->weight_decay(user_options.weight_decay());
      }
    }
    if constexpr (_has_dampening<Derived>::value) {
      if (user_options.dampening() != constructor_defaults.dampening()) {
        result->dampening(user_options.dampening());
      }
    }
    if constexpr (_has_nesterov<Derived>::value) {
      if (user_options.nesterov() != constructor_defaults.nesterov()) {
        result->nesterov(user_options.nesterov());
      }
    }
    if constexpr (_has_betas<Derived>::value) {
      if (user_options.betas() != constructor_defaults.betas()) {
        result->betas(user_options.betas());
      }
    }
    if constexpr (_has_eps<Derived>::value) {
      if (user_options.eps() != constructor_defaults.eps()) {
        result->eps(user_options.eps());
      }
    }
    if constexpr (_has_amsgrad<Derived>::value) {
      if (user_options.amsgrad() != constructor_defaults.amsgrad()) {
        result->amsgrad(user_options.amsgrad());
      }
    }

    // Optimizer-specific fields - automatically detected and handled
    if constexpr (_has_lr_decay<Derived>::value) {
      if (user_options.lr_decay() != constructor_defaults.lr_decay()) {
        result->lr_decay(user_options.lr_decay());
      }
    }
    if constexpr (_has_alpha<Derived>::value) {
      if (user_options.alpha() != constructor_defaults.alpha()) {
        result->alpha(user_options.alpha());
      }
    }
    if constexpr (_has_centered<Derived>::value) {
      if (user_options.centered() != constructor_defaults.centered()) {
        result->centered(user_options.centered());
      }
    }
    if constexpr (_has_initial_accumulator_value<Derived>::value) {
      if (user_options.initial_accumulator_value() !=
          constructor_defaults.initial_accumulator_value()) {
        result->initial_accumulator_value(
            user_options.initial_accumulator_value());
      }
    }

    // LBFGS-specific fields with appropriate types
    if constexpr (_has_max_iter<Derived>::value) {
      if (user_options.max_iter() != constructor_defaults.max_iter()) {
        result->max_iter(user_options.max_iter());
      }
    }
    if constexpr (_has_max_eval<Derived>::value) {
      if (user_options.max_eval() != constructor_defaults.max_eval()) {
        result->max_eval(user_options.max_eval());
      }
    }
    if constexpr (_has_tolerance_grad<Derived>::value) {
      if (user_options.tolerance_grad() !=
          constructor_defaults.tolerance_grad()) {
        result->tolerance_grad(user_options.tolerance_grad());
      }
    }
    if constexpr (_has_tolerance_change<Derived>::value) {
      if (user_options.tolerance_change() !=
          constructor_defaults.tolerance_change()) {
        result->tolerance_change(user_options.tolerance_change());
      }
    }
    if constexpr (_has_history_size<Derived>::value) {
      if (user_options.history_size() != constructor_defaults.history_size()) {
        result->history_size(user_options.history_size());
      }
    }
    if constexpr (_has_line_search_fn<Derived>::value) {
      if (user_options.line_search_fn() !=
          constructor_defaults.line_search_fn()) {
        result->line_search_fn(user_options.line_search_fn());
      }
    }
  }

  // Friend class for controlled access to private _merge_by_comparison method
  friend class Optimizer;
};

/// Stores parameters in the param_group and stores a pointer to the
/// OptimizerOptions
class TORCH_API OptimizerParamGroup {
 public:
  // NOTE: In order to store `OptimizerParamGroup` in a `std::vector`, it has to
  // be copy-constructible.
  OptimizerParamGroup(const OptimizerParamGroup& param_group)
      : params_(param_group.params()),
        options_(
            param_group.has_options() ? param_group.options().clone()
                                      : nullptr) {}
  OptimizerParamGroup(OptimizerParamGroup&& param_group) = default;
  OptimizerParamGroup(std::vector<Tensor> params)
      : params_(std::move(params)) {}
  OptimizerParamGroup(
      std::vector<Tensor> params,
      std::unique_ptr<OptimizerOptions> options)
      : params_(std::move(params)), options_(std::move(options)) {}

  OptimizerParamGroup& operator=(const OptimizerParamGroup& param_group) =
      delete;
  OptimizerParamGroup& operator=(OptimizerParamGroup&& param_group) noexcept =
      default;
  ~OptimizerParamGroup() = default;
  bool has_options() const;
  OptimizerOptions& options();
  const OptimizerOptions& options() const;
  void set_options(std::unique_ptr<OptimizerOptions> options);
  std::vector<Tensor>& params();
  const std::vector<Tensor>& params() const;

 protected:
  std::vector<Tensor> params_;
  std::unique_ptr<OptimizerOptions> options_;
};

class TORCH_API Optimizer {
 public:
  // The copy constructor is deleted, because the user should use the
  // `state_dict` / `load_state_dict` API to copy an optimizer instead.
  Optimizer(const Optimizer& optimizer) = delete;
  Optimizer(Optimizer&& optimizer) = default;
  Optimizer& operator=(const Optimizer& optimizer) = delete;
  Optimizer& operator=(Optimizer&& optimizer) = default;

  explicit Optimizer(
      const std::vector<OptimizerParamGroup>& param_groups,
      std::unique_ptr<OptimizerOptions> defaults)
      : defaults_(std::move(defaults)) {
    for (const auto& param_group : param_groups) {
      add_param_group(param_group);
    }
  }

  /// Constructs the `Optimizer` from a vector of parameters.
  explicit Optimizer(
      std::vector<Tensor> parameters,
      std::unique_ptr<OptimizerOptions> defaults)
      : Optimizer(
            {OptimizerParamGroup(std::move(parameters))},
            std::move(defaults)) {}

  /// Adds the given param_group to the optimizer's param_group list.
  void add_param_group(const OptimizerParamGroup& param_group);

  virtual ~Optimizer() = default;

  using LossClosure = std::function<Tensor()>;
  /// A loss function closure, which is expected to return the loss value.
  virtual Tensor step(LossClosure closure = nullptr) = 0;

  /// Adds the given vector of parameters to the optimizer's parameter list.
  void add_parameters(const std::vector<Tensor>& parameters);

  /// Zeros out the gradients of all parameters.
  void zero_grad(bool set_to_none = true);

  /// Provides a const reference to the parameters in the first param_group this
  /// optimizer holds.
  const std::vector<Tensor>& parameters() const noexcept;

  /// Provides a reference to the parameters in the first param_group this
  /// optimizer holds.
  std::vector<Tensor>& parameters() noexcept;

  /// Returns the number of parameters referenced by the optimizer.
  size_t size() const noexcept;

  OptimizerOptions& defaults() noexcept;

  const OptimizerOptions& defaults() const noexcept;

  /// Provides a reference to the param_groups this optimizer holds.
  std::vector<OptimizerParamGroup>& param_groups() noexcept;

  /// Provides a const reference to the param_groups this optimizer holds.
  const std::vector<OptimizerParamGroup>& param_groups() const noexcept;

  /// Provides a reference to the state this optimizer holds
  ska::flat_hash_map<void*, std::unique_ptr<OptimizerParamState>>&
  state() noexcept;

  /// Provides a const reference to the state this optimizer holds
  const ska::flat_hash_map<void*, std::unique_ptr<OptimizerParamState>>& state()
      const noexcept;

  /// Serializes the optimizer state into the given `archive`.
  virtual void save(serialize::OutputArchive& archive) const;

  /// Deserializes the optimizer state from the given `archive`.
  virtual void load(serialize::InputArchive& archive);

 private:
  /// Helper function to try merging for a specific optimizer type
  template <typename OptimizerType>
  static bool _try_merge_optimizer_type(
      std::unique_ptr<OptimizerOptions>& final_options,
      const OptimizerOptions& user_options,
      const OptimizerOptions& defaults) {
    auto* typed_final = dynamic_cast<OptimizerType*>(final_options.get());
    auto* typed_user = dynamic_cast<const OptimizerType*>(&user_options);
    auto* typed_defaults = dynamic_cast<const OptimizerType*>(&defaults);

    if (typed_final && typed_user && typed_defaults) {
      typed_final->_merge_by_comparison(*typed_defaults, *typed_user);
      return true;
    }
    return false;
  }

  /// Simple variadic dispatch helper - try all optimizer types in one call
  template <typename... OptimizerTypes>
  static void _try_merge_all_optimizer_types(
      std::unique_ptr<OptimizerOptions>& final_options,
      const OptimizerOptions& user_options,
      const OptimizerOptions& defaults) {
    // Try each optimizer type until one succeeds - much cleaner than manual
    // chain
    (void)(_try_merge_optimizer_type<OptimizerTypes>(
               final_options, user_options, defaults) ||
           ...);
  }

  /// Convenience function with all known PyTorch optimizers
  static void _try_merge_all_optimizers(
      std::unique_ptr<OptimizerOptions>& final_options,
      const OptimizerOptions& user_options,
      const OptimizerOptions& defaults);

 protected:
  std::vector<OptimizerParamGroup> param_groups_;
  ska::flat_hash_map<void*, std::unique_ptr<OptimizerParamState>> state_;
  std::unique_ptr<OptimizerOptions> defaults_;
};

/* How do we decide whether to serialize undefined tensors or
  std::nullopt values into the output archive?
Answer: we strictly follow the behavior of Python API. To be more specific:

For optimizer options:
a) For undefined tensor: currently no tensor is used as an options argument in
Python API, so we don't need to worry about it now. b) For std::nullopt value:
we serialize std::nullopt values into the output archive, to follow the exact
same behavior as Python API.

For optimizer param state:
a) For undefined tensor: in param state, undefined tensor in C++ impl is
equivalent to missing key in Python impl. Since we don't serialize missing keys
in Python API, we skip undefined tensors when serializing the param state. b)
For std::nullopt value: in param state, std::nullopt value in C++ impl is
equivalent to missing key in Python impl. Since we don't serialize missing keys
in Python API, we skip std::nullopt values when serializing the param state. */

/// Serializes an `Optimizer` into an `OutputArchive`.
TORCH_API serialize::OutputArchive& operator<<(
    serialize::OutputArchive& archive,
    const Optimizer& optimizer);

/// Deserializes a `Tensor` from an `InputArchive`.
TORCH_API serialize::InputArchive& operator>>(
    serialize::InputArchive& archive,
    Optimizer& optimizer);

} // namespace torch::optim
