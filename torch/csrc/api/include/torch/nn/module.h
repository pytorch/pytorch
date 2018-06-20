#pragma once

#include <torch/detail/ordered_dict.h>
#include <torch/nn/cursor.h>
#include <torch/nn/pimpl.h>
#include <torch/tensor.h>

#include <ATen/optional.h>

#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>

namespace torch {
namespace detail {
template <typename T>
class CursorBase;
} // namespace detail
} // namespace torch

namespace torch {
namespace nn {

class Module {
 public:
  /// Tells the base `Module` about the name of the submodule.
  explicit Module(std::string name);

  /// Constructs the base module without immediate knowledge of the submodule's
  /// name. The name of the submodule is inferred via RTTI the first time
  /// `.name()` is invoked.
  Module() = default;

  virtual ~Module() = default;

  /// Returns the name of the `Module`.
  const std::string& name() const noexcept;

  /// Performs a recursive deep copy of the module and all its registered
  /// parameters, buffers and submodules.
  virtual std::shared_ptr<Module> clone() const;

  /// Provides a means to traverse the `Module` tree.
  ModuleCursor modules();
  ConstModuleCursor modules() const;

  /// Traverses the (immediate) children of the `Module`.
  ModuleCursor children();
  ConstModuleCursor children() const;

  /// Provides a means to recursively access the parameters of the `Module`
  /// tree.
  ParameterCursor parameters();
  ConstParameterCursor parameters() const;

  /// Provides a means to recursively access the buffers of the `Module` tree.
  BufferCursor buffers();
  ConstBufferCursor buffers() const;

  /// Enables training mode.
  virtual void train();

  /// Disables training mode.
  virtual void eval();

  /// True if the module is in training mode.
  virtual bool is_training() const noexcept;

  /// Recursively moves all parameters to CPU memory (in place).
  virtual void cpu();

  /// Recursively moves all parameters to CUDA memory (in place).
  virtual void cuda();

  /// Recursively casts all parameters to the given type.
  virtual void to(at::Type& type);

  /// Recursively casts all parameters to the given scalar type.
  virtual void to(at::ScalarType scalar_type);

  /// Recursively moves all parameters to the given backend.
  virtual void to(at::Backend backend);

  /// Recursively zeros out the `grad` values of all parameters.
  virtual void zero_grad();

  template <class Archive>
  void save(Archive& ar) const {
    auto params = parameters();
    size_t size = params.size();
    ar(size);
    for (auto& p : params) {
      ar(p.key, p.value);
    }
  }

  template <class Archive>
  void load(Archive& ar) {
    auto params = parameters();
    size_t size;
    ar(size);
    std::string name;
    for (size_t i = 0; i < size; i++) {
      ar(name);
      ar(params[name]);
    }
  }

 protected:
  Variable& register_parameter(
      std::string name,
      Variable tensor,
      bool requires_grad = true);
  Variable& register_buffer(std::string name, Variable tensor);

  template <
      typename ModuleType,
      typename = torch::detail::disable_if_module_holder_t<ModuleType>>
  std::shared_ptr<ModuleType> register_module(
      std::string name,
      std::shared_ptr<ModuleType> module) {
    auto& base_module = children_.insert(std::move(name), std::move(module));
    return std::static_pointer_cast<ModuleType>(base_module);
  }

  template <typename ModuleHolderType>
  ModuleHolderType register_module(
      std::string name,
      ModuleHolderType module_holder) {
    register_module(std::move(name), module_holder.get());
    return module_holder;
  }

 private:
  template <typename T>
  using OrderedDict = torch::detail::OrderedDict<std::string, T>;

  template <typename Derived>
  friend class Cloneable;
  template <typename T>
  friend class detail::CursorBase;

  virtual void clone_(Module& other);

  OrderedDict<autograd::Variable> parameters_;
  OrderedDict<autograd::Variable> buffers_;
  OrderedDict<std::shared_ptr<Module>> children_;

  /// The module's name (e.g. "LSTM").
  mutable at::optional<std::string> name_;

  /// Whether the module is in training mode.
  bool is_training_{true};
};
} // namespace nn
} // namespace torch
