#pragma once

#include <torch/tensor.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace at {
enum class ScalarType;
} // namespace at

namespace torch { namespace nn {

struct Archive;
struct BufferCursor;
struct ModuleCursor;
struct ParameterCursor;

/// The base class for all torch modules.
class Module {
 public:
  /// Tells the base `Module` about the name of the submodule.
  explicit Module(std::string name);
  virtual ~Module();

  /// Returns the name of the `Module`.
  const std::string& name() const noexcept;

  /// Performs a recursive clone of the entire module hierarchy, including
  /// parameters and buffers. This is to enable deep copying in a polymorphic
  /// setting, i.e. when all you have is a `Module*` but want to clone the whole
  /// module hierarchy.
  virtual std::unique_ptr<Module> clone();

  /// The call operator delegates to `forward()` (for convenience).
  std::vector<Tensor> operator()(const std::vector<Tensor>& inputs);

  /// Takes a list of input variables and computes a list of output variables.
  virtual std::vector<Tensor> forward(const std::vector<Tensor>& inputs) = 0;

  /// Enables training mode.
  void train();

  /// Disables training mode.
  void eval();

  /// True if the `Module` is in training mode.
  bool is_training() const noexcept;

  /// Recursively moves all parameters and buffers to CPU memory (in-place).
  void cpu();

  /// Recursively moves all parameters and buffers to CUDA memory (in-place).
  void cuda();

  /// Recursively casts all parameters and buffers to the given type (in-place).
  void type(at::ScalarType new_type);

  /// Recursively zeros out the `grad` values of all parameters.
  void zero_grad();

  /// Provides a means to traverse the `Module` tree.
  ModuleCursor modules();

  /// Traverses the (immediate) children of the `Module`.
  ModuleCursor children();

  /// Provides a means to recursively access the parameters of the `Module`
  /// tree.
  ParameterCursor parameters();

  /// Provides a means to recursively access the buffers of the `Module` tree.
  BufferCursor buffers();

  /// Serializes this `Module`. The default implementation serializes the
  /// submodules, parameters and buffers registered with the base class. The
  /// method can be overriden to modify this behavior.
  virtual void serialize(Archive& archive);

  /// Deserializes the `Module` from the archive. It will restore the registered
  /// submodules, parameters and buffers, but can also be customized. This could
  /// be called from the constructor of the submodule. Maybe this method will
  /// also be protected only. Maybe submodules will have to create constructors
  /// from Archives. The specification is explicitly vague at the moment as the
  /// serialization protocol will require much more thought.
  virtual void deserialize(Archive&& archive);

 protected:
  /// Inserts the parameters into the parameters_ map.
  void register_parameters(
      const std::unordered_map<std::string, Tensor>& parameters);

  /// Inserts the buffers into the buffers_ map.
  void register_buffers(const std::unordered_map<std::string, Tensor>& buffers);

  /// Inserts the modules into the modules_ map.
  void register_modules(
      const std::unordered_map<std::string, Module*>& modules);

 private:
  /// The module's name (e.g. "LSTM").
  std::string name_;

  /// Whether the module is in training mode.
  bool is_training_;

  std::unordered_map<std::string, Module*> children_;
  std::unordered_map<std::string, Tensor> parameters_;
  std::unordered_map<std::string, Tensor> buffers_;
};

/// The `clone()` method in the base `Module` class does not have knowledge of
/// the concrete runtime type of its subclasses. Therefore, `clone()` must
/// either be called from within the subclass, or from a base class that has
/// knowledge of the concrete type. `CloneableModule` uses the CRTP to gain
/// knowledge of the subclass' static type and provide an implementation of the
/// `clone()` method. We do not want to use this pattern in the base class,
/// because then storing a module would always require templatizing it.
template <typename Derived>
class CloneableModule : public Module {
 public:
  explicit CloneableModule(std::string name) : Module(std::move(name)) {}

  std::unique_ptr<Module> clone() override {
    return std::unique_ptr<Module>(new Derived(static_cast<Derived&>(*this)));
  }
};
}} // namespace torch::nn
