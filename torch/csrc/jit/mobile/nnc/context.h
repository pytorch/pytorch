#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <ATen/core/ivalue.h>
#include <c10/core/ScalarType.h>

namespace torch {
namespace jit {
namespace mobile {
namespace nnc {

// Specify the requirements on an input tensor.
// TODO: support input tensor with dynamic shape (PR #54982)
struct TORCH_API InputSpec {
  InputSpec() = default;

  // Deserialize the spec from an IValue.
  explicit InputSpec(const c10::IValue& value);

  // Serialize the spec into an IValue.
  c10::IValue serialize() const;

  // Check whether the input tensor adheres to the spec.
  bool validate(const at::Tensor& input) const;

  std::vector<int64_t> sizes_;
  c10::ScalarType dtype_{c10::ScalarType::Undefined};
};

// Specify the sizes/dtype/... of output tensor to preallocate the output.
// TODO: support the case where kernel allocates output tensors dynamically.
struct TORCH_API OutputSpec {
  OutputSpec() = default;

  // Deserialize the spec from an IValue.
  explicit OutputSpec(const c10::IValue& value);

  // Serialize the spec into an IValue.
  c10::IValue serialize() const;

  // Allocate an output tensor in accordance with the spec.
  at::Tensor allocate() const;

  std::vector<int64_t> sizes_;
  c10::ScalarType dtype_{c10::ScalarType::Undefined};
};

// Hold the temporary buffers / states needed during the execution.
struct TORCH_API ExecutionState {
  // Preallocated buffers needed by the NNC kernel.
  std::vector<c10::DataPtr> preallocations;

  // The NNC kernel expects the following arguments layout:
  //   input tensor 1
  //   ...
  //   input tensor INPUT_NUM
  //   output tensor 1
  //   ...
  //   output tensor OUTPUT_NUM
  //   parameter tensor 1
  //   ...
  //   parameter tensor PARAM_NUM
  //   temporary buffer 1
  //   ...
  //   temporary buffer BUFFER_NUM
  std::vector<void*> arguments;
};

// Specify how to allocate temporary buffers at initialization.
struct TORCH_API MemoryPlan {
  MemoryPlan() = default;
  explicit MemoryPlan(const c10::IValue& value);

  c10::IValue serialize() const;

  void allocate(ExecutionState* state) const;

  std::vector<int64_t> buffer_sizes_;
};

// Represents a compiled NNC function which has a 1-1 correspondence with a
// `Method` (e.g. `forward`). It's similar as torch::jit::mobile::Function.
class TORCH_API Function {
 public:
  explicit Function() = default;

  // Deserialize from an IValue.
  explicit Function(const c10::IValue& value);

  // Serialize into an IValue.
  c10::IValue serialize() const;

  // Execute the compiled NNC function.
  c10::impl::GenericList run(const c10::impl::GenericList& inputs) const;

  // The name of the function as specified in the model code.
  c10::QualifiedName name() const {
    return name_;
  }

  void set_name(const c10::QualifiedName& name) {
    name_ = name;
  }

  // The unique id of the generated NNC kernel corresponding to the function.
  const std::string& nnc_kernel_id() const {
    return nnc_kernel_id_;
  }

  void set_nnc_kernel_id(const std::string& name) {
    nnc_kernel_id_ = name;
  }

  // The parameters (e.g. weights / bias tensors) to be passed to the generated
  // NNC kernel.
  const std::vector<at::Tensor>& parameters() const {
    return parameters_;
  }

  void set_parameters(const std::vector<at::Tensor>& parameters) {
    parameters_ = parameters;
  }

  const std::vector<InputSpec>& input_specs() const {
    return input_specs_;
  }

  void set_input_specs(const std::vector<InputSpec>& input_specs) {
    input_specs_ = input_specs;
  }

  const std::vector<OutputSpec>& output_specs() const {
    return output_specs_;
  }

  void set_output_spec(const std::vector<OutputSpec>& output_specs) {
    output_specs_ = output_specs;
  }

  const MemoryPlan& memory_plan() const {
    return memory_plan_;
  }

  void set_memory_plan(const MemoryPlan& memory_plan) {
    memory_plan_ = memory_plan;
  }

 private:
  void init_execution_state() const;

  c10::QualifiedName name_;
  std::string nnc_kernel_id_;
  std::vector<at::Tensor> parameters_;
  std::vector<InputSpec> input_specs_;
  std::vector<OutputSpec> output_specs_;
  MemoryPlan memory_plan_;
  mutable std::unique_ptr<ExecutionState> execution_state_;
};

// Represents a set of compiled NNC functions which has a 1-1 correspondence
// with a `Module`. It's similar as torch::jit::mobile::CompilationUnit.
class TORCH_API CompilationUnit {
 public:
  CompilationUnit() = default;

  // Deserialize from an IValue.
  explicit CompilationUnit(const c10::IValue& value);

  // Serialize all registered functions into an IValue. The IValue will be save
  // into the compiled TorchScript model file ahead-of-time on the host, and
  // will be deserialized at runtime on the target device.
  c10::IValue serialize() const;

  // Execute a registered function.
  c10::impl::GenericList run(
      const c10::QualifiedName& function_name,
      const c10::impl::GenericList& inputs) const;

  // Register a function to the compilation unit.
  void register_function(std::unique_ptr<Function> fn);

 private:
  Function* find_function(const c10::QualifiedName& qn) const;

  std::vector<std::unique_ptr<Function>> functions_;
};

} // namespace nnc
} // namespace mobile
} // namespace jit
} // namespace torch
