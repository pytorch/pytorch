#include <torch/csrc/jit/mobile/nnc/context.h>

#include <ATen/Functions.h>
#include <ATen/core/functional.h>
#include <c10/core/CPUAllocator.h>

#include <torch/csrc/jit/mobile/nnc/registry.h>

namespace torch {
namespace jit {
namespace mobile {
namespace nnc {

constexpr int64_t kProducedNNCFileFormatVersion = 0x1L;

namespace {

c10::IValue Tup(std::vector<c10::IValue>&& ivalues) {
  return c10::ivalue::Tuple::create(ivalues);
}

} // namespace

InputSpec::InputSpec(const c10::IValue& value) {
  auto dict = value.toGenericDict();
  sizes_ = dict.at("sizes").toIntVector();
  dtype_ = dict.at("dtype").toScalarType();
}

c10::IValue InputSpec::serialize() const {
  c10::Dict<c10::IValue, c10::IValue> dict(
      at::StringType::get(), at::AnyType::get());
  dict.insert("sizes", sizes_);
  dict.insert("dtype", dtype_);
  return dict;
}

bool InputSpec::validate(const at::Tensor& input) const {
  return input.sizes() == sizes_ && input.scalar_type() == dtype_;
}

OutputSpec::OutputSpec(const c10::IValue& value) {
  auto dict = value.toGenericDict();
  sizes_ = dict.at("sizes").toIntVector();
  dtype_ = dict.at("dtype").toScalarType();
}

c10::IValue OutputSpec::serialize() const {
  c10::Dict<c10::IValue, c10::IValue> dict(
      at::StringType::get(), at::AnyType::get());
  dict.insert("sizes", sizes_);
  dict.insert("dtype", dtype_);
  return dict;
}

at::Tensor OutputSpec::allocate() const {
  return at::empty(
      sizes_,
      at::TensorOptions()
          .dtype(dtype_)
          .layout(at::kStrided)
          .device(at::kCPU)
          .requires_grad(false));
}

MemoryPlan::MemoryPlan(const c10::IValue& value) {
  auto dict = value.toGenericDict();
  buffer_sizes_ = dict.at("buffer_sizes").toIntVector();
}

c10::IValue MemoryPlan::serialize() const {
  c10::Dict<c10::IValue, c10::IValue> dict(
      at::StringType::get(), at::AnyType::get());
  dict.insert("buffer_sizes", buffer_sizes_);
  return dict;
}

void MemoryPlan::allocate(ExecutionState* state) const {
  auto& allocations = state->preallocations_;
  allocations.clear();
  allocations.reserve(buffer_sizes_.size());
  for (int64_t buffer_size : buffer_sizes_) {
    at::DataPtr buffer = c10::GetCPUAllocator()->allocate(buffer_size);
    allocations.emplace_back(std::move(buffer));
  }
}

Function::Function(const c10::IValue& value) {
  auto dict = value.toGenericDict();
  name_ = c10::QualifiedName(dict.at("name").toStringRef());
  nnc_kernel_id_ = dict.at("nnc_kernel_id").toStringRef();
  parameters_ = dict.at("parameters").toTensorVector();

  // input_specs_
  for (const auto& input_value : dict.at("input_specs").toTuple()->elements()) {
    input_specs_.emplace_back(input_value);
  }

  // output_specs_
  for (const auto& output_value :
       dict.at("output_specs").toTuple()->elements()) {
    output_specs_.emplace_back(output_value);
  }

  // memory_plan_
  memory_plan_ = MemoryPlan(dict.at("memory_plan"));
}

c10::IValue Function::serialize() const {
  c10::Dict<c10::IValue, c10::IValue> dict(
      at::StringType::get(), at::AnyType::get());

  dict.insert("name", name_.qualifiedName());
  dict.insert("nnc_kernel_id", nnc_kernel_id_);
  // TODO: should serialize parameters with Module instead of with each Method.
  // And ideally the parameters should be shared between the compiled model
  // and the original model if we can serialize both in the same model file.
  dict.insert("parameters", parameters_);

  // input_specs_
  std::vector<c10::IValue> input_specs;
  for (const auto& input_spec : input_specs_) {
    input_specs.emplace_back(input_spec.serialize());
  }
  dict.insert("input_specs", Tup(std::move(input_specs)));

  // output_specs_
  std::vector<c10::IValue> output_specs;
  for (const auto& output_spec : output_specs_) {
    output_specs.emplace_back(output_spec.serialize());
  }
  dict.insert("output_specs", Tup(std::move(output_specs)));

  // memory_plan_
  dict.insert("memory_plan", memory_plan_.serialize());
  return dict;
}

void Function::init_execution_state() const {
  if (execution_state_.get() != nullptr) {
    return;
  }

  ExecutionState state;
  memory_plan_.allocate(&state);

  // The arguments vector consists of 4 sections: inputs, outputs, parameters
  // and buffers.
  auto input_args = input_specs_.size();
  auto output_args = output_specs_.size();
  auto param_args = parameters_.size();
  auto buffer_args = state.preallocations_.size();

  auto& arguments = state.arguments_;
  arguments.reserve(input_args + output_args + param_args + buffer_args);

  // Keep empty slots to fill in inputs/outputs pointers at execution time.
  arguments.resize(input_args + output_args);

  // Fill in parameter pointers.
  for (const auto& param : parameters_) {
    arguments.emplace_back(param.data_ptr());
  }

  // Fill in preallocated buffer pointers.
  for (const auto& preallocation : state.preallocations_) {
    arguments.emplace_back(preallocation.get());
  }

  execution_state_ = std::make_unique<ExecutionState>(std::move(state));
}

c10::impl::GenericList Function::run(
    const c10::impl::GenericList& inputs) const {
  TORCH_CHECK(
      registry::has_nnc_kernel(nnc_kernel_id_),
      "Cannot find NNC kernel: ",
      nnc_kernel_id_);

  init_execution_state();

  std::vector<void*>& args = execution_state_->arguments_;

  // Fill in input tensors.
  TORCH_CHECK(
      input_specs_.size() == inputs.size(),
      "Input size doesn't match the spec, expect: ",
      input_specs_.size(),
      " actual: ",
      inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    const c10::IValue& input = inputs[i];
    const auto& input_tensor = input.toTensor();
    TORCH_CHECK(
        input_specs_[i].validate(input_tensor), "Invalid input at pos: ", i);
    args[i] = input_tensor.data_ptr();
  }

  // Preallocate and fill in output tensors.
  c10::List<at::Tensor> outputs;
  outputs.reserve(output_specs_.size());
  for (size_t i = 0; i < output_specs_.size(); ++i) {
    at::Tensor output = output_specs_[i].allocate();
    outputs.emplace_back(output);
    args[inputs.size() + i] = output.data_ptr();
  }

  // TODO: check consistency, e.g.: code version, input shape and compiled
  // shape, etc.
  auto kernel = registry::get_nnc_kernel(nnc_kernel_id_);
  kernel->execute(args.data());

  return c10::impl::toList(outputs);
}

CompilationUnit::CompilationUnit(const c10::IValue& value) {
  const auto& root = value.toTuple()->elements();
  const auto& functions = root[1].toTuple()->elements();
  for (const auto& function : functions) {
    register_function(std::make_unique<Function>(function));
  }
}

c10::IValue CompilationUnit::serialize() const {
  auto functions =
      c10::fmap(functions_, [](decltype(functions_)::const_reference func) {
        return func.second->serialize();
      });
  return Tup({kProducedNNCFileFormatVersion, Tup(std::move(functions))});
}

c10::impl::GenericList CompilationUnit::run(
    const c10::QualifiedName& name,
    const c10::impl::GenericList& inputs) const {
  Function* func = find_function(name);
  TORCH_CHECK(
      func != nullptr, "Function '", name.qualifiedName(), "' is not defined.");
  return func->run(inputs);
}

void CompilationUnit::register_function(std::unique_ptr<Function> fn) {
  TORCH_CHECK(
      0 == functions_.count(fn->name()),
      "method '",
      fn->name().qualifiedName(),
      "' already defined.");
  const auto& name = fn->name();
  functions_.emplace(name, std::move(fn));
}

Function* CompilationUnit::find_function(const c10::QualifiedName& name) const {
  auto it = functions_.find(name);
  if (it == functions_.end()) {
    return nullptr;
  }
  return it->second.get();
}

} // namespace nnc
} // namespace mobile
} // namespace jit
} // namespace torch
