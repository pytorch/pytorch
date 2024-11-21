#include <torch/csrc/jit/mobile/nnc/context.h>

#include <ATen/Functions.h>
#include <ATen/core/functional.h>
#include <c10/core/CPUAllocator.h>
#include <c10/util/irange.h>

#include <torch/csrc/jit/mobile/nnc/registry.h>

namespace torch::jit::mobile::nnc {

constexpr int64_t kProducedNNCFileFormatVersion = 0x1L;

namespace {

c10::IValue Tup(std::initializer_list<c10::IValue> ivalues) {
  return c10::ivalue::Tuple::create(ivalues);
}

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
  if (sizes_.size() != input.sizes().size() || input.scalar_type() != dtype_) {
    return false;
  }
  auto spec_sizes = sizes_;
  for (const auto i : c10::irange(spec_sizes.size())) {
    // InputSpec size 0 means that the dimension is dynamic
    if (spec_sizes[i] != 0 && spec_sizes[i] != input.sizes()[i]) {
      return false;
    }
  }
  return true;
}

OutputSpec::OutputSpec(const c10::IValue& value) {
  auto dict = value.toGenericDict();
  sizes_ = dict.at("sizes").toIntVector();
  dtype_ = dict.at("dtype").toScalarType();
  if (dict.contains("qscale")) {
    qscale_ = dict.at("qscale").toDouble();
  }
  if (dict.contains("qzero")) {
    qzero_ = dict.at("qzero").toInt();
  }
}

c10::IValue OutputSpec::serialize() const {
  c10::Dict<c10::IValue, c10::IValue> dict(
      at::StringType::get(), at::AnyType::get());
  dict.insert("sizes", sizes_);
  dict.insert("dtype", dtype_);
  if (qscale_) {
    dict.insert("qscale", *qscale_);
  }
  if (qzero_) {
    dict.insert("qzero", *qzero_);
  }
  return dict;
}

at::Tensor OutputSpec::allocate() const {
  if (isQIntType(dtype_)) {
    TORCH_CHECK(
        qscale_ && qzero_,
        "Quantized output tensor must have qscale_ and qzero_");
    return at::_empty_affine_quantized(
        sizes_,
        at::TensorOptions()
            .dtype(dtype_)
            .layout(at::kStrided)
            .device(at::kCPU)
            .requires_grad(false),
        *qscale_,
        *qzero_);
  }
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
  parameters_ = dict.at("parameters").toList();

  // input_specs_
  for (const auto& input_value :
       dict.at("input_specs").toTupleRef().elements()) {
    input_specs_.emplace_back(input_value);
  }

  // output_specs_
  for (const auto& output_value :
       dict.at("output_specs").toTupleRef().elements()) {
    output_specs_.emplace_back(output_value);
  }

  // memory_plan_
  memory_plan_ = MemoryPlan(dict.at("memory_plan"));

  // symbolic shape positions
  for (const auto& sym_shape_pos :
       dict.at("sym_shape_pos").toTupleRef().elements()) {
    auto sym_shape_elements = sym_shape_pos.toTupleRef().elements();
    sym_shape_positions_.emplace_back(
        sym_shape_elements[0].toInt(), sym_shape_elements[1].toInt());
  }
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
  input_specs.reserve(input_specs_.size());
  for (const auto& input_spec : input_specs_) {
    input_specs.emplace_back(input_spec.serialize());
  }
  dict.insert("input_specs", Tup(std::move(input_specs)));

  // output_specs_
  std::vector<c10::IValue> output_specs;
  output_specs.reserve(output_specs_.size());
  for (const auto& output_spec : output_specs_) {
    output_specs.emplace_back(output_spec.serialize());
  }
  dict.insert("output_specs", Tup(std::move(output_specs)));

  // memory_plan_
  dict.insert("memory_plan", memory_plan_.serialize());

  // sym_shape_positions_
  std::vector<c10::IValue> sym_shape_pos_vec;
  sym_shape_pos_vec.reserve(sym_shape_positions_.size());
  for (const auto& sym_shape_pos : sym_shape_positions_) {
    sym_shape_pos_vec.emplace_back(
        Tup({sym_shape_pos.input_idx_, sym_shape_pos.dim_idx_}));
  }
  dict.insert("sym_shape_pos", Tup(std::move(sym_shape_pos_vec)));

  return dict;
}

void Function::init_execution_state() const {
  if (execution_state_ != nullptr) {
    return;
  }

  ExecutionState state;
  memory_plan_.allocate(&state);

  // The arguments vector consists of 5 sections: inputs, symbolic shapes,
  // outputs, parameters and buffers.
  auto input_args = input_specs_.size();
  auto sym_shape_args = sym_shape_positions_.size();
  auto output_args = output_specs_.size();
  auto param_args = parameters_.size();
  auto buffer_args = state.preallocations_.size();

  auto& arguments = state.arguments_;
  arguments.reserve(
      input_args + sym_shape_args + output_args + param_args + buffer_args);

  // Keep empty slots to fill in inputs/outputs pointers at execution time.
  arguments.resize(input_args + sym_shape_args + output_args);

  // Fill in parameters as untyped raw pointers.
  // The underlying storage of the parameters should be owned by `parameters_`,
  // which should be alive when `execution_state_` is being used.
  for (const auto& param : parameters_) {
    const c10::IValue& ivalue = (c10::IValue)param;
    if (ivalue.isTensor()) {
      arguments.emplace_back(ivalue.toTensor().data_ptr());
    } else if (torch::isCustomClass(ivalue)) {
      arguments.emplace_back(ivalue.toObjectRef().getSlot(0).toCapsule().get());
    } else {
      TORCH_CHECK(false, "Invalid parameter: ", ivalue);
    }
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
  std::vector<int64_t> scalar_values;
  int offset = 0;
  for (const auto i : c10::irange(inputs.size())) {
    const c10::IValue& input = inputs[i];
    const auto& spec = input_specs_[i];
    const auto& input_tensor = input.toTensor();
    TORCH_CHECK(spec.validate(input_tensor), "Invalid input at pos: ", i);
    args[i] = input_tensor.data_ptr();
  }
  offset += inputs.size();

  scalar_values.reserve(sym_shape_positions_.size());
  for (const auto i : c10::irange(sym_shape_positions_.size())) {
    const auto& sym_shape_pos = sym_shape_positions_[i];
    const c10::IValue& input = inputs[sym_shape_pos.input_idx_];
    auto dim = input.toTensor().size(sym_shape_pos.dim_idx_);
    scalar_values.push_back(dim);
    args[i + offset] = &scalar_values[scalar_values.size() - 1];
  }
  offset += sym_shape_positions_.size();

  // Preallocate and fill in output tensors.
  c10::List<at::Tensor> outputs;
  outputs.reserve(output_specs_.size());
  for (const auto i : c10::irange(output_specs_.size())) {
    at::Tensor output = output_specs_[i].allocate();
    outputs.emplace_back(output);
    args[i + offset] = output.data_ptr();
  }

  // TODO: check consistency, e.g.: code version, input shape and compiled
  // shape, etc.
  auto kernel = registry::get_nnc_kernel(nnc_kernel_id_);
  kernel->execute(args.data());

  return c10::impl::toList(outputs);
}

CompilationUnit::CompilationUnit(const c10::IValue& value) {
  const auto& root = value.toTupleRef().elements();
  const auto& functions = root[1].toTupleRef().elements();
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

} // namespace torch::jit::mobile::nnc
