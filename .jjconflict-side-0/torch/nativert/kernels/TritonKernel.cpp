#include <torch/nativert/kernels/TritonKernel.h>

#include <c10/util/Exception.h>

#include <algorithm>
#include <limits>
#include <optional>

#include <torch/nativert/executor/DelegateExecutor.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#endif

namespace torch::nativert {
namespace {

int checkedGridDim(const c10::IValue& value, size_t idx) {
  TORCH_CHECK(
      value.isInt(),
      "Expected integer Triton grid dimension at index ",
      idx,
      " but got ",
      value.tagKind());
  const auto dim = value.toInt();
  TORCH_CHECK(
      dim >= std::numeric_limits<int>::min() &&
          dim <= std::numeric_limits<int>::max(),
      "Triton grid dimension at index ",
      idx,
      " is out of range: ",
      dim);
  return static_cast<int>(dim);
}

GridDims gridDimsFromIValue(const c10::IValue& value) {
  TORCH_CHECK(
      value.isList(),
      "Expected Triton grid input to be List[int] but got ",
      value.tagKind());

  const auto list = value.toListRef();
  TORCH_CHECK(
      list.size() == 3,
      "Expected Triton grid input to have 3 dimensions but got ",
      list.size());

  GridDims grid_dims;
  size_t idx = 0;
  for (const auto& elem : list) {
    const auto dim = checkedGridDim(elem, idx);
    if (idx == 0) {
      grid_dims.x = dim;
    } else if (idx == 1) {
      grid_dims.y = dim;
    } else {
      grid_dims.z = dim;
    }
    ++idx;
  }
  return grid_dims;
}

bool isLaunchMetadataInput(std::string_view name) {
  return name == "grid";
}

bool isKernelParamName(const KernelInputParams& params, std::string_view name) {
  return std::find(
             params.kernel_param_names.begin(),
             params.kernel_param_names.end(),
             name) != params.kernel_param_names.end();
}

std::optional<size_t> findInputIndex(
    const std::vector<NamedArgument>& inputs,
    std::string_view name) {
  for (const auto i : c10::irange(inputs.size())) {
    if (inputs[i].name == name) {
      return i;
    }
  }
  return std::nullopt;
}

const Attribute* findAttribute(const Node* node, std::string_view name) {
  for (const auto& attr : node->attributes()) {
    if (attr.name == name) {
      return &attr;
    }
  }
  return nullptr;
}

void addKernelArg(
    KernelInputs& inputs,
    const c10::IValue& value,
    std::string_view param_name,
    std::string_view param_type,
    size_t source_idx) {
  if (value.isTensor()) {
    inputs.add_tensor_arg(value.toTensor());
  } else if (value.isInt() || value.isDouble() || value.isBool()) {
    inputs.add_scalar_arg(value, param_type);
  } else if (value.isNone()) {
    inputs.add_arg(nullptr);
  } else if (value.isList()) {
    TORCH_CHECK(
        false,
        "Unsupported Triton kernel input at index ",
        source_idx,
        " (",
        param_name,
        ": ",
        param_type,
        "): ",
        value.tagKind(),
        "(size=",
        value.toListRef().size(),
        ")");
  } else {
    TORCH_CHECK(
        false,
        "Unsupported Triton kernel input at index ",
        source_idx,
        " (",
        param_name,
        ": ",
        param_type,
        ")",
        ": ",
        value.tagKind());
  }
}

} // namespace

// in this case, we want to use the symbol from torch_cpu.dll
#ifndef NATIVERT_MSVC_TEST
C10_DEFINE_TYPED_REGISTRY(
    TritonKernelManagerRegistry,
    c10::DeviceType,
    TritonKernelManager,
    std::unique_ptr,
    std::string /* kernel_name */,
    std::string /* kernel_bin_path */,
    std::string /* kernel_launcher_bin_path */)
#endif

TritonKernel::TritonKernel(
    const Node* node,
    caffe2::serialize::PyTorchStreamReader* reader)
    : OpKernel(node, OpKernelKind::kTritonKernel) {
  TORCH_CHECK(reader != nullptr, "reader is null");

  std::string kernel_name{};
  std::string symbol_name{};

  // To prevent vector reallocation and dangling pointers
  size_t num_double_attrs = 0;
  for (const auto& attr : node_->attributes()) {
    if (attr.name.empty() && std::holds_alternative<double>(attr.value)) {
      ++num_double_attrs;
    }
  }
  float_attrs_.reserve(num_double_attrs);

  // Parse only TritonKernel-specific attributes here.
  // Launch parameters (grid, num_warps, etc.) are parsed by the target-specific
  // TritonKernelManager via createLaunchParams().
  for (const auto& attr : node_->attributes()) {
    if (attr.name.empty()) {
      attr_ptrs_.emplace_back(std::visit(
          [this](auto&& arg) -> void* {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, None>) {
              return nullptr;
            } else if constexpr (std::is_same_v<T, double>) {
              // Triton always uses fp32 for floats. See
              // create_specialize_impl in jit.py. However, due to the
              // Thrift schema, floats are serialized as doubles here. But,
              // Triton kernels read them as floats. So, we need to downcast
              // double to float here.
              float_attrs_.push_back(static_cast<float>(arg));
              return static_cast<void*>(&float_attrs_.back());
            }
            return static_cast<void*>(const_cast<T*>(&arg));
          },
          attr.value));
    } else if (attr.name == "name") {
      kernel_name = std::get<std::string>(attr.value);
      size_t last_underscore = kernel_name.find_last_of('_');
      symbol_name = kernel_name.substr(0, last_underscore);
    } else if (attr.name == "output_indices") {
      output_indices_ = std::get<std::vector<int64_t>>(attr.value);
      kernel_input_params_.output_indices = output_indices_;
    } else if (attr.name == "kernel_param_names") {
      kernel_input_params_.kernel_param_names =
          std::get<std::vector<std::string>>(attr.value);
    } else if (attr.name == "kernel_param_types") {
      kernel_input_params_.kernel_param_types =
          std::get<std::vector<std::string>>(attr.value);
    }
  }

  TORCH_CHECK(!kernel_name.empty(), "kernel name not found");
  TORCH_CHECK(!symbol_name.empty(), "symbol_name not found");
  TORCH_CHECK(!output_indices_.empty(), "output_indices attribute not found");

  auto kernel_prefix = std::string("data/triton") + "/" + kernel_name;

  auto tmp_dir = extractToTemporaryFolder(*reader, kernel_prefix) + "/";

  if (reader->hasRecord(kernel_prefix + "/" + kernel_name + ".cubin")) {
    loader_ = TritonKernelManagerRegistry()->Create(
        at::kCUDA, symbol_name, tmp_dir + kernel_name + ".cubin", "");
    TORCH_CHECK(
        loader_ != nullptr,
        "couldn't find cuda loader -- is this a gpu build?");
  } else if (reader->hasRecord(kernel_prefix + "/" + kernel_name + ".hsaco")) {
    loader_ = TritonKernelManagerRegistry()->Create(
        at::kHIP, symbol_name, tmp_dir + kernel_name + ".hsaco", "");
    TORCH_CHECK(
        loader_ != nullptr,
        "couldn't find cuda loader -- is this a gpu build?");
  } else if (reader->hasRecord(kernel_prefix + "/" + kernel_name + ".so")) {
    loader_ = TritonKernelManagerRegistry()->Create(
        at::kCPU,
        symbol_name,
        tmp_dir + kernel_name + ".so",
        tmp_dir + kernel_name + ".launcher.so");
    TORCH_CHECK(
        loader_ != nullptr, "couldn't find CPU loader -- is this a cpu build?");
  } else if (reader->hasRecord(kernel_prefix + "/" + kernel_name + ".bin")) {
    loader_ = TritonKernelManagerRegistry()->Create(
        at::kMTIA, symbol_name, tmp_dir + kernel_name + ".bin", "");
    TORCH_CHECK(
        loader_ != nullptr,
        "couldn't find MTIA loader -- is this a mtia build?");
  }

  TORCH_CHECK(
      loader_ != nullptr,
      "couldn't find triton kernel loader -- are you trying to run gpu kernels on a cpu build?");
}

TritonKernel::~TritonKernel() = default;

void TritonKernel::computeInternal(ExecutionFrame& executionFrame) const {
  const auto& node_inputs = node_->inputs();
  const auto num_attrs = attr_ptrs_.size();

  auto* loader = const_cast<TritonKernelManager*>(loader_.get());

  auto launch_params = loader->createLaunchParams(node_);

  const auto param_type = [this](size_t i) -> std::string_view {
    return i < kernel_input_params_.kernel_param_types.size()
        ? std::string_view(kernel_input_params_.kernel_param_types[i])
        : std::string_view();
  };
  const auto param_name = [this](size_t i) -> std::string_view {
    return i < kernel_input_params_.kernel_param_names.size()
        ? std::string_view(kernel_input_params_.kernel_param_names[i])
        : std::string_view();
  };

  const bool use_named_kernel_arg_packing =
      !kernel_input_params_.kernel_param_names.empty() &&
      std::any_of(
          node_inputs.begin(), node_inputs.end(), [this](const auto& input) {
            return !input.name.empty() &&
                isKernelParamName(kernel_input_params_, input.name);
          });

  if (use_named_kernel_arg_packing) {
    const auto num_kernel_args = kernel_input_params_.kernel_param_names.size();
    auto inputs =
        loader->create_inputs(num_kernel_args, 0, kernel_input_params_);
    std::vector<std::optional<size_t>> kernel_arg_input_indices(
        num_kernel_args);

    for (const auto i : c10::irange(node_inputs.size())) {
      const auto input_name = std::string_view(node_inputs[i].name);
      if (isLaunchMetadataInput(input_name)) {
        launch_params->grid_dims = gridDimsFromIValue(input(i, executionFrame));
      } else {
        TORCH_CHECK(
            isKernelParamName(kernel_input_params_, input_name),
            "Unsupported symbolic Triton launch input '",
            input_name,
            "' at index ",
            i,
            ": ",
            input(i, executionFrame).tagKind());
      }
    }

    for (const auto kernel_arg_idx : c10::irange(num_kernel_args)) {
      const auto name = param_name(kernel_arg_idx);
      if (const auto input_idx = findInputIndex(node_inputs, name)) {
        addKernelArg(
            *inputs,
            input(*input_idx, executionFrame),
            name,
            param_type(kernel_arg_idx),
            *input_idx);
        kernel_arg_input_indices[kernel_arg_idx] = *input_idx;
      } else if (const auto* attr = findAttribute(node_, name)) {
        addKernelArg(
            *inputs,
            constantToIValue(attr->value),
            name,
            param_type(kernel_arg_idx),
            kernel_arg_idx);
      } else {
        TORCH_CHECK(
            false,
            "Missing serialized Triton kernel parameter '",
            name,
            "' at index ",
            kernel_arg_idx);
      }
    }

    loader->launch(*launch_params, inputs->as_void());

    const auto outputTensor = [&](int64_t output_idx) -> at::Tensor {
      TORCH_CHECK(
          output_idx >= 0 && static_cast<size_t>(output_idx) < num_kernel_args,
          "Triton output index out of range: ",
          output_idx);
      const auto output_pos = static_cast<size_t>(output_idx);
      const auto input_idx = kernel_arg_input_indices[output_pos];
      TORCH_CHECK(
          input_idx.has_value(),
          "Triton output parameter '",
          param_name(output_pos),
          "' is not a runtime tensor input");
      return input(*input_idx, executionFrame).toTensor();
    };

    auto& out = output(0, executionFrame);
    if (out.isNone()) {
      auto list = c10::List<at::Tensor>();
      for (const auto& i : output_indices_) {
        list.emplace_back(outputTensor(i));
      }
      out = c10::IValue(std::move(list));
      return;
    }

    auto out_t = out.toTensorList();
    for (const auto i : c10::irange(output_indices_.size())) {
      out_t[i] = outputTensor(output_indices_[i]);
    }
    return;
  }

  size_t num_runtime_kernel_args = 0;
  for (const auto& node_input : node_inputs) {
    if (node_input.name.empty()) {
      ++num_runtime_kernel_args;
    }
  }

  if (!kernel_input_params_.kernel_param_names.empty()) {
    TORCH_CHECK(
        num_runtime_kernel_args + num_attrs ==
            kernel_input_params_.kernel_param_names.size(),
        "Triton kernel parameter count mismatch: got ",
        num_runtime_kernel_args,
        " runtime args and ",
        num_attrs,
        " constant attrs, but kernel metadata has ",
        kernel_input_params_.kernel_param_names.size(),
        " parameters");
  }

  auto inputs = loader->create_inputs(
      num_runtime_kernel_args, num_attrs, kernel_input_params_);

  size_t kernel_arg_idx = 0;
  for (const auto i : c10::irange(node_inputs.size())) {
    const auto input_name = std::string_view(node_inputs[i].name);
    const auto& value = input(i, executionFrame);
    if (isLaunchMetadataInput(input_name)) {
      launch_params->grid_dims = gridDimsFromIValue(value);
      continue;
    }

    TORCH_CHECK(
        input_name.empty(),
        "Unsupported symbolic Triton launch input '",
        input_name,
        "' at index ",
        i,
        ": ",
        value.tagKind());

    addKernelArg(
        *inputs,
        value,
        param_name(kernel_arg_idx),
        param_type(kernel_arg_idx),
        i);
    ++kernel_arg_idx;
  }

  for (const auto i : c10::irange(num_attrs)) {
    inputs->add_attribute(attr_ptrs_[i]);
  }

  loader->launch(*launch_params, inputs->as_void());

  auto& out = output(0, executionFrame);
  if (out.isNone()) {
    auto list = c10::List<at::Tensor>();
    for (const auto& i : output_indices_) {
      list.emplace_back(input(i, executionFrame).toTensor());
    }
    out = c10::IValue(std::move(list));
    return;
  }

  // todo: check if this is redundant
  auto out_t = out.toTensorList();
  for (const auto i : c10::irange(output_indices_.size())) {
    out_t[i] = input(output_indices_[i], executionFrame).toTensor();
  }
}

} // namespace torch::nativert
