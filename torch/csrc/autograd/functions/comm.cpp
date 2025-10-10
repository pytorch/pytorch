#include <torch/csrc/autograd/functions/comm.h>

#include <ATen/core/functional.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/cuda/comm.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <memory>
#include <vector>

namespace torch::autograd {
Scatter::Scatter(
    std::vector<at::Device> devices,
    std::optional<std::vector<int64_t>> chunk_sizes,
    int64_t dim,
    std::optional<std::vector<std::optional<at::cuda::CUDAStream>>> streams,
    bool unsqueeze_scalars)
    : devices_(std::move(devices)),
      chunk_sizes_(std::move(chunk_sizes)),
      dim_(dim),
      streams_(std::move(streams)),
      unsqueeze_scalars_(unsqueeze_scalars) {}

Scatter::~Scatter() = default;

variable_list Scatter::apply(variable_list&& inputs) {
  AT_ASSERT(inputs.size() == 1);
  auto& input = inputs.front();

  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad(input)) {
    grad_fn =
        std::make_shared<Gather>(/*destination_device=*/input.device(), dim_);
    grad_fn->set_next_edges(collect_next_edges(input));
  }

  auto device_indices = fmap(devices_, [](const at::Device& device) -> int64_t {
    return device.index();
  });
  auto tensors =
      torch::cuda::scatter(input, device_indices, chunk_sizes_, dim_, streams_);

  std::vector<Variable> variables;
  variables.reserve(tensors.size());
  for (auto& tensor : tensors) {
    AT_ASSERT(tensor.defined());
    if (unsqueeze_scalars_) {
      AT_ASSERT(tensor.dim() == 1 && tensor.numel() == 1);
      variables.push_back(tensor[0]);
    } else {
      variables.push_back(std::move(tensor));
    }
  }

  if (grad_fn) {
    set_history(variables, grad_fn);
  }

  return variables;
}

Gather::Gather(const at::Device& destination_device, int64_t dim)
    : destination_device_(destination_device), dim_(dim) {}

Gather::~Gather() = default;

variable_list Gather::apply(variable_list&& inputs) {
  bool all_are_zero_dim = true;
  for (const auto& input : inputs) {
    TORCH_CHECK(
        input.is_cuda(),
        "All inputs to Gather must be CUDA tensors, got ",
        input.toString());
    if (input.dim() > 0) {
      all_are_zero_dim = false;
    }
  }

  const bool unsqueeze_scalars = all_are_zero_dim && dim_ == 0;
  if (unsqueeze_scalars) {
    TORCH_WARN(
        "Was asked to gather along dimension 0, but all "
        "input tensors were scalars; will instead unsqueeze "
        "and return a vector.");
  }

  std::shared_ptr<Node> grad_fn;
  // compute this before moving variables from `inputs`
  if (compute_requires_grad(inputs)) {
    std::vector<at::Device> source_devices;
    source_devices.reserve(inputs.size());
    std::vector<int64_t> input_sizes;
    input_sizes.reserve(inputs.size());
    for (auto& input : inputs) {
      source_devices.push_back(input.device());
      input_sizes.push_back(input.size(dim_));
    }
    grad_fn = std::make_shared<Scatter>(
        std::move(source_devices),
        std::move(input_sizes),
        dim_,
        /*streams=*/std::nullopt,
        /*unsqueeze_scalars=*/unsqueeze_scalars);
    grad_fn->set_next_edges(collect_next_edges(inputs));
  }

  std::vector<at::Tensor> tensors;
  if (unsqueeze_scalars) {
    tensors.reserve(inputs.size());
    for (auto& variable : inputs) {
      tensors.push_back(variable.view(1));
    }
  } else {
    tensors = std::move(inputs);
  }

  // Disable the autograd during the actual computation
  // torch::cuda::gather does not return a view or change things inplace
  // so no need for extra logic here
  at::Tensor variable;
  {
    at::AutoDispatchBelowAutograd mode;
    // This is special logic for torch::cuda::gather!
    const auto destination_index =
        destination_device_.is_cpu() ? -1 : destination_device_.index();
    variable = torch::cuda::gather(tensors, dim_, destination_index);
  }
  if (grad_fn) {
    set_history(variable, grad_fn);
  }
  return {variable};
}

} // namespace torch::autograd
