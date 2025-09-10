#include <torch/nativert/kernels/TritonKernel.h>

#include <fmt/ostream.h>

#include <c10/util/Enumerate.h>
#include <c10/util/Exception.h>

#include <ATen/Tensor.h>
#include <ATen/core/op_registration/op_registration.h>

#include <torch/nativert/executor/DelegateExecutor.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

namespace torch::nativert {

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
  bool found_grid = false;
  for (const auto& attr : node_->attributes()) {
    if (attr.name.empty()) {
      attr_ptrs_.emplace_back(std::visit(
          [](auto&& arg) -> void* {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, None>) {
              return nullptr;
            }
            return static_cast<void*>(const_cast<T*>(&arg));
          },
          attr.value));
    } else if (attr.name == "name") {
      kernel_name = std::get<std::string>(attr.value);
    } else if (attr.name == "grid") {
      found_grid = true;
      auto grid = std::get<std::vector<int64_t>>(attr.value);
      TORCH_CHECK(grid.size() == 3, "grid must be a 3D vector");
      launch_params_.grid_dims = GridDims(
          static_cast<int>(grid[0]),
          static_cast<int>(grid[1]),
          static_cast<int>(grid[2]));
    } else if (attr.name == "num_warps") {
      if (const int num_warps = static_cast<int>(std::get<int64_t>(attr.value));
          num_warps > 0) {
        launch_params_.num_warps = num_warps;
      }
    } else if (attr.name == "shared_memory_bytes") {
      if (const int shared_memory_bytes =
              static_cast<int>(std::get<int64_t>(attr.value));
          shared_memory_bytes > 0) {
        launch_params_.shared_memory_bytes = shared_memory_bytes;
      }
    } else if (attr.name == "output_indices") {
      output_indices_ = std::get<std::vector<int64_t>>(attr.value);
    }
  }

  TORCH_CHECK(!kernel_name.empty(), "kernel name not found");
  TORCH_CHECK(found_grid, "grid attribute not found");
  TORCH_CHECK(!output_indices_.empty(), "output_indices attribute not found");

  auto kernel_prefix = std::string("data/triton") + "/" + kernel_name;

  auto tmp_dir = extractToTemporaryFolder(*reader, kernel_prefix) + "/";

  if (reader->hasRecord(kernel_prefix + "/" + kernel_name + ".cubin")) {
    loader_ = TritonKernelManagerRegistry()->Create(
        at::kCUDA, kernel_name, tmp_dir + kernel_name + ".cubin", "");
    TORCH_CHECK(
        loader_ != nullptr,
        "couldn't find cuda loader -- is this a gpu build?");
  } else if (reader->hasRecord(kernel_prefix + "/" + kernel_name + ".hsaco")) {
    loader_ = TritonKernelManagerRegistry()->Create(
        at::kHIP, kernel_name, tmp_dir + kernel_name + ".hsaco", "");
    TORCH_CHECK(
        loader_ != nullptr,
        "couldn't find cuda loader -- is this a gpu build?");
  } else {
    loader_ = TritonKernelManagerRegistry()->Create(
        at::kCPU,
        kernel_name,
        tmp_dir + kernel_name + ".so",
        tmp_dir + kernel_name + ".launcher.so");
  }

  TORCH_CHECK(
      loader_ != nullptr,
      "couldn't find triton kernel loader -- are you trying to run gpu kernels on a cpu build?");
}

TritonKernel::~TritonKernel() = default;

void TritonKernel::computeInternal(ExecutionFrame& executionFrame) const {
  const auto num_inputs = node_->inputs().size();
  const auto num_attrs = attr_ptrs_.size();

  auto* loader = const_cast<TritonKernelManager*>(loader_.get());

  auto inputs = loader->create_inputs(num_inputs, num_attrs);

  for (const auto i : c10::irange(num_inputs)) {
    inputs->add_arg(input(i, executionFrame).toTensor().data_ptr());
  }

  for (const auto i : c10::irange(num_attrs)) {
    inputs->add_attribute(attr_ptrs_[i]);
  }

  loader->launch(launch_params_, inputs->as_void());

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
  for (const auto& i : output_indices_) {
    out_t[i] = input(i, executionFrame).toTensor();
  }
}

} // namespace torch::nativert
