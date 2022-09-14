#include <torch/csrc/profiler/nvtx_observer.h>

#include <torch/csrc/profiler/util.h>

namespace torch {
namespace profiler {
namespace impl {

struct NVTXThreadLocalState : ProfilerStateBase {
  explicit NVTXThreadLocalState(const ProfilerConfig& config)
      : ProfilerStateBase(config) {
    // Only `report_input_shapes` makes sense in this context.
    TORCH_CHECK(!config.profile_memory);
    TORCH_CHECK(!config.with_stack);
    TORCH_CHECK(!config.with_flops);
    TORCH_CHECK(!config.with_modules);
  }
  ~NVTXThreadLocalState() override = default;

  ActiveProfilerType profilerType() override {
    return ActiveProfilerType::NVTX;
  }

  void reportMemoryUsage(void*, int64_t, int64_t, int64_t, c10::Device)
      override {}

  static NVTXThreadLocalState* getTLS() {
    auto tls = ProfilerStateBase::get(/*global=*/false);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        tls == nullptr || tls->profilerType() == ActiveProfilerType::NVTX);
    return static_cast<NVTXThreadLocalState*>(tls);
  }
  std::pair<at::RecordFunctionHandle, int> getOpIdFromInput(
      const at::Tensor& tensor);

  void setProducerTensorMap(
      at::TensorImpl* tensor,
      at::RecordFunctionHandle op_id,
      int output_nr) {
    producer_tensor_map_[(void*)tensor] =
        std::pair<at::RecordFunctionHandle, int>{op_id, output_nr};
  }

 protected:
  // Maps the address of an output Tensor to a unique op id and output
  // index of the tensor.
  // at::TensorImpl* is the actual type of the key, but using void*
  // to indicate the pointer is just being used as a key
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::unordered_map<void*, std::pair<at::RecordFunctionHandle, int>>
      producer_tensor_map_;
};

std::pair<at::RecordFunctionHandle, int> NVTXThreadLocalState::getOpIdFromInput(
    const at::Tensor& tensor) {
  std::pair<at::RecordFunctionHandle, int> producer_op_pair(0, -1);
  if (tensor.defined()) {
    at::TensorImpl* ten_addr = tensor.unsafeGetTensorImpl();
    // See if Address is in the map already
    if (producer_tensor_map_.count((void*)ten_addr) > 0) {
      producer_op_pair = producer_tensor_map_[(void*)ten_addr];
    }
  }
  return producer_op_pair;
}

std::list<std::pair<at::RecordFunctionHandle, int>> flattenOpIdList(
    c10::List<c10::IValue> list,
    std::string fn_name) {
  std::list<std::pair<at::RecordFunctionHandle, int>> input_op_id_list;
  auto state_ptr = NVTXThreadLocalState::getTLS();
  TORCH_INTERNAL_ASSERT(state_ptr, "Expected profiler state set");
  for (const c10::IValue input : list) {
    if (input.isTensor()) {
      const at::Tensor& tensor = input.toTensor();
      auto producer_op_pair = state_ptr->getOpIdFromInput(tensor);
      input_op_id_list.push_back(producer_op_pair);
    }
  }
  return input_op_id_list;
}

std::list<std::pair<at::RecordFunctionHandle, int>> getInputTensorOpIds(
    const at::RecordFunction& fn) {
  std::pair<at::RecordFunctionHandle, int> undefined_op_pair(0, -1);
  std::list<std::pair<at::RecordFunctionHandle, int>> input_producer_ops_;
  auto state_ptr = NVTXThreadLocalState::getTLS();
  TORCH_INTERNAL_ASSERT(state_ptr, "Expected profiler state set");
  for (const c10::IValue& input_item : fn.inputs()) {
    if (input_item.isTensor()) {
      const at::Tensor& tensor = input_item.toTensor();
      auto producer_pair = state_ptr->getOpIdFromInput(tensor);
      input_producer_ops_.push_back(producer_pair);
    } else {
      if (input_item.isList()) {
        std::list<std::pair<at::RecordFunctionHandle, int>> tmp_op_ids =
            flattenOpIdList(input_item.toList(), std::string(fn.name()));
        // Extend the current sizes array by the array returned from input sizes
        if (!tmp_op_ids.empty()) {
          input_producer_ops_.splice(input_producer_ops_.end(), tmp_op_ids);
        } else {
          input_producer_ops_.emplace_back(undefined_op_pair);
        }
      } else {
        input_producer_ops_.emplace_back(undefined_op_pair);
      }
    }
  }
  return input_producer_ops_;
}

void updateOutputTensorTracker(const at::RecordFunction& fn) {
  int output_nr = 0;
  auto state_ptr = NVTXThreadLocalState::getTLS();
  TORCH_INTERNAL_ASSERT(state_ptr, "Expected profiler state set");
  for (const c10::IValue& s_tensor : fn.outputs()) {
    if (s_tensor.isTensor()) {
      const at::Tensor& tensor = s_tensor.toTensor();
      if (tensor.defined()) {
        auto ten_addr = tensor.unsafeGetTensorImpl();
        state_ptr->setProducerTensorMap(ten_addr, fn.handle(), output_nr);
      }
    }
    output_nr++;
  }
}

template <bool report_input_shapes>
std::unique_ptr<at::ObserverContext> enterNVTX(const at::RecordFunction& fn) {
  if (NVTXThreadLocalState::getTLS() != nullptr) {
    auto input_op_ids = getInputTensorOpIds(fn);
    torch::profiler::impl::cudaStubs()->rangePush(
        torch::profiler::impl::getNvtxStr(
            fn.name(),
            fn.seqNr(),
            report_input_shapes ? torch::profiler::impl::inputSizes(fn, true)
                                : std::vector<std::vector<int64_t>>(),
            fn.handle(),
            report_input_shapes
                ? input_op_ids
                : std::list<std::pair<at::RecordFunctionHandle, int>>())
            .c_str());
  }
  return nullptr;
}

void pushNVTXCallbacks(
    const ProfilerConfig& config,
    const std::unordered_set<at::RecordScope>& scopes) {
  TORCH_CHECK(
      torch::profiler::impl::cudaStubs()->enabled(),
      "Can't use NVTX profiler - PyTorch was compiled without CUDA");

  c10::ThreadLocalDebugInfo::_push(
      c10::DebugInfoKind::PROFILER_STATE,
      std::make_shared<NVTXThreadLocalState>(config));

  auto state_ptr = NVTXThreadLocalState::getTLS();
  TORCH_INTERNAL_ASSERT(state_ptr, "Expected profiler state set");

  auto handle = at::addThreadLocalCallback(
      at::RecordFunctionCallback(
          state_ptr->config().report_input_shapes
              ? &enterNVTX</*report_input_shapes=*/true>
              : &enterNVTX</*report_input_shapes=*/false>,
          [](const at::RecordFunction& fn, at::ObserverContext* ctx) {
            torch::profiler::impl::cudaStubs()->rangePop();
            updateOutputTensorTracker(fn);
          })
          .needsInputs(config.report_input_shapes)
          .needsOutputs(config.report_input_shapes)
          .needsIds(true)
          .scopes(scopes));
  state_ptr->setCallbackHandle(handle);
}

} // namespace impl
} // namespace profiler
} // namespace torch
