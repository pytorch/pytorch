#include <torch/csrc/profiler/nvtx_observer.h>

#include <torch/csrc/profiler/util.h>

namespace torch {
namespace profiler {
namespace impl {

struct NVTXThreadLocalState : ProfilerThreadLocalStateBase {
  explicit NVTXThreadLocalState(const ProfilerConfig& config)
      : ProfilerThreadLocalStateBase(config) {
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
    auto tls = ProfilerThreadLocalStateBase::getTLS();
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        tls == nullptr || tls->profilerType() == ActiveProfilerType::NVTX);
    return static_cast<NVTXThreadLocalState*>(tls);
  }
};

std::pair<int, int> getOpIdFromInput(const c10::IValue& input_item) {
  std::pair<int, int> producer_op_pair(-1, -1);
  const at::Tensor& tensor = input_item.toTensor();
  auto state_ptr = NVTXThreadLocalState::getTLS();
  TORCH_INTERNAL_ASSERT(state_ptr, "Expected profiler state set");
  if (tensor.defined()) {
    auto ten_addr =  tensor.unsafeGetTensorImpl();
    // See if Address is in the map already
    auto op_id_tensor_map = state_ptr->producerTensorMap();
    if (op_id_tensor_map.count(ten_addr) > 0) {
      producer_op_pair  =  op_id_tensor_map[ten_addr];
    }
  }
  return producer_op_pair;
}

std::vector<std::pair<int, int>> flattenOpIdList(c10::List<c10::IValue> list, std::string fn_name) {
  std::pair<int, int> default_pair(-1, -1);
  std::vector<std::pair<int, int>> input_op_id_list;
  for (const c10::IValue input : list) {
    if (input.isTensor()) {
      auto producer_op_pair = getOpIdFromInput(input);
      input_op_id_list.push_back(producer_op_pair);
    }
    else {
      input_op_id_list.emplace_back(default_pair);
    }
  }
  return input_op_id_list;
}

std::vector<std::pair<int, int>> getInputTensorOpIds(const at::RecordFunction& fn) {
  int num_inputs = fn.inputs().size();
  std::pair<int, int> undefined_op_pair(-1,-1);
  std::vector<std::pair<int, int>> input_producer_ops_;
  input_producer_ops_.reserve(num_inputs);
  int idx = 0;
  for (const c10::IValue& input_item : fn.inputs()) {
    bool tensor_valid = false;
    if(input_item.isTensor()) {
      const at::Tensor& tensor = input_item.toTensor();
      auto producer_pair = getOpIdFromInput(input_item);
      input_producer_ops_.push_back(producer_pair);
    } else {
      if (input_item.isList()) {
        std::vector<std::pair<int, int>> tmp_op_ids = flattenOpIdList(input_item.toList(), std::string(fn.name()));
        // Extend the current sizes array by the array returned from input sizes
        if (!tmp_op_ids.empty()) {
          input_producer_ops_.insert(input_producer_ops_.end(), tmp_op_ids.begin(), tmp_op_ids.end());
        } else {
          input_producer_ops_.emplace_back(undefined_op_pair);
        }
      } else {
          input_producer_ops_.emplace_back(undefined_op_pair);
      }
    }
    idx++;
  }
  return input_producer_ops_;
}

void updateOutputTensorTracker(const at::RecordFunction& fn) {
  int producer_op_id = int(fn.handle());
  int output_nr = 0;
  auto state_ptr = NVTXThreadLocalState::getTLS();
  TORCH_INTERNAL_ASSERT(state_ptr, "Expected profiler state set");
  for (const c10::IValue& s_tensor : fn.outputs()){
    bool tensor_valid = false;
    if(s_tensor.isTensor()) {
      const at::Tensor& tensor = s_tensor.toTensor();
      if (tensor.defined()) {
        auto ten_addr =  tensor.unsafeGetTensorImpl();
        state_ptr->setProducerTensorMap(ten_addr, producer_op_id, output_nr);
      }
    }
    output_nr++;
  }
}

template <bool report_input_shapes>
std::unique_ptr<at::ObserverContext> enterNVTX(const at::RecordFunction& fn) {
  if (NVTXThreadLocalState::getTLS() != nullptr) {
    auto input_op_ids = getInputTensorOpIds(fn);
    torch::profiler::impl::cudaStubs()->nvtxRangePushA(
        torch::profiler::impl::getNvtxStr(
            fn.name(),
            fn.seqNr(),
            report_input_shapes ? torch::profiler::impl::inputSizes(fn)
                                : std::vector<std::vector<int64_t>>(),
            int(fn.handle()),
            report_input_shapes ? input_op_ids
                                : std::vector<std::pair<int, int>>())
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
            torch::profiler::impl::cudaStubs()->nvtxRangePop();
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
