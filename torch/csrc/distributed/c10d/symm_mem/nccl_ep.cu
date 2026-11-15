#include <torch/csrc/distributed/c10d/symm_mem/nccl_ep.hpp>

#ifdef USE_NCCL_EP

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/csrc/distributed/c10d/cuda/CUDAEventCache.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/NCCLSymmetricMemory.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_devcomm_manager.hpp>
#include <nccl_ep.h>

#include <mutex>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace c10d::nccl_ep {

static_assert(
    static_cast<int>(NcclEpHtFlow::Auto) == NCCL_EP_HT_FLOW_AUTO &&
    static_cast<int>(NcclEpHtFlow::PushPull) == NCCL_EP_HT_FLOW_PUSH_PULL &&
    static_cast<int>(NcclEpHtFlow::PullPush) == NCCL_EP_HT_FLOW_PULL_PUSH);

struct NcclEpCommState {
    c10::cuda::CUDAStream stream;
    std::shared_ptr<at::cuda::CUDAEvent> ready;
    std::mutex mutex;
    std::unordered_set<void*> handles;

    explicit NcclEpCommState(c10::DeviceIndex device)
        : stream(c10::cuda::getStreamFromPool(false, device)),
          ready(CUDAEventCache::get(device)->create(false)) {}
};

class NcclEpWork : public Work {
 public:
    NcclEpWork(
        c10::intrusive_ptr<NcclEpHandle> handle,
        std::shared_ptr<at::cuda::CUDAEvent> done,
        std::vector<at::Tensor> tensors)
        : Work(-1, OpType::ALLTOALL),
          handle_(std::move(handle)),
          done_(std::move(done)),
          tensors_(std::move(tensors)) {}

    bool isCompleted() override {
        return done_->query();
    }

    bool isSuccess() const override {
        return true;
    }

    void synchronize() override {
        done_->block(c10::cuda::getCurrentCUDAStream(done_->device_index()));
    }

    bool wait(std::chrono::milliseconds timeout = kNoTimeout) override {
        TORCH_CHECK_VALUE(
            timeout == kNoTimeout,
            "NCCL EP functional waits do not support a CPU timeout");
        synchronize();
        return true;
    }

 private:
    c10::intrusive_ptr<NcclEpHandle> handle_;
    std::shared_ptr<at::cuda::CUDAEvent> done_;
    std::vector<at::Tensor> tensors_;
};

class NcclEpOperation {
 public:
    // All ops on a group share its staging buffers, so they are serialized on
    // the group stream. Only async ops outlive the call and need their tensors
    // kept alive; sync ops join the caller stream before returning.
    explicit NcclEpOperation(
        const c10::intrusive_ptr<NcclEpGroup>& group,
        bool async_op = false)
        : state_(*group->comm_state),
          async_op_(async_op),
          lock_(state_.mutex),
          caller_(c10::cuda::getCurrentCUDAStream(state_.stream.device_index())) {
        TORCH_CHECK_VALUE(group->group != nullptr, "NCCL EP group is closed");
        state_.ready->record(caller_);
        state_.ready->block(state_.stream);
        guard_.emplace(state_.stream);
    }

    c10::cuda::CUDAStream stream() const {
        return state_.stream;
    }

    void record(const at::Tensor& tensor) {
        TORCH_CHECK_VALUE(
            tensor.is_cuda() && tensor.get_device() == state_.stream.device_index(),
            "NCCL EP tensors must be on the group's CUDA device");
        if (async_op_) {
            c10::cuda::CUDACachingAllocator::recordStream(
                tensor.storage().data_ptr(), state_.stream);
            tensors_.push_back(tensor);
        }
    }

    void finish(
        const c10::intrusive_ptr<NcclEpHandle>& handle,
        const std::vector<at::Tensor>& outputs = {}) {
        auto done = CUDAEventCache::get(state_.stream.device_index())->create(false);
        done->record(state_.stream);
        if (async_op_) {
            auto work = c10::make_intrusive<NcclEpWork>(
                handle, std::move(done), std::move(tensors_));
            for (const auto& output : outputs) {
                c10d::register_work(output, work);
            }
        } else {
            done->block(caller_);
        }
    }

 private:
    NcclEpCommState& state_;
    bool async_op_;
    std::unique_lock<std::mutex> lock_;
    c10::cuda::CUDAStream caller_;
    std::optional<c10::cuda::CUDAStreamGuard> guard_;
    std::vector<at::Tensor> tensors_;
};

// Wraps an at::Tensor as ncclEpTensor_t. Holding the at::Tensor by value bumps
// its refcount, keeping the device buffer alive for the descriptor's lifetime
// and letting desc.sizes alias the Tensor's own shape buffer (no copy).
struct EpTensor {
    at::Tensor t;
    ncclEpTensor_t desc = NCCL_EP_TENSOR_INIT;

    explicit EpTensor(at::Tensor tensor) : t(std::move(tensor)) {
        TORCH_CHECK_VALUE(
            t.is_contiguous(),
            "nccl_ep tensors must be memory-contiguous (call .contiguous())");
        TORCH_CHECK_VALUE(t.dim() > 0, "nccl_ep tensor must have rank >= 1");
        desc.ndim = static_cast<unsigned int>(t.dim());
        desc.datatype = c10d::getNcclDataType(t.scalar_type());
        desc.data = t.data_ptr();
        // at::IntArrayRef stores int64_t; ncclEpTensor_t expects size_t. Bit
        // pattern matches on 64-bit platforms for positive values, and the
        // library only reads sizes. const_cast because the C header doesn't
        // const-qualify the pointer.
        static_assert(sizeof(size_t) == sizeof(int64_t));
        desc.sizes = const_cast<size_t*>(
            reinterpret_cast<const size_t*>(t.sizes().data()));
    }

    // Window-backed constructor: NCCL-EP reads via RDMA from the registered
    // ncclWindow instead of a plain device pointer (zero-copy path).
    EpTensor(ncclWindow_t win, uint64_t win_offset, at::Tensor tensor)
        : t(std::move(tensor)) {
        TORCH_CHECK_VALUE(
            t.is_contiguous(),
            "nccl_ep tensors must be memory-contiguous (call .contiguous())");
        TORCH_CHECK_VALUE(t.dim() > 0, "nccl_ep tensor must have rank >= 1");
        desc.ndim = static_cast<unsigned int>(t.dim());
        desc.datatype = c10d::getNcclDataType(t.scalar_type());
        desc.data = nullptr;
        desc.win_hdl = win;
        desc.win_offset = win_offset;
        static_assert(sizeof(size_t) == sizeof(int64_t));
        desc.sizes = const_cast<size_t*>(
            reinterpret_cast<const size_t*>(t.sizes().data()));
    }
};

// Returns an EpTensor for t. If t is backed by NCCLSymmetricMemory registered
// under group_name, uses the window + offset (zero-copy RDMA path); otherwise
// falls back to the plain device-pointer path.
static EpTensor make_ep_tensor(const at::Tensor& t, std::string_view group_name) {
    namespace sm = c10d::symmetric_memory;
    if (sm::is_symm_mem_tensor(t)) {
        // rendezvous() takes std::optional<std::string> (no null-termination
        // requirement); it is cached after the first collective call.
        auto symm_mem = sm::rendezvous(t, std::string(group_name));
        auto* nccl_sm =
            dynamic_cast<sm::NCCLSymmetricMemory*>(symm_mem.get());
        if (nccl_sm != nullptr) {
            ncclWindow_t win = nccl_sm->get_window();
            TORCH_CHECK(win != ncclWindow_t{}, "nccl_ep: NCCL symmetric memory window is null");
            // window base is the signal pad, not the data buffer.
            // data_ptr - buffer_ptr misses buffer_offset_ and points into the pad.
            const uint64_t offset = nccl_sm->get_window_offset() + static_cast<uint64_t>(t.storage_offset()) * t.element_size();
            return EpTensor(win, offset, t);
        }
    }
    return EpTensor(t);
}

#define NCCL_EP_CHECK(expr)                                             \
    do {                                                                \
        ncclResult_t _r = (expr);                                       \
        TORCH_CHECK(_r == ncclSuccess, "nccl_ep error: ", ncclGetErrorString(_r)); \
    } while (0)

// Translate the header's standalone layout enum to the library enum.
static ncclEpLayout_t to_nccl_layout(NcclEpLayout layout) {
    switch (layout) {
        case NcclEpLayout::ExpertMajor:
            return NCCL_EP_LAYOUT_EXPERT_MAJOR;
        case NcclEpLayout::RankMajor:
            return NCCL_EP_LAYOUT_RANK_MAJOR;
        case NcclEpLayout::Flat:
            return NCCL_EP_LAYOUT_FLAT;
        case NcclEpLayout::Unset:
            break; // not a valid dispatch layout
    }
    TORCH_CHECK(
        false, "nccl_ep: layout must be set; got ", static_cast<int64_t>(layout));
}

static ncclComm_t get_nccl_comm(
    const c10::intrusive_ptr<::c10d::ProcessGroup>& pg) {
    auto device = c10::Device(at::kCUDA, at::cuda::current_device());
    return c10d::symmetric_memory::NCCLDevCommManager::get(device).get_comm(
        pg->getGroupName());
}

// Destructors are noexcept; a throwing CHECK would std::terminate. Warn
// instead of throwing as a defensive fallback.
static void warn_destroy_result(ncclResult_t r, const char* what) {
    if (r != ncclSuccess) {
        TORCH_WARN(what, " ignoring ", ncclGetErrorString(r));
    }
}

NcclEpGroup::NcclEpGroup(c10::DeviceIndex device)
    : device_index(device), comm_state(std::make_shared<NcclEpCommState>(device)) {}

NcclEpGroup::~NcclEpGroup() {
    try {
        close();
    } catch (const std::exception& error) {
        TORCH_WARN("Failed to close NCCL EP group: ", error.what());
    }
}

void NcclEpGroup::close() {
    std::lock_guard<std::mutex> lock(comm_state->mutex);
    if (!group) return;
    c10::cuda::CUDAGuard guard(device_index);
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    // Mark the group and every handle closed before destroying anything, so a
    // failure below leaks native objects instead of leaving dangling pointers.
    auto ep_group = reinterpret_cast<ncclEpGroup_t>(std::exchange(group, nullptr));
    auto handles = std::exchange(comm_state->handles, {});
    for (void* handle : handles) {
        NCCL_EP_CHECK(ncclEpHandleDestroy(reinterpret_cast<ncclEpHandle_t>(handle)));
    }
    NCCL_EP_CHECK(ncclEpGroupDestroy(ep_group));
}

NcclEpHandle::~NcclEpHandle() {
    std::lock_guard<std::mutex> lock(group->comm_state->mutex);
    if (handle && group->comm_state->handles.erase(handle)) {
        c10::cuda::CUDAGuard guard(group->device_index);
        warn_destroy_result(
            ncclEpHandleDestroy(reinterpret_cast<ncclEpHandle_t>(handle)),
            "NcclEpHandle::~NcclEpHandle()");
        handle = nullptr;
    }
}

c10::intrusive_ptr<NcclEpGroup> nccl_ep_create_group(
    const c10::intrusive_ptr<::c10d::ProcessGroup>& pg,
    int64_t num_experts,
    int64_t max_dispatch_tokens_per_rank,
    int64_t max_recv_tokens_per_rank,
    int64_t max_token_bytes, NcclEpHtFlow ht_flow) {
    const int64_t world_size = pg->getSize();
    TORCH_CHECK(
        world_size > 0 && num_experts % world_size == 0,
        "nccl_ep: num_experts (",
        num_experts,
        ") must be divisible by world size (",
        world_size,
        ")");
    const auto device_index = at::cuda::current_device();
    ncclComm_t comm = get_nccl_comm(pg);

    ncclEpGroupConfig_t config = NCCL_EP_GROUP_CONFIG_INIT;
    config.algorithm = NCCL_EP_ALGO_HIGH_THROUGHPUT;
    config.num_experts = static_cast<unsigned int>(num_experts);
    config.max_dispatch_tokens_per_rank =
        static_cast<unsigned int>(max_dispatch_tokens_per_rank);
    config.max_recv_tokens_per_rank =
        static_cast<unsigned int>(max_recv_tokens_per_rank);
    config.max_token_bytes = static_cast<unsigned int>(max_token_bytes);
    config.rdma_buffer_size = NCCL_EP_AUTO;
    config.num_qp_per_rank = NCCL_EP_AUTO;
    config.num_channels = NCCL_EP_AUTO;
    config.ht_flow = static_cast<ncclEpHtFlow_t>(ht_flow);
    auto result = c10::make_intrusive<NcclEpGroup>(device_index);
    result->group_name = pg->getGroupName();
    result->num_local_experts = num_experts / pg->getSize();
    result->max_dispatch_tokens_per_rank = max_dispatch_tokens_per_rank;
    result->max_recv_tokens_per_rank = max_recv_tokens_per_rank;
    result->max_token_bytes = max_token_bytes;
    ncclEpGroup_t ep_group = nullptr;
    NCCL_EP_CHECK(ncclEpCreateGroup(&ep_group, comm, &config));
    result->group = ep_group;
    return result;
}

c10::intrusive_ptr<NcclEpHandle> nccl_ep_create_handle(
    const c10::intrusive_ptr<NcclEpGroup>& group,
    const at::Tensor& topk_idx,
    const std::optional<at::Tensor>& recv_expert_counter,
    NcclEpLayout layout, int64_t alignment) {
    TORCH_CHECK_TYPE(
        topk_idx.scalar_type() == at::kInt || topk_idx.scalar_type() == at::kLong,
        "topk_idx must have dtype int32 or int64, got ",
        topk_idx.scalar_type());
    TORCH_CHECK_VALUE(
        topk_idx.dim() == 2 && topk_idx.size(0) <= group->max_dispatch_tokens_per_rank,
        "topk_idx must be [num_tokens, top_k] with at most ",
        group->max_dispatch_tokens_per_rank,
        " tokens (max_dispatch_tokens_per_rank), got ",
        topk_idx.sizes());
    TORCH_CHECK_VALUE(alignment > 0, "expert alignment must be positive");
    TORCH_CHECK_VALUE(layout == NcclEpLayout::ExpertMajor || alignment == 1,
        "expert alignment is only supported for expert_major routing");
    // Release the operation lock before handle cleanup on an exception.
    c10::intrusive_ptr<NcclEpHandle> handle;
    NcclEpOperation operation(group);
    operation.record(topk_idx);
    auto stream = operation.stream();
    auto ep_group = reinterpret_cast<ncclEpGroup_t>(group->group);

    auto recv_total_counter = at::empty(
        {1}, topk_idx.options().dtype(at::kInt));

    // HT FLAT needs a live int32 counter to use the pec scan kernel; mixing
    // pec then nopec on a reused group has hit illegal-address in the nopec JIT.
    // Expert-major uses the library's internal counter; rank-major is LL-only.
    at::Tensor expert_counter;
    if (recv_expert_counter.has_value()) {
      expert_counter = *recv_expert_counter;
      TORCH_CHECK(
          expert_counter.dim() == 1 &&
              expert_counter.scalar_type() == at::kInt &&
              expert_counter.device() == topk_idx.device() &&
              expert_counter.numel() == group->num_local_experts,
          "nccl_ep_create_handle: recv_expert_counter must be a 1D int32 tensor on ",
          topk_idx.device(),
          " with ",
          group->num_local_experts,
          " elements; got ",
          expert_counter.scalar_type(),
          " on ",
          expert_counter.device(),
          " with ",
          expert_counter.numel(),
          " elements");
      operation.record(expert_counter);
    } else if (layout == NcclEpLayout::Flat) {
      TORCH_CHECK(
          group->num_local_experts > 0,
          "nccl_ep_create_handle: group has no local experts");
      expert_counter = at::zeros(
          {group->num_local_experts}, topk_idx.options().dtype(at::kInt));
    }

    EpTensor topk(topk_idx.to(at::kLong));
    std::optional<EpTensor> total;
    if (recv_total_counter.defined()) total.emplace(recv_total_counter);

    ncclEpLayoutInfo_t layout_info = NCCL_EP_LAYOUT_INFO_INIT;
    layout_info.recv_total_counter = total ? &total->desc : nullptr;

    std::optional<EpTensor> counter;
    if (expert_counter.defined()) {
        counter.emplace(expert_counter);
        layout_info.expert_counters = &counter->desc;
    }

    ncclEpHandleConfig_t handle_config = NCCL_EP_HANDLE_CONFIG_INIT;
    handle_config.dispatch_output_per_expert_alignment = static_cast<size_t>(alignment);
    ncclEpHandle_t ep_handle = nullptr;
    NCCL_EP_CHECK(ncclEpCreateHandle(
        &ep_handle, ep_group,
        to_nccl_layout(layout),
        &topk.desc,
        &layout_info,
        &handle_config,
        stream));

    handle = c10::make_intrusive<NcclEpHandle>(
        group,
        ep_handle,
        layout,
        group->group_name,
        std::move(topk.t),
        std::move(recv_total_counter),
        std::move(expert_counter));
    try {
        group->comm_state->handles.insert(ep_handle);
    } catch (...) {
        ncclEpHandleDestroy(ep_handle);
        handle->handle = nullptr;
        throw;
    }
    operation.finish(handle);
    return handle;
}

void nccl_ep_update_handle(
    const c10::intrusive_ptr<NcclEpHandle>& handle,
    const at::Tensor& topk_idx,
    const std::optional<at::Tensor>& recv_expert_counter) {
    TORCH_CHECK_TYPE(
        topk_idx.scalar_type() == at::kInt || topk_idx.scalar_type() == at::kLong,
        "topk_idx must have dtype int32 or int64, got ",
        topk_idx.scalar_type());
    TORCH_CHECK_VALUE(
        topk_idx.sizes() == handle->topk_idx.sizes(),
        "Routing updates must preserve the original top-k shape");
    TORCH_CHECK_VALUE(
        topk_idx.device() == handle->topk_idx.device(),
        "Routing updates must preserve the original device");
    TORCH_CHECK_VALUE(
        handle->layout != NcclEpLayout::ExpertMajor || recv_expert_counter.has_value(),
        "expert_major routing updates require per-expert token counts");
    NcclEpOperation operation(handle->group);
    operation.record(topk_idx);
    if (handle->recv_total_counter.defined()) operation.record(handle->recv_total_counter);
    at::Tensor expert_counter = recv_expert_counter.value_or(handle->recv_expert_counter);
    if (expert_counter.defined()) {
        TORCH_CHECK_VALUE(
            expert_counter.dim() == 1 && expert_counter.scalar_type() == at::kInt &&
                expert_counter.device() == topk_idx.device() &&
                expert_counter.numel() == handle->group->num_local_experts,
            "recv_expert_counter must be a 1D int32 tensor with one entry per local expert");
        operation.record(expert_counter);
    }
    EpTensor topk(topk_idx.to(at::kLong));
    std::optional<EpTensor> total;
    if (handle->recv_total_counter.defined()) total.emplace(handle->recv_total_counter);
    std::optional<EpTensor> counter;
    if (expert_counter.defined()) counter.emplace(expert_counter);
    ncclEpLayoutInfo_t layout_info = NCCL_EP_LAYOUT_INFO_INIT;
    layout_info.recv_total_counter = total ? &total->desc : nullptr;
    layout_info.expert_counters = counter ? &counter->desc : nullptr;
    NCCL_EP_CHECK(ncclEpUpdateHandle(
        reinterpret_cast<ncclEpHandle_t>(handle->handle),
        &topk.desc,
        &layout_info,
        operation.stream()));
    handle->topk_idx = std::move(topk.t);
    handle->recv_expert_counter = std::move(expert_counter);
    operation.finish(handle);
}

int64_t nccl_ep_handle_get_num_recv_tokens(
    const c10::intrusive_ptr<NcclEpHandle>& handle) {
    NcclEpOperation operation(handle->group);
    TORCH_CHECK(
        handle->recv_total_counter.defined(),
        "nccl_ep_handle_get_num_recv_tokens: handle has no recv_total_counter");
    // The counter is written by the metadata kernel launched in
    // ncclEpCreateHandle / ncclEpUpdateHandle on the group stream; item() runs
    // under the operation's stream guard, so it reads after that kernel.
    return handle->recv_total_counter.item<int32_t>();
}

// libnccl_ep checks these shapes only with assert(), which Release builds
// compile out, so mismatches would become out-of-bounds device accesses.
// Combine inputs are not bounded here: they may be sliced to the device-side
// receive count.
static void check_rows(
    const NcclEpHandle& handle,
    const at::Tensor& tokens,
    const char* name,
    bool check_row_bytes = true) {
    TORCH_CHECK_VALUE(tokens.dim() == 2, name, " must be 2D, got ", tokens.sizes());
    if (!check_row_bytes) return;
    const int64_t row_bytes = tokens.size(1) * tokens.element_size();
    TORCH_CHECK_VALUE(
        row_bytes <= handle.group->max_token_bytes,
        name, " rows are ", row_bytes, " bytes, above max_token_bytes (",
        handle.group->max_token_bytes, ")");
}

static void check_weights(
    const at::Tensor& weights, int64_t top_k, bool expert_major, const char* name) {
    TORCH_CHECK_VALUE(
        weights.scalar_type() == at::kFloat &&
            (expert_major ? weights.dim() == 1
                          : weights.dim() == 2 && weights.size(1) == top_k),
        name, " must be float32 ", expert_major ? "[slots]" : "[slots, top_k]",
        " with top_k=", top_k, ", got ", weights.scalar_type(), " ", weights.sizes());
}

static void check_dispatch_shapes(
    const NcclEpHandle& handle,
    const at::Tensor& tokens,
    const std::optional<at::Tensor>& topk_weights,
    const at::Tensor& out_tokens,
    const std::optional<at::Tensor>& out_topk_weights,
    const std::optional<at::Tensor>& out_topk_idx) {
    const auto& topk = handle.topk_idx;
    const int64_t capacity = handle.group->max_recv_tokens_per_rank;
    check_rows(handle, tokens, "tokens");
    TORCH_CHECK_VALUE(
        tokens.size(0) == topk.size(0),
        "tokens must have one row per routed token (", topk.size(0), "), got ",
        tokens.size(0));
    if (topk_weights) {
        TORCH_CHECK_VALUE(
            topk_weights->scalar_type() == at::kFloat &&
                topk_weights->sizes() == topk.sizes(),
            "topk_weights must be float32 with the routing's shape ", topk.sizes(),
            ", got ", topk_weights->scalar_type(), " ", topk_weights->sizes());
    }
    TORCH_CHECK_VALUE(
        out_tokens.dim() == 2 && out_tokens.size(0) >= capacity &&
            out_tokens.size(1) == tokens.size(1) &&
            out_tokens.element_size() == tokens.element_size(),
        "out_tokens must have at least ", capacity,
        " rows (max_recv_tokens_per_rank) and match the token row width and "
        "element size, got ", out_tokens.scalar_type(), " ", out_tokens.sizes());
    if (out_topk_weights) {
        check_weights(
            *out_topk_weights, topk.size(1),
            handle.layout == NcclEpLayout::ExpertMajor, "out_topk_weights");
        TORCH_CHECK_VALUE(
            out_topk_weights->size(0) >= capacity,
            "out_topk_weights must have at least ", capacity, " rows");
    }
    if (out_topk_idx) {
        TORCH_CHECK_VALUE(
            out_topk_idx->dim() == 2 && out_topk_idx->size(0) >= capacity &&
                out_topk_idx->size(1) == topk.size(1),
            "out_topk_idx must be [>= ", capacity, ", ", topk.size(1), "], got ",
            out_topk_idx->sizes());
    }
}

static void check_combine_shapes(
    const NcclEpHandle& handle,
    const at::Tensor& expert_tokens,
    const at::Tensor& out_tokens,
    const std::optional<at::Tensor>& topk_weights,
    const std::optional<at::Tensor>& out_topk_weights,
    bool check_row_bytes) {
    const auto& topk = handle.topk_idx;
    check_rows(handle, expert_tokens, "expert_tokens", check_row_bytes);
    TORCH_CHECK_VALUE(
        out_tokens.dim() == 2 && out_tokens.size(0) == topk.size(0) &&
            out_tokens.size(1) == expert_tokens.size(1) &&
            out_tokens.element_size() == expert_tokens.element_size(),
        "combine output must be [", topk.size(0), ", ", expert_tokens.size(1),
        "] with the expert token element size, got ", out_tokens.scalar_type(),
        " ", out_tokens.sizes());
    if (topk_weights) {
        check_weights(
            *topk_weights, topk.size(1),
            handle.layout == NcclEpLayout::ExpertMajor, "topk_weights");
    }
    if (out_topk_weights) {
        TORCH_CHECK_VALUE(
            out_topk_weights->scalar_type() == at::kFloat &&
                out_topk_weights->sizes() == topk.sizes(),
            "out_topk_weights must be float32 with the routing's shape ",
            topk.sizes(), ", got ", out_topk_weights->scalar_type(), " ",
            out_topk_weights->sizes());
    }
}

static void nccl_ep_dispatch_impl(
    const c10::intrusive_ptr<NcclEpHandle>& handle,
    const at::Tensor& tokens,
    const std::optional<at::Tensor>& topk_weights,
    at::Tensor& out_tokens,
    std::optional<at::Tensor> out_topk_weights,
    std::optional<at::Tensor> out_topk_idx,
    bool backward,
    bool async_op,
    const std::optional<at::Tensor>& scales = std::nullopt,
    const std::optional<at::Tensor>& out_scales = std::nullopt,
    ncclEpDispQuant_t quant_recipe = NCCL_EP_DISP_QUANT_NONE) {
    TORCH_INTERNAL_ASSERT(
        scales.has_value() == (quant_recipe == NCCL_EP_DISP_QUANT_FWD) &&
        out_scales.has_value() == scales.has_value());
    check_dispatch_shapes(
        *handle, tokens, topk_weights, out_tokens, out_topk_weights, out_topk_idx);
    if (scales) {
        TORCH_CHECK_VALUE(
            scales->dim() == 2 && scales->size(0) == tokens.size(0) &&
                out_scales->dim() == 2 &&
                out_scales->size(0) >= out_tokens.size(0) &&
                out_scales->size(1) == scales->size(1) &&
                out_scales->element_size() == scales->element_size(),
            "scales must be [num_tokens, n] and out_scales [>= out_tokens rows, n] "
            "with the same element size, got ", scales->sizes(), " and ",
            out_scales->sizes());
        for (const at::Tensor* value : {&*scales, &*out_scales}) {
            TORCH_CHECK_VALUE(
                reinterpret_cast<uintptr_t>(value->data_ptr()) % 16 == 0,
                "MXFP8 scale storage must be 16-byte aligned");
        }
    }
    NcclEpOperation operation(handle->group, async_op);
    auto stream = operation.stream();
    auto ep_handle = reinterpret_cast<ncclEpHandle_t>(handle->handle);

    // topk_idx is bound to the handle at nccl_ep_create_handle time; the new
    // ncclEpDispatch API doesn't take it as a per-call input.
    operation.record(tokens);
    operation.record(out_tokens);
    if (topk_weights) operation.record(*topk_weights);
    if (out_topk_weights) operation.record(*out_topk_weights);
    if (out_topk_idx) operation.record(*out_topk_idx);
    std::optional<EpTensor> in_scales, out_scales_desc;
    if (scales) {
        operation.record(*scales);
        operation.record(*out_scales);
        in_scales.emplace(make_ep_tensor(*scales, handle->group_name));
        out_scales_desc.emplace(make_ep_tensor(*out_scales, handle->group_name));
    }
    EpTensor in_tokens = make_ep_tensor(tokens, handle->group_name);
    std::optional<EpTensor> in_weights;
    if (topk_weights) in_weights.emplace(*topk_weights);
    // Output may also be symm_mem-backed (zero-copy RDMA write path).
    EpTensor out_tok = make_ep_tensor(out_tokens, handle->group_name);
    std::optional<EpTensor> out_wts, out_idx;
    if (out_topk_weights) out_wts.emplace(*out_topk_weights);
    if (out_topk_idx) out_idx.emplace(*out_topk_idx);

    ncclEpDispatchInputs_t inputs = NCCL_EP_DISPATCH_INPUTS_INIT;
    inputs.tokens = &in_tokens.desc;
    inputs.scales = in_scales ? &in_scales->desc : nullptr;
    inputs.topk_weights = in_weights ? &in_weights->desc : nullptr;

    ncclEpDispatchOutputs_t outputs = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    outputs.tokens = &out_tok.desc;
    outputs.scales = out_scales_desc ? &out_scales_desc->desc : nullptr;
    outputs.topk_weights = out_wts ? &out_wts->desc : nullptr;
    outputs.topk_idx = out_idx ? &out_idx->desc : nullptr;

    ncclEpDispatchConfig_t config = NCCL_EP_DISPATCH_CONFIG_INIT;
    config.quant_recipe = quant_recipe;
    config.pass_direction = backward ? NCCL_EP_BWD_PASS : NCCL_EP_FWD_PASS;

    NCCL_EP_CHECK(ncclEpDispatch(
        ep_handle, &inputs, &outputs,
        /*layout_info=*/nullptr,
        &config, stream));
    std::vector<at::Tensor> result_tensors{out_tokens};
    if (out_scales) result_tensors.push_back(*out_scales);
    if (out_topk_weights) result_tensors.push_back(*out_topk_weights);
    if (out_topk_idx) result_tensors.push_back(*out_topk_idx);
    operation.finish(handle, result_tensors);
}

void nccl_ep_dispatch(
    const c10::intrusive_ptr<NcclEpHandle>& handle,
    const at::Tensor& tokens,
    const at::Tensor& topk_weights,
    at::Tensor& out_tokens,
    std::optional<at::Tensor> out_topk_weights,
    std::optional<at::Tensor> out_topk_idx,
    bool async_op) {
    nccl_ep_dispatch_impl(
        handle, tokens, topk_weights, out_tokens, out_topk_weights, out_topk_idx, false, async_op);
}

void nccl_ep_dispatch_backward(
    const c10::intrusive_ptr<NcclEpHandle>& handle,
    const at::Tensor& grad_tokens,
    at::Tensor& out_tokens,
    bool async_op) {
    nccl_ep_dispatch_impl(
        handle, grad_tokens, std::nullopt, out_tokens, std::nullopt, std::nullopt, true, async_op);
}

static void nccl_ep_combine_impl(
    const c10::intrusive_ptr<NcclEpHandle>& handle,
    const at::Tensor& expert_tokens,
    at::Tensor& out_tokens,
    const std::optional<at::Tensor>& topk_weights,
    const std::optional<at::Tensor>& out_topk_weights,
    bool backward,
    bool async_op,
    ncclEpCombQuant_t quant_recipe = NCCL_EP_COMB_QUANT_NONE) {
    // MXFP8 combine checks its quantized row size natively.
    check_combine_shapes(
        *handle, expert_tokens, out_tokens, topk_weights, out_topk_weights,
        quant_recipe == NCCL_EP_COMB_QUANT_NONE);
    NcclEpOperation operation(handle->group, async_op);
    auto stream = operation.stream();
    auto ep_handle = reinterpret_cast<ncclEpHandle_t>(handle->handle);

    operation.record(expert_tokens);
    operation.record(out_tokens);
    if (topk_weights) operation.record(*topk_weights);
    if (out_topk_weights) operation.record(*out_topk_weights);
    EpTensor in_tok = make_ep_tensor(expert_tokens, handle->group_name);
    EpTensor out_tok(out_tokens);
    std::optional<EpTensor> in_weights, out_weights;
    if (topk_weights) in_weights.emplace(*topk_weights);
    if (out_topk_weights) out_weights.emplace(*out_topk_weights);

    ncclEpCombineInputs_t inputs = NCCL_EP_COMBINE_INPUTS_INIT;
    inputs.tokens = &in_tok.desc;
    inputs.topk_weights = in_weights ? &in_weights->desc : nullptr;

    ncclEpCombineOutputs_t outputs = NCCL_EP_COMBINE_OUTPUTS_INIT;
    outputs.tokens = &out_tok.desc;
    outputs.topk_weights = out_weights ? &out_weights->desc : nullptr;

    ncclEpCombineConfig_t config = NCCL_EP_COMBINE_CONFIG_INIT;
    config.quant_recipe = quant_recipe;
    config.pass_direction = backward ? NCCL_EP_BWD_PASS : NCCL_EP_FWD_PASS;

    NCCL_EP_CHECK(ncclEpCombine(
        ep_handle, &inputs, &outputs, &config, stream));
    std::vector<at::Tensor> result_tensors{out_tokens};
    if (out_topk_weights) result_tensors.push_back(*out_topk_weights);
    operation.finish(handle, result_tensors);
}


void nccl_ep_combine(
    const c10::intrusive_ptr<NcclEpHandle>& handle,
    const at::Tensor& expert_tokens,
    at::Tensor& out_tokens,
    bool async_op) {
    nccl_ep_combine_impl(
        handle, expert_tokens, out_tokens, std::nullopt, std::nullopt, false, async_op);
}

void nccl_ep_combine_backward(
    const c10::intrusive_ptr<NcclEpHandle>& handle,
    const at::Tensor& grad_tokens,
    const at::Tensor& grad_topk_weights,
    at::Tensor& out_tokens,
    at::Tensor& out_topk_weights,
    bool async_op) {
    nccl_ep_combine_impl(
        handle, grad_tokens, out_tokens, grad_topk_weights, out_topk_weights, true, async_op);
}

void nccl_ep_dispatch_quantized(
    const c10::intrusive_ptr<NcclEpHandle>& handle,
    const at::Tensor& tokens,
    const at::Tensor& scales,
    const at::Tensor& topk_weights,
    at::Tensor& out_tokens,
    at::Tensor& out_scales,
    std::optional<at::Tensor> out_topk_weights,
    std::optional<at::Tensor> out_topk_idx,
    bool async_op) {
    nccl_ep_dispatch_impl(handle, tokens, topk_weights, out_tokens,
        out_topk_weights, out_topk_idx, false, async_op,
        scales, out_scales, NCCL_EP_DISP_QUANT_FWD);
}

void nccl_ep_dispatch_backward_quantized(
    const c10::intrusive_ptr<NcclEpHandle>& handle,
    const at::Tensor& tokens,
    const at::Tensor& scales,
    at::Tensor& out_tokens,
    at::Tensor& out_scales,
    bool async_op) {
    nccl_ep_dispatch_impl(handle, tokens, std::nullopt, out_tokens,
        std::nullopt, std::nullopt, true, async_op,
        scales, out_scales, NCCL_EP_DISP_QUANT_FWD);
}

void nccl_ep_combine_quantized(
    const c10::intrusive_ptr<NcclEpHandle>& handle,
    const at::Tensor& tokens,
    at::Tensor& out_tokens,
    bool async_op) {
    nccl_ep_combine_impl(handle, tokens, out_tokens, std::nullopt,
        std::nullopt, false, async_op, NCCL_EP_COMB_QUANT_MXFP8);
}

void nccl_ep_combine_backward_quantized(
    const c10::intrusive_ptr<NcclEpHandle>& handle,
    const at::Tensor& tokens,
    const at::Tensor& topk_weights,
    at::Tensor& out_tokens,
    at::Tensor& out_topk_weights,
    bool async_op) {
    nccl_ep_combine_impl(handle, tokens, out_tokens, topk_weights,
        out_topk_weights, true, async_op, NCCL_EP_COMB_QUANT_MXFP8);
}

} // namespace c10d::nccl_ep

#else // USE_NCCL_EP

namespace c10d::nccl_ep {

NcclEpGroup::NcclEpGroup(c10::DeviceIndex device) : device_index(device) {}
NcclEpGroup::~NcclEpGroup() = default;
NcclEpHandle::~NcclEpHandle() = default;

[[noreturn]] static void not_supported() {
    TORCH_CHECK(false, "PyTorch was not built with USE_NCCL_EP=1");
}

void NcclEpGroup::close() {
    not_supported();
}

c10::intrusive_ptr<NcclEpGroup> nccl_ep_create_group(
    const c10::intrusive_ptr<::c10d::ProcessGroup>&,
    int64_t, int64_t, int64_t, int64_t, NcclEpHtFlow) {
    not_supported();
}

c10::intrusive_ptr<NcclEpHandle> nccl_ep_create_handle(
    const c10::intrusive_ptr<NcclEpGroup>&,
    const at::Tensor&,
    const std::optional<at::Tensor>&,
    NcclEpLayout, int64_t) {
    not_supported();
}

void nccl_ep_update_handle(
    const c10::intrusive_ptr<NcclEpHandle>&,
    const at::Tensor&,
    const std::optional<at::Tensor>&) {
    not_supported();
}

int64_t nccl_ep_handle_get_num_recv_tokens(
    const c10::intrusive_ptr<NcclEpHandle>&) {
    not_supported();
}

void nccl_ep_dispatch(
    const c10::intrusive_ptr<NcclEpHandle>&,
    const at::Tensor&, const at::Tensor&,
    at::Tensor&, std::optional<at::Tensor>, std::optional<at::Tensor>, bool) {
    not_supported();
}

void nccl_ep_dispatch_backward(
    const c10::intrusive_ptr<NcclEpHandle>&,
    const at::Tensor&,
    at::Tensor&, bool) {
    not_supported();
}

void nccl_ep_combine(
    const c10::intrusive_ptr<NcclEpHandle>&,
    const at::Tensor&,
    at::Tensor&, bool) {
    not_supported();
}


void nccl_ep_combine_backward(
    const c10::intrusive_ptr<NcclEpHandle>&,
    const at::Tensor&,
    const at::Tensor&,
    at::Tensor&,
    at::Tensor&, bool) {
    not_supported();
}

void nccl_ep_dispatch_quantized(
    const c10::intrusive_ptr<NcclEpHandle>&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&,
    at::Tensor&, at::Tensor&,
    std::optional<at::Tensor>, std::optional<at::Tensor>, bool) {
    not_supported();
}

void nccl_ep_dispatch_backward_quantized(
    const c10::intrusive_ptr<NcclEpHandle>&,
    const at::Tensor&, const at::Tensor&,
    at::Tensor&, at::Tensor&, bool) {
    not_supported();
}

void nccl_ep_combine_quantized(
    const c10::intrusive_ptr<NcclEpHandle>&,
    const at::Tensor&, at::Tensor&, bool) {
    not_supported();
}

void nccl_ep_combine_backward_quantized(
    const c10::intrusive_ptr<NcclEpHandle>&,
    const at::Tensor&, const at::Tensor&,
    at::Tensor&, at::Tensor&, bool) {
    not_supported();
}

} // namespace c10d::nccl_ep

#endif // USE_NCCL_EP
