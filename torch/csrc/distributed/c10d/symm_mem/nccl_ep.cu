#include <torch/csrc/distributed/c10d/symm_mem/nccl_ep.hpp>

#ifdef USE_NCCL_EP

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/NCCLSymmetricMemory.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>
#include <nccl_ep.h>

#include <string_view>

namespace c10d::nccl_ep {

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
            void* base = nccl_sm->get_buffer_ptrs()[nccl_sm->get_rank()];
            uint64_t offset = static_cast<uint8_t*>(t.data_ptr()) -
                              static_cast<uint8_t*>(base);
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
    auto* ncclPg = dynamic_cast<c10d::ProcessGroupNCCL*>(
        pg->getBackend(c10::DeviceType::CUDA).get());
    TORCH_CHECK(ncclPg != nullptr, "backend must be a NCCL process group");
    return reinterpret_cast<ncclComm_t>(ncclPg->getCommPtr());
}

NcclEpGroup::~NcclEpGroup() {
    if (group) {
        ncclEpGroupDestroy(reinterpret_cast<ncclEpGroup_t>(group));
        group = nullptr;
    }
}

NcclEpHandle::~NcclEpHandle() {
    if (handle) {
        ncclEpHandleDestroy(reinterpret_cast<ncclEpHandle_t>(handle));
        handle = nullptr;
    }
}

c10::intrusive_ptr<NcclEpGroup> nccl_ep_create_group(
    const c10::intrusive_ptr<::c10d::ProcessGroup>& pg,
    int64_t num_experts,
    int64_t max_dispatch_tokens_per_rank,
    int64_t max_recv_tokens_per_rank,
    int64_t max_token_bytes) {
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

    ncclEpGroup_t ep_group = nullptr;
    NCCL_EP_CHECK(ncclEpCreateGroup(&ep_group, comm, &config));

    auto result = c10::make_intrusive<NcclEpGroup>();
    result->group = ep_group;
    result->group_name = pg->getGroupName();
    return result;
}

c10::intrusive_ptr<NcclEpHandle> nccl_ep_create_handle(
    const c10::intrusive_ptr<NcclEpGroup>& group,
    const at::Tensor& topk_idx,
    const std::optional<at::Tensor>& recv_expert_counter,
    NcclEpLayout layout) {
    auto stream = at::cuda::getCurrentCUDAStream();
    auto ep_group = reinterpret_cast<ncclEpGroup_t>(group->group);

    auto recv_total_counter = at::empty(
        {1}, topk_idx.options().dtype(at::kInt));

    EpTensor topk(topk_idx);
    EpTensor total(recv_total_counter);

    ncclEpLayoutInfo_t layout_info = NCCL_EP_LAYOUT_INFO_INIT;
    layout_info.recv_total_counter = &total.desc;

    std::optional<EpTensor> counter;
    if (recv_expert_counter) {
        counter.emplace(*recv_expert_counter);
        layout_info.expert_counters = &counter->desc;
    }

    ncclEpHandle_t ep_handle = nullptr;
    NCCL_EP_CHECK(ncclEpCreateHandle(
        &ep_handle, ep_group,
        to_nccl_layout(layout),
        &topk.desc,
        &layout_info,
        /*config=*/nullptr,
        stream));

    return c10::make_intrusive<NcclEpHandle>(
        ep_handle,
        layout,
        group->group_name,
        topk_idx,
        std::move(recv_total_counter));
}

int64_t nccl_ep_handle_get_num_recv_tokens(
    const c10::intrusive_ptr<NcclEpHandle>& handle) {
    TORCH_CHECK(
        handle->recv_total_counter.defined(),
        "nccl_ep_handle_get_num_recv_tokens: handle has no recv_total_counter");
    // The counter is written by the metadata kernel launched in
    // ncclEpCreateHandle / ncclEpUpdateHandle on the current stream; item()
    // does a stream-correct synchronizing read.
    return handle->recv_total_counter.item<int32_t>();
}

void nccl_ep_dispatch(
    const c10::intrusive_ptr<NcclEpHandle>& handle,
    const at::Tensor& tokens,
    const at::Tensor& topk_weights,
    at::Tensor& out_tokens,
    std::optional<at::Tensor> out_topk_weights,
    std::optional<at::Tensor> out_topk_idx) {
    auto stream = at::cuda::getCurrentCUDAStream();
    auto ep_handle = reinterpret_cast<ncclEpHandle_t>(handle->handle);

    // topk_idx is bound to the handle at nccl_ep_create_handle time; the new
    // ncclEpDispatch API doesn't take it as a per-call input.
    EpTensor in_tokens = make_ep_tensor(tokens, handle->group_name);
    EpTensor in_weights(topk_weights);
    // Output may also be symm_mem-backed (zero-copy RDMA write path).
    EpTensor out_tok = make_ep_tensor(out_tokens, handle->group_name);
    std::optional<EpTensor> out_wts, out_idx;
    if (out_topk_weights) out_wts.emplace(*out_topk_weights);
    if (out_topk_idx) out_idx.emplace(*out_topk_idx);

    ncclEpDispatchInputs_t inputs = NCCL_EP_DISPATCH_INPUTS_INIT;
    inputs.tokens = &in_tokens.desc;
    inputs.topk_weights = &in_weights.desc;

    ncclEpDispatchOutputs_t outputs = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    outputs.tokens = &out_tok.desc;
    outputs.topk_weights = out_wts ? &out_wts->desc : nullptr;
    outputs.topk_idx = out_idx ? &out_idx->desc : nullptr;

    ncclEpDispatchConfig_t config = NCCL_EP_DISPATCH_CONFIG_INIT;

    NCCL_EP_CHECK(ncclEpDispatch(
        ep_handle, &inputs, &outputs,
        /*layout_info=*/nullptr,
        &config, stream));
}

void nccl_ep_combine(
    const c10::intrusive_ptr<NcclEpHandle>& handle,
    const at::Tensor& expert_tokens,
    at::Tensor& out_tokens) {
    auto stream = at::cuda::getCurrentCUDAStream();
    auto ep_handle = reinterpret_cast<ncclEpHandle_t>(handle->handle);

    EpTensor in_tok = make_ep_tensor(expert_tokens, handle->group_name);
    EpTensor out_tok(out_tokens);

    ncclEpCombineInputs_t inputs = NCCL_EP_COMBINE_INPUTS_INIT;
    inputs.tokens = &in_tok.desc;

    ncclEpCombineOutputs_t outputs = NCCL_EP_COMBINE_OUTPUTS_INIT;
    outputs.tokens = &out_tok.desc;

    ncclEpCombineConfig_t config = NCCL_EP_COMBINE_CONFIG_INIT;

    NCCL_EP_CHECK(ncclEpCombine(
        ep_handle, &inputs, &outputs, &config, stream));
}

} // namespace c10d::nccl_ep

#else // USE_NCCL_EP

namespace c10d::nccl_ep {

NcclEpGroup::~NcclEpGroup() = default;
NcclEpHandle::~NcclEpHandle() = default;

[[noreturn]] static void not_supported() {
    TORCH_CHECK(false, "PyTorch was not built with USE_NCCL_EP=1");
}

c10::intrusive_ptr<NcclEpGroup> nccl_ep_create_group(
    const c10::intrusive_ptr<::c10d::ProcessGroup>&,
    int64_t, int64_t, int64_t, int64_t) {
    not_supported();
}

c10::intrusive_ptr<NcclEpHandle> nccl_ep_create_handle(
    const c10::intrusive_ptr<NcclEpGroup>&,
    const at::Tensor&,
    const std::optional<at::Tensor>&,
    NcclEpLayout) {
    not_supported();
}

int64_t nccl_ep_handle_get_num_recv_tokens(
    const c10::intrusive_ptr<NcclEpHandle>&) {
    not_supported();
}

void nccl_ep_dispatch(
    const c10::intrusive_ptr<NcclEpHandle>&,
    const at::Tensor&, const at::Tensor&,
    at::Tensor&, std::optional<at::Tensor>, std::optional<at::Tensor>) {
    not_supported();
}

void nccl_ep_combine(
    const c10::intrusive_ptr<NcclEpHandle>&,
    const at::Tensor&,
    at::Tensor&) {
    not_supported();
}

} // namespace c10d::nccl_ep

#endif // USE_NCCL_EP
