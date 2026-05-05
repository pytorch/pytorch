#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryUtils.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/NIXLSymmetricMemory.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <nixl.h>
#include <nixl_descriptors.h>
#include <unistd.h>
#include <chrono>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>

namespace c10d {
namespace symmetric_memory {

static std::mutex g_agent_mutex;
static std::unique_ptr<nixlAgent> g_agent;
static nixlBackendH* g_ucx_backend = nullptr;
static std::string g_agent_name;
// Tracks the last remote NIXL agent name seen for a process-group rank.
// Group scoping avoids invalidating metadata for unrelated subgroups that
// reuse the same rank index.
static std::map<std::pair<std::string, int>, std::string> g_peer_agent_names;

nixlAgent& ensure_nixl_agent() {
  std::lock_guard<std::mutex> lock(g_agent_mutex);
  if (g_agent) return *g_agent;
  char hostname[256] = {};
  gethostname(hostname, sizeof(hostname));
  g_agent_name = std::string("pt_nixl_") + hostname + "_" + std::to_string(getpid());
  nixlAgentConfig cfg;
  cfg.useProgThread = true;
  cfg.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT;
  g_agent = std::make_unique<nixlAgent>(g_agent_name, cfg);
  nixl_b_params_t params;
  auto status = g_agent->createBackend("UCX", params, g_ucx_backend);
  TORCH_CHECK(status == NIXL_SUCCESS, "NIXL: failed to create UCX backend");
  {
    int cur_dev = 0;
    AT_CUDA_CHECK(cudaGetDevice(&cur_dev));
    void* probe = nullptr;
    AT_CUDA_CHECK(cudaMalloc(&probe, 1));
    nixl_reg_dlist_t preg(VRAM_SEG);
    preg.addDesc(nixlBlobDesc(reinterpret_cast<uintptr_t>(probe), 1, uint64_t(cur_dev), ""));
    bool ok = (g_agent->registerMem(preg) == NIXL_SUCCESS);
    if (ok) {
      nixl_xfer_dlist_t px(VRAM_SEG);
      px.addDesc(nixlBasicDesc(reinterpret_cast<uintptr_t>(probe), 1, uint64_t(cur_dev)));
      nixlDlistH* dh = nullptr;
      ok = (g_agent->prepXferDlist(NIXL_INIT_AGENT, px, dh) == NIXL_SUCCESS);
      if (dh) g_agent->releasedDlistH(dh);
      g_agent->deregisterMem(preg);
    }
    AT_CUDA_CHECK(cudaFree(probe));
    TORCH_CHECK(ok, "NIXL: UCX lacks VRAM support. Install ucx-cuda.");
  }
  return *g_agent;
}

const std::string& nixl_agent_name() {
  std::lock_guard<std::mutex> lock(g_agent_mutex);
  TORCH_CHECK(!g_agent_name.empty(), "NIXL agent not initialized");
  return g_agent_name;
}

static StoreExchange storeExchange("NIXLSymmetricMemory");
static std::mutex g_md_exchange_mutex;
static std::unordered_map<std::string, size_t> g_md_exchange_seq;

static std::vector<nixl_blob_t> all_gather_nixl_metadata(
    const c10::intrusive_ptr<c10d::Store>& store, int rank, int ws,
    const std::string& gn, const nixl_blob_t& md) {
  size_t seq;
  { std::lock_guard<std::mutex> lk(g_md_exchange_mutex); seq = g_md_exchange_seq[gn]++; }
  std::vector<std::string> keys;
  for (int r = 0; r < ws; ++r)
    keys.push_back("nixl_md/" + gn + "/" + std::to_string(seq) + "/" + std::to_string(r));
  std::vector<uint8_t> payload(
      reinterpret_cast<const uint8_t*>(md.data()),
      reinterpret_cast<const uint8_t*>(md.data()) + md.size());
  store->set(keys[rank], payload);
  std::vector<nixl_blob_t> res(ws);
  res[rank] = md;
  for (int r = 0; r < ws; ++r) {
    if (r == rank) continue;
    store->wait({keys[r]});
    auto d = store->get(keys[r]);
    res[r] = nixl_blob_t(reinterpret_cast<const char*>(d.data()), d.size());
  }
  storeExchange.barrier(store, rank, ws);
  store->deleteKey(keys[rank]);
  return res;
}

struct NIXLRendezvousInfo {
  uintptr_t buffer_addr, signal_pad_addr;
  size_t buffer_size;
  int device_idx;
};

static size_t nixl_channel_signal_offset(int world_size, int channel, int rank) {
  TORCH_CHECK(channel >= 0, "NIXL: channel must be non-negative, got ", channel);
  TORCH_CHECK(rank >= 0 && rank < world_size, "NIXL: invalid rank ", rank);
  return kNixlChannelSignalOffset +
      (size_t(world_size) * size_t(channel) + size_t(rank)) * sizeof(uint32_t);
}

static void check_nixl_signal_pad_capacity(int world_size, int channel) {
  const auto required =
      nixl_channel_signal_offset(world_size, channel, world_size - 1) +
      sizeof(uint32_t);
  TORCH_CHECK(
      required <= get_signal_pad_size(),
      "NIXL: signal pad is too small for world_size=",
      world_size,
      ", channel=",
      channel,
      ". Required ",
      required,
      " bytes, but signal pad size is ",
      get_signal_pad_size());
}

struct NIXLAllocation {
  void* ptr;
  size_t buffer_size, signal_pad_offset, block_size;
  int device_idx;
  bool registered = false;
  NIXLAllocation(void* p, size_t bs, size_t spo, size_t blk, int d)
      : ptr(p), buffer_size(bs), signal_pad_offset(spo), block_size(blk), device_idx(d) {}
  NIXLAllocation(const NIXLAllocation&) = delete;
  NIXLAllocation& operator=(const NIXLAllocation&) = delete;
  ~NIXLAllocation() {
    if (is_finalizing()) return;
    if (registered) {
      std::lock_guard<std::mutex> lk(g_agent_mutex);
      if (g_agent) {
        nixl_reg_dlist_t d(VRAM_SEG);
        d.addDesc(nixlBlobDesc(reinterpret_cast<uintptr_t>(ptr), block_size, uint64_t(device_idx), ""));
        g_agent->deregisterMem(d);
      }
    }
    c10::cuda::CUDAGuard g(device_idx);
    AT_CUDA_CHECK(cudaFree(ptr));
  }
  void* signal_pad_ptr() const { return static_cast<char*>(ptr) + signal_pad_offset; }
};

class NIXLPeerAllocInfo : public c10::intrusive_ptr_target {
 public:
  NIXLPeerAllocInfo(NIXLAllocation* a, const std::string& gn)
      : group_name_(gn), buffer_size_(a->buffer_size), device_idx_(a->device_idx) {
    c10::cuda::CUDAGuard guard(device_idx_);
    auto group = c10d::resolve_process_group(gn);
    rank_ = group->getRank(); world_size_ = group->getSize();
    check_nixl_signal_pad_capacity(world_size_, 0);
    auto store = group->getStore();
    auto& agent = ensure_nixl_agent();
    if (!a->registered) {
      nixl_reg_dlist_t r(VRAM_SEG);
      r.addDesc(nixlBlobDesc(reinterpret_cast<uintptr_t>(a->ptr), a->block_size, uint64_t(device_idx_), ""));
      TORCH_CHECK(agent.registerMem(r) == NIXL_SUCCESS, "NIXL registerMem failed");
      a->registered = true;
    }
    // Serialize only this buffer's descriptor with connection info.
    // We always include conn info because we invalidate stale remote
    // metadata before reloading (see below).
    nixl_blob_t local_md;
    nixl_reg_dlist_t buf_desc(VRAM_SEG);
    buf_desc.addDesc(nixlBlobDesc(
        reinterpret_cast<uintptr_t>(a->ptr), a->block_size,
        uint64_t(device_idx_), ""));
    nixl_opt_args_t opts;
    opts.includeConnInfo = true;
    TORCH_CHECK(
        agent.getLocalPartialMD(buf_desc, local_md, &opts) == NIXL_SUCCESS,
        "NIXL getLocalPartialMD failed");
    auto mds = all_gather_nixl_metadata(store, rank_, world_size_, gn, local_md);
    NIXLRendezvousInfo li{reinterpret_cast<uintptr_t>(a->ptr),
        reinterpret_cast<uintptr_t>(a->signal_pad_ptr()), a->buffer_size, a->device_idx};
    auto pis = storeExchange.all_gather(store, rank_, world_size_, li);
    for (int r = 0; r < world_size_; ++r) {
      peer_infos_.push_back(pis[r]);
      if (r == rank_) {
        buffers_.push_back(a->ptr); signal_pads_.push_back(a->signal_pad_ptr());
        peer_agent_names_.push_back(nixl_agent_name()); continue;
      }
      // Try loading remote metadata. If it fails (stale descriptors
      // or connection from a prior rendezvous), invalidate that
      // agent's cached state and retry with fresh metadata.
      std::string rn;
      auto md_st = agent.loadRemoteMD(mds[r], rn);
      if (md_st != NIXL_SUCCESS) {
        std::lock_guard<std::mutex> lk(g_agent_mutex);
        auto it = g_peer_agent_names.find({group_name_, r});
        if (it != g_peer_agent_names.end()) {
          agent.invalidateRemoteMD(it->second);
          g_peer_agent_names.erase(it);
        }
        md_st = agent.loadRemoteMD(mds[r], rn);
      }
      TORCH_CHECK(md_st == NIXL_SUCCESS,
          "NIXL loadRemoteMD failed for rank ", r,
          ", status=", static_cast<int>(md_st));
      // Proactively establish UCX connection so that later transfers
      // don't need to do the handshake inline (which requires both
      // sides to progress UCX simultaneously).
      auto conn_st = agent.makeConnection(rn);
      TORCH_CHECK(
          conn_st == NIXL_SUCCESS,
          "NIXL makeConnection failed for rank ",
          r,
          ", status=",
          static_cast<int>(conn_st));
      {
        std::lock_guard<std::mutex> lk(g_agent_mutex);
        g_peer_agent_names[{group_name_, r}] = rn;
      }
      peer_agent_names_.push_back(rn);
      buffers_.push_back(nullptr); signal_pads_.push_back(nullptr);
    }
    size_t as = sizeof(void*) * world_size_;
    buffers_dev_ = (void**)c10::cuda::CUDACachingAllocator::raw_alloc(as);
    signal_pads_dev_ = (void**)c10::cuda::CUDACachingAllocator::raw_alloc(as);
    AT_CUDA_CHECK(cudaMemcpy(buffers_dev_, buffers_.data(), as, cudaMemcpyHostToDevice));
    AT_CUDA_CHECK(cudaMemcpy(signal_pads_dev_, signal_pads_.data(), as, cudaMemcpyHostToDevice));
    AT_CUDA_CHECK(cudaMalloc(&signal_staging_, kNixlSignalStagingBytes));
    AT_CUDA_CHECK(cudaMemset(signal_staging_, 0, kNixlSignalStagingBytes));
    uint32_t one = 1;
    AT_CUDA_CHECK(cudaMemcpy(
        static_cast<char*>(signal_staging_) + kNixlChannelSignalOffset,
        &one,
        sizeof(one),
        cudaMemcpyHostToDevice));
    nixl_reg_dlist_t sr(VRAM_SEG);
    sr.addDesc(nixlBlobDesc(
        reinterpret_cast<uintptr_t>(signal_staging_),
        kNixlSignalStagingBytes,
        uint64_t(device_idx_),
        ""));
    TORCH_CHECK(agent.registerMem(sr) == NIXL_SUCCESS, "staging reg failed");
    staging_reg_ = true;
  }
  ~NIXLPeerAllocInfo() override {
    if (is_finalizing()) return;
    if (staging_reg_ && signal_staging_) {
      std::lock_guard<std::mutex> lk(g_agent_mutex);
      if (g_agent) {
        nixl_reg_dlist_t d(VRAM_SEG);
        d.addDesc(nixlBlobDesc(
            reinterpret_cast<uintptr_t>(signal_staging_),
            kNixlSignalStagingBytes,
            uint64_t(device_idx_),
            ""));
        g_agent->deregisterMem(d);
      }
    }
    if (signal_staging_) {
      c10::cuda::CUDAGuard g(device_idx_);
      AT_CUDA_CHECK(cudaFree(signal_staging_));
    }
    if (buffers_dev_) c10::cuda::CUDACachingAllocator::raw_delete(buffers_dev_);
    if (signal_pads_dev_) c10::cuda::CUDACachingAllocator::raw_delete(signal_pads_dev_);
  }
 private:
  std::string group_name_;
  size_t buffer_size_; int rank_, world_size_, device_idx_;
  std::vector<void*> buffers_, signal_pads_;
  void** buffers_dev_ = nullptr; void** signal_pads_dev_ = nullptr;
  std::vector<std::string> peer_agent_names_;
  std::vector<NIXLRendezvousInfo> peer_infos_;
  void* signal_staging_ = nullptr; bool staging_reg_ = false;
  friend class NIXLSymmetricMemory;
};

class NIXLXferRequest {
 public:
  explicit NIXLXferRequest(nixlAgent& agent) : agent_(agent) {}
  NIXLXferRequest(const NIXLXferRequest&) = delete;
  NIXLXferRequest& operator=(const NIXLXferRequest&) = delete;
  ~NIXLXferRequest() {
    if (req_) {
      agent_.releaseXferReq(req_);
    }
  }

  nixlXferReqH*& out() {
    return req_;
  }

  nixlXferReqH* get() const {
    return req_;
  }

 private:
  nixlAgent& agent_;
  nixlXferReqH* req_ = nullptr;
};

void nixl_transfer(nixl_xfer_op_t op,
    uintptr_t la, size_t ls, uint64_t ld,
    uintptr_t ra, size_t rs, uint64_t rd, const std::string& rn) {
  if (ls == 0 && rs == 0) {
    return;
  }
  TORCH_CHECK(ls == rs, "NIXL transfer size mismatch: local=", ls, ", remote=", rs);
  TORCH_CHECK(!rn.empty(), "NIXL transfer requires a remote agent name");
  auto& agent = ensure_nixl_agent();
  nixl_xfer_dlist_t ll(VRAM_SEG); ll.addDesc(nixlBasicDesc(la, ls, ld));
  nixl_xfer_dlist_t rl(VRAM_SEG); rl.addDesc(nixlBasicDesc(ra, rs, rd));
  NIXLXferRequest req(agent);
  auto s = agent.createXferReq(op, ll, rl, rn, req.out());
  TORCH_CHECK(s == NIXL_SUCCESS, "NIXL createXferReq failed, status=", static_cast<int>(s));
  s = agent.postXferReq(req.get());
  TORCH_CHECK(s == NIXL_SUCCESS || s == NIXL_IN_PROG, "NIXL postXferReq failed, status=", static_cast<int>(s));
  auto deadline = std::chrono::steady_clock::now() +
      std::chrono::seconds(kNixlTransferTimeoutSeconds);
  while (s == NIXL_IN_PROG) {
    TORCH_CHECK(std::chrono::steady_clock::now() < deadline,
        "NIXL transfer timed out after ",
        kNixlTransferTimeoutSeconds,
        "s");
    std::this_thread::yield();
    s = agent.getXferStatus(req.get());
  }
  TORCH_CHECK(s == NIXL_SUCCESS, "NIXL transfer failed, status=", static_cast<int>(s));
}

NIXLSymmetricMemory::NIXLSymmetricMemory(c10::intrusive_ptr<NIXLPeerAllocInfo> p, size_t o)
    : device_idx_(p->device_idx_), pai_(std::move(p)), offset_(o) {}
NIXLSymmetricMemory::NIXLSymmetricMemory(const NIXLSymmetricMemory& o, size_t off)
    : device_idx_(o.device_idx_), pai_(o.pai_), offset_(off) {}
std::vector<void*> NIXLSymmetricMemory::get_buffer_ptrs() { return pai_->buffers_; }
std::vector<void*> NIXLSymmetricMemory::get_signal_pad_ptrs() { return pai_->signal_pads_; }
void** NIXLSymmetricMemory::get_buffer_ptrs_dev() { return pai_->buffers_dev_; }
void** NIXLSymmetricMemory::get_signal_pad_ptrs_dev() { return pai_->signal_pads_dev_; }
size_t NIXLSymmetricMemory::get_buffer_size() { return pai_->buffer_size_; }
size_t NIXLSymmetricMemory::get_offset() { return offset_; }
bool NIXLSymmetricMemory::has_multicast_support() { return false; }
void* NIXLSymmetricMemory::get_multicast_ptr() { return nullptr; }

void NIXLSymmetricMemory::put_signal(int dst, int ch, size_t) {
  TORCH_CHECK(dst >= 0 && dst < pai_->world_size_ && dst != pai_->rank_,
      "put_signal: invalid dst_rank ", dst);
  check_nixl_signal_pad_capacity(pai_->world_size_, ch);
  c10::cuda::CUDAGuard g(device_idx_);
  AT_CUDA_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream()));
  size_t off = nixl_channel_signal_offset(pai_->world_size_, ch, pai_->rank_);
  nixl_transfer(NIXL_WRITE,
      reinterpret_cast<uintptr_t>(pai_->signal_staging_) + kNixlChannelSignalOffset, sizeof(uint32_t), uint64_t(device_idx_),
      pai_->peer_infos_[dst].signal_pad_addr + off, sizeof(uint32_t), uint64_t(pai_->peer_infos_[dst].device_idx),
      pai_->peer_agent_names_[dst]);
}
void NIXLSymmetricMemory::wait_signal(int src, int ch, size_t tmo) {
  TORCH_CHECK(src >= 0 && src < pai_->world_size_ && src != pai_->rank_,
      "wait_signal: invalid src_rank ", src);
  check_nixl_signal_pad_capacity(pai_->world_size_, ch);
  nixl_launch_wait_signal_kernel(pai_->signal_pads_dev_, src, ch, pai_->rank_, pai_->world_size_, tmo, device_idx_);
}
void NIXLSymmetricMemory::barrier(int ch, size_t tmo) {
  for (int r = 0; r < pai_->world_size_; ++r) if (r != pai_->rank_) put_signal(r, ch, tmo);
  for (int r = 0; r < pai_->world_size_; ++r) if (r != pai_->rank_) wait_signal(r, ch, tmo);
}
int NIXLSymmetricMemory::get_rank() { return pai_->rank_; }
int NIXLSymmetricMemory::get_world_size() { return pai_->world_size_; }
c10::Device NIXLSymmetricMemory::get_device() { return c10::Device(c10::DeviceType::CUDA, device_idx_); }
bool NIXLSymmetricMemory::world_within_direct_access() { return false; }
const std::string& NIXLSymmetricMemory::get_peer_agent_name(int r) const { return pai_->peer_agent_names_[r]; }
uintptr_t NIXLSymmetricMemory::get_peer_buffer_addr(int r) const { return pai_->peer_infos_[r].buffer_addr; }
uintptr_t NIXLSymmetricMemory::get_peer_signal_pad_addr(int r) const { return pai_->peer_infos_[r].signal_pad_addr; }
int NIXLSymmetricMemory::get_peer_device_idx(int r) const { return pai_->peer_infos_[r].device_idx; }
int NIXLSymmetricMemory::get_local_device_idx() const { return device_idx_; }
void* NIXLSymmetricMemory::get_signal_staging_ptr() const { return pai_->signal_staging_; }

class NIXLSymmetricMemoryAllocator : public SymmetricMemoryAllocator {
 public:
  void* alloc(size_t sz, int dev, const std::optional<std::string>&) override {
    c10::cuda::CUDAGuard g(dev);
    size_t sp = get_signal_pad_size(), bs = sz + sp;
    void* p = nullptr; AT_CUDA_CHECK(cudaMalloc(&p, bs));
    AT_CUDA_CHECK(cudaMemset(static_cast<char*>(p) + sz, 0, sp));
    std::lock_guard<std::mutex> lk(mu_);
    allocs_.try_emplace(p, std::make_unique<NIXLAllocation>(p, sz, sz, bs, dev));
    return p;
  }
  void free(void* p) override {
    std::lock_guard<std::mutex> lk(mu_);
    allocs_.erase(p);
    // Evict stale rendezvous cache entries for this pointer so that if
    // cudaMalloc reuses the same address, we don't return a stale handle.
    for (auto it = sms_.begin(); it != sms_.end();) {
      if (std::get<0>(it->first) == p)
        it = sms_.erase(it);
      else
        ++it;
    }
  }
  size_t get_alloc_size(void* p) override {
    std::lock_guard<std::mutex> lk(mu_);
    auto i = allocs_.find(p);
    TORCH_CHECK(i != allocs_.end(), p, " is not allocated by NIXLSymmetricMemoryAllocator");
    return i->second->buffer_size;
  }
  c10::intrusive_ptr<SymmetricMemory> rendezvous(void* p, const std::optional<std::string>& gn) override {
    TORCH_CHECK(gn.has_value());
    std::lock_guard<std::mutex> lk(mu_);
    { auto i = sms_.find({p, *gn}); if (i != sms_.end()) return i->second; }
    auto ai = std::find_if(allocs_.begin(), allocs_.end(), [&](auto& kv) {
      auto pi = uintptr_t(p), bi = uintptr_t(kv.second->ptr);
      return pi >= bi && pi < bi + kv.second->buffer_size; });
    TORCH_CHECK(ai != allocs_.end());
    auto& a = ai->second;
    auto bi = sms_.find({a->ptr, *gn});
    c10::intrusive_ptr<NIXLSymmetricMemory> s;
    if (bi != sms_.end()) s = bi->second;
    else { s = c10::make_intrusive<NIXLSymmetricMemory>(
        c10::make_intrusive<NIXLPeerAllocInfo>(a.get(), *gn), 0);
      sms_[{a->ptr, *gn}] = s; }
    if (p == a->ptr) return s;
    return c10::make_intrusive<NIXLSymmetricMemory>(*s, uintptr_t(p) - uintptr_t(a->ptr));
  }
  bool has_multicast_support(int) override { return false; }
  c10::DeviceType supported_device_type() override { return c10::DeviceType::CUDA; }
  std::string name() override { return "NIXL"; }
 private:
  std::mutex mu_;
  std::unordered_map<void*, std::unique_ptr<NIXLAllocation>> allocs_;
  std::map<std::tuple<void*,std::string>, c10::intrusive_ptr<NIXLSymmetricMemory>> sms_;
};

struct RegNIXL { RegNIXL() {
  auto a = c10::make_intrusive<NIXLSymmetricMemoryAllocator>();
  if (getSymmMemBackendCUDA() == "NIXL") register_allocator(c10::DeviceType::CUDA, a);
  else register_availability("NIXL", a);
}};
static RegNIXL reg_;

} // namespace symmetric_memory
} // namespace c10d
