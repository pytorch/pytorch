#ifdef USE_C10D_TCCL

#include <torch/csrc/distributed/c10d/ProcessGroupTCCL.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupTCCLDetail.hpp>
#include <torch/csrc/distributed/c10d/TCCLUtils.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>

#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/util/Exception.h>
#include <c10/util/thread_name.h>

#include <ATen/Context.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/core/jit_type.h>
#include <ATen/detail/MPSHooksInterface.h>
#include <ATen/mps/MPSAllocatorInterface.h>

#include <fmt/format.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace c10d {

namespace {

// MPS staging helpers.
//
// Run the collective on a worker thread while the GPU keeps computing: main
// thread records a non-blocking MPS event (mpsEventRecord), worker blocks on it
// (mpsEventWait) - neither drains nor blocks the GPU - then reads/writes the
// tensor via a zero-copy CPU view (mpsSharedCpuView) over unified memory.

// Zero-copy CPU view over an MPS tensor's unified memory: getSharedBufferPtr
// gives the shared CPU pointer, from_blob wraps it. Uses the storage base
// pointer, NOT data_ptr() - on MPS data_ptr() folds storage_offset into
// MTLBuffer arithmetic, unmappable for offset views (e.g.
// all_gather_into_tensor).
at::Tensor mpsSharedCpuView(const at::Tensor& mpsTensor) {
  // A zero-element MPS tensor has no buffer at all; there is nothing to view.
  if (mpsTensor.numel() == 0) {
    return at::empty(
        mpsTensor.sizes(),
        at::TensorOptions().dtype(mpsTensor.dtype()).device(at::kCPU));
  }
  auto* allocator = at::mps::getIMPSAllocator();
  const void* storage_ptr = mpsTensor.storage().data_ptr().get();
  // Pair is {cpu_ptr, retain_count}; only the pointer is used here.
  const void* cpu_ptr = allocator->getSharedBufferPtr(storage_ptr).first;
  TORCH_INTERNAL_ASSERT(
      cpu_ptr != nullptr,
      fmt::format(
          "MPS tensor does not have a shared (unified memory) backing buffer: "
          "storage_ptr={} data_ptr={} size={} storage_offset={} device={}",
          storage_ptr,
          mpsTensor.data_ptr(),
          mpsTensor.nbytes(),
          mpsTensor.storage_offset(),
          mpsTensor.device().str()));
  void* offset_ptr = static_cast<char*>(const_cast<void*>(cpu_ptr)) +
      mpsTensor.storage_offset() * mpsTensor.itemsize();
  return at::from_blob(
      offset_ptr,
      mpsTensor.sizes(),
      mpsTensor.strides(),
      at::TensorOptions().dtype(mpsTensor.dtype()).device(at::kCPU));
}

// Non-blocking: encode a signal via commitAndContinue, return an event id.
// Does NOT drain the GPU or hold the serial queue beyond the encode.
uint32_t mpsEventRecord() {
  auto& hooks = at::detail::getMPSHooks();
  uint32_t eid = hooks.acquireEvent(false);
  // Main has several MPS streams; signal on the caller's current one, where
  // the work this collective waits for was encoded.
  const auto stream = c10::impl::getDeviceGuardImpl(c10::DeviceType::MPS)
                          ->getStream(c10::Device(c10::DeviceType::MPS, 0));
  hooks.recordEvent(eid, stream.id());
  return eid;
}

// Block the calling thread until the GPU reaches the recorded signal, then
// release the event. Returns immediately if the GPU already passed it.
void mpsEventWait(uint32_t eid) {
  auto& hooks = at::detail::getMPSHooks();
  hooks.synchronizeEvent(eid);
  hooks.releaseEvent(eid);
}

// The dtypes TCCL reduces; isSupportedReduceDtype checks the same set.
#define TCCL_REDUCE_DTYPES \
  AT_EXPAND(AT_INTEGRAL_TYPES), at::kFloat, at::kHalf, at::kBFloat16, at::kBool

inline bool isSupportedReduceDtype(at::ScalarType st) {
  return st == at::kFloat || st == at::kHalf || st == at::kBFloat16 ||
      st == at::kChar || st == at::kShort || st == at::kInt ||
      st == at::kLong || st == at::kByte || st == at::kBool;
}

// AVG (= SUM then x 1/world_size)
// AVG is gated to the float dtypes
inline bool isFloatReduceDtype(at::ScalarType st) {
  return st == at::kFloat || st == at::kHalf || st == at::kBFloat16;
}

// Argument checks shared by the collectives, raising DistBackendError.
// invalidArgument adapts the Utils.hpp asserts, as Gloo does; all_to_all's
// split checks are c10d's own (checkSplitSizes), with their error type.
std::function<void(const std::string&)> invalidArgument(const char* op) {
  return [op](const std::string& msg) {
    TORCH_CHECK_WITH(DistBackendError, false, "TCCL ", op, ": ", msg);
  };
}

void checkMpsTensor(const at::Tensor& t, const char* op) {
  TORCH_CHECK_WITH(
      DistBackendError,
      t.device().is_mps() && t.is_contiguous(),
      fmt::format(
          "TCCL {}: expected a contiguous MPS tensor, got a {} tensor on {}. "
          "Call .to(\"mps\").contiguous() first.",
          op,
          t.is_contiguous() ? "contiguous" : "non-contiguous",
          t.device().str()));
}

// A tensor-list entry: a contiguous MPS tensor with the dtype and numel of
// `like`.
void checkMatches(const at::Tensor& t, const at::Tensor& like, const char* op) {
  checkMpsTensor(t, op);
  TORCH_CHECK_WITH(
      DistBackendError,
      t.scalar_type() == like.scalar_type() && t.numel() == like.numel(),
      fmt::format(
          "TCCL {}: every list entry must have dtype {} and numel {}, got {} "
          "and {}.",
          op,
          c10::toString(like.scalar_type()),
          like.numel(),
          c10::toString(t.scalar_type()),
          t.numel()));
}

void checkReduce(const char* op, at::ScalarType st, const ReduceOp& reduceOp) {
  TORCH_CHECK_WITH(
      DistBackendError,
      reduceOp == ReduceOp::SUM || reduceOp == ReduceOp::AVG ||
          reduceOp == ReduceOp::MIN || reduceOp == ReduceOp::MAX ||
          reduceOp == ReduceOp::PRODUCT,
      "TCCL ",
      op,
      ": only SUM, AVG, MIN, MAX and PRODUCT are supported.");
  TORCH_CHECK_WITH(
      DistBackendError,
      isSupportedReduceDtype(st),
      "TCCL ",
      op,
      ": dtype ",
      st,
      " cannot be reduced; supported: float32, float16, bfloat16, int8, "
      "int16, int32, int64, uint8 and bool.");
  TORCH_CHECK_WITH(
      DistBackendError,
      reduceOp != ReduceOp::AVG || isFloatReduceDtype(st),
      "TCCL ",
      op,
      ": AVG needs float32, float16 or bfloat16 (integer AVG is undefined, as "
      "in NCCL), got ",
      st,
      ". Use SUM for integer dtypes.");
}

// Ring-vs-mesh selection for all_reduce (N>2): bf16 -> ring at any size,
// fp32/fp16 -> ring only >=8MB. TCCL_FORCE_ALGO=ring|mesh overrides per-call.
inline bool tcclUseRing(
    int worldSize,
    bool autoEnable,
    bool isBf16,
    std::size_t count,
    std::size_t elemBytes,
    bool forceRing = false) {
  if (worldSize <= 2) {
    return false;
  }
  if (forceRing) {
    return true;
  }
  // Auto-heuristic: bf16 any size, else >=1MB
  bool useRing =
      autoEnable && (isBf16 || count * elemBytes >= 1u * 1024u * 1024u);
  if (const char* f = std::getenv("TCCL_FORCE_ALGO")) {
    const std::string forced(f);
    if (forced == "ring") {
      useRing = true;
    } else if (forced == "mesh") {
      useRing = false;
    }
  }
  return useRing;
}

// Run an allreduce with a specific reduce-op functor, picking ring vs mesh.
template <typename T, typename Op>
void tcclAllreduceWith(
    TCCLEngine& mesh,
    T* data,
    std::size_t count,
    int worldSize,
    bool useRing) {
  if (useRing) {
    mesh.ring_all_reduce<T, Op>(data, count, Op{});
  } else {
    mesh.all_reduce<T, Op>(data, count, Op{});
  }
}

// Typed mesh allreduce. `op` selects the reduce op (SUM/MIN/MAX/AVG)
template <typename T>
void runMeshAllreduce(
    TCCLEngine& mesh,
    at::Tensor& view,
    ReduceOp::RedOpType op,
    int worldSize) {
  T* data = view.data_ptr<T>();
  const std::size_t count = static_cast<std::size_t>(view.numel());
  const bool useRing = tcclUseRing(
      worldSize,
      /*autoEnable=*/true,
      std::is_same_v<T, at::BFloat16>,
      count,
      sizeof(T),
      mesh.ringTopology());
  switch (op) {
    case ReduceOp::MIN:
      tcclAllreduceWith<T, TCCLMinOp<T>>(mesh, data, count, worldSize, useRing);
      break;
    case ReduceOp::MAX:
      tcclAllreduceWith<T, TCCLMaxOp<T>>(mesh, data, count, worldSize, useRing);
      break;
    case ReduceOp::PRODUCT:
      tcclAllreduceWith<T, TCCLProdOp<T>>(
          mesh, data, count, worldSize, useRing);
      break;
    // SUM or AVG
    default:
      tcclAllreduceWith<T, TCCLSumOp<T>>(mesh, data, count, worldSize, useRing);
      break;
  }
  if (op == ReduceOp::AVG) {
    const float inv = 1.0f / static_cast<float>(worldSize);
    for (std::size_t i = 0; i < count; ++i) {
      data[i] = static_cast<T>(static_cast<float>(data[i]) * inv);
    }
  }
}

// Typed reduce-scatter core: in_chunks[p] is this rank's contribution for peer
// p (count_per_rank elements). Picks ring vs mesh, applies the reduce functor,
// writes count_per_rank elements to `out`.
template <typename T>
void runReduceScatterChunks(
    TCCLEngine& mesh,
    const std::vector<const T*>& in_chunks,
    T* out,
    std::size_t count_per_rank,
    ReduceOp::RedOpType op,
    int worldSize) {
  const bool useRing = tcclUseRing(
      worldSize,
      /*autoEnable=*/false,
      std::is_same_v<T, at::BFloat16>,
      count_per_rank * static_cast<std::size_t>(worldSize),
      sizeof(T),
      mesh.ringTopology());
  switch (op) {
    case ReduceOp::MIN:
      if (useRing)
        mesh.ring_reduce_scatter<T, TCCLMinOp<T>>(
            in_chunks, out, count_per_rank, TCCLMinOp<T>{});
      else
        mesh.reduce_scatter<T, TCCLMinOp<T>>(
            in_chunks, out, count_per_rank, TCCLMinOp<T>{});
      break;
    case ReduceOp::MAX:
      if (useRing)
        mesh.ring_reduce_scatter<T, TCCLMaxOp<T>>(
            in_chunks, out, count_per_rank, TCCLMaxOp<T>{});
      else
        mesh.reduce_scatter<T, TCCLMaxOp<T>>(
            in_chunks, out, count_per_rank, TCCLMaxOp<T>{});
      break;
    case ReduceOp::PRODUCT:
      if (useRing)
        mesh.ring_reduce_scatter<T, TCCLProdOp<T>>(
            in_chunks, out, count_per_rank, TCCLProdOp<T>{});
      else
        mesh.reduce_scatter<T, TCCLProdOp<T>>(
            in_chunks, out, count_per_rank, TCCLProdOp<T>{});
      break;
    // SUM or AVG
    default:
      if (useRing)
        mesh.ring_reduce_scatter<T, TCCLSumOp<T>>(
            in_chunks, out, count_per_rank, TCCLSumOp<T>{});
      else
        mesh.reduce_scatter<T, TCCLSumOp<T>>(
            in_chunks, out, count_per_rank, TCCLSumOp<T>{});
      break;
  }
  if (op == ReduceOp::AVG) {
    const float inv = 1.0f / static_cast<float>(worldSize);
    for (std::size_t k = 0; k < count_per_rank; ++k) {
      out[k] = static_cast<T>(static_cast<float>(out[k]) * inv);
    }
  }
}

// Contiguous reduce-scatter: `inView` holds worldSize contiguous chunks of
// count_per_rank elements (chunk p destined for peer p). Builds the per-rank
// chunk pointers and runs the core.
template <typename T>
void runMeshReduceScatter(
    TCCLEngine& mesh,
    at::Tensor& inView,
    at::Tensor& outView,
    ReduceOp::RedOpType op,
    int worldSize) {
  const std::size_t count_per_rank = static_cast<std::size_t>(outView.numel());
  const T* in = inView.data_ptr<T>();
  std::vector<const T*> in_chunks(static_cast<std::size_t>(worldSize));
  for (int r = 0; r < worldSize; ++r) {
    in_chunks[r] = in + static_cast<std::size_t>(r) * count_per_rank;
  }
  runReduceScatterChunks<T>(
      mesh, in_chunks, outView.data_ptr<T>(), count_per_rank, op, worldSize);
}

// List-form reduce-scatter: inChunkViews[p] is this rank's CPU view of the
// contribution destined for peer p.
template <typename T>
void runMeshReduceScatterList(
    TCCLEngine& mesh,
    std::vector<at::Tensor>& inChunkViews,
    at::Tensor& outView,
    ReduceOp::RedOpType op,
    int worldSize) {
  const std::size_t count_per_rank = static_cast<std::size_t>(outView.numel());
  std::vector<const T*> in_chunks(static_cast<std::size_t>(worldSize));
  for (int r = 0; r < worldSize; ++r) {
    in_chunks[r] = inChunkViews[r].data_ptr<T>();
  }
  runReduceScatterChunks<T>(
      mesh, in_chunks, outView.data_ptr<T>(), count_per_rank, op, worldSize);
}

// Runtime-dtype dispatch for contiguous reduce-scatter
void dispatchReduceScatter(
    TCCLEngine& mesh,
    at::Tensor& inView,
    at::Tensor& outView,
    ReduceOp::RedOpType op,
    int worldSize) {
  AT_DISPATCH_V2(
      inView.scalar_type(),
      "tccl_reduce_scatter",
      AT_WRAP([&] {
        runMeshReduceScatter<scalar_t>(mesh, inView, outView, op, worldSize);
      }),
      TCCL_REDUCE_DTYPES);
}

// Runtime-dtype dispatch for list-form reduce-scatter (worldSize separate input
// views, one per destination rank).
void dispatchReduceScatterList(
    TCCLEngine& mesh,
    std::vector<at::Tensor>& inChunkViews,
    at::Tensor& outView,
    ReduceOp::RedOpType op,
    int worldSize) {
  AT_DISPATCH_V2(
      outView.scalar_type(),
      "tccl_reduce_scatter_list",
      AT_WRAP([&] {
        runMeshReduceScatterList<scalar_t>(
            mesh, inChunkViews, outView, op, worldSize);
      }),
      TCCL_REDUCE_DTYPES);
}
} // namespace

// Options

ProcessGroupTCCL::Options::Options(std::chrono::milliseconds timeout)
    : Backend::Options(TCCL_BACKEND_NAME, timeout) {}

// TCCLWork

ProcessGroupTCCL::TCCLWork::TCCLWork(
    OpType opType,
    uint64_t seq,
    std::vector<at::Tensor> outputs)
    : Work(/*rank=*/-1, opType), seq_(seq), outputs_(std::move(outputs)) {
  // No device list: nothing to order (MPS single-queue, collective completes on
  // the CPU worker before the Future fires). DDP chaining unaffected. A device
  // list cost ~140 us/op for zero synchronization.
  future_ = c10::make_intrusive<c10::ivalue::Future>(
      c10::ListType::create(c10::TensorType::get()));
}

// ProcessGroupTCCL

ProcessGroupTCCL::ProcessGroupTCCL(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size,
    c10::intrusive_ptr<Options> options)
    : Backend(rank, size), store_(store), options_(std::move(options)) {
  TORCH_CHECK_WITH(
      DistBackendError,
      store_ != nullptr,
      "ProcessGroupTCCL: store must not be null.");
  TORCH_CHECK_WITH(
      DistBackendError,
      size > 0 && rank >= 0 && rank < size,
      "ProcessGroupTCCL: invalid (rank=",
      rank,
      ", size=",
      size,
      ").");

  // 0. Resolve topology: Options field, overridden by env
  // TCCL_TOPOLOGY=ring|mesh.
  topology_ = options_->topology;
  if (const char* t = std::getenv("TCCL_TOPOLOGY")) {
    const std::string ts(t);
    if (ts == "ring") {
      topology_ = Topology::Ring;
    } else if (ts == "mesh") {
      topology_ = Topology::Mesh;
    } else {
      TORCH_CHECK_WITH(
          DistBackendError,
          false,
          "ProcessGroupTCCL: TCCL_TOPOLOGY must be 'ring' or 'mesh', got '",
          ts,
          "'.");
    }
  }
  const bool ringTopology = (topology_ == Topology::Ring);
  TORCH_CHECK_WITH(
      DistBackendError,
      !ringTopology || size > 2,
      "ProcessGroupTCCL: ring topology requires world size > 2 (got ",
      size,
      "); use mesh for <=2 ranks.");

  // 1. Resolve the per-peer RDMA device list.
  const std::vector<std::string> peerDevices =
      resolveTcclPeerDevices(rank, size, options_->device_name, ringTopology);

  // 2. num_wires sanity
  const int num_wires = options_->num_wires;
  TORCH_CHECK_WITH(
      DistBackendError,
      num_wires >= 1,
      "ProcessGroupTCCL: num_wires must be >= 1, got ",
      num_wires);

  // 3. Per-device validation.
  std::unordered_map<std::string, int> peersPerDevice;
  for (int peer = 0; peer < size; peer++) {
    if (peer != rank && !peerDevices[peer].empty()) {
      peersPerDevice[peerDevices[peer]]++;
    }
  }
  for (const auto& [dev, peers] : peersPerDevice) {
    checkLinkLayer(dev);
    const int qps = num_wires * peers;
    TORCH_CHECK_WITH(
        DistBackendError,
        qps <= 10,
        fmt::format(
            "ProcessGroupTCCL: device '{}' would host num_wires={} * {} "
            "peer(s) = {} QPs, exceeding the 10-UC-QP-per-device Thunderbolt "
            "limit (TN3205 sec. 12.1). Spread peers across more devices (full "
            "mesh) or reduce num_wires.",
            dev,
            num_wires,
            peers,
            qps));
  }

  // 4. Reserve a unique init-sequence number from the Store. The keys we
  //    use for destination exchange are prefixed with this number.
  const int64_t initSeq = tcclInitSequence(*store_, rank, options_->timeout);
  const std::string keyPrefix = "tccl_dest_s" + std::to_string(initSeq) + "_";

  // 5. Construct one TCCLConnection per (peer, wire). Each constructor opens
  // its
  //    own device context, allocates PD/CQ/QP, transitions the QP to INIT,
  //    and populates its localDestination.
  connections_.resize(static_cast<size_t>(size) * num_wires);
  std::vector<TCCLDestination> localDests(connections_.size());
  for (int peer = 0; peer < size; peer++) {
    if (peer == rank || peerDevices[peer].empty()) {
      // Self, or a non-neighbor in ring topology - no connection
      continue;
    }
    for (int wire = 0; wire < num_wires; wire++) {
      const size_t idx = static_cast<size_t>(peer) * num_wires + wire;
      connections_[idx] = std::make_unique<TCCLConnection>(peerDevices[peer]);
      localDests[idx] = connections_[idx]->localDestination();
    }
  }

  // 6. Allocate + register per-peer send/recv buffers
  TORCH_CHECK_WITH(
      DistBackendError,
      num_wires == 1,
      "ProcessGroupTCCL: num_wires must be 1 (multi-wire is not yet "
      "supported). Got num_wires=",
      num_wires);

  sendBuffers_.resize(static_cast<size_t>(size));
  recvBuffers_.resize(static_cast<size_t>(size));
  pipeSendBuffers_.resize(
      static_cast<size_t>(size) * TCCLEngine::kPipelineDepth);
  pipeRecvBuffers_.resize(
      static_cast<size_t>(size) * TCCLEngine::kPipelineDepth);
  for (int peer = 0; peer < size; peer++) {
    if (peer == rank || peerDevices[peer].empty()) {
      // Self, or a non-neighbor in ring topology
      continue;
    }
    sendBuffers_[peer] = TCCLSharedBuffer(TCCLEngine::kChunkSize);
    recvBuffers_[peer] = TCCLSharedBuffer(TCCLEngine::kChunkSize);
    auto* pd = connections_[peer]->protectionDomain();
    sendBuffers_[peer].registerToPD(pd);
    recvBuffers_[peer].registerToPD(pd);
    // Depth-indexed staging pool for the pipelined bidirectional ring.
    for (int slot = 0; slot < TCCLEngine::kPipelineDepth; slot++) {
      const size_t idx =
          static_cast<size_t>(peer) * TCCLEngine::kPipelineDepth + slot;
      pipeSendBuffers_[idx] = TCCLSharedBuffer(TCCLEngine::kChunkSize);
      pipeRecvBuffers_[idx] = TCCLSharedBuffer(TCCLEngine::kChunkSize);
      pipeSendBuffers_[idx].registerToPD(pd);
      pipeRecvBuffers_[idx].registerToPD(pd);
    }
  }

  // 7. All-to-all destination exchange via Store. Each rank publishes its
  //    full (size * num_wires) destination list.
  std::vector<std::vector<TCCLDestination>> remoteDests;
  allgatherDestinationsViaStore(
      *store_,
      rank,
      size,
      localDests,
      remoteDests,
      keyPrefix,
      options_->timeout);

  // 8. Transition each connection INIT -> RTR -> RTS. To find peer p's
  //    destination FOR ME on wire w, we read p's published list at the
  //    slot p.
  for (int peer = 0; peer < size; peer++) {
    if (peer == rank || peerDevices[peer].empty()) {
      // Self, or a non-neighbor in ring topology
      continue;
    }
    for (int wire = 0; wire < num_wires; wire++) {
      const size_t myConn = static_cast<size_t>(peer) * num_wires + wire;
      const size_t peerSlotForMe = static_cast<size_t>(rank) * num_wires + wire;
      const TCCLDestination& remote = remoteDests[peer][peerSlotForMe];
      connections_[myConn]->transitionToRTR(remote);
      connections_[myConn]->transitionToRTS();
    }
  }

  // 9. Final sync - every rank reaches RTS before any rank may post a send.
  tcclRtsBarrier(*store_, size, keyPrefix + "rts_ready", options_->timeout);

  // 10. Construct the collective engine (holds refs into our connection +
  // buffer vectors).
  engine_ = std::make_unique<TCCLEngine>(
      rank,
      size,
      connections_,
      sendBuffers_,
      recvBuffers_,
      pipeSendBuffers_,
      pipeRecvBuffers_,
      options_->timeout,
      /*ring_topology=*/ringTopology);

  // 11. Spawn the worker thread.
  workerThread_ = std::thread(&ProcessGroupTCCL::runLoop, this);
}

ProcessGroupTCCL::~ProcessGroupTCCL() {
  stop_.store(true);
  workCV_.notify_all();
  if (workerThread_.joinable()) {
    workerThread_.join();
  }
}

// Worker thread

void ProcessGroupTCCL::runLoop() {
  c10::setThreadName("pt_tccl_runloop");
  std::unique_lock<std::mutex> lock(workMutex_);
  while (!stop_.load()) {
    workCV_.wait(lock, [this] { return stop_.load() || !workQueue_.empty(); });
    if (stop_.load()) {
      return;
    }
    auto task = std::move(workQueue_.front());
    workQueue_.pop_front();
    lock.unlock();
    // Run without holding the queue mutex so the main thread can keep
    // enqueuing follow-on work for the next collective.
    if (task.p2p) {
      runP2p(std::move(task));
    } else {
      task.fn();
    }
    lock.lock();
  }
}

// enqueueCollective
//
// Shared async scaffold. Record the MPS event on the calling thread
// (non-blocking commitAndContinue keeps the serial queue free so autograd / DDP
// overlap the RDMA transfer), enqueue a lambda that waits the event and runs
// `fn` on the worker, return a TCCLWork the caller blocks on via wait().
std::string ProcessGroupTCCL::refusedMessage() const {
  return "TCCL: an earlier operation on this process group failed (" +
      brokenReason_ +
      "), so the group refuses new work. Destroy it and create a new one.";
}

void ProcessGroupTCCL::markBroken(const std::exception_ptr& ep) {
  if (broken_.load()) {
    return;
  }
  try {
    std::rethrow_exception(ep);
  } catch (const c10::Error& e) {
    brokenReason_ = e.what_without_backtrace();
  } catch (const std::exception& e) {
    brokenReason_ = e.what();
  } catch (...) {
    brokenReason_ = "unknown error";
  }
  // Flush whatever the failed operation left posted, so a peer cannot receive
  // it later.
  for (auto& connection : connections_) {
    if (connection) {
      connection->abort();
    }
  }
  broken_.store(true);
}

c10::intrusive_ptr<Work> ProcessGroupTCCL::enqueueCollective(
    OpType opType,
    std::vector<at::Tensor> outputs,
    std::function<void()> fn) {
  // A collective would run before the batch's deferred sends and recvs. With no
  // p2p recorded it runs as usual (the coalescing manager's fast path).
  if (coalescing_ && !coalescedOps_.empty()) {
    const auto ep = std::make_exception_ptr(C10_BUILD_ERROR(
        DistBackendError,
        "TCCL: a batch of sends and recvs cannot also run a collective; "
        "the batch is dropped."));
    closeCoalescing(ep);
    std::rethrow_exception(ep);
  }
  TORCH_CHECK_WITH(DistBackendError, !broken_.load(), refusedMessage());
  const uint32_t mpsEventId = mpsEventRecord();
  const uint64_t mySeq = ++seq_;
  auto work = c10::make_intrusive<TCCLWork>(opType, mySeq, std::move(outputs));
  {
    std::lock_guard<std::mutex> lock(workMutex_);
    workQueue_.push_back(Task{
        [this, mpsEventId, fn = std::move(fn), work]() mutable {
          std::exception_ptr ep;
          try {
            // Block only this worker until the GPU flushes writes to the
            // inputs
            mpsEventWait(mpsEventId);
            // The group may have broken while this work was queued.
            TORCH_CHECK_WITH(
                DistBackendError, !broken_.load(), refusedMessage());
            fn();
          } catch (...) {
            ep = std::current_exception();
            markBroken(ep);
          }
          // Store any exception and wake threads blocked on Work::wait().
          work->finishWork(ep);
        },
        std::nullopt,
        std::nullopt});
  }
  workCV_.notify_one();
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupTCCL::enqueueP2p(P2POp op) {
  TORCH_CHECK_WITH(DistBackendError, !broken_.load(), refusedMessage());
  auto work = op.work;
  const uint32_t mpsEventId = mpsEventRecord();
  {
    std::lock_guard<std::mutex> lock(workMutex_);
    workQueue_.push_back(Task{nullptr, std::move(op), mpsEventId});
  }
  workCV_.notify_one();
  return work;
}

void ProcessGroupTCCL::finishP2p(P2POp& op, const std::exception_ptr& ep) {
  op.work->finishWork(ep);
  P2pBatch* batch = op.batch.get();
  if (batch && batch->work && (ep || --batch->left == 0)) {
    batch->work->finishWork(ep);
    batch->work.reset();
  }
}

void ProcessGroupTCCL::runP2p(Task first) {
  TCCLEngine::P2pStreams streams(*engine_);
  // Admitted ops, indexed by their stream tag. A finished op drops its tensor,
  // view and Work, and the vectors restart whenever every stream has drained.
  std::vector<P2POp> ops;
  std::vector<at::Tensor> views;
  std::vector<bool> finished;
  std::deque<Task> pending;
  pending.push_back(std::move(first));
  std::vector<std::size_t> done;
  try {
    auto deadline = std::chrono::steady_clock::now() + options_->timeout;
    while (true) {
      // The p2p ops next in line: a recv queued after a send is posted while
      // the send waits for its peer.
      {
        std::lock_guard<std::mutex> lock(workMutex_);
        while (!workQueue_.empty() && workQueue_.front().p2p) {
          pending.push_back(std::move(workQueue_.front()));
          workQueue_.pop_front();
        }
      }
      // Admit them in queue order once the GPU is done with their tensors.
      // While others are in flight, a not-yet-ready op waits its turn instead
      // of blocking them.
      while (!pending.empty()) {
        Task& task = pending.front();
        if (task.mpsEventId) {
          if (!streams.empty() &&
              !at::detail::getMPSHooks().queryEvent(*task.mpsEventId)) {
            break;
          }
          mpsEventWait(*task.mpsEventId);
          task.mpsEventId.reset();
        }
        TORCH_CHECK_WITH(DistBackendError, !broken_.load(), refusedMessage());
        if (streams.empty()) {
          ops.clear();
          views.clear();
          finished.clear();
        }
        P2POp& op = *task.p2p;
        views.push_back(mpsSharedCpuView(op.tensor));
        const at::Tensor& v = views.back();
        const auto bytes = static_cast<std::size_t>(v.nbytes());
        if (op.isSend) {
          streams.addSend(
              op.peer,
              static_cast<const char*>(v.data_ptr()),
              bytes,
              ops.size());
        } else {
          streams.addRecv(
              op.peer, static_cast<char*>(v.data_ptr()), bytes, ops.size());
        }
        finished.push_back(false);
        ops.push_back(std::move(op));
        pending.pop_front();
        deadline = std::chrono::steady_clock::now() + options_->timeout;
      }
      if (streams.empty()) {
        return; // pending is empty too: admission stops only behind streams
      }
      done.clear();
      // Zero-byte ops complete without traffic, so `done` can fill even when
      // poll reports no progress.
      const bool progress = streams.poll(done, "TCCL p2p");
      for (const std::size_t tag : done) {
        finished[tag] = true;
        finishP2p(ops[tag], nullptr);
        ops[tag] = P2POp{};
        views[tag] = at::Tensor();
      }
      // An op still in `pending` waits for the GPU, which is unbounded as in a
      // collective's mpsEventWait; the timeout bounds waiting for peers.
      if (progress || !done.empty() || !pending.empty()) {
        deadline = std::chrono::steady_clock::now() + options_->timeout;
      } else if (std::chrono::steady_clock::now() > deadline) {
        throwTimeout("TCCL p2p", options_->timeout, "");
      }
    }
  } catch (...) {
    const auto ep = std::current_exception();
    markBroken(ep);
    for (std::size_t i = 0; i < ops.size(); ++i) {
      if (!finished[i]) {
        finishP2p(ops[i], ep);
      }
    }
    for (auto& task : pending) {
      if (task.mpsEventId) {
        at::detail::getMPSHooks().releaseEvent(*task.mpsEventId);
      }
      finishP2p(*task.p2p, ep);
    }
  }
}

// allreduce

c10::intrusive_ptr<Work> ProcessGroupTCCL::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  assertSingleElementInput(invalidArgument("allreduce"), tensors);
  auto& tensor = tensors[0];
  checkMpsTensor(tensor, "allreduce");
  checkReduce("allreduce", tensor.scalar_type(), opts.reduceOp);

  const ReduceOp::RedOpType op = opts.reduceOp;
  const int worldSize = size_;
  const at::ScalarType st = tensor.scalar_type();
  return enqueueCollective(
      OpType::ALLREDUCE,
      std::vector<at::Tensor>{tensor},
      [this, tensor, op, worldSize, st]() mutable {
        at::Tensor cpuView = mpsSharedCpuView(tensor);
        AT_DISPATCH_V2(
            st,
            "tccl_allreduce",
            AT_WRAP([&] {
              runMeshAllreduce<scalar_t>(*engine_, cpuView, op, worldSize);
            }),
            TCCL_REDUCE_DTYPES);
      });
}

// broadcast

c10::intrusive_ptr<Work> ProcessGroupTCCL::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  const auto invalid = invalidArgument("broadcast");
  assertSingleElementInput(invalid, tensors);
  auto& tensor = tensors[0];
  checkMpsTensor(tensor, "broadcast");
  assertRootRank(invalid, opts.rootRank, size_);
  assertRootTensor(invalid, opts.rootTensor, 1);
  // Byte-copy collective - any dtype. DDP construction broadcasts int32/int64
  // metadata tensors through this path.
  const int root = opts.rootRank;
  return enqueueCollective(
      OpType::BROADCAST,
      std::vector<at::Tensor>{tensor},
      [this, tensor, root]() mutable {
        at::Tensor view = mpsSharedCpuView(tensor);
        const std::size_t nbytes = static_cast<std::size_t>(view.nbytes());
        if (tcclUseRing(
                size_,
                /*autoEnable=*/false,
                /*isBf16=*/false,
                nbytes,
                1,
                engine_->ringTopology())) {
          engine_->ring_broadcast(view.data_ptr(), nbytes, root);
        } else {
          engine_->broadcast(view.data_ptr(), nbytes, root);
        }
      });
}

c10::intrusive_ptr<Work> ProcessGroupTCCL::reduce(
    std::vector<at::Tensor>& tensors,
    const ReduceOptions& opts) {
  const auto invalid = invalidArgument("reduce");
  assertSingleElementInput(invalid, tensors);
  auto& tensor = tensors[0];
  checkMpsTensor(tensor, "reduce");
  assertRootRank(invalid, opts.rootRank, size_);
  checkReduce("reduce", tensor.scalar_type(), opts.reduceOp);
  // Runs as all-reduce - reduced value on every rank (reduce guarantees only
  // the root). rootRank is validated, not used in the reduction.
  const ReduceOp::RedOpType op = opts.reduceOp;
  const int worldSize = size_;
  const at::ScalarType st = tensor.scalar_type();
  return enqueueCollective(
      OpType::REDUCE,
      std::vector<at::Tensor>{tensor},
      [this, tensor, op, worldSize, st]() mutable {
        at::Tensor cpuView = mpsSharedCpuView(tensor);
        AT_DISPATCH_V2(
            st,
            "tccl_allreduce",
            AT_WRAP([&] {
              runMeshAllreduce<scalar_t>(*engine_, cpuView, op, worldSize);
            }),
            TCCL_REDUCE_DTYPES);
      });
}

// allgather (list form)

c10::intrusive_ptr<Work> ProcessGroupTCCL::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& /*opts*/) {
  const auto invalid = invalidArgument("allgather");
  assertSingleElementInput(invalid, inputTensors);
  assertGatherOutputTensorList(invalid, outputTensors, size_);
  auto& input = inputTensors[0];
  auto& outList = outputTensors[0];
  checkMpsTensor(input, "allgather");
  for (const auto& o : outList) {
    checkMatches(o, input, "allgather");
  }
  // Byte-copy collective - any dtype (DDP construction allgathers kLong
  // param-size vectors).
  return enqueueCollective(
      OpType::ALLGATHER, outList, [this, input, outList]() mutable {
        at::Tensor inView = mpsSharedCpuView(input);
        const std::size_t per_rank_bytes =
            static_cast<std::size_t>(inView.nbytes());

        // Hold the per-slot CPU views alive for the duration of the gather;
        // out_ptrs points into their (unified-memory) storage.
        std::vector<at::Tensor> outViews;
        std::vector<void*> outPtrs;
        outViews.reserve(outList.size());
        outPtrs.reserve(outList.size());
        for (auto& o : outList) {
          outViews.push_back(mpsSharedCpuView(o));
          outPtrs.push_back(outViews.back().data_ptr());
        }

        if (tcclUseRing(
                size_,
                /*autoEnable=*/false,
                /*isBf16=*/false,
                per_rank_bytes * static_cast<std::size_t>(size_),
                1,
                engine_->ringTopology())) {
          engine_->ring_all_gather(inView.data_ptr(), outPtrs, per_rank_bytes);
        } else {
          engine_->all_gather(inView.data_ptr(), outPtrs, per_rank_bytes);
        }
      });
}

// all_gather_into_tensor, single and coalesced (TP all-gather)

c10::intrusive_ptr<Work> ProcessGroupTCCL::all_gather_single(
    at::Tensor& outputBuffer,
    at::Tensor& inputBuffer,
    const AllgatherOptions& /*opts*/) {
  std::vector<at::Tensor> outputs{outputBuffer};
  std::vector<at::Tensor> inputs{inputBuffer};
  return allGatherIntoTensors(outputs, inputs, OpType::_ALLGATHER_BASE);
}

c10::intrusive_ptr<Work> ProcessGroupTCCL::all_gather_single_coalesced(
    std::vector<at::Tensor>& outputs,
    std::vector<at::Tensor>& inputs,
    const AllgatherOptions& /*opts*/) {
  return allGatherIntoTensors(outputs, inputs, OpType::ALLGATHER_COALESCED);
}

c10::intrusive_ptr<Work> ProcessGroupTCCL::allGatherIntoTensors(
    std::vector<at::Tensor>& outputs,
    std::vector<at::Tensor>& inputs,
    OpType opType) {
  constexpr const char* name = "all_gather_into_tensor";
  assertInputOutputTensorListsSameSize(
      invalidArgument(name), outputs.size(), inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    checkMpsTensor(inputs[i], name);
    checkMpsTensor(outputs[i], name);
    TORCH_CHECK_WITH(
        DistBackendError,
        outputs[i].scalar_type() == inputs[i].scalar_type() &&
            outputs[i].numel() == inputs[i].numel() * size_,
        fmt::format(
            "TCCL all_gather_into_tensor: the output must have the input's "
            "dtype {} and world_size ({}) times its numel {}, got {} and {}.",
            c10::toString(inputs[i].scalar_type()),
            size_,
            inputs[i].numel(),
            c10::toString(outputs[i].scalar_type()),
            outputs[i].numel()));
  }

  // Byte-copy, any dtype. Each output is one contiguous rank-major buffer.
  return enqueueCollective(opType, outputs, [this, inputs, outputs]() mutable {
    for (size_t i = 0; i < inputs.size(); ++i) {
      at::Tensor inView = mpsSharedCpuView(inputs[i]);
      at::Tensor outView = mpsSharedCpuView(outputs[i]);
      const std::size_t per_rank_bytes =
          static_cast<std::size_t>(inView.nbytes());

      std::vector<void*> outPtrs(static_cast<std::size_t>(size_));
      char* ob = reinterpret_cast<char*>(outView.data_ptr());
      for (int r = 0; r < size_; ++r) {
        outPtrs[r] = ob + static_cast<std::size_t>(r) * per_rank_bytes;
      }

      if (tcclUseRing(
              size_,
              /*autoEnable=*/false,
              /*isBf16=*/false,
              per_rank_bytes * static_cast<std::size_t>(size_),
              1,
              engine_->ringTopology())) {
        engine_->ring_all_gather(inView.data_ptr(), outPtrs, per_rank_bytes);
      } else {
        engine_->all_gather(inView.data_ptr(), outPtrs, per_rank_bytes);
      }
    }
  });
}

c10::intrusive_ptr<Work> ProcessGroupTCCL::gather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const GatherOptions& opts) {
  const auto invalid = invalidArgument("gather");
  assertSingleElementInput(invalid, inputTensors);
  auto& input = inputTensors[0];
  const int root = opts.rootRank;
  assertRootRank(invalid, root, size_);
  checkMpsTensor(input, "gather");
  // Root needs a QP to every rank - mesh-only (ring has no non-neighbor link).
  TORCH_CHECK_WITH(
      DistBackendError,
      !engine_->ringTopology(),
      "TCCL gather: not supported on a ring topology (the root needs a QP to "
      "every rank); use a full-mesh cluster.");
  const bool isRoot = (rank_ == root);
  std::vector<at::Tensor> outList;
  if (isRoot) {
    assertGatherOutputTensorList(invalid, outputTensors, size_);
    outList = outputTensors[0];
    for (const auto& o : outList) {
      checkMatches(o, input, "gather");
    }
  }
  // Byte-copy - any dtype.
  return enqueueCollective(
      OpType::GATHER,
      isRoot ? outList : std::vector<at::Tensor>{},
      [this, input, outList, isRoot, root]() mutable {
        at::Tensor inView = mpsSharedCpuView(input);
        const std::size_t nbytes = static_cast<std::size_t>(inView.nbytes());
        // Keep the CPU views alive - out_ptrs alias their memory (root only).
        std::vector<at::Tensor> outViews;
        std::vector<void*> outPtrs;
        if (isRoot) {
          outViews.reserve(outList.size());
          outPtrs.reserve(outList.size());
          for (auto& o : outList) {
            outViews.push_back(mpsSharedCpuView(o));
            outPtrs.push_back(outViews.back().data_ptr());
          }
        }
        engine_->gather(inView.data_ptr(), outPtrs, nbytes, root);
      });
}

c10::intrusive_ptr<Work> ProcessGroupTCCL::scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ScatterOptions& opts) {
  const auto invalid = invalidArgument("scatter");
  assertSingleElementOutput(invalid, outputTensors);
  auto& output = outputTensors[0];
  const int root = opts.rootRank;
  assertRootRank(invalid, root, size_);
  checkMpsTensor(output, "scatter");
  TORCH_CHECK_WITH(
      DistBackendError,
      !engine_->ringTopology(),
      "TCCL scatter: not supported on a ring topology (the root needs a QP to "
      "every rank); use a full-mesh cluster.");
  const bool isRoot = (rank_ == root);
  std::vector<at::Tensor> inList;
  if (isRoot) {
    assertScatterInputTensorList(invalid, inputTensors, size_);
    inList = inputTensors[0];
    for (const auto& in : inList) {
      checkMatches(in, output, "scatter");
    }
  }
  // Byte-copy - any dtype.
  return enqueueCollective(
      OpType::SCATTER,
      std::vector<at::Tensor>{output},
      [this, output, inList, isRoot, root]() mutable {
        at::Tensor outView = mpsSharedCpuView(output);
        const std::size_t nbytes = static_cast<std::size_t>(outView.nbytes());
        // Keep the CPU views alive - in_ptrs alias their memory (root only).
        std::vector<at::Tensor> inViews;
        std::vector<const void*> inPtrs;
        if (isRoot) {
          inViews.reserve(inList.size());
          inPtrs.reserve(inList.size());
          for (auto& in : inList) {
            inViews.push_back(mpsSharedCpuView(in));
            inPtrs.push_back(inViews.back().data_ptr());
          }
        }
        engine_->scatter(inPtrs, outView.data_ptr(), nbytes, root);
      });
}

// reduce_scatter (list form)
//
// outputTensors[i] receives the element-wise reduction across ranks of
// inputTensors[i][rank_]; inputTensors[i] is this rank's world_size
// contributions (chunk p destined for peer p), fed to the engine directly as
// per-rank chunk pointers - no pre-gather copy.

c10::intrusive_ptr<Work> ProcessGroupTCCL::reduce_scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ReduceScatterOptions& opts) {
  const auto invalid = invalidArgument("reduce_scatter");
  assertInputOutputTensorListsSameSize(
      invalid, outputTensors.size(), inputTensors.size());
  for (size_t i = 0; i < outputTensors.size(); ++i) {
    auto& out = outputTensors[i];
    assertInputTensorListSizeEqualsWorldSize(
        invalid, inputTensors[i].size(), size_);
    checkMpsTensor(out, "reduce_scatter");
    checkReduce("reduce_scatter", out.scalar_type(), opts.reduceOp);
    for (const auto& chunk : inputTensors[i]) {
      checkMatches(chunk, out, "reduce_scatter");
    }
  }
  const ReduceOp::RedOpType op = opts.reduceOp;
  const int worldSize = size_;
  return enqueueCollective(
      OpType::REDUCE_SCATTER,
      outputTensors,
      [this, inputTensors, outputTensors, op, worldSize]() mutable {
        for (size_t i = 0; i < outputTensors.size(); ++i) {
          at::Tensor outView = mpsSharedCpuView(outputTensors[i]);
          // Hold the per-chunk CPU views alive for the duration of the call;
          // dispatchReduceScatterList reads their (unified-memory) storage.
          std::vector<at::Tensor> inChunkViews;
          inChunkViews.reserve(inputTensors[i].size());
          for (auto& chunk : inputTensors[i]) {
            inChunkViews.push_back(mpsSharedCpuView(chunk));
          }
          dispatchReduceScatterList(
              *engine_, inChunkViews, outView, op, worldSize);
        }
      });
}

// reduce_scatter_tensor, single and coalesced (FSDP, TP / sequence parallel)

c10::intrusive_ptr<Work> ProcessGroupTCCL::reduce_scatter_single(
    at::Tensor& outputBuffer,
    at::Tensor& inputBuffer,
    const ReduceScatterOptions& opts) {
  std::vector<at::Tensor> outputs{outputBuffer};
  std::vector<at::Tensor> inputs{inputBuffer};
  return reduceScatterTensors(
      outputs, inputs, opts, OpType::_REDUCE_SCATTER_BASE);
}

c10::intrusive_ptr<Work> ProcessGroupTCCL::reduce_scatter_single_coalesced(
    std::vector<at::Tensor>& outputs,
    std::vector<at::Tensor>& inputs,
    const ReduceScatterOptions& opts) {
  return reduceScatterTensors(
      outputs, inputs, opts, OpType::REDUCE_SCATTER_TENSOR_COALESCED);
}

c10::intrusive_ptr<Work> ProcessGroupTCCL::reduceScatterTensors(
    std::vector<at::Tensor>& outputs,
    std::vector<at::Tensor>& inputs,
    const ReduceScatterOptions& opts,
    OpType opType) {
  constexpr const char* name = "reduce_scatter_tensor";
  assertInputOutputTensorListsSameSize(
      invalidArgument(name), outputs.size(), inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    checkMpsTensor(inputs[i], name);
    checkMpsTensor(outputs[i], name);
    checkReduce(name, inputs[i].scalar_type(), opts.reduceOp);
    TORCH_CHECK_WITH(
        DistBackendError,
        outputs[i].scalar_type() == inputs[i].scalar_type() &&
            inputs[i].numel() == outputs[i].numel() * size_,
        fmt::format(
            "TCCL reduce_scatter_tensor: the input must have the output's "
            "dtype {} and world_size ({}) times its numel {}, got {} and {}.",
            c10::toString(outputs[i].scalar_type()),
            size_,
            outputs[i].numel(),
            c10::toString(inputs[i].scalar_type()),
            inputs[i].numel()));
  }

  const ReduceOp::RedOpType op = opts.reduceOp;
  const int worldSize = size_;
  return enqueueCollective(
      opType, outputs, [this, inputs, outputs, op, worldSize]() mutable {
        for (size_t i = 0; i < inputs.size(); ++i) {
          at::Tensor inView = mpsSharedCpuView(inputs[i]);
          at::Tensor outView = mpsSharedCpuView(outputs[i]);
          dispatchReduceScatter(*engine_, inView, outView, op, worldSize);
        }
      });
}

c10::intrusive_ptr<Work> ProcessGroupTCCL::all_to_all_single(
    at::Tensor& outputBuffer,
    at::Tensor& inputBuffer,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const AllToAllOptions& /*opts*/) {
  checkMpsTensor(inputBuffer, "all_to_all_single");
  checkMpsTensor(outputBuffer, "all_to_all_single");
  TORCH_CHECK_WITH(
      DistBackendError,
      inputBuffer.dim() >= 1 && outputBuffer.dim() >= 1 &&
          outputBuffer.scalar_type() == inputBuffer.scalar_type(),
      "TCCL all_to_all_single: input and output need dim >= 1 and one dtype.");
  checkSplitSizes(inputSplitSizes, inputBuffer, size_);
  checkSplitSizes(outputSplitSizes, outputBuffer, size_);

  // Split along dim 0, in elements, then in bytes for the engine.
  const int ws = size_;
  std::vector<std::size_t> sOff(ws), sBytes(ws), rOff(ws), rBytes(ws);
  computeLengthsAndOffsets(inputSplitSizes, inputBuffer, &sBytes, &sOff);
  computeLengthsAndOffsets(outputSplitSizes, outputBuffer, &rBytes, &rOff);
  const auto inElem = static_cast<std::size_t>(inputBuffer.element_size());
  const auto outElem = static_cast<std::size_t>(outputBuffer.element_size());
  for (int p = 0; p < ws; ++p) {
    sOff[p] *= inElem;
    sBytes[p] *= inElem;
    rOff[p] *= outElem;
    rBytes[p] *= outElem;
  }
  const bool equal = inputSplitSizes.empty() && outputSplitSizes.empty();

  // Ring requires equal (uniform) segments + ws > 2; tcclUseRing only returns
  // true under TCCL_FORCE_ALGO=ring or ring topology. Everything else (the
  // default, and all uneven splits) uses the mesh/direct form, which handles
  // variable per-peer sizes.
  const bool ringTop = engine_->ringTopology();
  TORCH_CHECK_WITH(
      DistBackendError,
      !ringTop || (equal && sBytes[0] == rBytes[0]),
      "TCCL all_to_all_single: ring topology supports only uniform (equal-split) "
      "all-to-all; uneven splits require the full mesh (non-neighbor peers are "
      "not connected on a ring).");
  const bool ringOk = equal && sBytes[0] == rBytes[0] &&
      tcclUseRing(ws,
                  /*autoEnable=*/false,
                  /*isBf16=*/false,
                  sBytes[0] * static_cast<std::size_t>(ws),
                  1,
                  ringTop);
  const std::size_t segBytes = equal ? sBytes[0] : 0;

  return enqueueCollective(
      OpType::ALLTOALL_BASE,
      std::vector<at::Tensor>{outputBuffer},
      [this,
       inputBuffer,
       outputBuffer,
       sOff,
       sBytes,
       rOff,
       rBytes,
       ringOk,
       segBytes]() mutable {
        at::Tensor inView = mpsSharedCpuView(inputBuffer);
        at::Tensor outView = mpsSharedCpuView(outputBuffer);
        const char* sb = reinterpret_cast<const char*>(inView.data_ptr());
        char* rb = reinterpret_cast<char*>(outView.data_ptr());
        if (ringOk) {
          engine_->ring_all_to_all(sb, rb, segBytes);
        } else {
          engine_->all_to_all(sb, rb, sOff, sBytes, rOff, rBytes);
        }
      });
}

void ProcessGroupTCCL::checkP2p(
    const std::vector<at::Tensor>& tensors,
    int peer,
    bool isSend) const {
  const char* op = isSend ? "send" : "recv";
  const char* arg = isSend ? "dstRank" : "srcRank";
  assertSingleElement(invalidArgument(op), tensors);
  checkMpsTensor(tensors[0], op);
  TORCH_CHECK_WITH(
      DistBackendError,
      peer >= 0 && peer < size_ && peer != rank_,
      fmt::format(
          "TCCL {}: invalid {} {} for rank {} (world size {}).",
          op,
          arg,
          peer,
          rank_,
          size_));
  // Ring topology only connects the two neighbors ((rank+/-1)%world); any
  // other peer has a null connection slot.
  TORCH_CHECK_WITH(
      DistBackendError,
      !engine_->ringTopology() || peer == (rank_ + 1) % size_ ||
          peer == (rank_ - 1 + size_) % size_,
      fmt::format(
          "TCCL {}: ring topology only connects a rank to its ring neighbors "
          "((rank+/-1)%world); {} {} is not a neighbor of rank {}.",
          op,
          arg,
          peer,
          rank_));
}

c10::intrusive_ptr<Work> ProcessGroupTCCL::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int /*tag*/) {
  try {
    checkP2p(tensors, dstRank, /*isSend=*/true);
  } catch (...) {
    closeCoalescing(std::current_exception());
    throw;
  }
  auto& tensor = tensors[0];
  // Byte movement - any dtype. tag is accepted but not hardware-matched (UC has
  // no tag matching); ordering is the per-peer FIFO of matched send/recv pairs.
  auto work = c10::make_intrusive<TCCLWork>(OpType::SEND, ++seq_);
  if (coalescing_) {
    coalescedOps_.push_back({/*isSend=*/true, dstRank, tensor, work});
    return work;
  }
  return enqueueP2p({/*isSend=*/true, dstRank, tensor, std::move(work)});
}

c10::intrusive_ptr<Work> ProcessGroupTCCL::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int /*tag*/) {
  try {
    checkP2p(tensors, srcRank, /*isSend=*/false);
  } catch (...) {
    closeCoalescing(std::current_exception());
    throw;
  }
  auto& tensor = tensors[0];
  auto work = c10::make_intrusive<TCCLWork>(
      OpType::RECV, ++seq_, std::vector<at::Tensor>{tensor});
  if (coalescing_) {
    coalescedOps_.push_back({/*isSend=*/false, srcRank, tensor, work});
    return work;
  }
  return enqueueP2p({/*isSend=*/false, srcRank, tensor, std::move(work)});
}

// barrier

c10::intrusive_ptr<Work> ProcessGroupTCCL::barrier(
    const BarrierOptions& /*opts*/) {
  // No data path - Store::barrier is the optimized server-side primitive.
  // Fresh sequence number per call so rapid-fire barriers don't collide.
  const uint64_t mySeq = ++barrierSeq_;
  const std::string key = "tccl_barrier_" + std::to_string(mySeq);

  // Run the barrier synchronously on the calling thread, then return a
  // pre-completed Work.
  tcclRtsBarrier(*store_, size_, key, options_->timeout);

  auto work = c10::make_intrusive<TCCLWork>(OpType::BARRIER, mySeq);
  work->finishWork();
  return work;
}

// batch_isend_irecv

void ProcessGroupTCCL::startCoalescing() {
  TORCH_CHECK_WITH(
      DistBackendError,
      !coalescing_,
      "TCCL: startCoalescing called while a batch is already open.");
  coalescing_ = true;
}

void ProcessGroupTCCL::closeCoalescing(const std::exception_ptr& ep) {
  if (!coalescing_) {
    return;
  }
  coalescing_ = false;
  for (auto& op : coalescedOps_) {
    op.work->finishWork(ep);
  }
  coalescedOps_.clear();
}

c10::intrusive_ptr<Work> ProcessGroupTCCL::endCoalescing() {
  TORCH_CHECK_WITH(
      DistBackendError,
      coalescing_,
      "TCCL: endCoalescing called without startCoalescing.");
  coalescing_ = false;
  std::vector<P2POp> ops = std::move(coalescedOps_);
  coalescedOps_.clear();
  if (ops.empty()) {
    // Nothing to run. _abort_process_group opens an empty batch, which must not
    // wait behind queued work.
    auto work = c10::make_intrusive<TCCLWork>(OpType::COALESCED, ++seq_);
    work->finishWork();
    return work;
  }
  if (broken_.load()) {
    const auto ep = std::make_exception_ptr(
        C10_BUILD_ERROR(DistBackendError, refusedMessage()));
    for (auto& op : ops) {
      op.work->finishWork(ep);
    }
    std::rethrow_exception(ep);
  }
  // The batch's ops join the worker's p2p streams (runP2p) like plain sends
  // and recvs, so their order inside the batch does not matter. One MPS event,
  // waited for by the first op, covers the whole batch.
  auto work = c10::make_intrusive<TCCLWork>(OpType::COALESCED, ++seq_);
  auto batch = std::make_shared<P2pBatch>(P2pBatch{ops.size(), work});
  const uint32_t mpsEventId = mpsEventRecord();
  {
    std::lock_guard<std::mutex> lock(workMutex_);
    std::optional<uint32_t> eid = mpsEventId;
    for (auto& op : ops) {
      op.batch = batch;
      workQueue_.push_back(Task{nullptr, std::move(op), eid});
      eid.reset();
    }
  }
  workCV_.notify_one();
  return work;
}

} // namespace c10d

#endif // USE_C10D_TCCL
