#ifdef USE_C10D_TCCL

#include <torch/csrc/distributed/c10d/ProcessGroupTCCL.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupTCCLDetail.hpp>
#include <torch/csrc/distributed/c10d/TCCLUtils.hpp>

#include <c10/util/Exception.h>
#include <c10/util/thread_name.h>

#include <ATen/core/jit_type.h>
#include <ATen/Context.h>
#include <ATen/mps/MPSAllocatorInterface.h>
#include <ATen/detail/MPSHooksInterface.h>

#include <cstddef>
#include <cstdlib>
#include <cstdint>
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
// MTLBuffer arithmetic, unmappable for offset views (e.g. _allgather_base).
at::Tensor mpsSharedCpuView(const at::Tensor& mpsTensor) {
  auto* allocator = at::mps::getIMPSAllocator();
  const void* storage_ptr = mpsTensor.storage().data_ptr().get();
  // Pair is {cpu_ptr, retain_count}; only the pointer is used here.
  const void* cpu_ptr = allocator->getSharedBufferPtr(storage_ptr).first;
  TORCH_INTERNAL_ASSERT(
      cpu_ptr != nullptr,
      "MPS tensor does not have a shared (unified memory) backing buffer. ",
      "storage_ptr=",
      storage_ptr,
      " data_ptr=",
      mpsTensor.data_ptr(),
      " size=",
      mpsTensor.nbytes(),
      " storage_offset=",
      mpsTensor.storage_offset(),
      " device=",
      mpsTensor.device());
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
  hooks.recordEvent(eid);
  return eid;
}

// Block the calling thread until the GPU reaches the recorded signal, then
// release the event. Returns immediately if the GPU already passed it.
void mpsEventWait(uint32_t eid) {
  auto& hooks = at::detail::getMPSHooks();
  hooks.synchronizeEvent(eid);
  hooks.releaseEvent(eid);
}

constexpr const char* kNotImplementedHint =
    "is not implemented for the TCCL backend. Implemented collectives: "
    "allreduce, broadcast, allgather, _allgather_base, "
    "allgather_into_tensor_coalesced, reduce_scatter, _reduce_scatter_base, "
    "reduce_scatter_tensor_coalesced, alltoall_base, send, recv, and barrier "
    "(reductions: SUM/MIN/MAX/PRODUCT for float32/float16/bfloat16 + "
    "int8/16/32/64/uint8/bool, AVG float-only; broadcast/allgather/alltoall/send/"
    "recv are byte-copy, any dtype).";

// SUM-capable reduce dtypes
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

// Ring-vs-mesh selection for all_reduce (N>2): bf16 -> ring at any size,
// fp32/fp16 -> ring only >=8MB. TCCL_FORCE_ALGO=ring|mesh overrides per-call.
inline bool tcclUseRing(
    int worldSize, bool autoEnable, bool isBf16, std::size_t count,
    std::size_t elemBytes, bool forceRing = false) {

  if (worldSize <= 2) {
    return false;
  }
  if (forceRing) {
    return true;
  }
  // Auto-heuristic: bf16 any size, else >=1MB
  bool useRing = autoEnable &&
      (isBf16 || count * elemBytes >= 1u * 1024u * 1024u);
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
    TCCLEngine& mesh, T* data, std::size_t count, int worldSize, bool useRing) {
  if (useRing) {
    mesh.ring_all_reduce<T, Op>(data, count, Op{});
  } else {
    mesh.all_reduce<T, Op>(data, count, Op{});
  }
}

// Typed mesh allreduce. `op` selects the reduce op (SUM/MIN/MAX/AVG)
template <typename T>
void runMeshAllreduce(
    TCCLEngine& mesh, at::Tensor& view, ReduceOp::RedOpType op, int worldSize) {
  T* data = view.data_ptr<T>();
  const std::size_t count = static_cast<std::size_t>(view.numel());
  const bool useRing =
      tcclUseRing(worldSize, /*autoEnable=*/true, std::is_same_v<T, at::BFloat16>, count, sizeof(T), mesh.ringTopology());
  switch (op) {
    case ReduceOp::MIN:
      tcclAllreduceWith<T, TCCLMinOp<T>>(mesh, data, count, worldSize, useRing);
      break;
    case ReduceOp::MAX:
      tcclAllreduceWith<T, TCCLMaxOp<T>>(mesh, data, count, worldSize, useRing);
      break;
    case ReduceOp::PRODUCT:
      tcclAllreduceWith<T, TCCLProdOp<T>>(mesh, data, count, worldSize, useRing);
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
      worldSize, /*autoEnable=*/false, std::is_same_v<T, at::BFloat16>,
      count_per_rank * static_cast<std::size_t>(worldSize), sizeof(T), mesh.ringTopology());
  switch (op) {
    case ReduceOp::MIN:
      if (useRing)
        mesh.ring_reduce_scatter<T, TCCLMinOp<T>>(in_chunks, out, count_per_rank, TCCLMinOp<T>{});
      else
        mesh.reduce_scatter<T, TCCLMinOp<T>>(in_chunks, out, count_per_rank, TCCLMinOp<T>{});
      break;
    case ReduceOp::MAX:
      if (useRing)
        mesh.ring_reduce_scatter<T, TCCLMaxOp<T>>(in_chunks, out, count_per_rank, TCCLMaxOp<T>{});
      else
        mesh.reduce_scatter<T, TCCLMaxOp<T>>(in_chunks, out, count_per_rank, TCCLMaxOp<T>{});
      break;
    case ReduceOp::PRODUCT:
      if (useRing)
        mesh.ring_reduce_scatter<T, TCCLProdOp<T>>(in_chunks, out, count_per_rank, TCCLProdOp<T>{});
      else
        mesh.reduce_scatter<T, TCCLProdOp<T>>(in_chunks, out, count_per_rank, TCCLProdOp<T>{});
      break;
    // SUM or AVG
    default:
      if (useRing)
        mesh.ring_reduce_scatter<T, TCCLSumOp<T>>(in_chunks, out, count_per_rank, TCCLSumOp<T>{});
      else
        mesh.reduce_scatter<T, TCCLSumOp<T>>(in_chunks, out, count_per_rank, TCCLSumOp<T>{});
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
  switch (inView.scalar_type()) {
    case at::kFloat: runMeshReduceScatter<float>(mesh, inView, outView, op, worldSize); break;
    case at::kHalf: runMeshReduceScatter<at::Half>(mesh, inView, outView, op, worldSize); break;
    case at::kBFloat16: runMeshReduceScatter<at::BFloat16>(mesh, inView, outView, op, worldSize); break;
    case at::kChar: runMeshReduceScatter<int8_t>(mesh, inView, outView, op, worldSize); break;
    case at::kShort: runMeshReduceScatter<int16_t>(mesh, inView, outView, op, worldSize); break;
    case at::kInt: runMeshReduceScatter<int32_t>(mesh, inView, outView, op, worldSize); break;
    case at::kLong: runMeshReduceScatter<int64_t>(mesh, inView, outView, op, worldSize); break;
    case at::kByte: runMeshReduceScatter<uint8_t>(mesh, inView, outView, op, worldSize); break;
    case at::kBool: runMeshReduceScatter<bool>(mesh, inView, outView, op, worldSize); break;
    // Unreachable - validated by caller
    default: break;
  }
}

// Runtime-dtype dispatch for list-form reduce-scatter (worldSize separate input
// views, one per destination rank).
void dispatchReduceScatterList(
    TCCLEngine& mesh,
    std::vector<at::Tensor>& inChunkViews,
    at::Tensor& outView,
    ReduceOp::RedOpType op,
    int worldSize) {
  switch (outView.scalar_type()) {
    case at::kFloat: runMeshReduceScatterList<float>(mesh, inChunkViews, outView, op, worldSize); break;
    case at::kHalf: runMeshReduceScatterList<at::Half>(mesh, inChunkViews, outView, op, worldSize); break;
    case at::kBFloat16: runMeshReduceScatterList<at::BFloat16>(mesh, inChunkViews, outView, op, worldSize); break;
    case at::kChar: runMeshReduceScatterList<int8_t>(mesh, inChunkViews, outView, op, worldSize); break;
    case at::kShort: runMeshReduceScatterList<int16_t>(mesh, inChunkViews, outView, op, worldSize); break;
    case at::kInt: runMeshReduceScatterList<int32_t>(mesh, inChunkViews, outView, op, worldSize); break;
    case at::kLong: runMeshReduceScatterList<int64_t>(mesh, inChunkViews, outView, op, worldSize); break;
    case at::kByte: runMeshReduceScatterList<uint8_t>(mesh, inChunkViews, outView, op, worldSize); break;
    case at::kBool: runMeshReduceScatterList<bool>(mesh, inChunkViews, outView, op, worldSize); break;
    // Unreachable - validated by caller
    default: break;
  }
}
} // namespace

// Options

ProcessGroupTCCL::Options::Options(std::chrono::milliseconds timeout)
    : Backend::Options(TCCL_BACKEND_NAME, timeout) {}

// TCCLWork

ProcessGroupTCCL::TCCLWork::TCCLWork(
    OpType opType, uint64_t seq, std::vector<at::Tensor> outputs)
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
    : Backend(rank, size),
      store_(store),
      options_(std::move(options)) {
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

  // 0. Resolve topology: Options field, overridden by env TCCL_TOPOLOGY=ring|mesh.
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
        "ProcessGroupTCCL: device '",
        dev,
        "' would host num_wires=",
        num_wires,
        " * ",
        peers,
        " peer(s) = ",
        qps,
        " QPs, exceeding the 10-UC-QP-per-device Thunderbolt limit "
        "(TN3205 sec. 12.1). Spread peers across more devices (full mesh) or "
        "reduce num_wires.");
  }

  // 4. Reserve a unique init-sequence number from the Store. The keys we
  //    use for destination exchange are prefixed with this number.
  const int64_t initSeq = tcclInitSequence(*store_, rank, options_->timeout);
  const std::string keyPrefix =
      "tccl_dest_s" + std::to_string(initSeq) + "_";

  // 5. Construct one TCCLConnection per (peer, wire). Each constructor opens its
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
  pipeSendBuffers_.resize(static_cast<size_t>(size) * TCCLEngine::kPipelineDepth);
  pipeRecvBuffers_.resize(static_cast<size_t>(size) * TCCLEngine::kPipelineDepth);
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
      const size_t myConn =
          static_cast<size_t>(peer) * num_wires + wire;
      const size_t peerSlotForMe =
          static_cast<size_t>(rank) * num_wires + wire;
      const TCCLDestination& remote =
          remoteDests[peer][peerSlotForMe];
      connections_[myConn]->transitionToRTR(remote);
      connections_[myConn]->transitionToRTS();
    }
  }

  // 9. Final sync - every rank reaches RTS before any rank may post a send.
  tcclRtsBarrier(
      *store_,
      size,
      keyPrefix + "rts_ready",
      options_->timeout);

  // 10. Construct the collective engine (holds refs into our connection + buffer vectors).
  engine_ = std::make_unique<TCCLEngine>(
      rank, size, connections_, sendBuffers_, recvBuffers_, pipeSendBuffers_,
      pipeRecvBuffers_, options_->timeout,
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
    workCV_.wait(lock, [this] {
      return stop_.load() || !workQueue_.empty();
    });
    if (stop_.load()) {
      return;
    }
    auto task = std::move(workQueue_.front());
    workQueue_.pop_front();
    lock.unlock();
    // Run without holding the queue mutex so the main thread can keep
    // enqueuing follow-on work for the next collective.
    task();
    lock.lock();
  }
}

// enqueueCollective
//
// Shared async scaffold. Record the MPS event on the calling thread
// (non-blocking commitAndContinue keeps the serial queue free so autograd / DDP
// overlap the RDMA transfer), enqueue a lambda that waits the event and runs
// `fn` on the worker, return a TCCLWork the caller blocks on via wait().
c10::intrusive_ptr<Work> ProcessGroupTCCL::enqueueCollective(
    OpType opType,
    std::vector<at::Tensor> outputs,
    std::function<void()> fn) {
  const uint32_t mpsEventId = mpsEventRecord();
  const uint64_t mySeq = ++seq_;
  auto work = c10::make_intrusive<TCCLWork>(opType, mySeq, std::move(outputs));
  {
    std::lock_guard<std::mutex> lock(workMutex_);
    workQueue_.push_back(
        [mpsEventId, fn = std::move(fn), work]() mutable {
          std::exception_ptr ep;
          try {
            // Block only this worker until the GPU flushes writes to the inputs
            mpsEventWait(mpsEventId);
            fn();
          } catch (...) {
            ep = std::current_exception();
          }
          // Store any exception and wake threads blocked on Work::wait().
          work->finishWork(ep);
        });
  }
  workCV_.notify_one();
  return work;
}

// allreduce

c10::intrusive_ptr<Work> ProcessGroupTCCL::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  TORCH_CHECK_WITH(
      DistBackendError,
      tensors.size() == 1,
      "TCCL allreduce: expected exactly one tensor, got ",
      tensors.size());
  auto& tensor = tensors[0];
  TORCH_CHECK_WITH(
      DistBackendError,
      tensor.device().is_mps(),
      "TCCL allreduce: tensor must be on MPS device, got ",
      tensor.device());
  TORCH_CHECK_WITH(
      DistBackendError,
      isSupportedReduceDtype(tensor.scalar_type()),
      "TCCL allreduce: unsupported dtype. Supported: float32/float16/bfloat16 "
      "and int8/int16/int32/int64/uint8/bool (SUM); uint16/32/64 have no MPS add "
      "and are unsupported. ",
      kNotImplementedHint,
      " Got dtype=",
      tensor.scalar_type());
  TORCH_CHECK_WITH(
      DistBackendError,
      opts.reduceOp == ReduceOp::SUM || opts.reduceOp == ReduceOp::AVG ||
          opts.reduceOp == ReduceOp::MIN || opts.reduceOp == ReduceOp::MAX ||
          opts.reduceOp == ReduceOp::PRODUCT,
      "TCCL allreduce: only SUM, AVG (= SUM / world_size), MIN, MAX and "
      "PRODUCT are supported. ",
      kNotImplementedHint);
  TORCH_CHECK_WITH(
      DistBackendError,
      opts.reduceOp != ReduceOp::AVG || isFloatReduceDtype(tensor.scalar_type()),
      "TCCL allreduce: AVG is only supported for float32/float16/bfloat16 "
      "(integer/bool AVG is undefined; matches NCCL). Use SUM for integer dtypes. ",
      " Got dtype=",
      tensor.scalar_type());
  TORCH_CHECK_WITH(
      DistBackendError,
      tensor.is_contiguous(),
      "TCCL allreduce: non-contiguous tensors are not supported. "
      "Call .contiguous() before allreduce.");

  const ReduceOp::RedOpType op = opts.reduceOp;
  const int worldSize = size_;
  const at::ScalarType st = tensor.scalar_type();
  return enqueueCollective(
      OpType::ALLREDUCE,
      std::vector<at::Tensor>{tensor},
      [this, tensor, op, worldSize, st]() mutable {
        at::Tensor cpuView = mpsSharedCpuView(tensor);
        switch (st) {
          case at::kFloat:
            runMeshAllreduce<float>(*engine_, cpuView, op, worldSize);
            break;
          case at::kHalf:
            runMeshAllreduce<at::Half>(*engine_, cpuView, op, worldSize);
            break;
          case at::kBFloat16:
            runMeshAllreduce<at::BFloat16>(*engine_, cpuView, op, worldSize);
            break;
          // Integer/bool: SUM (native add / OR) and MIN/MAX (std::min/max).
          // AVG is gated to floats above, so op is never AVG here.
          case at::kChar:
            runMeshAllreduce<int8_t>(*engine_, cpuView, op, worldSize);
            break;
          case at::kShort:
            runMeshAllreduce<int16_t>(*engine_, cpuView, op, worldSize);
            break;
          case at::kInt:
            runMeshAllreduce<int32_t>(*engine_, cpuView, op, worldSize);
            break;
          case at::kLong:
            runMeshAllreduce<int64_t>(*engine_, cpuView, op, worldSize);
            break;
          case at::kByte:
            runMeshAllreduce<uint8_t>(*engine_, cpuView, op, worldSize);
            break;
          case at::kBool:
            runMeshAllreduce<bool>(*engine_, cpuView, op, worldSize);
            break;
          default:
            // Unreachable - validated above
            break;
        }
      });
}

// broadcast

c10::intrusive_ptr<Work> ProcessGroupTCCL::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  TORCH_CHECK_WITH(
      DistBackendError,
      tensors.size() == 1,
      "TCCL broadcast: expected exactly one tensor, got ",
      tensors.size());
  auto& tensor = tensors[0];
  TORCH_CHECK_WITH(
      DistBackendError,
      tensor.device().is_mps(),
      "TCCL broadcast: tensor must be on MPS device, got ",
      tensor.device());
  TORCH_CHECK_WITH(
      DistBackendError,
      tensor.is_contiguous(),
      "TCCL broadcast: non-contiguous tensors are not supported. "
      "Call .contiguous() first.");
  TORCH_CHECK_WITH(
      DistBackendError,
      opts.rootRank >= 0 && opts.rootRank < size_,
      "TCCL broadcast: invalid rootRank ",
      opts.rootRank,
      " for world size ",
      size_);
  TORCH_CHECK_WITH(
      DistBackendError,
      opts.rootTensor == 0,
      "TCCL broadcast: rootTensor must be 0 (one tensor per rank), got ",
      opts.rootTensor);

  // Byte-copy collective - any dtype. DDP construction broadcasts int32/int64
  // metadata tensors through this path.
  const int root = opts.rootRank;
  return enqueueCollective(
      OpType::BROADCAST,
      std::vector<at::Tensor>{tensor},
      [this, tensor, root]() mutable {
        at::Tensor view = mpsSharedCpuView(tensor);
        const std::size_t nbytes = static_cast<std::size_t>(view.nbytes());
        if (tcclUseRing(size_, /*autoEnable=*/false, /*isBf16=*/false, nbytes, 1,
                        engine_->ringTopology())) {
          engine_->ring_broadcast(view.data_ptr(), nbytes, root);
        } else {
          engine_->broadcast(view.data_ptr(), nbytes, root);
        }
      });
}

c10::intrusive_ptr<Work> ProcessGroupTCCL::allreduce_coalesced(
    std::vector<at::Tensor>& /*tensors*/,
    const AllreduceCoalescedOptions& /*opts*/) {
  TORCH_CHECK(
      false, "ProcessGroupTCCL::allreduce_coalesced ", kNotImplementedHint);
}

c10::intrusive_ptr<Work> ProcessGroupTCCL::reduce(
    std::vector<at::Tensor>& tensors,
    const ReduceOptions& opts) {
  TORCH_CHECK_WITH(
      DistBackendError, tensors.size() == 1,
      "TCCL reduce: expected exactly one tensor, got ", tensors.size());
  auto& tensor = tensors[0];
  TORCH_CHECK_WITH(
      DistBackendError, tensor.device().is_mps(),
      "TCCL reduce: tensor must be on MPS device, got ", tensor.device());
  TORCH_CHECK_WITH(
      DistBackendError, opts.rootRank >= 0 && opts.rootRank < size_,
      "TCCL reduce: invalid rootRank ", opts.rootRank, " (world size ", size_, ").");
  TORCH_CHECK_WITH(
      DistBackendError, isSupportedReduceDtype(tensor.scalar_type()),
      "TCCL reduce: unsupported dtype. Supported: float32/float16/bfloat16 and "
      "int8/int16/int32/int64/uint8/bool. ", kNotImplementedHint,
      " Got dtype=", tensor.scalar_type());
  TORCH_CHECK_WITH(
      DistBackendError,
      opts.reduceOp == ReduceOp::SUM || opts.reduceOp == ReduceOp::AVG ||
          opts.reduceOp == ReduceOp::MIN || opts.reduceOp == ReduceOp::MAX ||
          opts.reduceOp == ReduceOp::PRODUCT,
      "TCCL reduce: only SUM, AVG, MIN, MAX and PRODUCT are supported. ",
      kNotImplementedHint);
  TORCH_CHECK_WITH(
      DistBackendError,
      opts.reduceOp != ReduceOp::AVG || isFloatReduceDtype(tensor.scalar_type()),
      "TCCL reduce: AVG is only supported for float32/float16/bfloat16. Got dtype=",
      tensor.scalar_type());
  TORCH_CHECK_WITH(
      DistBackendError, tensor.is_contiguous(),
      "TCCL reduce: non-contiguous tensors are not supported. Call .contiguous().");
  // Runs as all-reduce - reduced value on every rank (reduce guarantees only the
  // root). rootRank is validated, not used in the reduction.
  const ReduceOp::RedOpType op = opts.reduceOp;
  const int worldSize = size_;
  const at::ScalarType st = tensor.scalar_type();
  return enqueueCollective(
      OpType::REDUCE,
      std::vector<at::Tensor>{tensor},
      [this, tensor, op, worldSize, st]() mutable {
        at::Tensor cpuView = mpsSharedCpuView(tensor);
        switch (st) {
          case at::kFloat:
            runMeshAllreduce<float>(*engine_, cpuView, op, worldSize);
            break;
          case at::kHalf:
            runMeshAllreduce<at::Half>(*engine_, cpuView, op, worldSize);
            break;
          case at::kBFloat16:
            runMeshAllreduce<at::BFloat16>(*engine_, cpuView, op, worldSize);
            break;
          case at::kChar:
            runMeshAllreduce<int8_t>(*engine_, cpuView, op, worldSize);
            break;
          case at::kShort:
            runMeshAllreduce<int16_t>(*engine_, cpuView, op, worldSize);
            break;
          case at::kInt:
            runMeshAllreduce<int32_t>(*engine_, cpuView, op, worldSize);
            break;
          case at::kLong:
            runMeshAllreduce<int64_t>(*engine_, cpuView, op, worldSize);
            break;
          case at::kByte:
            runMeshAllreduce<uint8_t>(*engine_, cpuView, op, worldSize);
            break;
          case at::kBool:
            runMeshAllreduce<bool>(*engine_, cpuView, op, worldSize);
            break;
          default:
            // Unreachable - validated above
            break;
        }
      });
}

// allgather (list form)

c10::intrusive_ptr<Work> ProcessGroupTCCL::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& /*opts*/) {
  TORCH_CHECK_WITH(
      DistBackendError,
      inputTensors.size() == 1,
      "TCCL allgather: expected exactly one input tensor, got ",
      inputTensors.size());
  TORCH_CHECK_WITH(
      DistBackendError,
      outputTensors.size() == 1,
      "TCCL allgather: expected exactly one output list, got ",
      outputTensors.size());
  auto& input = inputTensors[0];
  auto& outList = outputTensors[0];
  TORCH_CHECK_WITH(
      DistBackendError,
      static_cast<int>(outList.size()) == size_,
      "TCCL allgather: output list must have world_size (",
      size_,
      ") tensors, got ",
      outList.size());
  TORCH_CHECK_WITH(
      DistBackendError,
      input.device().is_mps() && input.is_contiguous(),
      "TCCL allgather: input must be a contiguous MPS tensor.");
  for (const auto& o : outList) {
    TORCH_CHECK_WITH(
        DistBackendError,
        o.device().is_mps() && o.is_contiguous() &&
            o.numel() == input.numel() &&
            o.scalar_type() == input.scalar_type(),
        "TCCL allgather: every output slot must be a contiguous MPS tensor "
        "matching the input shape and dtype.");
  }

  // Byte-copy collective - any dtype (DDP construction allgathers kLong
  // param-size vectors).
  return enqueueCollective(
      OpType::ALLGATHER,
      outList,
      [this, input, outList]() mutable {
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

        if (tcclUseRing(size_, /*autoEnable=*/false, /*isBf16=*/false,
                        per_rank_bytes * static_cast<std::size_t>(size_), 1,
                        engine_->ringTopology())) {
          engine_->ring_all_gather(inView.data_ptr(), outPtrs, per_rank_bytes);
        } else {
          engine_->all_gather(inView.data_ptr(), outPtrs, per_rank_bytes);
        }
      });
}

// _allgather_base

c10::intrusive_ptr<Work> ProcessGroupTCCL::_allgather_base(
    at::Tensor& outputBuffer,
    at::Tensor& inputBuffer,
    const AllgatherOptions& /*opts*/) {
  TORCH_CHECK_WITH(
      DistBackendError,
      inputBuffer.device().is_mps() && inputBuffer.is_contiguous() &&
          outputBuffer.device().is_mps() && outputBuffer.is_contiguous(),
      "TCCL _allgather_base: input and output must be contiguous MPS tensors.");
  TORCH_CHECK_WITH(
      DistBackendError,
      outputBuffer.scalar_type() == inputBuffer.scalar_type(),
      "TCCL _allgather_base: output/input dtype mismatch.");
  TORCH_CHECK_WITH(
      DistBackendError,
      outputBuffer.numel() == inputBuffer.numel() * size_,
      "TCCL _allgather_base: output numel (",
      outputBuffer.numel(),
      ") must equal input numel (",
      inputBuffer.numel(),
      ") * world_size (",
      size_,
      ").");

  // Byte-copy, any dtype. Output is one contiguous rank-major buffer.
  return enqueueCollective(
      OpType::_ALLGATHER_BASE,
      std::vector<at::Tensor>{outputBuffer},
      [this, inputBuffer, outputBuffer]() mutable {
        at::Tensor inView = mpsSharedCpuView(inputBuffer);
        at::Tensor outView = mpsSharedCpuView(outputBuffer);
        const std::size_t per_rank_bytes =
            static_cast<std::size_t>(inView.nbytes());

        std::vector<void*> outPtrs(static_cast<std::size_t>(size_));
        char* ob = reinterpret_cast<char*>(outView.data_ptr());
        for (int r = 0; r < size_; ++r) {
          outPtrs[r] = ob + static_cast<std::size_t>(r) * per_rank_bytes;
        }

        if (tcclUseRing(size_, /*autoEnable=*/false, /*isBf16=*/false,
                        per_rank_bytes * static_cast<std::size_t>(size_), 1,
                        engine_->ringTopology())) {
          engine_->ring_all_gather(inView.data_ptr(), outPtrs, per_rank_bytes);
        } else {
          engine_->all_gather(inView.data_ptr(), outPtrs, per_rank_bytes);
        }
      });
}

// allgather_into_tensor_coalesced (TP all-gather)

c10::intrusive_ptr<Work> ProcessGroupTCCL::allgather_into_tensor_coalesced(
    std::vector<at::Tensor>& outputs,
    std::vector<at::Tensor>& inputs,
    const AllgatherOptions& /*opts*/) {
  TORCH_CHECK_WITH(
      DistBackendError,
      outputs.size() == inputs.size(),
      "TCCL allgather_into_tensor_coalesced: outputs/inputs size mismatch (",
      outputs.size(),
      " vs ",
      inputs.size(),
      ").");
  for (size_t i = 0; i < inputs.size(); ++i) {
    TORCH_CHECK_WITH(
        DistBackendError,
        inputs[i].device().is_mps() && inputs[i].is_contiguous() &&
            outputs[i].device().is_mps() && outputs[i].is_contiguous() &&
            outputs[i].scalar_type() == inputs[i].scalar_type() &&
            outputs[i].numel() == inputs[i].numel() * size_,
        "TCCL allgather_into_tensor_coalesced: each (output, input) pair must "
        "be contiguous MPS, same dtype, with output numel = input numel * "
        "world_size.");
  }

  return enqueueCollective(
      OpType::ALLGATHER_COALESCED,
      outputs,
      [this, inputs, outputs]() mutable {
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

          if (tcclUseRing(size_, /*autoEnable=*/false, /*isBf16=*/false,
                          per_rank_bytes * static_cast<std::size_t>(size_), 1,
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
  TORCH_CHECK_WITH(
      DistBackendError, inputTensors.size() == 1,
      "TCCL gather: expected exactly one input tensor, got ", inputTensors.size());
  auto& input = inputTensors[0];
  const int root = opts.rootRank;
  TORCH_CHECK_WITH(
      DistBackendError, root >= 0 && root < size_,
      "TCCL gather: invalid rootRank ", root, " (world size ", size_, ").");
  TORCH_CHECK_WITH(
      DistBackendError, input.device().is_mps() && input.is_contiguous(),
      "TCCL gather: input must be a contiguous MPS tensor.");
  // Root needs a QP to every rank - mesh-only (ring has no non-neighbor link).
  TORCH_CHECK_WITH(
      DistBackendError, !engine_->ringTopology(),
      "TCCL gather: not supported on a ring topology (the root needs a QP to "
      "every rank); use a full-mesh cluster.");
  const bool isRoot = (rank_ == root);
  std::vector<at::Tensor> outList;
  if (isRoot) {
    TORCH_CHECK_WITH(
        DistBackendError,
        outputTensors.size() == 1 &&
            static_cast<int>(outputTensors[0].size()) == size_,
        "TCCL gather: the root must supply one output list of world_size (",
        size_, ") tensors.");
    outList = outputTensors[0];
    for (const auto& o : outList) {
      TORCH_CHECK_WITH(
          DistBackendError,
          o.device().is_mps() && o.is_contiguous() &&
              o.numel() == input.numel() &&
              o.scalar_type() == input.scalar_type(),
          "TCCL gather: every output slot must be a contiguous MPS tensor "
          "matching the input shape and dtype.");
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
  TORCH_CHECK_WITH(
      DistBackendError, outputTensors.size() == 1,
      "TCCL scatter: expected exactly one output tensor, got ",
      outputTensors.size());
  auto& output = outputTensors[0];
  const int root = opts.rootRank;
  TORCH_CHECK_WITH(
      DistBackendError, root >= 0 && root < size_,
      "TCCL scatter: invalid rootRank ", root, " (world size ", size_, ").");
  TORCH_CHECK_WITH(
      DistBackendError, output.device().is_mps() && output.is_contiguous(),
      "TCCL scatter: output must be a contiguous MPS tensor.");
  TORCH_CHECK_WITH(
      DistBackendError, !engine_->ringTopology(),
      "TCCL scatter: not supported on a ring topology (the root needs a QP to "
      "every rank); use a full-mesh cluster.");
  const bool isRoot = (rank_ == root);
  std::vector<at::Tensor> inList;
  if (isRoot) {
    TORCH_CHECK_WITH(
        DistBackendError,
        inputTensors.size() == 1 &&
            static_cast<int>(inputTensors[0].size()) == size_,
        "TCCL scatter: the root must supply one input list of world_size (",
        size_, ") tensors.");
    inList = inputTensors[0];
    for (const auto& in : inList) {
      TORCH_CHECK_WITH(
          DistBackendError,
          in.device().is_mps() && in.is_contiguous() &&
              in.numel() == output.numel() &&
              in.scalar_type() == output.scalar_type(),
          "TCCL scatter: every input slot must be a contiguous MPS tensor "
          "matching the output shape and dtype.");
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
  TORCH_CHECK_WITH(
      DistBackendError,
      outputTensors.size() == inputTensors.size(),
      "TCCL reduce_scatter: outputs/inputs size mismatch (",
      outputTensors.size(),
      " vs ",
      inputTensors.size(),
      ").");
  TORCH_CHECK_WITH(
      DistBackendError,
      opts.reduceOp == ReduceOp::SUM || opts.reduceOp == ReduceOp::AVG ||
          opts.reduceOp == ReduceOp::MIN || opts.reduceOp == ReduceOp::MAX ||
          opts.reduceOp == ReduceOp::PRODUCT,
      "TCCL reduce_scatter: only SUM, AVG, MIN, MAX and PRODUCT are supported. ",
      kNotImplementedHint);
  for (size_t i = 0; i < outputTensors.size(); ++i) {
    auto& out = outputTensors[i];
    auto& inList = inputTensors[i];
    TORCH_CHECK_WITH(
        DistBackendError,
        static_cast<int>(inList.size()) == size_,
        "TCCL reduce_scatter: input list ",
        i,
        " must have world_size (",
        size_,
        ") tensors, got ",
        inList.size());
    TORCH_CHECK_WITH(
        DistBackendError,
        out.device().is_mps() && out.is_contiguous() &&
            isSupportedReduceDtype(out.scalar_type()),
        "TCCL reduce_scatter: each output must be a contiguous MPS tensor of a "
        "supported dtype (float32/float16/bfloat16, int8/16/32/64, uint8, bool). ",
        kNotImplementedHint);
    TORCH_CHECK_WITH(
        DistBackendError,
        opts.reduceOp != ReduceOp::AVG || isFloatReduceDtype(out.scalar_type()),
        "TCCL reduce_scatter: AVG is only supported for float32/float16/bfloat16 "
        "(integer/bool AVG is undefined; matches NCCL). Use SUM for integer "
        "dtypes.");
    for (const auto& chunk : inList) {
      TORCH_CHECK_WITH(
          DistBackendError,
          chunk.device().is_mps() && chunk.is_contiguous() &&
              chunk.scalar_type() == out.scalar_type() &&
              chunk.numel() == out.numel(),
          "TCCL reduce_scatter: every input chunk must be a contiguous MPS "
          "tensor matching the output dtype and numel.");
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
          dispatchReduceScatterList(*engine_, inChunkViews, outView, op, worldSize);
        }
      });
}

// _reduce_scatter_base (FSDP reduce_scatter_tensor)

c10::intrusive_ptr<Work> ProcessGroupTCCL::_reduce_scatter_base(
    at::Tensor& outputBuffer,
    at::Tensor& inputBuffer,
    const ReduceScatterOptions& opts) {
  TORCH_CHECK_WITH(
      DistBackendError,
      opts.reduceOp == ReduceOp::SUM || opts.reduceOp == ReduceOp::AVG ||
          opts.reduceOp == ReduceOp::MIN || opts.reduceOp == ReduceOp::MAX ||
          opts.reduceOp == ReduceOp::PRODUCT,
      "TCCL _reduce_scatter_base: only SUM, AVG, MIN, MAX and PRODUCT are supported. ",
      kNotImplementedHint);
  TORCH_CHECK_WITH(
      DistBackendError,
      inputBuffer.device().is_mps() && inputBuffer.is_contiguous() &&
          outputBuffer.device().is_mps() && outputBuffer.is_contiguous() &&
          isSupportedReduceDtype(inputBuffer.scalar_type()) &&
          outputBuffer.scalar_type() == inputBuffer.scalar_type(),
      "TCCL _reduce_scatter_base: unsupported dtype or layout. Supported: "
      "float32/float16/bfloat16 and int8/int16/int32/int64/uint8/bool (SUM); "
      "in/out dtype must match; contiguous MPS. ",
      kNotImplementedHint);
  TORCH_CHECK_WITH(
      DistBackendError,
      opts.reduceOp != ReduceOp::AVG ||
          isFloatReduceDtype(inputBuffer.scalar_type()),
      "TCCL _reduce_scatter_base: AVG is only supported for "
      "float32/float16/bfloat16 (integer/bool AVG is undefined; matches NCCL). "
      "Use SUM for integer dtypes.");
  TORCH_CHECK_WITH(
      DistBackendError,
      inputBuffer.numel() == outputBuffer.numel() * size_,
      "TCCL _reduce_scatter_base: input numel (",
      inputBuffer.numel(),
      ") must equal output numel (",
      outputBuffer.numel(),
      ") * world_size (",
      size_,
      ").");

  const ReduceOp::RedOpType op = opts.reduceOp;
  const int worldSize = size_;
  return enqueueCollective(
      OpType::_REDUCE_SCATTER_BASE,
      std::vector<at::Tensor>{outputBuffer},
      [this, inputBuffer, outputBuffer, op, worldSize]() mutable {
        at::Tensor inView = mpsSharedCpuView(inputBuffer);
        at::Tensor outView = mpsSharedCpuView(outputBuffer);
        dispatchReduceScatter(*engine_, inView, outView, op, worldSize);
      });
}

// reduce_scatter_tensor_coalesced (TP / sequence-parallel)

c10::intrusive_ptr<Work> ProcessGroupTCCL::reduce_scatter_tensor_coalesced(
    std::vector<at::Tensor>& outputs,
    std::vector<at::Tensor>& inputs,
    const ReduceScatterOptions& opts) {
  TORCH_CHECK_WITH(
      DistBackendError,
      outputs.size() == inputs.size(),
      "TCCL reduce_scatter_tensor_coalesced: outputs/inputs size mismatch (",
      outputs.size(),
      " vs ",
      inputs.size(),
      ").");
  TORCH_CHECK_WITH(
      DistBackendError,
      opts.reduceOp == ReduceOp::SUM || opts.reduceOp == ReduceOp::AVG ||
          opts.reduceOp == ReduceOp::MIN || opts.reduceOp == ReduceOp::MAX ||
          opts.reduceOp == ReduceOp::PRODUCT,
      "TCCL reduce_scatter_tensor_coalesced: only SUM, AVG, MIN, MAX and "
      "PRODUCT are supported. ",
      kNotImplementedHint);
  for (size_t i = 0; i < inputs.size(); ++i) {
    TORCH_CHECK_WITH(
        DistBackendError,
        inputs[i].device().is_mps() && inputs[i].is_contiguous() &&
            outputs[i].device().is_mps() && outputs[i].is_contiguous() &&
            isSupportedReduceDtype(inputs[i].scalar_type()) &&
            outputs[i].scalar_type() == inputs[i].scalar_type(),
        "TCCL reduce_scatter_tensor_coalesced: unsupported dtype. Supported: "
        "float32/float16/bfloat16 and int8/int16/int32/int64/uint8/bool (SUM); "
        "in/out dtype must match; contiguous MPS. ",
        kNotImplementedHint);
    TORCH_CHECK_WITH(
        DistBackendError,
        opts.reduceOp != ReduceOp::AVG ||
            isFloatReduceDtype(inputs[i].scalar_type()),
        "TCCL reduce_scatter_tensor_coalesced: AVG is only supported for "
        "float32/float16/bfloat16 (integer/bool AVG is undefined; matches NCCL). "
        "Use SUM for integer dtypes.");
    TORCH_CHECK_WITH(
        DistBackendError,
        inputs[i].numel() == outputs[i].numel() * size_,
        "TCCL reduce_scatter_tensor_coalesced: input numel (",
        inputs[i].numel(),
        ") must equal output numel (",
        outputs[i].numel(),
        ") * world_size (",
        size_,
        ").");
  }

  const ReduceOp::RedOpType op = opts.reduceOp;
  const int worldSize = size_;
  return enqueueCollective(
      OpType::REDUCE_SCATTER_TENSOR_COALESCED,
      outputs,
      [this, inputs, outputs, op, worldSize]() mutable {
        for (size_t i = 0; i < inputs.size(); ++i) {
          at::Tensor inView = mpsSharedCpuView(inputs[i]);
          at::Tensor outView = mpsSharedCpuView(outputs[i]);
          dispatchReduceScatter(*engine_, inView, outView, op, worldSize);
        }
      });
}

c10::intrusive_ptr<Work> ProcessGroupTCCL::alltoall_base(
    at::Tensor& outputBuffer,
    at::Tensor& inputBuffer,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const AllToAllOptions& /*opts*/) {
  TORCH_CHECK_WITH(
      DistBackendError,
      inputBuffer.device().is_mps() && inputBuffer.is_contiguous() &&
          outputBuffer.device().is_mps() && outputBuffer.is_contiguous() &&
          inputBuffer.dim() >= 1 && outputBuffer.dim() >= 1,
      "TCCL alltoall_base: input/output must be contiguous MPS tensors with dim >= 1.");
  TORCH_CHECK_WITH(
      DistBackendError,
      outputBuffer.scalar_type() == inputBuffer.scalar_type(),
      "TCCL alltoall_base: input/output dtype mismatch.");

  const int ws = size_;
  // Split is along dim 0; a "row" is the trailing-dims slab. in/out row bytes
  // match (same dtype + trailing dims for a valid all-to-all).
  const int64_t inDim0 = inputBuffer.size(0);
  const int64_t outDim0 = outputBuffer.size(0);
  const std::size_t inRow =
      inDim0 > 0 ? static_cast<std::size_t>(inputBuffer.nbytes()) / inDim0 : 0;
  const std::size_t outRow =
      outDim0 > 0 ? static_cast<std::size_t>(outputBuffer.nbytes()) / outDim0 : 0;

  const bool equal = inputSplitSizes.empty() && outputSplitSizes.empty();
  std::vector<std::size_t> sOff(ws), sBytes(ws), rOff(ws), rBytes(ws);
  if (equal) {
    TORCH_CHECK_WITH(
        DistBackendError,
        inDim0 % ws == 0 && outDim0 % ws == 0,
        "TCCL alltoall_base: equal split requires dim-0 (", inDim0, "/", outDim0,
        ") divisible by world_size ", ws, ".");
    const std::size_t sb = static_cast<std::size_t>(inDim0 / ws) * inRow;
    const std::size_t rb = static_cast<std::size_t>(outDim0 / ws) * outRow;
    for (int p = 0; p < ws; ++p) {
      sOff[p] = static_cast<std::size_t>(p) * sb;
      sBytes[p] = sb;
      rOff[p] = static_cast<std::size_t>(p) * rb;
      rBytes[p] = rb;
    }
  } else {
    TORCH_CHECK_WITH(
        DistBackendError,
        static_cast<int>(inputSplitSizes.size()) == ws &&
            static_cast<int>(outputSplitSizes.size()) == ws,
        "TCCL alltoall_base: split-size vectors must have world_size entries.");
    std::size_t so = 0, ro = 0;
    for (int p = 0; p < ws; ++p) {
      sOff[p] = so;
      sBytes[p] = static_cast<std::size_t>(inputSplitSizes[p]) * inRow;
      so += sBytes[p];
      rOff[p] = ro;
      rBytes[p] = static_cast<std::size_t>(outputSplitSizes[p]) * outRow;
      ro += rBytes[p];
    }
    TORCH_CHECK_WITH(
        DistBackendError,
        so == static_cast<std::size_t>(inputBuffer.nbytes()) &&
            ro == static_cast<std::size_t>(outputBuffer.nbytes()),
        "TCCL alltoall_base: split sizes do not sum to the buffer sizes.");
  }

  // Ring requires equal (uniform) segments + ws > 2; tcclUseRing only returns
  // true under TCCL_FORCE_ALGO=ring or ring topology. Everything else (the
  // default, and all uneven splits) uses the mesh/direct form, which handles
  // variable per-peer sizes.
  const bool ringTop = engine_->ringTopology();
  TORCH_CHECK_WITH(
      DistBackendError,
      !ringTop || (equal && sBytes[0] == rBytes[0]),
      "TCCL alltoall_base: ring topology supports only uniform (equal-split) "
      "all-to-all; uneven splits require the full mesh (non-neighbor peers are "
      "not connected on a ring).");
  const bool ringOk = equal && sBytes[0] == rBytes[0] &&
      tcclUseRing(ws, /*autoEnable=*/false, /*isBf16=*/false,
                  sBytes[0] * static_cast<std::size_t>(ws), 1, ringTop);
  const std::size_t segBytes = equal ? sBytes[0] : 0;

  return enqueueCollective(
      OpType::ALLTOALL_BASE,
      std::vector<at::Tensor>{outputBuffer},
      [this, inputBuffer, outputBuffer, sOff, sBytes, rOff, rBytes, ringOk,
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

c10::intrusive_ptr<Work> ProcessGroupTCCL::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int /*tag*/) {
  TORCH_CHECK_WITH(
      DistBackendError,
      tensors.size() == 1,
      "TCCL send: expected exactly one tensor, got ", tensors.size());
  auto& tensor = tensors[0];
  TORCH_CHECK_WITH(
      DistBackendError,
      tensor.device().is_mps() && tensor.is_contiguous(),
      "TCCL send: tensor must be a contiguous MPS tensor.");
  TORCH_CHECK_WITH(
      DistBackendError,
      dstRank >= 0 && dstRank < size_ && dstRank != rank_,
      "TCCL send: invalid dstRank ", dstRank, " for rank ", rank_,
      " (world size ", size_, ").");
  // Ring topology only connects the two neighbors ((rank+/-1)%world); a send to
  // any other rank has a null connection slot.
  TORCH_CHECK_WITH(
      DistBackendError,
      !engine_->ringTopology() ||
          dstRank == (rank_ + 1) % size_ ||
          dstRank == (rank_ - 1 + size_) % size_,
      "TCCL send: ring topology only permits sends to a ring neighbor "
      "((rank+/-1)%world); dstRank ", dstRank, " is not a neighbor of rank ",
      rank_, ".");
  // Byte movement - any dtype. tag is accepted but not hardware-matched (UC has
  // no tag matching); ordering is the per-peer FIFO of matched send/recv pairs.
  return enqueueCollective(
      OpType::SEND,
      std::vector<at::Tensor>{},
      [this, tensor, dstRank]() mutable {
        at::Tensor view = mpsSharedCpuView(tensor);
        engine_->p2p_send(
            dstRank,
            reinterpret_cast<const char*>(view.data_ptr()),
            static_cast<std::size_t>(view.nbytes()));
      });
}

c10::intrusive_ptr<Work> ProcessGroupTCCL::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int /*tag*/) {
  TORCH_CHECK_WITH(
      DistBackendError,
      tensors.size() == 1,
      "TCCL recv: expected exactly one tensor, got ", tensors.size());
  auto& tensor = tensors[0];
  TORCH_CHECK_WITH(
      DistBackendError,
      tensor.device().is_mps() && tensor.is_contiguous(),
      "TCCL recv: tensor must be a contiguous MPS tensor.");
  TORCH_CHECK_WITH(
      DistBackendError,
      srcRank >= 0 && srcRank < size_ && srcRank != rank_,
      "TCCL recv: invalid srcRank ", srcRank, " for rank ", rank_,
      " (world size ", size_, ").");
  // Ring topology only connects the two neighbors ((rank+/-1)%world); a recv from
  // any other rank has a null connection slot.
  TORCH_CHECK_WITH(
      DistBackendError,
      !engine_->ringTopology() ||
          srcRank == (rank_ + 1) % size_ ||
          srcRank == (rank_ - 1 + size_) % size_,
      "TCCL recv: ring topology only permits recvs from a ring neighbor "
      "((rank+/-1)%world); srcRank ", srcRank, " is not a neighbor of rank ",
      rank_, ".");
  return enqueueCollective(
      OpType::RECV,
      std::vector<at::Tensor>{tensor},
      [this, tensor, srcRank]() mutable {
        at::Tensor view = mpsSharedCpuView(tensor);
        engine_->p2p_recv(
            srcRank,
            reinterpret_cast<char*>(view.data_ptr()),
            static_cast<std::size_t>(view.nbytes()));
      });
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

} // namespace c10d

#endif // USE_C10D_TCCL
