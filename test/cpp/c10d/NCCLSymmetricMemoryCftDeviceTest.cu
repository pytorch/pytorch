#include <gtest/gtest.h>

// Include nccl_dev_cap.hpp first to define NCCL_HAS_HOST_CFT
#include <torch/csrc/distributed/c10d/symm_mem/nccl_dev_cap.hpp>

#ifdef NCCL_HAS_HOST_CFT

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Logging.h>
#include <c10/util/irange.h>
#include <torch/csrc/distributed/c10d/FileStore.hpp>
#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/NCCLSymmetricMemory.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>

#include "TestUtils.hpp"

#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace symm_mem = c10d::symmetric_memory;

namespace {

// One 256-byte slot per source rank. CFT transfers are 16-byte granular, so
// both the slot size and the per-source offset stay 16-byte aligned.
constexpr int kSlotWords = 64;
constexpr size_t kSlotBytes = kSlotWords * sizeof(uint32_t);
constexpr int kThreads = 64;
constexpr const char* kGroupName = "nccl_cft_device_test_group";

// Distinct payload per (src, dst) pair so a handle that addresses the wrong
// peer or the wrong offset shows up as a mismatch rather than passing by luck.
uint32_t valueBase(int src, int dst, int size) {
  return static_cast<uint32_t>((src * size + dst) * kSlotWords + 1);
}

// Push kSlotBytes into the logical endpoint `leId` at `leOffset`. Everything
// this kernel needs to reach peer memory is the (leId, leOffset) pair the host
// query returned -- no ncclDevComm is constructed anywhere in this test, which
// is the entire point of the host-side CFT API.
// NCCL's own NCCL_CFT_ENABLE cannot be used to gate this: cft__funcs.h ends
// with an unconditional `#undef NCCL_CFT_ENABLE`, so it always reads as 0 in
// consumer code and would silently compile the put away. Replicate NCCL's
// arch condition instead (nccl_device/impl/cft__funcs.h).
// Not `#define CFT_DEVICE_SUPPORTED (defined(...) && ...)`: `defined` inside
// a macro expansion is undefined behavior and clang rejects it under
// -Werror,-Wexpansion-to-defined.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && CUDART_VERSION >= 13030
#define CFT_DEVICE_SUPPORTED 1
#else
#define CFT_DEVICE_SUPPORTED 0
#endif

__global__ void cftPutKernel(uint32_t leId, size_t leOffset, uint32_t base) {
#if CFT_DEVICE_SUPPORTED
  __shared__ uint32_t payload[kSlotWords];
  __shared__ ncclCftSmem cftSmem;

  ncclCoopCta coop;
  ncclCft<ncclCoopCta> cft{coop, cftSmem};

  for (int w = threadIdx.x; w < kSlotWords; w += blockDim.x) {
    payload[w] = base + w;
  }
  // Publish the smem payload to the fabric proxy before handing it to CFT.
  ncclMemFence(
      coop,
      cuda::memory_order_release,
      ncclMemProxyType::Generic,
      ncclMemProxyType::Fabric,
      ncclMemFenceScope::Cta);

  cft.put(coop, leId, leOffset, payload, kSlotBytes);
  cft.submit(coop);
  // flushSmem waits only until the fabric engine has read `payload`; flush
  // waits for the transfer to land. Both are cheap here (one put, no buffer
  // reuse) and together they document the two distinct dependencies.
  cft.flushSmem(coop);
  cft.flush(coop);
#endif
}

// Rendezvous barrier for the rank threads. The ranks live in one process, so a
// plain host barrier is enough to order "everyone has put" against "everyone
// reads back"; there is no separate process to synchronize with.
class ThreadBarrier {
 public:
  explicit ThreadBarrier(int count) : threshold_(count), remaining_(count) {}

  void wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    const auto gen = generation_;
    if (--remaining_ == 0) {
      remaining_ = threshold_;
      ++generation_;
      cv_.notify_all();
    } else {
      cv_.wait(lock, [&] { return generation_ != gen; });
    }
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  int threshold_;
  int remaining_;
  uint64_t generation_{0};
};

struct RankFixture {
  c10::intrusive_ptr<c10d::ProcessGroup> pg;
  at::Tensor tensor;
  c10::intrusive_ptr<symm_mem::SymmetricMemory> hdl;

  symm_mem::NCCLSymmetricMemory* nccl() const {
    return dynamic_cast<symm_mem::NCCLSymmetricMemory*>(hdl.get());
  }
};

RankFixture setUpRank(const std::string& path, int rank, int size) {
  at::Device device(at::kCUDA, static_cast<c10::DeviceIndex>(rank));
  c10::cuda::CUDAGuard guard(device);

  auto store = c10::make_intrusive<c10d::FileStore>(path, size);
  auto opts = c10::make_intrusive<c10d::ProcessGroupNCCL::Options>();
  opts->group_name = kGroupName;
  opts->config.hostCftMode = ncclHostCftFallback;
  auto backend =
      c10::make_intrusive<c10d::ProcessGroupNCCL>(store, rank, size, opts);

  auto pg = c10::make_intrusive<c10d::ProcessGroup>(store, rank, size);
  pg->setDefaultBackend(c10d::ProcessGroup::BackendType::NCCL);
  pg->setBackend(
      at::kCUDA,
      c10d::ProcessGroup::BackendType::NCCL,
      c10::static_intrusive_pointer_cast<c10d::Backend>(backend));
  c10d::register_process_group(kGroupName, pg);
  backend->eagerConnectSingleDevice(device);

  // One slot per source rank, so every peer can write without colliding.
  auto tensor = symm_mem::empty_strided_p2p(
      {static_cast<int64_t>(size) * kSlotWords},
      {1},
      at::kInt,
      device,
      std::nullopt,
      std::nullopt);
  tensor.zero_();
  // Drain the zero-fill before rendezvous. It is enqueued on this rank's
  // stream, but peers write into this buffer through the fabric engine, which
  // that stream does not order against -- a peer's put becomes launchable as
  // soon as it clears rendezvous, so a still-pending zero_ could land on top
  // of the payload.
  AT_CUDA_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream(rank)));
  RankFixture fixture{pg, tensor, symm_mem::rendezvous(tensor, kGroupName)};
  EXPECT_NE(fixture.nccl(), nullptr)
      << "expected the NCCL symmetric memory backend to be active";
  return fixture;
}

// Querying our own rank is the only reliable probe for host-side CFT: it fails
// both on hardware without CFT and on a comm built without it, and neither is
// visible from the handle. The answer is uniform across ranks.
bool cftAvailable(symm_mem::NCCLSymmetricMemory* hdl) {
  try {
    (void)hdl->get_peer_cft_handle(hdl->get_rank());
    return true;
  } catch (const c10::Error& e) {
    LOG(WARNING) << "Skipping: host-side CFT unavailable: " << e.what();
    return false;
  }
}

// Each rank pushes its signature into every peer's slot `rank`, then checks
// that its own buffer holds exactly what the other ranks sent. This is what
// proves the (le_id, le_offset) pair actually addresses the peer's copy of the
// buffer -- the query returning success does not.
void testDevicePutCft(
    const std::string& path,
    int rank,
    int size,
    ThreadBarrier* barrier) {
  at::Device device(at::kCUDA, static_cast<c10::DeviceIndex>(rank));
  c10::cuda::CUDAGuard guard(device);

  auto fixture = setUpRank(path, rank, size);
  auto* hdl = fixture.nccl();
  if (hdl == nullptr || !cftAvailable(hdl)) {
    barrier->wait();
    c10d::unregister_process_group(kGroupName);
    return;
  }

  auto stream = at::cuda::getCurrentCUDAStream(rank);
  for (const auto peer : c10::irange(size)) {
    if (peer == rank) {
      continue;
    }
    const auto handle = hdl->get_peer_cft_handle(peer);
    cftPutKernel<<<1, kThreads, 0, stream>>>(
        handle.le_id,
        handle.le_offset + static_cast<size_t>(rank) * kSlotBytes,
        valueBase(rank, peer, size));
    C10_CUDA_CHECK(cudaGetLastError());
  }
  AT_CUDA_CHECK(cudaStreamSynchronize(stream));

  barrier->wait();

  const auto host = fixture.tensor.cpu();
  const auto* data = host.const_data_ptr<int32_t>();
  for (const auto src : c10::irange(size)) {
    for (const auto w : c10::irange(kSlotWords)) {
      const auto got = static_cast<uint32_t>(data[src * kSlotWords + w]);
      // Nobody writes into their own slot, so it must still read zero.
      const uint32_t want =
          src == rank ? 0u : valueBase(src, rank, size) + static_cast<uint32_t>(w);
      ASSERT_EQ(got, want) << "rank " << rank << " slot " << src << " word " << w;
    }
  }

  c10d::unregister_process_group(kGroupName);
}

class NCCLSymmetricMemoryCftDeviceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    c10::initLogging();
    if (auto* sizeEnv = std::getenv("WORLD_SIZE")) {
      size_ = std::stoi(std::string(sizeEnv));
    }
    c10d::set_thread_isolation_mode(true);
    symm_mem::set_backend("NCCL");
  }

  void TearDown() override {
    c10d::set_thread_isolation_mode(false);
  }

  bool skipTest() {
    if (!at::cuda::is_available()) {
      LOG(INFO) << "CUDA not available, skipping test";
      return true;
    }
    if (at::cuda::device_count() < size_) {
      LOG(INFO) << "Need " << size_ << " GPUs, skipping test";
      return true;
    }
    if (size_ < 2) {
      LOG(INFO) << "Need at least 2 ranks, skipping test";
      return true;
    }
    // cftPutKernel compiles to a no-op unless CFT_DEVICE_SUPPORTED held, and
    // the host-side queries can still succeed then (they only need the driver
    // and NCCL), so the readback would fail confusingly instead of skipping.
    // ptxVersion is the virtual arch the loaded kernel variant was compiled
    // for, i.e. its __CUDA_ARCH__/10; CUDART_VERSION is shared with the
    // device pass of this TU.
#if CUDART_VERSION >= 13030
    cudaFuncAttributes attr{};
    AT_CUDA_CHECK(cudaFuncGetAttributes(&attr, cftPutKernel));
    if (attr.ptxVersion >= 100) {
      return false;
    }
    LOG(INFO) << "cftPutKernel variant compiled for sm_" << attr.ptxVersion
              << " with CUDART " << CUDART_VERSION
              << ", needs sm_100+ and CUDA >= 13.3; skipping test";
#else
    LOG(INFO) << "cftPutKernel compiled with CUDART " << CUDART_VERSION
              << ", needs CUDA >= 13.3; skipping test";
#endif
    return true;
  }

  int size_{1};
};

TEST_F(NCCLSymmetricMemoryCftDeviceTest, testDevicePutCft) {
  if (skipTest()) {
    return;
  }
  // cftPutKernel compiles to a no-op unless CFT_DEVICE_SUPPORTED held, and
  // the host-side queries can still succeed then (they only need the driver
  // and NCCL), so the readback would fail confusingly instead of skipping.
  // ptxVersion is the virtual arch the loaded kernel variant was compiled
  // for, i.e. its __CUDA_ARCH__/10; CUDART_VERSION is shared with the device
  // pass of this TU.
#if CUDART_VERSION >= 13030
  cudaFuncAttributes attr{};
  AT_CUDA_CHECK(cudaFuncGetAttributes(&attr, cftPutKernel));
  const bool cftCompiled = attr.ptxVersion >= 100;
#else
  const bool cftCompiled = false;
#endif
  if (!cftCompiled) {
    GTEST_SKIP() << "cftPutKernel compiled without CFT device support "
                 << "(needs sm_100+ code and CUDA >= 13.3)";
  }
  c10d::test::TemporaryFile file;
  ThreadBarrier barrier(size_);
  std::vector<std::thread> threads;
  threads.reserve(size_);
  for (const auto rank : c10::irange(size_)) {
    threads.emplace_back(testDevicePutCft, file.path, rank, size_, &barrier);
  }
  for (auto& thread : threads) {
    thread.join();
  }
}

} // namespace

#else // NCCL_HAS_HOST_CFT

TEST(NCCLSymmetricMemoryCftDeviceTest, unsupported) {
  GTEST_SKIP() << "Host-side CFT requires NCCL >= 2.31";
}

#endif // NCCL_HAS_HOST_CFT
