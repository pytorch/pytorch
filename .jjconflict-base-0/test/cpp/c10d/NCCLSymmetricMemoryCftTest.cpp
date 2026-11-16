#include <gtest/gtest.h>

// Include nccl_dev_cap.hpp first to define NCCL_HAS_HOST_CFT
#include <torch/csrc/distributed/c10d/symm_mem/nccl_dev_cap.hpp>

#ifdef NCCL_HAS_HOST_CFT

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
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

#include <cstdlib>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace symm_mem = c10d::symmetric_memory;

namespace {

constexpr int64_t kNumel = 1024;
constexpr const char* kGroupName = "nccl_cft_test_group";

// Owns everything a rank needs alive for the duration of a test body: the
// symmetric allocation, its handle, and the process group backing both.
struct RankFixture {
  c10::intrusive_ptr<c10d::ProcessGroup> pg;
  at::Tensor tensor;
  c10::intrusive_ptr<symm_mem::SymmetricMemory> hdl;

  symm_mem::NCCLSymmetricMemory* nccl() const {
    return dynamic_cast<symm_mem::NCCLSymmetricMemory*>(hdl.get());
  }
};

// Build a process group whose communicator has host-side CFT turned on, then
// rendezvous a symmetric memory tensor with it. `host_cft_mode` is opt-in and
// has to be identical on every rank, otherwise NCCL fails the window
// registration underneath rendezvous.
RankFixture setUpRank(const std::string& path, int rank, int size) {
  at::Device device(at::kCUDA, static_cast<c10::DeviceIndex>(rank));
  c10::cuda::CUDAGuard guard(device);

  auto store = c10::make_intrusive<c10d::FileStore>(path, size);
  auto opts = c10::make_intrusive<c10d::ProcessGroupNCCL::Options>();
  opts->group_name = kGroupName;
  opts->config.hostCftMode = ncclHostCftFallback;
  auto backend =
      c10::make_intrusive<c10d::ProcessGroupNCCL>(store, rank, size, opts);

  // Symmetric memory resolves rank/world size through the group registry and
  // the communicator through NCCLDevCommManager, so the process group has to
  // be registered and eagerly connected before rendezvous.
  auto pg = c10::make_intrusive<c10d::ProcessGroup>(store, rank, size);
  pg->setDefaultBackend(c10d::ProcessGroup::BackendType::NCCL);
  pg->setBackend(
      at::kCUDA,
      c10d::ProcessGroup::BackendType::NCCL,
      c10::static_intrusive_pointer_cast<c10d::Backend>(backend));
  c10d::register_process_group(kGroupName, pg);
  backend->eagerConnectSingleDevice(device);

  auto tensor = symm_mem::empty_strided_p2p(
      {kNumel}, {1}, at::kFloat, device, std::nullopt, std::nullopt);
  RankFixture fixture{pg, tensor, symm_mem::rendezvous(tensor, kGroupName)};
  EXPECT_NE(fixture.nccl(), nullptr)
      << "expected the NCCL symmetric memory backend to be active";
  return fixture;
}

// Whether this system can actually serve host-side CFT queries. Querying our
// own rank is the only reliable probe: it fails both on hardware without CFT
// support and on a communicator built without host-side CFT, and neither is
// visible from the handle. The answer is a property of the hardware and the
// config, so it is the same on every rank -- no rank can skip while others
// proceed into a collective.
bool cftAvailable(symm_mem::NCCLSymmetricMemory* hdl) {
  try {
    (void)hdl->get_peer_cft_handle(hdl->get_rank());
    return true;
  } catch (const c10::Error& e) {
    LOG(WARNING) << "Skipping: host-side CFT unavailable: " << e.what();
    return false;
  }
}

// Unicast: one logical endpoint per peer over the same window.
void testUnicastCft(const std::string& path, int rank, int size) {
  auto fixture = setUpRank(path, rank, size);
  auto* hdl = fixture.nccl();
  if (hdl == nullptr || !cftAvailable(hdl)) {
    c10d::unregister_process_group(kGroupName);
    return;
  }

  const auto self = hdl->get_peer_cft_handle(rank);
  std::set<uint32_t> le_ids;
  for (int peer = 0; peer < size; ++peer) {
    const auto handle = hdl->get_peer_cft_handle(peer);
    EXPECT_TRUE(le_ids.insert(handle.le_id).second)
        << "peer " << peer << " reuses le_id " << handle.le_id;
    // Every rank maps the buffer at the same offset in the symmetric space, so
    // only the endpoint varies from peer to peer.
    EXPECT_EQ(handle.le_offset, self.le_offset);
  }

  EXPECT_THROW((void)hdl->get_peer_cft_handle(size), c10::Error);
  EXPECT_THROW((void)hdl->get_peer_cft_handle(-1), c10::Error);

  c10d::unregister_process_group(kGroupName);
}

// Multicast: the endpoint behind the device-side putMultimem / redMultimem.
void testMulticastCft(const std::string& path, int rank, int size) {
  auto fixture = setUpRank(path, rank, size);
  auto* hdl = fixture.nccl();
  if (hdl == nullptr || !cftAvailable(hdl)) {
    c10d::unregister_process_group(kGroupName);
    return;
  }

  const auto self = hdl->get_peer_cft_handle(rank);
  try {
    // Collective on first call unless the endpoint was created eagerly at
    // window registration, so every rank has to reach this.
    const auto mc = hdl->get_multimem_cft_handle();
    // The multicast endpoint is distinct from the unicast one, but it
    // addresses the same window, hence the same offset.
    EXPECT_EQ(mc.le_offset, self.le_offset);
  } catch (const c10::Error& e) {
    // NCCL disables CFT multicast when NVLS is unavailable. Uniform across
    // ranks, so nobody is left waiting in the collective above.
    LOG(WARNING) << "Skipping multicast CFT: " << e.what();
  }

  c10d::unregister_process_group(kGroupName);
}

using FuncType = void (*)(const std::string&, int, int);

class NCCLSymmetricMemoryCftTest : public ::testing::Test {
 protected:
  void SetUp() override {
    c10::initLogging();
    if (auto* sizeEnv = std::getenv("WORLD_SIZE")) {
      size_ = std::stoi(std::string(sizeEnv));
    }
    // Ranks share this process, so give each thread its own group registry.
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
    return false;
  }

  void multiThreadRun(FuncType testFunc) {
    c10d::test::TemporaryFile file;
    std::vector<std::thread> threads;
    threads.reserve(size_);
    for (const auto rank : c10::irange(size_)) {
      threads.emplace_back(testFunc, file.path, rank, size_);
    }
    for (auto& thread : threads) {
      thread.join();
    }
  }

  int size_{1};
};

TEST_F(NCCLSymmetricMemoryCftTest, testUnicastCft) {
  if (skipTest()) {
    return;
  }
  multiThreadRun(testUnicastCft);
}

TEST_F(NCCLSymmetricMemoryCftTest, testMulticastCft) {
  if (skipTest()) {
    return;
  }
  multiThreadRun(testMulticastCft);
}

} // namespace

#else // NCCL_HAS_HOST_CFT

TEST(NCCLSymmetricMemoryCftTest, unsupported) {
  GTEST_SKIP() << "Host-side CFT requires NCCL >= 2.31";
}

#endif // NCCL_HAS_HOST_CFT
