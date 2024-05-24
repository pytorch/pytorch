#ifdef USE_C10D_NCCL

#include <torch/csrc/distributed/c10d/HealthcheckNCCL.hpp>

#include <fmt/format.h>

#include <ATen/ATen.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/TensorOptions.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/distributed/c10d/Healthcheck.hpp>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/logging.h>
#include <ratio>

namespace {
int getDeviceIndex() {
  int deviceIndex = c10::cuda::current_device();
  if (deviceIndex < 0) {
    C10D_WARNING("got invalid device index, using 0");
    deviceIndex = 0;
  }
  return deviceIndex;
}
} // namespace

namespace c10d {

HealthcheckNCCL::HealthcheckNCCL(
    const c10::intrusive_ptr<::c10d::Store>& store,
    int rank,
    int worldSize,
    int localWorldSize,
    c10::optional<int> exitOnError,
    std::chrono::milliseconds interval,
    std::chrono::milliseconds timeout)
    : Healthcheck(exitOnError, interval, timeout),
      rank_(rank),
      worldSize_(worldSize),
      localWorldSize_(localWorldSize),
      deviceIndex_(getDeviceIndex()),
      store_(store) {
  if (worldSize % localWorldSize != 0) {
    throw std::runtime_error(
        "World size must be divisible by local world size");
  }
  if (rank >= worldSize) {
    throw std::runtime_error("Rank must be less than world size");
  }
  if (worldSize / localWorldSize < 2) {
    throw std::runtime_error("At least two hosts are required");
  }

  streams_.reserve(2);
  processGroups_.reserve(2);
}

void HealthcheckNCCL::setup(int side) {
  auto info =
      Healthcheck::calculateGroupInfo(side, rank_, worldSize_, localWorldSize_);

  auto group = std::get<0>(info);
  auto groupRank = std::get<1>(info);
  auto groupSize = std::get<2>(info);

  auto storePrefix = fmt::format("/healthcheck/{}/{}", side, group);
  auto store = c10::make_intrusive<PrefixStore>(storePrefix, store_);

  auto deviceIndex = deviceIndex_;
  if (deviceIndex < 0) {
    C10D_WARNING("got invalid device index, using 0");
    deviceIndex = 0;
  }
  C10D_WARNING(
      "setting device index to {}, class {}", deviceIndex, deviceIndex_);
  c10::cuda::set_device(deviceIndex);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  streams_.emplace_back(c10::cuda::getStreamFromExternal(stream, deviceIndex_));

  C10D_INFO(
      "Creating process group for side side={}, group={}, rank={}, size={}, store={}",
      side,
      group,
      groupRank,
      groupSize,
      storePrefix);

  processGroups_.emplace_back(
      c10::make_intrusive<ProcessGroupNCCL>(store, groupRank, groupSize));
}

void HealthcheckNCCL::runHealthcheck(int side) {
  auto device = deviceIndex_;
  c10::cuda::set_device(deviceIndex_);

  at::cuda::setCurrentCUDAStream(streams_.at(side));
  auto& pg = processGroups_.at(side);

  at::Tensor t = at::ones(
      {1},
      at::TensorOptions{}
          .device(at::Device(c10::DeviceType::CUDA, device))
          .dtype(at::kFloat));
  std::vector<at::Tensor> tensors{t};

  AllreduceOptions opts;
  opts.timeout = timeout_;
  auto work = pg->allreduce(tensors, opts);

  while (!work->isCompleted()) {
    waitFor(std::chrono::milliseconds(10));
    if (isShutdown()) {
      throw std::runtime_error("shutting down");
    }
  }
  work->wait();

  if (t.item().to<double>() != 2.0 * localWorldSize_) {
    throw std::runtime_error(
        "Health check all reduce returned invalid results");
  }
}

void HealthcheckNCCL::shutdown() {
  Healthcheck::shutdown();
  for (auto& group : processGroups_) {
    group->abort();
    group->shutdown();
  }
}

} // namespace c10d

#endif
