// Reconfigure (fault tolerance) implementation for ProcessGroupGloo. The rest
// of the backend lives in ProcessGroupGloo.cpp; this file holds the handle
// encoding and the reconfigure() entry point so the membership-change logic is
// isolated from the collective implementations.
#include <c10/util/Exception.h>
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>

#ifdef USE_C10D_GLOO

#include <torch/csrc/distributed/c10d/FlightRecorder.hpp>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>

#include <algorithm>
#include <unordered_set>
#include <variant>

namespace c10d {

namespace {

std::string getStoreAddress(const c10::intrusive_ptr<Store>& store) {
  auto* tcpStore = dynamic_cast<TCPStore*>(store.get());
  if (tcpStore == nullptr) {
    auto* prefixStore = dynamic_cast<PrefixStore*>(store.get());
    if (prefixStore != nullptr) {
      tcpStore = dynamic_cast<TCPStore*>(
          prefixStore->getUnderlyingNonPrefixStore().get());
    }
  }
  if (tcpStore == nullptr) {
    return "";
  }
  return c10::str(tcpStore->getHost(), ":", tcpStore->getPort());
}

struct GlooReconfigureHandle {
  int rank;
  int64_t uuid;
  uint64_t stableRank;
  std::string storeAddress;
};

GlooReconfigureHandle parseGlooReconfigureHandle(
    const ReconfigureHandle& handle) {
  auto first = handle.find(':');
  TORCH_CHECK(
      first != std::string::npos &&
          handle.substr(0, first) == GLOO_BACKEND_NAME,
      "Invalid Gloo reconfigure handle: ",
      handle);
  auto second = handle.find(':', first + 1);
  TORCH_CHECK(
      second != std::string::npos, "Invalid Gloo reconfigure handle: ", handle);
  auto third = handle.find(':', second + 1);
  TORCH_CHECK(
      third != std::string::npos, "Invalid Gloo reconfigure handle: ", handle);
  auto fourth = handle.find(':', third + 1);
  TORCH_CHECK(
      fourth != std::string::npos, "Invalid Gloo reconfigure handle: ", handle);
  return {
      .rank = std::stoi(handle.substr(first + 1, second - first - 1)),
      .uuid = std::stoll(handle.substr(second + 1, third - second - 1)),
      .stableRank = std::stoull(handle.substr(third + 1, fourth - third - 1)),
      .storeAddress = handle.substr(fourth + 1)};
}

std::vector<ReconfigureHandle> getOrderedReconfigureHandles(
    const ReconfigureOptions& opts) {
  std::vector<ReconfigureHandle> handles;
  std::visit(
      [&](const auto& inputHandles) {
        handles.assign(inputHandles.begin(), inputHandles.end());
      },
      opts.handles);
  if (std::holds_alternative<std::unordered_set<ReconfigureHandle>>(
          opts.handles)) {
    std::ranges::sort(handles);
  }
  TORCH_CHECK(!handles.empty(), "Reconfigure requires at least one handle");
  std::unordered_set<ReconfigureHandle> uniqueHandles(
      handles.begin(), handles.end());
  TORCH_CHECK(
      uniqueHandles.size() == handles.size(),
      "Reconfigure handles must be unique");
  for (const auto& handle : handles) {
    parseGlooReconfigureHandle(handle);
  }
  return handles;
}

c10::intrusive_ptr<Work> makeCompletedWork() {
  auto future = c10::make_intrusive<c10::ivalue::Future>(
      c10::ListType::create(c10::TensorType::get()), std::vector<at::Device>{});
  future->markCompleted(c10::IValue(std::vector<at::Tensor>()));
  return Work::create_from_future(future);
}

} // namespace

ReconfigureHandle ProcessGroupGloo::get_reconfigure_handle() const {
  const auto& ranks = groupRanks();
  uint64_t stableRank = rank_;
  if (rank_ >= 0 && static_cast<size_t>(rank_) < ranks.size()) {
    stableRank = ranks[rank_];
  }
  return c10::str(
      GLOO_BACKEND_NAME,
      ":",
      rank_,
      ":",
      reconfigureUuid_,
      ":",
      stableRank,
      ":",
      getStoreAddress(c10dStore_));
}

c10::intrusive_ptr<Work> ProcessGroupGloo::reconfigure(
    const ReconfigureOptions& opts) {
  auto handles = getOrderedReconfigureHandles(opts);
  std::vector<uint64_t> globalRanks;
  globalRanks.reserve(handles.size());
  for (const auto& handle : handles) {
    globalRanks.push_back(parseGlooReconfigureHandle(handle).stableRank);
  }
  auto localHandle = get_reconfigure_handle();
  auto localIt = std::ranges::find(handles, localHandle);
  TORCH_CHECK(
      localIt != handles.end(),
      "Local Gloo reconfigure handle is not part of the new communicator");

  auto newRank = static_cast<int>(std::distance(handles.begin(), localIt));
  auto newSize = static_cast<int>(handles.size());
  auto timeout = opts.timeout.value_or(options_->timeout);
  auto oldTimeout = options_->timeout;
  auto oldRank = rank_;
  auto oldSize = size_;
  auto oldInitialized = initialized_;
  auto oldDefaultRanks = defaultRanks_;
  auto prefixedStore = c10::make_intrusive<PrefixStore>(
      c10::str("reconfigure/", opts.uuid), c10dStore_);

  // The uuid namespaces this reconfigure's rendezvous keys; reusing it would
  // read stale rendezvous state. New rank 0 atomically claims the uuid: the
  // compareSet writes our handle only while "claimed" is unset, so a reused
  // uuid returns the prior claimant's handle, which differs from ours.
  if (newRank == 0) {
    auto claimedBy = prefixedStore->compareSet("claimed", "", localHandle);
    TORCH_CHECK(
        claimedBy == localHandle,
        "Gloo reconfigure uuid ",
        opts.uuid,
        " was already used; each reconfigure() requires a unique uuid");
  }

  rank_ = newRank;
  size_ = newSize;
  defaultRanks_ = std::move(globalRanks);
  collectiveCounter_ = 0;
  seq_ = 0;
  options_->timeout = timeout;
  try {
    connectContexts(rank_, size_, prefixedStore);
  } catch (...) {
    options_->timeout = oldTimeout;
    rank_ = oldRank;
    size_ = oldSize;
    initialized_ = oldInitialized;
    defaultRanks_ = std::move(oldDefaultRanks);
    throw;
  }
  options_->timeout = oldTimeout;
  for (auto& context : contexts_) {
    context->setTimeout(oldTimeout);
  }
  initialized_ = true;
  reconfigureUuid_ = opts.uuid;

  FlightRecorder<c10::Event>::get()->record_pg_ranks(
      std::make_tuple(pg_uid_, pg_desc_), groupRanks());
  return makeCompletedWork();
}

} // namespace c10d

#endif // USE_C10D_GLOO
