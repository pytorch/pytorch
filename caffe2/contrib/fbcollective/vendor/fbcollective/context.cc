#include "fbcollective/context.h"

#include "fbcollective/common/logging.h"
#include "fbcollective/transport/address.h"

namespace fbcollective {

Context::Context(int rank, int size) : rank_(rank), size_(size) {
  FBC_ENFORCE_GE(rank, 0);
  FBC_ENFORCE_LT(rank, size);
  FBC_ENFORCE_GE(size, 2);
}

void Context::connectFullMesh(
    rendezvous::Store& store,
    std::shared_ptr<transport::Device>& dev) {
  std::vector<std::unique_ptr<transport::Pair>> pairs(size_);

  // Create pair to connect to every other node in the collective
  for (int i = 0; i < size_; i++) {
    if (i == rank_) {
      continue;
    }

    auto pair = dev->createPair();
    pairs[i] = std::move(pair);

    // Store address for pair for this rank
    std::ostringstream key;
    key << rank_ << "/" << i;
    store.set(key.str(), pairs[i]->address().bytes());
  }

  // Connect every pair
  for (int i = 0; i < size_; i++) {
    if (i == rank_) {
      continue;
    }

    // Wait for address of other side of this pair to become available
    std::ostringstream key;
    key << i << "/" << rank_;
    store.wait({key.str()});

    // Connect to other side of this pair
    auto addr = store.get(key.str());
    pairs[i]->connect(addr);
  }

  pairs_ = std::move(pairs);
}

} // namespace fbcollective
