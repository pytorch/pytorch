#pragma once

#include <memory>
#include <vector>

#include "fbcollective/rendezvous/store.h"
#include "fbcollective/transport/device.h"
#include "fbcollective/transport/pair.h"

namespace fbcollective {

class Context {
 public:
  Context(int rank, int size);

  const int rank_;
  const int size_;

  void connectFullMesh(
      rendezvous::Store& store,
      std::shared_ptr<transport::Device>& dev);

  std::unique_ptr<transport::Pair>& getPair(int i) {
    return pairs_.at(i);
  }

 protected:
  std::vector<std::unique_ptr<transport::Pair>> pairs_;
};

} // namespace fbcollective
