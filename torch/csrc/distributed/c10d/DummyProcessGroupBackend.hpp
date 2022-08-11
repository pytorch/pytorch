#pragma once

#include <c10d/Backend.hpp>
#include <c10d/Work.hpp>

namespace c10d {

class TORCH_API DummyProcessGroupBackend : public Backend {
 public:
  explicit DummyProcessGroupBackend(int rank, int size) : Backend(rank, size){};

  ~DummyProcessGroupBackend() {}

  c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const BroadcastOptions& opts = BroadcastOptions()) override;
};

} // namespace c10d
