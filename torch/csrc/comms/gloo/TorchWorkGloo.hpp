// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/comms/TorchWork.hpp>
#include <torch/csrc/comms/utils/TracingGuard.hpp>

namespace torch::comms {

// Forward declaration
class TorchCommGloo;

class TorchWorkGloo : public TorchWork {
 public:
  TorchWorkGloo();
  ~TorchWorkGloo() override;

  // Delete copy and move operations
  TorchWorkGloo(const TorchWorkGloo&) = delete;
  TorchWorkGloo(TorchWorkGloo&&) = delete;
  TorchWorkGloo& operator=(const TorchWorkGloo&) = delete;
  TorchWorkGloo& operator=(TorchWorkGloo&&) = delete;

  // Override virtual functions from TorchWork
  void wait() override;

 protected:
  friend class TorchCommGloo;
  friend class TorchWorkGlooQueue;
};

} // namespace torch::comms
