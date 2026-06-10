// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <torch/csrc/comms/gloo/TorchWorkGloo.hpp>

#include <thread>

#include <torch/csrc/comms/gloo/TorchCommGloo.hpp>
#include <torch/csrc/comms/utils/Logging.hpp>

namespace torch::comms {

TorchWorkGloo::TorchWorkGloo() {
  setStatus(WorkStatus::COMPLETED);
}

TorchWorkGloo::~TorchWorkGloo() {
  TC_LOG(INFO, nullptr) << "TorchWorkGloo destroyed";
}

void TorchWorkGloo::wait() {
  runWaitPreHooks();
  runWaitPostHooks();
}

} // namespace torch::comms
