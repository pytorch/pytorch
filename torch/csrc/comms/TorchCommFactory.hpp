// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include <ATen/ATen.h>
#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <torch/csrc/comms/TorchCommBackend.hpp>
#include <torch/csrc/comms/TorchCommOptions.hpp>

namespace torch::comms {

class TorchCommFactory {
 public:
  static TorchCommFactory& get();

  std::shared_ptr<TorchCommBackend> create_backend(
      const std::string& backend,
      at::Device device,
      const std::string& name,
      const CommOptions& options = CommOptions());

  void register_backend(
      const std::string& backend,
      const std::function<std::shared_ptr<TorchCommBackend>()>& factory);

  // Allocator factory methods
  std::shared_ptr<c10::Allocator> get_allocator(const std::string& backend);

  void register_allocator_factory(
      const std::string& backend,
      const std::function<std::shared_ptr<c10::Allocator>()>& factory);

  bool is_backend_registered(const std::string& backend) const;

 private:
  std::shared_ptr<TorchCommBackend> create_generic_backend(
      const std::string& backend);

  mutable std::mutex mutex_;
  std::unordered_map<
      std::string,
      std::function<std::shared_ptr<TorchCommBackend>()>>
      backends_;
  std::unordered_map<
      std::string,
      std::function<std::shared_ptr<c10::Allocator>()>>
      allocator_factories_;
};
} // namespace torch::comms
