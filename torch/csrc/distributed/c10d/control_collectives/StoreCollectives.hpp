#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/FbcodeMaps.h>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/control_collectives/ControlCollectives.hpp>

namespace c10d {

class TORCH_API StoreCollectives : public ControlCollectives {
 public:
  explicit StoreCollectives(
      c10::intrusive_ptr<Store> store,
      int rank,
      int worldSize);

  void barrier(
      const std::string& key,
      std::chrono::milliseconds timeout = 5min,
      bool block = true) override;

  void broadcastSend(
      const std::string& key,
      const std::vector<uint8_t>& data,
      std::chrono::milliseconds timeout = 5min) override;
  std::vector<uint8_t> broadcastRecv(
      const std::string& key,
      std::chrono::milliseconds timeout = 5min) override;

  void gatherSend(
      const std::string& key,
      const std::vector<uint8_t>& data,
      std::chrono::milliseconds timeout = 5min) override;
  std::vector<std::vector<uint8_t>> gatherRecv(
      const std::string& key,
      const std::vector<uint8_t>& data,
      std::chrono::milliseconds timeout = 5min) override;

  std::vector<uint8_t> scatterSend(
      const std::string& key,
      const std::vector<std::vector<uint8_t>>& data,
      std::chrono::milliseconds timeout = 5min) override;
  std::vector<uint8_t> scatterRecv(
      const std::string& key,
      std::chrono::milliseconds timeout = 5min) override;

  std::vector<std::vector<uint8_t>> allGather(
      const std::string& key,
      const std::vector<uint8_t>& data,
      std::chrono::milliseconds timeout = 5min) override;

  int64_t allSum(
      const std::string& key,
      int64_t data,
      std::chrono::milliseconds timeout = 5min) override;

 private:
  void enforceUnique(const std::string& key);

 private:
  c10::intrusive_ptr<Store> store_;
  int rank_;
  int worldSize_;

  c10::FastSet<std::string> seenKeys_{};
};

} // namespace c10d
