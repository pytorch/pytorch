#pragma once

#include <ATen/core/ivalue.h>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include <c10/macros/Macros.h>
#include <torch/custom_class.h>

namespace c10d {

using namespace std::chrono_literals;

class TORCH_API ControlCollectives : public torch::CustomClassHolder {
 public:
  virtual void barrier(
      const std::string& key,
      std::chrono::milliseconds timeout = 5min,
      bool block = true) = 0;

  virtual void broadcastSend(
      const std::string& key,
      const std::vector<uint8_t>& data,
      std::chrono::milliseconds timeout = 5min) = 0;
  virtual std::vector<uint8_t> broadcastRecv(
      const std::string& key,
      std::chrono::milliseconds timeout = 5min) = 0;

  virtual void gatherSend(
      const std::string& key,
      const std::vector<uint8_t>& data,
      std::chrono::milliseconds timeout = 5min) = 0;
  virtual std::vector<std::vector<uint8_t>> gatherRecv(
      const std::string& key,
      const std::vector<uint8_t>& data,
      std::chrono::milliseconds timeout = 5min) = 0;

  virtual std::vector<uint8_t> scatterSend(
      const std::string& key,
      const std::vector<std::vector<uint8_t>>& data,
      std::chrono::milliseconds timeout = 5min) = 0;
  virtual std::vector<uint8_t> scatterRecv(
      const std::string& key,
      std::chrono::milliseconds timeout = 5min) = 0;

  virtual std::vector<std::vector<uint8_t>> allGather(
      const std::string& key,
      const std::vector<uint8_t>& data,
      std::chrono::milliseconds timeout = 5min) = 0;

  virtual int64_t allSum(
      const std::string& key,
      int64_t data,
      std::chrono::milliseconds timeout = 5min) = 0;
};

} // namespace c10d
