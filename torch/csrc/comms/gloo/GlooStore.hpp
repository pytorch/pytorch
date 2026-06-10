// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <c10/util/intrusive_ptr.h>
#include <chrono>
#include <string>
#include <vector>

#include <gloo/rendezvous/store.h>
#include <torch/csrc/Export.h> // @manual=//caffe2:torch-cpp-cpu
#include <torch/csrc/distributed/c10d/Store.hpp> // @manual=//caffe2:torch-cpp

namespace torch::comms {

// Wrap c10d store as Gloo store
class TORCH_API GlooStore : public ::gloo::rendezvous::Store {
 public:
  explicit GlooStore(c10::intrusive_ptr<::c10d::Store> store)
      : store_(std::move(store)) {}

  void setUint(const std::string& key, const std::vector<uint8_t>& value) {
    store_->set(key, value);
  }

  void set(const std::string& key, const std::vector<char>& value) override {
    std::vector<uint8_t> tmp(value.begin(), value.end());
    store_->set(key, tmp);
  }

  std::vector<uint8_t> getUint(const std::string& key) {
    auto value = store_->get(key);
    return value;
  }

  std::vector<char> get(const std::string& key) override {
    auto value = store_->get(key);
    return std::vector<char>(value.begin(), value.end());
  }

  void wait(const std::vector<std::string>& keys) override {
    store_->wait(keys, ::c10d::Store::kDefaultTimeout);
  }

  void wait(
      const std::vector<std::string>& keys,
      const std::chrono::milliseconds& timeout) override {
    store_->wait(keys, timeout);
  }

  bool has_v2_support() override {
    return store_->hasExtendedApi();
  }

  std::vector<std::vector<char>> multi_get(
      const std::vector<std::string>& keys) override {
    std::vector<std::vector<char>> res;
    for (auto& value : store_->multiGet(keys)) {
      res.emplace_back(value.begin(), value.end());
    }
    return res;
  }

  void multi_set(
      const std::vector<std::string>& keys,
      const std::vector<std::vector<char>>& values) override {
    std::vector<std::vector<uint8_t>> u_values;
    u_values.reserve(values.size());
    for (auto& value : values) {
      u_values.emplace_back(value.begin(), value.end());
    }
    store_->multiSet(keys, u_values);
  }

  void append(const std::string& key, const std::vector<char>& value) override {
    std::vector<uint8_t> tmp(value.begin(), value.end());
    return store_->append(key, tmp);
  }

  int64_t add(const std::string& key, int64_t value) override {
    return store_->add(key, value);
  }

  const c10::intrusive_ptr<::c10d::Store>& _getStore() const {
    return store_;
  }

 protected:
  c10::intrusive_ptr<::c10d::Store> store_;
};

} // namespace torch::comms
