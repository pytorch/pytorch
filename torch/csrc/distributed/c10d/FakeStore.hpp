#pragma once

#include <torch/csrc/distributed/c10d/Store.hpp>

namespace c10d {

// A no-op Store for use with the fake process group. The fake backend performs
// no real communication, so the store is never used for rendezvous and every
// operation is a stub. Crucially clone() is a real C++ method (not a Python
// trampoline override), so it stays callable after Python drops its reference
// to the store -- e.g. when only the process group holds it -- which is what
// split_group relies on.
class FakeStore : public Store {
 public:
  c10::intrusive_ptr<Store> clone() override {
    return c10::make_intrusive<FakeStore>();
  }

  void set(
      const std::string& /* key */,
      const std::vector<uint8_t>& /* value */) override {}

  std::vector<uint8_t> get(const std::string& /* key */) override {
    return {};
  }

  int64_t add(const std::string& /* key */, int64_t value) override {
    return value;
  }

  bool deleteKey(const std::string& /* key */) override {
    return true;
  }

  bool check(const std::vector<std::string>& /* keys */) override {
    return true;
  }

  int64_t getNumKeys() override {
    return 0;
  }

  void wait(const std::vector<std::string>& /* keys */) override {}

  void wait(
      const std::vector<std::string>& /* keys */,
      const std::chrono::milliseconds& /* timeout */) override {}
};

} // namespace c10d
