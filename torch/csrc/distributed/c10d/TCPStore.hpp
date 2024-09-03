#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include <torch/csrc/distributed/c10d/Store.hpp>

namespace c10d {
namespace detail {

class TCPServer;

class TCPClient;

struct SocketAddress {
  std::string host{};
  std::uint16_t port{};
};

class Counter {
 public:
  void update(double val);
  std::unordered_map<std::string, double> observe() const;

  double mean() const noexcept {
    return mean_;
  }
  int64_t count() const noexcept {
    return count_;
  }
  double variance() const noexcept {
    return m2_ / static_cast<double>(count_);
  }
  double sample_variance() const noexcept {
    return m2_ / static_cast<double>(count_ - 1);
  }

 private:
  int64_t count_ = 0;
  double mean_ = 0;
  double m2_ = 0;
};

} // namespace detail

struct TCPStoreOptions {
  static constexpr std::uint16_t kDefaultPort = 29500;

  std::uint16_t port = kDefaultPort;
  bool isServer = false;
  std::optional<std::size_t> numWorkers = std::nullopt;
  bool waitWorkers = true;
  std::chrono::milliseconds timeout = Store::kDefaultTimeout;

  // A boolean value indicating whether multiple store instances can be
  // initialized with the same host:port pair.
  bool multiTenant = false;

  // If specified, and if isServer is true, the underlying TCPServer will take
  // over the bound socket associated to this fd. This option is useful to avoid
  // port assignment races in certain scenarios.
  std::optional<int> masterListenFd = std::nullopt;

  // A boolean value indicating whether to use the experimental libUV backend.
  bool useLibUV = true;
};

class TORCH_API TCPStore : public Store {
 public:
  static constexpr std::chrono::milliseconds kConnectRetryDelay{1000};

  explicit TCPStore(std::string host, const TCPStoreOptions& opts = {});

  [[deprecated("Use TCPStore(host, opts) instead.")]] explicit TCPStore(
      const std::string& masterAddr,
      std::uint16_t masterPort,
      std::optional<int> numWorkers = std::nullopt,
      bool isServer = false,
      const std::chrono::milliseconds& timeout = kDefaultTimeout,
      bool waitWorkers = true);

  ~TCPStore() override;

  void set(const std::string& key, const std::vector<uint8_t>& value) override;

  std::vector<uint8_t> compareSet(
      const std::string& key,
      const std::vector<uint8_t>& expectedValue,
      const std::vector<uint8_t>& desiredValue) override;

  std::vector<uint8_t> get(const std::string& key) override;

  int64_t add(const std::string& key, int64_t value) override;

  bool deleteKey(const std::string& key) override;

  bool check(const std::vector<std::string>& keys) override;

  int64_t getNumKeys() override;

  void wait(const std::vector<std::string>& keys) override;

  void wait(
      const std::vector<std::string>& keys,
      const std::chrono::milliseconds& timeout) override;

  void append(const std::string& key, const std::vector<uint8_t>& value)
      override;

  std::vector<std::vector<uint8_t>> multiGet(
      const std::vector<std::string>& keys) override;

  void multiSet(
      const std::vector<std::string>& keys,
      const std::vector<std::vector<uint8_t>>& values) override;

  bool hasExtendedApi() const override;

  // Waits for all workers to join.
  void waitForWorkers();

  // Returns the hostname used by the TCPStore.
  const std::string& getHost() const noexcept {
    return addr_.host;
  }

  // Returns the port used by the TCPStore.
  std::uint16_t getPort() const noexcept {
    return addr_.port;
  }

  std::unordered_map<std::string, std::unordered_map<std::string, double>>
  collectClientCounters() const noexcept;

  bool isLibUvBackend() const noexcept {
    return usingLibUv_;
  }

  // note(xilunwu): this function is only for internal testing
  void _splitSet(const std::string& key, const std::vector<uint8_t>& data);

  std::string repr() const;

 private:
  int64_t incrementValueBy(const std::string& key, int64_t delta);

  void ping();
  void validate();

  std::vector<uint8_t> doGet(const std::string& key);

  void doWait(
      c10::ArrayRef<std::string> keys,
      std::chrono::milliseconds timeout);

  detail::SocketAddress addr_;
  std::shared_ptr<detail::TCPServer> server_;
  std::unique_ptr<detail::TCPClient> client_;
  std::optional<std::size_t> numWorkers_;

  const std::string initKey_ = "init/";
  const std::string keyPrefix_ = "/";
  std::mutex activeOpLock_;
  std::unordered_map<std::string, detail::Counter> clientCounters_;
  bool usingLibUv_ = true;
};

} // namespace c10d
