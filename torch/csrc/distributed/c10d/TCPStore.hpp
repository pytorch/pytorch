#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include <torch/csrc/distributed/c10d/Store.hpp>

namespace c10d {
namespace detail {

// TCPStore is a key-value store used by PyTorch mainly for distributed
// rendezvous, but for other purposes as well. (e.g., a centralized storage for
// synchronization among different processes.)
//
// It is run via a classic client-server architecture, where the server runs
// a separate background thread (alternatively we call it daemon thread). The
// client and server communicate via TCP sockets.
//
// Currently we have two types of server backends:
// 1. TCPStoreBackend: a single thread to handle all incoming request
// synchronously.
// 2. LibUVTCPStoreBackend: an event-driven asynchronous stream processing that
// leverages libuv library (https://github.com/libuv/libuv) for better
// performance. And this backend now is recommended to users. (We set the
// default value of `useLibUV` inside `TCPStoreOptions` to true now, so users
// should get it by default).
//
// Code structure:
// ├── TCPStore client side API and server setup code:
// │   TCPStore.hpp/TCPStore.cpp
// ├── TCPStoreBackend server side API implementation code:
// │   TCPStoreBackend.hpp/TCPStoreBackend.cpp
// |   (actual class:`TCPStoreMasterDaemon`)
// ├── LibUVTCPStoreBackend
// │   TCPStoreLibUvBackend.cpp
// |   (actual class: `LibUVStoreDaemon`)

class TCPServer;

class TCPClient;

struct SocketAddress {
  std::string host{};
  std::uint16_t port{};
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

  ~TCPStore() override;

  c10::intrusive_ptr<Store> clone() override;

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

  void queuePush(const std::string& key, const std::vector<uint8_t>& value)
      override;

  std::vector<uint8_t> queuePop(const std::string& key, bool block) override;

  int64_t queueLen(const std::string& key) override;

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
  bool usingLibUv_ = true;
};

} // namespace c10d
