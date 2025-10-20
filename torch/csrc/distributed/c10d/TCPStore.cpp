#include <c10/util/WaitCounter.h>
#include <c10/util/irange.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <torch/csrc/distributed/c10d/Backoff.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#include <torch/csrc/distributed/c10d/TCPStoreBackend.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/csrc/distributed/c10d/logging.h>

#include <chrono>
#include <fstream>
#include <optional>
#include <thread>
#include <unordered_map>
#include <utility>

namespace c10d {
namespace detail {

// Manages the lifecycle of a server daemon.
class TCPServer {
 public:
  static std::shared_ptr<TCPServer> start(const TCPStoreOptions& opts);

  std::uint16_t port() const noexcept {
    return port_;
  }

  explicit TCPServer(
      std::uint16_t port,
      std::unique_ptr<BackgroundThread>&& daemon)
      : port_{port}, daemon_{std::move(daemon)} {}

  std::string repr() const {
    return fmt::format("TCPServer(port={})", port_);
  }

 private:
  std::uint16_t port_;
  std::unique_ptr<BackgroundThread> daemon_;

  // We store weak references to all TCPServers for which the caller requested
  // multi-tenancy.
  static std::unordered_map<std::uint16_t, std::weak_ptr<TCPServer>>
      cachedServers_;

  static std::mutex cache_mutex_;
};

std::unordered_map<std::uint16_t, std::weak_ptr<TCPServer>>
    TCPServer::cachedServers_{};

std::mutex TCPServer::cache_mutex_{};

std::shared_ptr<TCPServer> TCPServer::start(const TCPStoreOptions& opts) {
  auto startCore = [&opts]() {
    auto daemon = opts.useLibUV ? create_libuv_tcpstore_backend(opts)
                                : create_tcpstore_backend(opts);
    daemon->start();
    return std::make_shared<TCPServer>(daemon->port(), std::move(daemon));
  };

  std::shared_ptr<TCPServer> server{};

  if (opts.multiTenant) {
    std::lock_guard<std::mutex> guard{cache_mutex_};

    // If the caller is okay with a multi-tenant store, first check if we
    // already have a TCPServer running on the specified port.
    if (opts.port > 0) {
      auto pos = cachedServers_.find(opts.port);
      if (pos != cachedServers_.end()) {
        server = pos->second.lock();
        if (server != nullptr) {
          return server;
        }

        // Looks like the TCPStore has been disposed, make sure that we release
        // the control block.
        cachedServers_.erase(pos);
      }
    }

    server = startCore();

    cachedServers_.emplace(server->port(), server);
  } else {
    server = startCore();
  }

  return server;
}

class TCPClient {
 public:
  static std::unique_ptr<TCPClient> connect(
      const SocketAddress& addr,
      const TCPStoreOptions& opts,
      std::shared_ptr<Backoff> backoff);

  void sendRaw(uint8_t* data, size_t length) {
    try {
      tcputil::sendBytes(socket_.handle(), data, length);
    } catch (const std::exception& e) {
      C10D_WARNING("sendBytes failed on {}: {}", socket_.repr(), e.what());
      throw;
    }
  }

  std::vector<std::uint8_t> receiveBits() {
    try {
      return tcputil::recvVector<std::uint8_t>(socket_.handle());
    } catch (const std::exception& e) {
      C10D_WARNING("recvVector failed on {}: {}", socket_.repr(), e.what());
      throw;
    }
  }

  template <typename T>
  T receiveValue() {
    try {
      return tcputil::recvValue<T>(socket_.handle());
    } catch (const std::exception& e) {
      C10D_WARNING("recvValue failed on {}: {}", socket_.repr(), e.what());
      throw;
    }
  }
  template <typename T>
  std::optional<T> receiveValueWithTimeout(std::chrono::milliseconds timeout) {
    if (!socket_.waitForInput(timeout)) {
      return {};
    }

    try {
      return tcputil::recvValue<T>(socket_.handle());
    } catch (const std::exception& e) {
      C10D_WARNING(
          "recvValueWithTimeout failed on {}: {}", socket_.repr(), e.what());
      throw;
    }
  }
  void setTimeout(std::chrono::milliseconds value);

  explicit TCPClient(Socket&& socket) : socket_{std::move(socket)} {}

  std::string repr() const {
    return fmt::format("TCPClient({})", socket_.repr());
  }

 private:
  Socket socket_;
};

std::unique_ptr<TCPClient> TCPClient::connect(
    const SocketAddress& addr,
    const TCPStoreOptions& opts,
    std::shared_ptr<Backoff> backoff) {
  Socket socket = Socket::connect(
      addr.host,
      addr.port,
      SocketOptions{}
          .connect_timeout(opts.timeout)
          .connect_backoff(std::move(backoff)));

  return std::make_unique<TCPClient>(std::move(socket));
}

void TCPClient::setTimeout(std::chrono::milliseconds value) {
  if (value == std::chrono::milliseconds::zero()) {
    return;
  }

#ifdef _WIN32
  struct timeval timeoutTV = {
      static_cast<long>(value.count() / 1000),
      static_cast<long>((value.count() % 1000) * 1000)};
#else
  struct timeval timeoutTV = {
      .tv_sec = value.count() / 1000,
      .tv_usec = static_cast<suseconds_t>((value.count() % 1000) * 1000),
  };
#endif
  SYSCHECK_ERR_RETURN_NEG1(::setsockopt(
      socket_.handle(),
      SOL_SOCKET,
      SO_RCVTIMEO,
      reinterpret_cast<char*>(&timeoutTV),
      sizeof(timeoutTV)));
}

class SendBuffer {
  // ethernet mtu 1500 - 40 (ip v6 header) - 20 (tcp header)
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const size_t FLUSH_WATERMARK = 1440;
  std::vector<uint8_t> buffer;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  detail::TCPClient& client;

  void maybeFlush() {
    if (buffer.size() >= FLUSH_WATERMARK) {
      flush();
    }
  }

 public:
  SendBuffer(detail::TCPClient& client, detail::QueryType cmd)
      : client(client) {
    buffer.reserve(32); // enough for most commands
    buffer.push_back((uint8_t)cmd);
  }

  void appendString(const std::string& str) {
    appendValue<uint64_t>(str.size());
    buffer.insert(buffer.end(), str.begin(), str.end());
    maybeFlush();
  }

  void appendBytes(const std::vector<uint8_t>& vec) {
    appendValue<uint64_t>(vec.size());
    buffer.insert(buffer.end(), vec.begin(), vec.end());
    maybeFlush();
  }

  template <typename T>
  void appendValue(T value) {
    uint8_t* begin = (uint8_t*)&value;
    buffer.insert(buffer.end(), begin, begin + sizeof(T));
    maybeFlush();
  }

  void flush() {
    if (!buffer.empty()) {
      client.sendRaw(buffer.data(), buffer.size());
      buffer.clear();
    }
  }
};

} // namespace detail

using detail::Socket;

// TCPStore class methods

// Although we still allow multi-params in ctor in Python, that behavior is
// removed from cpp and we construct the opts implicitly for users in the pybind
// of TCPStore.
TCPStore::TCPStore(std::string host, const TCPStoreOptions& opts)
    : Store{opts.timeout},
      addr_{std::move(host)},
      numWorkers_{opts.numWorkers},
      usingLibUv_{opts.useLibUV} {
  STATIC_SCOPED_WAIT_COUNTER(pytorch.wait_counter.TCPStore__init);

  if (opts.useLibUV) {
    TORCH_CHECK_WITH(
        DistStoreError,
        ::c10d::detail::is_libuv_tcpstore_backend_available(),
        "use_libuv was requested but PyTorch was built without libuv support, run with USE_LIBUV=0 to disable it.");
  }

  Socket::initialize();

  addr_.port = opts.port;

  if (opts.isServer) {
    try {
      server_ = detail::TCPServer::start(opts);
      // server successfully started
      C10D_DEBUG("The server has started on port = {}.", server_->port());
      addr_.port = server_->port();
    } catch (const SocketError& e) {
      bool useAgentStore = getCvarBool({"TORCHELASTIC_USE_AGENT_STORE"}, false);
      int masterPort = getCvarInt({"MASTER_PORT"}, 0);
      if (useAgentStore && masterPort == opts.port) {
        C10D_ERROR(
            "The server socket on {} has failed to bind. "
            "TORCHELASTIC_USE_AGENT_STORE is enabled so ignoring the error.",
            opts.port);
      } else {
        throw;
      }
    }

    std::ifstream maxconnFile("/proc/sys/net/core/somaxconn");
    if (maxconnFile.good() && numWorkers_.has_value()) {
      try {
        std::string str(
            (std::istreambuf_iterator<char>(maxconnFile)),
            std::istreambuf_iterator<char>());
        std::size_t somaxconn = std::stoll(str);
        if (somaxconn < *numWorkers_) {
          C10D_WARNING(
              "Starting store with {} workers but somaxconn is {}."
              "This might cause instability during bootstrap, consider increasing it.",
              *numWorkers_,
              somaxconn);
        }
      } catch (std::logic_error& e) {
        C10D_INFO("failed to parse somaxconn proc file due to {}", e.what());
      }
    }
  }

  // Try connecting several times -- if the server listen backlog is full it may
  // fail on the first send in validate.
  auto deadline = std::chrono::steady_clock::now() + opts.timeout;
  auto backoff = std::make_shared<ExponentialBackoffWithJitter>();

  auto retry = 0;
  do {
    try {
      client_ = detail::TCPClient::connect(addr_, opts, backoff);
      // TCP connection established
      C10D_DEBUG("TCP client connected to host {}:{}", addr_.host, addr_.port);

      // client's first query for validation
      validate();

      // ping to verify network connectivity
      ping();

      // success
      break;
    } catch (const c10::DistNetworkError& ex) {
      if (deadline < std::chrono::steady_clock::now()) {
        C10D_ERROR(
            "TCP client failed to connect/validate to host {}:{} - timed out (try={}, timeout={}ms): {}",
            addr_.host,
            addr_.port,
            retry,
            opts.timeout.count(),
            ex.what());
        throw;
      }

      auto delayDuration = backoff->nextBackoff();

      C10D_WARNING(
          "TCP client failed to connect/validate to host {}:{} - retrying (try={}, timeout={}ms, delay={}ms): {}",
          addr_.host,
          addr_.port,
          retry,
          opts.timeout.count(),
          delayDuration.count(),
          ex.what());

      std::this_thread::sleep_for(delayDuration);
      retry += 1;
    }
  } while (true);

  if (opts.waitWorkers) {
    waitForWorkers();
  }
}

TCPStore::~TCPStore() = default;

c10::intrusive_ptr<Store> TCPStore::clone() {
  TCPStoreOptions opts;
  opts.port = addr_.port;
  opts.isServer = false;
  opts.waitWorkers = false;
  opts.timeout = timeout_;
  opts.useLibUV = usingLibUv_;

  return c10::make_intrusive<TCPStore>(addr_.host, opts);
}

void TCPStore::waitForWorkers() {
  STATIC_SCOPED_WAIT_COUNTER(pytorch.wait_counter.TCPStore__waitForWorkers);
  if (!numWorkers_.has_value()) {
    return;
  }

  incrementValueBy(initKey_, 1);

  // Let server block until all workers have completed, this ensures that
  // the server daemon thread is always running until the very end
  if (server_) {
    const auto start = std::chrono::steady_clock::now();
    while (true) {
      // TODO: Any chance to make this cleaner?
      std::vector<uint8_t> value = doGet(initKey_);
      auto buf = reinterpret_cast<const char*>(value.data());
      auto len = value.size();
      int numWorkersCompleted = std::stoi(std::string(buf, len));
      if (numWorkersCompleted >= static_cast<int>(*numWorkers_)) {
        break;
      }
      const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::steady_clock::now() - start);
      if (timeout_ != kNoTimeout && elapsed > timeout_) {
        C10_THROW_ERROR(
            DistStoreError,
            fmt::format(
                "Timed out after {} seconds waiting for clients. {}/{} clients joined.",
                elapsed.count(),
                numWorkersCompleted,
                *numWorkers_));
      }
      /* sleep override */
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
}

void TCPStore::validate() {
  const std::lock_guard<std::mutex> lock(activeOpLock_);
  detail::SendBuffer buffer(*client_, detail::QueryType::VALIDATE);
  buffer.appendValue<std::uint32_t>(c10d::detail::validationMagicNumber);
  buffer.flush();
}

void TCPStore::ping() {
  const std::lock_guard<std::mutex> lock(activeOpLock_);
  detail::SendBuffer buffer(*client_, detail::QueryType::PING);

  uint32_t nonce = getpid();
  buffer.appendValue<std::uint32_t>(nonce);
  buffer.flush();

  uint32_t returnedNonce = client_->receiveValue<std::uint32_t>();
  if (nonce != returnedNonce) {
    C10_THROW_ERROR(
        DistNetworkError,
        fmt::format(
            "Ping failed, invalid value returned from server. Expected: {}, Got: {}",
            nonce,
            returnedNonce));
  }
}

void TCPStore::_splitSet(
    const std::string& key,
    const std::vector<uint8_t>& data) {
  const std::lock_guard<std::mutex> lock(activeOpLock_);
  detail::SendBuffer buffer(*client_, detail::QueryType::SET);
  buffer.appendString(keyPrefix_ + key);
  buffer.flush();
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  buffer.appendBytes(data);
  buffer.flush();
}

void TCPStore::set(const std::string& key, const std::vector<uint8_t>& data) {
  STATIC_SCOPED_WAIT_COUNTER(pytorch.wait_counter.TCPStore__set);
  const std::lock_guard<std::mutex> lock(activeOpLock_);
  detail::SendBuffer buffer(*client_, detail::QueryType::SET);
  buffer.appendString(keyPrefix_ + key);
  buffer.appendBytes(data);
  buffer.flush();
}

std::vector<uint8_t> TCPStore::compareSet(
    const std::string& key,
    const std::vector<uint8_t>& expectedValue,
    const std::vector<uint8_t>& desiredValue) {
  STATIC_SCOPED_WAIT_COUNTER(pytorch.wait_counter.TCPStore__compareSet);
  const std::lock_guard<std::mutex> lock(activeOpLock_);
  detail::SendBuffer buffer(*client_, detail::QueryType::COMPARE_SET);
  buffer.appendString(keyPrefix_ + key);
  buffer.appendBytes(expectedValue);
  buffer.appendBytes(desiredValue);
  buffer.flush();

  return client_->receiveBits();
}

std::vector<uint8_t> TCPStore::get(const std::string& key) {
  STATIC_SCOPED_WAIT_COUNTER(pytorch.wait_counter.TCPStore__get);
  const std::lock_guard<std::mutex> lock(activeOpLock_);
  return doGet(keyPrefix_ + key);
}

std::vector<uint8_t> TCPStore::doGet(const std::string& key) {
  doWait(key, timeout_);
  detail::SendBuffer buffer(*client_, detail::QueryType::GET);
  buffer.appendString(key);
  buffer.flush();

  return client_->receiveBits();
}

int64_t TCPStore::add(const std::string& key, int64_t value) {
  STATIC_SCOPED_WAIT_COUNTER(pytorch.wait_counter.TCPStore__add);
  const std::lock_guard<std::mutex> lock(activeOpLock_);
  return incrementValueBy(keyPrefix_ + key, value);
}

bool TCPStore::deleteKey(const std::string& key) {
  STATIC_SCOPED_WAIT_COUNTER(pytorch.wait_counter.TCPStore__delete);
  const std::lock_guard<std::mutex> lock(activeOpLock_);
  detail::SendBuffer buffer(*client_, detail::QueryType::DELETE_KEY);
  buffer.appendString(keyPrefix_ + key);
  buffer.flush();

  auto numDeleted = client_->receiveValue<std::int64_t>();
  return numDeleted == 1;
}

int64_t TCPStore::incrementValueBy(const std::string& key, int64_t delta) {
  detail::SendBuffer buff(*client_, detail::QueryType::ADD);
  buff.appendString(key);
  buff.appendValue<std::int64_t>(delta);
  buff.flush();

  return client_->receiveValue<std::int64_t>();
}

int64_t TCPStore::getNumKeys() {
  const std::lock_guard<std::mutex> lock(activeOpLock_);
  detail::SendBuffer buffer(*client_, detail::QueryType::GETNUMKEYS);
  buffer.flush();

  return client_->receiveValue<std::int64_t>();
}

bool TCPStore::check(const std::vector<std::string>& keys) {
  STATIC_SCOPED_WAIT_COUNTER(pytorch.wait_counter.TCPStore__check);
  const std::lock_guard<std::mutex> lock(activeOpLock_);
  detail::SendBuffer buffer(*client_, detail::QueryType::CHECK);
  buffer.appendValue(keys.size());

  for (const std::string& key : keys) {
    buffer.appendString(keyPrefix_ + key);
  }
  buffer.flush();

  auto response = client_->receiveValue<detail::CheckResponseType>();
  if (response == detail::CheckResponseType::READY) {
    return true;
  }
  if (response == detail::CheckResponseType::NOT_READY) {
    return false;
  }
  TORCH_CHECK_WITH(
      DistStoreError, false, "ready or not_ready response expected");
}

void TCPStore::wait(const std::vector<std::string>& keys) {
  wait(keys, timeout_);
}

void TCPStore::wait(
    const std::vector<std::string>& keys,
    const std::chrono::milliseconds& timeout) {
  STATIC_SCOPED_WAIT_COUNTER(pytorch.wait_counter.TCPStore__wait);
  const std::lock_guard<std::mutex> lock(activeOpLock_);
  std::vector<std::string> prefixedKeys{};
  prefixedKeys.reserve(keys.size());
  for (const std::string& key : keys) {
    prefixedKeys.emplace_back(keyPrefix_ + key);
  }

  doWait(prefixedKeys, timeout);
}

void TCPStore::doWait(
    c10::ArrayRef<std::string> keys,
    std::chrono::milliseconds timeout) {
  {
    detail::SendBuffer buffer(*client_, detail::QueryType::WAIT);
    buffer.appendValue(keys.size());
    for (const std::string& key : keys) {
      buffer.appendString(key);
    }
    buffer.flush();
  }

  auto response_opt =
      client_->receiveValueWithTimeout<detail::WaitResponseType>(timeout);
  if (response_opt.has_value()) {
    if (response_opt != detail::WaitResponseType::STOP_WAITING) {
      TORCH_CHECK_WITH(
          DistStoreError, false, "Stop_waiting response is expected");
    }
    return;
  }
  // this is the cancel wait timeout, once here we expect the server to respond
  // in a timely fashion
  {
    detail::SendBuffer buffer(*client_, detail::QueryType::CANCEL_WAIT);
    buffer.flush();
  }

  auto response = client_->receiveValue<detail::WaitResponseType>();
  // this can happen if the server responds before we cancel, just ignore it
  if (response != detail::WaitResponseType::WAIT_CANCELED) {
    if (response != detail::WaitResponseType::STOP_WAITING) {
      TORCH_CHECK_WITH(
          DistStoreError, false, "Stop_waiting response is expected");
    }

    response = client_->receiveValue<detail::WaitResponseType>(); // ignore
    if (response != detail::WaitResponseType::WAIT_CANCELED) {
      TORCH_CHECK_WITH(
          DistStoreError, false, "wait_canceled response is expected");
    }
  }
  C10_THROW_ERROR(
      DistStoreError,
      fmt::format(
          "wait timeout after {}ms, keys: {}",
          timeout.count(),
          fmt::join(keys, ", ")));
}

void TCPStore::append(
    const std::string& key,
    const std::vector<uint8_t>& data) {
  STATIC_SCOPED_WAIT_COUNTER(pytorch.wait_counter.TCPStore__append);
  const std::lock_guard<std::mutex> lock(activeOpLock_);
  detail::SendBuffer buffer(*client_, detail::QueryType::APPEND);
  buffer.appendString(keyPrefix_ + key);
  buffer.appendBytes(data);
  buffer.flush();
}

std::vector<std::vector<uint8_t>> TCPStore::multiGet(
    const std::vector<std::string>& keys) {
  STATIC_SCOPED_WAIT_COUNTER(pytorch.wait_counter.TCPStore__multiGet);
  const std::lock_guard<std::mutex> lock(activeOpLock_);
  std::vector<std::string> prefixedKeys;
  prefixedKeys.reserve(keys.size());
  for (const std::string& key : keys) {
    prefixedKeys.emplace_back(keyPrefix_ + key);
  }
  doWait(prefixedKeys, timeout_);

  detail::SendBuffer buffer(*client_, detail::QueryType::MULTI_GET);
  buffer.appendValue(keys.size());
  for (auto& key : prefixedKeys) {
    buffer.appendString(key);
  }
  buffer.flush();

  std::vector<std::vector<uint8_t>> result;
  result.reserve(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    result.emplace_back(client_->receiveBits());
  }
  return result;
}

void TCPStore::multiSet(
    const std::vector<std::string>& keys,
    const std::vector<std::vector<uint8_t>>& values) {
  STATIC_SCOPED_WAIT_COUNTER(pytorch.wait_counter.TCPStore__multiSet);
  TORCH_CHECK_WITH(
      DistStoreError,
      keys.size() == values.size(),
      "multiSet keys and values vectors must be of same size");
  const std::lock_guard<std::mutex> lock(activeOpLock_);

  detail::SendBuffer buffer(*client_, detail::QueryType::MULTI_SET);
  buffer.appendValue<std::int64_t>(static_cast<int64_t>(keys.size()));
  for (auto i : c10::irange(keys.size())) {
    buffer.appendString(keyPrefix_ + keys[i]);
    buffer.appendBytes(values[i]);
  }
  buffer.flush();
}

void TCPStore::queuePush(
    const std::string& key,
    const std::vector<uint8_t>& data) {
  TORCH_CHECK_WITH(
      NotImplementedError,
      usingLibUv_,
      "queues not implemented on legacy TCPStore backend");

  STATIC_SCOPED_WAIT_COUNTER(pytorch.wait_counter.TCPStore__queuePush);

  const std::lock_guard<std::mutex> lock(activeOpLock_);

  detail::SendBuffer buffer(*client_, detail::QueryType::QUEUE_PUSH);
  buffer.appendString(keyPrefix_ + key);
  buffer.appendBytes(data);
  buffer.flush();
}

std::vector<uint8_t> TCPStore::queuePop(const std::string& key, bool block) {
  TORCH_CHECK_WITH(
      NotImplementedError,
      usingLibUv_,
      "queues not implemented on legacy TCPStore backend");

  STATIC_SCOPED_WAIT_COUNTER(pytorch.wait_counter.TCPStore__queuePop);

  const std::lock_guard<std::mutex> lock(activeOpLock_);

  if (block) {
    doWait(keyPrefix_ + key, timeout_);
  }

  detail::SendBuffer buffer(*client_, detail::QueryType::QUEUE_POP);
  buffer.appendString(keyPrefix_ + key);
  buffer.flush();

  auto keys = client_->receiveValue<int64_t>();
  TORCH_CHECK_WITH(DistQueueEmptyError, keys > 0, "queue is empty");

  return client_->receiveBits();
}

int64_t TCPStore::queueLen(const std::string& key) {
  TORCH_CHECK_WITH(
      NotImplementedError,
      usingLibUv_,
      "queues not implemented on legacy TCPStore backend");

  STATIC_SCOPED_WAIT_COUNTER(pytorch.wait_counter.TCPStore__queueLen);

  const std::lock_guard<std::mutex> lock(activeOpLock_);

  detail::SendBuffer buffer(*client_, detail::QueryType::QUEUE_LEN);
  buffer.appendString(keyPrefix_ + key);
  buffer.flush();

  return client_->receiveValue<int64_t>();
}

bool TCPStore::hasExtendedApi() const {
  return true;
}

std::string TCPStore::repr() const {
  auto clientRepr = client_ ? client_->repr() : "<nullptr>";
  auto serverRepr = server_ ? server_->repr() : "<nullptr>";
  return fmt::format("TCPStore(client={}, server={})", clientRepr, serverRepr);
}

} // namespace c10d
