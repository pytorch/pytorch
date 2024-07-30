#include <c10/util/irange.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <torch/csrc/distributed/c10d/Backoff.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#include <torch/csrc/distributed/c10d/TCPStoreBackend.hpp>
#include <torch/csrc/distributed/c10d/logging.h>

#include <fcntl.h>
#include <chrono>
#include <fstream>
#include <random>
#include <thread>
#include <unordered_map>
#include <utility>

#ifdef _WIN32
#include <io.h>
#include <winsock2.h>
#else
#include <poll.h>
#include <unistd.h>
#endif

#ifdef _WIN32
#include <torch/csrc/distributed/c10d/WinSockUtils.hpp>
#else
#include <torch/csrc/distributed/c10d/UnixSockUtils.hpp>
#endif

#include <torch/csrc/distributed/c10d/socket.h>

namespace c10d {
namespace detail {

class timing_guard {
  Counter& counter_;
  typedef std::chrono::time_point<std::chrono::high_resolution_clock>
      time_point;
  time_point start_;

 public:
  timing_guard(Counter& counter)
      : counter_(counter), start_(std::chrono::high_resolution_clock::now()) {}

  ~timing_guard() {
    stop();
  }

  void stop() {
    if (start_ != time_point()) {
      auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::high_resolution_clock::now() - start_)
                      .count();
      counter_.update(diff);
      start_ = time_point();
    }
  }
};

void Counter::update(double val) {
  count_ += 1;

  auto delta = val - mean_;
  mean_ += delta / count_;

  auto delta2 = val - mean_;
  m2_ += delta2 * delta2;
}

std::unordered_map<std::string, double> Counter::observe() const {
  std::unordered_map<std::string, double> res;
  res["count"] = (double)count_;
  res["mean"] = mean_;
  if (count_ >= 2) {
    res["sample_variance"] = m2_ / (count_ - 1);
  } else {
    res["sample_variance"] = std::nan("1");
  }
  return res;
}

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

  void sendRaw(uint8_t* data, size_t lenght) {
    try {
      tcputil::sendBytes(socket_.handle(), data, lenght);
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
  bool receiveValueWithTimeout(T& t, std::chrono::milliseconds timeout) {
    if (!socket_.waitForInput(timeout))
      return false;
    t = tcputil::recvValue<T>(socket_.handle());
    return true;
  }
  void setTimeout(std::chrono::milliseconds value);

  explicit TCPClient(Socket&& socket) : socket_{std::move(socket)} {}

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
  const size_t FLUSH_WATERMARK = 1440;
  std::vector<uint8_t> buffer;
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
TCPStore::TCPStore(
    const std::string& masterAddr,
    std::uint16_t masterPort,
    std::optional<int> numWorkers,
    bool isServer,
    const std::chrono::milliseconds& timeout,
    bool waitWorkers)
    : TCPStore{
          masterAddr,
          TCPStoreOptions{
              masterPort,
              isServer,
              numWorkers ? std::optional<std::size_t>(*numWorkers)
                         : std::nullopt,
              waitWorkers,
              timeout}} {}

TCPStore::TCPStore(std::string host, const TCPStoreOptions& opts)
    : Store{opts.timeout},
      addr_{std::move(host)},
      numWorkers_{opts.numWorkers},
      usingLibUv_{opts.useLibUV} {
  if (opts.useLibUV) {
    TORCH_CHECK(
        ::c10d::detail::is_libuv_tcpstore_backend_available(),
        "use_libuv was requested but PyTorch was build without libuv support");

    if (opts.masterListenFd.has_value()) {
      // TODO(xilunwu): support this init method after testing
      constexpr auto* msg =
          "The libuv TCPStore backend does not support initialization with an listen fd. "
          "Please switch to the legacy TCPStore by setting environment variable USE_LIBUV "
          "to \"0\".";
      C10D_ERROR(msg);
      C10_THROW_ERROR(NotImplementedError, msg);
      return;
    }
  }

  Socket::initialize();

  if (opts.isServer) {
    server_ = detail::TCPServer::start(opts);
    // server successfully started
    C10D_DEBUG("The server has started on port = {}.", server_->port());

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

    addr_.port = server_->port();
  } else {
    addr_.port = opts.port;
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

void TCPStore::waitForWorkers() {
  detail::timing_guard tguard(clientCounters_["waitForWorkers"]);
  if (numWorkers_ == std::nullopt) {
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
  TORCH_INTERNAL_ASSERT(
      nonce == returnedNonce, "Ping failed, invalid nonce returned");
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
  detail::timing_guard tguard(clientCounters_["set"]);
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
  detail::timing_guard tguard(clientCounters_["compareSet"]);
  const std::lock_guard<std::mutex> lock(activeOpLock_);
  detail::SendBuffer buffer(*client_, detail::QueryType::COMPARE_SET);
  buffer.appendString(keyPrefix_ + key);
  buffer.appendBytes(expectedValue);
  buffer.appendBytes(desiredValue);
  buffer.flush();

  return client_->receiveBits();
}

std::vector<uint8_t> TCPStore::get(const std::string& key) {
  detail::timing_guard tguard(clientCounters_["get"]);
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
  detail::timing_guard tguard(clientCounters_["add"]);
  const std::lock_guard<std::mutex> lock(activeOpLock_);
  return incrementValueBy(keyPrefix_ + key, value);
}

bool TCPStore::deleteKey(const std::string& key) {
  detail::timing_guard tguard(clientCounters_["deleteKey"]);
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
  detail::timing_guard tguard(clientCounters_["check"]);
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
  TORCH_CHECK(false, "ready or not_ready response expected");
}

void TCPStore::wait(const std::vector<std::string>& keys) {
  wait(keys, timeout_);
}

void TCPStore::wait(
    const std::vector<std::string>& keys,
    const std::chrono::milliseconds& timeout) {
  detail::timing_guard tguard(clientCounters_["wait"]);
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

  detail::WaitResponseType response;
  if (client_->receiveValueWithTimeout<detail::WaitResponseType>(
          response, timeout)) {
    if (response != detail::WaitResponseType::STOP_WAITING) {
      TORCH_CHECK(false, "Stop_waiting response is expected");
    }
    return;
  }
  // this is the cancel wait timeout, once here we expect the server to respond
  // in a timely fashion
  {
    detail::SendBuffer buffer(*client_, detail::QueryType::CANCEL_WAIT);
    buffer.flush();
  }

  response = client_->receiveValue<detail::WaitResponseType>();
  // this can happen if the server responds before we cancel, just ignore it
  if (response != detail::WaitResponseType::WAIT_CANCELED) {
    if (response != detail::WaitResponseType::STOP_WAITING) {
      TORCH_CHECK(false, "Stop_waiting response is expected");
    }

    response = client_->receiveValue<detail::WaitResponseType>(); // ignore
    if (response != detail::WaitResponseType::WAIT_CANCELED) {
      TORCH_CHECK(false, "wait_canceled response is expected");
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
  detail::timing_guard tguard(clientCounters_["append"]);
  const std::lock_guard<std::mutex> lock(activeOpLock_);
  detail::SendBuffer buffer(*client_, detail::QueryType::APPEND);
  buffer.appendString(keyPrefix_ + key);
  buffer.appendBytes(data);
  buffer.flush();
}

std::vector<std::vector<uint8_t>> TCPStore::multiGet(
    const std::vector<std::string>& keys) {
  detail::timing_guard tguard(clientCounters_["multiGet"]);
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
  detail::timing_guard tguard(clientCounters_["multiSet"]);
  TORCH_CHECK(
      keys.size() == values.size(),
      "multiSet keys and values vectors must be of same size");
  const std::lock_guard<std::mutex> lock(activeOpLock_);

  detail::SendBuffer buffer(*client_, detail::QueryType::MULTI_SET);
  buffer.appendValue<std::int64_t>(keys.size());
  for (auto i : c10::irange(keys.size())) {
    buffer.appendString(keyPrefix_ + keys[i]);
    buffer.appendBytes(values[i]);
  }
  buffer.flush();
}

bool TCPStore::hasExtendedApi() const {
  return true;
}

std::unordered_map<std::string, std::unordered_map<std::string, double>>
TCPStore::collectClientCounters() const noexcept {
  std::unordered_map<std::string, std::unordered_map<std::string, double>> res;
  for (const auto& kv : clientCounters_) {
    res[kv.first] = kv.second.observe();
  }
  return res;
}

} // namespace c10d
