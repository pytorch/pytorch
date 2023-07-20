#include <c10/util/irange.h>
#include <torch/csrc/distributed/c10d/logging.h>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#include <torch/csrc/distributed/c10d/TCPStoreBackend.hpp>

#include <fcntl.h>
#include <algorithm>
#include <array>
#include <system_error>
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
    auto daemon = create_tcpstore_backend(opts);
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
      const TCPStoreOptions& opts);

  void sendRaw(uint8_t* data, size_t lenght) {
    tcputil::sendBytes(socket_.handle(), data, lenght);
  }

  std::vector<std::uint8_t> receiveBits() {
    return tcputil::recvVector<std::uint8_t>(socket_.handle());
  }

  template <typename T>
  T receiveValue() {
    return tcputil::recvValue<T>(socket_.handle());
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
    const TCPStoreOptions& opts) {
  auto timeout = std::chrono::duration_cast<std::chrono::seconds>(opts.timeout);
  Socket socket = Socket::connect(
      addr.host, addr.port, SocketOptions{}.connect_timeout(timeout));

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
    if (buffer.size() > 0) {
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
    c10::optional<int> numWorkers,
    bool isServer,
    const std::chrono::milliseconds& timeout,
    bool waitWorkers)
    : TCPStore{
          masterAddr,
          TCPStoreOptions{
              masterPort,
              isServer,
              numWorkers ? c10::optional<std::size_t>(*numWorkers)
                         : c10::nullopt,
              waitWorkers,
              timeout}} {}

TCPStore::TCPStore(std::string host, const TCPStoreOptions& opts)
    : Store{opts.timeout},
      addr_{std::move(host)},
      numWorkers_{opts.numWorkers} {
  Socket::initialize();

  if (opts.isServer) {
    server_ = detail::TCPServer::start(opts);
    // server successfully started
    C10D_DEBUG("The server has started on port = {}.", server_->port());

    addr_.port = server_->port();
  } else {
    addr_.port = opts.port;
  }

  client_ = detail::TCPClient::connect(addr_, opts);
  // TCP connection established
  C10D_DEBUG("TCP client connected to host {}:{}", addr_.host, addr_.port);

  if (opts.waitWorkers) {
    waitForWorkers();
  }
}

TCPStore::~TCPStore() = default;

void TCPStore::waitForWorkers() {
  if (numWorkers_ == c10::nullopt) {
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
        break;
      }
      /* sleep override */
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
}

void TCPStore::set(const std::string& key, const std::vector<uint8_t>& data) {
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
  const std::lock_guard<std::mutex> lock(activeOpLock_);
  detail::SendBuffer buffer(*client_, detail::QueryType::COMPARE_SET);
  buffer.appendString(keyPrefix_ + key);
  buffer.appendBytes(expectedValue);
  buffer.appendBytes(desiredValue);
  buffer.flush();

  return client_->receiveBits();
}

std::vector<uint8_t> TCPStore::get(const std::string& key) {
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
  const std::lock_guard<std::mutex> lock(activeOpLock_);
  return incrementValueBy(keyPrefix_ + key, value);
}

bool TCPStore::deleteKey(const std::string& key) {
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
  TORCH_CHECK(false, "Socket Timeout");
}

void TCPStore::append(
    const std::string& key,
    const std::vector<uint8_t>& data) {
  const std::lock_guard<std::mutex> lock(activeOpLock_);
  detail::SendBuffer buffer(*client_, detail::QueryType::APPEND);
  buffer.appendString(keyPrefix_ + key);
  buffer.appendBytes(data);
  buffer.flush();
}

std::vector<std::vector<uint8_t>> TCPStore::multiGet(
    const std::vector<std::string>& keys) {
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

} // namespace c10d
