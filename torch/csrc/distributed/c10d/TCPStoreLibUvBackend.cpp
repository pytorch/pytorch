#include <algorithm>
#include <deque>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <fmt/format.h>
#include <sys/socket.h>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#include <torch/csrc/distributed/c10d/TCPStoreBackend.hpp>
#include <torch/csrc/distributed/c10d/logging.h>

#ifdef TORCH_USE_LIBUV
#include <uv.h>
#endif

namespace c10d {
namespace detail {

#ifdef TORCH_USE_LIBUV

/*

Exception safety:

It's ok to use exceptions during client processing.
Other callbacks don't provide exception safety so avoid there.

*/

#define DEFAULT_BACKLOG 2048
#define MAX_KEY_COUNT (128 * 1024)
#define MAX_STRING_LEN (8 * 1024)
#define MAX_PAYLOAD_LEN (8 * 1024 * 1024)

// This controls the preferred size for buffers.
// Too small and we'll need multiple buffers for one request
// Too big and we might taxing malloc
#define ALLOC_BUFFER_SIZE ((size_t)4000)
class UvHandle {
 public:
  virtual ~UvHandle() {}

  static UvHandle* from_uv(uv_handle_t* stream) {
    return (UvHandle*)uv_handle_get_data(stream);
  }

  virtual void close() = 0;
  virtual uv_stream_t* as_stream() = 0;
};

class WriterPayload {
  static void write_done(uv_write_t* req, int status) {
    UvHandle* handle = UvHandle::from_uv((uv_handle_t*)req->handle);
    auto wp = (WriterPayload*)uv_req_get_data((uv_req_t*)req);
    delete wp;

    if (status) {
      C10D_INFO(
          "Write to client failed. code:{} name:{} desc:{}.",
          status,
          uv_err_name(status),
          uv_strerror(status));
      handle->close();
    }
  }

  std::vector<uint8_t> data;
  uv_write_t req;
  uv_buf_t buf;
  UvHandle* handle;

 public:
  WriterPayload(std::vector<uint8_t>&& in_data, UvHandle* handle)
      : data(in_data), handle(handle) {}

  bool send() {
    buf = uv_buf_init((char*)data.data(), data.size());
    int res = uv_write(&req, handle->as_stream(), &buf, 1, write_done);
    uv_req_set_data((uv_req_t*)&req, this);

    if (res) {
      C10D_INFO(
          "Write setup to client failed. code:{} name:{} desc:{}.",
          res,
          uv_err_name(res),
          uv_strerror(res));
      handle->close();
    }
    return res == 0;
  }
};

class StreamWriter {
  std::vector<uint8_t> data;
  UvHandle* handle;

  // must be stack allocated
  void* operator new(size_t);

 public:
  StreamWriter(UvHandle* handle) : handle(handle) {}

  void write1(uint8_t val) {
    data.push_back(val);
  }

  template <typename T>
  void write_value(T val) {
    uint8_t* val_ptr = (uint8_t*)&val;
    data.insert(data.end(), val_ptr, val_ptr + sizeof(T));
  }

  void write_vector(const std::vector<uint8_t>& val) {
    write_value<uint64_t>(val.size());
    data.insert(data.end(), val.begin(), val.end());
  }

  void write_string(const std::string& val) {
    write_value<uint64_t>(val.size());
    data.insert(data.end(), val.data(), val.data() + val.size());
  }
  void send() {
    auto wd = new WriterPayload(std::move(data), handle);
    if (!wd->send())
      delete wd;
  }
};

class ChunkedStream {
  std::deque<uv_buf_t> buffers;
  size_t buff_idx;
  size_t buff_offset;
  size_t capacity;
  size_t buff_offset_commit;
  size_t read_offset;

 public:
  ChunkedStream()
      : buff_idx(0),
        buff_offset(0),
        capacity(0),
        buff_offset_commit(0),
        read_offset(0) {}

  size_t buf_count() {
    return buffers.size();
  }

  void append(uv_buf_t buf) {
    if (buf.len == 0) {
      free(buf.base);
    } else {
      capacity += buf.len;
      buffers.push_back(buf);
    }
  }
  bool read_many(char* dest, size_t size) {
    if (available() < size) {
      return false;
    }

    size_t remaining = size;
    char* write_base = dest;
    while (remaining > 0) {
      auto to_read = std::min(buffers[buff_idx].len - buff_offset, remaining);
      ::memcpy(write_base, buffers[buff_idx].base + buff_offset, to_read);
      buff_offset += to_read;
      remaining -= to_read;
      write_base += to_read;
      if (buff_offset >= buffers[buff_idx].len) {
        buff_offset = 0;
        ++buff_idx;
        if (buff_idx >= buffers.size() && remaining > 0) {
          TORCH_CHECK(
              false,
              "Trying to read past end of buffer buffer_idx:{} available:{} remaining:{}",
              buff_idx,
              buffers.size(),
              remaining);
        }
      }
    }
    return true;
  }

  bool read1(uint8_t& byte) {
    while (true) {
      if (buff_idx >= buffers.size())
        return false;
      if (buff_offset >= buffers[buff_idx].len) {
        buff_offset = 0;
        ++buff_idx;
        continue;
      }
      break;
    }

    byte = buffers[buff_idx].base[buff_offset];
    ++buff_offset;
    ++read_offset;
    return true;
  }

  template <typename T>
  bool read_value(T& value) {
    return read_many((char*)&value, sizeof(T));
  }

  bool read_key(std::string& str) {
    uint64_t size = 0;
    if (!read_value(size))
      return false;
    TORCH_CHECK(
        size <= MAX_STRING_LEN,
        "Invalid string size. size:{} max:{}",
        size,
        MAX_STRING_LEN);

    if (available() < size)
      return false;
    str.resize(size);
    return read_many((char*)str.data(), size);
  }

  bool read_payload(std::vector<uint8_t>& data) {
    uint64_t size = 0;
    if (!read_value(size))
      return false;
    auto size_in_bytes = size * sizeof(uint8_t);
    TORCH_CHECK(
        size_in_bytes <= MAX_PAYLOAD_LEN,
        "Invalid payload size. size: {} max:{}",
        size_in_bytes,
        MAX_PAYLOAD_LEN);

    if (available() < size_in_bytes)
      return false;
    data.resize(size);
    return read_many((char*)data.data(), size_in_bytes);
  }

  size_t available() {
    return capacity - read_offset;
  }

  void commit() {
    if (buff_idx >= buffers.size() || buff_offset >= buffers[buff_idx].len) {
      buff_offset = 0;
      if (buff_idx < buffers.size())
        ++buff_idx;
    }

    for (size_t i = 0; i < buff_idx; ++i) {
      free(buffers[0].base);
      capacity -= buffers[0].len;
      buffers.pop_front();
    }
    buff_idx = 0;
    read_offset = buff_offset_commit = buff_offset;
  }

  void reset() {
    buff_idx = 0;
    read_offset = buff_offset = buff_offset_commit;
  }
};

class LibUVStoreDaemon : public BackgroundThread {
 public:
  explicit LibUVStoreDaemon(int port);
  ~LibUVStoreDaemon() override;

  uint16_t port() const override;

  void set(const std::string& key, const std::vector<uint8_t>& value);
  const std::vector<uint8_t>& compareAndSet(
      const std::string& key,
      const std::vector<uint8_t>& expectedValue,
      const std::vector<uint8_t>& newValue);
  const std::vector<uint8_t>& get(const std::string& key);
  int64_t add(const std::string& key, int64_t addVal);
  bool checkKeys(const std::vector<std::string>& keys);
  bool waitKeys(const std::vector<std::string>& keys, UvHandle* client);
  int64_t size();
  int64_t deleteKey(const std::string& key);
  void append(const std::string& key, const std::vector<uint8_t>& value);

  void registerClient(UvHandle* client);
  void unregisterClient(UvHandle* client);
  void clearClientWaitState(UvHandle* client);

  void init(const TCPStoreOptions& opts);

 protected:
  void run() override;
  void stop() override;

 private:
  uv_loop_t loop;
  uv_tcp_t server;
  uv_async_t exit_handle;
  std::unordered_map<std::string, std::vector<uint8_t>> tcpStore_;
  // From key -> the list of UvClient waiting on the key
  std::unordered_map<std::string, std::vector<UvHandle*>> waitingSockets_;
  // From socket -> number of keys awaited
  std::unordered_map<UvHandle*, size_t> keysAwaited_;
  std::unordered_set<UvHandle*> clients_;
  int port_;

  static LibUVStoreDaemon& from_uv(uv_handle_t* stream) {
    return *(LibUVStoreDaemon*)uv_handle_get_data(stream);
  }

  static void on_new_connection(uv_stream_t* server, int status) {
    from_uv((uv_handle_t*)server).onConnect(status);
  }

  static void on_exit_request(uv_async_t* handle) {
    from_uv((uv_handle_t*)handle).onExitRequest();
  }

  void onConnect(int status);
  void onExitRequest();
  void wakeupWaitingClients(const std::string& key);
  bool tryListen(bool use_ipv6);
};

class UvClient : public UvHandle {
  uv_tcp_t client;
  ChunkedStream stream;
  LibUVStoreDaemon* store;

  static void read_callback(
      uv_stream_t* client,
      ssize_t nread,
      const uv_buf_t* buf) {
    UvClient* uv_client = UvClient::from_uv((uv_handle_t*)client);

    if (nread < 0) {
      C10D_DEBUG(
          "Read callback failed. client:{} error: {} ",
          (void*)uv_client,
          nread,
          uv_err_name(nread),
          uv_strerror(nread));
      uv_client->close();
      return;
    }
    if (nread > 0) {
      try {
        uv_client->processBuf(buf, nread);
      } catch (std::exception& ex) {
        C10D_INFO("Error processing client message: {}", ex.what());
        uv_client->close();
      }
    }
  }
  static void on_close(uv_handle_t* handle) {
    UvClient* client = UvClient::from_uv(handle);
    client->store->unregisterClient(client);
    delete client;
  }

  static void alloc_buffer(
      uv_handle_t* handle,
      size_t suggested_size,
      uv_buf_t* buf) {
    suggested_size = std::min(suggested_size, (size_t)ALLOC_BUFFER_SIZE);
    buf->base = (char*)malloc(suggested_size);
    buf->len = suggested_size;
  }

  void processBuf(const uv_buf_t* buf, size_t nread) {
    auto tmp = *buf;
    tmp.len = nread;
    stream.append(tmp);

    while (true) {
      stream.reset();
      uint8_t command = -1;
      if (!stream.read1(command))
        break;
      switch ((QueryType)command) {
        case QueryType::SET:
          if (!parse_set_command())
            return;
          break;
        case QueryType::COMPARE_SET:
          if (!parse_compare_set_command())
            return;
          break;
        case QueryType::GET:
          if (!parse_get_command())
            return;
          break;
        case QueryType::ADD:
          if (!parse_add_command())
            return;
          break;
        case QueryType::CHECK:
          if (!parse_check_command())
            return;
          break;
        case QueryType::WAIT:
          if (!parse_wait_command())
            return;
          break;
        case QueryType::GETNUMKEYS:
          if (!parse_getnumkeys_command())
            return;
          break;
        case QueryType::DELETE_KEY:
          if (!parse_delete_key_command())
            return;
          break;
        case QueryType::APPEND:
          if (!parse_append_command())
            return;
          break;
        case QueryType::MULTI_GET:
          if (!parse_multi_get_command())
            return;
          break;
        case QueryType::MULTI_SET:
          if (!parse_multi_set_command())
            return;
          break;
        case QueryType::CANCEL_WAIT:
          if (!parse_cancel_wait_command())
            return;
          break;
        default:
          C10D_DEBUG(
              "Client sent invalid command. client:{} command:{}",
              (void*)this,
              (int)command);
          close();
          return;
      }
      stream.commit();
    }
  }

  bool parse_set_command() {
    std::string key;
    if (!stream.read_key(key))
      return false;

    std::vector<uint8_t> newData;
    if (!stream.read_payload(newData))
      return false;

    store->set(key, newData);
    return true;
  }

  bool parse_compare_set_command() {
    std::string key;
    if (!stream.read_key(key))
      return false;

    std::vector<uint8_t> currentValue;
    if (!stream.read_payload(currentValue))
      return false;

    std::vector<uint8_t> newValue;
    if (!stream.read_payload(newValue))
      return false;

    auto res = store->compareAndSet(key, currentValue, newValue);
    StreamWriter sw(this);
    sw.write_vector(res);
    sw.send();

    return true;
  }

  bool parse_get_command() {
    std::string key;
    if (!stream.read_key(key))
      return false;

    auto data = store->get(key);
    StreamWriter sw(this);
    sw.write_vector(data);
    sw.send();
    return true;
  }

  bool parse_add_command() {
    std::string key;
    if (!stream.read_key(key))
      return false;

    int64_t addVal = 0;
    if (!stream.read_value(addVal))
      return false;

    addVal = store->add(key, addVal);
    StreamWriter sw(this);
    sw.write_value(addVal);
    sw.send();

    return true;
  }

  bool parse_check_command() {
    uint64_t key_count = 0;
    if (!stream.read_value(key_count))
      return false;
    TORCH_CHECK(
        key_count <= MAX_KEY_COUNT,
        "Too many keys being waited. keys:{} max:{}",
        key_count,
        MAX_KEY_COUNT);

    std::vector<std::string> keys(key_count);
    for (uint64_t i = 0; i < key_count; ++i) {
      if (!stream.read_key(keys[i]))
        return false;
    }

    // Now we have received all the keys
    StreamWriter sw(this);
    if (store->checkKeys(keys)) {
      sw.write_value(CheckResponseType::READY);
    } else {
      sw.write_value(CheckResponseType::NOT_READY);
    }
    sw.send();
    return true;
  }

  bool parse_wait_command() {
    uint64_t key_count = 0;
    if (!stream.read_value(key_count)) {
      return false;
    }
    TORCH_CHECK(
        key_count <= MAX_KEY_COUNT,
        "Too many keys being waited. keys:{} max:{}",
        key_count,
        MAX_KEY_COUNT);

    std::vector<std::string> keys(key_count);
    for (uint64_t i = 0; i < key_count; ++i) {
      if (!stream.read_key(keys[i]))
        return false;
    }

    if (store->waitKeys(keys, this)) {
      StreamWriter sw(this);
      sw.write1((uint8_t)WaitResponseType::STOP_WAITING);
      sw.send();
    }

    return true;
  }

  bool parse_getnumkeys_command() {
    StreamWriter sw(this);
    sw.write_value<int64_t>(store->size());
    sw.send();

    return true;
  }

  bool parse_delete_key_command() {
    std::string key;
    if (!stream.read_key(key))
      return false;

    auto numDeleted = store->deleteKey(key);
    StreamWriter sw(this);
    sw.write_value<int64_t>(numDeleted);
    sw.send();

    return true;
  }

  bool parse_append_command() {
    std::string key;
    if (!stream.read_key(key)) {
      return false;
    }

    std::vector<uint8_t> data;
    if (!stream.read_payload(data)) {
      return false;
    }

    store->append(key, data);
    return true;
  }

  bool parse_multi_get_command() {
    uint64_t key_count = 0;
    if (!stream.read_value(key_count)) {
      return false;
    }
    TORCH_CHECK(
        key_count <= MAX_KEY_COUNT,
        "Too many keys with multi_get. keys:{} max:{}",
        key_count,
        MAX_KEY_COUNT);

    StreamWriter sw(this);
    for (const auto _ : c10::irange(key_count)) {
      (void)_; // Suppress unused variable warning
      std::string key;
      if (!stream.read_key(key)) {
        return false;
      }

      auto data = store->get(key);
      sw.write_vector(data);
    }
    sw.send();

    return true;
  }

  bool parse_multi_set_command() {
    uint64_t key_count = 0;
    if (!stream.read_value(key_count)) {
      return false;
    }
    TORCH_CHECK(
        key_count <= MAX_KEY_COUNT,
        "Too many keys with multi_get. keys:{} max:{}",
        key_count,
        MAX_KEY_COUNT);

    for (const auto _ : c10::irange(key_count)) {
      (void)_; // Suppress unused variable warning

      std::string key;
      if (!stream.read_key(key)) {
        return false;
      }

      std::vector<uint8_t> newData;
      if (!stream.read_payload(newData))
        return false;
      store->set(key, newData);
    }

    return true;
  }

  bool parse_cancel_wait_command() {
    store->clearClientWaitState(this);

    StreamWriter sw(this);
    sw.write1((uint8_t)WaitResponseType::WAIT_CANCELED);
    sw.send();

    return true;
  }

 public:
  explicit UvClient(uv_loop_t* loop, LibUVStoreDaemon* store) : store(store) {
    uv_tcp_init(loop, &client);
    uv_handle_set_data((uv_handle_t*)&client, this);
    C10D_DEBUG("Accepted new client: {}\n", (void*)this);
  }

  void startRead() {
    int res =
        uv_read_start((uv_stream_t*)as_stream(), alloc_buffer, read_callback);
    if (res) {
      C10D_INFO(
          "Failed to setup read callback. client:{} code:{} name:{} desc:{}.",
          (void*)this,
          res,
          uv_err_name(res),
          uv_strerror(res));
      close();
    }
  }

  static UvClient* from_uv(uv_handle_t* handle) {
    return (UvClient*)uv_handle_get_data(handle);
  }

  void close() override {
    if (!uv_is_closing((uv_handle_t*)&client)) {
      uv_close((uv_handle_t*)&client, on_close);
    }
  }

  uv_stream_t* as_stream() override {
    return (uv_stream_t*)&client;
  }
};

void LibUVStoreDaemon::onConnect(int status) {
  UvClient* client = new UvClient(&loop, this);

  registerClient(client);
  int res = uv_accept((uv_stream_t*)&server, client->as_stream());
  if (res == 0) {
    client->startRead();
  } else {
    C10D_INFO(
        "Failed to accept client. client:{} code:{} name:{} desc:{}.",
        (void*)client,
        res,
        uv_err_name(res),
        uv_strerror(res));
    client->close();
  }
}

void LibUVStoreDaemon::onExitRequest() {
  C10D_DEBUG("Store exit requested\n");
  uv_close((uv_handle_t*)&exit_handle, nullptr);
  uv_stop(&loop);
}

uint16_t get_socket_port(uv_tcp_t* handle) {
  sockaddr_storage addr_s{};

  int addr_len = sizeof(addr_s);

  if (uv_tcp_getsockname(
          handle, reinterpret_cast<sockaddr*>(&addr_s), &addr_len) != 0) {
    throw std::runtime_error(
        "The port number of the socket cannot be retrieved.");
  }

  if (addr_s.ss_family == AF_INET) {
    return ntohs(reinterpret_cast<sockaddr_in*>(&addr_s)->sin_port);
  } else {
    return ntohs(reinterpret_cast<sockaddr_in6*>(&addr_s)->sin6_port);
  }
}

void LibUVStoreDaemon::init(const TCPStoreOptions& opts) {
  uv_handle_set_data((uv_handle_t*)&server, this);

  if (opts.masterListenFd.has_value()) {
    int res = uv_tcp_open(&server, *opts.masterListenFd);
    TORCH_CHECK(
        res == 0,
        "Failed to open an existing socket. code:{} name:{} message:{}",
        res,
        uv_err_name(res),
        uv_strerror(res));

    auto port = get_socket_port(&server);
    TORCH_CHECK(
        port == opts.port,
        "listen fd {} is bound to port {}, expected to be bound to port {}",
        *opts.masterListenFd,
        port,
        opts.port);

    return;
  }

  if (tryListen(true)) {
    return;
  }
  uv_close((uv_handle_t*)&server, nullptr);
  uv_run(&loop, UV_RUN_ONCE);

  if (tryListen(false)) {
    return;
  }
  uv_close((uv_handle_t*)&server, nullptr);
  uv_run(&loop, UV_RUN_ONCE);
  TORCH_CHECK(false, "failed to init store, no bind possible");
}

bool LibUVStoreDaemon::tryListen(bool use_ipv6) {
  int res = uv_tcp_init(&loop, &server);
  struct sockaddr_storage addr;
  if (res) {
    C10D_WARNING(
        "UV Store init tcp socket. ipv6:{} message:{}",
        use_ipv6,
        uv_strerror(res));
    return false;
  }

  if (use_ipv6) {
    res = uv_ip6_addr("::", port_, (struct sockaddr_in6*)&addr);
  } else {
    res = uv_ip4_addr("0.0.0.0", port_, (struct sockaddr_in*)&addr);
  }
  if (res) {
    C10D_WARNING(
        "UV Store addr parsing failure. ipv6:{} message:{}",
        use_ipv6,
        uv_strerror(res));
    return false;
  }
  res = uv_tcp_bind(&server, (const struct sockaddr*)&addr, 0);
  if (res) {
    C10D_WARNING(
        "UV Store tcp bind failure. ipv6:{} message:{}",
        use_ipv6,
        uv_strerror(res));
    return false;
  }
  res = uv_listen(
      (uv_stream_t*)&server,
      DEFAULT_BACKLOG,
      LibUVStoreDaemon::on_new_connection);
  if (res) {
    C10D_WARNING(
        "UV Store listen failure. ipv6:{} message:{}",
        use_ipv6,
        uv_strerror(res));
  }
  return res == 0;
}

LibUVStoreDaemon::LibUVStoreDaemon(int port) : port_(port) {
  TORCH_CHECK(uv_loop_init(&loop) == 0, "Failed to init uv loop");
  TORCH_CHECK(
      uv_async_init(&loop, &exit_handle, LibUVStoreDaemon::on_exit_request) ==
          0,
      "Failed to init uv async event");
  uv_handle_set_data((uv_handle_t*)&exit_handle, this);
}

LibUVStoreDaemon::~LibUVStoreDaemon() {
  if (!is_running()) {
    uv_close((uv_handle_t*)&exit_handle, nullptr);
    uv_run(&loop, UV_RUN_NOWAIT);
    TORCH_CHECK(uv_loop_close(&loop) == 0, "loop cleanup didn't work");
  } else {
    // the daemon thread cleanup libuv
    dispose();
  }
}

uint16_t LibUVStoreDaemon::port() const {
  return port_;
}

void close_all_stuff(uv_handle_t* handle, void* arg) {
  C10D_DEBUG(
      "UV live handle type {} active:{} is-closing:{}",
      handle->type,
      uv_is_active(handle),
      uv_is_closing(handle));
}

void LibUVStoreDaemon::run() {
  C10D_DEBUG("Uv main loop running");
  int res = uv_run(&loop, UV_RUN_DEFAULT);
  if (res) {
    C10D_DEBUG("UV main loop done: res:{}", res);
  }
  bool debug_enabled =
      c10d::detail::isLogLevelEnabled(c10d::detail::LogLevel::Debug);

  if (debug_enabled) {
    C10D_DEBUG("Walking live handles prior to closing clients");
    uv_walk(&loop, close_all_stuff, nullptr);
  }

  for (auto it = clients_.begin(); it != clients_.end(); ++it) {
    (*it)->close();
  }
  uv_close((uv_handle_t*)&server, nullptr);

  if (debug_enabled) {
    C10D_DEBUG("Walking live handles after closing clients");
    uv_walk(&loop, close_all_stuff, nullptr);
  }

  while (1) {
    res = uv_loop_close(&loop);
    if (res == 0) {
      break;
    }
    C10D_INFO(
        "uv_loop_close failed with:{} errn:{} desc:{}",
        res,
        uv_err_name(res),
        uv_strerror(res));
    res = uv_run(&loop, UV_RUN_NOWAIT);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }
  C10D_INFO("uv_loop cleanup finished.");
}

void LibUVStoreDaemon::stop() {
  int res = uv_async_send(&exit_handle);
  if (res) {
    C10D_INFO(
        "uv_async_send failed with:{} errn:{} desc:{}\n",
        res,
        uv_err_name(res),
        uv_strerror(res));
  }
}

void LibUVStoreDaemon::registerClient(UvHandle* client) {
  clients_.insert(client);
}

void LibUVStoreDaemon::unregisterClient(UvHandle* client) {
  clients_.erase(client);
  clearClientWaitState(client);
}

void LibUVStoreDaemon::clearClientWaitState(UvHandle* client) {
  if (keysAwaited_.find(client) == keysAwaited_.end()) {
    return;
  }
  keysAwaited_.erase(client);
  for (auto it = waitingSockets_.begin(); it != waitingSockets_.end();) {
    for (auto vecIt = it->second.begin(); vecIt != it->second.end();) {
      if (*vecIt == client) {
        vecIt = it->second.erase(vecIt);
      } else {
        ++vecIt;
      }
    }
    if (it->second.empty()) {
      it = waitingSockets_.erase(it);
    } else {
      ++it;
    }
  }
}

void LibUVStoreDaemon::set(
    const std::string& key,
    const std::vector<uint8_t>& value) {
  tcpStore_[key] = value;
  // On "set", wake up all clients that have been waiting
  wakeupWaitingClients(key);
}

const std::vector<uint8_t>& LibUVStoreDaemon::compareAndSet(
    const std::string& key,
    const std::vector<uint8_t>& expectedValue,
    const std::vector<uint8_t>& newValue) {
  auto pos = tcpStore_.find(key);
  if (pos == tcpStore_.end()) {
    if (expectedValue.empty()) {
      tcpStore_[key] = newValue;
      wakeupWaitingClients(key);
      return newValue;
    } else {
      // TODO: This code path is not ideal as we are "lying" to the caller in
      // case the key does not exist. We should come up with a working solution.
      // It might make more sense to return ""
      wakeupWaitingClients(key);
      return expectedValue;
    }
  } else {
    if (pos->second == expectedValue) {
      pos->second = std::move(newValue);
    }
    wakeupWaitingClients(key);
    return pos->second;
  }
}

const std::vector<uint8_t>& LibUVStoreDaemon::get(const std::string& key) {
  static std::vector<uint8_t> missing_key;
  return tcpStore_.count(key) ? tcpStore_.at(key) : missing_key;
}

int64_t LibUVStoreDaemon::add(const std::string& key, int64_t addVal) {
  std::vector<uint8_t> oldData;
  auto it = tcpStore_.find(key);
  if (it != tcpStore_.end()) {
    oldData = it->second;
    auto buf = reinterpret_cast<const char*>(it->second.data());
    auto len = it->second.size();
    addVal += std::stoll(std::string(buf, len));
  }
  auto addValStr = std::to_string(addVal);
  std::vector<uint8_t> newData =
      std::vector<uint8_t>(addValStr.begin(), addValStr.end());
  tcpStore_[key] = newData;

  // On "add", wake up all clients that have been waiting
  wakeupWaitingClients(key);

  return addVal;
}

bool LibUVStoreDaemon::checkKeys(const std::vector<std::string>& keys) {
  return std::all_of(keys.begin(), keys.end(), [&](const std::string& s) {
    return tcpStore_.count(s) > 0;
  });
}

bool LibUVStoreDaemon::waitKeys(
    const std::vector<std::string>& keys,
    UvHandle* client) {
  if (checkKeys(keys)) {
    return true;
  }
  int numKeysToAwait = 0;
  for (auto& key : keys) {
    // Only count keys that have not already been set
    if (tcpStore_.find(key) == tcpStore_.end()) {
      waitingSockets_[key].push_back(client);
      numKeysToAwait++;
    }
  }
  keysAwaited_[client] = numKeysToAwait;
  return false;
}

int64_t LibUVStoreDaemon::size() {
  return tcpStore_.size();
}

int64_t LibUVStoreDaemon::deleteKey(const std::string& key) {
  return tcpStore_.erase(key);
}

void LibUVStoreDaemon::append(
    const std::string& key,
    const std::vector<uint8_t>& value) {
  std::vector<uint8_t> oldData;
  auto it = tcpStore_.find(key);
  if (it != tcpStore_.end()) {
    it->second.insert(it->second.end(), value.begin(), value.end());
  } else {
    tcpStore_[key] = value;
  }

  // we should not have clients waiting if we're appending, so it's all fine
  wakeupWaitingClients(key);
}

void LibUVStoreDaemon::wakeupWaitingClients(const std::string& key) {
  auto socketsToWait = waitingSockets_.find(key);
  if (socketsToWait != waitingSockets_.end()) {
    for (UvHandle* client : socketsToWait->second) {
      if (--keysAwaited_[client] == 0) {
        StreamWriter sw(client);
        sw.write1((uint8_t)WaitResponseType::STOP_WAITING);
        sw.send();
      }
    }
    waitingSockets_.erase(socketsToWait);
  }
}

#endif

std::unique_ptr<BackgroundThread> create_libuv_tcpstore_backend(
    const TCPStoreOptions& opts) {
#ifdef TORCH_USE_LIBUV
  auto res = std::make_unique<LibUVStoreDaemon>(opts.port);
  res->init(opts);
  return res;
#else
  TORCH_CHECK(false, "Implementation missing");
#endif
}

bool is_libuv_tcpstore_backend_available() {
#ifdef TORCH_USE_LIBUV
  return true;
#else
  return false;
#endif
}

} // namespace detail
} // namespace c10d
