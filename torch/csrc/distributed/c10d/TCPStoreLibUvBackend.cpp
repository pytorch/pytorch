#include <algorithm>
#include <deque>
#include <exception>
#include <memory>
#include <ostream>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#include <torch/csrc/distributed/c10d/TCPStoreBackend.hpp>
#include <torch/csrc/distributed/c10d/logging.h>

#ifdef TORCH_USE_LIBUV
#include <uv.h>
#endif

namespace c10d::detail {

#ifdef TORCH_USE_LIBUV

/*

Exception safety:

It's ok to use exceptions during client processing.
Other callbacks don't provide exception safety so avoid there.

*/

#define DEFAULT_BACKLOG 16384
#define MAX_KEY_COUNT (128 * 1024)
#define MAX_STRING_LEN (8 * 1024)
#define MAX_PAYLOAD_LEN (8 * 1024 * 1024)

// This controls the preferred size for buffers.
// Too small and we'll need multiple buffers for one request
// Too big and we might taxing malloc
#define ALLOC_BUFFER_SIZE ((size_t)4000)
class UvHandle : public c10::intrusive_ptr_target {
 public:
  ~UvHandle() override = default;

  c10::intrusive_ptr<UvHandle> iptr() {
    return c10::intrusive_ptr<UvHandle>::reclaim_copy(this);
  }

  void close() {
    if (uv_is_closing(unsafeGetHandle())) {
      return;
    }
    uv_close(unsafeGetHandle(), on_close);
  }

  virtual uv_handle_t* unsafeGetHandle() = 0;

 protected:
  void handleReady() {
    /*
    This method must be called once the handle is ready and registered with the
    loop.

    Do not call this in the ctor, make_intrusive reset refcounts to one after
    construction.
    */
    uv_handle_set_data(unsafeGetHandle(), this);
    at::raw::intrusive_ptr::incref(this);
  }

  virtual void onClose() = 0;

 private:
  static c10::intrusive_ptr<UvHandle> reclaim(uv_handle_t* handle) {
    auto h = (UvHandle*)uv_handle_get_data(handle);
    return c10::intrusive_ptr<UvHandle>::reclaim(h);
  }

  static void on_close(uv_handle_t* uv_handle) {
    auto handle = reclaim(uv_handle);
    handle->onClose();
  }
};

class UvTcpSocket : public UvHandle {
  uv_tcp_t client{};

  c10::intrusive_ptr<UvTcpSocket> iptr() {
    return c10::intrusive_ptr<UvTcpSocket>::reclaim_copy(this);
  }

  static c10::intrusive_ptr<UvTcpSocket> borrow(uv_stream_t* handle) {
    auto h = (UvTcpSocket*)uv_handle_get_data((uv_handle_t*)handle);
    return h->iptr();
  }

  static void alloc_buffer(

      uv_handle_t* handle,
      size_t suggested_size,
      uv_buf_t* buf) {
    suggested_size = std::min(suggested_size, (size_t)ALLOC_BUFFER_SIZE);
    buf->base = (char*)malloc(suggested_size);
    buf->len = suggested_size;
  }

  static void read_callback(
      uv_stream_t* client,
      ssize_t nread,
      const uv_buf_t* buf) {
    auto uv_socket = UvTcpSocket::borrow(client);

    if (nread < 0) {
      C10D_DEBUG(
          "Read callback failed. code:{} name:{} desc:{}",
          nread,
          uv_err_name(nread),
          uv_strerror(nread));
      uv_socket->close();
      return;
    }
    if (nread > 0) {
      try {
        uv_socket->processBuf(buf, nread);
      } catch (std::exception& ex) {
        C10D_INFO("Error processing client message: {}", ex.what());
        uv_socket->close();
      }
    }
  }

 public:
  explicit UvTcpSocket(uv_loop_t* loop) {
    uv_tcp_init(loop, &client);
  }

  void startRead() {
    int res = uv_read_start((uv_stream_t*)&client, alloc_buffer, read_callback);
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

  uv_handle_t* unsafeGetHandle() override {
    return (uv_handle_t*)&client;
  }

 protected:
  uv_stream_t* unsafeGetStream() {
    return (uv_stream_t*)&client;
  }

  uv_tcp_t* unsafeGetSocket() {
    return &client;
  }

  virtual void processBuf(const uv_buf_t* buf, size_t nread) {
    TORCH_CHECK(
        false, "Trying to read from a socket subclass that lacks processBuf");
  }

  void onClose() override {
    // TODO use registerClient (and rename it to registerHandle) - this will
    // significantly simplify things.
  }
};

class UvTcpServer : public UvTcpSocket {
 public:
  typedef std::function<void(int)> OnConnectCallback;
  explicit UvTcpServer(uv_loop_t* loop)
      : UvTcpSocket(loop), onConnectCb(missingOnConnect) {}

  static c10::intrusive_ptr<UvTcpServer> makeWithSocket(
      uv_loop_t* loop,
      int socket) {
    auto res = c10::make_intrusive<UvTcpServer>(loop);
    res->handleReady();
    try {
      int uv_res = uv_tcp_open((uv_tcp_t*)res->unsafeGetStream(), socket);
      TORCH_CHECK(
          uv_res == 0,
          "Failed to open existing socket. ",
          "socket: ",
          socket,
          ", code: ",
          uv_res,
          ", name: ",
          uv_err_name(uv_res),
          ", message: ",
          uv_strerror(uv_res));

      res->cacheSocketPort();
    } catch (std::exception& ex) {
      res->close();
      throw;
    }

    return res;
  }

  void setOnConnectCallback(OnConnectCallback&& callback) {
    onConnectCb = std::move(callback);
  }

  static c10::intrusive_ptr<UvTcpServer> makeWithPort(
      uv_loop_t* loop,
      uint16_t port,
      bool useIpv6) {
    auto res = c10::make_intrusive<UvTcpServer>(loop);
    res->handleReady();
    try {
      struct sockaddr_storage addr {};
      int uv_res = 0;
      if (useIpv6) {
        uv_res = uv_ip6_addr("::", port, (struct sockaddr_in6*)&addr);
      } else {
        uv_res = uv_ip4_addr("0.0.0.0", port, (struct sockaddr_in*)&addr);
      }
      TORCH_CHECK(
          uv_res == 0,
          "UV Store addr parsing failure. ",
          "useIpv6: ",
          useIpv6,
          ", code: ",
          uv_res,
          ", name: ",
          uv_err_name(uv_res),
          ", message: ",
          uv_strerror(uv_res));

      uv_res =
          uv_tcp_bind(res->unsafeGetSocket(), (const struct sockaddr*)&addr, 0);
      TORCH_CHECK(
          uv_res == 0,
          "The server socket has failed to bind. ",
          "useIpv6: ",
          useIpv6,
          ", code: ",
          uv_res,
          ", name: ",
          uv_err_name(uv_res),
          ", message: ",
          uv_strerror(uv_res));

      uv_res =
          uv_listen(res->unsafeGetStream(), DEFAULT_BACKLOG, on_new_connection);
      TORCH_CHECK(
          uv_res == 0,
          "The server socket has failed to listen on any local network address. ",
          "useIpv6: ",
          useIpv6,
          ", code: ",
          uv_res,
          ", name: ",
          uv_err_name(uv_res),
          ", message: ",
          uv_strerror(uv_res));

      res->cacheSocketPort();
    } catch (std::exception& ex) {
      res->close();
      throw;
    }

    return res;
  }

  uint16_t port() const {
    return portNum;
  }

  void accept(const c10::intrusive_ptr<UvTcpSocket>& socket) {
    int res =
        uv_accept(unsafeGetStream(), (uv_stream_t*)socket->unsafeGetHandle());
    TORCH_CHECK(
        res == 0,
        "Failed to accept socket. ",
        "code: ",
        res,
        ", name: ",
        uv_err_name(res),
        ", message: ",
        uv_strerror(res));
  }

 private:
  OnConnectCallback onConnectCb;
  uint16_t portNum{};

  c10::intrusive_ptr<UvTcpServer> iptr() {
    return c10::intrusive_ptr<UvTcpServer>::reclaim_copy(this);
  }

  static c10::intrusive_ptr<UvTcpServer> borrow(uv_stream_t* handle) {
    auto h = (UvTcpServer*)uv_handle_get_data((uv_handle_t*)handle);
    return h->iptr();
  }

  void cacheSocketPort() {
    sockaddr_storage addr_s{};

    int addr_len = sizeof(addr_s);

    if (uv_tcp_getsockname(
            (uv_tcp_t*)unsafeGetStream(),
            reinterpret_cast<sockaddr*>(&addr_s),
            &addr_len) != 0) {
      throw std::runtime_error(
          "The port number of the socket cannot be retrieved.");
    }

    if (addr_s.ss_family == AF_INET) {
      portNum = ntohs(reinterpret_cast<sockaddr_in*>(&addr_s)->sin_port);
    } else {
      portNum = ntohs(reinterpret_cast<sockaddr_in6*>(&addr_s)->sin6_port);
    }
  }

  static void missingOnConnect(int status) {
    TORCH_CHECK(false, "Socket accepted byt onConnect callback missing");
  }

  static void on_new_connection(uv_stream_t* server, int status) {
    borrow(server)->onConnectCb(status);
  }
};

class WriterPayload : public c10::intrusive_ptr_target {
  static c10::intrusive_ptr<WriterPayload> reclaim(uv_write_t* request) {
    /* This method returns a intrusive_ptr that does not increase the refcount.
     */
    auto h = (WriterPayload*)uv_req_get_data((uv_req_t*)request);
    return c10::intrusive_ptr<WriterPayload>::reclaim(h);
  }

  void registeredInLoop() {
    /*
    This refcount increment must be matched by a reclaim call.
    Call this method after sucessfully scheduling this handle with a loop.
    */
    at::raw::intrusive_ptr::incref(this);
  }

  static void write_done(uv_write_t* req, int status) {
    /* Since we're no longer actively used by the event loop, transfer ownership
     * to this frame. */
    auto wp = WriterPayload::reclaim(req);
    auto handle = wp->handle;

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
  uv_write_t req = {};
  uv_buf_t buf = {};
  c10::intrusive_ptr<UvHandle> handle;

 public:
  WriterPayload(
      std::vector<uint8_t>&& in_data,
      c10::intrusive_ptr<UvHandle> handle)
      : data(std::move(in_data)), handle(std::move(handle)) {
    uv_req_set_data((uv_req_t*)&req, this);
  }

  ~WriterPayload() override = default;

  void send() {
    buf = uv_buf_init((char*)data.data(), data.size());
    int res = uv_write(
        &req, (uv_stream_t*)handle->unsafeGetHandle(), &buf, 1, write_done);

    if (res) {
      C10D_INFO(
          "Write setup to client failed. code:{} name:{} desc:{}.",
          res,
          uv_err_name(res),
          uv_strerror(res));
      handle->close();
    } else {
      /* This object was successfully registered with the event loop, so keep it
       * alive until it's unregistered. */
      registeredInLoop();
    }
  }
};

class StreamWriter {
  std::vector<uint8_t> data;
  c10::intrusive_ptr<UvHandle> handle;

  // must be stack allocated
  void* operator new(size_t);

 public:
  StreamWriter(c10::intrusive_ptr<UvHandle> handle)
      : handle(std::move(handle)) {}

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
    auto wd = c10::make_intrusive<WriterPayload>(std::move(data), handle);
    wd->send();
  }
};

class ChunkedStream {
  std::deque<uv_buf_t> buffers;
  size_t buff_idx{0};
  size_t buff_offset{0};
  size_t capacity{0};
  size_t buff_offset_commit{0};
  size_t read_offset{0};

 public:
  ChunkedStream() = default;

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
              "Trying to read past end of buffer. ",
              "buffer_idx: ",
              buff_idx,
              ", available: ",
              buffers.size(),
              ", remaining: ",
              remaining);
        }
      }
    }
    read_offset += size;
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
        "Invalid string size. ",
        "size: ",
        size,
        ", max: ",
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
        "Invalid payload size. ",
        "size: ",
        size_in_bytes,
        ", max: ",
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
  bool waitKeys(
      const std::vector<std::string>& keys,
      const c10::intrusive_ptr<UvHandle>& client);
  int64_t size();
  int64_t deleteKey(const std::string& key);
  void append(const std::string& key, const std::vector<uint8_t>& value);

  void registerClient(const c10::intrusive_ptr<UvHandle>& client);
  void unregisterClient(const c10::intrusive_ptr<UvHandle>& client);
  void clearClientWaitState(const c10::intrusive_ptr<UvHandle>& client);
  bool isMiscellaneousClient(const c10::intrusive_ptr<UvHandle>& client);

  uint16_t get_socket_port(uv_tcp_t* handle);
  void init(const TCPStoreOptions& opts);

 protected:
  void run() override;
  void stop() override;

 private:
  uv_loop_t loop{};
  c10::intrusive_ptr<UvTcpServer> tcpServer;

  uv_async_t exit_handle{};
  std::unordered_map<std::string, std::vector<uint8_t>> tcpStore_;
  // From key -> the list of UvClient waiting on the key
  std::unordered_map<std::string, std::vector<c10::intrusive_ptr<UvHandle>>>
      waitingSockets_;
  // From socket -> number of keys awaited
  std::unordered_map<c10::intrusive_ptr<UvHandle>, size_t> keysAwaited_;
  std::unordered_set<c10::intrusive_ptr<UvHandle>> clients_;
  std::unordered_set<c10::intrusive_ptr<UvHandle>> miscellaneousClients_;
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
  // bool tryListen(bool use_ipv6);

  static void print_active_handles(uv_handle_t* handle, void* arg);
};

class UvClient : public UvTcpSocket {
  ChunkedStream stream;
  LibUVStoreDaemon* store;

 protected:
  void processBuf(const uv_buf_t* buf, size_t nread) override {
    auto tmp = *buf;
    tmp.len = nread;
    stream.append(tmp);

    while (true) {
      stream.reset();
      uint8_t command = -1;
      if (!stream.read1(command))
        break;
      if (store->isMiscellaneousClient(iptr())) {
        if ((QueryType)command != QueryType::VALIDATE)
          return;
        if (!parse_validate_command())
          return;
      } else {
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
      }
      stream.commit();
    }
  }

  bool parse_validate_command() {
    uint32_t validateNumber = 0;
    if (!stream.read_value(validateNumber))
      return false;

    if (validateNumber != c10d::detail::validationMagicNumber)
      return false;
    return true;
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
    StreamWriter sw(iptr());
    sw.write_vector(res);
    sw.send();

    return true;
  }

  bool parse_get_command() {
    std::string key;
    if (!stream.read_key(key))
      return false;

    const auto& data = store->get(key);
    StreamWriter sw(iptr());
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
    StreamWriter sw(iptr());
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
        "Too many keys being waited. ",
        "keys: ",
        key_count,
        ", max: ",
        MAX_KEY_COUNT);

    std::vector<std::string> keys(key_count);
    for (uint64_t i = 0; i < key_count; ++i) {
      if (!stream.read_key(keys[i]))
        return false;
    }

    // Now we have received all the keys
    StreamWriter sw(iptr());
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
        "Too many keys being waited. ",
        "keys: ",
        key_count,
        ", max: ",
        MAX_KEY_COUNT);

    std::vector<std::string> keys(key_count);
    for (uint64_t i = 0; i < key_count; ++i) {
      if (!stream.read_key(keys[i]))
        return false;
    }

    if (store->waitKeys(keys, iptr())) {
      StreamWriter sw(iptr());
      sw.write1((uint8_t)WaitResponseType::STOP_WAITING);
      sw.send();
    }

    return true;
  }

  bool parse_getnumkeys_command() {
    StreamWriter sw(iptr());
    sw.write_value<int64_t>(store->size());
    sw.send();

    return true;
  }

  bool parse_delete_key_command() {
    std::string key;
    if (!stream.read_key(key))
      return false;

    auto numDeleted = store->deleteKey(key);
    StreamWriter sw(iptr());
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
        "Too many keys with multi_get. ",
        "keys: ",
        key_count,
        ", max: ",
        MAX_KEY_COUNT);

    StreamWriter sw(iptr());
    for (const auto _ : c10::irange(key_count)) {
      (void)_; // Suppress unused variable warning
      std::string key;
      if (!stream.read_key(key)) {
        return false;
      }

      sw.write_vector(store->get(key));
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
        "Too many keys with multi_get. ",
        "keys: ",
        key_count,
        ", max: ",
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
    store->clearClientWaitState(iptr());

    StreamWriter sw(iptr());
    sw.write1((uint8_t)WaitResponseType::WAIT_CANCELED);
    sw.send();

    return true;
  }

 public:
  explicit UvClient(uv_loop_t* loop, LibUVStoreDaemon* store)
      : UvTcpSocket(loop), store(store) {}

  static c10::intrusive_ptr<UvClient> make(
      uv_loop_t* loop,
      LibUVStoreDaemon* store) {
    auto res = c10::make_intrusive<UvClient>(loop, store);
    res->handleReady();
    return res;
  }

  c10::intrusive_ptr<UvClient> iptr() {
    return c10::intrusive_ptr<UvClient>::reclaim_copy(this);
  }

 protected:
  void onClose() override {
    store->unregisterClient(iptr());
  }
};

void LibUVStoreDaemon::onConnect(int status) {
  auto client = UvClient::make(&loop, this);
  registerClient(client);
  try {
    tcpServer->accept(client);
    client->startRead();
  } catch (std::exception& e) {
    C10D_INFO("Failed to accept client due to {}", e.what());
    client->close();
  }
}

void LibUVStoreDaemon::onExitRequest() {
  C10D_DEBUG("Store exit requested\n");
  uv_close((uv_handle_t*)&exit_handle, nullptr);
  uv_stop(&loop);
}

void LibUVStoreDaemon::init(const TCPStoreOptions& opts) {
  if (opts.masterListenFd.has_value()) {
    tcpServer = UvTcpServer::makeWithSocket(&loop, *opts.masterListenFd);
  } else {
    try {
      tcpServer = UvTcpServer::makeWithPort(&loop, opts.port, /*useIpv6=*/true);
    } catch (std::exception& ex) {
      C10D_INFO(
          "Failed to bind to ipv6 address, trying ipv4. Error: {}", ex.what());
      tcpServer =
          UvTcpServer::makeWithPort(&loop, opts.port, /*useIpv6=*/false);
    }
  }
  tcpServer->setOnConnectCallback(
      [this](auto status) { this->onConnect(status); });

  port_ = tcpServer->port();
  TORCH_CHECK(
      port_ == opts.port || opts.port == 0, // zero means use any port
      "listen fd ",
      *opts.masterListenFd,
      " is bound to port ",
      port_,
      ", expected to be bound to port ",
      opts.port);
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

void LibUVStoreDaemon::print_active_handles(uv_handle_t* handle, void* arg) {
  C10D_DEBUG(
      "UV live handle type {} active:{} is-closing:{}",
      (int)handle->type,
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
    uv_walk(&loop, LibUVStoreDaemon::print_active_handles, nullptr);
  }

  for (const auto& client : clients_) {
    client->close();
  }
  tcpServer->close();

  if (debug_enabled) {
    C10D_DEBUG("Walking live handles after closing clients");
    uv_walk(&loop, LibUVStoreDaemon::print_active_handles, nullptr);
  }

  while (true) {
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
    if (res != 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
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

bool LibUVStoreDaemon::isMiscellaneousClient(
    const c10::intrusive_ptr<UvHandle>& client) {
  if (miscellaneousClients_.find(client) != miscellaneousClients_.end()) {
    miscellaneousClients_.erase(client);
    return true;
  }
  return false;
}

void LibUVStoreDaemon::registerClient(
    const c10::intrusive_ptr<UvHandle>& client) {
  clients_.insert(client);
  miscellaneousClients_.insert(client);
}

void LibUVStoreDaemon::unregisterClient(
    const c10::intrusive_ptr<UvHandle>& client) {
  clients_.erase(client);
  if (miscellaneousClients_.find(client) != miscellaneousClients_.end()) {
    miscellaneousClients_.erase(client);
  }
  clearClientWaitState(client);
}

void LibUVStoreDaemon::clearClientWaitState(
    const c10::intrusive_ptr<UvHandle>& client) {
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
      pos->second = newValue;
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
    const c10::intrusive_ptr<UvHandle>& client) {
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
    for (const auto& client : socketsToWait->second) {
      if (--keysAwaited_[client] == 0) {
        StreamWriter sw(client->iptr());
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
  TORCH_CHECK(false, "LibUV TCPStore implementation missing");
#endif
}

bool is_libuv_tcpstore_backend_available() {
#ifdef TORCH_USE_LIBUV
  return true;
#else
  return false;
#endif
}

} // namespace c10d::detail
