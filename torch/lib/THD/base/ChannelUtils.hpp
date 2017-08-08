#pragma once

#include <THPP/Tensor.hpp>
#include <sys/socket.h>
#include <sys/types.h>
#include <cstdlib>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <system_error>
#include <tuple>
#include <vector>


inline void hash_combine(std::size_t& seed) { }

template <typename T, typename... Rest>
inline void hash_combine(std::size_t& seed, const T& v, Rest... rest) {
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  hash_combine(seed, rest...);
}

#define MAKE_HASHABLE(type, ...)                                              \
  namespace std {                                                             \
    template<> struct hash<type> {                                            \
      std::size_t operator()(const type &t) const {                           \
        std::size_t ret = 0;                                                  \
        hash_combine(ret, __VA_ARGS__);                                       \
        return ret;                                                           \
      }                                                                       \
    };                                                                        \
  }


namespace thd {

enum class CollectiveType : std::uint8_t {
  ALL_GATHER = 0,
  GATHER,
  SCATTER,
  ALL_REDUCE,
  REDUCE,
  BROADCAST,
  SEND,
  BARRIER,
  LAST
};

enum class DeviceType : std::uint8_t {
  CPU,
  CUDA,
  LAST
};

inline DeviceType getDeviceType(thpp::Tensor& tensor) {
    return tensor.isCuda() ? DeviceType::CUDA : DeviceType::CPU;
}

} // namespace thd

MAKE_HASHABLE(::thd::CollectiveType, static_cast<std::uint8_t>(t));
MAKE_HASHABLE(::thd::DeviceType, static_cast<std::uint8_t>(t));


namespace thd {

using rank_type = std::uint32_t;
using port_type = std::uint16_t;
using size_type = std::uint64_t;

#define SYSCHECK(expr) { \
  errno = 0; (expr);     \
  if (errno != 0) throw std::system_error(errno, std::system_category()); \
}

template<typename T>
void send_bytes(int socket, const T* buffer, std::size_t length, bool more_data = false)
{
  std::size_t bytes_to_send = sizeof(T) * length;
  if (bytes_to_send == 0)
    return;

  auto bytes = reinterpret_cast<const std::uint8_t*>(buffer);
  std::uint8_t *current_bytes = const_cast<std::uint8_t*>(bytes);

  int flags = 0;
#ifdef MSG_MORE
  if (more_data) { // there is more data to send
    flags |= MSG_MORE;
  }
#endif

  while (bytes_to_send > 0) {
    ssize_t bytes_sent;
    SYSCHECK(bytes_sent = ::send(socket, current_bytes, bytes_to_send, flags))
    if (bytes_sent == 0)
      throw std::system_error(ECONNRESET, std::system_category());

    bytes_to_send -= bytes_sent;
    current_bytes += bytes_sent;
  }
}


template<typename T>
void recv_bytes(int socket, T* buffer, std::size_t length)
{
  std::size_t bytes_to_receive = sizeof(T) * length;
  if (bytes_to_receive == 0)
    return;

  auto bytes = reinterpret_cast<std::uint8_t*>(buffer);
  std::uint8_t *current_bytes = bytes;

  while (bytes_to_receive > 0) {
    ssize_t bytes_received;
    SYSCHECK(bytes_received = ::recv(socket, current_bytes, bytes_to_receive, 0))
    if (bytes_received == 0)
      throw std::system_error(ECONNRESET, std::system_category());

    bytes_to_receive -= bytes_received;
    current_bytes += bytes_received;
  }
}

inline port_type convertToPort(long port) {
  if ((port < 0) || (port >= std::numeric_limits<port_type>::max()))
    throw std::domain_error("invalid port (value out of range)");

  return static_cast<port_type>(port);
}

inline rank_type convertToRank(long rank, long min = 0) {
  if ((rank < min) || (rank >= std::numeric_limits<rank_type>::max()))
    throw std::domain_error("invalid rank (value out of range)");

  return static_cast<rank_type>(rank);
}

std::pair<int, port_type> listen(port_type port = 0);
int connect(const std::string& address, port_type port, bool wait = true, int timeout = -1);
std::tuple<int, std::string> accept(int listen_socket, int timeout = -1);

std::string sockaddrToString(struct sockaddr *addr);
std::pair<std::string, std::string> splitAddress(const std::string &addr);

/* send a string's length and data */
inline void send_string(int socket, const std::string& str,
                        bool more_data = false) {
  size_type size = str.size();
  send_bytes<size_type>(socket, &size, 1, true);
  send_bytes<char>(socket, str.data(), size, more_data);
}

/* receive a string as sent in send_string */
inline std::string recv_string(int socket) {
  size_type value_size;
  recv_bytes<size_type>(socket, &value_size, 1);
  std::vector<char> value(value_size);
  recv_bytes<char>(socket, value.data(), value.size());
  return std::string(value.data(), value.size());
}

/* send a vector's length and data */
template<typename T>
void send_vector(int socket, const std::vector<T>& vec,
                 bool more_data = false) {
  size_type size = vec.size();
  send_bytes<size_type>(socket, &size, 1, true);
  send_bytes<T>(socket, vec.data(), size, more_data);
}

/* receive a vector as sent in send_vector */
template<typename T>
std::vector<T> recv_vector(int socket) {
  size_type value_size;
  recv_bytes<size_type>(socket, &value_size, 1);
  std::vector<char> value(value_size);
  recv_bytes<char>(socket, value.data(), value.size());
  return value;
}

/* this is only for convenience when sending rvalues */
template<typename T>
void send_value(int socket, const T& value, bool more_data = false) {
  send_bytes<T>(socket, &value, 1, more_data);
}

template<typename T>
T recv_value(int socket) {
  T value;
  recv_bytes<T>(socket, &value, 1);
  return value;
}

class ResourceGuard {
  std::function<void()> _destructor;
  bool _released;

public:
  ResourceGuard(std::function<void()> destructor)
    : _destructor(std::move(destructor))
    , _released(false) {}

  ~ResourceGuard() {
    if (!_released) _destructor();
  }

  void release() {
    _released = true;
  }
};

} // namespace thd
