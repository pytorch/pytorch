#pragma once

#include <sys/socket.h>
#include <sys/types.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <string>
#include <system_error>
#include <tuple>
#include <vector>

#include <ATen/ATen.h>

#include <c10d/Types.hpp>

namespace c10d {

// Turns at::IntArrayRef into "(1, 2, 3, 4)".
inline std::string toString(at::IntArrayRef l) {
  std::stringstream ss;
  ss << "(";
  for (size_t i = 0; i < l.size(); i++) {
    if (i > 0) {
      ss << ", ";
    }
    ss << l[i];
  }
  ss << ")";
  return ss.str();
}

inline std::string toString(const c10::Layout& layout) {
  std::stringstream ss;
  ss << layout;
  return ss.str();
}

inline void assertSameType(
    const at::DeprecatedTypeProperties& type,
    const std::vector<at::Tensor>& tensors) {
  for (size_t i = 0; i < tensors.size(); i++) {
    if (tensors[i].type() != type) {
      const std::string expected = type.toString();
      const std::string actual = tensors[i].type().toString();
      throw std::invalid_argument(
          "mixed types (" + expected + " and " + actual + ")");
    }
  }
}

inline void assertSameSizes(
    const at::IntArrayRef& sizes,
    const std::vector<at::Tensor>& tensors) {
  for (size_t i = 0; i < tensors.size(); i++) {
    if (!tensors[i].sizes().equals(sizes)) {
      const auto expected = toString(sizes);
      const auto actual = toString(tensors[i].sizes());
      throw std::invalid_argument(
          "mixed sizes (" + expected + " and " + actual + ")");
    }
  }
}

inline void assertSameSizeAndType(const std::vector<at::Tensor>& tensors) {
  // Ensure we have at least one tensor
  if (tensors.size() == 0) {
    throw std::invalid_argument("argument is empty");
  }

  // Ensure all tensors have identical type and shape
  auto type = tensors[0].type();
  auto sizes = tensors[0].sizes();
  for (size_t i = 1; i < tensors.size(); i++) {
    if (tensors[i].type() != type) {
      const std::string expected = type.toString();
      const std::string actual = tensors[i].type().toString();
      throw std::invalid_argument(
          "argument contains mixed types (" + expected + " and " + actual +
          ")");
    }
    if (!tensors[i].sizes().equals(sizes)) {
      const auto expected = toString(sizes);
      const auto actual = toString(tensors[i].sizes());
      throw std::invalid_argument(
          "argument contains mixed sizes (" + expected + " and " + actual +
          ")");
    }
  }
}

inline void assertTypeMatch(
    std::function<void(const std::string&)> fn,
    const at::DeprecatedTypeProperties& type,
    const at::ArrayRef<at::Tensor>& tensors,
    size_t index) {
  if (tensors[index].type() != type) {
    fn("invalid tensor type at index " + std::to_string(index) + " (expected " +
       type.toString() + ", got " + tensors[index].type().toString() + ")");
  }
}

inline void assertSizesMatch(
    std::function<void(const std::string&)> fn,
    const at::IntArrayRef& sizes,
    const at::ArrayRef<at::Tensor>& tensors,
    size_t index) {
  if (tensors[index].sizes() != sizes) {
    fn("invalid tensor size at index " + std::to_string(index) + " (expected " +
       toString(sizes) + ", got " + toString(tensors[index].sizes()) + ")");
  }
}

inline void assertLayoutMatch(
    std::function<void(const std::string&)> fn,
    const c10::Layout& expected,
    const at::ArrayRef<at::Tensor>& tensors,
    size_t index) {
  const auto& actual = tensors[index].layout();
  if (actual != expected) {
    fn("invalid tensor layout at index " + std::to_string(index) +
       " (expected " + toString(expected) + ", got " + toString(actual) + ")");
  }
}

inline void assertLayoutMatch(
    std::function<void(const std::string&)> fn,
    const at::ArrayRef<at::Tensor>& tensors) {
  const auto& layout = tensors[0].layout();
  for (size_t i = 1; i < tensors.size(); i++) {
    assertLayoutMatch(fn, layout, tensors, i);
  }
}

inline void assertNonEmpty(
    std::function<void(const std::string&)> fn,
    const at::ArrayRef<at::Tensor>& tensors) {
  if (tensors.size() == 0) {
    fn("requires non-empty tensor list");
  }
}

inline void assertSingleElement(
    std::function<void(const std::string&)> fn,
    const at::ArrayRef<at::Tensor>& tensors) {
  if (tensors.size() != 1) {
    fn("requires a single-element tensor list");
  }
}

inline void assertSingleElementInput(
    std::function<void(const std::string&)> fn,
    const at::ArrayRef<at::Tensor>& tensors) {
  if (tensors.size() != 1) {
    fn("requires a single-element input tensor list");
  }
}

inline void assertSingleElementOutput(
    std::function<void(const std::string&)> fn,
    const at::ArrayRef<at::Tensor>& tensors) {
  if (tensors.size() != 1) {
    fn("requires a single-element output tensor list");
  }
}

inline void assertRootRank(
    std::function<void(const std::string&)> fn,
    int rank,
    int size) {
  if (rank < 0 || rank >= size) {
    fn("invalid root rank: " + std::to_string(rank));
  }
}

inline void assertRootTensor(
    std::function<void(const std::string&)> fn,
    int rank,
    int size) {
  if (rank < 0 || rank >= size) {
    fn("invalid root tensor: " + std::to_string(rank));
  }
}

inline void assertDense(
    std::function<void(const std::string&)> fn,
    const at::ArrayRef<at::Tensor>& tensors) {
  const auto& layout = tensors[0].layout();
  if (layout != at::kStrided) {
    fn("only supports dense tensors");
  }
}

inline void assertCPU(
    std::function<void(const std::string&)> fn,
    const at::ArrayRef<at::Tensor>& tensors) {
  const auto& device = tensors[0].device();
  if (device.type() != at::kCPU) {
    fn("only supports CPU tensors");
  }
}

inline void assertTypeAndSizesMatch(
    std::function<void(const std::string&)> fn,
    const at::ArrayRef<at::Tensor>& tensors,
    const at::DeprecatedTypeProperties& type,
    const at::IntArrayRef& sizes) {
  for (size_t i = 0; i < tensors.size(); i++) {
    assertTypeMatch(fn, type, tensors, i);
    assertSizesMatch(fn, sizes, tensors, i);
  }
}

inline void assertTypeAndSizesMatch(
    std::function<void(const std::string&)> fn,
    const at::ArrayRef<at::Tensor>& tensors) {
  const auto& type = tensors[0].type();
  const auto sizes = tensors[0].sizes();
  assertTypeAndSizesMatch(fn, tensors.slice(1), type, sizes);
}

// Copied from ATen/core/functional.h.
template <typename F, typename T>
inline auto fmap(T& inputs, const F& fn)
    -> std::vector<decltype(fn(*inputs.begin()))> {
  std::vector<decltype(fn(*inputs.begin()))> r;
  r.reserve(inputs.size());
  for (auto& input : inputs) {
    r.push_back(fn(input));
  }
  return r;
}

// Copied from torch/csrc/utils/tensor_flatten.h.
inline at::Tensor flattenDenseTensors(at::TensorList tensors) {
  static const auto flatten = [](const at::Tensor& t) {
    return t.contiguous().view({-1});
  };
  if (tensors.size() == 1) {
    return flatten(tensors[0]);
  }
  return at::cat(::c10d::fmap(tensors, flatten));
}

inline at::Tensor newLikeFlat(
    std::vector<std::vector<at::Tensor>>& tensors,
    size_t deviceIdx) {
  if (tensors.size() == 0 || tensors[0].size() == 0) {
    throw std::runtime_error("Received an empty list");
  }
  if (deviceIdx >= tensors.size()) {
    throw std::runtime_error("Invalid device index");
  }
  auto& t = tensors[deviceIdx][0];
  auto device = t.device();
  for (size_t i = 1; i < tensors[deviceIdx].size(); ++i) {
    if (tensors[deviceIdx][i].device() != device) {
      throw std::runtime_error("Expecting all tensors on the same device");
    }
  }
  at::DeviceGuard gpuGuard(device);
  std::vector<int64_t> sizes{static_cast<int64_t>(tensors[deviceIdx].size())};
  sizes.insert(sizes.end(), t.sizes().begin(), t.sizes().end());
  return at::empty(sizes, t.options());
}

inline at::Tensor newLikeFlat(std::vector<at::Tensor>& tensors) {
  if (tensors.size() == 0) {
    throw std::runtime_error("Received an empty list");
  }
  auto& t = tensors[0];
  at::DeviceGuard gpuGuard(t.device());
  std::vector<int64_t> sizes{static_cast<int64_t>(tensors.size())};
  sizes.insert(sizes.end(), t.sizes().begin(), t.sizes().end());
  return at::empty(sizes, t.options());
}

inline std::vector<std::vector<int64_t>> getSizes(
    const std::vector<at::Tensor>& tensors) {
  std::vector<std::vector<int64_t>> sizes(tensors.size());
  for (size_t i = 0; i < tensors.size(); i++) {
    sizes[i] = tensors[i].sizes().vec();
  }
  return sizes;
}

inline std::vector<int> getDevices(const std::vector<at::Tensor>& tensors) {
  std::vector<int> devices(tensors.size(), -1);
  if (tensors[0].type().is_cuda()) {
    for (size_t i = 0; i < tensors.size(); i++) {
      devices[i] = tensors[i].storage().device().index();
    }
  }
  return devices;
}

template <typename T>
inline T* getDataPointer(const at::Tensor& tensor) {
  // This method is only used in ProcessGroupGloo for now. Call sites must make
  // sure that the input tensor is contiguous. It is OK if the tensor does not
  // start from the beginning of the storage. For example, it could come from
  // chunk(..., dim=0)[1]. Hence, we need to use data_ptr() instead of
  // tensor.storage().data()
  // NB: not using tensor.data<T>() because tensor is not aware of gloo::TYPE
  return static_cast<T*>(tensor.data_ptr());
}

template <typename T>
std::vector<T*> getDataPointers(const std::vector<at::Tensor>& tensors) {
  std::vector<T*> ptrs(tensors.size());
  for (size_t i = 0; i < tensors.size(); i++) {
    ptrs[i] = getDataPointer<T>(tensors[i]);
  }
  return ptrs;
}

using RankType = uint32_t;
using PortType = uint16_t;
using SizeType = uint64_t;

// `errno` is only meaningful when it fails. E.g., a  successful `fork()` sets
// `errno` to `EINVAL` in child process on some macos
// (https://stackoverflow.com/a/20295079), and thus `errno` should really only
// be inspected if an error occured.
//
// `success_cond` is an expression used to check if an error has happend. So for
// `fork()`, we can use `SYSCHECK(pid = fork(), pid != -1)`. The function output
// is stored in variable `__output` and may be used in `success_cond`.
#define SYSCHECK(expr, success_cond)                            \
  while (true) {                                                \
    auto __output = (expr);                                     \
    (void)__output;                                             \
    if (!(success_cond)) {                                      \
      if (errno == EINTR) {                                     \
        continue;                                               \
      } else if (errno == EAGAIN || errno == EWOULDBLOCK) {     \
        throw std::runtime_error("Socket Timeout");             \
      } else {                                                  \
        throw std::system_error(errno, std::system_category()); \
      }                                                         \
    } else {                                                    \
      break;                                                    \
    }                                                           \
  }

// Most functions indicate error by returning `-1`. This is a helper macro for
// this common case with `SYSCHECK`.
#define SYSCHECK_ERR_RETURN_NEG1(expr) SYSCHECK(expr, __output != -1)

// Helper resource guard class
class ResourceGuard {
 public:
  ResourceGuard(std::function<void()> destructor)
      : destructor_(std::move(destructor)), released_(false) {}

  ~ResourceGuard() {
    if (!released_) {
      destructor_();
    }
  }

  void release() {
    released_ = true;
  }

 private:
  std::function<void()> destructor_;
  bool released_;
};

namespace tcputil {

constexpr std::chrono::milliseconds kNoTimeout = std::chrono::milliseconds(-1);

// Send and receive
template <typename T>
void sendBytes(
    int socket,
    const T* buffer,
    size_t length,
    bool moreData = false) {
  size_t bytesToSend = sizeof(T) * length;
  if (bytesToSend == 0) {
    return;
  }

  auto bytes = reinterpret_cast<const uint8_t*>(buffer);
  uint8_t* currentBytes = const_cast<uint8_t*>(bytes);

  int flags = 0;

#ifdef MSG_MORE
  if (moreData) { // there is more data to send
    flags |= MSG_MORE;
  }
#endif

  while (bytesToSend > 0) {
    ssize_t bytesSent;
    SYSCHECK_ERR_RETURN_NEG1(
        bytesSent = ::send(socket, currentBytes, bytesToSend, flags))
    if (bytesSent == 0) {
      throw std::system_error(ECONNRESET, std::system_category());
    }

    bytesToSend -= bytesSent;
    currentBytes += bytesSent;
  }
}

template <typename T>
void recvBytes(int socket, T* buffer, size_t length) {
  size_t bytesToReceive = sizeof(T) * length;
  if (bytesToReceive == 0) {
    return;
  }

  auto bytes = reinterpret_cast<uint8_t*>(buffer);
  uint8_t* currentBytes = bytes;

  while (bytesToReceive > 0) {
    ssize_t bytesReceived;
    SYSCHECK_ERR_RETURN_NEG1(
        bytesReceived = ::recv(socket, currentBytes, bytesToReceive, 0))
    if (bytesReceived == 0) {
      throw std::system_error(ECONNRESET, std::system_category());
    }

    bytesToReceive -= bytesReceived;
    currentBytes += bytesReceived;
  }
}

// send a vector's length and data
template <typename T>
void sendVector(int socket, const std::vector<T>& vec, bool moreData = false) {
  SizeType size = vec.size();
  sendBytes<SizeType>(socket, &size, 1, true);
  sendBytes<T>(socket, vec.data(), size, moreData);
}

// receive a vector as sent in sendVector
template <typename T>
std::vector<T> recvVector(int socket) {
  SizeType valueSize;
  recvBytes<SizeType>(socket, &valueSize, 1);
  std::vector<T> value(valueSize);
  recvBytes<T>(socket, value.data(), value.size());
  return value;
}

// this is only for convenience when sending rvalues
template <typename T>
void sendValue(int socket, const T& value, bool moreData = false) {
  sendBytes<T>(socket, &value, 1, moreData);
}

template <typename T>
T recvValue(int socket) {
  T value;
  recvBytes<T>(socket, &value, 1);
  return value;
}

// send a string's length and data
inline void sendString(
    int socket,
    const std::string& str,
    bool moreData = false) {
  SizeType size = str.size();
  sendBytes<SizeType>(socket, &size, 1, true);
  sendBytes<char>(socket, str.data(), size, moreData);
}

// receive a string as sent in sendString
inline std::string recvString(int socket) {
  SizeType valueSize;
  recvBytes<SizeType>(socket, &valueSize, 1);
  std::vector<char> value(valueSize);
  recvBytes<char>(socket, value.data(), value.size());
  return std::string(value.data(), value.size());
}

// Other helpers
std::string sockaddrToString(struct sockaddr* addr);

std::pair<int, PortType> listen(PortType port);

int connect(
    const std::string& address,
    PortType port,
    bool wait = true,
    const std::chrono::milliseconds& timeout = kNoTimeout);

std::tuple<int, std::string> accept(
    int listenSocket,
    const std::chrono::milliseconds& timeout = kNoTimeout);

} // namespace tcputil
} // namespace c10d
