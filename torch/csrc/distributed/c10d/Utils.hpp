#pragma once

#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <c10/util/accumulate.h>
#include <c10/util/irange.h>
#include <torch/csrc/distributed/c10d/Types.hpp>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
typedef SSIZE_T ssize_t;
#pragma comment(lib, "Ws2_32.lib")
#else
#include <fcntl.h>
#include <netdb.h>
#include <sys/poll.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

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

namespace c10d {

// Retrieve tensor shapes from a given tensor.
TORCH_API std::vector<at::Tensor> getTensorShapes(
    const std::vector<at::Tensor>& tensors);

// Use -2 to represent unset state of env vars
#define C10D_ENV_NOT_SET -2

// Turns at::IntArrayRef into "(1, 2, 3, 4)".
inline std::string toString(at::IntArrayRef l) {
  std::stringstream ss;
  ss << "(";
  for (const auto i : c10::irange(l.size())) {
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
  for (const auto i : c10::irange(tensors.size())) {
    if (!tensors[i].options().type_equal(type.options())) {
      const std::string expected = type.toString();
      const std::string actual = tensors[i].toString();
      throw std::invalid_argument(
          "mixed types (" + expected + " and " + actual + ")");
    }
  }
}

inline std::vector<std::string> split(
    char separator,
    const std::string& string) {
  std::vector<std::string> pieces;
  std::stringstream ss(string);
  std::string item;
  while (std::getline(ss, item, separator)) {
    pieces.push_back(std::move(item));
  }
  return pieces;
}

inline std::string getCvarString(
    const std::vector<std::string>& env,
    const char* def) {
  const char* ret = def;

  if (env.empty()) {
    TORCH_CHECK(false, "No environment variables passed");
    return ret;
  }

  /* parse environment variable in reverse order, so the early
   * versions of a variable get higher priority than the latter
   * versions of the same variable */
  for (int i = env.size() - 1; i >= 0; i--) {
    const char* val = std::getenv(env[i].c_str());
    if (val == nullptr) {
      continue;
    } else if (i) {
      TORCH_WARN(
          "Environment variable " + env[i] + " is deprecated; use " + env[0] +
          " instead");
    }

    ret = val;
  }

  return ret;
}

inline int getCvarInt(const std::vector<std::string>& env, int def) {
  int ret = def;

  if (env.empty()) {
    TORCH_CHECK(false, "No environment variables passed");
    return ret;
  }

  /* parse environment variable in reverse order, so the early
   * versions of a variable get higher priority than the latter
   * versions of the same variable */
  for (int i = env.size() - 1; i >= 0; i--) {
    char* val = std::getenv(env[i].c_str());
    if (val == nullptr) {
      continue;
    } else if (i) {
      TORCH_WARN(
          "Environment variable " + env[i] + " is deprecated; use " + env[0] +
          " instead");
    }

    try {
      ret = std::stoi(val);
    } catch (std::exception& e) {
      TORCH_CHECK(false, "Invalid value for environment variable: " + env[i]);
    }
  }

  return ret;
}

inline bool getCvarBool(const std::vector<std::string>& env, bool def) {
  bool ret = def;

  if (env.empty()) {
    TORCH_CHECK(false, "No environment variables passed");
    return ret;
  }

  /* parse environment variable in reverse order, so the early
   * versions of a variable get higher priority than the latter
   * versions of the same variable */
  for (int i = env.size() - 1; i >= 0; i--) {
    char* val_ = std::getenv(env[i].c_str());
    if (val_ == nullptr) {
      continue;
    } else if (i) {
      TORCH_WARN(
          "Environment variable " + env[i] + " is deprecated; use " + env[0] +
          " instead");
    }

    std::string val = std::string(val_);
    for (auto& x : val) {
      x = std::tolower(x);
    }

    if (val == "y" || val == "yes" || val == "1" || val == "t" ||
        val == "true") {
      ret = true;
    } else if (
        val == "n" || val == "no" || val == "0" || val == "f" ||
        val == "false") {
      ret = false;
    } else {
      TORCH_CHECK(false, "Invalid value for environment variable: " + env[i]);
      return ret;
    }
  }

  return ret;
}

inline void assertSameSizes(
    const at::IntArrayRef& sizes,
    const std::vector<at::Tensor>& tensors) {
  for (const auto i : c10::irange(tensors.size())) {
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
  if (tensors.empty()) {
    throw std::invalid_argument("argument is empty");
  }

  // Ensure all tensors have identical type and shape
  auto options = tensors[0].options();
  auto sizes = tensors[0].sizes();
  for (const auto i : c10::irange(1, tensors.size())) {
    if (!tensors[i].options().type_equal(options)) {
      const auto expected = toString(options);
      const auto actual = toString(tensors[i].options());
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
    const at::ArrayRef<at::Tensor> tensors,
    size_t index) {
  if (!tensors[index].options().type_equal(type.options())) {
    fn("invalid tensor type at index " + std::to_string(index) + " (expected " +
       type.toString() + ", got " + tensors[index].toString() + ")");
  }
}

inline void assertTypeMatch(
    std::function<void(const std::string&)> fn,
    const at::TensorOptions& options,
    const at::ArrayRef<at::Tensor> tensors,
    size_t index) {
  if (!tensors[index].options().type_equal(options)) {
    fn("invalid tensor type at index " + std::to_string(index) + " (expected " +
       toString(options) + ", got " + toString(tensors[index].options()) + ")");
  }
}

inline void assertSizesMatch(
    std::function<void(const std::string&)> fn,
    const at::IntArrayRef& sizes,
    const at::ArrayRef<at::Tensor> tensors,
    size_t index) {
  if (tensors[index].sizes() != sizes) {
    fn("invalid tensor size at index " + std::to_string(index) + " (expected " +
       toString(sizes) + ", got " + toString(tensors[index].sizes()) + ")");
  }
}

inline void assertLayoutMatch(
    std::function<void(const std::string&)> fn,
    const c10::Layout& expected,
    const at::ArrayRef<at::Tensor> tensors,
    size_t index) {
  const auto& actual = tensors[index].layout();
  if (actual != expected) {
    fn("invalid tensor layout at index " + std::to_string(index) +
       " (expected " + toString(expected) + ", got " + toString(actual) + ")");
  }
}

inline void assertLayoutMatch(
    std::function<void(const std::string&)> fn,
    const at::ArrayRef<at::Tensor> tensors) {
  const auto& layout = tensors[0].layout();
  for (const auto i : c10::irange(1, tensors.size())) {
    assertLayoutMatch(fn, layout, tensors, i);
  }
}

inline void assertNonEmpty(
    std::function<void(const std::string&)> fn,
    const at::ArrayRef<at::Tensor> tensors) {
  if (tensors.empty()) {
    fn("requires non-empty tensor list");
  }
}

inline void assertSingleElement(
    std::function<void(const std::string&)> fn,
    const at::ArrayRef<at::Tensor> tensors) {
  if (tensors.size() != 1) {
    fn("requires a single-element tensor list");
  }
}

inline void assertSingleElementInput(
    std::function<void(const std::string&)> fn,
    const at::ArrayRef<at::Tensor> tensors) {
  if (tensors.size() != 1) {
    fn("requires a single-element input tensor list");
  }
}

inline void assertSingleElementOutput(
    std::function<void(const std::string&)> fn,
    const at::ArrayRef<at::Tensor> tensors) {
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
    const at::ArrayRef<at::Tensor> tensors) {
  const auto& layout = tensors[0].layout();
  if (layout != at::kStrided) {
    fn("only supports dense tensors");
  }
}

inline void assertCPU(
    std::function<void(const std::string&)> fn,
    const at::ArrayRef<at::Tensor> tensors) {
  const auto& device = tensors[0].device();
  if (device.type() != at::kCPU) {
    fn("only supports CPU tensors");
  }
}

inline void assertSameDevice(
    std::function<void(const std::string&)> fn,
    const at::ArrayRef<at::Tensor> tensors) {
  if (tensors.size() < 2) {
    return;
  }
  const auto& device = tensors[0].device();
  for (const auto i : c10::irange(1, tensors.size())) {
    if (tensors[i].device() != device) {
      fn("tensors should be on the same device");
    }
  }
}

inline void assertTypeAndSizesMatch(
    std::function<void(const std::string&)> fn,
    const at::ArrayRef<at::Tensor> tensors,
    const at::DeprecatedTypeProperties& type,
    const at::IntArrayRef& sizes) {
  for (const auto i : c10::irange(tensors.size())) {
    assertTypeMatch(fn, type, tensors, i);
    assertSizesMatch(fn, sizes, tensors, i);
  }
}

inline void assertTypeAndSizesMatch(
    std::function<void(const std::string&)> fn,
    const at::ArrayRef<at::Tensor> tensors,
    const at::TensorOptions& options,
    const at::IntArrayRef& sizes) {
  for (const auto i : c10::irange(tensors.size())) {
    assertTypeMatch(fn, options, tensors, i);
    assertSizesMatch(fn, sizes, tensors, i);
  }
}

inline void assertTypeAndSizesMatch(
    std::function<void(const std::string&)> fn,
    const at::ArrayRef<at::Tensor> tensors) {
  const auto& options = tensors[0].options();
  const auto sizes = tensors[0].sizes();
  assertTypeAndSizesMatch(fn, tensors.slice(1), options, sizes);
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
  if (tensors.empty() || tensors[0].empty()) {
    TORCH_CHECK(false, "Received an empty list");
  }
  if (deviceIdx >= tensors.size()) {
    TORCH_CHECK(false, "Invalid device index");
  }
  auto& t = tensors[deviceIdx][0];
  auto device = t.device();
  for (const auto i : c10::irange(1, tensors[deviceIdx].size())) {
    if (tensors[deviceIdx][i].device() != device) {
      TORCH_CHECK(false, "Expecting all tensors on the same device");
    }
  }
  at::DeviceGuard gpuGuard(device);
  std::vector<int64_t> sizes{static_cast<int64_t>(tensors[deviceIdx].size())};
  std::vector<int64_t> strides{static_cast<int64_t>(t.numel())};
  sizes.insert(sizes.end(), t.sizes().begin(), t.sizes().end());
  strides.insert(strides.end(), t.strides().begin(), t.strides().end());
  return at::empty_strided(
      sizes, strides, t.options().memory_format(c10::nullopt));
}

inline at::Tensor newLikeFlat(std::vector<at::Tensor>& tensors) {
  if (tensors.empty()) {
    TORCH_CHECK(false, "Received an empty list");
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
  for (const auto i : c10::irange(tensors.size())) {
    sizes[i] = tensors[i].sizes().vec();
  }
  return sizes;
}

inline std::vector<int> getDevices(const std::vector<at::Tensor>& tensors) {
  std::vector<int> devices(tensors.size(), -1);
  if (tensors[0].device().is_cuda()) {
    for (const auto i : c10::irange(tensors.size())) {
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
  for (const auto i : c10::irange(tensors.size())) {
    ptrs[i] = getDataPointer<T>(tensors[i]);
  }
  return ptrs;
}

// For alltoall split size sanity check
inline void checkSplitSizes(
    const std::vector<int64_t>& split_sizes,
    const at::Tensor& tensor,
    int group_size) {
  if (split_sizes.empty()) {
    TORCH_CHECK(
        tensor.size(0) % group_size == 0,
        "Tensor's dim 0 does not divide equally across group size");
  } else {
    TORCH_CHECK(
        split_sizes.size() == static_cast<size_t>(group_size),
        "Number of tensor splits not equal to group size");
    const auto sum = c10::sum_integers(split_sizes);
    TORCH_CHECK(
        sum == tensor.size(0), "Split sizes doesn't match total dim 0 size");
  }
}

// Compute alltoall lengths and offsets, handling multi-dimension tensors
template <typename T>
size_t computeLengthsAndOffsets(
    const std::vector<int64_t>& split_sizes,
    const at::Tensor& tensor,
    std::vector<T>* lengths,
    std::vector<T>* offsets) {
  size_t group_size = lengths->size();
  bool equal_splits = false;
  size_t dim0_size = tensor.size(0);
  size_t row_size = (dim0_size ? tensor.numel() / dim0_size : 1);
  size_t split_size = 0;
  size_t offset = 0;

  if (split_sizes.empty()) {
    equal_splits = true;
    split_size = tensor.size(0) / group_size;
  }
  for (const auto i : c10::irange(group_size)) {
    size_t length = row_size * (equal_splits ? split_size : split_sizes[i]);
    (*lengths)[i] = length;
    (*offsets)[i] = offset;
    // TODO: see if we should add overflow protection for offset
    offset += length;
  }
  return offset;
}

template <typename T>
size_t computeLengthsAndOffsets(
    const std::vector<at::Tensor>& tensors,
    std::vector<T>* lengths,
    std::vector<T>* offsets) {
  size_t group_size = lengths->size();
  size_t offset = 0;
  for (const auto i : c10::irange(group_size)) {
    size_t length = tensors[i].numel();
    (*lengths)[i] = length;
    (*offsets)[i] = offset;
    offset += length;
  }
  return offset;
}

using RankType = uint32_t;
using SizeType = uint64_t;

// `errno` is only meaningful when it fails. E.g., a  successful `fork()` sets
// `errno` to `EINVAL` in child process on some macos
// (https://stackoverflow.com/a/20295079), and thus `errno` should really only
// be inspected if an error occurred.
//
// `success_cond` is an expression used to check if an error has happend. So for
// `fork()`, we can use `SYSCHECK(pid = fork(), pid != -1)`. The function output
// is stored in variable `__output` and may be used in `success_cond`.
#ifdef _WIN32
#define SYSCHECK(expr, success_cond)                                      \
  while (true) {                                                          \
    auto __output = (expr);                                               \
    auto errno_local = WSAGetLastError();                                 \
    (void)__output;                                                       \
    if (!(success_cond)) {                                                \
      if (errno == EINTR) {                                               \
        continue;                                                         \
      } else if (                                                         \
          errno_local == WSAETIMEDOUT || errno_local == WSAEWOULDBLOCK) { \
        C10_THROW_ERROR(DistNetworkError, "Socket Timeout");              \
      } else {                                                            \
        C10_THROW_ERROR(DistNetworkError, std::strerror(errno_local));    \
      }                                                                   \
    } else {                                                              \
      break;                                                              \
    }                                                                     \
  }
#else
#define SYSCHECK(expr, success_cond)                             \
  while (true) {                                                 \
    auto __output = (expr);                                      \
    (void)__output;                                              \
    if (!(success_cond)) {                                       \
      if (errno == EINTR) {                                      \
        continue;                                                \
      } else if (errno == EAGAIN || errno == EWOULDBLOCK) {      \
        C10_THROW_ERROR(DistNetworkError, "Socket Timeout");     \
      } else {                                                   \
        C10_THROW_ERROR(DistNetworkError, std::strerror(errno)); \
      }                                                          \
    } else {                                                     \
      break;                                                     \
    }                                                            \
  }
#endif

// Most functions indicate error by returning `-1`. This is a helper macro for
// this common case with `SYSCHECK`.
// Since SOCKET_ERROR = -1 in MSVC, so also leverage SYSCHECK_ERR_RETURN_NEG1
#define SYSCHECK_ERR_RETURN_NEG1(expr) SYSCHECK(expr, __output != -1)

namespace tcputil {

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

// Ignore SIGPIPE as the send() return value is always checked for error
#ifdef MSG_NOSIGNAL
  flags |= MSG_NOSIGNAL;
#endif

  while (bytesToSend > 0) {
    ssize_t bytesSent;
    SYSCHECK_ERR_RETURN_NEG1(
        bytesSent =
            ::send(socket, (const char*)currentBytes, bytesToSend, flags))
    if (bytesSent == 0) {
      C10_THROW_ERROR(DistNetworkError, std::strerror(ECONNRESET));
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
        bytesReceived = recv(socket, (char*)currentBytes, bytesToReceive, 0))
    if (bytesReceived == 0) {
      C10_THROW_ERROR(DistNetworkError, std::strerror(ECONNRESET));
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

} // namespace tcputil
} // namespace c10d
