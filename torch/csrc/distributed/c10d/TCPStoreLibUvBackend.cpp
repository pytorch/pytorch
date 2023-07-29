#include <algorithm>
#include <deque>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <fmt/format.h>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#include <torch/csrc/distributed/c10d/TCPStoreBackend.hpp>

#ifdef TORCH_USE_LIBUV
#include <uv.h>
#endif

namespace c10d {
namespace detail {

std::unique_ptr<BackgroundThread> create_libuv_tcpstore_backend(
    const TCPStoreOptions& opts) {
#ifdef TORCH_USE_LIBUV
  TORCH_CHECK(false, "Libuv implementation missing");
#else
  TORCH_CHECK(false, "Libuv not available");
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
