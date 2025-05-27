#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <mutex>

namespace {
// We use both a mutex and an atomic here because:
// (1) Mutex is needed during writing:
//     We need to first check the value and potentially error,
//     before setting the value (without any one else racing in the middle).
//     It's also totally fine for this to be slow, since it happens exactly once
//     at import time.
// (2) Atomic is needed during reading:
//     Whenever a user prints a privateuse1 device name, they need to read this
//     variable. Although unlikely, we'll data race if someone else is trying to
//     set this variable at the same time that another thread is print the
//     device name. We could re-use the same mutex, but reading the atomic will
//     be much faster.
std::atomic<bool> privateuse1_backend_name_set;
std::string privateuse1_backend_name;
std::mutex privateuse1_lock;
} // namespace

namespace torch::standalone {
std::string get_privateuse1_backend(bool lower_case) {
  // Applying the same atomic read memory ordering logic as in Note [Memory
  // ordering on Python interpreter tag].
  auto name_registered =
      privateuse1_backend_name_set.load(std::memory_order_acquire);
  // Guaranteed that if the flag is set, then privateuse1_backend_name has been
  // set, and will never be written to.
  auto backend_name =
      name_registered ? privateuse1_backend_name : "privateuseone";
  auto op_case = lower_case ? ::tolower : ::toupper;
  std::transform(
      backend_name.begin(), backend_name.end(), backend_name.begin(), op_case);
  return backend_name;
}
} // namespace torch::standalone

namespace c10 {
void register_privateuse1_backend(const std::string& backend_name) {
  std::lock_guard<std::mutex> guard(privateuse1_lock);
  TORCH_CHECK(
      !privateuse1_backend_name_set.load() ||
          privateuse1_backend_name == backend_name,
      "torch.register_privateuse1_backend() has already been set! Current backend: ",
      privateuse1_backend_name);

  static const std::array<std::string, 6> types = {
      "cpu", "cuda", "hip", "mps", "xpu", "mtia"};
  TORCH_CHECK(
      std::find(types.begin(), types.end(), backend_name) == types.end(),
      "Cannot register privateuse1 backend with in-tree device name: ",
      backend_name);

  privateuse1_backend_name = backend_name;
  // Invariant: once this flag is set, privateuse1_backend_name is NEVER written
  // to.
  privateuse1_backend_name_set.store(true, std::memory_order_relaxed);
}

bool is_privateuse1_backend_registered() {
  return privateuse1_backend_name_set.load(std::memory_order_acquire);
}
} // namespace c10
