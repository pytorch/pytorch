#pragma once

#include <functional>
#include <memory>
#include <mutex>

namespace torch::comms {

class RemovableHandle {
 public:
  explicit RemovableHandle(std::function<void()>&& callback)
      : callback_(std::move(callback)) {}
  ~RemovableHandle() = default;

  RemovableHandle(const RemovableHandle&) = delete;
  RemovableHandle& operator=(const RemovableHandle&) = delete;
  RemovableHandle(RemovableHandle&&) = delete;
  RemovableHandle& operator=(RemovableHandle&&) = delete;

  void remove() {
    // NOLINTNEXTLINE(facebook-hte-c10::call_once)
    c10::call_once(once_, [this]() noexcept {
      callback_();
      callback_ = nullptr;
    });
  }

  // Factory function to create a unique_ptr to RemovableHandle
  // This allows the handle to be moved via the unique_ptr
  static std::unique_ptr<RemovableHandle> create(
      std::function<void()>&& callback) {
    return std::unique_ptr<RemovableHandle>(
        new RemovableHandle(std::move(callback)));
  }

 private:
  std::function<void()> callback_;
  c10::once_flag once_;
};

} // namespace torch::comms
