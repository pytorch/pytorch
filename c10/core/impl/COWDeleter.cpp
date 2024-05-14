#include <c10/core/impl/COWDeleter.h>
#include <c10/util/Exception.h>
#include <iostream>
#include <mutex>

namespace c10::impl {

void cow::cow_deleter(void* ctx) {
  static_cast<cow::COWDeleterContext*>(ctx)->decrement_refcount();
}

void cow::cowsim_deleter(void* ctx) {
  static_cast<cow::COWSimDeleterContext*>(ctx)->decrement_refcount();
}

cow::COWDeleterContext::COWDeleterContext(
    std::unique_ptr<void, DeleterFnPtr> data)
    : data_(std::move(data)) {
  // We never wrap a COWDeleterContext.
  TORCH_INTERNAL_ASSERT(data_.get_deleter() != cow::cow_deleter);

  // We never wrap a COWSimDeleterContext.
  TORCH_INTERNAL_ASSERT(data_.get_deleter() != cow::cowsim_deleter);
}

auto cow::COWDeleterContext::increment_refcount() -> void {
  auto refcount = ++refcount_;
  TORCH_INTERNAL_ASSERT(refcount > 1);
}

auto cow::COWDeleterContext::decrement_refcount()
    -> std::variant<NotLastReference, LastReference> {
  auto refcount = --refcount_;
  TORCH_INTERNAL_ASSERT(refcount >= 0, refcount);
  if (refcount == 0) {
    std::unique_lock lock(mutex_);
    auto result = std::move(data_);
    lock.unlock();
    delete this;
    return {std::move(result)};
  }

  return std::shared_lock(mutex_);
}

cow::COWDeleterContext::~COWDeleterContext() {
  TORCH_INTERNAL_ASSERT(refcount_ == 0);
}

cow::COWSimDeleterContext::COWSimDeleterContext(
    std::unique_ptr<void, DeleterFnPtr> data)
    : cow::COWDeleterContext(std::move(data)),
      has_first_writer_(false),
      has_raised_(false),
      first_writer_(0) {}

void cow::COWSimDeleterContext::raise_warning(cow::AccessType access_type) {
  if (!has_raised_) {
    // TODO: Improve this message
    TORCH_WARN(
        "Detected divergent behavior on ",
        (access_type == cow::AccessType::READ) ? "read" : "write");
    has_raised_ = true;
  }
}

void cow::COWSimDeleterContext::check_write(cow::COWSimAccessorID writer) {
  if (!has_first_writer_) {
    has_first_writer_ = true;
    first_writer_ = writer;
  } else if (writer != first_writer_) {
    raise_warning(cow::AccessType::WRITE);
  }
}

void cow::COWSimDeleterContext::check_read(cow::COWSimAccessorID reader) {
  if (has_first_writer_ && reader != first_writer_) {
    raise_warning(cow::AccessType::READ);
  }
}

} // namespace c10::impl
