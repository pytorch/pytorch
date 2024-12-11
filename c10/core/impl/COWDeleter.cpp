#include <c10/core/impl/COWDeleter.h>
#include <c10/util/Exception.h>
#include <mutex>

namespace c10::impl {

void cow::cow_deleter(void* ctx) {
  static_cast<cow::COWDeleterContext*>(ctx)->decrement_refcount();
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

} // namespace c10::impl

namespace c10::impl::cow {

void cowsim_deleter(void* ctx) {
  static_cast<COWSimDeleterContext*>(ctx)->decrement_refcount();
}

COWSimDeleterContext::COWSimDeleterContext(
    std::unique_ptr<void, DeleterFnPtr> data)
    : COWDeleterContext(std::move(data)),
      has_first_writer_(false),
      has_raised_(false),
      first_writer_(0) {}

enum class AccessType { READ, WRITE };

void COWSimDeleterContext::raise_warning(AccessType access_type) {
  if (!has_raised_) {
    // TODO: Improve this message
    alert_cowsim(
        "Detected divergent behavior on ",
        (access_type == AccessType::READ) ? "read" : "write");
    has_raised_ = true;
  }
}

void COWSimDeleterContext::check_write(COWSimAccessorID writer) {
  if (!has_first_writer_) {
    if (get_extra_conditional_view_warnings()) {
      alert_cowsim("Detected first write to a deprecated conditional view");
    }
    has_first_writer_ = true;
    first_writer_ = writer;
  } else if (writer != first_writer_) {
    raise_warning(AccessType::WRITE);
  }
}

void COWSimDeleterContext::check_read(COWSimAccessorID reader) {
  if (has_first_writer_ && reader != first_writer_) {
    raise_warning(AccessType::READ);
  }
}

static bool _extra_conditional_view_warnings = false;

C10_API void set_extra_conditional_view_warnings(bool mode) {
  _extra_conditional_view_warnings = mode;
}

C10_API bool get_extra_conditional_view_warnings() {
  return _extra_conditional_view_warnings;
}

static bool _error_on_conditional_view_warnings = false;

C10_API void set_error_on_conditional_view_warnings(bool mode) {
  _error_on_conditional_view_warnings = mode;
}

C10_API bool get_error_on_conditional_view_warnings() {
  return _error_on_conditional_view_warnings;
}

} // namespace c10::impl::cow
