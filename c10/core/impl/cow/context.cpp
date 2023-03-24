#include <c10/core/impl/cow/context.h>

namespace c10::impl {

cow::Context::Context(std::unique_ptr<void, DeleterFnPtr> data)
    : data_(std::move(data)) {
  assert(data_.get_deleter() != delete_instance); // we never wrap a Context
}

/* static */ auto cow::Context::delete_instance(void* ctx) -> void {
  static_cast<cow::Context*>(ctx)->decrement_refcount();
}

auto cow::Context::refcount() const -> std::int64_t {
  auto refcount = refcount_.load();
  assert(refcount > 0);
  return refcount;
}

auto cow::Context::increment_refcount() -> void {
  [[maybe_unused]] auto refcount = ++refcount_;
  assert(refcount >= 1);
}

auto cow::Context::decrement_refcount() -> void {
  auto refcount = --refcount_;
  assert(refcount >= 0);
  if (refcount == 0) {
    delete this;
  }
}

cow::Context::~Context() {
  assert(refcount_ == 0);
}

} // namespace c10::impl
