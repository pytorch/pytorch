#include <c10/core/DeviceGuard.h>
#include <c10/core/impl/COWDeleter.h>
#include <c10/util/Exception.h>
#include <mutex>
#include <utility>

namespace c10::impl {

void cow::cow_deleter(void* ctx) {
  delete static_cast<cow::COWReference*>(ctx);
}

cow::COWDeleterContext::COWDeleterContext(
    std::unique_ptr<void, DeleterFnPtr> data,
    std::optional<Stream> stream,
    uint64_t capture_id)
    : data_(std::move(data)), stream_(stream), capture_id_(capture_id) {
  // We never wrap a COWDeleterContext.
  TORCH_INTERNAL_ASSERT(data_.get_deleter() != cow::cow_deleter);
}

auto cow::COWDeleterContext::increment_refcount() -> void {
  auto refcount = ++refcount_;
  TORCH_INTERNAL_ASSERT(refcount > 1);
}

auto cow::COWDeleterContext::decrement_refcount(
    std::unique_ptr<COWCompletion> completion) -> LastReference {
  if (stream_) {
    std::lock_guard lock(mutex_);
    if (completion) {
      completion->next = std::move(completions_);
      completions_ = std::move(completion);
    } else {
      needs_stream_wait_ = true;
    }
  }
  // Publish dependencies and unlock before the decrement can delete us.
  auto refcount = --refcount_;
  TORCH_INTERNAL_ASSERT(refcount >= 0, refcount);
  if (refcount == 0) {
    auto result = std::move(data_);
    delete this;
    return result;
  }

  return {nullptr, nullptr};
}

void cow::COWDeleterContext::wait_for_completion(Stream stream) {
  TORCH_INTERNAL_ASSERT(is_unique() && stream_);
  const auto* impl = getDeviceGuardImpl(stream.device_type());
  // A unique captured output keeps its address outside capture. Its replay
  // ordering comes from the caller, not from events recorded during capture.
  if (needs_stream_wait_ || impl->getStreamCaptureId(stream) != capture_id_) {
    impl->waitStream(stream, *stream_);
  } else {
    for (auto* node = completions_.get(); node; node = node->next.get()) {
      node->event.block(stream);
    }
  }
}

cow::COWDeleterContext::~COWDeleterContext() {
  TORCH_INTERNAL_ASSERT(refcount_ == 0);
  while (completions_) {
    auto next = std::move(completions_->next);
    completions_ = std::move(next);
  }
}

cow::COWReference::~COWReference() {
  if (ctx) {
    release();
  }
}

auto cow::COWReference::release() -> COWDeleterContext::LastReference {
  auto* context = std::exchange(ctx, nullptr);
  return context->decrement_refcount(std::move(completion));
}

void cow::COWReference::record_done() {
  TORCH_INTERNAL_ASSERT(ctx->stream());
  const auto reuse_stream = *ctx->stream();
  c10::DeviceGuard guard(reuse_stream.device());
  const auto* impl = getDeviceGuardImpl(reuse_stream.device_type());
  const auto stream = impl->getStream(reuse_stream.device());
  TORCH_CHECK(
      impl->getStreamCaptureId(stream) == ctx->capture_id() &&
          impl->getStreamCaptureId(reuse_stream) == ctx->capture_id(),
      "Cannot complete COW uses across CUDA graph capture boundaries");
  if (completion) {
    return;
  }
  auto done = std::make_unique<COWCompletion>(stream.device_type());
  done->event.record(stream);
  // Establish reuse ordering here, while capture state is known. Destruction
  // only publishes this fixed event and drops the reference; it enqueues no
  // work.
  done->event.block(reuse_stream);
  completion = std::move(done);
}

} // namespace c10::impl
