#include <c10/core/impl/cow/materialize.h>

#include <c10/core/Allocator.h>
#include <c10/core/StorageImpl.h>
#include <c10/core/impl/cow/context.h>
#include <c10/core/impl/cow/deleter.h>
#include <c10/util/Exception.h>

#include <cstring>
#include <memory>
#include <optional>
#include <utility>
#include <variant>

namespace c10::impl {

auto C10_API cow::materialize(StorageImpl& storage) -> void {
  at::DataPtr& data_ptr = storage.mutable_data_ptr();

  auto* ctx = data_ptr.cast_context<cow::Context>(cow::delete_context);
  TORCH_INTERNAL_ASSERT(ctx != nullptr);

  auto result = ctx->decrement_refcount();

  // This must be set by each of the branches below.
  std::optional<DataPtr> new_data_ptr;

  if (std::holds_alternative<cow::Context::LastReference>(result)) {
    // This is the only alias left on the data. If there were any
    // racing writes, the context ensured they finished before giving
    // us the result.
    std::unique_ptr<void, DeleterFnPtr> data =
        std::get<cow::Context::LastReference>(std::move(result));
    TORCH_INTERNAL_ASSERT(data.get() == data_ptr.get());
    new_data_ptr = DataPtr(
        data.release(),
        data_ptr.mutable_get(),
        data.get_deleter(),
        data_ptr.device());
  } else {
    TORCH_INTERNAL_ASSERT(
        std::holds_alternative<cow::Context::NotLastReference>(result));
    // We don't need to consume the result, it's just a shared lock
    // ensuring that the data will remain while we copy it.
    new_data_ptr = storage.allocator()->allocate(storage.nbytes());
    std::memcpy(new_data_ptr->get(), data_ptr.get(), storage.nbytes());
  }

  TORCH_INTERNAL_ASSERT(new_data_ptr.has_value());
  DataPtr old_data_ptr = storage.set_data_ptr(*std::move(new_data_ptr));
  old_data_ptr.release_context(); // already deleted by decrement_refcount
}

} // namespace c10::impl
