#include <c10/core/impl/cow/shadow_storage.h>

#include <c10/util/Exception.h>

#include <limits>

namespace c10::impl {

template <bool intrusive>
cow::detail::ShadowStorageImpl<intrusive>::ShadowStorageImpl(
    std::uint64_t generation) noexcept
    : generation_(generation) {}

template <bool intrusive>
auto cow::detail::ShadowStorageImpl<intrusive>::generation() const noexcept
    -> std::uint64_t {
  return generation_;
}

template <bool intrusive>
auto cow::detail::ShadowStorageImpl<intrusive>::bump_generation() noexcept
    -> std::uint64_t {
  TORCH_INTERNAL_ASSERT(
      generation_ != std::numeric_limits<std::uint64_t>::max());
  return ++generation_;
}

template class cow::detail::ShadowStorageImpl<true>;
template class cow::detail::ShadowStorageImpl<false>;

} // namespace c10::impl
