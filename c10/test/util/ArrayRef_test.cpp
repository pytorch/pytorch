#include <c10/util/ArrayRef.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <utility>
#include <vector>

namespace {

template <typename T>
class ctor_from_container_test_span_ {
  T* data_;
  std::size_t sz_;

 public:
  template <typename V = std::vector<std::remove_const_t<T>>>
  constexpr explicit ctor_from_container_test_span_(
      std::conditional_t<std::is_const_v<T>, const V, V>& vec) noexcept
      : data_(vec.data()), sz_(vec.size()) {}

  [[nodiscard]] constexpr auto data() const noexcept {
    return data_;
  }

  [[nodiscard]] constexpr auto size() const noexcept {
    return sz_;
  }
};

TEST(ArrayRefTest, ctor_from_container_test) {
  using value_type = int;
  std::vector<value_type> test_vec{1, 6, 32, 4, 68, 3, 7};
  const ctor_from_container_test_span_<value_type> test_mspan{test_vec};
  const ctor_from_container_test_span_<const value_type> test_cspan{
      std::as_const(test_vec)};

  const auto test_ref_mspan = c10::ArrayRef<value_type>(test_mspan);
  const auto test_ref_cspan = c10::ArrayRef<value_type>(test_cspan);

  EXPECT_EQ(std::as_const(test_vec), test_ref_mspan);
  EXPECT_EQ(std::as_const(test_vec), test_ref_cspan);
}

} // namespace
