#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <c10/core/UndefinedTensorImpl.h>
#include <string>

using namespace at;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(TestUndefined, UndefinedTest) {
  manual_seed(123);

  // mainly test ops on undefined tensors don't segfault and give a reasonable errror message.
  Tensor und;
  Tensor ft = ones({1}, CPU(kFloat));

  std::stringstream ss;
  ss << und << std::endl;
  ASSERT_FALSE(und.defined());
  ASSERT_EQ(std::string("UndefinedType"), und.toString());

  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(und.strides());
  ASSERT_EQ(und.dim(), 1);
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW([]() { return Tensor(); }() = Scalar(5));
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(und.add(und));
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(und.add(ft));
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(ft.add(und));
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(und.add(5));
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(und.mm(und));

  // public variable API
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(und.variable_data());
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(und.tensor_data());
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(und.is_view());
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(und._base());
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(und.name());
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(und.grad_fn());
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(und.remove_hook(0));
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(und.register_hook([](const Tensor& x) -> Tensor { return x; }));

  // copy_
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(und.copy_(und));
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(und.copy_(ft));
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(ft.copy_(und));

  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(und.toBackend(Backend::CPU));
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(ft.toBackend(Backend::Undefined));

  Tensor to_move = ones({1}, CPU(kFloat));
  Tensor m(std::move(to_move));
  // NOLINTNEXTLINE(bugprone-use-after-move)
  ASSERT_FALSE(to_move.defined());
  ASSERT_EQ(to_move.unsafeGetTensorImpl(), UndefinedTensorImpl::singleton());
}
