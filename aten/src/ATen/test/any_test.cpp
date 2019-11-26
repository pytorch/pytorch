#include <gtest/gtest.h>

#include <ATen/ATen.h>

#include <c10/util/any.h>
#include <c10/util/ordered_dict.h>

void func1(at::Tensor x) {
  x.add_(1);
}

at::Tensor func2(const at::Tensor& x) {
  return x.add(2);
}

TEST(AnyTest, Basic) {
  c10::OrderedDict<int, c10::any> hooks_dict;

  hooks_dict.insert(1, std::function<void(at::Tensor)>(func1));
  hooks_dict.insert(2, std::function<at::Tensor(const at::Tensor&)>(func2));

  auto hook1 = c10::any_cast<std::function<void(at::Tensor)>>(hooks_dict[1]);
  auto* hook1_ptr = hook1.target<void(*)(at::Tensor)>();
  ASSERT_TRUE(hook1_ptr && *hook1_ptr == func1);

  auto hook2 = c10::any_cast<std::function<at::Tensor(const at::Tensor&)>>(hooks_dict[2]);
  auto* hook2_ptr = hook2.target<at::Tensor(*)(const at::Tensor&)>();
  ASSERT_TRUE(hook2_ptr && *hook2_ptr == func2);

  ASSERT_THROW(
    c10::any_cast<std::function<at::Tensor(const at::Tensor&)>>(hooks_dict[1]),
    c10::bad_any_cast);
}
