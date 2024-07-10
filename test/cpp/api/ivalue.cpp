#include <gtest/gtest.h>

#include <ATen/core/ivalue.h>

#include <c10/util/flat_hash_map.h>
#include <c10/util/irange.h>
#include <c10/util/tempfile.h>

#include <torch/torch.h>

#include <test/cpp/api/support.h>

#include <cstdio>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace torch::test;
using namespace torch::nn;
using namespace torch::optim;

TEST(IValueTest, DeepcopyTensors) {
  torch::Tensor t0 = torch::randn({2, 3});
  torch::Tensor t1 = torch::randn({3, 4});
  torch::Tensor t2 = t0.detach();
  torch::Tensor t3 = t0;
  torch::Tensor t4 = t1.as_strided({2, 3}, {3, 1}, 2);
  std::vector<torch::Tensor> tensor_vector = {t0, t1, t2, t3, t4};
  c10::List<torch::Tensor> tensor_list(tensor_vector);
  torch::IValue tensor_list_ivalue(tensor_list);

  c10::IValue::CompIdentityIValues ivalue_compare;

  // Make sure our setup configuration is correct
  ASSERT_TRUE(ivalue_compare(tensor_list[0].get(), tensor_list[3].get()));
  ASSERT_FALSE(ivalue_compare(tensor_list[0].get(), tensor_list[1].get()));
  ASSERT_FALSE(ivalue_compare(tensor_list[0].get(), tensor_list[2].get()));
  ASSERT_FALSE(ivalue_compare(tensor_list[1].get(), tensor_list[4].get()));
  ASSERT_TRUE(tensor_list[0].get().isAliasOf(tensor_list[2].get()));

  c10::IValue copied_ivalue = tensor_list_ivalue.deepcopy();
  c10::List<torch::IValue> copied_list = copied_ivalue.toList();

  // Make sure our setup configuration is correct
  ASSERT_TRUE(ivalue_compare(copied_list[0].get(), copied_list[3].get()));
  ASSERT_FALSE(ivalue_compare(copied_list[0].get(), copied_list[1].get()));
  ASSERT_FALSE(ivalue_compare(copied_list[0].get(), copied_list[2].get()));
  ASSERT_FALSE(ivalue_compare(copied_list[1].get(), copied_list[4].get()));
  // NOTE: this is actually incorrect. Ideally, these _should_ be aliases.
  ASSERT_FALSE(copied_list[0].get().isAliasOf(copied_list[2].get()));

  ASSERT_TRUE(copied_list[0].get().toTensor().allclose(
      tensor_list[0].get().toTensor()));
  ASSERT_TRUE(copied_list[1].get().toTensor().allclose(
      tensor_list[1].get().toTensor()));
  ASSERT_TRUE(copied_list[2].get().toTensor().allclose(
      tensor_list[2].get().toTensor()));
  ASSERT_TRUE(copied_list[3].get().toTensor().allclose(
      tensor_list[3].get().toTensor()));
  ASSERT_TRUE(copied_list[4].get().toTensor().allclose(
      tensor_list[4].get().toTensor()));
}
