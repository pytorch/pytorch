#include <gtest/gtest.h>

#include <torch/torch.h>
#include <torch/csrc/distributed/rpc/utils.h>

#include <memory>
#include <string>
#include <vector>

using namespace torch::distributed::rpc;

TEST(WireSerialize, Base) {
  auto run = [](const std::string& payload,
                const std::vector<at::Tensor>& tensors) {
    std::string serialized;
    {
      std::vector<char> mpayload(payload.begin(), payload.end());
      std::vector<at::Tensor> mtensors = tensors;
      serialized = torch::distributed::rpc::wireSerialize(
          std::move(mpayload), std::move(mtensors));
    }
    auto deser = torch::distributed::rpc::wireDeserialize(
        serialized.data(), serialized.size());
    EXPECT_EQ(payload.size(), deser.first.size());
    EXPECT_EQ(tensors.size(), deser.second.size());
    if (payload.size() > 0) {
      EXPECT_TRUE(
          memcmp(deser.first.data(), payload.data(), payload.size()) == 0);
    }
    for (size_t i = 0; i < tensors.size(); ++i) {
      EXPECT_TRUE(torch::equal(tensors[i], deser.second[i]));
    }
  };
  run("", {});
  run("hi", {});
  run("", {torch::randn({5, 5})});
  run("hi", {torch::randn({5, 5})});
  run("more", {torch::randn({5, 5}), torch::rand({10, 10})});
}

TEST(WireSerialize, RecopySparseTensors) {
  // Take a 1K row of a 1M tensors, and make sure we don't send across 1M rows.
  constexpr size_t k1K = 1024;
  at::Tensor main = torch::randn({k1K, k1K});
  at::Tensor tiny = main.select(0, 2); // Select a row in the middle
  EXPECT_EQ(tiny.numel(), k1K);
  EXPECT_EQ(tiny.storage().numel(), k1K * k1K);
  auto ser = torch::distributed::rpc::wireSerialize({}, {tiny});
  auto deser = torch::distributed::rpc::wireDeserialize(ser.data(), ser.size());
  EXPECT_TRUE(torch::equal(tiny, deser.second[0]));
  EXPECT_LT(ser.size(), (tiny.element_size() * k1K) + k1K);
}
