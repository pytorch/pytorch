#include <catch.hpp>

#include <torch/nn/modules/pooling.h>

using namespace torch::nn;

TEST_CASE("Pooling/MaxPool1d") {
  auto input = torch::arange(0, 10).expand({1, 1, 10});

  {
    auto output = MaxPool1d(3)->forward({input})[0];
    REQUIRE(output.numel() == 3);
    REQUIRE(output.data().allclose(at::tensor({2, 5, 8}, at::kFloat)));
  }
  {
    auto output = MaxPool1d(MaxPool1dOptions(3).stride(1))->forward({input})[0];
    REQUIRE(output.numel() == 8);
    REQUIRE(output.data().allclose(
        at::tensor({2, 3, 4, 5, 6, 7, 8, 9}, at::kFloat)));
  }
  {
    auto output = MaxPool1d(MaxPool1dOptions(3).stride(1).padding(1))
                      ->forward({input})[0];
    REQUIRE(output.numel() == 10);
    REQUIRE(output.data().allclose(
        at::tensor({1, 2, 3, 4, 5, 6, 7, 8, 9, 9}, at::kFloat)));
  }
  {
    auto output = MaxPool1d(MaxPool1dOptions(3).stride(1).dilation(2))
                      ->forward({input})[0];
    REQUIRE(output.numel() == 6);
    REQUIRE(output.data().allclose(at::tensor({4, 5, 6, 7, 8, 9}, at::kFloat)));
  }
  {
    auto output =
        MaxPool1d(MaxPool1dOptions(3).ceil_mode(true))->forward({input})[0];
    REQUIRE(output.numel() == 4);
    REQUIRE(output.data().allclose(at::tensor({2, 5, 8, 9}, at::kFloat)));
  }
}

TEST_CASE("Pooling/MaxPool2d") {
  auto input = torch::arange(0, 25).reshape({1, 1, 5, 5});

  {
    auto output = MaxPool2d(2)->forward({input})[0];
    REQUIRE(output.size(0) == 1);
    REQUIRE(output.size(1) == 1);
    REQUIRE(output.size(2) == 2);
    REQUIRE(output.size(3) == 2);
    REQUIRE(output.data().flatten().allclose(
        at::tensor({6, 8, 16, 18}, at::kFloat)));
  }
  {
    auto output = MaxPool2d(MaxPool2dOptions({2, 1}))->forward({input})[0];
    REQUIRE(output.size(0) == 1);
    REQUIRE(output.size(1) == 1);
    REQUIRE(output.size(2) == 2);
    REQUIRE(output.size(3) == 5);
    REQUIRE(output.data().flatten().allclose(
        at::tensor({5, 6, 7, 8, 9, 15, 16, 17, 18, 19}, at::kFloat)));
  }
  {
    auto output = MaxPool2d(MaxPool2dOptions(2).stride(3))->forward({input})[0];
    REQUIRE(output.size(0) == 1);
    REQUIRE(output.size(1) == 1);
    REQUIRE(output.size(2) == 2);
    REQUIRE(output.size(3) == 2);
    REQUIRE(output.data().flatten().allclose(
        at::tensor({6, 9, 21, 24}, at::kFloat)));
  }
  {
    auto output =
        MaxPool2d(MaxPool2dOptions(2).stride({1, 2}))->forward({input})[0];
    REQUIRE(output.size(0) == 1);
    REQUIRE(output.size(1) == 1);
    REQUIRE(output.size(2) == 4);
    REQUIRE(output.size(3) == 2);
    REQUIRE(output.data().flatten().allclose(
        at::tensor({6, 8, 11, 13, 16, 18, 21, 23}, at::kFloat)));
  }
  {
    auto output =
        MaxPool2d(MaxPool2dOptions(2).padding(1))->forward({input})[0];
    REQUIRE(output.size(0) == 1);
    REQUIRE(output.size(1) == 1);
    REQUIRE(output.size(2) == 3);
    REQUIRE(output.size(3) == 3);
    REQUIRE(output.data().flatten().allclose(
        at::tensor({0, 2, 4, 10, 12, 14, 20, 22, 24}, at::kFloat)));
  }
  {
    auto output =
        MaxPool2d(MaxPool2dOptions(2).padding({0, 1}))->forward({input})[0];
    REQUIRE(output.size(0) == 1);
    REQUIRE(output.size(1) == 1);
    REQUIRE(output.size(2) == 2);
    REQUIRE(output.size(3) == 3);
    REQUIRE(output.data().flatten().allclose(
        at::tensor({5, 7, 9, 15, 17, 19}, at::kFloat)));
  }
  {
    auto output = MaxPool2d(MaxPool2dOptions(2).stride(1).dilation(2))
                      ->forward({input})[0];
    REQUIRE(output.size(0) == 1);
    REQUIRE(output.size(1) == 1);
    REQUIRE(output.size(2) == 3);
    REQUIRE(output.size(3) == 3);
    REQUIRE(output.data().flatten().allclose(
        at::tensor({12, 13, 14, 17, 18, 19, 22, 23, 24}, at::kFloat)));
  }
  {
    auto output =
        MaxPool2d(MaxPool2dOptions(2).dilation({1, 2}))->forward({input})[0];
    REQUIRE(output.size(0) == 1);
    REQUIRE(output.size(1) == 1);
    REQUIRE(output.size(2) == 2);
    REQUIRE(output.size(3) == 2);
    REQUIRE(output.data().flatten().allclose(
        at::tensor({7, 9, 17, 19}, at::kFloat)));
  }
  {
    auto output =
        MaxPool2d(MaxPool2dOptions(3).ceil_mode(true))->forward({input})[0];
    REQUIRE(output.size(0) == 1);
    REQUIRE(output.size(1) == 1);
    REQUIRE(output.size(2) == 2);
    REQUIRE(output.size(3) == 2);
    REQUIRE(output.data().flatten().allclose(
        at::tensor({12, 14, 22, 24}, at::kFloat)));
  }
}

TEST_CASE("Pooling/AvgPool1d") {
  auto input = torch::arange(0, 10).expand({1, 1, 10});

  {
    auto output = AvgPool1d(3)->forward({input})[0];
    REQUIRE(output.numel() == 3);
    REQUIRE(output.data().allclose(at::tensor({1, 4, 7}, at::kFloat)));
  }
  {
    auto output = AvgPool1d(AvgPool1dOptions(3).stride(1))->forward({input})[0];
    REQUIRE(output.numel() == 8);
    REQUIRE(output.data().allclose(
        at::tensor({1, 2, 3, 4, 5, 6, 7, 8}, at::kFloat)));
  }
  {
    auto output = AvgPool1d(AvgPool1dOptions(3).stride(1).padding(1))
                      ->forward({input})[0];
    REQUIRE(output.numel() == 10);
    REQUIRE(output.data().allclose(at::tensor(
        {1.0 / 3, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 17.0 / 3},
        at::kFloat)));
  }
  {
    auto output =
        AvgPool1d(AvgPool1dOptions(3).ceil_mode(true))->forward({input})[0];
    REQUIRE(output.numel() == 4);
    REQUIRE(output.data().allclose(at::tensor({1, 4, 7, 9}, at::kFloat)));
  }
  {
    auto output =
        AvgPool1d(
            AvgPool1dOptions(3).stride(1).padding(1).count_include_pad(false))
            ->forward({input})[0];
    REQUIRE(output.numel() == 10);
    REQUIRE(output.data().allclose(at::tensor(
        {0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.5}, at::kFloat)));
  }
}

TEST_CASE("Pooling/AvgPool2d") {
  auto input = torch::arange(0, 25).reshape({1, 1, 5, 5});

  {
    auto output = AvgPool2d(2)->forward({input})[0];
    REQUIRE(output.size(0) == 1);
    REQUIRE(output.size(1) == 1);
    REQUIRE(output.size(2) == 2);
    REQUIRE(output.size(3) == 2);
    REQUIRE(output.data().flatten().allclose(
        at::tensor({3, 5, 13, 15}, at::kFloat)));
  }
  {
    auto output = AvgPool2d(AvgPool2dOptions({2, 1}))->forward({input})[0];
    REQUIRE(output.size(0) == 1);
    REQUIRE(output.size(1) == 1);
    REQUIRE(output.size(2) == 2);
    REQUIRE(output.size(3) == 5);
    REQUIRE(output.data().flatten().allclose(at::tensor(
        {2.5, 3.5, 4.5, 5.5, 6.5, 12.5, 13.5, 14.5, 15.5, 16.5}, at::kFloat)));
  }
  {
    auto output = AvgPool2d(AvgPool2dOptions(2).stride(3))->forward({input})[0];
    REQUIRE(output.size(0) == 1);
    REQUIRE(output.size(1) == 1);
    REQUIRE(output.size(2) == 2);
    REQUIRE(output.size(3) == 2);
    REQUIRE(output.data().flatten().allclose(
        at::tensor({3, 6, 18, 21}, at::kFloat)));
  }
  {
    auto output =
        AvgPool2d(AvgPool2dOptions(2).stride({1, 2}))->forward({input})[0];
    REQUIRE(output.size(0) == 1);
    REQUIRE(output.size(1) == 1);
    REQUIRE(output.size(2) == 4);
    REQUIRE(output.size(3) == 2);
    REQUIRE(output.data().flatten().allclose(
        at::tensor({3, 5, 8, 10, 13, 15, 18, 20}, at::kFloat)));
  }
  {
    auto output =
        AvgPool2d(AvgPool2dOptions(2).padding(1))->forward({input})[0];
    REQUIRE(output.size(0) == 1);
    REQUIRE(output.size(1) == 1);
    REQUIRE(output.size(2) == 3);
    REQUIRE(output.size(3) == 3);
    REQUIRE(output.data().flatten().allclose(at::tensor(
        {0.0, 0.75, 1.75, 3.75, 9.0, 11.0, 8.75, 19.0, 21.0}, at::kFloat)));
  }
  {
    auto output =
        AvgPool2d(AvgPool2dOptions(2).padding({0, 1}))->forward({input})[0];
    REQUIRE(output.size(0) == 1);
    REQUIRE(output.size(1) == 1);
    REQUIRE(output.size(2) == 2);
    REQUIRE(output.size(3) == 3);
    REQUIRE(output.data().flatten().allclose(
        at::tensor({1.25, 4.0, 6.0, 6.25, 14.0, 16.0}, at::kFloat)));
  }
  {
    auto output =
        AvgPool2d(AvgPool2dOptions(3).ceil_mode(true))->forward({input})[0];
    REQUIRE(output.size(0) == 1);
    REQUIRE(output.size(1) == 1);
    REQUIRE(output.size(2) == 2);
    REQUIRE(output.size(3) == 2);
    REQUIRE(output.data().flatten().allclose(
        at::tensor({6.0, 8.5, 18.5, 21.0}, at::kFloat)));
  }
  {
    auto output =
        AvgPool2d(AvgPool2dOptions(2).padding(1).count_include_pad(false))
            ->forward({input})[0];
    REQUIRE(output.size(0) == 1);
    REQUIRE(output.size(1) == 1);
    REQUIRE(output.size(2) == 3);
    REQUIRE(output.size(3) == 3);
    REQUIRE(output.data().flatten().allclose(at::tensor(
        {0.0, 1.5, 3.5, 7.5, 9.0, 11.0, 17.5, 19.0, 21.0}, at::kFloat)));
  }
}
