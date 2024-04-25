#include <gtest/gtest.h>
#include <c10/util/irange.h>
#include <test/cpp/api/support.h>
#include <torch/torch.h>

TEST(PrintTest, Precision) {
    auto t = torch::tensor({{0.21861312}});
    torch::print(t, 80, 4, true);
    torch::print(t, 80, 8, true);
}

TEST(PrintTest, Fixed) {
    auto t = torch::tensor({{0.85897932}});
    torch::print(t, 80, 8, true);
    torch::print(t, 80, 8, false);
}

TEST(PrintTest, PrintFlexibility) {
    auto t = torch::tensor({{83902821284}});
    torch::print(t, 80, 11, false);
    torch::print(t, 80, 2, true);
}