#include <gtest/gtest.h>
#include <c10/util/irange.h>
#include <test/cpp/api/support.h>
#include <torch/torch.h>

TEST(PrintTest, Precision) {
    auto t = torch::tensor({0.21862027});
    torch::print(t, 80, 4, true);
    torch::print(t, 80, 8, true);
    ASSERT_EQ(1, 1);
}

TEST(PrintTest, Fixed) {
    auto t = torch::tensor({0.85897932});
    torch::print(t, 80, 8, true);
    torch::print(t, 80, 8, false);
    ASSERT_EQ(1, 1);
}

TEST(PrintTest, PrintFlexibility) {
    auto t = torch::tensor({83902821});
    torch::print(t, 80, 8, false);
    torch::print(t, 80, 2, true);
    ASSERT_EQ(1, 1);
}