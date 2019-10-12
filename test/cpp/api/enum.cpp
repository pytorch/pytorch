#include <gtest/gtest.h>

#include <torch/torch.h>

#include <test/cpp/api/support.h>

TEST_F(EnumTest, AllEnums) {
  c10::variant<
    torch::enumtype::kLinear,
    torch::enumtype::kConv1D,
    torch::enumtype::kConv2D,
    torch::enumtype::kConv3D,
    torch::enumtype::kConvTranspose1D,
    torch::enumtype::kConvTranspose2D,
    torch::enumtype::kConvTranspose3D,
    torch::enumtype::kSigmoid,
    torch::enumtype::kTanh,
    torch::enumtype::kReLU,
    torch::enumtype::kLeakyReLU,
    torch::enumtype::kFanIn,
    torch::enumtype::kFanOut
  > v;

  // yf225 TODO: let's macro generate these:
  {
    v = torch::kLinear;
    ASSERT_EQ(c10::visit(torch::enumtype::enum_name{}, v), "Linear");
  }
  {
    v = torch::kConv1D;
    ASSERT_EQ(c10::visit(torch::enumtype::enum_name{}, v), "Conv1D");
  }
  {
    v = torch::kConv2D;
    ASSERT_EQ(c10::visit(torch::enumtype::enum_name{}, v), "Conv2D");
  }
  {
    v = torch::kConv3D;
    ASSERT_EQ(c10::visit(torch::enumtype::enum_name{}, v), "Conv3D");
  }
  {
    v = torch::kConvTranspose1D;
    ASSERT_EQ(c10::visit(torch::enumtype::enum_name{}, v), "ConvTranspose1D");
  }
  {
    v = torch::kConvTranspose2D;
    ASSERT_EQ(c10::visit(torch::enumtype::enum_name{}, v), "ConvTranspose2D");
  }
  {
    v = torch::kConvTranspose3D;
    ASSERT_EQ(c10::visit(torch::enumtype::enum_name{}, v), "ConvTranspose3D");
  }
  {
    v = torch::kSigmoid;
    ASSERT_EQ(c10::visit(torch::enumtype::enum_name{}, v), "Sigmoid");
  }
  {
    v = torch::kTanh;
    ASSERT_EQ(c10::visit(torch::enumtype::enum_name{}, v), "Tanh");
  }
  {
    v = torch::kReLU;
    ASSERT_EQ(c10::visit(torch::enumtype::enum_name{}, v), "ReLU");
  }
  {
    v = torch::kLeakyReLU;
    ASSERT_EQ(c10::visit(torch::enumtype::enum_name{}, v), "LeakyReLU");
  }
  {
    v = torch::kFanIn;
    ASSERT_EQ(c10::visit(torch::enumtype::enum_name{}, v), "FanIn");
  }
  {
    v = torch::kFanOut;
    ASSERT_EQ(c10::visit(torch::enumtype::enum_name{}, v), "FanOut");
  }
}
