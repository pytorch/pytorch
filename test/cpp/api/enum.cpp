#include <gtest/gtest.h>

#include <c10/util/variant.h>
#include <torch/torch.h>

#include <test/cpp/api/support.h>

#define TORCH_ENUM_PRETTY_PRINT_TEST(name) \
{ \
  v = torch::k##name; \
  std::string pretty_print_name("k"); \
  pretty_print_name.append(#name); \
  ASSERT_EQ(c10::visit(torch::enumtype::enum_name{}, v), pretty_print_name); \
}

TEST(EnumTest, AllEnums) {
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
    torch::enumtype::kFanOut,
    torch::enumtype::kConstant,
    torch::enumtype::kReflect,
    torch::enumtype::kReplicate,
    torch::enumtype::kCircular,
    torch::enumtype::kNearest,
    torch::enumtype::kBilinear,
    torch::enumtype::kBicubic,
    torch::enumtype::kTrilinear,
    torch::enumtype::kArea,
    torch::enumtype::kSum,
    torch::enumtype::kMean,
    torch::enumtype::kMax
  > v;

  TORCH_ENUM_PRETTY_PRINT_TEST(Linear)
  TORCH_ENUM_PRETTY_PRINT_TEST(Conv1D)
  TORCH_ENUM_PRETTY_PRINT_TEST(Conv2D)
  TORCH_ENUM_PRETTY_PRINT_TEST(Conv3D)
  TORCH_ENUM_PRETTY_PRINT_TEST(ConvTranspose1D)
  TORCH_ENUM_PRETTY_PRINT_TEST(ConvTranspose2D)
  TORCH_ENUM_PRETTY_PRINT_TEST(ConvTranspose3D)
  TORCH_ENUM_PRETTY_PRINT_TEST(Sigmoid)
  TORCH_ENUM_PRETTY_PRINT_TEST(Tanh)
  TORCH_ENUM_PRETTY_PRINT_TEST(ReLU)
  TORCH_ENUM_PRETTY_PRINT_TEST(LeakyReLU)
  TORCH_ENUM_PRETTY_PRINT_TEST(FanIn)
  TORCH_ENUM_PRETTY_PRINT_TEST(FanOut)
  TORCH_ENUM_PRETTY_PRINT_TEST(Constant)
  TORCH_ENUM_PRETTY_PRINT_TEST(Reflect)
  TORCH_ENUM_PRETTY_PRINT_TEST(Replicate)
  TORCH_ENUM_PRETTY_PRINT_TEST(Circular)
  TORCH_ENUM_PRETTY_PRINT_TEST(Nearest)
  TORCH_ENUM_PRETTY_PRINT_TEST(Bilinear)
  TORCH_ENUM_PRETTY_PRINT_TEST(Bicubic)
  TORCH_ENUM_PRETTY_PRINT_TEST(Trilinear)
  TORCH_ENUM_PRETTY_PRINT_TEST(Area)
  TORCH_ENUM_PRETTY_PRINT_TEST(Sum)
  TORCH_ENUM_PRETTY_PRINT_TEST(Mean)
  TORCH_ENUM_PRETTY_PRINT_TEST(Max)
}
