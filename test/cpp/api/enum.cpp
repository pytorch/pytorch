#include <gtest/gtest.h>

#include <torch/torch.h>
#include <variant>

#include <test/cpp/api/support.h>

#define TORCH_ENUM_PRETTY_PRINT_TEST(name)                           \
  {                                                                  \
    v = torch::k##name;                                              \
    std::string pretty_print_name("k");                              \
    pretty_print_name.append(#name);                                 \
    ASSERT_EQ(torch::enumtype::get_enum_name(v), pretty_print_name); \
  }

TEST(EnumTest, AllEnums) {
  std::variant<
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
      torch::enumtype::kMax,
      torch::enumtype::kNone,
      torch::enumtype::kBatchMean,
      torch::enumtype::kZeros,
      torch::enumtype::kBorder,
      torch::enumtype::kReflection,
      torch::enumtype::kRNN_TANH,
      torch::enumtype::kRNN_RELU,
      torch::enumtype::kLSTM,
      torch::enumtype::kGRU>
      v;

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
  TORCH_ENUM_PRETTY_PRINT_TEST(None)
  TORCH_ENUM_PRETTY_PRINT_TEST(BatchMean)
  TORCH_ENUM_PRETTY_PRINT_TEST(Zeros)
  TORCH_ENUM_PRETTY_PRINT_TEST(Border)
  TORCH_ENUM_PRETTY_PRINT_TEST(Reflection)
  TORCH_ENUM_PRETTY_PRINT_TEST(RNN_TANH)
  TORCH_ENUM_PRETTY_PRINT_TEST(RNN_RELU)
  TORCH_ENUM_PRETTY_PRINT_TEST(LSTM)
  TORCH_ENUM_PRETTY_PRINT_TEST(GRU)
}
