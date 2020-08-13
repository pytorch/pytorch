#include <gtest/gtest.h>

#include <torch/torch.h>

#include <test/cpp/api/support.h>

using namespace torch::nn;
using namespace torch::test;

struct TransformerTest : torch::test::SeedingFixture {};

void transformer_encoder_layer_test_helper(bool is_cuda) {
  // this is a deterministic test for TransformerEncoderLayer
  int64_t d_model = 4;
  int64_t nhead = 2;
  int64_t dim_feedforward = 16;
  double dropout = 0.0;

  TransformerEncoderLayer model(
    TransformerEncoderLayerOptions(
      d_model, nhead).dim_feedforward(dim_feedforward).dropout(dropout));
  if (is_cuda) {
    model->to(torch::kCUDA);
  }

  torch::Device device = is_cuda ? torch::kCUDA : torch::kCPU;
  torch::TensorOptions tensor_options = torch::TensorOptions().dtype(torch::kFloat32).device(device);

  // set constant weights of the model
  {
    torch::NoGradGuard guard;
    for (auto& p : model->parameters()) {
      auto sz = p.view(-1).size(0);
      p.copy_(torch::cos(torch::arange(0, sz, tensor_options).view(p.sizes())));
    }
  }

  // relu test case 1
  torch::Tensor encoder_input = torch::tensor({{{20, 30, 40, 50}}}, tensor_options);
  torch::Tensor result = model(encoder_input).detach();
  torch::Tensor ref_output = torch::tensor({{{2.258703, 0.127985, -0.697881, 0.170862}}}, tensor_options);
  ASSERT_EQ(result.sizes(), ref_output.sizes());
  ASSERT_TRUE(torch::allclose(result, ref_output, 1e-7, 1e-5, /*equal_nan=*/true));

  // all 0 values are NOT masked. This should't mask anything
  torch::Tensor mask = torch::tensor({{0}}, tensor_options) == 1;
  result = model(encoder_input, /*src_mask=*/torch::Tensor{}, /*src_key_padding_mask=*/mask).detach();
  ASSERT_EQ(result.sizes(), ref_output.sizes());
  ASSERT_TRUE(torch::allclose(result, ref_output, 1e-7, 1e-5, /*equal_nan=*/true));

  // all 1 values are masked. Since there is only 1 input embedding this will result in nan.
  mask = torch::tensor({{1}}, tensor_options) == 1;
  result = model(encoder_input, /*src_mask=*/torch::Tensor{}, /*src_key_padding_mask=*/mask).detach();
  ASSERT_TRUE(torch::isnan(result).all().item().to<bool>());

  // relu test case 2
  encoder_input = torch::tensor({{{1, 2, 3, 4}}, {{5, 6, 7, 8}}}, tensor_options);
  result = model(encoder_input).detach();
  ref_output = torch::tensor({
    {{2.272644, 0.119035, -0.691669, 0.153486}},
    {{2.272644, 0.119035, -0.691669, 0.153486}}}, tensor_options);
  ASSERT_EQ(result.sizes(), ref_output.sizes());
  ASSERT_TRUE(torch::allclose(result, ref_output, 1e-7, 1e-5, /*equal_nan=*/true));

  // all 0 values are NOT masked
  mask = torch::tensor({{0, 0}}, tensor_options) == 1;
  result = model(encoder_input, /*src_mask=*/torch::Tensor{}, /*src_key_padding_mask=*/mask).detach();
  ASSERT_EQ(result.sizes(), ref_output.sizes());
  ASSERT_TRUE(torch::allclose(result, ref_output, 1e-7, 1e-5, /*equal_nan=*/true));

  // mask with 1 and 0
  mask = torch::tensor({{1, 0}}, tensor_options) == 1;
  result = model(encoder_input, /*src_mask=*/torch::Tensor{}, /*src_key_padding_mask=*/mask).detach();
  ref_output = torch::tensor({
    {{2.301516, 0.092249, -0.679101, 0.103088}},
    {{2.301516, 0.092249, -0.679101, 0.103088}}}, tensor_options);
  ASSERT_EQ(result.sizes(), ref_output.sizes());
  ASSERT_TRUE(torch::allclose(result, ref_output, 1e-7, 1e-5, /*equal_nan=*/true));

  // relu test case 3
  encoder_input = torch::tensor({
    {{0.7462, 0.6653, 0.5679, 0.4891}, {0.5387, 0.1655, 0.3565, 0.0471}},
    {{0.8335, 0.2799, 0.5031, 0.2947}, {0.1402, 0.0318, 0.7636, 0.1346}},
    {{0.6333, 0.9344, 0.1376, 0.9938}, {0.8924, 0.2872, 0.6692, 0.2944}},
    {{0.9897, 0.6915, 0.3154, 0.1733}, {0.8645, 0.3513, 0.3064, 0.0767}},
    {{0.8117, 0.2366, 0.4838, 0.7881}, {0.3718, 0.4945, 0.9511, 0.0864}}}, tensor_options);
  result = model(encoder_input).detach();
  ref_output = torch::tensor({
    {{2.428589, 0.020835, -0.602055, -0.085249}, {2.427987, 0.021213, -0.602496, -0.084103}},
    {{2.424689, 0.019155, -0.604793, -0.085672}, {2.413863, 0.022211, -0.612486, -0.072490}},
    {{2.433774, 0.021598, -0.598343, -0.087548}, {2.425104, 0.019748, -0.604515, -0.084839}},
    {{2.436185, 0.022682, -0.596625, -0.087261}, {2.433556, 0.021891, -0.598509, -0.086832}},
    {{2.416246, 0.017512, -0.610712, -0.082961}, {2.422901, 0.024187, -0.606178, -0.074929}}}, tensor_options);
  ASSERT_EQ(result.sizes(), ref_output.sizes());
  ASSERT_TRUE(torch::allclose(result, ref_output, 1e-7, 1e-5, /*equal_nan=*/true));

  // all 0 values are NOT masked
  mask = torch::zeros({2, 5}, tensor_options) == 1;
  result = model(encoder_input, /*src_mask=*/torch::Tensor{}, /*src_key_padding_mask=*/mask).detach();
  ASSERT_EQ(result.sizes(), ref_output.sizes());
  ASSERT_TRUE(torch::allclose(result, ref_output, 1e-7, 1e-5, /*equal_nan=*/true));

  // mask with 0s and 1s
  mask[0][1] = 1;
  mask[1][3] = 1;
  mask[1][4] = 1;
  result = model(encoder_input, /*src_mask=*/torch::Tensor{}, /*src_key_padding_mask=*/mask).detach();
  ref_output = torch::tensor({
    {{2.429026, 0.020793, -0.601741, -0.085642}, {2.428811, 0.021445, -0.601912, -0.084252}},
    {{2.425009, 0.019155, -0.604566, -0.085899}, {2.415408, 0.02249 , -0.611415, -0.073}},
    {{2.434199, 0.021682, -0.598039, -0.087699}, {2.42598, 0.019941, -0.603896, -0.085091}},
    {{2.436457, 0.022736, -0.59643 , -0.08736},  {2.434021, 0.022093, -0.598179, -0.08679}},
    {{2.416531, 0.017498, -0.610513, -0.083181}, {2.4242, 0.024653, -0.605266, -0.074959}}}, tensor_options);
  ASSERT_EQ(result.sizes(), ref_output.sizes());
  ASSERT_TRUE(torch::allclose(result, ref_output, 1e-7, 1e-5, /*equal_nan=*/true));

  // gelu test case 1
  model.get()->options.activation(torch::kGELU);
  encoder_input = torch::tensor({{{20, 30, 40, 50}}}, tensor_options);
  result = model(encoder_input).detach();
  ref_output = torch::tensor({{{2.249815, 0.131006, -0.702199, 0.177868}}}, tensor_options);
  ASSERT_EQ(result.sizes(), ref_output.sizes());
  ASSERT_TRUE(torch::allclose(result, ref_output, 1e-7, 1e-5, /*equal_nan=*/true));

  // gelu test case 2
  encoder_input = torch::tensor({
    {{0.7462, 0.6653, 0.5679, 0.4891}, {0.5387, 0.1655, 0.3565, 0.0471}},
    {{0.8335, 0.2799, 0.5031, 0.2947}, {0.1402, 0.0318, 0.7636, 0.1346}},
    {{0.6333, 0.9344, 0.1376, 0.9938}, {0.8924, 0.2872, 0.6692, 0.2944}},
    {{0.9897, 0.6915, 0.3154, 0.1733}, {0.8645, 0.3513, 0.3064, 0.0767}},
    {{0.8117, 0.2366, 0.4838, 0.7881}, {0.3718, 0.4945, 0.9511, 0.0864}}}, tensor_options);
  result = model(encoder_input);
  ref_output = torch::tensor({
    {{2.42163188, 0.03227153, -0.60714219, -0.05908082}, {2.42151276, 0.03302179, -0.60722523, -0.05762651}},
    {{2.41926761, 0.02974034, -0.60879519, -0.0621269}, {2.41626395, 0.03539356, -0.61087842, -0.04978623}},
    {{2.42382808, 0.03218872, -0.6055963, -0.06073591}, {2.41983477, 0.03085259, -0.60840145, -0.06046414}},
    {{2.42500749, 0.03328855, -0.60476388, -0.0595334}, {2.4237977, 0.03290575, -0.60561789, -0.05940082}},
    {{2.41383916, 0.02686345, -0.61256377, -0.06380707}, {2.42000277, 0.03800944, -0.60824798, -0.04754947}}},
    tensor_options);
  ASSERT_EQ(result.sizes(), ref_output.sizes());
  ASSERT_TRUE(torch::allclose(result, ref_output, 1e-7, 1e-5, /*equal_nan=*/true));
}

TEST_F(TransformerTest, TestTransformerEncoderLayer) {
  transformer_encoder_layer_test_helper(false);
}

TEST_F(TransformerTest, TestTransformerEncoderLayer_CUDA) {
  transformer_encoder_layer_test_helper(true);
}
