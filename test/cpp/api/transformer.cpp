#include <gtest/gtest.h>

#include <torch/torch.h>

#include <test/cpp/api/support.h>

using namespace torch::nn;

struct TransformerTest : torch::test::SeedingFixture {};

// a generic function to set constants for parameters so we have fixed result for deterministic test
template<typename Model>
void set_parameter_to_constants(Model& model, const torch::TensorOptions& tensor_options) {
  torch::NoGradGuard guard;
  for (auto& p : model->parameters()) {
    auto sz = p.view(-1).size(0);
    p.copy_(torch::cos(torch::arange(0, sz, tensor_options).view(p.sizes())));
  }
}

// a generic function to provide consistent encoder/decoder layer for all the transformer tests
template<typename T_LAYER, typename T_OPTIONS>
T_LAYER get_a_test_layer(const torch::TensorOptions& tensor_options) {
  int64_t d_model = 4;
  int64_t nhead = 2;
  int64_t dim_feedforward = 16;
  double dropout = 0.0;

  // activation is always ReLU here and it can be adjusted later depending on the usage
  T_LAYER layer(T_OPTIONS(d_model, nhead).dim_feedforward(dim_feedforward).dropout(dropout));
  if (tensor_options.device() == torch::kCUDA) {
    layer->to(torch::kCUDA);
  }

  // set constant weights of the model
  set_parameter_to_constants<T_LAYER>(layer, tensor_options);

  return layer;
}

void transformer_encoder_layer_test_helper(bool is_cuda) {
  // this is a deterministic test for TransformerEncoderLayer
  torch::Device device = is_cuda ? torch::kCUDA : torch::kCPU;
  torch::TensorOptions tensor_options = torch::TensorOptions().dtype(torch::kFloat32).device(device);

  TransformerEncoderLayer model =
    get_a_test_layer<TransformerEncoderLayer, TransformerEncoderLayerOptions>(tensor_options);

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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(TransformerTest, TransformerEncoderLayer) {
  transformer_encoder_layer_test_helper(false);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(TransformerTest, TransformerEncoderLayer_CUDA) {
  transformer_encoder_layer_test_helper(true);
}

void transformer_decoder_layer_test_helper(bool is_cuda){

  torch::Device device = is_cuda ? torch::kCUDA : torch::kCPU;
  torch::TensorOptions tensor_options = torch::TensorOptions()
    .dtype(torch::kFloat32).device(device);

  TransformerDecoderLayer model = get_a_test_layer<
    TransformerDecoderLayer,
    TransformerDecoderLayerOptions>(tensor_options);

  // deterministic input
  torch::Tensor decoder_input = torch::tensor({{{20, 30, 40, 50}}},
                                           tensor_options);
  torch::Tensor memory_input = torch::tensor({{{60, 70, 80, 90}}},
                                          tensor_options);
  torch::Tensor result = model(decoder_input, memory_input).detach();
  torch::Tensor ref_output = torch::tensor(
    {{{2.314351, 0.094805, -0.671322, 0.101977}}},
    tensor_options);
  ASSERT_EQ(result.sizes().size(),ref_output.sizes().size());
  ASSERT_TRUE(torch::allclose(result, ref_output, 1e-7, 1e-5,
                              /*equal_nan=*/true));

  // deterministic input
  decoder_input = torch::tensor({{{9, 10, 11, 12}},
                                 {{11, 12, 13, 14}}}, tensor_options);
  memory_input = torch::tensor({{{1, 2, 3, 4}}}, tensor_options);
  result = model(decoder_input, memory_input).detach();
  ref_output = torch::tensor({{{2.422245,  0.051716, -0.606338, -0.024756}},
                              {{2.422245,  0.051716, -0.606338, -0.024756}}},
                              tensor_options);
  ASSERT_EQ(result.sizes().size(),ref_output.sizes().size());
  ASSERT_TRUE(torch::allclose(result, ref_output, 1e-7, 1e-5,
                              /*equal_nan=*/true));

  // deterministic input
  decoder_input = torch::tensor({{{1, 2, 3, 4}},
                                 {{5, 6, 7, 8}}}, tensor_options);
  memory_input = torch::tensor({{{9, 10, 11, 12}},
                                {{11, 12, 13, 14}}}, tensor_options);
  result = model(decoder_input, memory_input).detach();
  ref_output = torch::tensor({{{2.343536,  0.085561, -0.654954, 0.074991}},
                              {{2.343536,  0.085561, -0.654954, 0.074991}}},
                              tensor_options);
  ASSERT_EQ(result.sizes().size(),ref_output.sizes().size());
  ASSERT_TRUE(torch::allclose(result, ref_output, 1e-7, 1e-5,
                              /*equal_nan=*/true));


   // deterministic input
  decoder_input = torch::tensor({{{0.4517, 0.6793, 0.5313, 0.0034},
                                  {0.2678, 0.3677, 0.4459, 0.7166}},
                                 {{0.8100, 0.3716, 0.4096, 0.1976},
                                  {0.6958, 0.8844, 0.6081, 0.8315}},
                                 {{0.0494, 0.9343, 0.5955, 0.3830},
                                  {0.5404, 0.3464, 0.9378, 0.6200}}},
                                  tensor_options);
  memory_input = torch::tensor({{{0.7462, 0.6653, 0.5679, 0.4891},
                                 {0.5387, 0.1655, 0.3565, 0.0471}},
                                {{0.8335, 0.2799, 0.5031, 0.2947},
                                 {0.1402, 0.0318, 0.7636, 0.1346}},
                                {{0.6333, 0.9344, 0.1376, 0.9938},
                                 {0.8924, 0.2872, 0.6692, 0.2944}},
                                {{0.9897, 0.6915, 0.3154, 0.1733},
                                 {0.8645, 0.3513, 0.3064, 0.0767}},
                                {{0.8117, 0.2366, 0.4838, 0.7881},
                                 {0.3718, 0.4945, 0.9511, 0.0864}}},
                                 tensor_options);
  result = model(decoder_input, memory_input).detach();
  ref_output = torch::tensor({{{2.430065, 0.027862, -0.601136, -0.073096},
                               {2.431935, 0.028907, -0.599809, -0.072488}},
                              {{2.428457, 0.027053, -0.602275, -0.073462},
                               {2.431970, 0.029387, -0.599789, -0.071621}},
                              {{2.431934, 0.028196, -0.599802, -0.073809},
                               {2.432306, 0.028858, -0.599542, -0.072846}}},
                               tensor_options);
  ASSERT_EQ(result.sizes().size(),ref_output.sizes().size());
  ASSERT_TRUE(torch::allclose(result, ref_output, 1e-7, 1e-5,
                              /*equal_nan=*/true));

  // key_padding_mask
  torch::Tensor t_mask = {};
  torch::Tensor m_mask = {};
  torch::Tensor key_padding_mask = torch::zeros({2, 3}, tensor_options) == 1;
  result = model(decoder_input, memory_input, t_mask, m_mask,
                 key_padding_mask).detach();
  ref_output = torch::tensor({{{2.430065, 0.027862, -0.601136, -0.073096},
                               {2.431935, 0.028907, -0.599809, -0.072488}},
                              {{2.428457, 0.027053, -0.602275, -0.073462},
                               {2.431970, 0.029387, -0.599789, -0.071621}},
                              {{2.431934, 0.028196, -0.599802, -0.073809},
                               {2.432306, 0.028858, -0.599542, -0.072846}}},
                               tensor_options);
  ASSERT_EQ(result.sizes().size(),ref_output.sizes().size());
  ASSERT_TRUE(torch::allclose(result, ref_output, 1e-7, 1e-5,
                              /*equal_nan=*/true));

  // key_padding_mask
  key_padding_mask[0][2] = 1;
  key_padding_mask[1][1] = 1;
  key_padding_mask[1][2] = 1;
  result = model(decoder_input, memory_input, t_mask, m_mask,
                 key_padding_mask).detach();
  ref_output = torch::tensor({{{2.430025, 0.027643, -0.601164, -0.073476},
                               {2.4323, 0.029375, -0.599553, -0.071881}},
                              {{2.428523, 0.026838, -0.602226, -0.07391},
                               {2.432634, 0.029842, -0.599318, -0.071253}},
                              {{2.432278, 0.028152, -0.599555, -0.074139},
                               {2.432659, 0.029244, -0.599294, -0.072382}}},
                               tensor_options);
  ASSERT_EQ(result.sizes().size(),ref_output.sizes().size());
  ASSERT_TRUE(torch::allclose(result, ref_output, 1e-7, 1e-5,
                              /*equal_nan=*/true));

  // memory_key_padding_mask
  torch::Tensor t_key_padding_mask = {};
  key_padding_mask = torch::zeros({2, 5}, tensor_options) == 1;
  result = model(decoder_input, memory_input, t_mask, m_mask,
                 t_key_padding_mask, key_padding_mask).detach();
  ref_output = torch::tensor({{{2.430065, 0.027862, -0.601136, -0.073096},
                               {2.431935, 0.028907, -0.599809, -0.072488}},
                              {{2.428457, 0.027053, -0.602275, -0.073462},
                               {2.431970, 0.029387, -0.599789, -0.071621}},
                              {{2.431934, 0.028196, -0.599802, -0.073809},
                               {2.432306, 0.028858, -0.599542, -0.072846}}},
                               tensor_options);
  ASSERT_EQ(result.sizes().size(),ref_output.sizes().size());
  ASSERT_TRUE(torch::allclose(result, ref_output, 1e-7, 1e-5,
                              /*equal_nan=*/true));

  // memory_key_padding_mask
  key_padding_mask[0][4] = 1;
  key_padding_mask[1][3] = 1;
  key_padding_mask[1][4] = 1;
  result = model(decoder_input, memory_input, t_mask, m_mask,
                 t_key_padding_mask, key_padding_mask).detach();
  ref_output = torch::tensor({{{2.429757, 0.027358, -0.601351, -0.073816},
                               {2.432692, 0.028583, -0.599263, -0.073634}},
                              {{2.428247, 0.02662, -0.602419, -0.074123},
                               {2.432657, 0.029055, -0.599293, -0.072732}},
                              {{2.431515, 0.027687, -0.600096, -0.074459},
                               {2.433075, 0.028543, -0.598987, -0.073985}}},
                               tensor_options);
  ASSERT_EQ(result.sizes().size(),ref_output.sizes().size());
  ASSERT_TRUE(torch::allclose(result, ref_output, 1e-7, 1e-5,
                              /*equal_nan=*/true));

}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(TransformerTest, TransformerDecoderLayer){
  transformer_decoder_layer_test_helper(false);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(TransformerTest, TransformerDecoderLayer_CUDA){
    transformer_decoder_layer_test_helper(true);
}

void transformer_decoder_layer_test_helper_gelu(bool is_cuda) {

  torch::Device device = is_cuda ? torch::kCUDA : torch::kCPU;
  torch::TensorOptions tensor_options = torch::TensorOptions()
    .dtype(torch::kFloat32).device(device);

  TransformerDecoderLayer model = get_a_test_layer<
    TransformerDecoderLayer,
    TransformerDecoderLayerOptions>(tensor_options);
  model.get()->options.activation(torch::kGELU);

  // deterministic input
  torch::Tensor decoder_input = torch::tensor({{{20, 30, 40, 50}}},
                                           tensor_options);
  torch::Tensor memory_input = torch::tensor({{{60, 70, 80, 90}}},
                                          tensor_options);
  torch::Tensor result = model(decoder_input, memory_input).detach();
  torch::Tensor ref_output = torch::tensor(
    {{{2.306435, 0.095946, -0.675796, 0.10687}}},
    tensor_options);
  ASSERT_EQ(result.sizes().size(),ref_output.sizes().size());
  ASSERT_TRUE(torch::allclose(result, ref_output, 1e-7, 1e-5,
                              /*equal_nan=*/true));

  // deterministic input
  decoder_input = torch::tensor({{{9, 10, 11, 12}},
                                 {{11, 12, 13, 14}}},
                                 tensor_options);
  memory_input = torch::tensor({{{1, 2, 3, 4}}}, tensor_options);
  result = model(decoder_input, memory_input).detach();
  ref_output = torch::tensor({{{2.415448, 0.054389, -0.610932, -0.0156613}},
                              {{2.415448, 0.054389, -0.610932, -0.0156613}}},
                              tensor_options);
  ASSERT_EQ(result.sizes().size(),ref_output.sizes().size());
  ASSERT_TRUE(torch::allclose(result, ref_output, 1e-7, 1e-5,
                              /*equal_nan=*/true));

  // deterministic input
  decoder_input = torch::tensor({{{1, 2, 3, 4}},
                                 {{5, 6, 7, 8}}},
                                 tensor_options);
  memory_input = torch::tensor({{{9, 10, 11, 12}},
                                {{11, 12, 13, 14}}},
                                tensor_options);
  result = model(decoder_input, memory_input).detach();
  ref_output = torch::tensor({{{2.338531, 0.087709, -0.65776, 0.080646}},
                              {{2.338531, 0.087709, -0.65776, 0.080646}}},
                              tensor_options);
  ASSERT_EQ(result.sizes().size(),ref_output.sizes().size());
  ASSERT_TRUE(torch::allclose(result, ref_output, 1e-7, 1e-5,
                              /*equal_nan=*/true));


  // deterministic input
  decoder_input = torch::tensor({{{0.4517, 0.6793, 0.5313, 0.0034},
                                  {0.2678, 0.3677, 0.4459, 0.7166}},
                                 {{0.8100, 0.3716, 0.4096, 0.1976},
                                  {0.6958, 0.8844, 0.6081, 0.8315}},
                                 {{0.0494, 0.9343, 0.5955, 0.3830},
                                  {0.5404, 0.3464, 0.9378, 0.6200}}},
                                 tensor_options);
  memory_input = torch::tensor({{{0.7462, 0.6653, 0.5679, 0.4891},
                                 {0.5387, 0.1655, 0.3565, 0.0471}},
                                {{0.8335, 0.2799, 0.5031, 0.2947},
                                 {0.1402, 0.0318, 0.7636, 0.1346}},
                                {{0.6333, 0.9344, 0.1376, 0.9938},
                                 {0.8924, 0.2872, 0.6692, 0.2944}},
                                {{0.9897, 0.6915, 0.3154, 0.1733},
                                 {0.8645, 0.3513, 0.3064, 0.0767}},
                                {{0.8117, 0.2366, 0.4838, 0.7881},
                                 {0.3718, 0.4945, 0.9511, 0.0864}}},
                                tensor_options);
  result = model(decoder_input, memory_input).detach();
  ref_output = torch::tensor(
    {{{2.42049104, 0.03443088, -0.60793706, -0.05436271},
     {2.42210631, 0.03546578, -0.60679895, -0.05357488}},
    {{2.41907674, 0.0336104, -0.60892977, -0.05490462},
     {2.42216881, 0.03586554, -0.6067524, -0.05289126}},
    {{2.42205716, 0.03488046, -0.60683681, -0.05460596},
     {2.42240309, 0.0354595, -0.60659063, -0.05378816}}},
    tensor_options);
  ASSERT_EQ(result.sizes().size(),ref_output.sizes().size());
  ASSERT_TRUE(torch::allclose(result, ref_output, 1e-7, 1e-5,
                              /*equal_nan=*/true));

}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(TransformerTest, TransformerDecoderLayer_gelu) {
  transformer_decoder_layer_test_helper_gelu(false);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(TransformerTest, TransformerDecoderLayer_gelu_CUDA) {
  transformer_decoder_layer_test_helper_gelu(true);
}

void transformer_encoder_test_helper(bool is_cuda) {
  // this is a deterministic test for TransformerEncoderLayer
  torch::Device device = is_cuda ? torch::kCUDA : torch::kCPU;
  torch::TensorOptions tensor_options = torch::TensorOptions().dtype(torch::kFloat32).device(device);

  TransformerEncoderLayer encoder_layer =
    get_a_test_layer<TransformerEncoderLayer, TransformerEncoderLayerOptions>(tensor_options);

  TransformerEncoder model(TransformerEncoderOptions(encoder_layer, 1));
  if (is_cuda) {
    model->to(torch::kCUDA);
  }

  torch::Tensor encoder_input = torch::tensor({
    {{0.7462, 0.6653, 0.5679, 0.4891}, {0.5387, 0.1655, 0.3565, 0.0471}},
    {{0.8335, 0.2799, 0.5031, 0.2947}, {0.1402, 0.0318, 0.7636, 0.1346}},
    {{0.6333, 0.9344, 0.1376, 0.9938}, {0.8924, 0.2872, 0.6692, 0.2944}},
    {{0.9897, 0.6915, 0.3154, 0.1733}, {0.8645, 0.3513, 0.3064, 0.0767}},
    {{0.8117, 0.2366, 0.4838, 0.7881}, {0.3718, 0.4945, 0.9511, 0.0864}}}, tensor_options);
  torch::Tensor result = model(encoder_input).detach();
  torch::Tensor ref_output = torch::tensor({
    {{2.428589, 0.020835, -0.602055, -0.085249}, {2.427987, 0.021213, -0.602496, -0.084103}},
    {{2.424689, 0.019155, -0.604793, -0.085672}, {2.413863, 0.022211, -0.612486, -0.072490}},
    {{2.433774, 0.021598, -0.598343, -0.087548}, {2.425104, 0.019748, -0.604515, -0.084839}},
    {{2.436185, 0.022682, -0.596625, -0.087261}, {2.433556, 0.021891, -0.598509, -0.086832}},
    {{2.416246, 0.017512, -0.610712, -0.082961}, {2.422901, 0.024187, -0.606178, -0.074929}}}, tensor_options);
  ASSERT_EQ(result.sizes(), ref_output.sizes());
  ASSERT_TRUE(torch::allclose(result, ref_output, 1e-7, 1e-5, /*equal_nan=*/true));

  // all 0 values are NOT masked
  torch::Tensor mask = torch::zeros({2, 5}, tensor_options) == 1;
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

  // test case 2, multiple layers no norm
  model = TransformerEncoder(TransformerEncoderOptions(encoder_layer, 2));
  if (is_cuda) {
    model->to(torch::kCUDA);
  }
  result = model(encoder_input, /*src_mask=*/torch::Tensor{}, /*src_key_padding_mask=*/mask).detach();
  ref_output = torch::tensor({
    {{2.419051, 0.017446, -0.608738, -0.085003}, {2.419102, 0.017452, -0.608703, -0.085026}},
    {{2.419043, 0.017445, -0.608744, -0.084999}, {2.419052, 0.017446, -0.608738, -0.085004}},
    {{2.419067, 0.017448, -0.608727, -0.085010}, {2.419098, 0.017452, -0.608706, -0.085024}},
    {{2.419072, 0.017449, -0.608724, -0.085012}, {2.419119, 0.017455, -0.608691, -0.085034}},
    {{2.419019, 0.017442, -0.608761, -0.084989}, {2.419075, 0.017449, -0.608722, -0.085014}}}, tensor_options);
  ASSERT_EQ(result.sizes(), ref_output.sizes());
  ASSERT_TRUE(torch::allclose(result, ref_output, 1e-7, 1e-5, /*equal_nan=*/true));

  model = TransformerEncoder(TransformerEncoderOptions(encoder_layer, 6));
  if (is_cuda) {
    model->to(torch::kCUDA);
  }
  result = model(encoder_input, /*src_mask=*/torch::Tensor{}, /*src_key_padding_mask=*/mask).detach();
  ref_output = torch::tensor({
    {{2.419101, 0.017453, -0.608703, -0.085025}, {2.419101, 0.017453, -0.608704, -0.085025}},
    {{2.419101, 0.017453, -0.608703, -0.085025}, {2.419101, 0.017453, -0.608704, -0.085025}},
    {{2.419101, 0.017453, -0.608703, -0.085025}, {2.419101, 0.017453, -0.608704, -0.085025}},
    {{2.419101, 0.017453, -0.608703, -0.085025}, {2.419101, 0.017453, -0.608704, -0.085025}},
    {{2.419101, 0.017453, -0.608703, -0.085025}, {2.419101, 0.017453, -0.608704, -0.085025}}}, tensor_options);
  ASSERT_EQ(result.sizes(), ref_output.sizes());
  ASSERT_TRUE(torch::allclose(result, ref_output, 1e-7, 1e-5, /*equal_nan=*/true));

  // test case 3, multiple layers with norm
  LayerNorm norm(LayerNormOptions({encoder_layer.get()->options.d_model()}));
  model = TransformerEncoder(TransformerEncoderOptions(encoder_layer, 2).norm(AnyModule(norm)));
  if (is_cuda) {
    model->to(torch::kCUDA);
  }
  result = model(encoder_input, /*src_mask=*/torch::Tensor{}, /*src_key_padding_mask=*/mask).detach();
  ref_output = torch::tensor({
    {{1.695949, -0.357635, -0.893077, -0.445238}, {1.695955, -0.357639, -0.893050, -0.445266}},
    {{1.695948, -0.357634, -0.893082, -0.445233}, {1.695950, -0.357635, -0.893077, -0.445238}},
    {{1.695951, -0.357636, -0.893069, -0.445246}, {1.695955, -0.357639, -0.893052, -0.445264}},
    {{1.695952, -0.357636, -0.893066, -0.445249}, {1.695957, -0.357641, -0.893041, -0.445276}},
    {{1.695946, -0.357632, -0.893095, -0.445220}, {1.695952, -0.357637, -0.893065, -0.445251}}}, tensor_options);
  ASSERT_EQ(result.sizes(), ref_output.sizes());
  ASSERT_TRUE(torch::allclose(result, ref_output, 1e-7, 1e-5, /*equal_nan=*/true));

  model = TransformerEncoder(TransformerEncoderOptions(encoder_layer, 6).norm(AnyModule(norm)));
  if (is_cuda) {
    model->to(torch::kCUDA);
  }
  result = model(encoder_input, /*src_mask=*/torch::Tensor{}, /*src_key_padding_mask=*/mask).detach();
  ref_output = torch::tensor({
    {{1.695955, -0.357639, -0.893051, -0.445265}, {1.695955, -0.357639, -0.893051, -0.445265}},
    {{1.695955, -0.357639, -0.893051, -0.445265}, {1.695955, -0.357639, -0.893051, -0.445265}},
    {{1.695955, -0.357639, -0.893051, -0.445265}, {1.695955, -0.357639, -0.893051, -0.445265}},
    {{1.695955, -0.357639, -0.893051, -0.445265}, {1.695955, -0.357639, -0.893051, -0.445265}},
    {{1.695955, -0.357639, -0.893051, -0.445265}, {1.695955, -0.357639, -0.893051, -0.445265}}}, tensor_options);
  ASSERT_EQ(result.sizes(), ref_output.sizes());
  ASSERT_TRUE(torch::allclose(result, ref_output, 1e-7, 1e-5, /*equal_nan=*/true));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(TransformerTest, TransformerEncoder) {
  transformer_encoder_test_helper(false);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(TransformerTest, TransformerEncoder_CUDA) {
  transformer_encoder_test_helper(true);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(TransformerTest, PrettyPrintTransformerEncoderLayer) {
  ASSERT_EQ(
      c10::str(TransformerEncoderLayer(4, 2)),
      "torch::nn::TransformerEncoderLayerImpl(\n"
      "  (self_attn): torch::nn::MultiheadAttention(\n"
      "    (out_proj): torch::nn::Linear(in_features=4, out_features=4, bias=true)\n"
      "  )\n"
      "  (linear1): torch::nn::Linear(in_features=4, out_features=2048, bias=true)\n"
      "  (dropout): torch::nn::Dropout(p=0.1, inplace=false)\n"
      "  (linear2): torch::nn::Linear(in_features=2048, out_features=4, bias=true)\n"
      "  (norm1): torch::nn::LayerNorm([4], eps=1e-05, elementwise_affine=true)\n"
      "  (norm2): torch::nn::LayerNorm([4], eps=1e-05, elementwise_affine=true)\n"
      "  (dropout1): torch::nn::Dropout(p=0.1, inplace=false)\n"
      "  (dropout2): torch::nn::Dropout(p=0.1, inplace=false)\n"
      ")");
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(TransformerTest, PrettyPrintTransformerEncoder) {
  LayerNorm norm = LayerNorm(LayerNormOptions({4}));
  TransformerEncoderOptions options(
    TransformerEncoderOptions(TransformerEncoderLayerOptions(4, 2),2).norm(AnyModule(norm)));
  ASSERT_EQ(
      c10::str(TransformerEncoder(options)),
      "torch::nn::TransformerEncoderImpl(\n"
      "  (layers): torch::nn::ModuleList(\n"
      "    (0): torch::nn::TransformerEncoderLayerImpl(\n"
      "      (self_attn): torch::nn::MultiheadAttention(\n"
      "        (out_proj): torch::nn::Linear(in_features=4, out_features=4, bias=true)\n"
      "      )\n"
      "      (linear1): torch::nn::Linear(in_features=4, out_features=2048, bias=true)\n"
      "      (dropout): torch::nn::Dropout(p=0.1, inplace=false)\n"
      "      (linear2): torch::nn::Linear(in_features=2048, out_features=4, bias=true)\n"
      "      (norm1): torch::nn::LayerNorm([4], eps=1e-05, elementwise_affine=true)\n"
      "      (norm2): torch::nn::LayerNorm([4], eps=1e-05, elementwise_affine=true)\n"
      "      (dropout1): torch::nn::Dropout(p=0.1, inplace=false)\n"
      "      (dropout2): torch::nn::Dropout(p=0.1, inplace=false)\n"
      "    )\n"
      "    (1): torch::nn::TransformerEncoderLayerImpl(\n"
      "      (self_attn): torch::nn::MultiheadAttention(\n"
      "        (out_proj): torch::nn::Linear(in_features=4, out_features=4, bias=true)\n"
      "      )\n"
      "      (linear1): torch::nn::Linear(in_features=4, out_features=2048, bias=true)\n"
      "      (dropout): torch::nn::Dropout(p=0.1, inplace=false)\n"
      "      (linear2): torch::nn::Linear(in_features=2048, out_features=4, bias=true)\n"
      "      (norm1): torch::nn::LayerNorm([4], eps=1e-05, elementwise_affine=true)\n"
      "      (norm2): torch::nn::LayerNorm([4], eps=1e-05, elementwise_affine=true)\n"
      "      (dropout1): torch::nn::Dropout(p=0.1, inplace=false)\n"
      "      (dropout2): torch::nn::Dropout(p=0.1, inplace=false)\n"
      "    )\n"
      "  )\n"
      "  (norm): torch::nn::LayerNorm([4], eps=1e-05, elementwise_affine=true)\n"
      ")");
}


// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(TransformerTest, PrettyPrintTransformerDecoderLayer) {
  ASSERT_EQ(
      c10::str(TransformerDecoderLayer(4, 2)),
      "torch::nn::TransformerDecoderLayerImpl(\n"
      "  (self_attn): torch::nn::MultiheadAttention(\n"
      "    (out_proj): torch::nn::Linear(in_features=4, out_features=4, bias=true)\n"
      "  )\n"
      "  (multihead_attn): torch::nn::MultiheadAttention(\n"
      "    (out_proj): torch::nn::Linear(in_features=4, out_features=4, bias=true)\n"
      "  )\n"
      "  (linear1): torch::nn::Linear(in_features=4, out_features=2048, bias=true)\n"
      "  (dropout): torch::nn::Dropout(p=0.1, inplace=false)\n"
      "  (linear2): torch::nn::Linear(in_features=2048, out_features=4, bias=true)\n"
      "  (norm1): torch::nn::LayerNorm([4], eps=1e-05, elementwise_affine=true)\n"
      "  (norm2): torch::nn::LayerNorm([4], eps=1e-05, elementwise_affine=true)\n"
      "  (norm3): torch::nn::LayerNorm([4], eps=1e-05, elementwise_affine=true)\n"
      "  (dropout1): torch::nn::Dropout(p=0.1, inplace=false)\n"
      "  (dropout2): torch::nn::Dropout(p=0.1, inplace=false)\n"
      "  (dropout3): torch::nn::Dropout(p=0.1, inplace=false)\n"
      ")");
}

void transformer_decoder_test_helper(bool is_cuda) {
  // this is a deterministic test for TransformerDecoder
  torch::Device device = is_cuda ? torch::kCUDA : torch::kCPU;
  torch::TensorOptions tensor_options =
    torch::TensorOptions().dtype(torch::kFloat32).device(device);

  TransformerDecoderLayer decoder_layer = get_a_test_layer<
    TransformerDecoderLayer,
    TransformerDecoderLayerOptions>(tensor_options);

  TransformerDecoder model(TransformerDecoderOptions(decoder_layer, 1));
  if (is_cuda) {
    model->to(torch::kCUDA);
  }


  torch::Tensor decoder_input = torch::tensor({{{20, 30, 40, 50}}},
                                           tensor_options);
  torch::Tensor memory_input = torch::tensor({{{60, 70, 80, 90}}},
                                          tensor_options);
  torch::Tensor result = model(decoder_input, memory_input).detach();
  torch::Tensor ref_output = torch::tensor(
    {{{2.314351, 0.094805, -0.671322, 0.101977}}},
    tensor_options);
  ASSERT_EQ(result.sizes().size(),ref_output.sizes().size());
  ASSERT_TRUE(torch::allclose(result, ref_output, 1e-7, 1e-5,
                              /*equal_nan=*/true));

// deterministic input
  decoder_input = torch::tensor({{{9, 10, 11, 12}},
                                 {{11, 12, 13, 14}}}, tensor_options);
  memory_input = torch::tensor({{{1, 2, 3, 4}}}, tensor_options);
  result = model(decoder_input, memory_input).detach();
  ref_output = torch::tensor({{{2.422245,  0.051716, -0.606338, -0.024756}},
                              {{2.422245,  0.051716, -0.606338, -0.024756}}},
                              tensor_options);
  ASSERT_EQ(result.sizes().size(),ref_output.sizes().size());
  ASSERT_TRUE(torch::allclose(result, ref_output, 1e-7, 1e-5,
                              /*equal_nan=*/true));

  // deterministic input
  decoder_input = torch::tensor({{{1, 2, 3, 4}},
                                 {{5, 6, 7, 8}}}, tensor_options);
  memory_input = torch::tensor({{{9, 10, 11, 12}},
                                {{11, 12, 13, 14}}}, tensor_options);
  result = model(decoder_input, memory_input).detach();
  ref_output = torch::tensor({{{2.343536,  0.085561, -0.654954, 0.074991}},
                              {{2.343536,  0.085561, -0.654954, 0.074991}}},
                              tensor_options);
  ASSERT_EQ(result.sizes().size(),ref_output.sizes().size());
  ASSERT_TRUE(torch::allclose(result, ref_output, 1e-7, 1e-5,
                              /*equal_nan=*/true));


   // deterministic input
  decoder_input = torch::tensor({{{0.4517, 0.6793, 0.5313, 0.0034},
                                  {0.2678, 0.3677, 0.4459, 0.7166}},
                                 {{0.8100, 0.3716, 0.4096, 0.1976},
                                  {0.6958, 0.8844, 0.6081, 0.8315}},
                                 {{0.0494, 0.9343, 0.5955, 0.3830},
                                  {0.5404, 0.3464, 0.9378, 0.6200}}},
                                  tensor_options);
  memory_input = torch::tensor({{{0.7462, 0.6653, 0.5679, 0.4891},
                                 {0.5387, 0.1655, 0.3565, 0.0471}},
                                {{0.8335, 0.2799, 0.5031, 0.2947},
                                 {0.1402, 0.0318, 0.7636, 0.1346}},
                                {{0.6333, 0.9344, 0.1376, 0.9938},
                                 {0.8924, 0.2872, 0.6692, 0.2944}},
                                {{0.9897, 0.6915, 0.3154, 0.1733},
                                 {0.8645, 0.3513, 0.3064, 0.0767}},
                                {{0.8117, 0.2366, 0.4838, 0.7881},
                                 {0.3718, 0.4945, 0.9511, 0.0864}}},
                                 tensor_options);
  result = model(decoder_input, memory_input).detach();
  ref_output = torch::tensor({{{2.430065, 0.027862, -0.601136, -0.073096},
                               {2.431935, 0.028907, -0.599809, -0.072488}},
                              {{2.428457, 0.027053, -0.602275, -0.073462},
                               {2.431970, 0.029387, -0.599789, -0.071621}},
                              {{2.431934, 0.028196, -0.599802, -0.073809},
                               {2.432306, 0.028858, -0.599542, -0.072846}}},
                               tensor_options);
  ASSERT_EQ(result.sizes().size(),ref_output.sizes().size());
  ASSERT_TRUE(torch::allclose(result, ref_output, 1e-7, 1e-5,
                              /*equal_nan=*/true));

  // key_padding_mask
  torch::Tensor t_mask = {};
  torch::Tensor m_mask = {};
  torch::Tensor key_padding_mask = torch::zeros({2, 3}, tensor_options) == 1;
  result = model(decoder_input, memory_input, t_mask, m_mask,
                 key_padding_mask).detach();
  ref_output = torch::tensor({{{2.430065, 0.027862, -0.601136, -0.073096},
                               {2.431935, 0.028907, -0.599809, -0.072488}},
                              {{2.428457, 0.027053, -0.602275, -0.073462},
                               {2.431970, 0.029387, -0.599789, -0.071621}},
                              {{2.431934, 0.028196, -0.599802, -0.073809},
                               {2.432306, 0.028858, -0.599542, -0.072846}}},
                               tensor_options);
  ASSERT_EQ(result.sizes().size(),ref_output.sizes().size());
  ASSERT_TRUE(torch::allclose(result, ref_output, 1e-7, 1e-5,
                              /*equal_nan=*/true));

  // key_padding_mask
  key_padding_mask[0][2] = 1;
  key_padding_mask[1][1] = 1;
  key_padding_mask[1][2] = 1;
  result = model(decoder_input, memory_input, t_mask, m_mask,
                 key_padding_mask).detach();
  ref_output = torch::tensor({{{2.430025, 0.027643, -0.601164, -0.073476},
                               {2.4323, 0.029375, -0.599553, -0.071881}},
                              {{2.428523, 0.026838, -0.602226, -0.07391},
                               {2.432634, 0.029842, -0.599318, -0.071253}},
                              {{2.432278, 0.028152, -0.599555, -0.074139},
                               {2.432659, 0.029244, -0.599294, -0.072382}}},
                               tensor_options);
  ASSERT_EQ(result.sizes().size(),ref_output.sizes().size());
  ASSERT_TRUE(torch::allclose(result, ref_output, 1e-7, 1e-5,
                              /*equal_nan=*/true));

  // memory_key_padding_mask
  torch::Tensor t_key_padding_mask = {};
  key_padding_mask = torch::zeros({2, 5}, tensor_options) == 1;
  result = model(decoder_input, memory_input, t_mask, m_mask,
                 t_key_padding_mask, key_padding_mask).detach();
  ref_output = torch::tensor({{{2.430065, 0.027862, -0.601136, -0.073096},
                               {2.431935, 0.028907, -0.599809, -0.072488}},
                              {{2.428457, 0.027053, -0.602275, -0.073462},
                               {2.431970, 0.029387, -0.599789, -0.071621}},
                              {{2.431934, 0.028196, -0.599802, -0.073809},
                               {2.432306, 0.028858, -0.599542, -0.072846}}},
                               tensor_options);
  ASSERT_EQ(result.sizes().size(),ref_output.sizes().size());
  ASSERT_TRUE(torch::allclose(result, ref_output, 1e-7, 1e-5,
                              /*equal_nan=*/true));

  // memory_key_padding_mask
  key_padding_mask[0][4] = 1;
  key_padding_mask[1][3] = 1;
  key_padding_mask[1][4] = 1;
  result = model(decoder_input, memory_input, t_mask, m_mask,
                 t_key_padding_mask, key_padding_mask).detach();
  ref_output = torch::tensor({{{2.429757, 0.027358, -0.601351, -0.073816},
                               {2.432692, 0.028583, -0.599263, -0.073634}},
                              {{2.428247, 0.02662, -0.602419, -0.074123},
                               {2.432657, 0.029055, -0.599293, -0.072732}},
                              {{2.431515, 0.027687, -0.600096, -0.074459},
                               {2.433075, 0.028543, -0.598987, -0.073985}}},
                               tensor_options);
  ASSERT_EQ(result.sizes().size(),ref_output.sizes().size());
  ASSERT_TRUE(torch::allclose(result, ref_output, 1e-7, 1e-5,
                              /*equal_nan=*/true));

  // multiple layers no norm
  model = TransformerDecoder(TransformerDecoderOptions(decoder_layer, 2));
  if (is_cuda) {
    model->to(torch::kCUDA);
  }

  decoder_input = torch::tensor({{{20, 30, 40, 50}}}, tensor_options);
  memory_input = torch::tensor({{{60, 70, 80, 90}}}, tensor_options);
  result = model(decoder_input, memory_input).detach();
  ref_output = torch::tensor(
    {{{2.31316, 0.0950293, -0.671995, 0.102802}}},
    tensor_options);
  ASSERT_EQ(result.sizes().size(),ref_output.sizes().size());
  ASSERT_TRUE(torch::allclose(result, ref_output, 1e-7, 1e-5,
                              /*equal_nan=*/true));

  // multiple layers no norm
  model = TransformerDecoder(TransformerDecoderOptions(decoder_layer, 6));
  if (is_cuda) {
    model->to(torch::kCUDA);
  }
   // deterministic input
  decoder_input = torch::tensor({{{0.4517, 0.6793, 0.5313, 0.0034},
                                  {0.2678, 0.3677, 0.4459, 0.7166}},
                                 {{0.8100, 0.3716, 0.4096, 0.1976},
                                  {0.6958, 0.8844, 0.6081, 0.8315}},
                                 {{0.0494, 0.9343, 0.5955, 0.3830},
                                  {0.5404, 0.3464, 0.9378, 0.6200}}},
                                  tensor_options);
  memory_input = torch::tensor({{{0.7462, 0.6653, 0.5679, 0.4891},
                                 {0.5387, 0.1655, 0.3565, 0.0471}},
                                {{0.8335, 0.2799, 0.5031, 0.2947},
                                 {0.1402, 0.0318, 0.7636, 0.1346}},
                                {{0.6333, 0.9344, 0.1376, 0.9938},
                                 {0.8924, 0.2872, 0.6692, 0.2944}},
                                {{0.9897, 0.6915, 0.3154, 0.1733},
                                 {0.8645, 0.3513, 0.3064, 0.0767}},
                                {{0.8117, 0.2366, 0.4838, 0.7881},
                                 {0.3718, 0.4945, 0.9511, 0.0864}}},
                                 tensor_options);
  result = model(decoder_input, memory_input).detach();
  ref_output = torch::tensor({{{2.42794, 0.026164, -0.60263, -0.0747591},
                               {2.43113, 0.0279516, -0.600376, -0.0736896}},
                              {{2.42794, 0.026164, -0.60263, -0.0747591},
                               {2.43113, 0.0279516, -0.600376, -0.0736896}},
                              {{2.42794, 0.026164, -0.60263, -0.0747591},
                               {2.43113, 0.0279516, -0.600376, -0.0736896}}},
                               tensor_options);
  ASSERT_EQ(result.sizes().size(),ref_output.sizes().size());
  ASSERT_TRUE(torch::allclose(result, ref_output, 1e-7, 1e-5,
                              /*equal_nan=*/true));


  // multiple layers with norm
  LayerNorm norm(LayerNormOptions({decoder_layer.get()->options.d_model()}));
  model = TransformerDecoder(
    TransformerDecoderOptions(decoder_layer, 2).norm(AnyModule(norm)));
  if (is_cuda) {
    model->to(torch::kCUDA);
  }

  decoder_input = torch::tensor({{{20, 30, 40, 50}}}, tensor_options);
  memory_input = torch::tensor({{{60, 70, 80, 90}}}, tensor_options);
  result = model(decoder_input, memory_input).detach();
  ref_output = torch::tensor(
    {{{1.66166, -0.326986, -1.01466, -0.320017}}},
    tensor_options);
  ASSERT_EQ(result.sizes().size(),ref_output.sizes().size());
  ASSERT_TRUE(torch::allclose(result, ref_output, 1e-7, 1e-5,
                              /*equal_nan=*/true));

  // multiple layers with norm
  model = TransformerDecoder(
    TransformerDecoderOptions(decoder_layer, 6).norm(AnyModule(norm)));
  if (is_cuda) {
    model->to(torch::kCUDA);
  }
   // deterministic input
  decoder_input = torch::tensor({{{0.4517, 0.6793, 0.5313, 0.0034},
                                  {0.2678, 0.3677, 0.4459, 0.7166}},
                                 {{0.8100, 0.3716, 0.4096, 0.1976},
                                  {0.6958, 0.8844, 0.6081, 0.8315}},
                                 {{0.0494, 0.9343, 0.5955, 0.3830},
                                  {0.5404, 0.3464, 0.9378, 0.6200}}},
                                  tensor_options);
  memory_input = torch::tensor({{{0.7462, 0.6653, 0.5679, 0.4891},
                                 {0.5387, 0.1655, 0.3565, 0.0471}},
                                {{0.8335, 0.2799, 0.5031, 0.2947},
                                 {0.1402, 0.0318, 0.7636, 0.1346}},
                                {{0.6333, 0.9344, 0.1376, 0.9938},
                                 {0.8924, 0.2872, 0.6692, 0.2944}},
                                {{0.9897, 0.6915, 0.3154, 0.1733},
                                 {0.8645, 0.3513, 0.3064, 0.0767}},
                                {{0.8117, 0.2366, 0.4838, 0.7881},
                                 {0.3718, 0.4945, 0.9511, 0.0864}}},
                                 tensor_options);
  result = model(decoder_input, memory_input).detach();
  ref_output = torch::tensor({{{1.69559, -0.357291, -0.894741, -0.443553},
                               {1.69571, -0.357363, -0.894154, -0.444196}},
                              {{1.69559, -0.357291, -0.894741, -0.443553},
                               {1.69571, -0.357363, -0.894154, -0.444196}},
                              {{1.69559, -0.357291, -0.894741, -0.443553},
                               {1.69571, -0.357363, -0.894154, -0.444196}}},
                               tensor_options);
  ASSERT_EQ(result.sizes().size(),ref_output.sizes().size());
  ASSERT_TRUE(torch::allclose(result, ref_output, 1e-7, 1e-5,
                              /*equal_nan=*/true));

  //gelu activation test cases
  decoder_layer.get()->options.activation(torch::kGELU);
  model = TransformerDecoder(TransformerDecoderOptions(decoder_layer, 1));
  if (is_cuda) {
    model->to(torch::kCUDA);
  }

  // deterministic input
  decoder_input = torch::tensor({{{20, 30, 40, 50}}},
                                           tensor_options);
  memory_input = torch::tensor({{{60, 70, 80, 90}}},
                                          tensor_options);
  result = model(decoder_input, memory_input).detach();
  ref_output = torch::tensor(
    {{{2.306435, 0.095946, -0.675796, 0.10687}}},
    tensor_options);
  ASSERT_EQ(result.sizes().size(),ref_output.sizes().size());
  ASSERT_TRUE(torch::allclose(result, ref_output, 1e-7, 1e-5,
                              /*equal_nan=*/true));

  // deterministic input
  decoder_input = torch::tensor({{{9, 10, 11, 12}},
                                 {{11, 12, 13, 14}}},
                                 tensor_options);
  memory_input = torch::tensor({{{1, 2, 3, 4}}}, tensor_options);
  result = model(decoder_input, memory_input).detach();
  ref_output = torch::tensor({{{2.415448, 0.054389, -0.610932, -0.0156613}},
                              {{2.415448, 0.054389, -0.610932, -0.0156613}}},
                              tensor_options);
  ASSERT_EQ(result.sizes().size(),ref_output.sizes().size());
  ASSERT_TRUE(torch::allclose(result, ref_output, 1e-7, 1e-5,
                              /*equal_nan=*/true));

  // deterministic input
  decoder_input = torch::tensor({{{1, 2, 3, 4}},
                                 {{5, 6, 7, 8}}},
                                 tensor_options);
  memory_input = torch::tensor({{{9, 10, 11, 12}},
                                {{11, 12, 13, 14}}},
                                tensor_options);
  result = model(decoder_input, memory_input).detach();
  ref_output = torch::tensor({{{2.338531, 0.087709, -0.65776, 0.080646}},
                              {{2.338531, 0.087709, -0.65776, 0.080646}}},
                              tensor_options);
  ASSERT_EQ(result.sizes().size(),ref_output.sizes().size());
  ASSERT_TRUE(torch::allclose(result, ref_output, 1e-7, 1e-5,
                              /*equal_nan=*/true));

  // deterministic input
  decoder_input = torch::tensor({{{0.4517, 0.6793, 0.5313, 0.0034},
                                  {0.2678, 0.3677, 0.4459, 0.7166}},
                                 {{0.8100, 0.3716, 0.4096, 0.1976},
                                  {0.6958, 0.8844, 0.6081, 0.8315}},
                                 {{0.0494, 0.9343, 0.5955, 0.3830},
                                  {0.5404, 0.3464, 0.9378, 0.6200}}},
                                 tensor_options);
  memory_input = torch::tensor({{{0.7462, 0.6653, 0.5679, 0.4891},
                                 {0.5387, 0.1655, 0.3565, 0.0471}},
                                {{0.8335, 0.2799, 0.5031, 0.2947},
                                 {0.1402, 0.0318, 0.7636, 0.1346}},
                                {{0.6333, 0.9344, 0.1376, 0.9938},
                                 {0.8924, 0.2872, 0.6692, 0.2944}},
                                {{0.9897, 0.6915, 0.3154, 0.1733},
                                 {0.8645, 0.3513, 0.3064, 0.0767}},
                                {{0.8117, 0.2366, 0.4838, 0.7881},
                                 {0.3718, 0.4945, 0.9511, 0.0864}}},
                                tensor_options);
  result = model(decoder_input, memory_input).detach();
  ref_output = torch::tensor(
    {{{2.42049104, 0.03443088, -0.60793706, -0.05436271},
     {2.42210631, 0.03546578, -0.60679895, -0.05357488}},
    {{2.41907674, 0.0336104, -0.60892977, -0.05490462},
     {2.42216881, 0.03586554, -0.6067524, -0.05289126}},
    {{2.42205716, 0.03488046, -0.60683681, -0.05460596},
     {2.42240309, 0.0354595, -0.60659063, -0.05378816}}},
    tensor_options);
  ASSERT_EQ(result.sizes().size(),ref_output.sizes().size());
  ASSERT_TRUE(torch::allclose(result, ref_output, 1e-7, 1e-5,
                              /*equal_nan=*/true));

  // Multiple layers no norm
  model = TransformerDecoder(TransformerDecoderOptions(decoder_layer, 6));
  if (is_cuda) {
    model->to(torch::kCUDA);
  }
  decoder_input = torch::tensor({{{0.4517, 0.6793, 0.5313, 0.0034},
                                  {0.2678, 0.3677, 0.4459, 0.7166}},
                                 {{0.8100, 0.3716, 0.4096, 0.1976},
                                  {0.6958, 0.8844, 0.6081, 0.8315}},
                                 {{0.0494, 0.9343, 0.5955, 0.3830},
                                  {0.5404, 0.3464, 0.9378, 0.6200}}},
                                 tensor_options);
  memory_input = torch::tensor({{{0.7462, 0.6653, 0.5679, 0.4891},
                                 {0.5387, 0.1655, 0.3565, 0.0471}},
                                {{0.8335, 0.2799, 0.5031, 0.2947},
                                 {0.1402, 0.0318, 0.7636, 0.1346}},
                                {{0.6333, 0.9344, 0.1376, 0.9938},
                                 {0.8924, 0.2872, 0.6692, 0.2944}},
                                {{0.9897, 0.6915, 0.3154, 0.1733},
                                 {0.8645, 0.3513, 0.3064, 0.0767}},
                                {{0.8117, 0.2366, 0.4838, 0.7881},
                                 {0.3718, 0.4945, 0.9511, 0.0864}}},
                                tensor_options);
  result = model(decoder_input, memory_input).detach();
  ref_output = torch::tensor({{{2.41859, 0.0328114, -0.609269, -0.0560386},
                               {2.42138, 0.034598, -0.607316, -0.0546574}},
                              {{2.41859, 0.0328114, -0.609269, -0.0560386},
                               {2.42138, 0.034598, -0.607316, -0.0546574}},
                              {{2.41859, 0.0328114, -0.609269, -0.0560386},
                               {2.42138, 0.034598, -0.607316, -0.0546574}}},
                              tensor_options);
  ASSERT_EQ(result.sizes().size(),ref_output.sizes().size());
  ASSERT_TRUE(torch::allclose(result, ref_output, 1e-7, 1e-5,
                              /*equal_nan=*/true));

  // Multiple layers with norm
  norm = LayerNorm(LayerNormOptions({decoder_layer.get()->options.d_model()}));
  model = TransformerDecoder(TransformerDecoderOptions(decoder_layer, 6).norm(AnyModule(norm)));
  if (is_cuda) {
    model->to(torch::kCUDA);
  }

  decoder_input = torch::tensor({{{0.4517, 0.6793, 0.5313, 0.0034},
                                  {0.2678, 0.3677, 0.4459, 0.7166}},
                                 {{0.8100, 0.3716, 0.4096, 0.1976},
                                  {0.6958, 0.8844, 0.6081, 0.8315}},
                                 {{0.0494, 0.9343, 0.5955, 0.3830},
                                  {0.5404, 0.3464, 0.9378, 0.6200}}},
                                 tensor_options);
  memory_input = torch::tensor({{{0.7462, 0.6653, 0.5679, 0.4891},
                                 {0.5387, 0.1655, 0.3565, 0.0471}},
                                {{0.8335, 0.2799, 0.5031, 0.2947},
                                 {0.1402, 0.0318, 0.7636, 0.1346}},
                                {{0.6333, 0.9344, 0.1376, 0.9938},
                                 {0.8924, 0.2872, 0.6692, 0.2944}},
                                {{0.9897, 0.6915, 0.3154, 0.1733},
                                 {0.8645, 0.3513, 0.3064, 0.0767}},
                                {{0.8117, 0.2366, 0.4838, 0.7881},
                                 {0.3718, 0.4945, 0.9511, 0.0864}}},
                                tensor_options);
  result = model(decoder_input, memory_input).detach();
  ref_output = torch::tensor({{{1.69298, -0.355163, -0.906375, -0.431439},
                               {1.69305, -0.355195, -0.906062, -0.431791}},
                              {{1.69298, -0.355163, -0.906375, -0.431439},
                               {1.69305, -0.355195, -0.906062, -0.431791}},
                              {{1.69298, -0.355163, -0.906375, -0.431439},
                               {1.69305, -0.355195, -0.906062, -0.431791}}},
                             tensor_options);
  ASSERT_EQ(result.sizes().size(),ref_output.sizes().size());
  ASSERT_TRUE(torch::allclose(result, ref_output, 1e-7, 1e-5,
                              /*equal_nan=*/true));

}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(TransformerTest, TransformerDecoder) {
  transformer_decoder_test_helper(false);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(TransformerTest, TransformerDecoder_CUDA) {
  transformer_decoder_test_helper(true);
}


// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(TransformerTest, PrettyPrintTransformerDecoder) {
  LayerNorm norm = LayerNorm(LayerNormOptions({4}));
  TransformerDecoderOptions options(
    TransformerDecoderOptions(
      TransformerDecoderLayerOptions(4, 2),2).norm(AnyModule(norm)));
  ASSERT_EQ(
      c10::str(TransformerDecoder(options)),
      "torch::nn::TransformerDecoderImpl(\n"
      "  (layers): torch::nn::ModuleList(\n"
      "    (0): torch::nn::TransformerDecoderLayerImpl(\n"
      "      (self_attn): torch::nn::MultiheadAttention(\n"
      "        (out_proj): torch::nn::Linear(in_features=4, out_features=4, bias=true)\n"
      "      )\n"
      "      (multihead_attn): torch::nn::MultiheadAttention(\n"
      "        (out_proj): torch::nn::Linear(in_features=4, out_features=4, bias=true)\n"
      "      )\n"
      "      (linear1): torch::nn::Linear(in_features=4, out_features=2048, bias=true)\n"
      "      (dropout): torch::nn::Dropout(p=0.1, inplace=false)\n"
      "      (linear2): torch::nn::Linear(in_features=2048, out_features=4, bias=true)\n"
      "      (norm1): torch::nn::LayerNorm([4], eps=1e-05, elementwise_affine=true)\n"
      "      (norm2): torch::nn::LayerNorm([4], eps=1e-05, elementwise_affine=true)\n"
      "      (norm3): torch::nn::LayerNorm([4], eps=1e-05, elementwise_affine=true)\n"
      "      (dropout1): torch::nn::Dropout(p=0.1, inplace=false)\n"
      "      (dropout2): torch::nn::Dropout(p=0.1, inplace=false)\n"
      "      (dropout3): torch::nn::Dropout(p=0.1, inplace=false)\n"
      "    )\n"
      "    (1): torch::nn::TransformerDecoderLayerImpl(\n"
      "      (self_attn): torch::nn::MultiheadAttention(\n"
      "        (out_proj): torch::nn::Linear(in_features=4, out_features=4, bias=true)\n"
      "      )\n"
      "      (multihead_attn): torch::nn::MultiheadAttention(\n"
      "        (out_proj): torch::nn::Linear(in_features=4, out_features=4, bias=true)\n"
      "      )\n"
      "      (linear1): torch::nn::Linear(in_features=4, out_features=2048, bias=true)\n"
      "      (dropout): torch::nn::Dropout(p=0.1, inplace=false)\n"
      "      (linear2): torch::nn::Linear(in_features=2048, out_features=4, bias=true)\n"
      "      (norm1): torch::nn::LayerNorm([4], eps=1e-05, elementwise_affine=true)\n"
      "      (norm2): torch::nn::LayerNorm([4], eps=1e-05, elementwise_affine=true)\n"
      "      (norm3): torch::nn::LayerNorm([4], eps=1e-05, elementwise_affine=true)\n"
      "      (dropout1): torch::nn::Dropout(p=0.1, inplace=false)\n"
      "      (dropout2): torch::nn::Dropout(p=0.1, inplace=false)\n"
      "      (dropout3): torch::nn::Dropout(p=0.1, inplace=false)\n"
      "    )\n"
      "  )\n"
      "  (norm): torch::nn::LayerNorm([4], eps=1e-05, elementwise_affine=true)\n"
      ")");
}

void transformer_test_helper(bool is_cuda) {
    // this is a deterministic test for Transformere
    torch::Device device = is_cuda ? torch::kCUDA : torch::kCPU;
    torch::TensorOptions tensor_options = torch::TensorOptions().dtype(torch::kFloat32).device(device);

    // transformer created encoder/decoder
    Transformer model(TransformerOptions()
      .d_model(4)
      .nhead(2)
      .num_encoder_layers(2)
      .num_decoder_layers(1)
      .dim_feedforward(16)
      .dropout(0.0)
      .activation(torch::kReLU));

    set_parameter_to_constants<Transformer>(model, tensor_options);
    if (tensor_options.device() == torch::kCUDA) {
      model->to(torch::kCUDA);
    }

    // transformer with customized encoder/decoder
    LayerNorm enorm(LayerNormOptions({4}));
    TransformerEncoder encoder(TransformerEncoderOptions(
      TransformerEncoderLayerOptions(4, 2).dim_feedforward(16).dropout(0.0), 2).norm(AnyModule(enorm)));

    LayerNorm dnorm(LayerNormOptions({4}));
    TransformerDecoder decoder(TransformerDecoderOptions(
      TransformerDecoderLayerOptions(4, 2).dim_feedforward(16).dropout(0.0), 1).norm(AnyModule(dnorm)));

    Transformer model_cus(TransformerOptions()
      .d_model(4)
      .nhead(2)
      .custom_encoder(AnyModule(encoder))
      .custom_decoder(AnyModule(decoder)));

    set_parameter_to_constants<Transformer>(model_cus, tensor_options);
    if (tensor_options.device() == torch::kCUDA) {
      model_cus->to(torch::kCUDA);
    }

    // test cases
    torch::Tensor src = torch::tensor({
      {{1.0,  2.0,  3.0,  4.0},  {5.0, 6.0, 7.0, 8.0}},
      {{9.0,  10.0, 11.0, 12.0}, {13.0, 14.0, 15.0, 16.0}},
      {{17.0, 18.0, 19.0, 20.0}, {21.0, 22.0, 23.0, 24.0}}}, tensor_options);

    torch::Tensor tgt = torch::tensor({
      {{1.0,  2.0,  3.0,  4.0},  {5.0, 6.0, 7.0, 8.0}},
      {{9.0,  10.0, 11.0, 12.0}, {13.0, 14.0, 15.0, 16.0}}}, tensor_options);

    torch::Tensor ref_output = torch::tensor({
      {{2.695875, 0.347114, -0.044355, -0.549541}, {2.696091, 0.347015, -0.044770, -0.548522}},
      {{2.695875, 0.347114, -0.044355, -0.549541}, {2.696091, 0.347015, -0.044770, -0.548522}}}, tensor_options);
    torch::Tensor result = model(src, tgt);
    torch::Tensor result_cus = model_cus(src, tgt);
    ASSERT_EQ(result.sizes(), ref_output.sizes());
    ASSERT_TRUE(result.equal(result_cus));
    ASSERT_TRUE(torch::allclose(result, ref_output, 1e-7, 1e-5, /*equal_nan=*/true));

    torch::Tensor src_mask = Transformer::Impl::generate_square_subsequent_mask(src.size(0)).to(tensor_options);
    ref_output = torch::tensor({
      {{2.695875, 0.347114, -0.044355, -0.549541}, {2.696091, 0.347015, -0.044770, -0.548522}},
      {{2.695875, 0.347114, -0.044355, -0.549541}, {2.696091, 0.347015, -0.044770, -0.548522}}}, tensor_options);
    result = model(src, tgt, src_mask);
    result_cus = model_cus(src, tgt, src_mask);
    ASSERT_EQ(result.sizes(), ref_output.sizes());
    ASSERT_TRUE(result.equal(result_cus));
    ASSERT_TRUE(torch::allclose(result, ref_output, 1e-7, 1e-5, /*equal_nan=*/true));

    torch::Tensor tgt_key_padding_mask = torch::zeros({tgt.size(1), tgt.size(0)}, tensor_options) == 1;
    tgt_key_padding_mask[0][0] = 1;
    tgt_key_padding_mask[1][1] = 1;
    ref_output = torch::tensor({
      {{2.696114, 0.347004, -0.044813, -0.548417}, {2.696091, 0.347015, -0.044770, -0.548522}},
      {{2.696114, 0.347004, -0.044813, -0.548417}, {2.696091, 0.347015, -0.044770, -0.548522}}}, tensor_options);
    result = model(src, tgt, src_mask, torch::Tensor(), torch::Tensor(), torch::Tensor(), tgt_key_padding_mask);
    result_cus = model_cus(src, tgt, src_mask, torch::Tensor(), torch::Tensor(), torch::Tensor(), tgt_key_padding_mask);
    ASSERT_EQ(result.sizes(), ref_output.sizes());
    ASSERT_TRUE(result.equal(result_cus));
    ASSERT_TRUE(torch::allclose(result, ref_output, 1e-7, 1e-5, /*equal_nan=*/true));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(TransformerTest, Transformer) {
  transformer_test_helper(false);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(TransformerTest, Transformer_CUDA) {
  transformer_test_helper(true);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(TransformerTest, TransformerArgsCorrectness) {
  Transformer model(TransformerOptions()
    .d_model(4)
    .nhead(2)
    .num_encoder_layers(2)
    .num_decoder_layers(1)
    .dim_feedforward(16)
    .dropout(0.0)
    .activation(torch::kReLU));

  torch::Tensor src = torch::randn({2, 3, 4});
  torch::Tensor tgt = torch::randn({3, 2, 4});

  ASSERT_THROWS_WITH(model(src, tgt), "src and tgt should have equal batch size");

  tgt = torch::randn({2, 3, 3});
  ASSERT_THROWS_WITH(model(src, tgt), "src and tgt should have same feature size as d_model");

  src = torch::randn({2, 3});
  ASSERT_THROWS_WITH(model(src, tgt), "src and tgt should have 3 dimensions");
}
