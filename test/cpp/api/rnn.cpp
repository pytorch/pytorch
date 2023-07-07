#include <gtest/gtest.h>

#include <torch/torch.h>

#include <test/cpp/api/support.h>

using namespace torch::nn;
using namespace torch::test;

template <typename R, typename Func>
bool test_RNN_xor(Func&& model_maker, bool cuda = false) {
  torch::manual_seed(0);

  auto nhid = 32;
  auto model = std::make_shared<SimpleContainer>();
  auto l1 = model->add(Linear(1, nhid), "l1");
  auto rnn_model = model_maker(nhid);
  auto rnn = model->add(rnn_model, "rnn");
  auto nout = nhid;
  if (rnn_model.get()->options_base.proj_size() > 0) {
    nout = rnn_model.get()->options_base.proj_size();
  }
  auto lo = model->add(Linear(nout, 1), "lo");

  torch::optim::Adam optimizer(model->parameters(), 1e-2);
  auto forward_op = [&](torch::Tensor x) {
    auto T = x.size(0);
    auto B = x.size(1);
    x = x.view({T * B, 1});
    x = l1->forward(x).view({T, B, nhid}).tanh_();
    x = std::get<0>(rnn->forward(x))[T - 1];
    x = lo->forward(x);
    return x;
  };

  if (cuda) {
    model->to(torch::kCUDA);
  }

  float running_loss = 1;
  int epoch = 0;
  auto max_epoch = 1500;
  while (running_loss > 1e-2) {
    auto bs = 16U;
    auto nlen = 5U;

    const auto backend = cuda ? torch::kCUDA : torch::kCPU;
    auto inputs =
        torch::rand({nlen, bs, 1}, backend).round().to(torch::kFloat32);
    auto labels = inputs.sum(0).detach();
    inputs.set_requires_grad(true);
    auto outputs = forward_op(inputs);
    torch::Tensor loss = torch::mse_loss(outputs, labels);

    optimizer.zero_grad();
    loss.backward();
    optimizer.step();

    // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,cppcoreguidelines-avoid-magic-numbers,bugprone-narrowing-conversions)
    running_loss = running_loss * 0.99 + loss.item<float>() * 0.01;
    if (epoch > max_epoch) {
      return false;
    }
    epoch++;
  }
  return true;
};

void check_lstm_sizes(
    std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>>
        lstm_output) {
  // Expect the LSTM to have 64 outputs and 3 layers, with an input of batch
  // 10 and 16 time steps (10 x 16 x n)

  torch::Tensor output = std::get<0>(lstm_output);
  std::tuple<torch::Tensor, torch::Tensor> state = std::get<1>(lstm_output);
  torch::Tensor hx = std::get<0>(state);
  torch::Tensor cx = std::get<1>(state);

  ASSERT_EQ(output.ndimension(), 3);
  ASSERT_EQ(output.size(0), 10);
  ASSERT_EQ(output.size(1), 16);
  ASSERT_EQ(output.size(2), 64);

  ASSERT_EQ(hx.ndimension(), 3);
  ASSERT_EQ(hx.size(0), 3); // layers
  ASSERT_EQ(hx.size(1), 16); // Batchsize
  ASSERT_EQ(hx.size(2), 64); // 64 hidden dims

  ASSERT_EQ(cx.ndimension(), 3);
  ASSERT_EQ(cx.size(0), 3); // layers
  ASSERT_EQ(cx.size(1), 16); // Batchsize
  ASSERT_EQ(cx.size(2), 64); // 64 hidden dims

  // Something is in the hiddens
  ASSERT_GT(hx.norm().item<float>(), 0);
  ASSERT_GT(cx.norm().item<float>(), 0);
}

void check_lstm_sizes_proj(
    std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>>
        lstm_output) {
  // Expect the LSTM to have 32 outputs and 3 layers, with an input of batch
  // 10 and 16 time steps (10 x 16 x n)

  torch::Tensor output = std::get<0>(lstm_output);
  std::tuple<torch::Tensor, torch::Tensor> state = std::get<1>(lstm_output);
  torch::Tensor hx = std::get<0>(state);
  torch::Tensor cx = std::get<1>(state);

  ASSERT_EQ(output.ndimension(), 3);
  ASSERT_EQ(output.size(0), 10);
  ASSERT_EQ(output.size(1), 16);
  ASSERT_EQ(output.size(2), 32);

  ASSERT_EQ(hx.ndimension(), 3);
  ASSERT_EQ(hx.size(0), 3); // layers
  ASSERT_EQ(hx.size(1), 16); // Batchsize
  ASSERT_EQ(hx.size(2), 32); // 32 hidden dims

  ASSERT_EQ(cx.ndimension(), 3);
  ASSERT_EQ(cx.size(0), 3); // layers
  ASSERT_EQ(cx.size(1), 16); // Batchsize
  ASSERT_EQ(cx.size(2), 64); // 64 cell dims

  // Something is in the hiddens
  ASSERT_GT(hx.norm().item<float>(), 0);
  ASSERT_GT(cx.norm().item<float>(), 0);
}

struct RNNTest : torch::test::SeedingFixture {};

TEST_F(RNNTest, CheckOutputSizes) {
  LSTM model(LSTMOptions(128, 64).num_layers(3).dropout(0.2));
  // Input size is: sequence length, batch size, input size
  auto x = torch::randn({10, 16, 128}, torch::requires_grad());
  auto output = model->forward(x);
  auto y = x.mean();

  y.backward();
  check_lstm_sizes(output);

  auto next = model->forward(x, std::get<1>(output));

  check_lstm_sizes(next);

  auto output_hx = std::get<0>(std::get<1>(output));
  auto output_cx = std::get<1>(std::get<1>(output));

  auto next_hx = std::get<0>(std::get<1>(next));
  auto next_cx = std::get<1>(std::get<1>(next));

  torch::Tensor diff =
      torch::cat({next_hx, next_cx}, 0) - torch::cat({output_hx, output_cx}, 0);

  // Hiddens changed
  ASSERT_GT(diff.abs().sum().item<float>(), 1e-3);
}

TEST_F(RNNTest, CheckOutputSizesProj) {
  LSTM model(LSTMOptions(128, 64).num_layers(3).dropout(0.2).proj_size(32));
  // Input size is: sequence length, batch size, input size
  auto x = torch::randn({10, 16, 128}, torch::requires_grad());
  auto output = model->forward(x);
  auto y = x.mean();

  y.backward();
  check_lstm_sizes_proj(output);

  auto next = model->forward(x, std::get<1>(output));

  check_lstm_sizes_proj(next);

  auto output_hx = std::get<0>(std::get<1>(output));
  auto output_cx = std::get<1>(std::get<1>(output));

  auto next_hx = std::get<0>(std::get<1>(next));
  auto next_cx = std::get<1>(std::get<1>(next));

  torch::Tensor diff = next_hx - output_hx;
  // Hiddens changed
  ASSERT_GT(diff.abs().sum().item<float>(), 1e-3);
  diff = next_cx - output_cx;
  ASSERT_GT(diff.abs().sum().item<float>(), 1e-3);
}

TEST_F(RNNTest, CheckOutputValuesMatchPyTorch) {
  torch::manual_seed(0);
  // Make sure the outputs match pytorch outputs
  LSTM model(2, 2);
  for (auto& v : model->parameters()) {
    float size = v.numel();
    auto p = static_cast<float*>(v.storage().mutable_data());
    for (size_t i = 0; i < size; i++) {
      p[i] = i / size;
    }
  }

  auto x = torch::empty({3, 4, 2}, torch::requires_grad());
  float size = x.numel();
  auto p = static_cast<float*>(x.storage().mutable_data());
  for (size_t i = 0; i < size; i++) {
    p[i] = (size - i) / size;
  }

  auto out = model->forward(x);
  ASSERT_EQ(std::get<0>(out).ndimension(), 3);
  ASSERT_EQ(std::get<0>(out).size(0), 3);
  ASSERT_EQ(std::get<0>(out).size(1), 4);
  ASSERT_EQ(std::get<0>(out).size(2), 2);

  auto flat = std::get<0>(out).view(3 * 4 * 2);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
  float c_out[] = {0.4391, 0.5402, 0.4330, 0.5324, 0.4261, 0.5239,
                   0.4183, 0.5147, 0.6822, 0.8064, 0.6726, 0.7968,
                   0.6620, 0.7860, 0.6501, 0.7741, 0.7889, 0.9003,
                   0.7769, 0.8905, 0.7635, 0.8794, 0.7484, 0.8666};
  for (size_t i = 0; i < 3 * 4 * 2; i++) {
    ASSERT_LT(std::abs(flat[i].item<float>() - c_out[i]), 1e-3);
  }

  auto hx = std::get<0>(std::get<1>(out));
  auto cx = std::get<1>(std::get<1>(out));

  ASSERT_EQ(hx.ndimension(), 3); // layers x B x 2
  ASSERT_EQ(hx.size(0), 1);
  ASSERT_EQ(hx.size(1), 4);
  ASSERT_EQ(hx.size(2), 2);

  ASSERT_EQ(cx.ndimension(), 3); // layers x B x 2
  ASSERT_EQ(cx.size(0), 1);
  ASSERT_EQ(cx.size(1), 4);
  ASSERT_EQ(cx.size(2), 2);

  flat = torch::cat({hx, cx}, 0).view(16);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
  float h_out[] = {
      0.7889,
      0.9003,
      0.7769,
      0.8905,
      0.7635,
      0.8794,
      0.7484,
      0.8666,
      1.1647,
      1.6106,
      1.1425,
      1.5726,
      1.1187,
      1.5329,
      1.0931,
      1.4911};
  for (size_t i = 0; i < 16; i++) {
    ASSERT_LT(std::abs(flat[i].item<float>() - h_out[i]), 1e-3);
  }
}

TEST_F(RNNTest, EndToEndLSTM) {
  ASSERT_TRUE(test_RNN_xor<LSTM>(
      [](int s) { return LSTM(LSTMOptions(s, s).num_layers(2)); }));
}

TEST_F(RNNTest, EndToEndLSTMProj) {
  ASSERT_TRUE(test_RNN_xor<LSTM>([](int s) {
    return LSTM(LSTMOptions(s, s).num_layers(2).proj_size(s / 2));
  }));
}

TEST_F(RNNTest, EndToEndGRU) {
  ASSERT_TRUE(test_RNN_xor<GRU>(
      [](int s) { return GRU(GRUOptions(s, s).num_layers(2)); }));
}

TEST_F(RNNTest, EndToEndRNNRelu) {
  ASSERT_TRUE(test_RNN_xor<RNN>([](int s) {
    return RNN(RNNOptions(s, s).nonlinearity(torch::kReLU).num_layers(2));
  }));
}

TEST_F(RNNTest, EndToEndRNNTanh) {
  ASSERT_TRUE(test_RNN_xor<RNN>([](int s) {
    return RNN(RNNOptions(s, s).nonlinearity(torch::kTanh).num_layers(2));
  }));
}

TEST_F(RNNTest, Sizes_CUDA) {
  torch::manual_seed(0);
  LSTM model(LSTMOptions(128, 64).num_layers(3).dropout(0.2));
  model->to(torch::kCUDA);
  auto x =
      torch::randn({10, 16, 128}, torch::requires_grad().device(torch::kCUDA));
  auto output = model->forward(x);
  auto y = x.mean();

  y.backward();
  check_lstm_sizes(output);

  auto next = model->forward(x, std::get<1>(output));

  check_lstm_sizes(next);

  auto output_hx = std::get<0>(std::get<1>(output));
  auto output_cx = std::get<1>(std::get<1>(output));

  auto next_hx = std::get<0>(std::get<1>(next));
  auto next_cx = std::get<1>(std::get<1>(next));

  torch::Tensor diff =
      torch::cat({next_hx, next_cx}, 0) - torch::cat({output_hx, output_cx}, 0);

  // Hiddens changed
  ASSERT_GT(diff.abs().sum().item<float>(), 1e-3);
}

TEST_F(RNNTest, SizesProj_CUDA) {
  torch::manual_seed(0);
  LSTM model(LSTMOptions(128, 64).num_layers(3).dropout(0.2).proj_size(32));
  model->to(torch::kCUDA);
  auto x =
      torch::randn({10, 16, 128}, torch::requires_grad().device(torch::kCUDA));
  auto output = model->forward(x);
  auto y = x.mean();

  y.backward();
  check_lstm_sizes_proj(output);

  auto next = model->forward(x, std::get<1>(output));

  check_lstm_sizes_proj(next);

  auto output_hx = std::get<0>(std::get<1>(output));
  auto output_cx = std::get<1>(std::get<1>(output));

  auto next_hx = std::get<0>(std::get<1>(next));
  auto next_cx = std::get<1>(std::get<1>(next));

  torch::Tensor diff = next_hx - output_hx;
  // Hiddens changed
  ASSERT_GT(diff.abs().sum().item<float>(), 1e-3);
  diff = next_cx - output_cx;
  ASSERT_GT(diff.abs().sum().item<float>(), 1e-3);
}

TEST_F(RNNTest, EndToEndLSTM_CUDA) {
  ASSERT_TRUE(test_RNN_xor<LSTM>(
      [](int s) { return LSTM(LSTMOptions(s, s).num_layers(2)); }, true));
}

TEST_F(RNNTest, EndToEndLSTMProj_CUDA) {
  ASSERT_TRUE(test_RNN_xor<LSTM>(
      [](int s) {
        return LSTM(LSTMOptions(s, s).num_layers(2).proj_size(s / 2));
      },
      true));
}

TEST_F(RNNTest, EndToEndGRU_CUDA) {
  ASSERT_TRUE(test_RNN_xor<GRU>(
      [](int s) { return GRU(GRUOptions(s, s).num_layers(2)); }, true));
}

TEST_F(RNNTest, EndToEndRNNRelu_CUDA) {
  ASSERT_TRUE(test_RNN_xor<RNN>(
      [](int s) {
        return RNN(RNNOptions(s, s).nonlinearity(torch::kReLU).num_layers(2));
      },
      true));
}
TEST_F(RNNTest, EndToEndRNNTanh_CUDA) {
  ASSERT_TRUE(test_RNN_xor<RNN>(
      [](int s) {
        return RNN(RNNOptions(s, s).nonlinearity(torch::kTanh).num_layers(2));
      },
      true));
}

TEST_F(RNNTest, PrettyPrintRNNs) {
  ASSERT_EQ(
      c10::str(LSTM(LSTMOptions(128, 64).num_layers(3).dropout(0.2))),
      "torch::nn::LSTM(input_size=128, hidden_size=64, num_layers=3, bias=true, batch_first=false, dropout=0.2, bidirectional=false)");
  ASSERT_EQ(
      c10::str(
          LSTM(LSTMOptions(128, 64).num_layers(3).dropout(0.2).proj_size(32))),
      "torch::nn::LSTM(input_size=128, hidden_size=64, num_layers=3, bias=true, batch_first=false, dropout=0.2, bidirectional=false, proj_size=32)");
  ASSERT_EQ(
      c10::str(GRU(GRUOptions(128, 64).num_layers(3).dropout(0.5))),
      "torch::nn::GRU(input_size=128, hidden_size=64, num_layers=3, bias=true, batch_first=false, dropout=0.5, bidirectional=false)");
  ASSERT_EQ(
      c10::str(RNN(RNNOptions(128, 64).num_layers(3).dropout(0.2).nonlinearity(
          torch::kTanh))),
      "torch::nn::RNN(input_size=128, hidden_size=64, num_layers=3, bias=true, batch_first=false, dropout=0.2, bidirectional=false)");
}

// This test assures that flatten_parameters does not crash,
// when bidirectional is set to true
// https://github.com/pytorch/pytorch/issues/19545
TEST_F(RNNTest, BidirectionalFlattenParameters) {
  GRU gru(GRUOptions(100, 256).num_layers(2).bidirectional(true));
  gru->flatten_parameters();
}

template <typename Impl>
void copyParameters(
    torch::nn::ModuleHolder<Impl>& target,
    std::string t_suffix,
    const torch::nn::ModuleHolder<Impl>& source,
    std::string s_suffix) {
  at::NoGradGuard guard;
  target->named_parameters()["weight_ih_l" + t_suffix].copy_(
      source->named_parameters()["weight_ih_l" + s_suffix]);
  target->named_parameters()["weight_hh_l" + t_suffix].copy_(
      source->named_parameters()["weight_hh_l" + s_suffix]);
  target->named_parameters()["bias_ih_l" + t_suffix].copy_(
      source->named_parameters()["bias_ih_l" + s_suffix]);
  target->named_parameters()["bias_hh_l" + t_suffix].copy_(
      source->named_parameters()["bias_hh_l" + s_suffix]);
}

std::tuple<torch::Tensor, torch::Tensor> gru_output_to_device(
    std::tuple<torch::Tensor, torch::Tensor> gru_output,
    torch::Device device) {
  return std::make_tuple(
      std::get<0>(gru_output).to(device), std::get<1>(gru_output).to(device));
}

std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>>
lstm_output_to_device(
    std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>>
        lstm_output,
    torch::Device device) {
  auto hidden_states = std::get<1>(lstm_output);
  return std::make_tuple(
      std::get<0>(lstm_output).to(device),
      std::make_tuple(
          std::get<0>(hidden_states).to(device),
          std::get<1>(hidden_states).to(device)));
}

// This test is a port of python code introduced here:
// https://towardsdatascience.com/understanding-bidirectional-rnn-in-pytorch-5bd25a5dd66
// Reverse forward of bidirectional GRU should act
// as regular forward of unidirectional GRU
void BidirectionalGRUReverseForward(bool cuda) {
  auto opt = torch::TensorOptions()
                 .dtype(torch::kFloat32)
                 .requires_grad(false)
                 .device(cuda ? torch::kCUDA : torch::kCPU);
  auto input = torch::tensor({1, 2, 3, 4, 5}, opt).reshape({5, 1, 1});
  auto input_reversed = torch::tensor({5, 4, 3, 2, 1}, opt).reshape({5, 1, 1});

  auto gru_options = GRUOptions(1, 1).num_layers(1).batch_first(false);
  GRU bi_grus{gru_options.bidirectional(true)};
  GRU reverse_gru{gru_options.bidirectional(false)};

  if (cuda) {
    bi_grus->to(torch::kCUDA);
    reverse_gru->to(torch::kCUDA);
  }

  // Now make sure the weights of the reverse gru layer match
  // ones of the (reversed) bidirectional's:
  copyParameters(reverse_gru, "0", bi_grus, "0_reverse");

  auto bi_output = bi_grus->forward(input);
  auto reverse_output = reverse_gru->forward(input_reversed);

  if (cuda) {
    bi_output = gru_output_to_device(bi_output, torch::kCPU);
    reverse_output = gru_output_to_device(reverse_output, torch::kCPU);
  }

  ASSERT_EQ(
      std::get<0>(bi_output).size(0), std::get<0>(reverse_output).size(0));
  auto size = std::get<0>(bi_output).size(0);
  for (int i = 0; i < size; i++) {
    ASSERT_EQ(
        std::get<0>(bi_output)[i][0][1].item<float>(),
        std::get<0>(reverse_output)[size - 1 - i][0][0].item<float>());
  }
  // The hidden states of the reversed GRUs sits
  // in the odd indices in the first dimension.
  ASSERT_EQ(
      std::get<1>(bi_output)[1][0][0].item<float>(),
      std::get<1>(reverse_output)[0][0][0].item<float>());
}

TEST_F(RNNTest, BidirectionalGRUReverseForward) {
  BidirectionalGRUReverseForward(false);
}

TEST_F(RNNTest, BidirectionalGRUReverseForward_CUDA) {
  BidirectionalGRUReverseForward(true);
}

// Reverse forward of bidirectional LSTM should act
// as regular forward of unidirectional LSTM
void BidirectionalLSTMReverseForwardTest(bool cuda) {
  auto opt = torch::TensorOptions()
                 .dtype(torch::kFloat32)
                 .requires_grad(false)
                 .device(cuda ? torch::kCUDA : torch::kCPU);
  auto input = torch::tensor({1, 2, 3, 4, 5}, opt).reshape({5, 1, 1});
  auto input_reversed = torch::tensor({5, 4, 3, 2, 1}, opt).reshape({5, 1, 1});

  auto lstm_opt = LSTMOptions(1, 1).num_layers(1).batch_first(false);

  LSTM bi_lstm{lstm_opt.bidirectional(true)};
  LSTM reverse_lstm{lstm_opt.bidirectional(false)};

  if (cuda) {
    bi_lstm->to(torch::kCUDA);
    reverse_lstm->to(torch::kCUDA);
  }

  // Now make sure the weights of the reverse lstm layer match
  // ones of the (reversed) bidirectional's:
  copyParameters(reverse_lstm, "0", bi_lstm, "0_reverse");

  auto bi_output = bi_lstm->forward(input);
  auto reverse_output = reverse_lstm->forward(input_reversed);

  if (cuda) {
    bi_output = lstm_output_to_device(bi_output, torch::kCPU);
    reverse_output = lstm_output_to_device(reverse_output, torch::kCPU);
  }

  ASSERT_EQ(
      std::get<0>(bi_output).size(0), std::get<0>(reverse_output).size(0));
  auto size = std::get<0>(bi_output).size(0);
  for (int i = 0; i < size; i++) {
    ASSERT_EQ(
        std::get<0>(bi_output)[i][0][1].item<float>(),
        std::get<0>(reverse_output)[size - 1 - i][0][0].item<float>());
  }
  // The hidden states of the reversed LSTM sits
  // in the odd indices in the first dimension.
  ASSERT_EQ(
      std::get<0>(std::get<1>(bi_output))[1][0][0].item<float>(),
      std::get<0>(std::get<1>(reverse_output))[0][0][0].item<float>());
  ASSERT_EQ(
      std::get<1>(std::get<1>(bi_output))[1][0][0].item<float>(),
      std::get<1>(std::get<1>(reverse_output))[0][0][0].item<float>());
}

TEST_F(RNNTest, BidirectionalLSTMReverseForward) {
  BidirectionalLSTMReverseForwardTest(false);
}

TEST_F(RNNTest, BidirectionalLSTMReverseForward_CUDA) {
  BidirectionalLSTMReverseForwardTest(true);
}

TEST_F(RNNTest, BidirectionalMultilayerGRU_CPU_vs_CUDA) {
  // Create two GRUs with the same options
  auto opt =
      GRUOptions(2, 4).num_layers(3).batch_first(false).bidirectional(true);
  GRU gru_cpu{opt};
  GRU gru_cuda{opt};

  // Copy weights and biases from CPU GRU to CUDA GRU
  {
    at::NoGradGuard guard;
    for (const auto& param : gru_cpu->named_parameters(/*recurse=*/false)) {
      gru_cuda->named_parameters()[param.key()].copy_(
          gru_cpu->named_parameters()[param.key()]);
    }
  }

  gru_cpu->flatten_parameters();
  gru_cuda->flatten_parameters();

  // Move GRU to CUDA
  gru_cuda->to(torch::kCUDA);

  // Create the same inputs
  auto input_opt =
      torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
  auto input_cpu =
      torch::tensor({1, 2, 3, 4, 5, 6}, input_opt).reshape({3, 1, 2});
  auto input_cuda = torch::tensor({1, 2, 3, 4, 5, 6}, input_opt)
                        .reshape({3, 1, 2})
                        .to(torch::kCUDA);

  // Call forward on both GRUs
  auto output_cpu = gru_cpu->forward(input_cpu);
  auto output_cuda = gru_cuda->forward(input_cuda);

  output_cpu = gru_output_to_device(output_cpu, torch::kCPU);

  // Assert that the output and state are equal on CPU and CUDA
  ASSERT_EQ(std::get<0>(output_cpu).dim(), std::get<0>(output_cuda).dim());
  for (int i = 0; i < std::get<0>(output_cpu).dim(); i++) {
    ASSERT_EQ(
        std::get<0>(output_cpu).size(i), std::get<0>(output_cuda).size(i));
  }
  for (int i = 0; i < std::get<0>(output_cpu).size(0); i++) {
    for (int j = 0; j < std::get<0>(output_cpu).size(1); j++) {
      for (int k = 0; k < std::get<0>(output_cpu).size(2); k++) {
        ASSERT_NEAR(
            std::get<0>(output_cpu)[i][j][k].item<float>(),
            std::get<0>(output_cuda)[i][j][k].item<float>(),
            1e-5);
      }
    }
  }
}

TEST_F(RNNTest, BidirectionalMultilayerLSTM_CPU_vs_CUDA) {
  // Create two LSTMs with the same options
  auto opt =
      LSTMOptions(2, 4).num_layers(3).batch_first(false).bidirectional(true);
  LSTM lstm_cpu{opt};
  LSTM lstm_cuda{opt};

  // Copy weights and biases from CPU LSTM to CUDA LSTM
  {
    at::NoGradGuard guard;
    for (const auto& param : lstm_cpu->named_parameters(/*recurse=*/false)) {
      lstm_cuda->named_parameters()[param.key()].copy_(
          lstm_cpu->named_parameters()[param.key()]);
    }
  }

  lstm_cpu->flatten_parameters();
  lstm_cuda->flatten_parameters();

  // Move LSTM to CUDA
  lstm_cuda->to(torch::kCUDA);

  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
  auto input_cpu =
      torch::tensor({1, 2, 3, 4, 5, 6}, options).reshape({3, 1, 2});
  auto input_cuda = torch::tensor({1, 2, 3, 4, 5, 6}, options)
                        .reshape({3, 1, 2})
                        .to(torch::kCUDA);

  // Call forward on both LSTMs
  auto output_cpu = lstm_cpu->forward(input_cpu);
  auto output_cuda = lstm_cuda->forward(input_cuda);

  output_cpu = lstm_output_to_device(output_cpu, torch::kCPU);

  // Assert that the output and state are equal on CPU and CUDA
  ASSERT_EQ(std::get<0>(output_cpu).dim(), std::get<0>(output_cuda).dim());
  for (int i = 0; i < std::get<0>(output_cpu).dim(); i++) {
    ASSERT_EQ(
        std::get<0>(output_cpu).size(i), std::get<0>(output_cuda).size(i));
  }
  for (int i = 0; i < std::get<0>(output_cpu).size(0); i++) {
    for (int j = 0; j < std::get<0>(output_cpu).size(1); j++) {
      for (int k = 0; k < std::get<0>(output_cpu).size(2); k++) {
        ASSERT_NEAR(
            std::get<0>(output_cpu)[i][j][k].item<float>(),
            std::get<0>(output_cuda)[i][j][k].item<float>(),
            1e-5);
      }
    }
  }
}

TEST_F(RNNTest, BidirectionalMultilayerLSTMProj_CPU_vs_CUDA) {
  // Create two LSTMs with the same options
  auto opt = LSTMOptions(2, 4)
                 .num_layers(3)
                 .batch_first(false)
                 .bidirectional(true)
                 .proj_size(2);
  LSTM lstm_cpu{opt};
  LSTM lstm_cuda{opt};

  // Copy weights and biases from CPU LSTM to CUDA LSTM
  {
    at::NoGradGuard guard;
    for (const auto& param : lstm_cpu->named_parameters(/*recurse=*/false)) {
      lstm_cuda->named_parameters()[param.key()].copy_(
          lstm_cpu->named_parameters()[param.key()]);
    }
  }

  lstm_cpu->flatten_parameters();
  lstm_cuda->flatten_parameters();

  // Move LSTM to CUDA
  lstm_cuda->to(torch::kCUDA);

  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
  auto input_cpu =
      torch::tensor({1, 2, 3, 4, 5, 6}, options).reshape({3, 1, 2});
  auto input_cuda = torch::tensor({1, 2, 3, 4, 5, 6}, options)
                        .reshape({3, 1, 2})
                        .to(torch::kCUDA);

  // Call forward on both LSTMs
  auto output_cpu = lstm_cpu->forward(input_cpu);
  auto output_cuda = lstm_cuda->forward(input_cuda);

  output_cpu = lstm_output_to_device(output_cpu, torch::kCPU);

  // Assert that the output and state are equal on CPU and CUDA
  ASSERT_EQ(std::get<0>(output_cpu).dim(), std::get<0>(output_cuda).dim());
  for (int i = 0; i < std::get<0>(output_cpu).dim(); i++) {
    ASSERT_EQ(
        std::get<0>(output_cpu).size(i), std::get<0>(output_cuda).size(i));
  }
  for (int i = 0; i < std::get<0>(output_cpu).size(0); i++) {
    for (int j = 0; j < std::get<0>(output_cpu).size(1); j++) {
      for (int k = 0; k < std::get<0>(output_cpu).size(2); k++) {
        ASSERT_NEAR(
            std::get<0>(output_cpu)[i][j][k].item<float>(),
            std::get<0>(output_cuda)[i][j][k].item<float>(),
            1e-5);
      }
    }
  }
}

TEST_F(RNNTest, UsePackedSequenceAsInput) {
  {
    torch::manual_seed(0);
    auto m = RNN(2, 3);
    torch::nn::utils::rnn::PackedSequence packed_input =
        torch::nn::utils::rnn::pack_sequence({torch::ones({3, 2})});
    auto rnn_output = m->forward_with_packed_input(packed_input);
    auto expected_output = torch::tensor(
        {{-0.0645, -0.7274, 0.4531},
         {-0.3970, -0.6950, 0.6009},
         {-0.3877, -0.7310, 0.6806}});
    ASSERT_TRUE(torch::allclose(
        std::get<0>(rnn_output).data(), expected_output, 1e-05, 2e-04));

    // Test passing optional argument to `RNN::forward_with_packed_input`
    rnn_output = m->forward_with_packed_input(packed_input, torch::Tensor());
    ASSERT_TRUE(torch::allclose(
        std::get<0>(rnn_output).data(), expected_output, 1e-05, 2e-04));
  }
  {
    torch::manual_seed(0);
    auto m = LSTM(2, 3);
    torch::nn::utils::rnn::PackedSequence packed_input =
        torch::nn::utils::rnn::pack_sequence({torch::ones({3, 2})});
    auto rnn_output = m->forward_with_packed_input(packed_input);
    auto expected_output = torch::tensor(
        {{-0.2693, -0.1240, 0.0744},
         {-0.3889, -0.1919, 0.1183},
         {-0.4425, -0.2314, 0.1386}});
    ASSERT_TRUE(torch::allclose(
        std::get<0>(rnn_output).data(), expected_output, 1e-05, 2e-04));

    // Test passing optional argument to `LSTM::forward_with_packed_input`
    rnn_output = m->forward_with_packed_input(packed_input, torch::nullopt);
    ASSERT_TRUE(torch::allclose(
        std::get<0>(rnn_output).data(), expected_output, 1e-05, 2e-04));
  }
  {
    torch::manual_seed(0);
    auto m = GRU(2, 3);
    torch::nn::utils::rnn::PackedSequence packed_input =
        torch::nn::utils::rnn::pack_sequence({torch::ones({3, 2})});
    auto rnn_output = m->forward_with_packed_input(packed_input);
    auto expected_output = torch::tensor(
        {{-0.1134, 0.0467, 0.2336},
         {-0.1189, 0.0502, 0.2960},
         {-0.1138, 0.0484, 0.3110}});
    ASSERT_TRUE(torch::allclose(
        std::get<0>(rnn_output).data(), expected_output, 1e-05, 2e-04));

    // Test passing optional argument to `GRU::forward_with_packed_input`
    rnn_output = m->forward_with_packed_input(packed_input, torch::Tensor());
    ASSERT_TRUE(torch::allclose(
        std::get<0>(rnn_output).data(), expected_output, 1e-05, 2e-04));
  }
}
