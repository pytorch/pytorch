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
  auto rnn = model->add(model_maker(nhid), "rnn");
  auto lo = model->add(Linear(nhid, 1), "lo");

  torch::optim::Adam optimizer(model->parameters(), 1e-2);
  auto forward_op = [&](torch::Tensor x) {
    auto T = x.size(0);
    auto B = x.size(1);
    x = x.view({T * B, 1});
    x = l1->forward(x).view({T, B, nhid}).tanh_();
    x = rnn->forward(x).output[T - 1];
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

    running_loss = running_loss * 0.99 + loss.item<float>() * 0.01;
    if (epoch > max_epoch) {
      return false;
    }
    epoch++;
  }
  return true;
};

void check_lstm_sizes(RNNOutput output) {
  // Expect the LSTM to have 64 outputs and 3 layers, with an input of batch
  // 10 and 16 time steps (10 x 16 x n)

  ASSERT_EQ(output.output.ndimension(), 3);
  ASSERT_EQ(output.output.size(0), 10);
  ASSERT_EQ(output.output.size(1), 16);
  ASSERT_EQ(output.output.size(2), 64);

  ASSERT_EQ(output.state.ndimension(), 4);
  ASSERT_EQ(output.state.size(0), 2); // (hx, cx)
  ASSERT_EQ(output.state.size(1), 3); // layers
  ASSERT_EQ(output.state.size(2), 16); // Batchsize
  ASSERT_EQ(output.state.size(3), 64); // 64 hidden dims

  // Something is in the hiddens
  ASSERT_GT(output.state.norm().item<float>(), 0);
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

  auto next = model->forward(x, output.state);

  check_lstm_sizes(next);

  torch::Tensor diff = next.state - output.state;

  // Hiddens changed
  ASSERT_GT(diff.abs().sum().item<float>(), 1e-3);
}

TEST_F(RNNTest, CheckOutputValuesMatchPyTorch) {
  torch::manual_seed(0);
  // Make sure the outputs match pytorch outputs
  LSTM model(2, 2);
  for (auto& v : model->parameters()) {
    float size = v.numel();
    auto p = static_cast<float*>(v.storage().data());
    for (size_t i = 0; i < size; i++) {
      p[i] = i / size;
    }
  }

  auto x = torch::empty({3, 4, 2}, torch::requires_grad());
  float size = x.numel();
  auto p = static_cast<float*>(x.storage().data());
  for (size_t i = 0; i < size; i++) {
    p[i] = (size - i) / size;
  }

  auto out = model->forward(x);
  ASSERT_EQ(out.output.ndimension(), 3);
  ASSERT_EQ(out.output.size(0), 3);
  ASSERT_EQ(out.output.size(1), 4);
  ASSERT_EQ(out.output.size(2), 2);

  auto flat = out.output.view(3 * 4 * 2);
  float c_out[] = {0.4391, 0.5402, 0.4330, 0.5324, 0.4261, 0.5239,
                   0.4183, 0.5147, 0.6822, 0.8064, 0.6726, 0.7968,
                   0.6620, 0.7860, 0.6501, 0.7741, 0.7889, 0.9003,
                   0.7769, 0.8905, 0.7635, 0.8794, 0.7484, 0.8666};
  for (size_t i = 0; i < 3 * 4 * 2; i++) {
    ASSERT_LT(std::abs(flat[i].item<float>() - c_out[i]), 1e-3);
  }

  ASSERT_EQ(out.state.ndimension(), 4); // (hx, cx) x layers x B x 2
  ASSERT_EQ(out.state.size(0), 2);
  ASSERT_EQ(out.state.size(1), 1);
  ASSERT_EQ(out.state.size(2), 4);
  ASSERT_EQ(out.state.size(3), 2);
  flat = out.state.view(16);
  float h_out[] = {0.7889,
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

TEST_F(RNNTest, EndToEndGRU) {
  ASSERT_TRUE(
      test_RNN_xor<GRU>([](int s) { return GRU(GRUOptions(s, s).num_layers(2)); }));
}

TEST_F(RNNTest, EndToEndRNNRelu) {
  ASSERT_TRUE(test_RNN_xor<RNN>(
      [](int s) { return RNN(RNNOptions(s, s).nonlinearity(torch::kReLU).num_layers(2)); }));
}

TEST_F(RNNTest, EndToEndRNNTanh) {
  ASSERT_TRUE(test_RNN_xor<RNN>(
      [](int s) { return RNN(RNNOptions(s, s).nonlinearity(torch::kTanh).num_layers(2)); }));
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

  auto next = model->forward(x, output.state);

  check_lstm_sizes(next);

  torch::Tensor diff = next.state - output.state;

  // Hiddens changed
  ASSERT_GT(diff.abs().sum().item<float>(), 1e-3);
}

TEST_F(RNNTest, EndToEndLSTM_CUDA) {
  ASSERT_TRUE(test_RNN_xor<LSTM>(
      [](int s) { return LSTM(LSTMOptions(s, s).num_layers(2)); }, true));
}

TEST_F(RNNTest, EndToEndGRU_CUDA) {
  ASSERT_TRUE(test_RNN_xor<GRU>(
      [](int s) { return GRU(GRUOptions(s, s).num_layers(2)); }, true));
}

TEST_F(RNNTest, EndToEndRNNRelu_CUDA) {
  ASSERT_TRUE(test_RNN_xor<RNN>(
      [](int s) { return RNN(RNNOptions(s, s).nonlinearity(torch::kReLU).num_layers(2)); }, true));
}
TEST_F(RNNTest, EndToEndRNNTanh_CUDA) {
  ASSERT_TRUE(test_RNN_xor<RNN>(
      [](int s) { return RNN(RNNOptions(s, s).nonlinearity(torch::kTanh).num_layers(2)); }, true));
}

TEST_F(RNNTest, PrettyPrintRNNs) {
  ASSERT_EQ(
      c10::str(LSTM(LSTMOptions(128, 64).num_layers(3).dropout(0.2))),
      "torch::nn::LSTM(128, 64, num_layers=3, dropout=0.2)");
  ASSERT_EQ(
      c10::str(GRU(GRUOptions(128, 64).num_layers(3).dropout(0.5))),
      "torch::nn::GRU(128, 64, num_layers=3, dropout=0.5)");
  ASSERT_EQ(
      c10::str(RNN(RNNOptions(128, 64).num_layers(3).dropout(0.2).nonlinearity(torch::kTanh))),
      "torch::nn::RNN(128, 64, num_layers=3, dropout=0.2)");
}

// This test assures that flatten_parameters does not crash,
// when bidirectional is set to true
// https://github.com/pytorch/pytorch/issues/19545
TEST_F(RNNTest, BidirectionalFlattenParameters) {
  GRU gru(GRUOptions(100, 256).num_layers(2).bidirectional(true));
  gru->flatten_parameters();
}

template <typename Impl>
void copyParameters(torch::nn::ModuleHolder<Impl>& target, size_t t_i,
                    const torch::nn::ModuleHolder<Impl>& source, size_t s_i) {
  at::NoGradGuard guard;
  target->w_ih[t_i].copy_(source->w_ih[s_i]);
  target->w_hh[t_i].copy_(source->w_hh[s_i]);
  target->b_ih[t_i].copy_(source->b_ih[s_i]);
  target->b_hh[t_i].copy_(source->b_hh[s_i]);
}

// This test is a port of python code introduced here:
// https://towardsdatascience.com/understanding-bidirectional-rnn-in-pytorch-5bd25a5dd66
// Reverse forward of bidrectional GRU should act
// as regular forward of unidirectional GRU
void BidirectionalGRUReverseForward(bool cuda) {
  auto opt = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false)
                                   .device(cuda ? torch::kCUDA : torch::kCPU);
  auto input = torch::tensor({1, 2, 3, 4, 5}, opt).reshape({5, 1, 1});
  auto input_reversed = torch::tensor({5, 4, 3, 2, 1}, opt).reshape({5, 1, 1});

  auto gru_options = GRUOptions(1, 1).num_layers(1).batch_first(false);
  GRU bi_grus {gru_options.bidirectional(true)};
  GRU reverse_gru {gru_options.bidirectional(false)};

  if (cuda) {
    bi_grus->to(torch::kCUDA);
    reverse_gru->to(torch::kCUDA);
  }

  // Now make sure the weights of the reverse gru layer match
  // ones of the (reversed) bidirectional's:
  copyParameters(reverse_gru, 0, bi_grus, 1);

  auto bi_output = bi_grus->forward(input);
  auto reverse_output = reverse_gru->forward(input_reversed);

  if (cuda) {
    bi_output.output = bi_output.output.to(torch::kCPU);
    bi_output.state = bi_output.state.to(torch::kCPU);
    reverse_output.output = reverse_output.output.to(torch::kCPU);
    reverse_output.state = reverse_output.state.to(torch::kCPU);
  }

  ASSERT_EQ(bi_output.output.size(0), reverse_output.output.size(0));
  auto size = bi_output.output.size(0);
  for (int i = 0; i < size; i++) {
    ASSERT_EQ(bi_output.output[i][0][1].item<float>(),
              reverse_output.output[size - 1 - i][0][0].item<float>());
  }
  // The hidden states of the reversed GRUs sits
  // in the odd indices in the first dimension.
  ASSERT_EQ(bi_output.state[1][0][0].item<float>(),
            reverse_output.state[0][0][0].item<float>());
}

TEST_F(RNNTest, BidirectionalGRUReverseForward) {
  BidirectionalGRUReverseForward(false);
}

TEST_F(RNNTest, BidirectionalGRUReverseForward_CUDA) {
  BidirectionalGRUReverseForward(true);
}

// Reverse forward of bidrectional LSTM should act
// as regular forward of unidirectional LSTM
void BidirectionalLSTMReverseForwardTest(bool cuda) {
  auto opt = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false)
                                   .device(cuda ? torch::kCUDA : torch::kCPU);
  auto input = torch::tensor({1, 2, 3, 4, 5}, opt).reshape({5, 1, 1});
  auto input_reversed = torch::tensor({5, 4, 3, 2, 1}, opt).reshape({5, 1, 1});

  auto lstm_opt = GRUOptions(1, 1).num_layers(1).batch_first(false);

  LSTM bi_lstm {lstm_opt.bidirectional(true)};
  LSTM reverse_lstm {lstm_opt.bidirectional(false)};

  if (cuda) {
    bi_lstm->to(torch::kCUDA);
    reverse_lstm->to(torch::kCUDA);
  }

  // Now make sure the weights of the reverse lstm layer match
  // ones of the (reversed) bidirectional's:
  copyParameters(reverse_lstm, 0, bi_lstm, 1);

  auto bi_output = bi_lstm->forward(input);
  auto reverse_output = reverse_lstm->forward(input_reversed);

  if (cuda) {
    bi_output.output = bi_output.output.to(torch::kCPU);
    bi_output.state = bi_output.state.to(torch::kCPU);
    reverse_output.output = reverse_output.output.to(torch::kCPU);
    reverse_output.state = reverse_output.state.to(torch::kCPU);
  }

  ASSERT_EQ(bi_output.output.size(0), reverse_output.output.size(0));
  auto size = bi_output.output.size(0);
  for (int i = 0; i < size; i++) {
    ASSERT_EQ(bi_output.output[i][0][1].item<float>(),
              reverse_output.output[size - 1 - i][0][0].item<float>());
  }
  // The hidden states of the reversed LSTM sits
  // in the odd indices in the first dimension.
  ASSERT_EQ(bi_output.state[0][1][0][0].item<float>(),
            reverse_output.state[0][0][0][0].item<float>());
  ASSERT_EQ(bi_output.state[1][1][0][0].item<float>(),
            reverse_output.state[1][0][0][0].item<float>());
}

TEST_F(RNNTest, BidirectionalLSTMReverseForward) {
  BidirectionalLSTMReverseForwardTest(false);
}

TEST_F(RNNTest, BidirectionalLSTMReverseForward_CUDA) {
  BidirectionalLSTMReverseForwardTest(true);
}

TEST_F(RNNTest, BidirectionalMultilayerGRU_CPU_vs_CUDA) {
  // Create two GRUs with the same options
  auto opt = GRUOptions(2, 4).num_layers(3).batch_first(false).bidirectional(true);
  GRU gru_cpu {opt};
  GRU gru_cuda {opt};

  // Copy weights and biases from CPU GRU to CUDA GRU
  {
    at::NoGradGuard guard;
    const auto num_directions = gru_cpu->options.bidirectional() ? 2 : 1;
    for (int64_t layer = 0; layer < gru_cpu->options.num_layers(); layer++) {
      for (auto direction = 0; direction < num_directions; direction++) {
        const auto layer_idx = (layer * num_directions) + direction;
        copyParameters(gru_cuda, layer_idx, gru_cpu, layer_idx);
      }
    }
  }

  gru_cpu->flatten_parameters();
  gru_cuda->flatten_parameters();

  // Move GRU to CUDA
  gru_cuda->to(torch::kCUDA);

  // Create the same inputs
  auto input_opt = torch::TensorOptions()
                    .dtype(torch::kFloat32).requires_grad(false);
  auto input_cpu = torch::tensor({1, 2, 3, 4, 5, 6}, input_opt)
                    .reshape({3, 1, 2});
  auto input_cuda = torch::tensor({1, 2, 3, 4, 5, 6}, input_opt)
                    .reshape({3, 1, 2}).to(torch::kCUDA);

  // Call forward on both GRUs
  auto output_cpu = gru_cpu->forward(input_cpu);
  auto output_cuda = gru_cuda->forward(input_cuda);

  output_cpu.output = output_cpu.output.to(torch::kCPU);
  output_cpu.state = output_cpu.state.to(torch::kCPU);

  // Assert that the output and state are equal on CPU and CUDA
  ASSERT_EQ(output_cpu.output.dim(), output_cuda.output.dim());
  for (int i = 0; i < output_cpu.output.dim(); i++) {
    ASSERT_EQ(output_cpu.output.size(i), output_cuda.output.size(i));
  }
  for (int i = 0; i < output_cpu.output.size(0); i++) {
    for (int j = 0; j < output_cpu.output.size(1); j++) {
      for (int k = 0; k < output_cpu.output.size(2); k++) {
        ASSERT_NEAR(
          output_cpu.output[i][j][k].item<float>(),
          output_cuda.output[i][j][k].item<float>(), 1e-5);
      }
    }
  }
}

TEST_F(RNNTest, BidirectionalMultilayerLSTM_CPU_vs_CUDA) {
  // Create two LSTMs with the same options
  auto opt = LSTMOptions(2, 4).num_layers(3).batch_first(false).bidirectional(true);
  LSTM lstm_cpu {opt};
  LSTM lstm_cuda {opt};

  // Copy weights and biases from CPU LSTM to CUDA LSTM
  {
    at::NoGradGuard guard;
    const auto num_directions = lstm_cpu->options.bidirectional() ? 2 : 1;
    for (int64_t layer = 0; layer < lstm_cpu->options.num_layers(); layer++) {
      for (auto direction = 0; direction < num_directions; direction++) {
        const auto layer_idx = (layer * num_directions) + direction;
        copyParameters(lstm_cuda, layer_idx, lstm_cpu, layer_idx);
      }
    }
  }

  lstm_cpu->flatten_parameters();
  lstm_cuda->flatten_parameters();

  // Move LSTM to CUDA
  lstm_cuda->to(torch::kCUDA);

  auto options = torch::TensorOptions()
                  .dtype(torch::kFloat32).requires_grad(false);
  auto input_cpu = torch::tensor({1, 2, 3, 4, 5, 6}, options)
                  .reshape({3, 1, 2});
  auto input_cuda = torch::tensor({1, 2, 3, 4, 5, 6}, options)
                  .reshape({3, 1, 2}).to(torch::kCUDA);

  // Call forward on both LSTMs
  auto output_cpu = lstm_cpu->forward(input_cpu);
  auto output_cuda = lstm_cuda->forward(input_cuda);

  output_cpu.output = output_cpu.output.to(torch::kCPU);
  output_cpu.state = output_cpu.state.to(torch::kCPU);

  // Assert that the output and state are equal on CPU and CUDA
  ASSERT_EQ(output_cpu.output.dim(), output_cuda.output.dim());
  for (int i = 0; i < output_cpu.output.dim(); i++) {
    ASSERT_EQ(output_cpu.output.size(i), output_cuda.output.size(i));
  }
  for (int i = 0; i < output_cpu.output.size(0); i++) {
    for (int j = 0; j < output_cpu.output.size(1); j++) {
      for (int k = 0; k < output_cpu.output.size(2); k++) {
        ASSERT_NEAR(
          output_cpu.output[i][j][k].item<float>(),
          output_cuda.output[i][j][k].item<float>(), 1e-5);
      }
    }
  }
}
