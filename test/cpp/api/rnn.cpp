#include "catch_utils.hpp"

#include <torch/nn/modules/linear.h>
#include <torch/nn/modules/rnn.h>
#include <torch/optim/adam.h>
#include <torch/tensor.h>
#include <torch/utils.h>

#include <test/cpp/api/util.h>

using Catch::StartsWith;

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
        torch::rand({nlen, bs, 1}, backend).round().toType(torch::kFloat32);
    auto labels = inputs.sum(0).detach();
    inputs.set_requires_grad(true);

    auto outputs = forward_op(inputs);
    torch::Tensor loss = torch::mse_loss(outputs, labels);

    optimizer.zero_grad();
    loss.backward();
    optimizer.step();

    running_loss = running_loss * 0.99 + loss.toCFloat() * 0.01;
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

  CATCH_REQUIRE(output.output.ndimension() == 3);
  CATCH_REQUIRE(output.output.size(0) == 10);
  CATCH_REQUIRE(output.output.size(1) == 16);
  CATCH_REQUIRE(output.output.size(2) == 64);

  CATCH_REQUIRE(output.state.ndimension() == 4);
  CATCH_REQUIRE(output.state.size(0) == 2); // (hx, cx)
  CATCH_REQUIRE(output.state.size(1) == 3); // layers
  CATCH_REQUIRE(output.state.size(2) == 16); // Batchsize
  CATCH_REQUIRE(output.state.size(3) == 64); // 64 hidden dims

  // Something is in the hiddens
  CATCH_REQUIRE(output.state.norm().toCFloat() > 0);
}

CATCH_TEST_CASE("RNN/CheckOutputSizes") {
  torch::manual_seed(0);
  LSTM model(LSTMOptions(128, 64).layers(3).dropout(0.2));
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
  CATCH_REQUIRE(diff.abs().sum().toCFloat() > 1e-3);
}

CATCH_TEST_CASE("RNN/CheckOutputValuesMatchPyTorch") {
  torch::manual_seed(0);
  // Make sure the outputs match pytorch outputs
  LSTM model(2, 2);
  for (auto& v : model->parameters()) {
    float size = v->numel();
    auto p = static_cast<float*>(v->storage().data());
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
  CATCH_REQUIRE(out.output.ndimension() == 3);
  CATCH_REQUIRE(out.output.size(0) == 3);
  CATCH_REQUIRE(out.output.size(1) == 4);
  CATCH_REQUIRE(out.output.size(2) == 2);

  auto flat = out.output.view(3 * 4 * 2);
  float c_out[] = {0.4391, 0.5402, 0.4330, 0.5324, 0.4261, 0.5239,
                   0.4183, 0.5147, 0.6822, 0.8064, 0.6726, 0.7968,
                   0.6620, 0.7860, 0.6501, 0.7741, 0.7889, 0.9003,
                   0.7769, 0.8905, 0.7635, 0.8794, 0.7484, 0.8666};
  for (size_t i = 0; i < 3 * 4 * 2; i++) {
    CATCH_REQUIRE(std::abs(flat[i].toCFloat() - c_out[i]) < 1e-3);
  }

  CATCH_REQUIRE(out.state.ndimension() == 4); // (hx, cx) x layers x B x 2
  CATCH_REQUIRE(out.state.size(0) == 2);
  CATCH_REQUIRE(out.state.size(1) == 1);
  CATCH_REQUIRE(out.state.size(2) == 4);
  CATCH_REQUIRE(out.state.size(3) == 2);
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
    CATCH_REQUIRE(std::abs(flat[i].toCFloat() - h_out[i]) < 1e-3);
  }
}

CATCH_TEST_CASE("RNN/integration/LSTM") {
  CATCH_REQUIRE(test_RNN_xor<LSTM>(
      [](int s) { return LSTM(LSTMOptions(s, s).layers(2)); }));
}

CATCH_TEST_CASE("RNN/integration/GRU") {
  CATCH_REQUIRE(
      test_RNN_xor<GRU>([](int s) { return GRU(GRUOptions(s, s).layers(2)); }));
}

CATCH_TEST_CASE("RNN/integration/RNN") {
  CATCH_SECTION("relu") {
    CATCH_REQUIRE(test_RNN_xor<RNN>(
        [](int s) { return RNN(RNNOptions(s, s).relu().layers(2)); }));
  }
  CATCH_SECTION("tanh") {
    CATCH_REQUIRE(test_RNN_xor<RNN>(
        [](int s) { return RNN(RNNOptions(s, s).tanh().layers(2)); }));
  }
}

CATCH_TEST_CASE("rnn_cuda", "[cuda]") {
  CATCH_SECTION("sizes") {
    torch::manual_seed(0);
    LSTM model(LSTMOptions(128, 64).layers(3).dropout(0.2));
    model->to(torch::kCUDA);
    auto x = torch::randn(
        {10, 16, 128}, torch::requires_grad().device(torch::kCUDA));
    auto output = model->forward(x);
    auto y = x.mean();

    y.backward();
    check_lstm_sizes(output);

    auto next = model->forward(x, output.state);

    check_lstm_sizes(next);

    torch::Tensor diff = next.state - output.state;

    // Hiddens changed
    CATCH_REQUIRE(diff.abs().sum().toCFloat() > 1e-3);
  }

  CATCH_SECTION("lstm") {
    CATCH_REQUIRE(test_RNN_xor<LSTM>(
        [](int s) { return LSTM(LSTMOptions(s, s).layers(2)); }, true));
  }

  CATCH_SECTION("gru") {
    CATCH_REQUIRE(test_RNN_xor<GRU>(
        [](int s) { return GRU(GRUOptions(s, s).layers(2)); }, true));
  }

  CATCH_SECTION("rnn") {
    CATCH_SECTION("relu") {
      CATCH_REQUIRE(test_RNN_xor<RNN>(
          [](int s) { return RNN(RNNOptions(s, s).relu().layers(2)); }, true));
    }
    CATCH_SECTION("tanh") {
      CATCH_REQUIRE(test_RNN_xor<RNN>(
          [](int s) { return RNN(RNNOptions(s, s).tanh().layers(2)); }, true));
    }
  }
}
