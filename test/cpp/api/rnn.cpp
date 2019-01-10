#include <catch.hpp>

#include <torch/torch.h>

using namespace torch;
using namespace torch::nn;

template <typename R, typename Func>
bool test_RNN_xor(Func&& model_maker, bool cuda = false) {
  auto nhid = 32;
  auto model = make(SimpleContainer());
  auto l1 = model->add(make(Linear(1, nhid)), "l1");
  auto rnn = model->add(model_maker(nhid), "rnn");
  auto lo = model->add(make(Linear(nhid, 1)), "lo");

  auto optim = Adam(model, 1e-2).make();

  auto forward_op = [&](Variable x) {
    auto T = x.size(0);
    auto B = x.size(1);
    x = x.view({T * B, 1});
    x = l1->forward({x})[0].view({T, B, nhid}).tanh_();
    x = rnn->forward({x})[0][T - 1];
    x = lo->forward({x})[0];
    return x;
  };

  if (cuda) {
    model->cuda();
  }

  float running_loss = 1;
  int epoch = 0;
  auto max_epoch = 1500;
  while (running_loss > 1e-2) {
    auto bs = 16U;
    auto nlen = 5U;
    auto inp =
        at::CPU(at::kFloat).rand({nlen, bs, 1}).round().toType(at::kFloat);
    auto lab = inp.sum(0);

    if (cuda) {
      inp = inp.toBackend(at::kCUDA);
      lab = lab.toBackend(at::kCUDA);
    }

    auto x = Var(inp);
    auto y = Var(lab, false);
    x = forward_op(x);
    Variable loss = at::mse_loss(x, y);

    optim->zero_grad();
    backward(loss);
    optim->step();

    running_loss = running_loss * 0.99 + loss.toCFloat() * 0.01;
    if (epoch > max_epoch) {
      return false;
    }
    epoch++;
  }
  return true;
};

void check_lstm_sizes(variable_list tup) {
  // Expect the LSTM to have 64 outputs and 3 layers, with an input of batch
  // 10 and 16 time steps (10 x 16 x n)

  auto out = tup[0];
  auto hids = tup[1];

  REQUIRE(out.ndimension() == 3);
  REQUIRE(out.size(0) == 10);
  REQUIRE(out.size(1) == 16);
  REQUIRE(out.size(2) == 64);

  REQUIRE(hids.ndimension() == 4);
  REQUIRE(hids.size(0) == 2); // (hx, cx)
  REQUIRE(hids.size(1) == 3); // layers
  REQUIRE(hids.size(2) == 16); // Batchsize
  REQUIRE(hids.size(3) == 64); // 64 hidden dims

  // Something is in the hiddens
  REQUIRE(hids.norm().toCFloat() > 0);
}

TEST_CASE("rnn") {
  SECTION("lstm") {
    SECTION("sizes") {
      auto model = make(LSTM(128, 64).nlayers(3).dropout(0.2));
      Variable x = Var(at::CPU(at::kFloat).randn({10, 16, 128}));
      auto tup = model->forward({x});
      auto y = x.mean();

      backward(y);
      check_lstm_sizes(tup);

      auto next = model->forward({x, tup[1]});

      check_lstm_sizes(next);

      Variable diff = next[1] - tup[1];

      // Hiddens changed
      REQUIRE(diff.data().abs().sum().toCFloat() > 1e-3);
    };

    SECTION("outputs") {
      // Make sure the outputs match pytorch outputs
      auto model = make(LSTM(2, 2));
      for (auto& v : model->parameters()) {
        float size = v.second.numel();
        auto p = static_cast<float*>(v.second.data().storage()->data());
        for (size_t i = 0; i < size; i++) {
          p[i] = i / size;
        }
      }

      Variable x = Var(at::CPU(at::kFloat).tensor({3, 4, 2}));
      float size = x.data().numel();
      auto p = static_cast<float*>(x.data().storage()->data());
      for (size_t i = 0; i < size; i++) {
        p[i] = (size - i) / size;
      }

      auto out = model->forward({x});
      REQUIRE(out[0].ndimension() == 3);
      REQUIRE(out[0].size(0) == 3);
      REQUIRE(out[0].size(1) == 4);
      REQUIRE(out[0].size(2) == 2);

      auto flat = out[0].data().view(3 * 4 * 2);
      float c_out[] = {0.4391, 0.5402, 0.4330, 0.5324, 0.4261, 0.5239,
                       0.4183, 0.5147, 0.6822, 0.8064, 0.6726, 0.7968,
                       0.6620, 0.7860, 0.6501, 0.7741, 0.7889, 0.9003,
                       0.7769, 0.8905, 0.7635, 0.8794, 0.7484, 0.8666};
      for (size_t i = 0; i < 3 * 4 * 2; i++) {
        REQUIRE(std::abs(flat[i].toCFloat() - c_out[i]) < 1e-3);
      }

      REQUIRE(out[1].ndimension() == 4); // (hx, cx) x layers x B x 2
      REQUIRE(out[1].size(0) == 2);
      REQUIRE(out[1].size(1) == 1);
      REQUIRE(out[1].size(2) == 4);
      REQUIRE(out[1].size(3) == 2);
      flat = out[1].data().view(16);
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
        REQUIRE(std::abs(flat[i].toCFloat() - h_out[i]) < 1e-3);
      }
    };
  }
  SECTION("integration") {
    SECTION("LSTM") {
      REQUIRE(test_RNN_xor<LSTM>(
          [](int s) { return make(LSTM(s, s).nlayers(2)); }));
    }

    SECTION("gru") {
      REQUIRE(
          test_RNN_xor<GRU>([](int s) { return make(GRU(s, s).nlayers(2)); }));
    }

    SECTION("rnn") {
      SECTION("relu") {
        REQUIRE(test_RNN_xor<RNN>(
            [](int s) { return make(RNN(s, s, RNN::Mode::Relu).nlayers(2)); }));
      }

      SECTION("tanh") {
        REQUIRE(test_RNN_xor<RNN>(
            [](int s) { return make(RNN(s, s, RNN::Mode::Tanh).nlayers(2)); }));
      }
    }
  }
}

TEST_CASE("rnn_cuda", "[cuda]") {
  SECTION("sizes") {
    auto model = make(LSTM(128, 64).nlayers(3).dropout(0.2));
    model->cuda();
    Variable x = Var(at::CUDA(at::kFloat).randn({10, 16, 128}));
    auto tup = model->forward({x});
    auto y = x.mean();

    backward(y);
    check_lstm_sizes(tup);

    auto next = model->forward({x, tup[1]});

    check_lstm_sizes(next);

    Variable diff = next[1] - tup[1];

    // Hiddens changed
    REQUIRE(diff.data().abs().sum().toCFloat() > 1e-3);
  };

  SECTION("lstm") {
    REQUIRE(test_RNN_xor<LSTM>(
        [](int s) { return make(LSTM(s, s).nlayers(2)); }, true));
  }

  SECTION("gru") {
    REQUIRE(test_RNN_xor<GRU>(
        [](int s) { return make(GRU(s, s).nlayers(2)); }, true));
  }

  SECTION("rnn") {
    SECTION("Relu") {
      REQUIRE(test_RNN_xor<RNN>(
          [](int s) { return make(RNN(s, s, RNN::Mode::Relu).nlayers(2)); },
          true));
    }

    SECTION("tanh") {
      REQUIRE(test_RNN_xor<RNN>(
          [](int s) { return make(RNN(s, s, RNN::Mode::Tanh).nlayers(2)); },
          true));
    }
  }
}
