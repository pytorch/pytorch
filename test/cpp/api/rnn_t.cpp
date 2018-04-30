#include "test.h"

template <typename R, typename Func>
bool test_RNN_xor(Func&& model_maker, bool cuda = false) {
  auto nhid = 32;
  auto model = SimpleContainer().make();
  auto l1 = model->add(Linear(1, nhid).make(), "l1");
  auto rnn = model->add(model_maker(nhid), "rnn");
  auto lo = model->add(Linear(nhid, 1).make(), "lo");

  auto optim = Adam(model, 1e-2).make();

  auto forward_op = [&](Variable x) {
    auto T = x.size(0);
    auto B = x.size(1);
    x = x.view({T * B, 1});
    x = l1->forward({x})[0].view({T, B, nhid}).tanh_();
    x = rnn->forward({x})[0][T-1];
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
    auto inp = at::CPU(at::kFloat).rand({nlen, bs, 1}).round().toType(at::kFloat);
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

CASE("RNN/LSTM/sizes") {
  auto model = LSTM(128, 64).nlayers(2).dropout(0.2).make();
  Variable x = Var(at::CPU(at::kFloat).randn({10, 16, 128}));
  auto tup = model->forward({x});
  auto y = x.mean();

  auto out = tup[0];
  auto hids = tup[1];

  backward(y);
  EXPECT(out.ndimension() == 3);
  EXPECT(out.size(0) == 10);
  EXPECT(out.size(1) == 16);
  EXPECT(out.size(2) == 64);

  EXPECT(hids.ndimension() == 4);
  EXPECT(hids.size(0) == 2);  // 2 layers
  EXPECT(hids.size(1) == 2);  // c and h
  EXPECT(hids.size(2) == 16); // Batch size of 16
  EXPECT(hids.size(3) == 64); // 64 hidden dims

  // Something is in the hiddens
  EXPECT(hids.norm().toCFloat() > 0);

  Variable diff = model->forward({x, hids})[1] - hids;

  // Hiddens changed
  EXPECT(diff.data().abs().sum().toCFloat() > 1e-3);
};

CASE("RNN/LSTM/outputs") {
  // Make sure the outputs match pytorch outputs
  auto model = LSTM(2, 2).make();
  for (auto& v : model->parameters()) {
    float size = v.second.numel();
    auto p = static_cast<float*>(v.second.data().storage()->data());
    for (size_t i = 0; i < size; i++) {
      p[i] = i/size;
    }
  }

  Variable x = Var(at::CPU(at::kFloat).tensor({3, 4, 2}));
  float size = x.data().numel();
  auto p = static_cast<float*>(x.data().storage()->data());
  for (size_t i = 0; i < size; i++) {
    p[i] = (size - i) / size;
  }

  auto out = model->forward({x});
  EXPECT(out[0].ndimension() == 3);
  EXPECT(out[0].size(0) == 3);
  EXPECT(out[0].size(1) == 4);
  EXPECT(out[0].size(2) == 2);

  auto flat = out[0].data().view(3*4*2);
  float c_out[] =  {0.4391, 0.5402, 0.4330, 0.5324, 0.4261, 0.5239, 0.4183, 
    0.5147, 0.6822, 0.8064, 0.6726, 0.7968, 0.6620, 0.7860, 0.6501, 0.7741, 
    0.7889, 0.9003, 0.7769, 0.8905, 0.7635, 0.8794, 0.7484, 0.8666};
  for (size_t i = 0; i < 3*4*2; i++) {
    EXPECT(std::abs(flat[i].toCFloat() - c_out[i]) < 1e-3);
  }

  EXPECT(out[1].ndimension() == 4); // T x (hx, cx) x B x 2
  EXPECT(out[1].size(0) == 1);
  EXPECT(out[1].size(1) == 2);
  EXPECT(out[1].size(2) == 4);
  EXPECT(out[1].size(3) == 2);
  flat = out[1].data().view(16);
  float h_out[] = {0.7889, 0.9003, 0.7769, 0.8905, 0.7635, 0.8794, 0.7484,
    0.8666, 1.1647, 1.6106, 1.1425, 1.5726, 1.1187, 1.5329, 1.0931, 1.4911};
  for (size_t i = 0; i < 16; i++) {
    EXPECT(std::abs(flat[i].toCFloat() - h_out[i]) < 1e-3);
  }
};

CASE("integration/RNN/LSTM") {
  EXPECT(test_RNN_xor<LSTM>([](int s) { return LSTM(s, s).nlayers(2).make(); }));
};

CASE("integration/RNN/GRU") {
  EXPECT(test_RNN_xor<GRU>([](int s) { return GRU(s, s).nlayers(2).make(); }));
};

CASE("integration/RNN/RNN/Relu") {
  EXPECT(test_RNN_xor<RNN>([](int s) { return RNN(s, s, RNN::Mode::Relu).nlayers(2).make(); }));
};

CASE("integration/RNN/RNN/Tanh") {
  EXPECT(test_RNN_xor<RNN>([](int s) { return RNN(s, s, RNN::Mode::Tanh).nlayers(2).make(); }));
};

CASE("integration/RNN/cuda/LSTM") {
  CUDA_GUARD;
  EXPECT(test_RNN_xor<LSTM>([](int s) { return LSTM(s, s).nlayers(2).make(); }, true));
};

CASE("integration/RNN/cuda/GRU") {
  CUDA_GUARD;
  EXPECT(test_RNN_xor<GRU>([](int s) { return GRU(s, s).nlayers(2).make(); }, true));
};

CASE("integration/RNN/cuda/RNN/Relu") {
  CUDA_GUARD;
  EXPECT(test_RNN_xor<RNN>([](int s) { return RNN(s, s, RNN::Mode::Relu).nlayers(2).make(); }, true));
};

CASE("integration/RNN/cuda/RNN/Tanh") {
  CUDA_GUARD;
  EXPECT(test_RNN_xor<RNN>([](int s) { return RNN(s, s, RNN::Mode::Tanh).nlayers(2).make(); }, true));
};
