#include <catch.hpp>

#include <torch/torch.h>

using namespace torch;
using namespace torch::nn;

#include <iostream>

class CartPole {
  // Translated from openai/gym's cartpole.py
 public:
  double gravity = 9.8;
  double masscart = 1.0;
  double masspole = 0.1;
  double total_mass = (masspole + masscart);
  double length = 0.5; // actually half the pole's length;
  double polemass_length = (masspole * length);
  double force_mag = 10.0;
  double tau = 0.02; // seconds between state updates;

  // Angle at which to fail the episode
  double theta_threshold_radians = 12 * 2 * M_PI / 360;
  double x_threshold = 2.4;
  int steps_beyond_done = -1;

  at::Tensor state;
  double reward;
  bool done;
  int step_ = 0;

  at::Tensor getState() {
    return state;
  }

  double getReward() {
    return reward;
  }

  double isDone() {
    return done;
  }

  void reset() {
    state = at::CPU(at::kFloat).tensor({4}).uniform_(-0.05, 0.05);
    steps_beyond_done = -1;
    step_ = 0;
  }

  CartPole() {
    reset();
  }

  void step(int action) {
    auto x = state[0].toCFloat();
    auto x_dot = state[1].toCFloat();
    auto theta = state[2].toCFloat();
    auto theta_dot = state[3].toCFloat();

    auto force = (action == 1) ? force_mag : -force_mag;
    auto costheta = std::cos(theta);
    auto sintheta = std::sin(theta);
    auto temp = (force + polemass_length * theta_dot * theta_dot * sintheta) /
        total_mass;
    auto thetaacc = (gravity * sintheta - costheta * temp) /
        (length * (4.0 / 3.0 - masspole * costheta * costheta / total_mass));
    auto xacc = temp - polemass_length * thetaacc * costheta / total_mass;

    x = x + tau * x_dot;
    x_dot = x_dot + tau * xacc;
    theta = theta + tau * theta_dot;
    theta_dot = theta_dot + tau * thetaacc;
    state[0] = x;
    state[1] = x_dot;
    state[2] = theta;
    state[3] = theta_dot;
    done = x < -x_threshold || x > x_threshold ||
        theta < -theta_threshold_radians || theta > theta_threshold_radians ||
        step_ > 200;

    if (!done) {
      reward = 1.0;
    } else if (steps_beyond_done == -1) {
      // Pole just fell!
      steps_beyond_done = 0;
      reward = 0;
    } else {
      if (steps_beyond_done == 0) {
        assert(false); // Can't do this
      }
    }
    step_++;
  }
};

template <typename M, typename F, typename O>
bool test_mnist(
    uint32_t batch_size,
    uint32_t num_epochs,
    bool useGPU,
    M&& model,
    F&& forward_op,
    O&& optim) {
  std::cout << "Training MNIST for " << num_epochs
            << " epochs, rest your eyes for a bit!\n";
  struct MNIST_Reader {
    FILE* fp_;

    MNIST_Reader(const char* path) {
      fp_ = fopen(path, "rb");
      if (!fp_)
        throw std::runtime_error("failed to open file");
    }

    ~MNIST_Reader() {
      if (fp_)
        fclose(fp_);
    }

    int32_t read_int() {
      uint8_t buf[4];
      if (fread(buf, sizeof(buf), 1, fp_) != 1)
        throw std::runtime_error("failed to read an integer");
      return int32_t(buf[0] << 24 | buf[1] << 16 | buf[2] << 8 | buf[3]);
    }

    uint8_t read_byte() {
      uint8_t i;
      if (fread(&i, sizeof(i), 1, fp_) != 1)
        throw std::runtime_error("failed to read an byte");
      return i;
    }
  };

  auto readData = [&](std::string fn) {
    MNIST_Reader rd(fn.c_str());

    /* int image_magic = */ rd.read_int();
    int image_count = rd.read_int();
    int image_rows = rd.read_int();
    int image_cols = rd.read_int();

    auto data =
        at::CPU(at::kFloat).tensor({image_count, 1, image_rows, image_cols});
    auto a_data = data.accessor<float, 4>();

    for (int c = 0; c < image_count; c++) {
      for (int i = 0; i < image_rows; i++) {
        for (int j = 0; j < image_cols; j++) {
          a_data[c][0][i][j] = float(rd.read_byte()) / 255;
        }
      }
    }

    return data.toBackend(useGPU ? at::kCUDA : at::kCPU);
  };

  auto readLabels = [&](std::string fn) {
    MNIST_Reader rd(fn.c_str());
    /* int label_magic = */ rd.read_int();
    int label_count = rd.read_int();

    auto data = at::CPU(at::kLong).tensor({label_count});
    auto a_data = data.accessor<int64_t, 1>();

    for (int i = 0; i < label_count; ++i) {
      a_data[i] = long(rd.read_byte());
    }
    return data.toBackend(useGPU ? at::kCUDA : at::kCPU);
  };

  auto trdata = readData("test/cpp/api/mnist/train-images-idx3-ubyte");
  auto trlabel = readLabels("test/cpp/api/mnist/train-labels-idx1-ubyte");
  auto tedata = readData("test/cpp/api/mnist/t10k-images-idx3-ubyte");
  auto telabel = readLabels("test/cpp/api/mnist/t10k-labels-idx1-ubyte");

  if (useGPU) {
    model->cuda();
  }

  for (auto epoch = 0U; epoch < num_epochs; epoch++) {
    auto shuffled_inds = std::vector<int>(trdata.size(0));
    for (int i = 0; i < trdata.size(0); i++) {
      shuffled_inds[i] = i;
    }
    std::random_shuffle(shuffled_inds.begin(), shuffled_inds.end());

    auto inp = (useGPU ? at::CUDA : at::CPU)(at::kFloat)
                   .tensor({batch_size, 1, trdata.size(2), trdata.size(3)});
    auto lab = (useGPU ? at::CUDA : at::CPU)(at::kLong).tensor({batch_size});
    for (auto p = 0U; p < shuffled_inds.size() - batch_size; p++) {
      inp[p % batch_size] = trdata[shuffled_inds[p]];
      lab[p % batch_size] = trlabel[shuffled_inds[p]];

      if (p % batch_size != batch_size - 1)
        continue;
      Variable x = forward_op(Var(inp));
      Variable y = Var(lab, false);
      Variable loss = at::nll_loss(x, y);

      optim->zero_grad();
      backward(loss);
      optim->step();
    }
  }

  no_grad_guard guard;
  auto result = std::get<1>(forward_op(Var(tedata, false)).max(1));
  Variable correct = (result == Var(telabel)).toType(at::kFloat);
  std::cout << "Num correct: " << correct.data().sum().toCFloat() << " out of "
            << telabel.size(0) << std::endl;
  return correct.data().sum().toCFloat() > telabel.size(0) * 0.8;
};

TEST_CASE("integration") {
  SECTION("cartpole") {
    std::cerr
        << "Training episodic policy gradient with a critic for up to 3000"
           " episodes, rest your eyes for a bit!\n";
    auto model = make(SimpleContainer());
    auto linear = model->add(make(Linear(4, 128)), "linear");
    auto policyHead = model->add(make(Linear(128, 2)), "policy");
    auto valueHead = model->add(make(Linear(128, 1)), "action");
    auto optim = Adam(model, 1e-3).make();

    std::vector<Variable> saved_log_probs;
    std::vector<Variable> saved_values;
    std::vector<float> rewards;

    auto forward = [&](variable_list inp) {
      auto x = linear->forward(inp)[0].clamp_min(0);
      Variable actions = policyHead->forward({x})[0];
      Variable value = valueHead->forward({x})[0];
      return std::make_tuple(at::softmax(actions, -1), value);
    };

    auto selectAction = [&](at::Tensor state) {
      // Only work on single state right now, change index to gather for batch
      auto out = forward({Var(state, false)});
      auto probs = Variable(std::get<0>(out));
      auto value = Variable(std::get<1>(out));
      auto action = probs.data().multinomial(1)[0].toCInt();
      // Compute the log prob of a multinomial distribution.
      // This should probably be actually implemented in autogradpp...
      auto p = probs / probs.sum(-1, true);
      auto log_prob = p[action].log();
      saved_log_probs.push_back(log_prob);
      saved_values.push_back(value);
      return action;
    };

    auto finishEpisode = [&]() {
      auto R = 0.;
      for (int i = rewards.size() - 1; i >= 0; i--) {
        R = rewards[i] + 0.99 * R;
        rewards[i] = R;
      }
      auto r_t =
          at::CPU(at::kFloat)
              .tensorFromBlob(
                  rewards.data(), {static_cast<int64_t>(rewards.size())});
      r_t = (r_t - r_t.mean()) / (r_t.std() + 1e-5);

      std::vector<at::Tensor> policy_loss;
      std::vector<at::Tensor> value_loss;
      for (auto i = 0U; i < saved_log_probs.size(); i++) {
        auto r = rewards[i] - saved_values[i].toCFloat();
        policy_loss.push_back(-r * saved_log_probs[i]);
        value_loss.push_back(at::smooth_l1_loss(
            saved_values[i],
            Var(at::CPU(at::kFloat).scalarTensor(at::Scalar(rewards[i])),
                false)));
      }
      auto loss = at::stack(policy_loss).sum() + at::stack(value_loss).sum();

      optim->zero_grad();
      backward(loss);
      optim->step();

      rewards.clear();
      saved_log_probs.clear();
      saved_values.clear();
    };

    auto env = CartPole();
    double running_reward = 10.0;
    for (auto episode = 0;; episode++) {
      env.reset();
      auto state = env.getState();
      int t = 0;
      for (; t < 10000; t++) {
        auto action = selectAction(state);
        env.step(action);
        state = env.getState();
        auto reward = env.getReward();
        auto done = env.isDone();

        rewards.push_back(reward);
        if (done)
          break;
      }

      running_reward = running_reward * 0.99 + t * 0.01;
      finishEpisode();
      /*
      if (episode % 10 == 0) {
        printf("Episode %i\tLast length: %5d\tAverage length: %.2f\n",
                episode, t, running_reward);
      }
      */
      if (running_reward > 150)
        break;
      REQUIRE(episode < 3000);
    }
  }
}

TEST_CASE("integration_cuda", "[cuda]") {
  SECTION("mnist") {
    auto model = make(SimpleContainer());
    auto conv1 = model->add(make(Conv2d(1, 10, 5)), "conv1");
    auto conv2 = model->add(make(Conv2d(10, 20, 5)), "conv2");
    auto drop = make(Dropout(0.3));
    auto drop2d = make(Dropout2d(0.3));
    auto linear1 = model->add(make(Linear(320, 50)), "linear1");
    auto linear2 = model->add(make(Linear(50, 10)), "linear2");

    auto forward = [&](Variable x) {
      x = std::get<0>(at::max_pool2d(conv1->forward({x})[0], {2, 2}))
              .clamp_min(0);
      x = conv2->forward({x})[0];
      x = drop2d->forward({x})[0];
      x = std::get<0>(at::max_pool2d(x, {2, 2})).clamp_min(0);

      x = x.view({-1, 320});
      x = linear1->forward({x})[0].clamp_min(0);
      x = drop->forward({x})[0];
      x = linear2->forward({x})[0];
      x = at::log_softmax(x, 1);
      return x;
    };

    auto optim = SGD(model, 1e-2).momentum(0.5).make();

    REQUIRE(test_mnist(
        32, // batch_size
        3, // num_epochs
        true, // useGPU
        model,
        forward,
        optim));
  }

  SECTION("mnist_batchnorm") {
    auto model = make(SimpleContainer());
    auto conv1 = model->add(make(Conv2d(1, 10, 5)), "conv1");
    auto batchnorm2d =
        model->add(make(BatchNorm(10).stateful()), "batchnorm2d");
    auto conv2 = model->add(make(Conv2d(10, 20, 5)), "conv2");
    auto linear1 = model->add(make(Linear(320, 50)), "linear1");
    auto batchnorm1 = model->add(make(BatchNorm(50).stateful()), "batchnorm1");
    auto linear2 = model->add(make(Linear(50, 10)), "linear2");

    auto forward = [&](Variable x) {
      x = std::get<0>(at::max_pool2d(conv1->forward({x})[0], {2, 2}))
              .clamp_min(0);
      x = batchnorm2d->forward({x})[0];
      x = conv2->forward({x})[0];
      x = std::get<0>(at::max_pool2d(x, {2, 2})).clamp_min(0);

      x = x.view({-1, 320});
      x = linear1->forward({x})[0].clamp_min(0);
      x = batchnorm1->forward({x})[0];
      x = linear2->forward({x})[0];
      x = at::log_softmax(x, 1);
      return x;
    };

    auto optim = SGD(model, 1e-2).momentum(0.5).make();

    REQUIRE(test_mnist(
        32, // batch_size
        3, // num_epochs
        true, // useGPU
        model,
        forward,
        optim));
  }
}
