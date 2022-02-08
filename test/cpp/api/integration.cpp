#include <gtest/gtest.h>

#include <c10/util/irange.h>
#include <torch/torch.h>

#include <test/cpp/api/support.h>

#include <cmath>
#include <cstdlib>
#include <random>

using namespace torch::nn;
using namespace torch::test;

const double kPi = 3.1415926535898;

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
  double theta_threshold_radians = 12 * 2 * kPi / 360;
  double x_threshold = 2.4;
  int steps_beyond_done = -1;

  torch::Tensor state;
  double reward;
  bool done;
  int step_ = 0;

  torch::Tensor getState() {
    return state;
  }

  double getReward() {
    return reward;
  }

  double isDone() {
    return done;
  }

  void reset() {
    state = torch::empty({4}).uniform_(-0.05, 0.05);
    steps_beyond_done = -1;
    step_ = 0;
  }

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  CartPole() {
    reset();
  }

  void step(int action) {
    auto x = state[0].item<float>();
    auto x_dot = state[1].item<float>();
    auto theta = state[2].item<float>();
    auto theta_dot = state[3].item<float>();

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
    state = torch::tensor({x, x_dot, theta, theta_dot});

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
        AT_ASSERT(false); // Can't do this
      }
    }
    step_++;
  }
};

template <typename M, typename F, typename O>
bool test_mnist(
    size_t batch_size,
    size_t number_of_epochs,
    bool with_cuda,
    M&& model,
    F&& forward_op,
    O&& optimizer) {
  std::string mnist_path = "mnist";
  if (const char* user_mnist_path = getenv("TORCH_CPP_TEST_MNIST_PATH")) {
    mnist_path = user_mnist_path;
  }

  auto train_dataset =
      torch::data::datasets::MNIST(
          mnist_path, torch::data::datasets::MNIST::Mode::kTrain)
          .map(torch::data::transforms::Stack<>());

  auto data_loader =
      torch::data::make_data_loader(std::move(train_dataset), batch_size);

  torch::Device device(with_cuda ? torch::kCUDA : torch::kCPU);
  model->to(device);

  for (const auto epoch : c10::irange(number_of_epochs)) {
    (void)epoch; // Suppress unused variable warning
    // NOLINTNEXTLINE(performance-for-range-copy)
    for (torch::data::Example<> batch : *data_loader) {
      auto data = batch.data.to(device);
      auto targets = batch.target.to(device);
      torch::Tensor prediction = forward_op(std::move(data));
      // NOLINTNEXTLINE(performance-move-const-arg)
      torch::Tensor loss = torch::nll_loss(prediction, std::move(targets));
      AT_ASSERT(!torch::isnan(loss).any().item<int64_t>());
      optimizer.zero_grad();
      loss.backward();
      optimizer.step();
    }
  }

  torch::NoGradGuard guard;
  torch::data::datasets::MNIST test_dataset(
      mnist_path, torch::data::datasets::MNIST::Mode::kTest);
  auto images = test_dataset.images().to(device),
       targets = test_dataset.targets().to(device);

  auto result = std::get<1>(forward_op(images).max(/*dim=*/1));
  torch::Tensor correct = (result == targets).to(torch::kFloat32);
  return correct.sum().item<float>() > (test_dataset.size().value() * 0.8);
}

struct IntegrationTest : torch::test::SeedingFixture {};

TEST_F(IntegrationTest, CartPole) {
  torch::manual_seed(0);
  auto model = std::make_shared<SimpleContainer>();
  auto linear = model->add(Linear(4, 128), "linear");
  auto policyHead = model->add(Linear(128, 2), "policy");
  auto valueHead = model->add(Linear(128, 1), "action");
  auto optimizer = torch::optim::Adam(model->parameters(), 1e-3);

  std::vector<torch::Tensor> saved_log_probs;
  std::vector<torch::Tensor> saved_values;
  std::vector<float> rewards;

  auto forward = [&](torch::Tensor inp) {
    auto x = linear->forward(inp).clamp_min(0);
    torch::Tensor actions = policyHead->forward(x);
    torch::Tensor value = valueHead->forward(x);
    return std::make_tuple(torch::softmax(actions, -1), value);
  };

  auto selectAction = [&](torch::Tensor state) {
    // Only work on single state right now, change index to gather for batch
    auto out = forward(state);
    auto probs = torch::Tensor(std::get<0>(out));
    auto value = torch::Tensor(std::get<1>(out));
    auto action = probs.multinomial(1)[0].item<int32_t>();
    // Compute the log prob of a multinomial distribution.
    // This should probably be actually implemented in autogradpp...
    auto p = probs / probs.sum(-1, true);
    auto log_prob = p[action].log();
    saved_log_probs.emplace_back(log_prob);
    saved_values.push_back(value);
    return action;
  };

  auto finishEpisode = [&] {
    auto R = 0.;
    // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
    for (int i = rewards.size() - 1; i >= 0; i--) {
      R = rewards[i] + 0.99 * R;
      rewards[i] = R;
    }
    auto r_t = torch::from_blob(
        rewards.data(), {static_cast<int64_t>(rewards.size())});
    r_t = (r_t - r_t.mean()) / (r_t.std() + 1e-5);

    std::vector<torch::Tensor> policy_loss;
    std::vector<torch::Tensor> value_loss;
    for (const auto i : c10::irange(0U, saved_log_probs.size())) {
      auto advantage = r_t[i] - saved_values[i].item<float>();
      policy_loss.push_back(-advantage * saved_log_probs[i]);
      value_loss.push_back(
          torch::smooth_l1_loss(saved_values[i], torch::ones(1) * r_t[i]));
    }

    auto loss =
        torch::stack(policy_loss).sum() + torch::stack(value_loss).sum();

    optimizer.zero_grad();
    loss.backward();
    optimizer.step();

    rewards.clear();
    saved_log_probs.clear();
    saved_values.clear();
  };

  auto env = CartPole();
  double running_reward = 10.0;
  for (size_t episode = 0;; episode++) {
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
    if (running_reward > 150) {
      break;
    }
    ASSERT_LT(episode, 3000);
  }
}

TEST_F(IntegrationTest, MNIST_CUDA) {
  torch::manual_seed(0);
  auto model = std::make_shared<SimpleContainer>();
  auto conv1 = model->add(Conv2d(1, 10, 5), "conv1");
  auto conv2 = model->add(Conv2d(10, 20, 5), "conv2");
  auto drop = Dropout(0.3);
  auto drop2d = Dropout2d(0.3);
  auto linear1 = model->add(Linear(320, 50), "linear1");
  auto linear2 = model->add(Linear(50, 10), "linear2");

  auto forward = [&](torch::Tensor x) {
    x = torch::max_pool2d(conv1->forward(x), {2, 2}).relu();
    x = conv2->forward(x);
    x = drop2d->forward(x);
    x = torch::max_pool2d(x, {2, 2}).relu();

    x = x.view({-1, 320});
    x = linear1->forward(x).clamp_min(0);
    x = drop->forward(x);
    x = linear2->forward(x);
    x = torch::log_softmax(x, 1);
    return x;
  };

  auto optimizer = torch::optim::SGD(
      model->parameters(), torch::optim::SGDOptions(1e-2).momentum(0.5));

  ASSERT_TRUE(test_mnist(
      32, // batch_size
      3, // number_of_epochs
      true, // with_cuda
      model,
      forward,
      optimizer));
}

TEST_F(IntegrationTest, MNISTBatchNorm_CUDA) {
  torch::manual_seed(0);
  auto model = std::make_shared<SimpleContainer>();
  auto conv1 = model->add(Conv2d(1, 10, 5), "conv1");
  auto batchnorm2d = model->add(BatchNorm2d(10), "batchnorm2d");
  auto conv2 = model->add(Conv2d(10, 20, 5), "conv2");
  auto linear1 = model->add(Linear(320, 50), "linear1");
  auto batchnorm1 = model->add(BatchNorm1d(50), "batchnorm1");
  auto linear2 = model->add(Linear(50, 10), "linear2");

  auto forward = [&](torch::Tensor x) {
    x = torch::max_pool2d(conv1->forward(x), {2, 2}).relu();
    x = batchnorm2d->forward(x);
    x = conv2->forward(x);
    x = torch::max_pool2d(x, {2, 2}).relu();

    x = x.view({-1, 320});
    x = linear1->forward(x).clamp_min(0);
    x = batchnorm1->forward(x);
    x = linear2->forward(x);
    x = torch::log_softmax(x, 1);
    return x;
  };

  auto optimizer = torch::optim::SGD(
      model->parameters(), torch::optim::SGDOptions(1e-2).momentum(0.5));

  ASSERT_TRUE(test_mnist(
      32, // batch_size
      3, // number_of_epochs
      true, // with_cuda
      model,
      forward,
      optimizer));
}
