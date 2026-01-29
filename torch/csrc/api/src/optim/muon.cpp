#include <torch/optim/muon.h>

#include <torch/csrc/autograd/variable.h>
#include <torch/nn/module.h>
#include <torch/serialize/archive.h>
#include <torch/utils.h>

#include <ATen/ATen.h>
#include <c10/util/irange.h>

#include <cmath>
#include <functional>

namespace torch::optim {

MuonOptions::MuonOptions(double lr) : lr_(lr) {}

bool operator==(const MuonOptions& lhs, const MuonOptions& rhs) {
  return (lhs.lr() == rhs.lr()) &&
      (std::get<0>(lhs.ns_coefficients()) ==
       std::get<0>(rhs.ns_coefficients())) &&
      (std::get<1>(lhs.ns_coefficients()) ==
       std::get<1>(rhs.ns_coefficients())) &&
      (std::get<2>(lhs.ns_coefficients()) ==
       std::get<2>(rhs.ns_coefficients())) &&
      (lhs.eps() == rhs.eps()) && (lhs.weight_decay() == rhs.weight_decay()) &&
      (lhs.momentum() == rhs.momentum()) &&
      (lhs.nesterov() == rhs.nesterov()) &&
      (lhs.ns_steps() == rhs.ns_steps()) &&
      (lhs.match_rms_adamw() == rhs.match_rms_adamw());
}

void MuonOptions::serialize(torch::serialize::OutputArchive& archive) const {
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(lr);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(weight_decay);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(momentum);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(nesterov);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(ns_coefficients);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(eps);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(ns_steps);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(match_rms_adamw);
}

void MuonOptions::serialize(torch::serialize::InputArchive& archive) {
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, lr);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, weight_decay);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, momentum);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(bool, nesterov);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(ns_coefficients_t, ns_coefficients);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, eps);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int, ns_steps);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(bool, match_rms_adamw);
}

double MuonOptions::get_lr() const {
  return lr();
}

void MuonOptions::set_lr(const double lr) {
  this->lr(lr);
}

bool operator==(const MuonParamState& lhs, const MuonParamState& rhs) {
  return (lhs.step() == rhs.step()) &&
      torch::equal(lhs.momentum_buffer(), rhs.momentum_buffer());
}

void MuonParamState::serialize(torch::serialize::OutputArchive& archive) const {
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(step);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(momentum_buffer);
}

void MuonParamState::serialize(torch::serialize::InputArchive& archive) {
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int64_t, step);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, momentum_buffer);
}

Tensor _zeropower_via_newtonschulz(
    Tensor grad,
    std::tuple<double, double, double> ns_coefficients,
    int ns_steps,
    double eps) {
  TORCH_CHECK(
      ns_steps < 100,
      "Number of steps must be less than 100 for computational efficiency");
  TORCH_CHECK(grad.dim() == 2, "Muon only supports 2D gradients");

  auto a = std::get<0>(ns_coefficients);
  auto b = std::get<1>(ns_coefficients);
  auto c = std::get<2>(ns_coefficients);

  auto ortho_grad = grad.to(torch::kBFloat16);
  if (grad.size(0) > grad.size(1)) {
    ortho_grad = ortho_grad.t();
  }

  // Ensure spectral norm is at most 1
  ortho_grad.div_(ortho_grad.norm().clamp(eps));

  // Perform the NS iterations
  for (int i = 0; i < ns_steps; ++i) {
    auto gram_matrix = ortho_grad.mm(ortho_grad.t());
    auto gram_update =
        torch::addmm(gram_matrix, gram_matrix, gram_matrix, b, c);
    ortho_grad = torch::addmm(ortho_grad, gram_update, ortho_grad, a);
  }

  if (grad.size(0) > grad.size(1)) {
    ortho_grad = ortho_grad.t();
  }
  return ortho_grad;
}

Tensor Muon::step(LossClosure closure) {
  NoGradGuard no_grad;
  Tensor loss = {};
  if (closure != nullptr) {
    at::AutoGradMode enable_grad(true);
    loss = closure();
  }
  for (auto& group : param_groups_) {
    for (auto& p : group.params()) {
      if (!p.grad().defined()) {
        continue;
      }
      TORCH_CHECK(p.grad().dim() == 2, "Muon only supports 2D gradients");
      const auto& grad = p.grad();
      TORCH_CHECK(!grad.is_sparse(), "Muon does not support sparse gradients");
      auto param_state = state_.find(p.unsafeGetTensorImpl());
      auto& options = static_cast<MuonOptions&>(group.options());

      // Perform stepweight decay
      if (options.weight_decay() != 0) {
        p.mul_(1 - options.lr() * options.weight_decay());
      }

      // State initialization
      if (param_state == state_.end()) {
        auto state = std::make_unique<MuonParamState>();
        state->step(0);
        state->momentum_buffer(torch::zeros_like(p, MemoryFormat::Preserve));
        state_[p.unsafeGetTensorImpl()] = std::move(state);
      }

      auto& state =
          static_cast<MuonParamState&>(*state_[p.unsafeGetTensorImpl()]);
      auto& buf = state.momentum_buffer();
      auto& momentum = options.momentum();
      auto nesterov = options.nesterov();
      auto ns_coefficients = options.ns_coefficients();
      auto weight_decay = options.weight_decay();
      auto ns_steps = options.ns_steps();
      auto eps = options.eps();
      auto lr = options.lr();
      auto match_rms_adamw = options.match_rms_adamw();

      state.step(state.step() + 1);

      buf.lerp_(grad, 1 - momentum);

      auto update = buf;
      if (nesterov) {
        update = grad.lerp(buf, momentum);
      }

      update =
          _zeropower_via_newtonschulz(update, ns_coefficients, ns_steps, eps);

      // Adjust learning rate
      double adjusted_lr, adjusted_ratio;
      double A = p.size(0);
      double B = p.size(1);
      if (match_rms_adamw) {
        adjusted_ratio = 0.2 * std::sqrt(std::max(A, B));
      } else {
        adjusted_ratio = std::sqrt(std::max(1.0, A / B));
      }
      adjusted_lr = lr * adjusted_ratio;

      p.mul_(1 - lr * weight_decay);
      p.add_(update, -adjusted_lr);
    }
  }
  return loss;
}

void Muon::save(serialize::OutputArchive& archive) const {
  serialize(*this, archive);
}

void Muon::load(serialize::InputArchive& archive) {
  IValue pytorch_version;
  if (archive.try_read("pytorch_version", pytorch_version)) {
    serialize(*this, archive);
  } else { // deserializing archives saved in old format (prior to
           // version 1.5.0)
    TORCH_WARN(
        "Your serialized Muon optimizer is still using the old serialization format. "
        "You should re-save your Muon optimizer to use the new serialization format.");
    std::vector<int64_t> step_buffers;
    std::vector<at::Tensor> momentum_buffers;
    torch::optim::serialize(archive, "step_buffers", step_buffers);
    torch::optim::serialize(archive, "momentum_buffers", momentum_buffers);
    // since there were no param_groups prior to version 1.5.0, assuming all
    // tensors are now in one param_group
    std::vector<Tensor> params = param_groups_.at(0).params();
    for (const auto idx : c10::irange(step_buffers.size())) {
      auto state = std::make_unique<MuonParamState>();
      state->step(step_buffers.at(idx));
      state->momentum_buffer(momentum_buffers.at(idx));
      state_[params.at(idx).unsafeGetTensorImpl()] = std::move(state);
    }
  }
}
} // namespace torch::optim
