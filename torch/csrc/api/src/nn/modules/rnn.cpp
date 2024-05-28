#include <torch/nn/modules/rnn.h>

#include <torch/nn/init.h>
#include <torch/types.h>
#include <torch/utils.h>

#include <c10/util/Exception.h>
#include <c10/util/irange.h>

#include <cmath>
#include <cstdint>
#include <regex>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace torch::nn::utils::rnn;

namespace torch {
namespace nn {

/// These must line up with the CUDNN mode codes:
/// https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnRNNMode_t
enum class CuDNNMode { RNN_RELU = 0, RNN_TANH = 1, LSTM = 2, GRU = 3 };

static CuDNNMode get_cudnn_mode_for_rnn(
    detail::RNNOptionsBase::rnn_options_base_mode_t mode) {
  if (std::holds_alternative<enumtype::kRNN_RELU>(mode)) {
    return CuDNNMode::RNN_RELU;
  } else if (std::holds_alternative<enumtype::kRNN_TANH>(mode)) {
    return CuDNNMode::RNN_TANH;
  } else if (std::holds_alternative<enumtype::kLSTM>(mode)) {
    return CuDNNMode::LSTM;
  } else if (std::holds_alternative<enumtype::kGRU>(mode)) {
    return CuDNNMode::GRU;
  } else {
    TORCH_CHECK(false, "Unknown mode: ", torch::enumtype::get_enum_name(mode));
  }
}

static Tensor apply_permutation(
    const Tensor& tensor,
    const Tensor& permutation,
    int64_t dim = 1) {
  return tensor.index_select(dim, permutation);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ RNNImplBase ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
namespace detail {
template <typename Derived>
RNNImplBase<Derived>::RNNImplBase(const RNNOptionsBase& options_)
    : options_base(options_) {
  // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
  reset();
}

template <typename Derived>
void RNNImplBase<Derived>::reset() {
  const int64_t num_directions = options_base.bidirectional() ? 2 : 1;

  TORCH_CHECK(
      0 <= options_base.dropout() && options_base.dropout() <= 1,
      "dropout should be a number in range [0, 1] ",
      "representing the probability of an element being ",
      "zeroed");

  if (options_base.dropout() > 0 && options_base.num_layers() == 1) {
    TORCH_WARN(
        "dropout option adds dropout after all but last ",
        "recurrent layer, so non-zero dropout expects ",
        "num_layers greater than 1, but got dropout=",
        options_base.dropout(),
        " and ",
        "num_layers=",
        options_base.num_layers());
  }

  TORCH_CHECK(
      options_base.hidden_size() > 0, "hidden_size must be greater than zero");

  TORCH_CHECK(
      options_base.num_layers() > 0, "num_layers must be greater than zero");

  TORCH_CHECK(
      0 <= options_base.proj_size() &&
          options_base.proj_size() < options_base.hidden_size(),
      "proj_size has to be a positive integer, smaller than ",
      "hidden_size or zero to disable projections");

  if (options_base.proj_size() > 0) {
    TORCH_CHECK(
        std::get_if<enumtype::kLSTM>(&options_base.mode()),
        "proj_size argument is only supported for LSTM, not RNN or GRU");
  }

  int64_t gate_size = 0;
  if (std::holds_alternative<enumtype::kLSTM>(options_base.mode())) {
    gate_size = 4 * options_base.hidden_size();
  } else if (std::holds_alternative<enumtype::kGRU>(options_base.mode())) {
    gate_size = 3 * options_base.hidden_size();
    // NOLINTNEXTLINE(bugprone-branch-clone)
  } else if (std::holds_alternative<enumtype::kRNN_TANH>(options_base.mode())) {
    gate_size = options_base.hidden_size();
  } else if (std::holds_alternative<enumtype::kRNN_RELU>(options_base.mode())) {
    gate_size = options_base.hidden_size();
  } else {
    TORCH_CHECK(
        false,
        "Unrecognized RNN mode: " +
            torch::enumtype::get_enum_name(options_base.mode()));
  }

  flat_weights_names_ = {};
  all_weights_ = {};

  for (const auto layer : c10::irange(options_base.num_layers())) {
    for (const auto direction : c10::irange(num_directions)) {
      int64_t real_hidden_size = options_base.proj_size() > 0
          ? options_base.proj_size()
          : options_base.hidden_size();
      int64_t layer_input_size = layer == 0 ? options_base.input_size()
                                            : real_hidden_size * num_directions;

      auto w_ih = torch::empty({gate_size, layer_input_size});
      auto w_hh = torch::empty({gate_size, real_hidden_size});
      auto b_ih = torch::empty({gate_size});
      // Second bias vector included for CuDNN compatibility. Only one
      // bias vector is needed in standard definition.
      auto b_hh = torch::empty({gate_size});
      std::vector<Tensor> layer_params = {w_ih, w_hh};

      std::string suffix = direction == 1 ? "_reverse" : "";
      std::vector<std::string> param_names = {
          "weight_ih_l{layer}{suffix}", "weight_hh_l{layer}{suffix}"};
      if (options_base.bias()) {
        param_names.emplace_back("bias_ih_l{layer}{suffix}");
        param_names.emplace_back("bias_hh_l{layer}{suffix}");
        layer_params.emplace_back(b_ih);
        layer_params.emplace_back(b_hh);
      }
      if (options_base.proj_size() > 0) {
        auto w_hr = torch::empty(
            {options_base.proj_size(), options_base.hidden_size()});
        layer_params.emplace_back(w_hr);
        param_names.emplace_back("weight_hr_l{layer}{suffix}");
      }
      for (auto& param_name : param_names) {
        std::string x = std::regex_replace(
            param_name, std::regex("\\{layer\\}"), c10::str(layer));
        param_name =
            std::regex_replace(x, std::regex("\\{suffix\\}"), c10::str(suffix));
      }

      for (const auto i : c10::irange(param_names.size())) {
        this->register_parameter(param_names[i], std::move(layer_params[i]));
      }
      flat_weights_names_.insert(
          flat_weights_names_.end(), param_names.begin(), param_names.end());
      all_weights_.emplace_back(std::move(param_names));
    }
  }

  flat_weights_ = {};
  for (const auto& wn : flat_weights_names_) {
    auto named_parameters = this->named_parameters(/*recurse=*/false);
    if (named_parameters.contains(wn)) {
      flat_weights_.emplace_back(named_parameters[wn]);
    } else {
      flat_weights_.emplace_back();
    }
  }

  this->flatten_parameters();
  this->reset_parameters();
}

template <typename Derived>
void RNNImplBase<Derived>::flatten_parameters() {
  // Resets parameter data pointer so that they can use faster code paths.
  //
  // Right now, this works only if the module is on the GPU and cuDNN is
  // enabled. Otherwise, it's a no-op.

  // Short-circuits if flat_weights_ is only partially instantiated
  if (flat_weights_.size() != flat_weights_names_.size()) {
    return;
  }

  // Short-circuits if any tensor in self.flat_weights_ is not acceptable to
  // cuDNN or the tensors in flat_weights_ are of different dtypes

  auto first_fw = flat_weights_[0];
  auto dtype = first_fw.dtype();
  for (const auto& fw : flat_weights_) {
    if (!(fw.dtype() == dtype) || !fw.is_cuda() ||
        !torch::cudnn_is_acceptable(fw)) {
      return;
    }
  }

  // If any parameters alias, we fall back to the slower, copying code path.
  // This is a sufficient check, because overlapping parameter buffers that
  // don't completely alias would break the assumptions of the uniqueness check
  // in Module::named_parameters().
  std::unordered_set<void*> unique_data_ptrs;
  for (const auto& p : flat_weights_) {
    unique_data_ptrs.emplace(p.data_ptr());
  }
  if (unique_data_ptrs.size() != flat_weights_.size()) {
    return;
  }

  {
    torch::DeviceGuard device_guard(first_fw.device());

    // Note: no_grad() is necessary since _cudnn_rnn_flatten_weight is
    // an inplace operation on self.flat_weights_
    {
      torch::NoGradGuard no_grad;
      if (torch::_use_cudnn_rnn_flatten_weight()) {
        int64_t num_weights = options_base.bias() ? 4 : 2;
        if (options_base.proj_size() > 0) {
          ++num_weights;
        }
        torch::_cudnn_rnn_flatten_weight(
            flat_weights_,
            num_weights,
            options_base.input_size(),
            static_cast<int64_t>(get_cudnn_mode_for_rnn(options_base.mode())),
            options_base.hidden_size(),
            options_base.proj_size(),
            options_base.num_layers(),
            options_base.batch_first(),
            options_base.bidirectional());
      }
    }
  }
}

template <typename Derived>
void RNNImplBase<Derived>::reset_flat_weights() {
  flat_weights_ = {};
  for (const auto& wn : flat_weights_names_) {
    auto named_parameters = this->named_parameters(/*recurse=*/false);
    if (named_parameters.contains(wn)) {
      flat_weights_.emplace_back(named_parameters[wn]);
    } else {
      flat_weights_.emplace_back();
    }
  }
}

template <typename Derived>
void RNNImplBase<Derived>::to(
    torch::Device device,
    torch::Dtype dtype,
    bool non_blocking) {
  nn::Module::to(device, dtype, non_blocking);
  reset_flat_weights();
  flatten_parameters();
}

template <typename Derived>
void RNNImplBase<Derived>::to(torch::Dtype dtype, bool non_blocking) {
  nn::Module::to(dtype, non_blocking);
  reset_flat_weights();
  flatten_parameters();
}

template <typename Derived>
void RNNImplBase<Derived>::to(torch::Device device, bool non_blocking) {
  nn::Module::to(device, non_blocking);
  reset_flat_weights();
  flatten_parameters();
}

template <typename Derived>
void RNNImplBase<Derived>::reset_parameters() {
  const double stdv = 1.0 / std::sqrt(options_base.hidden_size());
  for (auto& weight : this->parameters()) {
    init::uniform_(weight, -stdv, stdv);
  }
}

template <typename Derived>
void RNNImplBase<Derived>::check_input(
    const Tensor& input,
    const Tensor& batch_sizes) const {
  int64_t expected_input_dim = batch_sizes.defined() ? 2 : 3;
  TORCH_CHECK(
      input.dim() == expected_input_dim,
      "input must have ",
      expected_input_dim,
      " dimensions, got ",
      input.dim());
  TORCH_CHECK(
      options_base.input_size() == input.size(-1),
      "input.size(-1) must be equal to input_size. Expected ",
      options_base.input_size(),
      ", got ",
      input.size(-1));
}

template <typename Derived>
std::tuple<int64_t, int64_t, int64_t> RNNImplBase<Derived>::
    get_expected_hidden_size(const Tensor& input, const Tensor& batch_sizes)
        const {
  int64_t mini_batch = 0;
  if (batch_sizes.defined()) {
    mini_batch = batch_sizes[0].item<int64_t>();
  } else {
    mini_batch = options_base.batch_first() ? input.size(0) : input.size(1);
  }
  int64_t num_directions = options_base.bidirectional() ? 2 : 1;
  int64_t real_hidden_size = options_base.proj_size() > 0
      ? options_base.proj_size()
      : options_base.hidden_size();
  return std::make_tuple(
      options_base.num_layers() * num_directions, mini_batch, real_hidden_size);
}

template <typename Derived>
void RNNImplBase<Derived>::check_hidden_size(
    const Tensor& hx,
    std::tuple<int64_t, int64_t, int64_t> expected_hidden_size,
    std::string msg) const {
  auto expected_hidden_size_vec = std::vector<int64_t>({
      std::get<0>(expected_hidden_size),
      std::get<1>(expected_hidden_size),
      std::get<2>(expected_hidden_size),
  });
  if (hx.sizes() != expected_hidden_size_vec) {
    msg = std::regex_replace(
        msg, std::regex("\\{1\\}"), c10::str(expected_hidden_size_vec));
    msg = std::regex_replace(msg, std::regex("\\{2\\}"), c10::str(hx.sizes()));
    TORCH_CHECK(false, msg);
  }
}

template <typename Derived>
void RNNImplBase<Derived>::check_forward_args(
    Tensor input,
    Tensor hidden,
    Tensor batch_sizes) const {
  this->check_input(input, batch_sizes);
  auto expected_hidden_size =
      this->get_expected_hidden_size(input, batch_sizes);

  this->check_hidden_size(hidden, expected_hidden_size);
}

template <typename Derived>
Tensor RNNImplBase<Derived>::permute_hidden(
    Tensor hx,
    const Tensor& permutation) const {
  if (!permutation.defined()) {
    return hx;
  }
  return apply_permutation(hx, permutation);
}

template <typename Derived>
void RNNImplBase<Derived>::pretty_print(std::ostream& stream) const {
  const std::string name = this->name();
  const std::string name_without_impl = name.substr(0, name.size() - 4);
  stream << std::boolalpha << name_without_impl
         << "(input_size=" << options_base.input_size()
         << ", hidden_size=" << options_base.hidden_size()
         << ", num_layers=" << options_base.num_layers()
         << ", bias=" << options_base.bias()
         << ", batch_first=" << options_base.batch_first()
         << ", dropout=" << options_base.dropout()
         << ", bidirectional=" << options_base.bidirectional();
  if (options_base.proj_size() > 0) {
    stream << ", proj_size=" << options_base.proj_size();
  }
  stream << ")";
}

template <typename Derived>
std::vector<Tensor> RNNImplBase<Derived>::all_weights() const {
  std::vector<Tensor> result = {};
  auto named_parameters = this->named_parameters(/*recurse=*/false);
  for (const auto& weights : all_weights_) {
    for (const auto& weight : weights) {
      result.emplace_back(named_parameters[weight]);
    }
  }
  return result;
}

template class RNNImplBase<LSTMImpl>;
template class RNNImplBase<GRUImpl>;
template class RNNImplBase<RNNImpl>;
} // namespace detail

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ RNN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

static detail::RNNOptionsBase::rnn_options_base_mode_t
compute_rnn_options_base_mode(RNNOptions::nonlinearity_t nonlinearity) {
  if (std::holds_alternative<enumtype::kTanh>(nonlinearity)) {
    return torch::kRNN_TANH;
  } else if (std::holds_alternative<enumtype::kReLU>(nonlinearity)) {
    return torch::kRNN_RELU;
  } else {
    TORCH_CHECK(
        false,
        "Unknown nonlinearity ",
        torch::enumtype::get_enum_name(nonlinearity));
  }
}

RNNImpl::RNNImpl(const RNNOptions& options_)
    : detail::RNNImplBase<RNNImpl>(
          detail::RNNOptionsBase(
              compute_rnn_options_base_mode(options_.nonlinearity()),
              options_.input_size(),
              options_.hidden_size())
              .num_layers(options_.num_layers())
              .bias(options_.bias())
              .batch_first(options_.batch_first())
              .dropout(options_.dropout())
              .bidirectional(options_.bidirectional())),
      options(options_) {}

std::tuple<Tensor, Tensor> RNNImpl::forward_helper(
    const Tensor& input,
    const Tensor& batch_sizes,
    const Tensor& sorted_indices,
    int64_t max_batch_size,
    Tensor hx) {
  if (!hx.defined()) {
    int64_t num_directions = options_base.bidirectional() ? 2 : 1;
    hx = torch::zeros(
        {options_base.num_layers() * num_directions,
         max_batch_size,
         options_base.hidden_size()},
        torch::dtype(input.dtype()).device(input.device()));
  } else {
    // Each batch of the hidden state should match the input sequence that
    // the user believes he/she is passing in.
    hx = this->permute_hidden(hx, sorted_indices);
  }

  this->check_forward_args(input, hx, batch_sizes);

  std::tuple<Tensor, Tensor> result;
  if (!batch_sizes.defined()) {
    if (std::holds_alternative<enumtype::kRNN_TANH>(options_base.mode())) {
      result = torch::rnn_tanh(
          input,
          hx,
          flat_weights_,
          options_base.bias(),
          options_base.num_layers(),
          options_base.dropout(),
          this->is_training(),
          options_base.bidirectional(),
          options_base.batch_first());
    } else if (std::holds_alternative<enumtype::kRNN_RELU>(
                   options_base.mode())) {
      result = torch::rnn_relu(
          input,
          hx,
          flat_weights_,
          options_base.bias(),
          options_base.num_layers(),
          options_base.dropout(),
          this->is_training(),
          options_base.bidirectional(),
          options_base.batch_first());
    } else {
      TORCH_CHECK(
          false,
          "Unknown mode: ",
          torch::enumtype::get_enum_name(options_base.mode()));
    }
  } else {
    if (std::holds_alternative<enumtype::kRNN_TANH>(options_base.mode())) {
      result = torch::rnn_tanh(
          input,
          batch_sizes,
          hx,
          flat_weights_,
          options_base.bias(),
          options_base.num_layers(),
          options_base.dropout(),
          this->is_training(),
          options_base.bidirectional());
    } else if (std::holds_alternative<enumtype::kRNN_RELU>(
                   options_base.mode())) {
      result = torch::rnn_relu(
          input,
          batch_sizes,
          hx,
          flat_weights_,
          options_base.bias(),
          options_base.num_layers(),
          options_base.dropout(),
          this->is_training(),
          options_base.bidirectional());
    } else {
      TORCH_CHECK(
          false,
          "Unknown mode: ",
          torch::enumtype::get_enum_name(options_base.mode()));
    }
  }
  auto output = std::get<0>(result);
  auto hidden = std::get<1>(result);

  return std::make_tuple(output, hidden);
}

std::tuple<Tensor, Tensor> RNNImpl::forward(const Tensor& input, Tensor hx) {
  auto batch_sizes = torch::Tensor();
  auto max_batch_size =
      options_base.batch_first() ? input.size(0) : input.size(1);
  auto sorted_indices = torch::Tensor();
  auto unsorted_indices = torch::Tensor();

  auto [output, hidden] = this->forward_helper(
      input, batch_sizes, sorted_indices, max_batch_size, std::move(hx));

  return std::make_tuple(
      output, this->permute_hidden(hidden, unsorted_indices));
}

std::tuple<PackedSequence, Tensor> RNNImpl::forward_with_packed_input(
    const PackedSequence& packed_input,
    Tensor hx) {
  const auto& input = packed_input.data();
  const auto& batch_sizes = packed_input.batch_sizes();
  const auto& sorted_indices = packed_input.sorted_indices();
  const auto& unsorted_indices = packed_input.unsorted_indices();
  auto max_batch_size = batch_sizes[0].item<int64_t>();

  auto [output, hidden] = this->forward_helper(
      input, batch_sizes, sorted_indices, max_batch_size, std::move(hx));

  auto output_packed =
      PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices);
  return std::make_tuple(
      output_packed, this->permute_hidden(hidden, unsorted_indices));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LSTM ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LSTMImpl::LSTMImpl(const LSTMOptions& options_)
    : detail::RNNImplBase<LSTMImpl>(detail::RNNOptionsBase(
                                        torch::kLSTM,
                                        options_.input_size(),
                                        options_.hidden_size())
                                        .num_layers(options_.num_layers())
                                        .bias(options_.bias())
                                        .batch_first(options_.batch_first())
                                        .dropout(options_.dropout())
                                        .bidirectional(options_.bidirectional())
                                        .proj_size(options_.proj_size())),
      options(options_) {}

std::tuple<int64_t, int64_t, int64_t> LSTMImpl::get_expected_cell_size(
    const Tensor& input,
    const Tensor& batch_sizes) const {
  int64_t mini_batch = 0;
  if (batch_sizes.defined()) {
    mini_batch = batch_sizes[0].item<int64_t>();
  } else {
    mini_batch = options_base.batch_first() ? input.size(0) : input.size(1);
  }
  int64_t num_directions = options_base.bidirectional() ? 2 : 1;
  return std::make_tuple(
      options_base.num_layers() * num_directions,
      mini_batch,
      options_base.hidden_size());
}

void LSTMImpl::check_forward_args(
    const Tensor& input,
    std::tuple<Tensor, Tensor> hidden,
    const Tensor& batch_sizes) const {
  this->check_input(input, batch_sizes);
  this->check_hidden_size(
      std::get<0>(hidden),
      this->get_expected_hidden_size(input, batch_sizes),
      "Expected hidden[0] size {1}, got {2}");
  this->check_hidden_size(
      std::get<1>(hidden),
      this->get_expected_cell_size(input, batch_sizes),
      "Expected hidden[1] size {1}, got {2}");
}

std::tuple<Tensor, Tensor> LSTMImpl::permute_hidden(
    std::tuple<Tensor, Tensor> hx,
    const Tensor& permutation) const {
  if (!permutation.defined()) {
    return hx;
  }
  return std::make_tuple(
      apply_permutation(std::get<0>(hx), permutation),
      apply_permutation(std::get<1>(hx), permutation));
}

std::tuple<Tensor, std::tuple<Tensor, Tensor>> LSTMImpl::forward_helper(
    const Tensor& input,
    const Tensor& batch_sizes,
    const Tensor& sorted_indices,
    int64_t max_batch_size,
    torch::optional<std::tuple<Tensor, Tensor>> hx_opt) {
  std::tuple<Tensor, Tensor> hx;
  if (!hx_opt.has_value()) {
    int64_t num_directions = options.bidirectional() ? 2 : 1;
    int64_t real_hidden_size =
        options.proj_size() > 0 ? options.proj_size() : options.hidden_size();
    auto h_zeros = torch::zeros(
        {options.num_layers() * num_directions,
         max_batch_size,
         real_hidden_size},
        torch::dtype(input.dtype()).device(input.device()));
    auto c_zeros = torch::zeros(
        {options.num_layers() * num_directions,
         max_batch_size,
         options.hidden_size()},
        torch::dtype(input.dtype()).device(input.device()));
    hx = std::make_tuple(h_zeros, c_zeros);
  } else {
    hx = hx_opt.value();
    // Each batch of the hidden state should match the input sequence that
    // the user believes he/she is passing in.
    hx = this->permute_hidden(hx, sorted_indices);
  }

  this->check_forward_args(input, hx, batch_sizes);
  std::tuple<Tensor, Tensor, Tensor> result;
  if (!batch_sizes.defined()) {
    result = torch::lstm(
        input,
        {std::get<0>(hx), std::get<1>(hx)},
        flat_weights_,
        options.bias(),
        options.num_layers(),
        options.dropout(),
        this->is_training(),
        options.bidirectional(),
        options.batch_first());
  } else {
    result = torch::lstm(
        input,
        batch_sizes,
        {std::get<0>(hx), std::get<1>(hx)},
        flat_weights_,
        options.bias(),
        options.num_layers(),
        options.dropout(),
        this->is_training(),
        options.bidirectional());
  }
  auto output = std::get<0>(result);
  auto hidden = std::make_tuple(std::get<1>(result), std::get<2>(result));

  return std::make_tuple(output, hidden);
}

std::tuple<Tensor, std::tuple<Tensor, Tensor>> LSTMImpl::forward(
    const Tensor& input,
    torch::optional<std::tuple<Tensor, Tensor>> hx_opt) {
  auto batch_sizes = torch::Tensor();
  auto max_batch_size = options.batch_first() ? input.size(0) : input.size(1);
  auto sorted_indices = torch::Tensor();
  auto unsorted_indices = torch::Tensor();

  auto [output, hidden] = this->forward_helper(
      input, batch_sizes, sorted_indices, max_batch_size, std::move(hx_opt));

  return std::make_tuple(
      output, this->permute_hidden(hidden, unsorted_indices));
}

std::tuple<PackedSequence, std::tuple<Tensor, Tensor>> LSTMImpl::
    forward_with_packed_input(
        const PackedSequence& packed_input,
        torch::optional<std::tuple<Tensor, Tensor>> hx_opt) {
  const auto& input = packed_input.data();
  const auto& batch_sizes = packed_input.batch_sizes();
  const auto& sorted_indices = packed_input.sorted_indices();
  const auto& unsorted_indices = packed_input.unsorted_indices();
  auto max_batch_size = batch_sizes[0].item<int64_t>();

  auto [output, hidden] = this->forward_helper(
      input, batch_sizes, sorted_indices, max_batch_size, std::move(hx_opt));

  auto output_packed =
      PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices);
  return std::make_tuple(
      output_packed, this->permute_hidden(hidden, unsorted_indices));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GRU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GRUImpl::GRUImpl(const GRUOptions& options_)
    : detail::RNNImplBase<GRUImpl>(
          detail::RNNOptionsBase(
              torch::kGRU,
              options_.input_size(),
              options_.hidden_size())
              .num_layers(options_.num_layers())
              .bias(options_.bias())
              .batch_first(options_.batch_first())
              .dropout(options_.dropout())
              .bidirectional(options_.bidirectional())),
      options(options_) {}

std::tuple<Tensor, Tensor> GRUImpl::forward_helper(
    const Tensor& input,
    const Tensor& batch_sizes,
    const Tensor& sorted_indices,
    int64_t max_batch_size,
    Tensor hx) {
  if (!hx.defined()) {
    int64_t num_directions = options.bidirectional() ? 2 : 1;
    hx = torch::zeros(
        {options.num_layers() * num_directions,
         max_batch_size,
         options.hidden_size()},
        torch::dtype(input.dtype()).device(input.device()));
  } else {
    // Each batch of the hidden state should match the input sequence that
    // the user believes he/she is passing in.
    hx = this->permute_hidden(hx, sorted_indices);
  }

  this->check_forward_args(input, hx, batch_sizes);
  std::tuple<Tensor, Tensor> result;
  if (!batch_sizes.defined()) {
    result = torch::gru(
        input,
        hx,
        flat_weights_,
        options.bias(),
        options.num_layers(),
        options.dropout(),
        this->is_training(),
        options.bidirectional(),
        options.batch_first());
  } else {
    result = torch::gru(
        input,
        batch_sizes,
        hx,
        flat_weights_,
        options.bias(),
        options.num_layers(),
        options.dropout(),
        this->is_training(),
        options.bidirectional());
  }
  auto output = std::get<0>(result);
  auto hidden = std::get<1>(result);

  return std::make_tuple(output, hidden);
}

std::tuple<Tensor, Tensor> GRUImpl::forward(const Tensor& input, Tensor hx) {
  auto batch_sizes = torch::Tensor();
  auto max_batch_size = options.batch_first() ? input.size(0) : input.size(1);
  auto sorted_indices = torch::Tensor();
  auto unsorted_indices = torch::Tensor();

  auto [output, hidden] = this->forward_helper(
      input, batch_sizes, sorted_indices, max_batch_size, std::move(hx));

  return std::make_tuple(
      output, this->permute_hidden(hidden, unsorted_indices));
}

std::tuple<PackedSequence, Tensor> GRUImpl::forward_with_packed_input(
    const PackedSequence& packed_input,
    Tensor hx) {
  const auto& input = packed_input.data();
  const auto& batch_sizes = packed_input.batch_sizes();
  const auto& sorted_indices = packed_input.sorted_indices();
  const auto& unsorted_indices = packed_input.unsorted_indices();
  auto max_batch_size = batch_sizes[0].item<int64_t>();

  auto [output, hidden] = this->forward_helper(
      input, batch_sizes, sorted_indices, max_batch_size, std::move(hx));

  auto output_packed =
      PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices);
  return std::make_tuple(
      output_packed, this->permute_hidden(hidden, unsorted_indices));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ RNNCellImplBase
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace detail {
template <typename Derived>
RNNCellImplBase<Derived>::RNNCellImplBase(const RNNCellOptionsBase& options_)
    : options_base(options_) {
  // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
  reset();
}

template <typename Derived>
void RNNCellImplBase<Derived>::reset() {
  weight_ih = this->register_parameter(
      "weight_ih",
      torch::empty(
          {options_base.num_chunks() * options_base.hidden_size(),
           options_base.input_size()}));
  weight_hh = this->register_parameter(
      "weight_hh",
      torch::empty(
          {options_base.num_chunks() * options_base.hidden_size(),
           options_base.hidden_size()}));

  if (options_base.bias()) {
    bias_ih = this->register_parameter(
        "bias_ih",
        torch::empty({options_base.num_chunks() * options_base.hidden_size()}));
    bias_hh = this->register_parameter(
        "bias_hh",
        torch::empty({options_base.num_chunks() * options_base.hidden_size()}));
  } else {
    bias_ih =
        this->register_parameter("bias_ih", Tensor(), /*requires_grad=*/false);
    bias_hh =
        this->register_parameter("bias_hh", Tensor(), /*requires_grad=*/false);
  }

  reset_parameters();
}

template <typename Derived>
void RNNCellImplBase<Derived>::reset_parameters() {
  const double stdv = 1.0 / std::sqrt(options_base.hidden_size());
  for (auto& weight : this->parameters()) {
    init::uniform_(weight, -stdv, stdv);
  }
}

template <typename Derived>
void RNNCellImplBase<Derived>::pretty_print(std::ostream& stream) const {
  const std::string name = this->name();
  const std::string name_without_impl = name.substr(0, name.size() - 4);
  stream << name_without_impl << "(" << options_base.input_size() << ", "
         << options_base.hidden_size();
  if (!options_base.bias()) {
    stream << ", bias=" << std::boolalpha << false;
  }
  auto nonlinearity_str = this->get_nonlinearity_str();
  if (!nonlinearity_str.empty() && nonlinearity_str != "kTanh") {
    stream << ", nonlinearity=" << nonlinearity_str;
  }
  stream << ")";
}

template <typename Derived>
void RNNCellImplBase<Derived>::check_forward_input(
    const Tensor& input,
    const string& name) const {
  TORCH_CHECK(
      input.dim() == 1 || input.dim() == 2,
      "Expected ",
      name.c_str(),
      " to be 1D or 2D, got ",
      input.dim(),
      "D instead");
}

template <typename Derived>
std::string RNNCellImplBase<Derived>::get_nonlinearity_str() const {
  return "";
}

template class RNNCellImplBase<LSTMCellImpl>;
template class RNNCellImplBase<GRUCellImpl>;
template class RNNCellImplBase<RNNCellImpl>;
} // namespace detail

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ RNNCell
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RNNCellImpl::RNNCellImpl(const RNNCellOptions& options_)
    : detail::RNNCellImplBase<RNNCellImpl>(detail::RNNCellOptionsBase(
          options_.input_size(),
          options_.hidden_size(),
          options_.bias(),
          /*num_chunks=*/1)),
      options(options_) {}

Tensor RNNCellImpl::forward(const Tensor& input, Tensor hx) {
  this->check_forward_input(input, "input");
  this->check_forward_input(hx, "hidden");

  Tensor r_hx, ret;

  bool is_batched = input.dim() == 2;
  Tensor r_input = is_batched ? input : input.unsqueeze(0);

  if (!hx.defined()) {
    r_hx = torch::zeros(
        {input.size(0), options.hidden_size()},
        torch::dtype(input.dtype()).device(input.device()));
  } else {
    r_hx = is_batched ? hx : hx.unsqueeze(0);
  }

  if (std::holds_alternative<enumtype::kTanh>(options.nonlinearity())) {
    ret = torch::rnn_tanh_cell(
        r_input, r_hx, weight_ih, weight_hh, bias_ih, bias_hh);
  } else if (std::holds_alternative<enumtype::kReLU>(options.nonlinearity())) {
    ret = torch::rnn_relu_cell(
        r_input, r_hx, weight_ih, weight_hh, bias_ih, bias_hh);
  } else {
    TORCH_CHECK(
        false,
        "Unknown nonlinearity: ",
        torch::enumtype::get_enum_name(options.nonlinearity()));
  }

  if (!is_batched) {
    ret = ret.squeeze(0);
  }

  return ret;
}

std::string RNNCellImpl::get_nonlinearity_str() const {
  return get_enum_name(options.nonlinearity());
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LSTMCell
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LSTMCellImpl::LSTMCellImpl(const LSTMCellOptions& options_)
    : detail::RNNCellImplBase<LSTMCellImpl>(detail::RNNCellOptionsBase(
          options_.input_size(),
          options_.hidden_size(),
          options_.bias(),
          /*num_chunks=*/4)),
      options(options_) {}

std::tuple<Tensor, Tensor> LSTMCellImpl::forward(
    const Tensor& input,
    torch::optional<std::tuple<Tensor, Tensor>> hx_opt) {
  this->check_forward_input(input, "input");
  if (hx_opt.has_value()) {
    this->check_forward_input(std::get<0>(hx_opt.value()), "hx[0]");
    this->check_forward_input(std::get<1>(hx_opt.value()), "hx[1]");
  }

  std::tuple<Tensor, Tensor> r_hx, ret;

  bool is_batched = input.dim() == 2;
  Tensor r_input = is_batched ? input : input.unsqueeze(0);

  if (!hx_opt.has_value()) {
    auto zeros = torch::zeros(
        {input.size(0), options.hidden_size()},
        torch::dtype(input.dtype()).device(input.device()));
    r_hx = std::make_tuple(zeros, zeros);
  } else {
    if (!is_batched) {
      r_hx = std::make_tuple(
          std::get<0>(hx_opt.value()).unsqueeze(0),
          std::get<1>(hx_opt.value()).unsqueeze(0));
    } else {
      r_hx = hx_opt.value();
    }
  }

  ret = torch::lstm_cell(
      r_input,
      {std::get<0>(r_hx), std::get<1>(r_hx)},
      weight_ih,
      weight_hh,
      bias_ih,
      bias_hh);

  if (!is_batched) {
    ret = std::make_tuple(
        std::get<0>(ret).squeeze(0), std::get<1>(ret).squeeze(0));
  }

  return ret;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GRUCell
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GRUCellImpl::GRUCellImpl(const GRUCellOptions& options_)
    : detail::RNNCellImplBase<GRUCellImpl>(detail::RNNCellOptionsBase(
          options_.input_size(),
          options_.hidden_size(),
          options_.bias(),
          /*num_chunks=*/3)),
      options(options_) {}

Tensor GRUCellImpl::forward(const Tensor& input, Tensor hx) {
  this->check_forward_input(input, "input");
  this->check_forward_input(hx, "hidden");

  Tensor r_hx, ret;

  bool is_batched = input.dim() == 2;
  Tensor r_input = is_batched ? input : input.unsqueeze(0);

  if (!hx.defined()) {
    r_hx = torch::zeros(
        {input.size(0), options.hidden_size()},
        torch::dtype(input.dtype()).device(input.device()));
  } else {
    r_hx = is_batched ? hx : hx.unsqueeze(0);
  }

  ret = torch::gru_cell(r_input, r_hx, weight_ih, weight_hh, bias_ih, bias_hh);

  if (!is_batched) {
    ret = ret.squeeze(0);
  }

  return ret;
}

} // namespace nn
} // namespace torch
