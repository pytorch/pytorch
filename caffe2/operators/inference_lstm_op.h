#ifndef LSTM_OP_H_
#define LSTM_OP_H_

#include <algorithm>
#include <sstream>
#include <unordered_map>
#include <vector>
#include "caffe2/core/blob_serialization.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"
#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"
#include "lstm_utils.h"

C10_DECLARE_CAFFE2_OPERATOR(LSTMOp);

namespace caffe2 {
namespace {

using t_tuple = std::tuple<Tensor, Tensor>;

struct CellParams {
  CellParams(
      const Tensor& _w_ih,
      const Tensor& _w_hh,
      const Tensor& _b_ih,
      const Tensor& _b_hh,
      CPUContext* _context) {
    initParams(_w_ih, _w_hh, _b_ih, _b_hh, _context);
  }

  CellParams(const CellParams& rhs) {
    initParams(rhs.w_ih, rhs.w_hh, rhs.b_ih, rhs.b_hh, rhs.context);
  }

  CellParams& operator=(const CellParams& rhs) {
    initParams(rhs.w_ih, rhs.w_hh, rhs.b_ih, rhs.b_hh, rhs.context);
    return *this;
  }

  void initParams(
      const Tensor& _w_ih,
      const Tensor& _w_hh,
      const Tensor& _b_ih,
      const Tensor& _b_hh,
      CPUContext* _context) {
    w_ih = copy_ctor(_w_ih);
    w_hh = copy_ctor(_w_hh);
    b_ih = copy_ctor(_b_ih);
    b_hh = copy_ctor(_b_hh);
    context = _context;
  }

  Tensor w_ih;
  Tensor w_hh;
  Tensor b_ih; /* optional */
  Tensor b_hh; /* optional */
  CPUContext* context;

  Tensor linear_ih(const Tensor& input) const {
    return linear(input, w_ih, b_ih, context);
  }
  Tensor linear_hh(const Tensor& h) const {
    return linear(h, w_hh, b_hh, context);
  }
};

struct LSTMCell {
  explicit LSTMCell(CPUContext* context) : context_(context) {}
  t_tuple operator()(
      const Tensor& input,
      const t_tuple& hidden,
      const CellParams& params) const {
    const auto& hx = std::get<0>(hidden);
    const auto& cx = std::get<1>(hidden);
    auto linear_ih = params.linear_ih(input);
    auto linear_hh = params.linear_hh(hx);
    auto gates = add(linear_ih, linear_hh, context_);
    auto chunked_gates = chunk(gates, 4, 1, context_);
    auto ingate = sigmoid(chunked_gates[0]);
    auto forgetgate = sigmoid(chunked_gates[1]);
    auto cellgate = tanh(chunked_gates[2], context_);
    auto outgate = sigmoid(chunked_gates[3]);

    auto cy =
        add(mul(forgetgate, cx, context_),
            mul(ingate, cellgate, context_),
            context_);
    auto hy = mul(outgate, tanh(cy, context_), context_);
    return std::make_tuple(std::move(hy), std::move(cy));
  }
  CPUContext* context_;
};

template <typename output_type, typename hidden_type>
struct LayerOutput {
  output_type outputs;
  hidden_type final_hidden;

  LayerOutput(const output_type& _outputs, const hidden_type& _hidden) {
    outputs = copy_ctor(_outputs);
    final_hidden = copy_ctor(_hidden);
  }
};

template <typename hidden_type, typename param_type>
struct Layer {
  using output_type = LayerOutput<Tensor, hidden_type>;
  virtual ~Layer() {}
  virtual output_type operator()(
      const Tensor& input,
      const hidden_type& input_hidden,
      const param_type& params) const = 0;
};

struct FullLSTMLayer : Layer<t_tuple, CellParams> {
  FullLSTMLayer(LSTMCell& cell, CPUContext* context)
      : cell_(cell), context_(context) {}

  LayerOutput<std::vector<Tensor>, t_tuple> operator()(
      const std::vector<Tensor>& step_inputs,
      const std::tuple<Tensor, Tensor>& input_hidden,
      const CellParams& params) const {
    std::vector<Tensor> step_outputs;
    auto hidden = copy_ctor(input_hidden);

    for (size_t i = 0; i < step_inputs.size(); i++) {
      hidden = cell_(step_inputs[i], hidden, params);
      step_outputs.push_back(copy_ctor(std::get<0>(hidden)));
    }

    return {step_outputs, hidden};
  }

  LayerOutput<Tensor, t_tuple> operator()(
      const Tensor& inputs,
      const std::tuple<Tensor, Tensor>& input_hidden,
      const CellParams& params) const override {
    auto unstacked_output =
        (*this)(unbind(inputs, 0, context_), input_hidden, params);
    return {stack(unstacked_output.outputs, 0, context_),
            unstacked_output.final_hidden};
  }
  LSTMCell cell_;
  CPUContext* context_;
};

struct FullBidirectionalLSTMLayer
    : Layer<std::pair<t_tuple, t_tuple>, std::pair<CellParams, CellParams>> {
  using bidir_hidden_type = std::pair<t_tuple, t_tuple>;
  using param_type = std::pair<CellParams, CellParams>;
  using output_type = LayerOutput<Tensor, bidir_hidden_type>;

  FullBidirectionalLSTMLayer(LSTMCell& cell, CPUContext* context)
      : layer_(cell, context), context_(context) {}

  output_type operator()(
      const Tensor& input,
      const bidir_hidden_type& input_hidden,
      const param_type& params) const override {
    std::vector<Tensor> outputs;
    auto step_inputs = unbind(input, 0, context_);
    auto fw_result = layer_(step_inputs, input_hidden.first, params.first);
    auto fw_output = stack(fw_result.outputs, 0, context_);
    outputs.push_back(copy_ctor(fw_output));
    auto rev_step_inputs = reverse(std::move(step_inputs));
    auto rev_result =
        layer_(rev_step_inputs, input_hidden.second, params.second);
    std::reverse(rev_result.outputs.begin(), rev_result.outputs.end());
    auto rev_output = stack(rev_result.outputs, 0, context_);
    outputs.push_back(copy_ctor(rev_output));
    return {cat(outputs, fw_output.dim() - 1, context_),
            std::make_pair(
                std::move(fw_result.final_hidden),
                std::move(rev_result.final_hidden))};
  }

  inline std::vector<Tensor> reverse(std::vector<Tensor>&& x) const {
    std::reverse(x.begin(), x.end());
    return std::move(x);
  }

 private:
  FullLSTMLayer layer_;
  CPUContext* context_;
};

template <typename hidden_type, typename weight_type>
LayerOutput<Tensor, std::vector<hidden_type>> apply_layer_stack(
    const Layer<hidden_type, weight_type>& layer,
    const Tensor& input,
    const std::vector<hidden_type>& hiddens,
    const std::vector<weight_type>& weights,
    int64_t num_layers) {
  CAFFE_ENFORCE(
      num_layers == hiddens.size(),
      "Expected more hidden states in stacked_rnn");
  CAFFE_ENFORCE(
      num_layers == weights.size(), "Expected more weights in stacked_rnn");

  auto layer_input = input.UnsafeSharedInstance();
  auto hidden_it = hiddens.begin();
  auto weight_it = weights.begin();
  std::vector<hidden_type> final_hiddens(num_layers);
  for (int64_t l = 0; l < num_layers; ++l) {
    auto layer_output = layer(layer_input, *(hidden_it++), *(weight_it++));
    final_hiddens.at(l) = std::move(layer_output.final_hidden);
    layer_input = std::move(layer_output.outputs);
  }
  return {layer_input, final_hiddens};
}

std::tuple<Tensor, Tensor, Tensor> _lstm_impl(
    const Tensor& input,
    const std::vector<CellParams>& params,
    const Tensor& hx,
    const Tensor& cx,
    int64_t num_layers,
    bool bidirectional,
    CPUContext* context) {
  using stack_output = LayerOutput<Tensor, std::vector<t_tuple>>;
  auto layer_hx = unbind(hx, 0, context);
  auto layer_cx = unbind(cx, 0, context);
  int64_t total_layers = layer_hx.size();
  std::vector<std::tuple<Tensor, Tensor>> hiddens;
  hiddens.reserve(total_layers);
  for (int64_t i = 0; i < total_layers; ++i) {
    hiddens.emplace_back(std::move(layer_hx[i]), std::move(layer_cx[i]));
  }
  LSTMCell cell(context);
  std::shared_ptr<stack_output> stack_output_ptr;
  if (bidirectional) {
    auto bidir_result = apply_layer_stack(
        FullBidirectionalLSTMLayer{cell, context},
        input,
        pair_vec(hiddens),
        pair_vec(params),
        num_layers);
    stack_output_ptr.reset(new stack_output(
        bidir_result.outputs,
        unpair_vec(std::move(bidir_result.final_hidden))));
  } else {
    auto result = apply_layer_stack(
        FullLSTMLayer{cell, context}, input, hiddens, params, num_layers);
    stack_output_ptr = std::make_shared<stack_output>(std::move(result));
  }

  std::vector<Tensor> hy, cy;
  hy.reserve(total_layers);
  cy.reserve(total_layers);
  for (auto& hidden : stack_output_ptr->final_hidden) {
    hy.push_back(std::move(std::get<0>(hidden)));
    cy.push_back(std::move(std::get<1>(hidden)));
  }
  return std::make_tuple(
      std::move(stack_output_ptr->outputs),
      stack(hy, 0, context),
      stack(cy, 0, context));
}

// Parses a flat list of parameter tensors into a list of CellParams
std::vector<CellParams> gather_params(
    const std::vector<Tensor>& params,
    bool has_biases,
    CPUContext* context) {
  Tensor undefined;
  std::vector<CellParams> result;
  if (has_biases) {
    CAFFE_ENFORCE_EQ(
        params.size() % 4, 0, "got an incorrect number of LSTM parameters");
    for (size_t i = 0; i < params.size(); i += 4) {
      result.emplace_back(
          params[i], params[i + 1], params[i + 2], params[i + 3], context);
    }
  } else {
    CAFFE_ENFORCE_EQ(
        params.size() % 2, 0, "got an incorrect number of LSTM parameters");
    for (size_t i = 0; i < params.size(); i += 2) {
      result.emplace_back(
          params[i], params[i + 1], undefined, undefined, context);
    }
  }
  return result;
}

class InferenceLSTMOp : public Operator<CPUContext> {
 public:
  template <class... Args>
  explicit InferenceLSTMOp(Args&&... args)
      : Operator(std::forward<Args>(args)...),
        num_layers_(this->template GetSingleArgument<int64_t>("num_layers", 1)),
        bidirectional_(
            this->template GetSingleArgument<bool>("bidirectional", false)),
        has_biases_(this->template GetSingleArgument<bool>("has_biases", true)),
        batch_first_(
            this->template GetSingleArgument<bool>("batch_first", false)) {}

  bool RunOnDevice() override;

 protected:
  int64_t num_layers_;
  bool bidirectional_;
  bool has_biases_;
  bool batch_first_;
};

} // namespace
} // namespace caffe2
#endif // LSTM_OP_H_
