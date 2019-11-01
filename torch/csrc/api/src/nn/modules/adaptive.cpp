#include <torch/nn/modules/adaptive.h>
#include <torch/nn/options/activation.h>
#include <torch/nn/options/linear.h>

namespace torch {
namespace nn {

ASMoutput::ASMoutput(const Tensor& output_, const double& loss_): output(output_), loss(loss_) {}

AdaptiveLogSoftmaxWithLossImpl::AdaptiveLogSoftmaxWithLossImpl(const AdaptiveLogSoftmaxWithLossOptions& options_)
    : options(options_) {
  TORCH_CHECK( std::is_sorted(options.cutoffs().begin(), options.cutoffs().end()) &&
          *std::min_element(options.cutoffs().begin(), options.cutoffs().end()) > 0 &&
          *std::max_element(options.cutoffs().begin(), options.cutoffs().end()) <= options.n_classes() &&
          std::set<int64_t>(options.cutoffs().begin(), options.cutoffs().end()).size() == options.cutoffs().size(),
          "cutoffs should be a sequence of unique, positive integers sorted in an increasing order, \
           where each value is between 1 and n_classes-1");

  cutoffs = options.cutoffs();
  cutoffs.push_back(options.n_classes());

  shortlist_size = cutoffs[0];
  n_clusters = cutoffs.size() - 1;
  head_size = shortlist_size + n_clusters;

  head = Linear(LinearOptions(options.in_features(), head_size).bias(options.head_bias()));
  tail = ModuleList();
  for (size_t i = 0; i < n_clusters; i++) {
    int64_t hsz = int64_t(options.in_features() / (std::pow(options.div_value(), (i + 1))));
    int64_t osz = cutoffs[i + 1] - cutoffs[i];

    Sequential projection(
        Linear(LinearOptions(options.in_features(), hsz).bias(false)),
        Linear(LinearOptions(hsz, osz).bias(false)));
    tail->push_back(projection);
  }
}

void AdaptiveLogSoftmaxWithLossImpl::reset() {
  head->reset();
  for (size_t i = 0; i < tail->size(); ++i) {
    auto modules = tail[i]->modules();
    for (size_t j = 0; j < modules.size(); ++j) {
      modules[j]->as<Linear>()->reset();
    }
  }
}

ASMoutput AdaptiveLogSoftmaxWithLossImpl::forward(const Tensor& input, const Tensor& target) {
  TORCH_CHECK( input.size(0) == target.size(0),
      "Input and target should have the same size in the batch dimension.");

  int64_t used_rows = 0;
  const int64_t batch_size = target.size(0);

  Tensor output = input.new_zeros(batch_size);
  Tensor gather_inds = target.new_empty(batch_size);

  auto cutoff_values = cutoffs;
  cutoff_values.insert(cutoff_values.begin(), 0);

  for (size_t i = 0; i < cutoff_values.size() - 1; ++i) {
    int64_t low_idx = cutoff_values[i];
    int64_t high_idx = cutoff_values[i + 1];

    const Tensor target_mask = (target >= low_idx) * (target < high_idx);
    const Tensor row_indices = target_mask.nonzero().squeeze();

  if (row_indices.numel() == 0) {
    continue;
  }

  if (i == 0) {
    gather_inds = gather_inds.index_copy(0, row_indices, target.index_select(0, row_indices));
  }
  else {
    Tensor relative_target = target.index_select(0, row_indices) - low_idx;
    Tensor input_subset = input.index_select(0, row_indices);

    const Tensor cluster_output = tail[i - 1]->as<Sequential>()->forward(input_subset);
    int64_t cluster_index = shortlist_size + i - 1;
    gather_inds.index_fill_(0, row_indices, cluster_index);

    const Tensor cluster_logprob = log_softmax(cluster_output, 1);
    const Tensor local_logprob = cluster_logprob.gather(1, relative_target.unsqueeze(1));
    output = output.index_copy_(0, row_indices, local_logprob.squeeze(1));
  }
    used_rows += row_indices.numel();
 }

  TORCH_CHECK(used_rows == batch_size, "Target values should be in [0, ", options.n_classes() - 1,"],\
   but values in range [",target.min().item().toFloat(),", ",target.max().item().toFloat(),"] were found. ");

  const Tensor head_output = head(input);
  const Tensor head_logprob = log_softmax(head_output, 1);
  output += head_logprob.gather(1, gather_inds.unsqueeze(1)).squeeze();
  const double loss = (-output).mean().item().toDouble();
  return ASMoutput(output, loss);
}

Tensor AdaptiveLogSoftmaxWithLossImpl::_get_full_log_prob(const Tensor& input, const Tensor& head_output) {
  Tensor out = input.new_empty({head_output.size(0), options.n_classes()});
  const Tensor head_logprob = log_softmax(head_output, 1);
  auto shotlisted_head_logprob = head_logprob.narrow(1, 0, shortlist_size);
  auto indices = torch::arange(0, shortlist_size, torch::kLong);
  out = out.index_copy_(1, indices, shotlisted_head_logprob);
  
  for (size_t i = 0; i < cutoffs.size() - 1; ++i) {
    int64_t start_idx = cutoffs[i];
    int64_t stop_idx = cutoffs[i+1];
    const Tensor cluster_output = tail[i]->as<Sequential>()->forward(input);
    const Tensor cluster_logprob = log_softmax(cluster_output, 1);
    auto output_logprob = cluster_logprob + head_logprob.narrow(1, shortlist_size + i, 1);
    out = out.index_copy_(1, torch::arange(start_idx, stop_idx, torch::kLong), output_logprob);
  }
  return out;
}

Tensor AdaptiveLogSoftmaxWithLossImpl::AdaptiveLogSoftmaxWithLossImpl::log_prob(const Tensor& input) {
  const Tensor head_output = head(input);
  return _get_full_log_prob(input, head_output);
}

Tensor  AdaptiveLogSoftmaxWithLossImpl::predict(const Tensor& input) {
  const Tensor head_output = head(input);
  Tensor output = torch::argmax(head_output, 1);
  auto not_in_shortlist = (output >= shortlist_size);
  auto all_in_shortlist = bitwise_not(not_in_shortlist.any());

  if(all_in_shortlist.item().toBool()) {
    return output;
  }
  else if (not_in_shortlist.all().item().toBool()) {
    const Tensor log_prob = _get_full_log_prob(input, head_output);
    return torch::argmax(log_prob, 1);
  }
  else {
    auto not_in_shortlist_idices = not_in_shortlist.nonzero().squeeze();
    const Tensor log_prob = _get_full_log_prob(
                        input.index_select(0, not_in_shortlist_idices),
                        head_output.index_select(0, not_in_shortlist_idices));
    output = output.index_copy_(0, not_in_shortlist.nonzero().squeeze(), torch::argmax(log_prob, 1));
    return output;
  }
}

void AdaptiveLogSoftmaxWithLossImpl::pretty_print(std::ostream& stream) const {
  //stream << "torch::nn::AdaptiveLogSoftmaxWithLoss()"; 
}

} // namespace nn
} // namespace torch