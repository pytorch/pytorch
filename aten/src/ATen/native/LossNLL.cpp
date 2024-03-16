#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIndexing.h>
#include <ATen/TensorMeta.h>
#include <ATen/TensorOperators.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/cpu/utils.h>
#include <ATen/native/Resize.h>
#include <c10/util/SmallBuffer.h>
#include <ATen/TensorSubclassLikeUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/cross_entropy_loss_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/log_softmax.h>
#include <ATen/ops/nll_loss.h>
#include <ATen/ops/nll_loss2d.h>
#include <ATen/ops/nll_loss_backward_native.h>
#include <ATen/ops/nll_loss_forward.h>
#include <ATen/ops/nll_loss_forward_native.h>
#include <ATen/ops/nll_loss_native.h>
#include <ATen/ops/nll_loss_nd.h>
#include <ATen/ops/nll_loss_nd_native.h>
#endif

#include <c10/core/TensorOptions.h>
#include <c10/util/irange.h>

#include <utility>

namespace at::meta {
TORCH_META_FUNC(nll_loss_forward)
(const Tensor& self,
 const Tensor& target,
 const OptionalTensorRef weight_opt,
 int64_t reduction,
 int64_t ignore_index) {
  const Tensor& weight = weight_opt.getTensorRef();

  TORCH_CHECK(
      self.dim() > 0 && self.dim() <= 2, "input tensor should be 1D or 2D");
  TORCH_CHECK(
      target.dim() <= 1,
      "0D or 1D target tensor expected, multi-target not supported");

  auto no_batch_dim = self.dim() == 1  && target.dim() == 0;
  TORCH_CHECK(
      no_batch_dim || (self.size(0) == target.size(0)),
      "size mismatch (got input: ",
      self.sizes(),
      ", target: ",
      target.sizes(),
      ")")

  const auto n_classes = self.size(-1);

  TORCH_CHECK(
      !weight.defined() || (weight.dim() == 1 && weight.numel() == n_classes),
      "weight tensor should be defined either for all ",
      n_classes,
      " classes or no classes"
      " but got weight tensor of shape: ",
      weight.sizes());

  const auto n_dims = self.dim();
  const auto batch_size = self.size(0);

  if (reduction == Reduction::None && n_dims == 2) {
    set_output_raw_strided(0, {batch_size}, {}, self.options());
  } else {
    // produce scalar output when reducing or input is 1d
    set_output_raw_strided(0, {}, {}, self.options());
  }

  set_output_raw_strided(1, {}, {}, self.options());
}

TORCH_META_FUNC(nll_loss_backward)
(const Tensor& grad_output,
 const Tensor& self,
 const Tensor& target,
 OptionalTensorRef weight_opt,
 int64_t reduction,
 int64_t ignore_index,
 const Tensor& total_weight) {
  TORCH_CHECK(
      self.dim() > 0 && self.dim() <= 2, "input tensor should be 1D or 2D");
  TORCH_CHECK(
      target.dim() <= 1,
      "0D or 1D target tensor expected, multi-target not supported");

  auto no_batch_dim = self.dim() == 1  && target.dim() == 0;
  TORCH_CHECK(
      no_batch_dim || (self.size(0) == target.size(0)),
      "size mismatch (got input: ",
      self.sizes(),
      ", target: ",
      target.sizes(),
      ")")
  TORCH_CHECK(
      total_weight.numel() == 1,
      "expected total_weight to be a  single element tensor, got: ",
      total_weight.sizes(),
      " (",
      total_weight.numel(),
      " elements)");

  const auto& weight = weight_opt.getTensorRef();

  TORCH_CHECK(
      !weight.defined() || weight.numel() == self.size(-1),
      "weight tensor should be defined either for all or no classes");

  const auto n_dims = self.dim();

  if (reduction == Reduction::None && n_dims == 2) {
    const auto batch_size = self.size(0);
    check_dim_size(grad_output, 1, 0, batch_size);
  } else {
    TORCH_CHECK(
        grad_output.dim() <= 1 && grad_output.numel() == 1,
        "Expected a single element grad_output tensor, but got: ",
        grad_output.sizes());
  }

  set_output_raw_strided(0, self.sizes(), {}, self.options().memory_format(LEGACY_CONTIGUOUS_MEMORY_FORMAT));
}
} // namespace at::meta

namespace at::native {

namespace {

// Returns a contiguous tensor if the source tensor
// is defined. Otherwise returns the undefined
// source tensor unmodified.
inline Tensor optional_contiguous(const Tensor& source) {
  return source.defined() ? source.contiguous() : source;
}

// Returns the address of the first element of a tensor
// or nullptr if the tensor is undefined.
template <typename scalar_t>
inline scalar_t* optional_data(const Tensor& source) {
  if constexpr (std::is_const<scalar_t>::value) {
    return source.defined() ? source.const_data_ptr<scalar_t>() : nullptr;
  } else {
    return source.defined() ? source.data_ptr<scalar_t>() : nullptr;
  }
}

template <typename scalar_t, typename target_t>
static void nll_loss_out_frame(
    const Tensor& output,
    const Tensor& total_weight,
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index) {
  const auto n_dims = input.dim();
  const auto n_classes = input.size(-1);

  scalar_t* total_weight_data = total_weight.data_ptr<scalar_t>();
  *total_weight_data = 0;

  auto weight_contiguous = optional_contiguous(weight);
  const scalar_t* weight_data = optional_data<const scalar_t>(weight_contiguous);

  if (reduction == Reduction::None && n_dims == 2) {
    const auto batch_size = input.size(0);
    at::native::resize_output(output, {batch_size});

    auto input_acc = input.accessor<const scalar_t, 2>();
    auto target_acc = target.accessor<const target_t, 1>();
    auto output_acc = output.accessor<scalar_t, 1>();

    at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
      for (const auto i : c10::irange(start, end)) {
        const auto cur_target = target_acc[i];

        if (cur_target == ignore_index) {
          output_acc[i] = 0;
          continue;
        }

        TORCH_CHECK_INDEX(
            cur_target >= 0 && cur_target < n_classes,
            "Target ",
            cur_target,
            " is out of bounds.");

        scalar_t cur_weight = weight_data != nullptr ? weight_data[cur_target]
                                                     : static_cast<scalar_t>(1);
        output_acc[i] = -input_acc[i][cur_target] * cur_weight;
      }
    });

    return;
  }

  // produce scalar outputs for the reduction case
  at::native::resize_output(output, {});

  if (target.numel() == 0) {
    // Here target (and input) have zero elements
    // Mean reduction on empty tensors produces NaN. See the discussion in
    // https://github.com/pytorch/pytorch/pull/64572#issuecomment-926504162
    if (reduction == Reduction::Mean) {
      output.fill_(std::numeric_limits<double>::quiet_NaN());
    } else {
      output.zero_();
    }
    total_weight.zero_();
    return;
  }

  auto input_contiguous = input.contiguous();
  auto target_contiguous = target.contiguous();

  const scalar_t* input_data = input_contiguous.const_data_ptr<scalar_t>();
  const target_t* target_data = target_contiguous.const_data_ptr<target_t>();

  const int64_t ndim = input.dim();
  const int64_t batch_size = ndim == 1 ? 1 : input.size(0);

  constexpr int64_t cascade_sum_num_levels = 8;
  const int64_t level_power =
      std::max(int64_t(4), utils::CeilLog2(batch_size) / cascade_sum_num_levels);
  const int64_t level_step = (1 << level_power);
  const int64_t level_mask = level_step - 1;

  int64_t num_ignored = 0;

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  scalar_t weight_partial_sums[cascade_sum_num_levels] = {0};
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  scalar_t loss_partial_sums[cascade_sum_num_levels] = {0};
  for (const auto b : c10::irange(batch_size)) {
    const int64_t cur_target = target_data[b];
    if (cur_target == ignore_index) {
      ++num_ignored;
      continue;
    }

    TORCH_CHECK_INDEX(
        cur_target >= 0 && cur_target < n_classes,
        "Target ",
        cur_target,
        " is out of bounds.");

    const auto data = input_data[b * n_classes + cur_target];
    if (weight_data) {
      const scalar_t weight_val = weight_data[cur_target];
      loss_partial_sums[0] -= data * weight_val;
      weight_partial_sums[0] += weight_val;
    } else {
      loss_partial_sums[0] -= data;
    }

    for (int64_t j = 0; j + 1 < cascade_sum_num_levels; ++j) {
      const auto mask = (level_mask << (j * level_power));
      if (C10_LIKELY((b & mask) != 0)) {
        break;
      }

      weight_partial_sums[j + 1] += weight_partial_sums[j];
      loss_partial_sums[j + 1] += loss_partial_sums[j];

      weight_partial_sums[j] = 0;
      loss_partial_sums[j] = 0;
    }
  }

  const scalar_t total_weight_val = !weight_data ?
    static_cast<scalar_t>(batch_size - num_ignored) :
    std::accumulate(std::begin(weight_partial_sums),
                    std::end(weight_partial_sums),
                    scalar_t{0});

  scalar_t output_val = std::accumulate(std::begin(loss_partial_sums),
                                        std::end(loss_partial_sums),
                                        scalar_t{0});

  if (reduction == Reduction::Mean) {
    output_val /= total_weight_val;
  }

  // write result to output tensors
  *output.data_ptr<scalar_t>() = output_val;
  *total_weight_data = total_weight_val;
}

void nll_loss_forward_out_cpu_template(
    const Tensor& output,
    const Tensor& total_weight,
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index) {
  AT_DISPATCH_FLOATING_TYPES_AND(
      ScalarType::BFloat16, input.scalar_type(), "nll_loss_out_frame", [&] {
        if (target.scalar_type() == kByte) {
          nll_loss_out_frame<scalar_t, uint8_t>(
              output,
              total_weight,
              input,
              target,
              weight,
              reduction,
              ignore_index);
        } else {
          // assumed to be int64
          nll_loss_out_frame<scalar_t, int64_t>(
              output,
              total_weight,
              input,
              target,
              weight,
              reduction,
              ignore_index);
        }
      });
}

template <typename scalar_t, typename target_t>
static void nll_loss_backward_out_frame(
    const Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight) {
  const auto n_dims = input.dim();
  const auto n_classes = input.size(-1);

  auto target_ = target;
  if (target.dim() == 0) {
    target_ = target.unsqueeze(0);
  }
  auto target_acc = target_.accessor<target_t, 1>();

  auto weight_contiguous = optional_contiguous(weight);
  const scalar_t* weight_data = optional_data<scalar_t>(weight_contiguous);

  if (reduction == Reduction::None && n_dims == 2) {
    const auto batch_size = input.size(0);
    auto grad_input_acc = grad_input.accessor<scalar_t, 2>();
    auto grad_output_acc = grad_output.accessor<scalar_t, 1>();
    at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
      for (const auto i : c10::irange(start, end)) {
        auto cur_target = target_acc[i];
        if (cur_target == ignore_index) {
          continue;
        }
        const scalar_t w =
            weight_data ? weight_data[cur_target] : static_cast<scalar_t>(1);
        grad_input_acc[i][cur_target] = -w * grad_output_acc[i];
      }
    });
    return;
  }

  const scalar_t total_weight_value = *total_weight.data_ptr<scalar_t>();

  const scalar_t grad_output_value = *grad_output.data_ptr<scalar_t>();

  if (input.dim() == 1) {
    auto grad_input_acc = grad_input.accessor<scalar_t, 1>();

    const auto t = target_acc[0];
    if (t != ignore_index) {
      TORCH_CHECK_INDEX(t >= 0 && t < n_classes, "Target ", t, " is out of bounds.");
      const auto grad = -(reduction == Reduction::Mean ? grad_output_value / total_weight_value
                                                       : grad_output_value);
      grad_input_acc[t] = weight_data != nullptr ? weight_data[t] * grad
                                                 : grad;
    }
  } else if (input.dim() == 2) {
    auto grad_input_acc = grad_input.accessor<scalar_t, 2>();
    const auto grad = -(reduction == Reduction::Mean ? grad_output_value / total_weight_value
                                                     : grad_output_value);

    const auto batch_size = input.size(0);

    at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
      for (const auto i : c10::irange(start, end)) {
        const auto t = target_acc[i];
        if (t != ignore_index) {
          TORCH_CHECK_INDEX(t >= 0 && t < n_classes, "Target ", t, " is out of bounds.");
          grad_input_acc[i][t] = weight_data != nullptr ? weight_data[t] * grad
                                                        : grad;
        }
      }
    });
  }
}

void nll_loss_backward_out_cpu_template(
    const Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight) {
  grad_input.zero_();

  AT_DISPATCH_FLOATING_TYPES_AND(
      ScalarType::BFloat16,
      input.scalar_type(),
      "nll_loss_backward_out_frame",
      [&] {
        if (target.scalar_type() == kByte) {
          nll_loss_backward_out_frame<scalar_t, uint8_t>(
              grad_input,
              grad_output,
              input,
              target,
              weight,
              reduction,
              ignore_index,
              total_weight);
        } else {
          // assumed to be uint64
          nll_loss_backward_out_frame<scalar_t, int64_t>(
              grad_input,
              grad_output,
              input,
              target,
              weight,
              reduction,
              ignore_index,
              total_weight);
        }
      });
}

} // namespace

TORCH_IMPL_FUNC(nll_loss_forward_out_cpu)
(const Tensor& self,
 const Tensor& target,
 const OptionalTensorRef weight_opt,
 int64_t reduction,
 int64_t ignore_index,
 const Tensor& output,
 const Tensor& total_weight) {
  const Tensor& weight = weight_opt.getTensorRef();
  nll_loss_forward_out_cpu_template(
      output, total_weight, self, target, weight, reduction, ignore_index);
}

TORCH_IMPL_FUNC(nll_loss_backward_out_cpu)
(const Tensor& grad_output,
 const Tensor& self,
 const Tensor& target,
 OptionalTensorRef weight_opt,
 int64_t reduction,
 int64_t ignore_index,
 const Tensor& total_weight,
 const Tensor& grad_input
) {
  const Tensor& weight = weight_opt.getTensorRef();
  nll_loss_backward_out_cpu_template(
      grad_input,
      grad_output,
      self,
      target,
      weight,
      reduction,
      ignore_index,
      total_weight);
}

static Tensor cross_entropy_loss_prob_target(
    const Tensor& self,
    const Tensor& target_,
    const Tensor& weight,
    int64_t reduction,
    double label_smoothing) {
  const auto class_dim = self.dim() == 1 ? 0 : 1;
  const auto n_classes = self.size(class_dim);
  TORCH_CHECK(
      !weight.defined() || (weight.dim() == 1 && weight.numel() == n_classes),
      "cross_entropy: weight tensor should be defined either for all ",
      n_classes,
      " classes or no classes"
      " but got weight tensor of shape: ",
      weight.sizes());

  auto input = at::log_softmax(self, class_dim, self.scalar_type());
  Tensor target;

  if (label_smoothing > 0.0) {
    TORCH_CHECK(label_smoothing <= 1.0, "label_smoothing must be between 0.0 and 1.0. Got: ", label_smoothing);
    target = target_ * (1 - label_smoothing) + label_smoothing / n_classes;
  } else {
    target = target_;
  }

  if (weight.defined()) {
    // Expand weight to the correct number of dims for broadcasting with input / target
    Tensor weight_ = weight;
    if (input.dim() > 1) {
        auto weight_broadcast_shape = SmallBuffer<int64_t, 5>(input.dim());
        std::fill(weight_broadcast_shape.begin(), weight_broadcast_shape.end(), 1);
        weight_broadcast_shape[1] = weight.size(0);
        weight_ = weight.view(weight_broadcast_shape);
    }

    switch (reduction) {
      case Reduction::Mean:
        if (input.numel()==0){
          return -(input * target * weight_).sum().fill_(std::numeric_limits<double>::quiet_NaN());
        } else {
          return -(input * target * weight_).sum() / (input.numel() / n_classes);
        }
      case Reduction::Sum:
        return -(input * target * weight_).sum();
      case Reduction::None:
        return -(input * target * weight_).sum(class_dim);
      default:
        TORCH_CHECK(false, "Invalid reduction type encountered in cross_entropy: ", reduction);
    }
  } else {
    switch (reduction) {
      case Reduction::Mean:
        if (input.numel()==0){
          return -(input * target).sum().fill_(std::numeric_limits<double>::quiet_NaN());
        } else {
          return -(input * target).sum() / (input.numel() / n_classes);
        }
      case Reduction::Sum:
        return -(input * target).sum();
      case Reduction::None:
        return -(input * target).sum(class_dim);
      default:
        TORCH_CHECK(false, "Invalid reduction type encountered in cross_entropy: ", reduction);
    }
  }
}

static Tensor cross_entropy_loss_label_smoothing(
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    c10::SymInt ignore_index,
    double label_smoothing) {
    auto class_dim = self.dim() == 1 ? 0 : 1;
    auto input = at::log_softmax(self, class_dim, self.scalar_type());
    auto nllloss = at::nll_loss_nd_symint(input, target, weight, reduction, ignore_index);

    auto n_classes = input.sym_size(class_dim);

    Tensor smooth_loss;
    if (weight.defined()) {
      // Expand weight to the correct number of dims for broadcasting with input / target
      auto weight_broadcast_shape = SmallBuffer<int64_t, 5>(input.dim());
      std::fill(weight_broadcast_shape.begin(), weight_broadcast_shape.end(), 1);
      weight_broadcast_shape[class_dim] = weight.size(0);
      Tensor weight_ = weight.view(weight_broadcast_shape);

      smooth_loss = -(input * weight_).sum(class_dim);
    } else {
      smooth_loss = -input.sum(class_dim);
    }

    auto ignore_mask = target == std::move(ignore_index);
    smooth_loss.masked_fill_(ignore_mask, 0.0);

    Tensor ret;
    switch (reduction) {
      case Reduction::Mean:
        if (weight.defined()) {
          if (isTensorSubclassLike(weight)){
            // we will collect weights from 0 index which is always valid
            // and mask them out if they are ignored
            auto filtered_target = target.masked_fill(ignore_mask, 0);
            auto tgt_weights = weight.gather(0, filtered_target.flatten());
            auto weight_sum =
                tgt_weights.masked_fill_(ignore_mask.flatten(), 0).sum();
            ret = smooth_loss.sum() / weight_sum;
          } else {
            // TODO: This code can path can be removed if #61309 is resolved
            // loss is normalized by the weights to be consistent with
            // nll_loss_nd
            ret = smooth_loss.sum() /
                weight.gather(0, target.masked_select(~ignore_mask).flatten())
                    .sum();
          }
        } else {
          auto true_mask = ~ignore_mask;
          ret = smooth_loss.sum()/ true_mask.sum();
        }
        break;
      case Reduction::Sum:
        ret = smooth_loss.sum();
        break;
      case Reduction::None:
        ret = smooth_loss;
        break;
      default:
        TORCH_CHECK(false, "Invalid reduction type encountered in cross_entropy: ", reduction);
    }
    return (1 - label_smoothing) * nllloss + ret * (label_smoothing / n_classes);
}

Tensor cross_entropy_loss_symint(
    const Tensor& self,
    const Tensor& target,
    const c10::optional<Tensor>& weight,
    int64_t reduction,
    c10::SymInt ignore_index,
    double label_smoothing) {
  Tensor ret;
  if (self.sym_sizes() == target.sym_sizes()) {
    // Assume soft targets when input and target shapes are the same
    TORCH_CHECK(at::isFloatingType(target.scalar_type()),
        "Expected floating point type for target with class probabilities, got ", target.scalar_type());
    TORCH_CHECK(ignore_index < 0, "ignore_index is not supported for floating point target");

    // See [Note: hacky wrapper removal for optional tensor]
    c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight);
    const Tensor& weight_ = *weight_maybe_owned;
    ret = cross_entropy_loss_prob_target(self, target, weight_, reduction, label_smoothing);
  } else if (label_smoothing > 0.0) {
    TORCH_CHECK(label_smoothing <= 1.0, "label_smoothing must be between 0.0 and 1.0. Got: ", label_smoothing);

    // See [Note: hacky wrapper removal for optional tensor]
    c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight);
    const Tensor& weight_ = *weight_maybe_owned;
    ret = cross_entropy_loss_label_smoothing(self, target, weight_, reduction, std::move(ignore_index), label_smoothing);
  } else {
    auto class_dim = self.dim() == 1 ? 0 : 1;
    ret = at::nll_loss_nd_symint(
        at::log_softmax(self, class_dim, self.scalar_type()),
        target,
        weight,
        reduction,
        std::move(ignore_index));
  }
  return ret;
}

Tensor & nll_loss_out(const Tensor & self, const Tensor & target, const c10::optional<Tensor>& weight_opt, int64_t reduction, int64_t ignore_index, Tensor & output) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  Tensor total_weight = at::empty({0}, self.options());
  return std::get<0>(at::nll_loss_forward_out(output, total_weight, self, target, weight, reduction, ignore_index));
}

Tensor nll_loss_symint(const Tensor & self, const Tensor & target, const c10::optional<Tensor>& weight_opt, int64_t reduction, c10::SymInt ignore_index) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  return std::get<0>(at::nll_loss_forward_symint(self, target, weight, reduction, std::move(ignore_index)));
}

// Duplicate of above code for non-symbolic ints. Kept for BC purposes and to minimize breakages.
static Tensor nll_loss(const Tensor & self, const Tensor & target, const c10::optional<Tensor>& weight_opt, int64_t reduction, int64_t ignore_index) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  return std::get<0>(at::nll_loss_forward_symint(self, target, weight, reduction, ignore_index));
}

Tensor nll_loss_nd_symint(
    const Tensor& self,
    const Tensor& target,
    const c10::optional<Tensor>& weight,
    int64_t reduction,
    c10::SymInt ignore_index) {
  if (self.dim() < 1) {
    TORCH_CHECK_VALUE(
        false, "Expected 1 or more dimensions (got ", self.dim(), ")");
  }

  if (self.dim() != 1 && self.sym_sizes()[0] != target.sym_sizes()[0]) {
    TORCH_CHECK_VALUE(
        false,
        "Expected input batch_size (",
        self.sym_sizes()[0],
        ") to match target batch_size (",
        target.sym_sizes()[0],
        ").");
  }

  Tensor ret;
  Tensor input_ = self;
  Tensor target_ = target;
  if (input_.dim() == 1 || input_.dim() == 2) {
    ret = at::nll_loss_symint(input_, target_, weight, reduction, std::move(ignore_index));
  } else if (input_.dim() == 4) {
    ret = at::nll_loss2d_symint(input_, target_, weight, reduction, std::move(ignore_index));
  } else {
    // dim == 3 or dim > 4
    auto n = input_.sym_sizes()[0];
    auto c = input_.sym_sizes()[1];
    auto out_size = input_.sym_sizes().slice(2).vec();
    out_size.insert(out_size.begin(), n);
    if (target_.sym_sizes().slice(1) != input_.sym_sizes().slice(2)) {
      TORCH_CHECK(
          false,
          "Expected target size ",
          SymIntArrayRef(out_size),
          ", got ",
          target_.sym_sizes());
    }
    input_ = input_.contiguous();
    target_ = target_.contiguous();
    // support empty batches, see #15870
    if (input_.sym_numel() > 0) {
      input_ = input_.view_symint({n, std::move(c), 1, -1});
    } else {
      input_ = input_.view_symint({n, std::move(c), 0, 0});
    }
    if (target_.sym_numel() > 0) {
      target_ = target_.view_symint({std::move(n), 1, -1});
    } else {
      target_ = target_.view_symint({std::move(n), 0, 0});
    }
    if (reduction != Reduction::None) {
      ret = at::nll_loss2d_symint(input_, target_, weight, reduction, std::move(ignore_index));
    } else {
      auto out =
          at::nll_loss2d_symint(input_, target_, weight, reduction, std::move(ignore_index));
      ret = out.view_symint(out_size);
    }
  }
  return ret;
}

} // namespace at::native
