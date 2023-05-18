#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/native/cpu/utils.h>
#include <ATen/native/Resize.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/nll_loss2d_backward_native.h>
#include <ATen/ops/nll_loss2d_forward.h>
#include <ATen/ops/nll_loss2d_forward_native.h>
#include <ATen/ops/nll_loss2d_native.h>
#include <ATen/ops/zeros_like.h>

#include <utility>
#endif

namespace at {
namespace native {

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
  return source.defined() ? source.data_ptr<scalar_t>() : nullptr;
}

inline void check_inputs_nll_loss2d(
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight) {
  TORCH_CHECK(
      target.dim() == 3,
      "only batches of spatial targets supported (3D tensors)"
      " but got targets of dimension: ",
      target.dim());
  TORCH_CHECK(
      input.dim() == 4,
      "only batches of spatial inputs supported (4D tensors), "
      "but got input of dimension: ",
      input.dim());
  TORCH_CHECK(
      !weight.defined() || weight.numel() == input.size(1),
      "weight tensor should be defined either for all or no classes");

  const int64_t input0 = input.size(0);
  const int64_t input2 = input.size(2);
  const int64_t input3 = input.size(3);
  const int64_t target0 = target.size(0);
  const int64_t target1 = target.size(1);
  const int64_t target2 = target.size(2);
  TORCH_CHECK(
      input0 == target0 && input2 == target1 && input3 == target2,
      "size mismatch (got input: ",
      input.sizes(),
      " , target: ",
      target.sizes());
}

inline void check_gradout_shape_nll_loss2d(
    const Tensor& grad_output,
    const Tensor& target) {
  TORCH_CHECK(
      grad_output.dim() == 3,
      "grad_output must have same dimension as target (3) but got dimension: ",
      grad_output.sizes());

  const int64_t grad_output0 = grad_output.size(0);
  const int64_t grad_output1 = grad_output.size(1);
  const int64_t grad_output2 = grad_output.size(2);
  const int64_t target0 = target.size(0);
  const int64_t target1 = target.size(1);
  const int64_t target2 = target.size(2);
  TORCH_CHECK(
      grad_output0 == target0 && grad_output1 == target1 &&
          grad_output2 == target2,
      "size mismatch (got grad_output: ",
      grad_output.sizes(),
      " target: ",
      target.sizes());
}


template <typename scalar_t>
static void nll_loss2d_forward_out_frame(
    Tensor& output,
    Tensor& total_weight,
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index) {
  const int64_t n_classes = input.size(1);

  scalar_t* total_weight_data = total_weight.data_ptr<scalar_t>();
  *total_weight_data = 0;

  auto weight_contiguous = optional_contiguous(weight);
  const scalar_t* weight_data = optional_data<scalar_t>(weight_contiguous);

  if (reduction == Reduction::None) {
    const int64_t batch_size = input.size(0);
    const int64_t H = input.size(2);
    const int64_t W = input.size(3);

    at::native::resize_output(output, {batch_size, H, W});
    auto input_acc = input.accessor<scalar_t, 4>();
    auto output_acc = output.accessor<scalar_t, 3>();
    auto target_acc = target.accessor<int64_t, 3>();

    at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
      for (const auto b : c10::irange(start, end)) {
        for (const auto h : c10::irange(H)) {
          for (const auto w : c10::irange(W)) {
            const int64_t cur_target = (int64_t)target_acc[b][h][w];

            if (cur_target == ignore_index) {
              output_acc[b][h][w] = static_cast<scalar_t>(0);
              continue;
            }

            TORCH_CHECK_INDEX(
                cur_target >= 0 && cur_target < n_classes,
                "Target ",
                cur_target,
                " is out of bounds.");

            // load optional weight value
            const scalar_t cur_weight = weight_data != nullptr
                ? weight_data[cur_target]
                : static_cast<scalar_t>(1);
            output_acc[b][h][w] = -input_acc[b][cur_target][h][w] * cur_weight;
          }
        }
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

  const scalar_t* input_data = input_contiguous.data_ptr<scalar_t>();
  const int64_t* target_data = target_contiguous.data_ptr<int64_t>();

  const int64_t batch_size = input.size(0);
  const int64_t map_size = input.size(2) * input.size(3);
  const int64_t sample_size = map_size * n_classes;
  const int64_t numiter = batch_size * map_size;

  constexpr int64_t cascade_sum_num_levels = 8;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  scalar_t weight_partial_sums[cascade_sum_num_levels] = {0};
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  scalar_t loss_partial_sums[cascade_sum_num_levels] = {0};
  const int64_t level_power =
      std::max(int64_t(4), utils::CeilLog2(numiter) / cascade_sum_num_levels);
  const int64_t level_step = (1 << level_power);
  const int64_t level_mask = level_step - 1;

  int64_t num_ignored = 0;
  for (const auto b : c10::irange(batch_size)) {
    for (const auto elem : c10::irange(map_size)) {
      const int64_t cur_target = target_data[b * map_size + elem];
      if (cur_target == ignore_index) {
        ++num_ignored;
        continue;
      }

      TORCH_CHECK_INDEX(
          cur_target >= 0 && cur_target < n_classes,
          "Target ",
          cur_target,
          " is out of bounds.");

      const auto data = input_data[b * sample_size + cur_target * map_size + elem];
      if (weight_data) {
        const scalar_t weight_val = weight_data[cur_target];
        loss_partial_sums[0] -= data * weight_val;
        weight_partial_sums[0] += weight_val;
      } else {
        loss_partial_sums[0] -= data;
      }

      const int64_t linear_idx = b * map_size + elem;
      for (int64_t j = 0; j + 1 < cascade_sum_num_levels; ++j) {
        const auto mask = (level_mask << (j * level_power));
        if (C10_LIKELY((linear_idx & mask) != 0)) {
          break;
        }

        weight_partial_sums[j + 1] += weight_partial_sums[j];
        loss_partial_sums[j + 1] += loss_partial_sums[j];

        weight_partial_sums[j] = 0;
        loss_partial_sums[j] = 0;
      }
    }
  }


  const scalar_t total_weight_val = !weight_data ?
    static_cast<scalar_t>(numiter - num_ignored) :
    std::accumulate(std::begin(weight_partial_sums),
                    std::end(weight_partial_sums),
                    scalar_t{0});

  scalar_t output_val = std::accumulate(std::begin(loss_partial_sums),
                                        std::end(loss_partial_sums),
                                        scalar_t{0});

  if (reduction == Reduction::Mean) {
    output_val /= total_weight_val;
  }

  *total_weight_data = total_weight_val;
  *output.data_ptr<scalar_t>() = output_val;
}

void nll_loss2d_forward_out_cpu_template(
    Tensor& output,
    Tensor& total_weight,
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index) {
  check_inputs_nll_loss2d(input, target, weight);
  total_weight.resize_({});

  AT_DISPATCH_FLOATING_TYPES_AND(
      ScalarType::BFloat16,
      input.scalar_type(),
      "nll_loss2d_forward_out_frame",
      [&] {
        nll_loss2d_forward_out_frame<scalar_t>(
            output,
            total_weight,
            input,
            target,
            weight,
            reduction,
            ignore_index);
      });
}

template <typename scalar_t>
static void nll_loss2d_backward_out_frame(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight) {
  auto weight_contiguous = optional_contiguous(weight);
  const scalar_t* weight_data = optional_data<scalar_t>(weight_contiguous);

  if (reduction == at::Reduction::None) {
    check_gradout_shape_nll_loss2d(grad_output, target);

    const int64_t batch_size = input.size(0);
    const int64_t H = input.size(2);
    const int64_t W = input.size(3);

    auto grad_input_acc = grad_input.accessor<scalar_t, 4>();
    auto grad_output_acc = grad_output.accessor<scalar_t, 3>();
    auto target_acc = target.accessor<int64_t, 3>();

    at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
      for (const auto b : c10::irange(start, end)) {
        for (const auto h : c10::irange(H)) {
          for (const auto w : c10::irange(W)) {
            const int64_t cur_target = target_acc[b][h][w];
            if (cur_target == ignore_index) {
              continue;
            }
            const scalar_t value =
                -(weight_data ? weight_data[cur_target]
                              : static_cast<scalar_t>(1));
            const scalar_t grad_output_value = grad_output_acc[b][h][w];
            grad_input_acc[b][cur_target][h][w] = value * grad_output_value;
          }
        }
      }
    });

    return;
  }

  const scalar_t total_weight_value = *total_weight.data_ptr<scalar_t>();

  TORCH_CHECK(
      grad_output.dim() <= 1 && grad_output.numel() == 1,
      "Expected a single element grad_output tensor, but got: ",
      grad_output.sizes());

  const scalar_t grad_output_value = *grad_output.data_ptr<scalar_t>();

  const auto target_contiguous = target.contiguous();
  const int64_t* target_data = target_contiguous.data_ptr<int64_t>();

  scalar_t* grad_input_data = grad_input.data_ptr<scalar_t>();

  const int64_t batch_size = input.size(0);
  const int64_t n_classes = input.size(1);
  const int64_t map_size = input.size(2) * input.size(3);
  const int64_t sample_size = map_size * n_classes;

  const auto grad = -(reduction == Reduction::Mean ? grad_output_value / total_weight_value
                                                   : grad_output_value);

  at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
    for (const auto b : c10::irange(start, end)) {
      for (const auto elem : c10::irange(map_size)) {
        const int64_t t = target_data[b * map_size + elem];

        if (t != ignore_index) {
          TORCH_CHECK_INDEX(t >= 0 && t < n_classes, "Target ", t, " is out of bounds.");

          const int64_t index = b * sample_size + t * map_size + elem;
          grad_input_data[index] = weight_data != nullptr ? weight_data[t] * grad
                                                          : grad;
        }
      }
    }
  });
}

void nll_loss2d_backward_out_cpu_template(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight) {
  check_inputs_nll_loss2d(input, target, weight);
  grad_input.resize_as_(input);
  grad_input.zero_();
  TORCH_CHECK(grad_input.is_contiguous(), "grad_input must be contiguous");
  TORCH_CHECK(
      total_weight.numel() == 1,
      "expected total_weight to be a single element tensor, got: ",
      total_weight.sizes(),
      " (",
      total_weight.numel(),
      " elements)");

  AT_DISPATCH_FLOATING_TYPES_AND(
      ScalarType::BFloat16,
      input.scalar_type(),
      "nll_loss2d_backward_out_frame",
      [&] {
        nll_loss2d_backward_out_frame<scalar_t>(
            grad_input,
            grad_output,
            input,
            target,
            weight,
            reduction,
            ignore_index,
            total_weight);
      });
}

} // namespace

std::tuple<Tensor&, Tensor&> nll_loss2d_forward_out_cpu(const Tensor& self,
    const Tensor& target, const c10::optional<Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index,
    Tensor& output,
    Tensor& total_weight) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  nll_loss2d_forward_out_cpu_template(
      output, total_weight, self, target, weight, reduction, ignore_index);
  return std::tuple<Tensor&, Tensor&>(output, total_weight);
}

std::tuple<Tensor, Tensor> nll_loss2d_forward_cpu(
    const Tensor& self,
    const Tensor& target, const c10::optional<Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  auto output = at::empty({0}, self.options());
  auto total_weight = at::empty({0}, self.options());
  at::native::nll_loss2d_forward_out_cpu(
      self, target, weight, reduction, ignore_index, output, total_weight);
  return std::make_tuple(output, total_weight);
}

Tensor& nll_loss2d_backward_out_cpu(const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target, const c10::optional<Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight,
    Tensor& grad_input) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  nll_loss2d_backward_out_cpu_template(
      grad_input,
      grad_output,
      self,
      target,
      weight,
      reduction,
      ignore_index,
      total_weight);
  return grad_input;
}

Tensor nll_loss2d_backward_cpu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target, const c10::optional<Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  auto grad_input = at::zeros_like(self);
  at::native::nll_loss2d_backward_out_cpu(
      grad_output,
      self,
      target,
      weight,
      reduction,
      ignore_index,
      total_weight,
      grad_input);
  return grad_input;
}

Tensor & nll_loss2d_out(const Tensor & self, const Tensor & target, const c10::optional<Tensor>& weight_opt, int64_t reduction, int64_t ignore_index, Tensor & output) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  Tensor total_weight = at::empty({0}, self.options());
  return std::get<0>(at::nll_loss2d_forward_out(output, total_weight, self, target, weight, reduction, ignore_index));
}

Tensor nll_loss2d_symint(const Tensor & self, const Tensor & target, const c10::optional<Tensor>& weight_opt, int64_t reduction, c10::SymInt ignore_index) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  return std::get<0>(at::nll_loss2d_forward_symint(self, target, weight, reduction, std::move(ignore_index)));
}

// Duplicate of above code for non-symbolic ints. Kept for BC purposes and to minimize breakages.
static Tensor nll_loss2d(const Tensor & self, const Tensor & target, const c10::optional<Tensor>& weight_opt, int64_t reduction, int64_t ignore_index) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  return std::get<0>(at::nll_loss2d_forward_symint(self, target, weight, reduction, ignore_index));
}

} // namespace native
} // namespace at
