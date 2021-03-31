#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorUtils.h>

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

template <typename scalar_t>
static void nll_loss_out_frame(
    Tensor& output,
    Tensor& total_weight,
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
  const scalar_t* weight_data = optional_data<scalar_t>(weight_contiguous);

  if (reduction == Reduction::None && n_dims == 2) {
    const auto batch_size = input.size(0);
    output.resize_({batch_size});

    auto input_acc = input.accessor<scalar_t, 2>();
    auto target_acc = target.accessor<int64_t, 1>();
    auto output_acc = output.accessor<scalar_t, 1>();

    at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
      for (auto i = start; i < end; i++) {
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

  // produce scalar output when reducing or input is 1d
  output.resize_({});

  auto input_contiguous = input.contiguous();
  auto target_contiguous = target.contiguous();

  const scalar_t* input_data = input_contiguous.data_ptr<scalar_t>();
  const int64_t* target_data = target_contiguous.data_ptr<int64_t>();

  scalar_t output_val = 0;
  scalar_t total_weight_val = 0;

  if (input.dim() == 1) {
    const auto cur_target = target_data[0];
    if (cur_target != ignore_index) {
      TORCH_CHECK_INDEX(
          cur_target >= 0 && cur_target < n_classes,
          "Target ",
          cur_target,
          " is out of bounds.");
      total_weight_val =
          weight_data ? weight_data[cur_target] : static_cast<scalar_t>(1);
      output_val = -input_data[cur_target] * total_weight_val;
    }
  } else if (input.dim() == 2) {
    const auto batch_size = input.size(0);
    TORCH_CHECK(target.size(0) == batch_size);
    const auto n_target = input.size(1);

    for (int64_t i = 0; i < batch_size; i++) {
      const auto cur_target = target_data[i];
      if (cur_target != ignore_index) {
        TORCH_CHECK_INDEX(
            cur_target >= 0 && cur_target < n_classes,
            "Target ",
            cur_target,
            " is out of bounds.");

        scalar_t cur_weight =
            weight_data ? weight_data[cur_target] : static_cast<scalar_t>(1);
        total_weight_val += cur_weight;
        output_val -= input_data[i * n_target + cur_target] * cur_weight;
      }
    }
  }

  if (reduction == Reduction::Mean &&
      (total_weight_val != 0 || input.numel() == 0)) {
    // allow NaN result for total_weight_val == 0 case, see #15870
    output_val /= total_weight_val;
  }

  // write result to output tensors
  *output.data_ptr<scalar_t>() = output_val;
  *total_weight_data = total_weight_val;
}

void nll_loss_forward_out_cpu_template(
    Tensor& output,
    Tensor& total_weight,
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index) {
  TORCH_CHECK(
      input.dim() > 0 && input.dim() <= 2, "input tensor should be 1D or 2D");
  TORCH_CHECK(
      target.dim() == 1,
      "1D target tensor expected, multi-target not supported");
  TORCH_CHECK(
      input.size(0) == target.size(0),
      "size mismatch (got input: ",
      input.sizes(),
      ", target: ",
      target.sizes(),
      ")")

  const auto n_classes = input.size(-1);

  TORCH_CHECK(
      !weight.defined() || weight.numel() == n_classes,
      "weight tensor should be defined either for all ",
      n_classes,
      " classes or no classes"
      " but got weight tensor of shape: ",
      weight.sizes());

  total_weight.resize_({});

  AT_DISPATCH_FLOATING_TYPES_AND(
      ScalarType::BFloat16, input.scalar_type(), "nll_loss_out_frame", [&] {
        nll_loss_out_frame<scalar_t>(
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
static void nll_loss_backward_out_frame(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight) {
  const auto n_dims = input.dim();
  const auto n_classes = input.size(-1);

  auto target_acc = target.accessor<int64_t, 1>();

  auto weight_contiguous = optional_contiguous(weight);
  const scalar_t* weight_data = optional_data<scalar_t>(weight_contiguous);

  if (reduction == Reduction::None && n_dims == 2) {
    const auto batch_size = input.size(0);
    check_dim_size(grad_output, 1, 0, batch_size);
    auto grad_input_acc = grad_input.accessor<scalar_t, 2>();
    auto grad_output_acc = grad_output.accessor<scalar_t, 1>();
    at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
      for (auto i = start; i < end; i++) {
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
  if (total_weight_value <= 0) {
    return;
  }

  TORCH_CHECK(
      grad_output.dim() <= 1 && grad_output.numel() == 1,
      "Expected a single element grad_output tensor, but got: ",
      grad_output.sizes());
  const scalar_t grad_output_value = *grad_output.data_ptr<scalar_t>();

  if (input.dim() == 1) {
    auto grad_input_acc = grad_input.accessor<scalar_t, 1>();

    const auto cur_target = target_acc[0];
    if (cur_target != ignore_index) {
      TORCH_CHECK_INDEX(
          cur_target >= 0 && cur_target < n_classes,
          "Target ",
          cur_target,
          " is out of bounds.");

      grad_input_acc[cur_target] =
          (reduction != Reduction::Mean && weight_data != nullptr)
          ? -weight_data[cur_target]
          : static_cast<scalar_t>(-1);
      grad_input_acc[cur_target] *= grad_output_value;
    }
  } else if (input.dim() == 2) {
    auto grad_input_acc = grad_input.accessor<scalar_t, 2>();

    const auto batch_size = input.size(0);
    TORCH_CHECK(target.size(0) == batch_size);

    for (int64_t i = 0; i < batch_size; i++) {
      const auto cur_target = target_acc[i];

      if (cur_target != ignore_index) {
        TORCH_CHECK_INDEX(
            cur_target >= 0 && cur_target < n_classes,
            "Target ",
            cur_target,
            " is out of bounds.");

        const scalar_t w = weight_data != nullptr ? weight_data[cur_target]
                                                  : static_cast<scalar_t>(1);
        grad_input_acc[i][cur_target] = -w * grad_output_value;

        if (reduction == Reduction::Mean) {
          grad_input_acc[i][cur_target] /= total_weight_value;
        }
      }
    }
  }
}

void nll_loss_backward_out_cpu_template(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight) {
  TORCH_CHECK(
      input.dim() > 0 && input.dim() <= 2, "input tensor should be 1D or 2D");
  TORCH_CHECK(
      target.dim() == 1,
      "1D target tensor expected, multi-target not supported");
  TORCH_CHECK(
      input.size(0) == target.size(0),
      "size mismatch (got input: ",
      input.sizes(),
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

  grad_input.resize_as_(input);
  grad_input.zero_();

  TORCH_CHECK(grad_input.is_contiguous(), "grad_input must be contiguous");
  TORCH_CHECK(
      !weight.defined() || weight.numel() == input.size(-1),
      "weight tensor should be defined either for all or no classes");

  AT_DISPATCH_FLOATING_TYPES_AND(
      ScalarType::BFloat16,
      input.scalar_type(),
      "nll_loss_backward_out_frame",
      [&] {
        nll_loss_backward_out_frame<scalar_t>(
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

std::tuple<Tensor&, Tensor&> nll_loss_forward_out_cpu(const Tensor& self,
    const Tensor& target, const c10::optional<Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index,
    Tensor& output,
    Tensor& total_weight) {
  // See [Note: hacky wrapper removal for optional tensor]
  const Tensor& weight = c10::value_or_else(weight_opt, [] {return Tensor();});

  nll_loss_forward_out_cpu_template(
      output, total_weight, self, target, weight, reduction, ignore_index);
  return std::tuple<Tensor&, Tensor&>(output, total_weight);
}

std::tuple<Tensor, Tensor> nll_loss_forward_cpu(
    const Tensor& self,
    const Tensor& target, const c10::optional<Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index) {
  // See [Note: hacky wrapper removal for optional tensor]
  const Tensor& weight = c10::value_or_else(weight_opt, [] {return Tensor();});

  auto output = at::empty({0}, self.options());
  auto total_weight = at::empty({0}, self.options());
  at::native::nll_loss_forward_out_cpu(
      self, target, weight, reduction, ignore_index, output, total_weight);
  return std::make_tuple(output, total_weight);
}

Tensor& nll_loss_backward_out_cpu(const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target, const c10::optional<Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight,
    Tensor& grad_input) {
  // See [Note: hacky wrapper removal for optional tensor]
  const Tensor& weight = c10::value_or_else(weight_opt, [] {return Tensor();});

  nll_loss_backward_out_cpu_template(
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

Tensor nll_loss_backward_cpu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target, const c10::optional<Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight) {
  // See [Note: hacky wrapper removal for optional tensor]
  const Tensor& weight = c10::value_or_else(weight_opt, [] {return Tensor();});

  auto grad_input = at::zeros_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  at::native::nll_loss_backward_out_cpu(
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

Tensor cross_entropy_loss(
    const Tensor& self,
    const Tensor& target,
    const c10::optional<Tensor>& weight,
    int64_t reduction,
    int64_t ignore_index) {
  return at::nll_loss_nd(
      at::log_softmax(
          self, 1, optTypeMetaToScalarType(self.options().dtype_opt())),
      target,
      weight,
      reduction,
      ignore_index);
}

Tensor & nll_loss_out(const Tensor & self, const Tensor & target, const c10::optional<Tensor>& weight_opt, int64_t reduction, int64_t ignore_index, Tensor & output) {
  // See [Note: hacky wrapper removal for optional tensor]
  const Tensor& weight = c10::value_or_else(weight_opt, [] {return Tensor();});

  Tensor total_weight = at::empty({0}, self.options());
  return std::get<0>(at::nll_loss_forward_out(output, total_weight, self, target, weight, reduction, ignore_index));
}

Tensor nll_loss(const Tensor & self, const Tensor & target, const c10::optional<Tensor>& weight_opt, int64_t reduction, int64_t ignore_index) {
  // See [Note: hacky wrapper removal for optional tensor]
  const Tensor& weight = c10::value_or_else(weight_opt, [] {return Tensor();});

  return std::get<0>(at::nll_loss_forward(self, target, weight, reduction, ignore_index));
}

Tensor nll_loss_nd(
    const Tensor& self,
    const Tensor& target,
    const c10::optional<Tensor>& weight,
    int64_t reduction,
    int64_t ignore_index) {
  if (self.dim() < 2) {
    TORCH_CHECK_VALUE(
        false, "Expected 2 or more dimensions (got ", self.dim(), ")");
  }

  if (self.sizes()[0] != target.sizes()[0]) {
    TORCH_CHECK_VALUE(
        false,
        "Expected input batch_size (",
        self.sizes()[0],
        ") to match target batch_size (",
        target.sizes()[0],
        ").");
  }

  Tensor ret;
  Tensor input_ = self;
  Tensor target_ = target;
  if (input_.dim() == 2) {
    ret = at::nll_loss(input_, target_, weight, reduction, ignore_index);
  } else if (input_.dim() == 4) {
    ret = at::nll_loss2d(input_, target_, weight, reduction, ignore_index);
  } else {
    // dim == 3 or dim > 4
    auto n = input_.sizes()[0];
    auto c = input_.sizes()[1];
    auto out_size = input_.sizes().slice(2).vec();
    out_size.insert(out_size.begin(), n);
    if (target_.sizes().slice(1) != input_.sizes().slice(2)) {
      TORCH_CHECK(
          false,
          "Expected target size ",
          IntArrayRef(out_size),
          ", got ",
          target_.sizes());
    }
    input_ = input_.contiguous();
    target_ = target_.contiguous();
    // support empty batches, see #15870
    if (input_.numel() > 0) {
      input_ = input_.view({n, c, 1, -1});
    } else {
      input_ = input_.view({n, c, 0, 0});
    }
    if (target_.numel() > 0) {
      target_ = target_.view({n, 1, -1});
    } else {
      target_ = target_.view({n, 0, 0});
    }
    if (!(reduction == Reduction::None)) {
      ret = at::nll_loss2d(input_, target_, weight, reduction, ignore_index);
    } else {
      auto out =
          at::nll_loss2d(input_, target_, weight, reduction, ignore_index);
      ret = out.view(out_size);
    }
  }
  return ret;
}

} // namespace native
} // namespace at
