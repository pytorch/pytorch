#include <ATen/Batching.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <ATen/WrapDimUtils.h>

namespace at {

/////////////////////////////////////////////////////////////
// --------------------[ UTILITIES ]-------------------------
/////////////////////////////////////////////////////////////

bool isBatchTensor(const Tensor& tensor) {
  return tensor.unsafeGetTensorImpl()->key_set().has(BatchTensorKey);
}

int64_t maxLevel(const std::vector<Tensor>& maybeBatchTensors) {
  int64_t max = -1;
  auto it = maybeBatchTensors.begin();
  auto end_it = maybeBatchTensors.end();
  while (it != end_it) {
    it = std::find_if(it, end_it, isBatchTensor);
    if (it != end_it) {
      const auto* batchTensor = static_cast<const BatchTensorImpl*>(it->unsafeGetTensorImpl());
      if (batchTensor->level_ > max) {
        max = batchTensor->level_;
      }
      it++;
    }
  }
  return max;
}

std::pair<Tensor,optional<int64_t>> unwrapAtLevel(const Tensor& tensor, int64_t level) {
  if (!isBatchTensor(tensor)) {
    return { tensor, nullopt };
  }
  auto* batch_tensor = getBatched(tensor);
  if (batch_tensor->level_ != level) {
    TORCH_INTERNAL_ASSERT(batch_tensor->level_ < level);
    return { tensor, nullopt };
  }
  return { batch_tensor->rep_, batch_tensor->batch_dim_ };
}

Tensor broadcastTo(const Tensor& tensor, int64_t ndim) {
  auto old_sizes = tensor.sizes();
  if (old_sizes.size() == ndim) {
    return tensor;
  }
  TORCH_INTERNAL_ASSERT(old_sizes.size() <= ndim);
  // TODO: This is really slow, we should probably write a new operator for
  // this. Note that we can't call view because it is not "semantic" enough.
  // It might be possible to just call reshape here.
  int64_t diff = ndim - old_sizes.size();
  Tensor result = tensor;
  for (int64_t i = 0; i < diff; ++i) {
    result = result.unsqueeze(0);  
  }
  return result;
}

Tensor moveBatchDimToFront(
    const Tensor& tensor,
    optional<int64_t> batch_dim,
    int64_t result_dim) {
  if (!batch_dim) {
    return broadcastTo(tensor, result_dim);
  }
  auto bdim = *batch_dim;
  auto extra_dims = result_dim - tensor.dim();
  auto result = broadcastTo(tensor, result_dim);
  auto transpose_dim = bdim + extra_dims;
  if (transpose_dim == 0) {
    return result;
  }
  return result.transpose(0, bdim + extra_dims);
}

int64_t actualDim(int64_t dim, optional<int64_t> maybe_batch_dim) {
  if (maybe_batch_dim && dim >= *maybe_batch_dim) {
    return dim + 1;
  }
  return dim;
}

int64_t minRequiredDim(const Tensor& tensor, optional<int64_t> batch_dim) {
  auto result = tensor.dim(); 
  if (!batch_dim) {
    result += 1;
  }
  return result;
}

typedef std::pair<Tensor,optional<int64_t>> TensorAndBdim;



/////////////////////////////////////////////////////////////
// --------------------[ BATCHING RULES ]--------------------
/////////////////////////////////////////////////////////////

std::pair<Tensor,optional<int64_t>> unsqueeze_batching_rule(
    const Tensor& self, optional<int64_t> self_bdim, int64_t dim) {
  const auto& tensor = self;
  const auto& maybe_batch_dim = self_bdim;

  dim = maybe_wrap_dim(dim, tensor.dim());
  auto actual_dim = actualDim(dim, maybe_batch_dim);
  optional<int64_t> new_batch_dim = nullopt;
  if (maybe_batch_dim) {
    auto bdim = *maybe_batch_dim;
    new_batch_dim = bdim < actual_dim ? bdim : bdim + 1;
  }
  return { tensor.unsqueeze(actual_dim), new_batch_dim };
}

std::pair<Tensor,optional<int64_t>> mul_batching_rule(
    const Tensor& self, optional<int64_t> self_bdim,
    const Tensor& other, optional<int64_t> other_bdim) {
  auto self_dim = minRequiredDim(self, self_bdim);
  auto other_dim = minRequiredDim(other, other_bdim);
  auto result_dim = std::max({self_dim, other_dim});

  auto self_value = moveBatchDimToFront(self, self_bdim, result_dim);
  auto other_value = moveBatchDimToFront(other, other_bdim, result_dim);
  return { at::mul(self_value, other_value), 0 };
}

std::pair<Tensor,optional<int64_t>> add_batching_rule(
    const Tensor& self, optional<int64_t> self_bdim,
    const Tensor& other, optional<int64_t> other_bdim) {
  auto self_dim = minRequiredDim(self, self_bdim);
  auto other_dim = minRequiredDim(other, other_bdim);
  auto result_dim = std::max({self_dim, other_dim});

  auto self_value = moveBatchDimToFront(self, self_bdim, result_dim);
  auto other_value = moveBatchDimToFront(other, other_bdim, result_dim);
  return { at::add(self_value, other_value), 0 };
}

std::pair<Tensor,optional<int64_t>> conv2d_batching_rule(
    const Tensor& self, optional<int64_t> self_bdim,
    const Tensor& weight, optional<int64_t> weight_bdim,
    const Tensor& bias, optional<int64_t> bias_bdim,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups) {
  if (weight_bdim) {
    TORCH_CHECK(false, "NYI: conv2d_batching_rule for batched weight");
  }
  if (bias_bdim) {
    TORCH_CHECK(false, "NYI: conv2d_batching_rule for batched bias");
  }
  auto result_dim = minRequiredDim(self, self_bdim);
  auto self_ = moveBatchDimToFront(self, self_bdim, result_dim);
  if (self_.dim() == 4) {
    // User used vmap over a batch of 3D tensors.
    return { at::conv2d(self_, weight, bias, stride, padding, dilation), 0 };
  }
  // self_ either has dim 5 or a user passed in a tensor that is too small.
  TORCH_INTERNAL_ASSERT(self_.dim() <= 5);
  auto self_sizes = self_.sizes();
  auto self_4d = self_.flatten(0, 1);
  auto result_4d = at::conv2d(
      self_4d, weight, bias, stride, padding, dilation);
  return {
    result_4d.unflatten(0, {self_sizes.begin(), self_sizes.begin() + 2}),
    /*result_bdim=*/0
  };
}

/////////////////////////////////////////////////////////////
// --------------------[ FALLBACK IMPLEMENTATION ]-----------
/////////////////////////////////////////////////////////////

void batchTensorFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  TORCH_CHECK(false, "Batching rule not implemented for ", op.schema(), ".");
}


/////////////////////////////////////////////////////////////
// --------------------[ REGISTRY DECLARATIONS ]-------------
/////////////////////////////////////////////////////////////

// TODO: The dispatcher has trouble with this so we register an unboxed kernel.
Tensor BatchedTensor_conv2d(const Tensor& input, const Tensor& weight,
                            const Tensor& bias, IntArrayRef stride,
                            IntArrayRef padding, IntArrayRef dilation,
                            int64_t groups) {
  // The following lines need to happen in each kernel
  auto cur_level = maxLevel({input, weight, bias});
  auto input_and_bdim = unwrapAtLevel(input, cur_level);
  auto weight_and_bdim = unwrapAtLevel(weight, cur_level);
  auto bias_and_bdim = unwrapAtLevel(bias, cur_level);

  auto result_and_bdim = conv2d_batching_rule(
      input_and_bdim.first,
      input_and_bdim.second,
      weight_and_bdim.first,
      weight_and_bdim.second,
      bias_and_bdim.first,
      bias_and_bdim.second,
      stride, padding, dilation, groups);
  return makeBatched(
      result_and_bdim.first,
      result_and_bdim.second,
      cur_level);
}

auto registry = c10::Dispatcher::singleton().registerBackendFallbackKernel(
    BatchTensorKey,
    KernelFunction::makeFromBoxedFunction<&batchTensorFallback>()
);

static auto registry2 = torch::RegisterOperators()
  // Some operations need to be transformed to their batched versions
  .op(torch::RegisterOperators::options()
      .schema("aten::_make_batched(Tensor self, int? batch_dim, int level) -> Tensor")
      .kernel(BatchTensorKey, &at::native::_make_batched))
  .op(torch::RegisterOperators::options()
      .schema("aten::_unwrap_batched(Tensor self, int level) -> Tensor")
      .kernel(BatchTensorKey, &at::native::_unwrap_batched))
  .op(torch::RegisterOperators::options()
      .schema("aten::unsqueeze(Tensor(a) self, int dim) -> Tensor(a)")
      .kernel(BatchTensorKey, [] (const Tensor& self, int64_t dim) -> Tensor {
        auto cur_level = maxLevel({self});
        auto self_and_bdim = unwrapAtLevel(self, cur_level);
        auto result_and_bdim = unsqueeze_batching_rule(
            self_and_bdim.first, self_and_bdim.second, dim);
        return makeBatched(
            result_and_bdim.first,
            result_and_bdim.second,
            cur_level);
      }))
  .op(torch::RegisterOperators::options()
      .schema("aten::transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)")
      .kernel(BatchTensorKey, [] (const Tensor& self, int64_t dim0, int64_t dim1) -> Tensor {
        // TODO: don't forget to wrap dim0 & dim1
        auto* self_batched = getBatched(self);
        auto batch_dim = self_batched->batch_dim_;
        return makeBatched(
          self_batched->rep_.transpose(
            actualDim(dim0, batch_dim),
            actualDim(dim1, batch_dim)),
          batch_dim,
          self_batched->level_);
      }))
  .op(torch::RegisterOperators::options()
      .schema("aten::mul.Tensor(Tensor self, Tensor other) -> Tensor")
      .kernel(BatchTensorKey, [] (const Tensor& self, const Tensor& other) -> Tensor {
        // The following lines need to happen in each kernel
        auto cur_level = maxLevel({self, other});
        auto self_and_bdim = unwrapAtLevel(self, cur_level);
        auto other_and_bdim = unwrapAtLevel(other, cur_level);

        auto result_with_batch = mul_batching_rule(
            self_and_bdim.first, self_and_bdim.second,
            other_and_bdim.first, other_and_bdim.second);
        return makeBatched(
            result_with_batch.first,
            result_with_batch.second,
            cur_level);
      }))
  .op(torch::RegisterOperators::options()
      .schema("aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor")
      .kernel(BatchTensorKey, [] (const Tensor& self, const Tensor& other, Scalar alpha) -> Tensor {
        // The following lines need to happen in each kernel
        auto cur_level = maxLevel({self, other});
        auto self_and_bdim = unwrapAtLevel(self, cur_level);
        auto other_and_bdim = unwrapAtLevel(other, cur_level);

        // TODO: this assumes that alpha = 1.
        auto result_with_batch = add_batching_rule(
            self_and_bdim.first, self_and_bdim.second,
            other_and_bdim.first, other_and_bdim.second);
        return makeBatched(
            result_with_batch.first,
            result_with_batch.second,
            cur_level);
      }))
  .op(torch::RegisterOperators::options()
      .schema("aten::detach(Tensor self) -> (Tensor)")
      .kernel(BatchTensorKey, [] (const Tensor& self) -> Tensor {
        auto* batched = getBatched(self);
        return makeBatched(
            batched->rep_.detach(),
            batched->batch_dim_,
            batched->level_);
      }))
  .op(torch::RegisterOperators::options()
      .schema("aten::_is_batched(Tensor self) -> bool")
      .kernel(BatchTensorKey, [] (const Tensor& self) -> bool {
        return true;
      }))
  .op(torch::RegisterOperators::options()
      .schema("aten::_batch_dim(Tensor self) -> int")
      .kernel(BatchTensorKey, [] (const Tensor& self) -> int64_t {
        return native::_batch_dim(self); // wut
      }))
  .op(torch::RegisterOperators::options()
      .schema("aten::size.int(Tensor self, int dim) -> int")
      .kernel(BatchTensorKey, [] (const Tensor& self, int64_t dim) -> int64_t {
        dim = maybe_wrap_dim(dim, self.dim());
        return self.sizes()[dim];
      }))
  .op(torch::RegisterOperators::options()
      .schema("aten::conv2d(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor")
      .impl_unboxedOnlyKernel<Tensor (const Tensor&, const Tensor&, const Tensor&, IntArrayRef, IntArrayRef, IntArrayRef, int64_t), &BatchedTensor_conv2d>(BatchTensorKey))
  ;

}
