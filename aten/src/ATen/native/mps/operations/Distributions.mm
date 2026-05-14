//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/TensorOperators.h>
#include <ATen/ceil_div.h>
#include <ATen/mps/MPSGeneratorImpl.h>
#include <ATen/native/DistributionTemplates.h>
#include <ATen/native/Distributions.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_standard_gamma_grad_native.h>
#include <ATen/ops/_standard_gamma_native.h>
#include <ATen/ops/argmax.h>
#include <ATen/ops/bernoulli_native.h>
#include <ATen/ops/clamp.h>
#include <ATen/ops/cumsum.h>
#include <ATen/ops/div.h>
#include <ATen/ops/multinomial_native.h>
#include <ATen/ops/rand.h>
#include <ATen/ops/randperm.h>
#include <ATen/ops/randperm_native.h>
#include <ATen/ops/searchsorted.h>
#include <ATen/ops/topk.h>
#endif

namespace at::native {
namespace mps {
#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/Distributions_metallib.h>
#endif

struct RandomCachedGraph : public MPSCachedGraph {
  RandomCachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
  // Only relevant for multinomial
  MPSGraphTensor* probTensor = nil;
  MPSGraphTensor* resultTensor = nil;
  MPSGraphTensor* stateTensor = nil;
  // used for Normal distributions only
  MPSGraphTensor *meanTensor = nil, *stdTensor = nil;
};

typedef MPSGraphTensor* (^RandomOpBlock)(RandomCachedGraph*, MPSGraphTensor*);
#define RandomOpFn(graph, randomTensor) MPSGraphTensor*(mps::RandomCachedGraph * graph, MPSGraphTensor * randomTensor)

// for Uniform distributions with scalar from (val1) and to (val2) intervals
// for Normal distributions with scalar mean (val1) and std (val2) values
template <typename scalar_t>
Tensor& random_mps_impl(Tensor& self,
                        scalar_t val1,
                        scalar_t val2,
                        const std::optional<Tensor>& mean_opt,
                        const std::optional<Tensor>& std_opt,
                        MPSGraphRandomDistribution distribution,
                        std::optional<Generator> gen,
                        std::string op_name,
                        RandomOpBlock randomBlock) {
  if (self.numel() == 0) {
    return self;
  }
  at::assert_no_internal_overlap(self);
  // MPS random is broken for 5D+ tensors, see https://github.com/pytorch/pytorch/issues/147624
  const auto need_reshape = self.ndimension() > 4;
  auto mps_gen = get_generator_or_default<MPSGeneratorImpl>(gen, at::mps::detail::getDefaultMPSGenerator());
  auto stream = getCurrentMPSStream();

  @autoreleasepool {
    auto key = op_name + getTensorsStringKey({self, mean_opt.value_or(Tensor()), std_opt.value_or(Tensor())}) + ":" +
        std::to_string(val1) + ":" + std::to_string(val2);
    auto cachedGraph = LookUpOrCreateCachedGraph<RandomCachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      newCachedGraph->stateTensor =
          mpsGraphRankedPlaceHolder(mpsGraph, MPSDataTypeInt32, @[ @(at::mps::detail::PHILOX_STATE_N) ]);

      // BF16, FP16, FP32 and Int32 are the only data types supported for distributions on MPS backend.
      const MPSDataType inputDataType = [&] {
        // only for random_mps, we pass interval range of type int64_t
        if constexpr (std::is_same_v<scalar_t, int64_t>) {
          return MPSDataTypeInt32;
        }
        // for bernoully always use float32
        if constexpr (std::is_same_v<scalar_t, bool>) {
          return MPSDataTypeFloat32;
        }
        switch (self.scalar_type()) {
          case kHalf:
            return MPSDataTypeFloat16;
          case kFloat:
            return MPSDataTypeFloat32;
          case kBFloat16: {
            return MPSDataTypeBFloat16;
          }
          default:
            TORCH_CHECK_TYPE(false, "Unsupported type ", self.scalar_type(), " for operation ", op_name);
        }
      }();
      const MPSDataType outputDataType = std::is_same_v<scalar_t, bool> ? MPSDataTypeBool : inputDataType;

      MPSGraphRandomOpDescriptor* desc = [MPSGraphRandomOpDescriptor descriptorWithDistribution:distribution
                                                                                       dataType:inputDataType];
      if (distribution == MPSGraphRandomDistributionUniform) {
        if (inputDataType == MPSDataTypeInt32) {
          desc.minInteger = static_cast<NSInteger>(val1);
          desc.maxInteger = static_cast<NSInteger>(val2);
        } else {
          desc.min = static_cast<float>(val1);
          desc.max = static_cast<float>(val2);
        }
      } else if (distribution == MPSGraphRandomDistributionNormal) {
        desc.mean = static_cast<float>(val1);
        desc.standardDeviation = static_cast<float>(val2);
      }
      // we don't use the output state tensor from the MPSGraph API as it requires reading back from GPU to CPU.
      // Instead, we keep the Philox state in the MPSGenerator and use the PyTorch's philox_engine to maintain
      // the counters, and feed them to the graph manually
      auto self_shape = getMPSShape(self);
      NSArray<MPSGraphTensor*>* resultTensors =
          [mpsGraph randomTensorWithShape:need_reshape ? @[ @(self.numel()) ] : self_shape
                               descriptor:desc
                              stateTensor:newCachedGraph->stateTensor
                                     name:nil];
      newCachedGraph->resultTensor =
          need_reshape ? [mpsGraph reshapeTensor:resultTensors[0] withShape:self_shape name:nil] : resultTensors[0];
      if (randomBlock) {
        newCachedGraph->resultTensor = randomBlock(newCachedGraph, newCachedGraph->resultTensor);
      }
      // results will be cast if self's scalar type isn't directly supported by MPS backend.
      if (getMPSDataType(self) != outputDataType)
        newCachedGraph->resultTensor = castMPSTensor(mpsGraph, newCachedGraph->resultTensor, self.scalar_type());
    });
    // feed the updated state values to the graph
    MPSNDArrayDescriptor* stateDesc =
        [MPSNDArrayDescriptor descriptorWithDataType:MPSDataTypeInt32 shape:@[ @(at::mps::detail::PHILOX_STATE_N) ]];
    MPSNDArray* stateNDArray = [[[MPSNDArray alloc] initWithDevice:stream->device() descriptor:stateDesc] autorelease];
    {
      // See Note [Acquire lock when using random generators]
      std::lock_guard<std::mutex> lock(mps_gen->mutex_);
      // update the Philox state values on each run
      mps_gen->update_philox_counters();
      [stateNDArray writeBytes:mps_gen->state_data() strideBytes:nil];
    }
    MPSGraphTensorData* stateTensorData = [[[MPSGraphTensorData alloc] initWithMPSNDArray:stateNDArray] autorelease];

    Placeholder meanPlaceholder, stdPlaceholder;
    NSMutableDictionary* feeds = [[NSMutableDictionary new] autorelease];
    feeds[cachedGraph->stateTensor] = stateTensorData;

    if (cachedGraph->stdTensor) {
      const Tensor& stdTensor = *(at::borrow_from_optional_tensor(std_opt));
      stdPlaceholder = Placeholder(cachedGraph->stdTensor, stdTensor);
      feeds[stdPlaceholder.getMPSGraphTensor()] = stdPlaceholder.getMPSGraphTensorData();
    }
    if (cachedGraph->meanTensor) {
      const Tensor& meanTensor = *(at::borrow_from_optional_tensor(mean_opt));
      meanPlaceholder = Placeholder(cachedGraph->meanTensor, meanTensor);
      feeds[meanPlaceholder.getMPSGraphTensor()] = meanPlaceholder.getMPSGraphTensorData();
    }

    // Handle non-contiguous output tensors by creating a contiguous temporary
    const auto needs_gather = needsGather(self);
    Tensor self_ = needs_gather ? at::empty_like(self, MemoryFormat::Contiguous) : self;
    Placeholder outputPlaceholder = Placeholder(cachedGraph->resultTensor, self_);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);

    // Copy results back to original non-contiguous output
    if (needs_gather) {
      self.copy_(self_);
    }
  }

  return self;
}

} // namespace mps

// `randoms_per_thread` is the number of Philox-4x32-10 calls each thread
// makes; `elements_per_thread` is how many output elements that thread
// writes. The host advances the generator offset by exactly the number of
// philox indices the kernel consumes (`randoms_per_thread * threads`),
// rather than by `randoms_per_element * numel` as before, which could
// over-advance by 4x for kernels that pack 4 elements per thread.
//
// Decomposes tensors that exceed 32-bit indexing into sub-iters first; each
// sub-dispatch pulls a fresh philox offset chunk so the chunks sample
// disjoint streams of the same seed. The kernel itself takes `numel` as
// `uint32_t`, so the splitting is what keeps it correct for >2^31-element
// tensors instead of silently wrapping.
// `params_t` is the array type bound to buffer 1 of the kernel; `float2` for
// the bulk of distributions and `long2` for the int64 random path. The
// templated dispatcher keeps the recursion / generator-advance / 32-bit-iter
// boilerplate in one place. A `(double, double)` overload below preserves the
// ergonomics for the common float-param callers.
template <typename params_t>
static void distribution_kernel_mps_impl(TensorIteratorBase& iter,
                                         params_t params,
                                         const std::string& kernel_name,
                                         int64_t randoms_per_thread,
                                         std::optional<Generator> gen,
                                         int64_t elements_per_thread = 4) {
  if (iter.numel() == 0) {
    return;
  }

  using namespace mps;

  // Non-contiguous outputs can't go through the linear-index kernel: write
  // into a contiguous temp first, then scatter back through the iter. Handle
  // this *before* `with_32bit_indexing` decomposition - sub-iters of a
  // non-contiguous iter still report the parent's full `iter.tensor(0)`, so
  // taking the temp's shape from a sub-iter would over-allocate and re-fill
  // the whole output once per sub-iter. At the top level of the recursion
  // `iter.tensor(0)` is unambiguously the user's output tensor.
  if (!iter.is_contiguous()) {
    Tensor tmp = at::empty(iter.tensor(0).sizes(), iter.tensor(0).options());
    auto tmp_iter = at::TensorIterator::borrowing_nullary_op(tmp);
    distribution_kernel_mps_impl(tmp_iter, params, kernel_name, randoms_per_thread, gen, elements_per_thread);
    iter.tensor(0).copy_(tmp);
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto&& sub_iter : iter.with_32bit_indexing()) {
      distribution_kernel_mps_impl(sub_iter, params, kernel_name, randoms_per_thread, gen, elements_per_thread);
    }
    return;
  }

  // After `with_32bit_indexing` decomposition `iter.numel()` fits in uint32,
  // but `checked_convert` keeps us honest if anything ever reaches the kernel
  // with a count that would silently truncate.
  const uint32_t numel = c10::checked_convert<uint32_t>(iter.numel(), "uint32_t");
  const int64_t threads = at::ceil_div<int64_t>(numel, elements_per_thread);

  auto mps_gen = get_generator_or_default<MPSGeneratorImpl>(gen, at::mps::detail::getDefaultMPSGenerator());
  auto stream = getCurrentMPSStream();

  @autoreleasepool {
    // Sub-iters share the original tensor's storage but have their own
    // `data_ptr` offset; `bind_iter_tensors` computes that for buffer 0.
    auto pso = lib.getPipelineStateForFunc(kernel_name + "_" + scalarToMetalTypeString(iter.tensor(0)));

    int64_t seed;
    int64_t base_offset;
    {
      // See Note [Acquire lock when using random generators]
      std::lock_guard<std::mutex> lock(mps_gen->mutex_);
      seed = static_cast<int64_t>(mps_gen->current_seed());
      base_offset = static_cast<int64_t>(mps_gen->get_offset());
      mps_gen->set_offset(base_offset + randoms_per_thread * threads);
    }

    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        auto computeEncoder = stream->commandEncoder();
        [computeEncoder setComputePipelineState:pso];
        bind_iter_tensors(computeEncoder, iter, /*ntensors=*/1);
        mtl_setArgs<1>(computeEncoder, params, std::array<long, 2>{seed, base_offset}, numel);
        mtl_dispatch1DJob(computeEncoder, pso, threads);
      }
    });
  }
}

// Float-param convenience overload for callers that don't need the int64
// kernel path. Forwards to the templated dispatcher above.
static void distribution_kernel_mps_impl(TensorIteratorBase& iter,
                                         double param1,
                                         double param2,
                                         const std::string& kernel_name,
                                         int64_t randoms_per_thread,
                                         std::optional<Generator> gen,
                                         int64_t elements_per_thread = 4) {
  distribution_kernel_mps_impl(iter,
                               std::array<float, 2>{static_cast<float>(param1), static_cast<float>(param2)},
                               kernel_name,
                               randoms_per_thread,
                               gen,
                               elements_per_thread);
}

// Tensor-p Bernoulli: dispatches the Metal `bernoulli_tensor` kernel with a
// flat float32 probability buffer matching `self.numel()`.
static Tensor& bernoulli_tensor_mps_impl(Tensor& self, const Tensor& p_, std::optional<Generator> gen) {
  if (self.numel() == 0) {
    return self;
  }
  TORCH_CHECK(p_.is_same_size(self) || p_.dim() == 0,
              "bernoulli_mps_: probability and self tensor should be of the same shape");

  using namespace mps;

  // 0-dim p is a scalar — go through the scalar path.
  if (p_.dim() == 0) {
    double p_val = p_.item<double>();
    TORCH_CHECK(0.0 <= p_val && p_val <= 1.0, "bernoulli_mps_ expects p to be in [0, 1], but got p=", p_val);
    auto iter = at::TensorIterator::borrowing_nullary_op(self);
    distribution_kernel_mps_impl(iter, p_val, 0.0, "bernoulli_scalar", 1, gen);
    return self;
  }

  // Both `to` and `contiguous` short-circuit when no work is needed.
  auto p_float = p_.to(kFloat).contiguous();
  auto mps_gen = get_generator_or_default<MPSGeneratorImpl>(gen, at::mps::detail::getDefaultMPSGenerator());
  auto stream = getCurrentMPSStream();
  const auto needs_copy = !self.is_contiguous();
  auto output = needs_copy ? at::empty_like(self, MemoryFormat::Contiguous) : self;

  @autoreleasepool {
    auto pso = lib.getPipelineStateForFunc("bernoulli_tensor_" + scalarToMetalTypeString(output));

    int64_t seed;
    int64_t base_offset;
    {
      // See Note [Acquire lock when using random generators]
      std::lock_guard<std::mutex> lock(mps_gen->mutex_);
      seed = static_cast<int64_t>(mps_gen->current_seed());
      base_offset = static_cast<int64_t>(mps_gen->get_offset());
      mps_gen->set_offset(base_offset + output.numel());
    }

    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        auto computeEncoder = stream->commandEncoder();
        [computeEncoder setComputePipelineState:pso];
        const auto numel = c10::checked_convert<uint32_t>(output.numel(), "uint32_t");
        mtl_setArgs(computeEncoder, output, p_float, std::array<long, 2>{seed, base_offset}, numel);
        mtl_dispatch1DJob(computeEncoder, pso, at::ceil_div(numel, 4u));
      }
    });
  }

  if (needs_copy) {
    self.copy_(output);
  }
  return self;
}

// Stub-style entry points so MPS shares the dispatch template with CPU/CUDA.
// `const TensorBase&` mirrors the stub signature; the underlying storage is
// still the inplace target (TensorIterator uses the same idiom).
static void bernoulli_scalar_kernel_mps(const TensorBase& self, double p, std::optional<Generator> gen) {
  // 0 <= p <= 1 is already enforced by `bernoulli_impl_` before the stub is dispatched.
  auto iter = at::TensorIterator::borrowing_nullary_op(self);
  distribution_kernel_mps_impl(iter, p, 0.0, "bernoulli_scalar", 1, gen);
}

static void bernoulli_tensor_kernel_mps(const TensorBase& self, const TensorBase& p_, std::optional<Generator> gen) {
  Tensor& self_t = const_cast<Tensor&>(static_cast<const Tensor&>(self));
  const Tensor& p_t = static_cast<const Tensor&>(p_);
  bernoulli_tensor_mps_impl(self_t, p_t, gen);
}

REGISTER_MPS_DISPATCH(bernoulli_scalar_stub, &bernoulli_scalar_kernel_mps)
REGISTER_MPS_DISPATCH(bernoulli_tensor_stub, &bernoulli_tensor_kernel_mps)

static void uniform_kernel_mps(TensorIteratorBase& iter, double from, double to, std::optional<Generator> gen) {
  distribution_kernel_mps_impl(iter, from, to, "uniform_dist", 1, gen);
}

static void normal_kernel_mps(const TensorBase& self, double mean, double std, std::optional<Generator> gen) {
  // Match CPU/CUDA: only floating dtypes are supported. Without this the
  // kernel-name lookup downstream produces a confusing
  // "Failed to create function state object for: normal_int" RuntimeError.
  TORCH_CHECK_TYPE(
      at::isFloatingType(self.scalar_type()), "normal_kernel_mps not implemented for '", self.scalar_type(), "'");
  auto iter = at::TensorIterator::borrowing_nullary_op(self);
  distribution_kernel_mps_impl(iter, mean, std, "normal", 1, gen);
}

static void cauchy_kernel_mps(TensorIteratorBase& iter, double median, double sigma, std::optional<Generator> gen) {
  distribution_kernel_mps_impl(iter, median, sigma, "cauchy", 1, gen);
}

static void exponential_kernel_mps(TensorIteratorBase& iter, double lambda, std::optional<Generator> gen) {
  distribution_kernel_mps_impl(iter, lambda, 0.0, "exponential", 1, gen);
}

static void log_normal_kernel_mps(TensorIteratorBase& iter, double mean, double std, std::optional<Generator> gen) {
  distribution_kernel_mps_impl(iter, mean, std, "log_normal", 1, gen);
}

static void geometric_kernel_mps(TensorIteratorBase& iter, double p, std::optional<Generator> gen) {
  // `geometric_impl_` already enforces 0 < p < 1; we pass `log1p(-p)` so the
  // kernel can apply the inverse-CDF (`floor(log(u) / log1p(-p))`) directly.
  distribution_kernel_mps_impl(iter, std::log1p(-p), 0.0, "geometric", 1, gen);
}

// Dispatch the packed `random_int` kernel. `range_param == 0` is the
// kernel-side sentinel for "full T-bit-width range" (used both for the int64
// `[int64_min, int64_max]` case where range = 2^64 doesn't fit in any signed
// type and for `random_(self)` callers that want each output to span the full
// dtype). The kernel keeps a dtype-stable layout for narrow types - 4 outputs
// per thread for `sizeof(T) <= 4`, 2 for `sizeof(T) == 8` - so `randint_like`
// produces value-identical results across float/half/int dtypes.
static void random_int_dispatch(TensorIteratorBase& iter,
                                int64_t base,
                                int64_t range_param,
                                std::optional<Generator> gen) {
  const int64_t elts_per_thread = (iter.element_size(0) == 8) ? 2 : 4;
  distribution_kernel_mps_impl(iter,
                               std::array<long, 2>{base, range_param},
                               "random_int",
                               /*randoms_per_thread=*/1,
                               gen,
                               elts_per_thread);
}

static void random_from_to_kernel_mps(TensorIteratorBase& iter,
                                      uint64_t range,
                                      int64_t base,
                                      std::optional<Generator> gen) {
  // `range` is `to - from`. The kernel's `range == 0` sentinel covers the
  // full 2^64 case (where `static_cast<int64_t>(UINT64_MAX) == -1` would
  // otherwise look like an empty range). Smaller ranges fit in `int64_t`
  // directly because `random_from_to_impl` already clamps `to` to the dtype's
  // max + 1.
  const int64_t range_param = (range == std::numeric_limits<uint64_t>::max()) ? 0 : static_cast<int64_t>(range);
  random_int_dispatch(iter, base, range_param, gen);
}

static void random_kernel_mps(TensorIteratorBase& iter, std::optional<Generator> gen) {
  // No-args `random_(self)`: fill with values uniform over `[0, 2^digits)` for
  // floating dtypes and `[0, dtype_max + 1)` for integer dtypes. Note that for
  // signed integers this includes only non-negative values, matching the
  // pre-existing MPS behaviour.
  int64_t to = 0;
  const auto dtype = iter.dtype();
  if (isFloatingType(dtype)) {
    AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, dtype, "random_kernel_mps_range_calc", [&] {
      constexpr int64_t scalar_t_max = static_cast<int64_t>(1) << std::numeric_limits<scalar_t>::digits;
      to = scalar_t_max > std::numeric_limits<int64_t>::max() ? std::numeric_limits<int64_t>::max() : scalar_t_max;
    });
  } else if (isIntegralType(dtype, /*includeBool=*/true)) {
    AT_DISPATCH_INTEGRAL_TYPES_AND(kBool, dtype, "random_kernel_mps_range_calc", [&] {
      if constexpr (std::is_same_v<scalar_t, int64_t>) {
        to = std::numeric_limits<int64_t>::max();
      } else {
        to = static_cast<uint64_t>(std::numeric_limits<scalar_t>::max()) + 1;
      }
    });
  } else {
    TORCH_CHECK(false, "random_mps handles only integral, floating-point and boolean types");
  }
  random_int_dispatch(iter, /*base=*/0, /*range_param=*/to, gen);
}

static void random_full_64_bits_range_kernel_mps(TensorIteratorBase& iter, std::optional<Generator> gen) {
  // [int64_min, int64_max]: full uint64 sentinel (`range_param = 0`) plus a
  // base shift that moves the unsigned output back into the signed range.
  random_int_dispatch(iter, /*base=*/std::numeric_limits<int64_t>::min(), /*range_param=*/0, gen);
}

REGISTER_MPS_DISPATCH(uniform_stub, &uniform_kernel_mps)
REGISTER_MPS_DISPATCH(normal_stub, &normal_kernel_mps)
REGISTER_MPS_DISPATCH(cauchy_stub, &cauchy_kernel_mps)
REGISTER_MPS_DISPATCH(exponential_stub, &exponential_kernel_mps)
REGISTER_MPS_DISPATCH(log_normal_stub, &log_normal_kernel_mps)
REGISTER_MPS_DISPATCH(geometric_stub, &geometric_kernel_mps)
REGISTER_MPS_DISPATCH(random_stub, &random_kernel_mps)
REGISTER_MPS_DISPATCH(random_from_to_stub, &random_from_to_kernel_mps)
REGISTER_MPS_DISPATCH(random_full_64_bits_range_stub, &random_full_64_bits_range_kernel_mps)

Tensor _s_gamma_mps(const Tensor& alpha, std::optional<Generator> gen) {
  if (alpha.numel() == 0) {
    return at::empty_like(alpha);
  }

  using namespace mps;

  auto mps_gen = get_generator_or_default<MPSGeneratorImpl>(gen, at::mps::detail::getDefaultMPSGenerator());
  auto stream = getCurrentMPSStream();
  Tensor ret = at::empty_like(alpha, alpha.options(), at::MemoryFormat::Contiguous);
  auto alpha_contig = alpha.contiguous();

  @autoreleasepool {
    auto pso = lib.getPipelineStateForFunc("standard_gamma_" + scalarToMetalTypeString(ret));

    int64_t seed;
    int64_t base_offset;
    // Each thread may consume up to GAMMA_RANDOMS_STRIDE random numbers
    constexpr int64_t GAMMA_RANDOMS_STRIDE = 32;
    {
      // See Note [Acquire lock when using random generators]
      std::lock_guard<std::mutex> lock(mps_gen->mutex_);
      seed = static_cast<int64_t>(mps_gen->current_seed());
      base_offset = static_cast<int64_t>(mps_gen->get_offset());
      mps_gen->set_offset(base_offset + GAMMA_RANDOMS_STRIDE * ret.numel());
    }

    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        auto computeEncoder = stream->commandEncoder();
        [computeEncoder setComputePipelineState:pso];
        mtl_setArgs(computeEncoder, ret, alpha_contig, std::array<long, 2>{seed, base_offset});
        mtl_dispatch1DJob(computeEncoder, pso, ret.numel());
      }
    });
  }

  return ret;
}

Tensor _standard_gamma_grad_mps(const Tensor& self, const Tensor& output) {
  if (self.numel() == 0) {
    return at::empty_like(self);
  }

  using namespace mps;

  auto stream = getCurrentMPSStream();
  Tensor ret = at::empty_like(self, self.options(), at::MemoryFormat::Contiguous);
  const auto self_contig = self.contiguous();
  const auto output_contig = output.contiguous();

  @autoreleasepool {
    auto pso = lib.getPipelineStateForFunc("standard_gamma_grad_" + scalarToMetalTypeString(ret));

    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        auto computeEncoder = stream->commandEncoder();
        [computeEncoder setComputePipelineState:pso];
        mtl_setArgs(computeEncoder, ret, self_contig, output_contig);
        mtl_dispatch1DJob(computeEncoder, pso, ret.numel());
      }
    });
  }

  return ret;
}

Tensor& randperm_out_mps(int64_t n, std::optional<Generator> generator, Tensor& result) {
  TORCH_CHECK(n >= 0, "n must be non-negative, got", n);
  TORCH_CHECK(!generator.has_value() || (generator.has_value() && result.device() == generator->device()),
              "Expected a '",
              result.device(),
              "' generator device but found '",
              generator->device(),
              "'");
  check_supported_max_int_with_precision(n, result);

  result.resize_({n});
  if (n == 0) {
    return result;
  }

  mps::RandomOpBlock random_op_block = ^RandomOpFn(cachedGraph, randomTensor) {
    MPSGraph* mpsGraph = cachedGraph->graph();
    MPSGraphTensor* argsortTensor = [mpsGraph argSortWithTensor:randomTensor axis:0 name:nil];
    if (result.scalar_type() != kInt) {
      argsortTensor = [mpsGraph castTensor:argsortTensor toType:mps::getMPSDataType(result) name:@"castOutput"];
    }
    return argsortTensor;
  };

  return mps::random_mps_impl<int64_t>(result,
                                       std::numeric_limits<int64_t>::min(),
                                       std::numeric_limits<int64_t>::max(),
                                       std::nullopt,
                                       std::nullopt,
                                       MPSGraphRandomDistributionUniform,
                                       generator,
                                       "ranperm_out_mps:" + mps::getTensorsStringKey({result}),
                                       random_op_block);
}

static Tensor& multinomial_with_replacement_mps_kernel(const Tensor& self,
                                                       const int64_t n_sample,
                                                       std::optional<Generator> generator,
                                                       Tensor& result) {
  auto numCategories = self.size(-1);
  // CDF accumulated in float32 since bfloat16/float16 lose precision summing many small probabilities.
  // Sample u from U[0, total) and search the unnormalized CDF,
  // equivalent to normalizing then sampling u from U[0, 1)
  auto cdf = self.cumsum(-1, /*dtype=*/kFloat);
  auto uniform = at::rand(result.sizes(), generator, self.options().dtype(kFloat))
                     .mul_(cdf.select(-1, numCategories - 1).unsqueeze(-1));
  at::searchsorted_out(result, cdf, uniform);
  return result.clamp_(0, numCategories - 1);
}

/* The largest consecutive integer representable in float32 (2^24) */
constexpr int64_t FLOAT32_MAX_CONSECUTIVE_INT = 1 << (FLT_MANT_DIG);

Tensor& multinomial_out_mps(const Tensor& self,
                            int64_t n_sample,
                            bool with_replacement,
                            std::optional<Generator> gen,
                            Tensor& result) {
  TORCH_CHECK(result.device() == self.device(), "multinomial arguments must have the same device");
  TORCH_CHECK(self.dim() > 0 && self.dim() <= 2, "prob_dist must be 1 or 2 dim");
  TORCH_CHECK(at::isFloatingType(self.scalar_type()),
              "multinomial only supports floating-point dtypes for input, got: ",
              self.scalar_type());
  TORCH_CHECK(
      result.scalar_type() == ScalarType::Long, "multinomial expects Long tensor out, got: ", result.scalar_type());
  TORCH_CHECK(n_sample > 0, "cannot sample n_sample <= 0 samples");
  int64_t n_categories = self.size(-1);
  TORCH_CHECK(with_replacement || (n_sample <= n_categories),
              "cannot sample n_sample > prob_dist.size(-1) samples without replacement");
  // Since the index tensor is float, numCategories cannot exceed max
  // float integer precision
  TORCH_CHECK(n_categories <= FLOAT32_MAX_CONSECUTIVE_INT, "number of categories cannot exceed 2^24");

  if (self.dim() == 1) {
    result.resize_({n_sample});
  } else {
    const int64_t n_dist = self.size(0);
    result.resize_({n_dist, n_sample});
  }
  if (result.numel() == 0) {
    return result;
  }

  // Fast-path for no replacement (or if only one sample draw).
  // Reference:
  // https://github.com/pytorch/pytorch/issues/11931#issuecomment-625882503
  if (!with_replacement || n_sample == 1) {
    // Sanity checks on `self`.
    auto is_valid = ((self.max() < INFINITY) & (self.min() >= 0)).item();
    TORCH_CHECK(is_valid.to<bool>(), "probability tensor contains either `inf`, `nan` or element < 0");
    bool zero_prob_condition = false;
    if (self.dim() == 1) {
      zero_prob_condition = (self.sum() == 0).item().to<bool>();
    } else {
      zero_prob_condition = (self.sum(1) == 0).sum().item().to<bool>();
    }
    TORCH_CHECK(!zero_prob_condition, "invalid multinomial distribution (sum of probabilities <= 0)");

    // The algorithm is from gumbel softmax.
    // s = argmax( logp - log(-log(eps)) ) where eps ~ U(0, 1)
    // Here we can apply exp to the formula which will not affect result of
    // argmax or topk. Then we have
    // s = argmax( p / (-log(eps)) ) where eps ~ U(0, 1).
    // We can also simplify the formula above by
    // s = argmax( p / q ) where q ~ Exp(1)
    // If needed, create `q` as contiguous tensor to ensure memory layout supports inplace operations
    const auto has_strided_api = is_macos_13_or_newer(MacOSVersion::MACOS_VER_15_0_PLUS);
    auto q = at::empty_like(self, {}, has_strided_api ? std::nullopt : std::optional(MemoryFormat::Contiguous));
    q.exponential_(1, gen);
    // In theory the probability to generate 0 from exponential distribution is
    // 0. However, on CUDA side there is a protection to avoid 0s, but on CPU
    // side, there is a very low probability to generate 0 from
    // exponential<double>. The probability is about 2^(-DBL_MANT_DIG). We just
    // ignore it here, but there may be some risk to get invalid output on CPU.
    at::div_out(q, self, q);
    if (n_sample == 1) {
      at::argmax_out(result, q, /*dim=*/-1, /*keepdim=*/true);
    } else {
      Tensor vals = at::empty(result.sizes(), self.options());
      at::topk_out(vals, result, q, n_sample);
    }
    return result;
  }

  result = multinomial_with_replacement_mps_kernel(const_cast<Tensor&>(self), n_sample, gen, result);

  return result;
}

Tensor multinomial_mps(const Tensor& self, int64_t n_sample, bool with_replacement, std::optional<Generator> gen) {
  Tensor result = at::empty({0}, self.options().dtype(kLong));
  multinomial_out_mps(self, n_sample, with_replacement, gen, result);
  return result;
}

} // namespace at::native
