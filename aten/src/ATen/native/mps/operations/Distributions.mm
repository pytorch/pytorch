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
#include <ATen/ops/_dirichlet_grad_native.h>
#include <ATen/ops/_sample_dirichlet_native.h>
#include <ATen/ops/_standard_gamma_grad_native.h>
#include <ATen/ops/_standard_gamma_native.h>
#include <ATen/ops/argsort.h>
#include <ATen/ops/bernoulli_native.h>
#include <ATen/ops/binomial_native.h>
#include <ATen/ops/clamp.h>
#include <ATen/ops/cumsum.h>
#include <ATen/ops/poisson_native.h>
#include <ATen/ops/rand.h>
#include <ATen/ops/randperm.h>
#include <ATen/ops/randperm_native.h>
#include <ATen/ops/searchsorted.h>
#include <ATen/ops/sort.h>
#include <ATen/ops/topk.h>
#endif

namespace at::native {
namespace mps {
#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/Distributions_metallib.h>
#endif
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
        mtl_setArgs(
            computeEncoder, output, p_float, std::array<long, 2>{seed, base_offset}, numel, stream->getErrorBuffer());
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

Tensor _s_poisson_mps(const Tensor& lambda, std::optional<Generator> gen) {
  if (lambda.numel() == 0) {
    return at::empty_like(lambda);
  }

  using namespace mps;

  auto mps_gen = get_generator_or_default<MPSGeneratorImpl>(gen, at::mps::detail::getDefaultMPSGenerator());
  auto stream = getCurrentMPSStream();
  Tensor ret = at::empty_like(lambda, lambda.options(), at::MemoryFormat::Contiguous);
  auto lambda_contig = lambda.contiguous();

  @autoreleasepool {
    auto pso = lib.getPipelineStateForFunc("poisson_" + scalarToMetalTypeString(ret));

    int64_t seed;
    int64_t base_offset;
    // Each thread may consume up to POISSON_RANDOMS_STRIDE random numbers
    constexpr int64_t POISSON_RANDOMS_STRIDE = 32;
    {
      // See Note [Acquire lock when using random generators]
      std::lock_guard<std::mutex> lock(mps_gen->mutex_);
      seed = static_cast<int64_t>(mps_gen->current_seed());
      base_offset = static_cast<int64_t>(mps_gen->get_offset());
      mps_gen->set_offset(base_offset + POISSON_RANDOMS_STRIDE * ret.numel());
    }

    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        auto computeEncoder = stream->commandEncoder();
        [computeEncoder setComputePipelineState:pso];
        mtl_setArgs(computeEncoder, ret, lambda_contig, std::array<long, 2>{seed, base_offset});
        mtl_dispatch1DJob(computeEncoder, pso, ret.numel());
      }
    });
  }

  return ret;
}

Tensor _s_dirichlet_mps(const Tensor& alpha, std::optional<Generator> gen) {
  if (alpha.numel() == 0) {
    return at::empty_like(alpha);
  }

  using namespace mps;

  auto mps_gen = get_generator_or_default<MPSGeneratorImpl>(gen, at::mps::detail::getDefaultMPSGenerator());
  auto stream = getCurrentMPSStream();
  Tensor ret = at::empty_like(alpha, alpha.options(), at::MemoryFormat::Contiguous);
  auto alpha_contig = alpha.contiguous();

  const int64_t num_alpha = (alpha.dim() > 0) ? alpha.size(-1) : 1;
  const int64_t num_batches = ret.numel() / num_alpha;

  @autoreleasepool {
    auto pso = lib.getPipelineStateForFunc("standard_dirichlet_" + scalarToMetalTypeString(ret));
    auto tptg = std::min(num_alpha, int64_t(pso.maxTotalThreadsPerThreadgroup));

    int64_t seed;
    int64_t base_offset;
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
        mtl_setArgs(computeEncoder,
                    ret,
                    alpha_contig,
                    std::array<long, 2>{seed, base_offset},
                    static_cast<uint32_t>(num_alpha));
        [computeEncoder dispatchThreadgroups:MTLSizeMake(num_batches, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(tptg, 1, 1)];
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

Tensor _dirichlet_grad_mps(const Tensor& x, const Tensor& alpha, const Tensor& total) {
  if (x.numel() == 0) {
    return at::empty_like(x);
  }

  using namespace mps;

  auto stream = getCurrentMPSStream();
  Tensor ret = at::empty_like(x, x.options(), at::MemoryFormat::Contiguous);
  const auto x_contig = x.contiguous();
  const auto alpha_contig = alpha.contiguous();
  const auto total_contig = total.contiguous();

  @autoreleasepool {
    auto pso = lib.getPipelineStateForFunc("dirichlet_grad_" + scalarToMetalTypeString(ret));

    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        auto computeEncoder = stream->commandEncoder();
        [computeEncoder setComputePipelineState:pso];
        mtl_setArgs(computeEncoder, ret, x_contig, alpha_contig, total_contig);
        mtl_dispatch1DJob(computeEncoder, pso, ret.numel());
      }
    });
  }

  return ret;
}

Tensor _s_binomial_mps(const Tensor& count, const Tensor& prob, std::optional<Generator> gen_) {
  using namespace mps;

  TORCH_CHECK_VALUE(at::isFloatingType(count.scalar_type()),
                    "binomial only supports floating-point dtypes for count, got: ",
                    count.scalar_type());
  TORCH_CHECK_VALUE(at::isFloatingType(prob.scalar_type()),
                    "binomial only supports floating-point dtypes for prob, got: ",
                    prob.scalar_type());
  Tensor ret = at::empty_like(count, count.options(), at::MemoryFormat::Contiguous);
  if (count.numel() == 0) {
    return ret;
  }

  auto count_contig = count.contiguous();
  auto prob_contig = prob.expand(count.sizes())
                         .to(count.scalar_type(), /*non_blocking=*/false, /*copy=*/false, MemoryFormat::Contiguous)
                         .contiguous();
  auto mps_gen = get_generator_or_default<MPSGeneratorImpl>(gen_, at::mps::detail::getDefaultMPSGenerator());
  auto stream = getCurrentMPSStream();

  @autoreleasepool {
    auto pso = lib.getPipelineStateForFunc("standard_binomial_" + scalarToMetalTypeString(ret));

    int64_t seed;
    int64_t base_offset;
    constexpr int64_t BINOMIAL_RANDOMS_STRIDE = 32;
    {
      // See Note [Acquire lock when using random generators]
      std::lock_guard<std::mutex> lock(mps_gen->mutex_);
      seed = static_cast<int64_t>(mps_gen->current_seed());
      base_offset = static_cast<int64_t>(mps_gen->get_offset());
      mps_gen->set_offset(base_offset + BINOMIAL_RANDOMS_STRIDE * ret.numel());
    }

    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        auto computeEncoder = stream->commandEncoder();
        [computeEncoder setComputePipelineState:pso];
        mtl_setArgs(computeEncoder, ret, count_contig, prob_contig, std::array<long, 2>{seed, base_offset});
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

  using namespace mps;

  // Small-n fast path: single-threadgroup Fisher-Yates kernel writes the
  // permutation directly into `result`, skipping the keys + argsort + copy
  // pipeline. Threadgroup memory caps us at 4096 uint indices, but the swap
  // loop runs serially on thread 0 — measurement (M4 Max) puts the crossover
  // with the parallel sort path near n = 384 (past that the serial swap loop
  // costs more than launching the sort).
  constexpr int64_t kRandpermSmallThreshold = 384;
  const auto stype = result.scalar_type();
  const bool small_path_supported = (stype == kInt || stype == kLong) && result.is_contiguous();
  if (n <= kRandpermSmallThreshold && small_path_supported) {
    auto mps_gen = get_generator_or_default<MPSGeneratorImpl>(generator, at::mps::detail::getDefaultMPSGenerator());
    auto stream = getCurrentMPSStream();

    @autoreleasepool {
      auto pso = lib.getPipelineStateForFunc("randperm_small_" + scalarToMetalTypeString(result));

      int64_t seed;
      int64_t base_offset;
      // n - 1 uniforms consumed by Fisher-Yates; one Philox round emits 4.
      const int64_t philox_rounds = (n - 1 + 3) / 4;
      {
        std::lock_guard<std::mutex> lock(mps_gen->mutex_);
        seed = static_cast<int64_t>(mps_gen->current_seed());
        base_offset = static_cast<int64_t>(mps_gen->get_offset());
        mps_gen->set_offset(base_offset + philox_rounds);
      }

      dispatch_sync_with_rethrow(stream->queue(), ^() {
        @autoreleasepool {
          auto computeEncoder = stream->commandEncoder();
          [computeEncoder setComputePipelineState:pso];
          uint32_t numel = static_cast<uint32_t>(n);
          mtl_setArgs(computeEncoder, result, std::array<long, 2>{seed, base_offset});
          mtl_setBytes(computeEncoder, numel, 2);
          NSUInteger tg = std::min<NSUInteger>(256, static_cast<NSUInteger>(n));
          [computeEncoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        }
      });
    }
    return result;
  }

  // Large-n path: sort random integer keys, take the argsort permutation, then
  // de-bias. A permutation of [0, n) doesn't need a *full* sort - sorting random
  // keys by only their low bits already orders the distinct keys uniformly; an
  // exact Fisher-Yates shuffle of each equal-key island (randperm_dedup_islands)
  // removes the residual tie-bias so the result is uniform over all n!, matching
  // CPU and CUDA. 24-bit keys keep the sort at 3 radix passes (vs 4 for full
  // 32-bit keys); the islands they leave (~6% of elements at n = 1M) are cheap
  // to dedup. Mirrors CUDA's radix_sort_pairs + randperm_handle_duplicate_keys.
  constexpr int64_t kRandpermRadixThreshold = 1 << 15;
  constexpr int kRandpermKeyBits = 24;
  constexpr int kRandpermRadixPasses = 3; // ceil(kRandpermKeyBits / 8-bit radix)
  if (n >= kRandpermRadixThreshold) {
    Tensor keys = at::empty({n}, result.options().dtype(kInt));
    keys.random_(0, int64_t(1) << kRandpermKeyBits, generator);
    Tensor sorted_keys;
    Tensor perm = randperm_argsort_lowbits_metal(keys, kRandpermRadixPasses, sorted_keys);

    auto mps_gen = get_generator_or_default<MPSGeneratorImpl>(generator, at::mps::detail::getDefaultMPSGenerator());
    auto stream = getCurrentMPSStream();
    @autoreleasepool {
      auto pso = lib.getPipelineStateForFunc("randperm_dedup_islands_" + scalarToMetalTypeString(perm));
      int64_t seed;
      int64_t base_offset;
      {
        std::lock_guard<std::mutex> lock(mps_gen->mutex_);
        seed = static_cast<int64_t>(mps_gen->current_seed());
        base_offset = static_cast<int64_t>(mps_gen->get_offset());
        mps_gen->set_offset(base_offset + n);
      }
      dispatch_sync_with_rethrow(stream->queue(), ^() {
        @autoreleasepool {
          auto computeEncoder = stream->commandEncoder();
          [computeEncoder setComputePipelineState:pso];
          uint32_t numel = static_cast<uint32_t>(n);
          mtl_setArgs(computeEncoder, perm, sorted_keys, std::array<long, 2>{seed, base_offset});
          mtl_setBytes(computeEncoder, numel, 3);
          mtl_dispatch1DJob(computeEncoder, pso, numel);
        }
      });
    }
    result.copy_(perm);
    return result;
  }

  // Mid-n path: uniform-float keys + argsort. The argsort permutation is a
  // uniformly-random permutation of [0, n). When `result` is already int64
  // and contiguous, route the indices straight into `result` via `sort_out`
  // to avoid an extra cast / copy.
  Tensor keys = at::empty({n}, result.options().dtype(kFloat));
  auto keys_iter = at::TensorIterator::borrowing_nullary_op(keys);
  distribution_kernel_mps_impl(keys_iter, 0.0, 1.0, "uniform_dist", 1, generator);
  if (stype == kLong && result.is_contiguous()) {
    Tensor values = at::empty_like(keys);
    at::sort_out(values, result, keys);
  } else {
    Tensor perm = at::argsort(keys);
    if (perm.scalar_type() != stype) {
      perm = perm.to(stype);
    }
    result.copy_(perm);
  }
  return result;
}

// MPS kernel for the with-replacement, multi-sample case. The shared frontend
// in Distributions.cpp handles validation and the no-replacement / single-sample
// fast path (the latter relies on `_assert_async`, now implemented for MPS).
static void multinomial_with_replacement_mps_kernel(Tensor& result,
                                                    const Tensor& self,
                                                    int64_t n_sample,
                                                    std::optional<Generator> generator) {
  auto numCategories = self.size(-1);
  // CDF accumulated in float32 since bfloat16/float16 lose precision summing many small probabilities.
  // Sample u from U[0, total) and search the unnormalized CDF,
  // equivalent to normalizing then sampling u from U[0, 1)
  auto cdf = self.cumsum(-1, /*dtype=*/kFloat);
  auto uniform = at::rand(result.sizes(), generator, self.options().dtype(kFloat))
                     .mul_(cdf.select(-1, numCategories - 1).unsqueeze(-1));
  at::searchsorted_out(result, cdf, uniform);
  result.clamp_(0, numCategories - 1);
}

REGISTER_MPS_DISPATCH(multinomial_with_replacement_stub, &multinomial_with_replacement_mps_kernel)

} // namespace at::native
