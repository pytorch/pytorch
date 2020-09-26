#include <torch/library.h>
#include <ATen/core/boxing/KernelFunction.h>

using torch::CppFunction;

namespace at {

// Note: [DispatchKey::VmapMode usage]
// Whenever we're inside a vmap, all Tensors dispatch on this key. At the moment,
// this key is used to disable random operations inside of vmap. If you are looking
// for Batching Rules, those are registered with DispatchKey::Batched instead.
//
// Note: [Ambiguity of random operations inside vmap]
// Random operations have an ambiguity where it isn't clear if they should
// apply the same randomness or apply different randomness. For example:
//
// >>> vmap(lambda t: torch.rand(1))(torch.zeros(5))
// Should the above return the same random number 5 times, or a different one?
//
// We haven't made a decision on that yet so we are temporarily banning random
// operations inside of vmap while we gather user feedback.

template <typename... Args> Tensor unsupportedRandomOp(Args... args) {
  TORCH_CHECK(false, "vmap: We do not yet support calling random operations inside of vmap. ",
              "Please perform random operations outside of vmap as a workaround");
}

template <typename... Args> Tensor& unsupportedRandomOp_(Args... args) {
  TORCH_CHECK(false, "vmap: We do not yet support calling random operations inside of vmap. ",
              "Please perform random operations outside of vmap as a workaround");
}

TORCH_LIBRARY_IMPL(_, VmapMode, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(aten, VmapMode, m) {
  // NB: I'd really like to register a special kernel like
  // CppFunction::makeNamedNotSupported() to avoid listing out the types of everything.
  // However, registering e.g. CppFunction::makeNamedNotSupported() as an implementation
  // only works for operators that support boxing.
#define TENSOROPTIONS c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>, c10::optional<bool>

  // random operations (out-of-place)
  m.impl_UNBOXED("bernoulli", unsupportedRandomOp<const Tensor&, optional<Generator>>);
  m.impl_UNBOXED("bernoulli.out", unsupportedRandomOp_<Tensor&, const Tensor&, optional<Generator>>);
  m.impl_UNBOXED("bernoulli.p", unsupportedRandomOp<const Tensor&, double, optional<Generator>>);
  m.impl_UNBOXED("bernoulli_.Tensor", unsupportedRandomOp_<Tensor&, const Tensor&, optional<Generator>>);
  m.impl_UNBOXED("bernoulli_.float", unsupportedRandomOp_<Tensor&, double, optional<Generator>>);

  m.impl_UNBOXED("cauchy_", unsupportedRandomOp_<Tensor&, double, double, optional<Generator>>);
  m.impl_UNBOXED("exponential_", unsupportedRandomOp_<Tensor&, double, optional<Generator>>);
  m.impl_UNBOXED("geometric_", unsupportedRandomOp_<Tensor&, double, optional<Generator>>);
  m.impl_UNBOXED("log_normal_", unsupportedRandomOp_<Tensor&, double, double, optional<Generator>>);
  m.impl_UNBOXED("multinomial", unsupportedRandomOp<const Tensor&, int64_t, bool, optional<Generator>>);
  m.impl_UNBOXED("multinomial.out", unsupportedRandomOp_<Tensor&, const Tensor&, int64_t, bool, optional<Generator>>);

  m.impl_UNBOXED("normal.Tensor_float", unsupportedRandomOp<const Tensor&, double, optional<Generator>>);
  m.impl_UNBOXED("normal.Tensor_float_out", unsupportedRandomOp_<Tensor&, const Tensor&, double, optional<Generator>>);
  m.impl_UNBOXED("normal.float_Tensor_out", unsupportedRandomOp_<Tensor&, double, const Tensor&, optional<Generator>>);
  m.impl_UNBOXED("normal.float_Tensor", unsupportedRandomOp<double, const Tensor&, optional<Generator>>);
  m.impl_UNBOXED("normal.Tensor_Tensor", unsupportedRandomOp<const Tensor&, const Tensor&, optional<Generator>>);
  m.impl_UNBOXED("normal.Tensor_Tensor_out", unsupportedRandomOp_<Tensor&, const Tensor&, const Tensor&, optional<Generator>>);
  m.impl_UNBOXED("normal.float_float", unsupportedRandomOp<double, double, IntArrayRef, optional<Generator>, const TensorOptions&>);
  m.impl_UNBOXED("normal.float_float_out", unsupportedRandomOp_<Tensor&, double, double, IntArrayRef, optional<Generator>>);
  m.impl_UNBOXED("normal_", unsupportedRandomOp_<Tensor&, double, double, optional<Generator>>);

  m.impl_UNBOXED("poisson", unsupportedRandomOp<const Tensor&, optional<Generator>>);

  m.impl_UNBOXED("random_.from", unsupportedRandomOp_<Tensor&, int64_t, optional<int64_t>, optional<Generator>>);
  m.impl_UNBOXED("random_.to", unsupportedRandomOp_<Tensor&, int64_t, optional<Generator>>);
  m.impl_UNBOXED("random_", unsupportedRandomOp_<Tensor&, optional<Generator>>);

  m.impl_UNBOXED("rand_like", unsupportedRandomOp<const Tensor&, TENSOROPTIONS, optional<MemoryFormat>>);
  m.impl_UNBOXED("randn_like", unsupportedRandomOp<const Tensor&, TENSOROPTIONS, optional<MemoryFormat>>);

  m.impl_UNBOXED("randint_like", unsupportedRandomOp<const Tensor&, int64_t, TENSOROPTIONS, optional<MemoryFormat>>);
  m.impl_UNBOXED("randint_like.low_dtype", unsupportedRandomOp<const Tensor&, int64_t, int64_t, TENSOROPTIONS, optional<MemoryFormat>>);

  m.impl("rand", unsupportedRandomOp<IntArrayRef, TENSOROPTIONS>);
  m.impl_UNBOXED("rand.generator", unsupportedRandomOp<IntArrayRef, optional<Generator>, const TensorOptions&>);
  m.impl_UNBOXED("rand.names", unsupportedRandomOp<IntArrayRef, optional<DimnameList>, const TensorOptions&>);
  m.impl_UNBOXED("rand.generator_with_names", unsupportedRandomOp<IntArrayRef, optional<Generator>, optional<DimnameList>, const TensorOptions&>);
  m.impl_UNBOXED("rand.out", unsupportedRandomOp_<Tensor&, IntArrayRef>);
  m.impl_UNBOXED("rand.generator_out", unsupportedRandomOp_<Tensor&, IntArrayRef, optional<Generator>>);

  m.impl("randn", unsupportedRandomOp<IntArrayRef, TENSOROPTIONS>);
  m.impl_UNBOXED("randn.generator", unsupportedRandomOp<IntArrayRef, optional<Generator>, const TensorOptions&>);
  m.impl_UNBOXED("randn.names", unsupportedRandomOp<IntArrayRef, optional<DimnameList>, const TensorOptions&>);
  m.impl_UNBOXED("randn.generator_with_names", unsupportedRandomOp<IntArrayRef, optional<Generator>, optional<DimnameList>, const TensorOptions&>);
  m.impl_UNBOXED("randn.out", unsupportedRandomOp_<Tensor&, IntArrayRef>);
  m.impl_UNBOXED("randn.generator_out", unsupportedRandomOp_<Tensor&, IntArrayRef, optional<Generator>>);

  m.impl("randperm", unsupportedRandomOp<int64_t, TENSOROPTIONS>);
  m.impl_UNBOXED("randperm.generator", unsupportedRandomOp<int64_t, optional<Generator>, const TensorOptions&>);
  m.impl_UNBOXED("randperm.out", unsupportedRandomOp_<Tensor&, int64_t>);
  m.impl_UNBOXED("randperm.generator_out", unsupportedRandomOp_<Tensor&, int64_t, optional<Generator>>);

  m.impl("randint", unsupportedRandomOp<int64_t, IntArrayRef, TENSOROPTIONS>);
  m.impl_UNBOXED("randint.generator", unsupportedRandomOp<int64_t, IntArrayRef, optional<Generator>, const TensorOptions&>);
  m.impl("randint.low", unsupportedRandomOp<int64_t, int64_t, IntArrayRef, TENSOROPTIONS>);
  m.impl_UNBOXED("randint.low_generator", unsupportedRandomOp<int64_t, int64_t, IntArrayRef, optional<Generator>, const TensorOptions&>);
  m.impl_UNBOXED("randint.out", unsupportedRandomOp_<Tensor&, int64_t, IntArrayRef>);
  m.impl_UNBOXED("randint.generator_out", unsupportedRandomOp_<Tensor&, int64_t, IntArrayRef, optional<Generator>>);
  m.impl_UNBOXED("randint.low_out", unsupportedRandomOp_<Tensor&, int64_t, int64_t, IntArrayRef>);
  m.impl_UNBOXED("randint.low_generator_out", unsupportedRandomOp_<Tensor&, int64_t, int64_t, IntArrayRef, optional<Generator>>);

  m.impl_UNBOXED("uniform_", unsupportedRandomOp_<Tensor&, double, double, optional<Generator>>);

#undef TENSOROPTIONS
}


} // namespace at
