#include <functorch/csrc/BatchRulesHelper.h>

namespace at { namespace functorch {

using FactoryType = Tensor (*)(const Tensor&, optional<ScalarType>, optional<Layout>, optional<Device>, optional<bool>, optional<MemoryFormat>);
using RandIntType = Tensor (*)(const Tensor&, optional<ScalarType>, optional<Layout>, optional<Device>, optional<bool>, optional<MemoryFormat>);

#define SINGLE_ARG(...) __VA_ARGS__
#define FACTORY_BATCH_RULE(op) SINGLE_ARG(basic_unary_batch_rule<FactoryType, &op, optional<ScalarType>, optional<Layout>, optional<Device>, optional<bool>, optional<MemoryFormat>>)

TORCH_LIBRARY_IMPL(aten, FT_BATCHED_KEY, m) {
  VMAP_SUPPORT("ones_like", FACTORY_BATCH_RULE(at::ones_like));
  VMAP_SUPPORT("zeros_like", FACTORY_BATCH_RULE(at::zeros_like));
  VMAP_SUPPORT("empty_like", FACTORY_BATCH_RULE(at::empty_like));
  VMAP_SUPPORT("randn_like", FACTORY_BATCH_RULE(at::randn_like));
  VMAP_SUPPORT("rand_like", FACTORY_BATCH_RULE(at::rand_like));
  // Not sure how to add the ones with irregular args to the mix cleanly (i.e. randint takes an extra int parameter)
}
}}

