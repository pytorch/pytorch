
einops 0.6.1 is buggy - it added "Add allow_ops_in_compiled_graph to support torch.compile"
https://github.com/arogozhnikov/einops/pull/251/files

https://github.com/arogozhnikov/einops/releases/tag/v0.6.1



0.7.0rc1 still buggy - it added " automatically register torch ops in torchdynamo"
https://github.com/arogozhnikov/einops/pull/265

https://github.com/arogozhnikov/einops/releases/tag/v0.7.0rc1



0.7.0rc2 works fine - it added "cover dynamic shapes in torch.compile, introduce fallback if shape was not cacheable"

https://github.com/arogozhnikov/einops/pull/275

so, before 0.7.0rc2 there was an unhandled TypeError in _apply_recipe when "shape or one of passed axes lengths is not hashable (i.e. they are symbols)". when this error is thrown, they fallback to _result = _reconstruct_from_shape_uncached(recipe, backend.shape(tensor), axes_lengths)


ok so:
- we want to make torch.compile work for einops < 0.7.0rc2
- einops >= 0.7.0rc2 works fine, catches TypeError and runs _reconstruct_from_shape_uncached to return correct result
- einops < 0.7.0rc2 doesn't catch TypeError which is propagated to pytorch and graph breaks

TypeError in einops is thrown because "shape or one of passed axes lengths is not hashable (i.e. they are symbols)" - 'unhashable type: non-nested SymInt'

I need to understand what throws the error - is it torch ran inside of einops or einops ran inside of torch.compile?

the problem seem to be with dynamic shapes

possibilities to prevent graph break:
- return _correct_ hash in SymInt.__hash__ when not self.node.is_nested_int() - ezyang's "return hash(builtins.int(self))" doesn't work and results with an error that is widely spread (and expected) in many different places in pytorch - "SymIntArrayRef expected to contain only concrete integers". When I return a hardcoded value `1`, the og error doesn't appear, although lru_cache is definitely broken under the hood
