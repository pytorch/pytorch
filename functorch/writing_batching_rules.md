So, you want to write some batching rules? This is the guide to get started :)

First off, what are batching rules and why do we need so many of them? Well, to understand that, we need to understand how vmap works.

### How does vmap work?
Vmap is a function transform (pioneered by Jax) that allows one to batch functions. That is, given a function `f(x: [N]) -> [N]`, `vmap(f)` now transforms the signature to be `f(x: [B, N]) -> [B, N]`. That is - it adds a batch dimension to both the input and the output of the function.

This guide will gloss over all the cool things you can do this (there are many!), so let's focus on how we actually implement this.

One misconception is that this is some magic compiler voodoo, or that it is inherently some function transform. It is not - and there's another framing of it that might make it more clear.

Instead of providing `vmap`, imagine that we provide a `BatchedTensor` instead. This `BatchedTensor` wraps a `Tensor[B, N, M]`. *But*, to all the users of this tensor, it looks like a `Tensor[N, M]` (that is, without the `B` dimension). Then, when operations are done on this tensor, it transforms that operation to broadcast over the additional `B` dimension as well.

For example, let's say that we wanted to sum a `BatchedTensor` with shape `[5]` - that is, `torch.sum(x)`. This would give us back a `BatchedTensor` with shape `[]` (i.e. a scalar tensor). **But**, in reality, this is actually a `Tensor` with shape `[B]`. Instead of running `torch.sum(x: [5])`, we ran `torch.sum(x: [B, 5], dim=1)`. In other words, we transformed the sum operation so that instead of summing the whole tensor, it summed all the dimensions *except* the batch dimension.

That is how `vmap` works. For every single operator, we define how to transform that operator to broadcast over an additional batch dimension.

### Basic Batching Rule (unsqueeze)
Let's take a look at our batching rule API. For some reference, the function signature for unsqueeze is `unsqueeze(Tensor(a) self, int dim) -> Tensor(a)`. This can be found [here](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/functorch/BatchRulesViews.cpp).
```
std::tuple<Tensor,optional<int64_t>> unsqueeze_batch_rule(
    const Tensor& self,
    optional<int64_t> self_bdim,
    int64_t dim) {
  auto self_ = moveBatchDimToFront(self, self_bdim);
  auto rank = rankWithoutBatchDim(self, self_bdim);
  dim = maybe_wrap_dim(dim, rank + 1) + 1;
  return std::make_tuple(self_.unsqueeze(dim), 0);
}
```
Now, let's look at each part individually.
```
std::tuple<Tensor,optional<int64_t>> unsqueeze_batch_rule(
    const Tensor& self,
    optional<int64_t> self_bdim,
    int64_t dim) {
```
For the most part, the function signature for a batching rule is identical to the function signature for the operator. The only difference is that for each `Tensor` (both in the input and the output), we have an additional `optional<int64_t>`. This is the batch dimension. In the previous explanation, we implicitly assumed that the batch dimension was always at 0, but we allow for batch dimensions to be on arbitrary dimensions. The `optional` part reflects that not all tensors are batched - if a function takes multiple tensors then it's possible for only one of them to be a `BatchedTensor`. Note, however, that we guarantee that at least one tensor will always have a batch dimension.

```
  auto self_ = moveBatchDimToFront(self, self_bdim);
  auto rank = rankWithoutBatchDim(self, self_bdim);
  dim = maybe_wrap_dim(dim, rank + 1) + 1;
```
For `unsqueeze(x, dim)`, the strategy for the batching rule is pretty simple. We first move the batching dimension to the front. Then, instead of doing `unsqueeze(x, dim)`, we do `unsqueeze(x, dim + 1)` (since there's now an extra bdim).

```
return std::make_tuple(self_.unsqueeze(dim), 0);
```
Now, we return a tuple of the tensor along with its batch dimension (which is now 0 since we moved it to the front).

```
VMAP_SUPPORT(unsqueeze, unsqueeze_batch_rule);
```
Finally, we add support for it by using the `VMAP_SUPPORT` macro.

You may need to use the `VMAP_SUPPORT2` macro if the operator has an overload name.

### Implementing multiple batching rules with boxed fallbacks or templates
Often, we find that large classes of operators have similar patterns of batching rules. For example, every single pointwise op has a similar pattern. In that case, it's a bit ridiculous to separately write a batching rule for those situations.

In those cases, we have 2 primary tools - templates and boxed fallbacks. For example, we've written a boxed fallback that covers many reductions (see the [reduction batching rules](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/functorch/BatchRulesReduceOps.cpp)).

There are 3 primary boxed fallbacks that we've used (I'll refer to the macros here). If you feel that there's any pattern that we could/should abstract away, feel free to post an issue.

1. `POINTWISE_BOXED`: Handles pointwise ops. Takes all tensors in the arguments, moves batch dimensions to the front, and unsqueezes all tensors so that they broadcast.
2. `REDUCTION_BOXED`: Handles reduction ops. Moves batch dimension to the front, and then modifies the dim argument so that it works with the extra batch dimension. For example, if the dim is an integer, then we add one. If it's a dimarray, then we add one to all entries (unless it's empty!, in which case we fill in all the entries except 0).
3. `VARIADIC_BDIMS_BOXED`: Handles ops that already natively support arbitrary batch dimensions. For example, if it supports `[B1,B2,..., N]`. In this case, we can simply move the batch dimension to the front and we're done!

### Sidestepping batching rules by decomposing operators
Sometimes, it's difficult to implement a batching rule by transforming it into another operator. For example, `trace`. In that case, instead of transforming the operator, we can simply decompose it.

```
Tensor trace_decomp(const Tensor& self) {
  return at::sum(at::diagonal(self));
}
...
m.impl("trace", trace_decomp);
```
In general, this reduces the performance, since instead of launching one kernel we're launching multiple. So, we generally try to avoid this option :)

### Testing your batching rule
We generally use OpInfos to test our batching rules. OpInfos are great since they let us test the same operator in many different ways.

In general, if the operator you've added a batching rule for has an OpInfo test, that's good enough!

Generally, you can try running `pytest -k op_name` to use `pytest` to find all tests that test your operator. Sometimes, if your operator doesn't match the public API, you need to figure out the public API that corresponds to the operator you've implemented a batching rule for. For example, `torch.where` actually often executes `aten::_s_where` underneath.

Todo: Add more relevant details @zou

## Cool, I'm convinced! And I want to write batching rules! Where do I find some?
There's a couple different resources for finding batching rules to write.

1. [BatchingRegistrations.cpp](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/functorch/LegacyBatchingRegistrations.cpp): This is probably the easiest place to start. These were batching rules that were written with an old API, and thus have a lot of cruft in them that are no longer necessary. Porting these batching rules to using one of the above options is an easy way to get started and help us reduce tech debt :) Once you've gotten your footing with writing batching rules, you can start helping with writing new batching rules.
2. Popular operators. See [1](https://github.com/pytorch/functorch/issues/112), [2](https://github.com/pytorch/functorch/issues/101), [3](https://github.com/pytorch/functorch/issues/102), and [4](https://github.com/pytorch/functorch/issues/102). These contain lists of (user-facing) PyTorch operators sorted by usages, along with whether they have a batching rule implemented or not.
3. [Master List](https://docs.google.com/spreadsheets/d/1Sp4HUjxwMifS5oDQg0yvjqk7hKOpCfKO4jWH4MTGP-k/edit#gid=0). This is the master list of vmap operator support :). It's generated by [this script](op_analysis/gen_data.py). Theoretically, we want to support most of the operators in that list (that aren't composite or out variants).
