(zero-one-specialization)=
# The Zero-One Specialization Problem

Before you read this section, you should understand the basics of
dynamic shapes. Make sure you have read the following sections:

* {ref}`dynamic_shapes`
* {ref}`torch.export`
* {ref}`what_is_a_specialization`

In `torch.compile`, we specialize automatically on inputs with sizes
0 or 1 and assume that any remaining inputs cannot be 0 or 1. This
simplifies tasks like contiguity and broadcasting checks, as it
avoids adding extra guards. However, this can cause problems for
sparse models with many symbolic integers that in practice have
tensors of size 0, 1, or 2. For example, consider when you a task is
something like collecting likes on page.

While it's possible to stop specializing on 0/1 upfront, executing
normal PyTorch code often reintroduces 0/1 guards, as many conditions
in PyTorch check for values being 0 or 1. Although models that work
for `N > 2` often generalize to `N = 1`, this isn't guaranteed, especially
with symbolic variables. For example, in hand tracking, a dimension
size of `N = 0`, `1`, or `2` may lead to different graph behaviors.
Simply hoping that the `N > 2` model generalizes can expose soundness issues.

## Size-Oblivious Reasoning

The main technique to address the 0/1 specialization problem is to
use size-oblivious reasoning, which involves treating tensor dimensions
as if they are greater than or equal to `2`, even when they are actually `0` or `1`.
In PyTorch, this is achieved by using {ref}`unbacked SymInts <backed-vs-unbacked-symints>`.
When using unbacked SymInts, PyTorch would emulate the behavior as if the
tensor dimension is `>=2` even if it is `0` or `1`.

Consider the following example:

```python
torch.squeeze(torch.randn(s0, 20, 1))
```

{func}`torch.squeeze` checks if the tensor is of size 1 and removes the dimension.

If you mark `s0` as unbacked `SymInt`, the tensor will have a size of `[s0, 20]`,
even if `s0 = 1` at runtime. This approach avoids recompilation, though it
diverges from the eager execution behavior, which would yield `[20]`.
This is particularly useful in scenarios involving sparse models.

For example, your model tracks restaurants visits and you have a total of 100
restaurants to track, where many entries might be 0 or 1. In such cases, avoiding
recompilation for each tensor change prevents potentially 200 recompilations,
making the use of unbacked `SymInts` advantageous.


deviation from the eager behavior because with eager you would get `[20]`.
The main reason you might care about this is in sparse models, such as
those representing how many times you went to a restaurant. Say you have
100 restaurants; a lot of them would have a value 0 or 1. In those cases,
you don't want to recompile every time the tensor changes, because this
would create 200 recompilations, and that's when you would use unbacked.
However, you might avoid using unbacked `SymInts` if there are specific
runtime performance requirements or guards, as they can be less efficient.

```{seealso}
* {ref}`dynamic_shapes`
* {ref}`torch.export`
* {ref}`what_is_a_specialization`
* {ref}`backed-vs-unbacked-symints`
