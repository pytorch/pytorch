---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  execution_timeout: 30
  execution_show_tb: True
  merge_streams: True
---

```{code-cell}
:tags: [remove-cell]
import torch
import header_code
torch._logging.set_logs(graph_breaks=True, graph_code=True)
```

(practical_guide_reduce_recompilation)=

# A Practical Guide to Reducing Recompilations with Dynamic Shapes

This section describes solutions for developers experiencing
recompilations in their models that might be resolved by marking
some dimensions or integers as dynamic. It is not intended for recompilations due
to other reasons such as guards on object IDs or types.

## What does it mean for a size/integer to be dynamic?

Dynamic shapes allow avoiding recompilations by making certain dimensions or integers
dynamic. For example, if a function `f(x)` is compiled with a static size, it will need
recompilation for different sizes:

```{code-cell}
import torch
@torch.compile(dynamic=False)
def f(x):
     return x* x.size()[0]

f(torch.rand(10))
f(torch.rand(20))
f(torch.rand(30))
f(torch.rand(40))
```

In the produced output, you can see that four graphs were generated.
See the corresponding
<a href="../_static/img/dynamic_shapes/tlparse1_dynamic_shapes_false.png" target="_blank">tlparse output</a>.


By making the size dynamic, the function can handle various sizes without recompilation:

```{code-cell}
import torch
@torch.compile(dynamic=True)
def f(x):
     return x* x.size()[0]

f(torch.rand(10))
f(torch.rand(20))
f(torch.rand(30))
f(torch.rand(40))
```

With dynamic shapes enabled, only one graph is created. See the
corresponding
<a href="../_static/img/dynamic_shapes/tlparse2_dynamic_shapes_true.png" target="_blank">tlparse output</a>.

While compilation time differences
are minimal for this small example, more complex use cases would show significant
performance improvements.

(what_is_a_specialization)=
## What is a specialization?

**Specialization** refers to optimizing a computational graph for specific input shapes
by examining shape conditions during control flow. If a branch is taken based on a
shape condition, the graph is tailored for that condition. If a new input doesn't meet
this condition, the system will recompile the graph.

Specialization allows you to create optimized computational graphs for specific input
shapes, which can significantly improve execution speed.

Here is a more concrete example:

```{code-cell}
import torch
@torch.compile(dynamic=True)
def f(x):
    if x.size()[0] == 10:
        return x * 10

    if x.size()[0] <= 30:
        return x*200

    return x*x.size()[0]

f(torch.rand(10))
f(torch.rand(20))
f(torch.rand(30))
f(torch.rand(40))
f(torch.rand(50))
```

In the code above, we specialize that the graph requires an input size of 10, in which
case it will return `x * 10`. If the input size is less than 30, it will return `x * 200`.
In the output, you can see that this creates three graphs.

See the corresponding
<a href="../_static/img/dynamic_shapes/tlparse3_specialization.png" target="_blank">tlparse output</a>.

This is how graphs created for the above function:

```{image} ../_static/img/dynamic_shapes/dynamic_shapes_example_specialization.png
```

## Enabling Dynamic Behavior

There are the following ways to make things dynamic:

* Automatic dynamic
* torch.compile (dynamic=true)
* Profile-Guided Optimization (PGO)
* Compiler Collective
* Users annotations

Read below about each of this options.

### Automatic dynamic

This is the default behavior where Dynamo makes an input dynamic if it
sees different values for it.

### torch.compile (dynamic=true)

This setting forces all sizes and integers to be dynamic, increasing the
chance of encountering dynamic shape bugs. Setting this option is not recommended due to it  being error prone.
It would make every input size dynamic which may result it performance regressions and ultimately increase compilation time.

### Profile-Guided Optimization (PGO)

Profile-Guided Optimization (PGO) extends automatic dynamic across attempts, learning from previous runs to avoid initial static compilations. This means that things marked as dynamic in attempt 1 will remain dynamic in attempt 2 from the first compilation. If attempt 2 encounters different sizes for the same input, they will be marked as dynamic.

For example, for the program discussed earlier:

```python
def f(x):
    return x * x.size()[0]
```

In attempt 0, when we first encounter `f(x)` with `x=10`, we make it
static since it's the initial observation. The second time we encounter `f(x)`
with `x=20`, we have observed two different
values for `x`, so we make it dynamic through automatic dynamic behavior.

In attempt 1, we repeat the process above unless Profile-Guided Optimization (PGO) is enabled.
With PGO, we already know from attempt 0 that `f(x)` should be dynamic, so it is marked as
such the first time we encounter it.

(identifying-dynamic-elements-marked-by-pgo)=
#### Identifying Dynamic Elements Marked by PGO

Use `tlparse` to find line numbers of interest and check for multiple values
seen for inputs.

To determine which elements are marked as dynamic by Profile-Guided Optimization (PGO),
follow these steps using `tlparse`:

1. In the `tlparse` output, identify the line number of the frame of interest. Example:

   ```{image} ../_static/img/dynamic_shapes/tlparse4_pgo.png
   ```

2. Open `local_code` using `put_local_code_state_` or `put_remote_code_state_` for the
   latest frame (for example, 6/1).

   Each `?` indicates that multiple values have been observed for this input.

   For instance, the following output shows that the input `L['m']` has been seen with
   multiple sizes at `size[0]`, but the stride has consistently been 1:

   ```python
   /data/users/bobren/a/pytorch/r2.py:2:func:
   L['m']: fully dynamic scalar or tensor
   L['x']: tensor size=[?] stride=[1]
   L['y']: tensor size=[?] stride=[1]
   L['z']: tensor size=[?] stride=[1]
   ```

```{note}
If an element is marked as dynamic by PGO, it does not guarantee that it will remain dynamic in the graph. Specialization can revert it to a static state.
```

### Compiler Collective

Different ranks can communicate with each other to share observed sizes. In the second
iteration, automatic dynamic uses this information to determine which elements to mark
as dynamic based on inputs seen across all ranks. Check the PR for more details.
To enable this feature, use `enable_compiler_collectives=True` with the `@config.patch`
decorator.

```python
@config.patch(enable_compiler_collectives=True)
```

```{note}
This feature enables the use of collectives during compilation to
synchronize behavior across ranks. Currently, it is used to modify
automatic dynamic shapes behavior by inferring if an input is dynamic
based on whether its size varies across ranks. Since this synchronization
uses collectives, all ranks must run compilation simultaneously; ranks must
not diverge with graph breaks. This is most reliably achieved by ensuring
torch is only run on SPMD programs. Violating this invariant may result in
deadlocking NCCL and encountering a NCCL timeout.
```

(user_annotations)=
### User Annotations

Several tools allow users to explicitly mark specific inputs
by name or code as dynamic. This is useful for avoiding initial compilations that
would eventually become dynamic with the previous tools. It is also used to mark
elements that do not automatically get marked as dynamic, such as neural network
module parameters, and so on. These tools include:

(dynamic_sources_allow_list)=
#### Dynamic Allow List (DYNAMIC_SOURCES)

Use the evnironmental variable `TORCH_COMPILE_DYNAMIC_SOURCES` to pass a configuration
list of source names to be marked as dynamic. For example:
`TORCH_COMPILE_DYNAMIC_SOURCES=L[‘x’],L[‘y’]`
It's easiest to find these dynamic source names using the PGO artifact in `tlparse`.
See {ref}`identifying-dynamic-elements-marked-by-pgo` for more details. You can
copy and paste the dynamic source names from the PGO artifact. This method works
for integers and tensor sizes and has the highest precedence over all other flags
that force static shapes. It will not throw an error if what is marked dynamic
gets specialized or if the provided input does not exist.

#### mark_dynamic(tensor, size)

The `mark_dynamic` function marks a tensor dimension as dynamic and will fail if it
gets specialized. It does not work for integers. Use this function only if you know
all graphs in the frame using this input converge to a single dynamic graph.
Otherwise, you may encounter a misleading constraint violation error.
In such cases, consider using `maybe_mark_dynamic`. Currently, `mark_dynamic`
does not have precedence over `force_parameter_static_shapes = True` or `force_nn_module_property_static_shapes = True`.

#### maybe_mark_dynamic(tensor, size)

The `maybe_mark_dynamic` function shares all properties with `mark_dynamic`
but does not fail if the size gets specialized. Use it for inputs shared by
multiple graphs or if the number of graphs does not converge to one for a specific
frame. For instance, in the example above, use `maybe_mark_dynamic()` because graphs
with sizes 0 and 1 will specialize. However, you can use `mark_dynamic` to ensure
you never specialize.

#### mark_unbacked(tensor, size)

The `mark_unbacked` function marks a tensor dimension as unbacked. It is unlikely
to be the tool you need, but it could be useful if the specialization occurs inside
a condition `guard_size_oblivious(x)`, and if using it removes the specialization.
Ensure it fixes the specialization and does not introduce a data-dependent error
that converts to a graph break at or before the specialization location
you are trying to  avoid. It might be better to use the next option.

#### backed_size_oblivious

Use `backed_size_oblivious` if you want to avoid specialization
during `guard_size_oblivious`and aim to minimize compilations and
specialization. Set a flag to treat backed as unbacked for all checks in the code
that participate in size-oblivious reasoning, which avoids
0/1 specialization for backed elements.

## Reducing Compilations: Step by Step

If you have a model that you can run on your master job and have a `tlparse`,
here's whatyou should do next:

### Step 1: Mark Dynamic Elements

The first step is to reduce initial compilations that are eventually optimized away
by automatic dynamic or PGO. This is straightforward because we know it will work
upfront. If, in one run, a frame starts with static graphs and converges to
dynamic graphs, and if you notice a reduction in the number of compiled
frames in a second (warm) PGO-enabled run, it's likely due to this optimization.

This is a two-step process:

1. Find elements marked as dynamic by PGO or automatic dynamic.
2. Mark them as dynamic using one of the {ref}`user_annotations` tools.
   Using {ref}`dynamic_sources_allow_list` is easiest.

#### How to Identify Elements to Mark as Dynamic

Follow these guidelines:

1. **PGO artifact:** Follow the steps in {ref}`identifying-dynamic-elements-marked-by-pgo`.
2. **Dynamic Logs:** If you have a run with `TORCH_LOGS="+dynamic"`, each
time a new dynamic dimension is allocated, a debug line will specify it
along with the input name.
3. **Compare Graphs:** For frames with reduced compilations across runs,
inspect the Dynamo graphs in the second run or the latest runs in the
cold run. Look for elements marked as dynamic in those graphs. Specifically,
find graphs that are similar (once specialized and once dynamic).

Even without a warm run, you can inspect all graphs for a specific frame
to see if some are similar and converge to a dynamic version.

For example, in the following `tlparse` snapshot, Dynamo graphs 20/0,
20/1, and 20/2 are similar except for different sizes (for example,
graph 20/0 vs. graph 20/2). In the Dynamo graph of 20/2, sizes `s0`,
`s1`, and `s5` are used for `rotary_pos_emb_` and `x`.

```{image} ../_static/img/dynamic_shapes/tlparse5_dynamic_shapes.png
```

```{tip}
Two graphs are considered similar if they have the same sequence of calls for
torch operations and the same tensor inputs. Variations may exist in integer
inputs that could be inlined in the specialized version or arithmetic
computations that only exist in the dynamic version due to inlining in the
static version.
```

### Step 2: Debugging: Identifying Missed Opportunities

The complexity of debugging can vary greatly depending on the issues you
encounter. The end result is often to find a bug, enable a flag, or modify
user/framework code.

#### Finding Similar Graphs

Start by identifying a group of similar graphs that you might want to combine
into one dynamic graph, as discussed in the previous section on comparing
graphs. If you can't find any similar graphs, there's nothing further to do
in this step.

#### Quick Checks: Fail Fast

After finding similar graphs, you want to understand why the have recompilations.
Check the following:

1. **Check Recompile Reasons:** For graphs you believe are similar, click on
`recompile_reason` in the `tlparse` output for the later graph. Ensure the
reason is size-related and not due to other factors. For example, while
in these screenshot the recomplile reason is size-related:

```{image} ../_static/img/dynamic_shapes/tlparse6_size_related_recompilations.png
```

In the one below it is not, which indicates that dynamic shapes won't resolve it:

```{image} ../_static/img/dynamic_shapes/tlparse7_not_size_related_recompilations.png
```

2. **Compare Guards Files:** Ensure there are no guards on non-size-related
elementsthat exist in one graph but not the others.

3. **Early Check for Custom Triton Kernels:** Check if your model calls custom
Triton kernels with constant expression arguments, as these are always
specialized. If your model receives different values for these arguments,
it could be a source of recompilation.


### **Identifying and Fixing Recompilation Causes**

1. **Is Something Not Marked Dynamic but Should Be?** Determine if an input was
marked dynamic and got specialized or was not marked dynamic at all. You can
identify this by:

    * Checking the Dynamo graph - look for `Sym(number)`. For example:

      ```sh
      Sym(256) vs Sym(s0)
      ```

    * Using dynamic logs:

      ```sh
      +launcher.additional_environ=["TORCH_LOGS=+dynamic"]
      create_symbol s2 = 2 for L['self']._modules['cle ...
      ```

    * Reviewing guards files. If a tensor size is dynamic, it will be indicated as `None`:

      ```sh
      | | | | | | | | | | | +- TENSOR_MATCH:check_tensor(L['self']._modules['cle']._modules['compress']._parameters['weight'], Parameter, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=True, size=[None, None], stride=[None, 1])
      ```

2. **Why Is It Not Marked Dynamic?** If you determine an element is not marked dynamic, consider:

    * Checking if it's an `nn` module property, parameter, or field. Verify setting for the flags:
      * `force_parameter_static_shapes = True`
      * `force_nn_module_property_static_shapes = True`
      * `allow_unspec_int_on_nn_module = False`
      * Or using the dynamic allow list to mark it dynamic, which should have the highest priority.

    ```{tip}
    Marking elements one by one can be time-consuming. Initially, flip the flags to
    identify any blocking specializations, then decide how to mark them
    dynamic at the end of the process.
    ```

    * If you feel, like it could be a bug, please file a bug report and mark
    with the `module: dynamic shapes` label. Check the list of known issues in
    [this list](https://github.com/pytorch/pytorch/issues?q=sort%3Aupdated-desc+state%3Aopen+label%3A%22module%3A+dynamic+shapes%22).

3. **Is a Dynamic Element Getting Specialized?** Determine why it is specialized.
It could be due to user code (such as an `if` condition), framework code, or a
call  to a Triton kernel. To identify the reason for specialization:

    * **Using tlparse:** Check the `compilation_metrics` for a specialization section, which will indicate what got specialized and the user and framework stack when it happened. Example:

    ```{image} ../_static/img/dynamic_shapes/tlparse8_compilation_metrics.png
    ```

    The log above indicates that `s0` is specialized to `33` due to the following code:

    ```sh
    `if self.x ==33` at example4.py line 16.
    ```

    * **+Dynamic Logs:** Ppass `+launcher.additional_environ=["TORCH_LOGS=+dynamic"]`. Look for the first specialization, as once a variable is specialized, all dependent variables get specialized too.

    Example log:

    ```sh
    torch/fx/experimental/symbolic_shapes.py:6557] [0/2] eval Eq(s0, 33) [guard added] if self.x ==33:  # example4.py:16 in forward (_dynamo/variables/tensor.py:1242 in evaluate_expr), for more info run with TORCHDYNAMO_EXTENDED_DEBUG_GUARD_ADDED="Eq(s0, 33)"
    V0228 12:04:24.190000 2990033 torch/fx/experimental/symbolic_shapes.py:6000] [0/2] _update_var_to_range s0 = VR[33, 33] (update)
    ```

    The log above indicates that `s0` is specialized to `33` due to the following code:
    ```sh
    if self.x ==33. At example4.py like 16.
    ```

## Where Do I Go From Here?

If you encounter a framework code bug or an issue with specialization,
file an issue so it can be reviewed and potentially improved. If the issue
is within your user code, consider whether you are willing to rewrite your
code to avoid it. Determine if it affects correctness or if it's a redundant
check. If the issue involves a Triton custom kernel with a `constexpr`
argument, evaluate whether you can rewrite it to address the problem.
