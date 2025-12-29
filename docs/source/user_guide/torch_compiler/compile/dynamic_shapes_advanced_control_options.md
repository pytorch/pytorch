(dynamic_shapes_advanced_control_options)=
# Advanced Options to Control Dynamic Behavior

PyTorch provides several advanced options to control dynamic behavior.
These options requires a deep understanding of the PyTorch internals and
may inlvolve setting additional tools. These options include:

* Profile-Guided Optimization (PGO) is a technique that allows the compiler
  to save automatic dynamic decisions and reuse them across jobs.
* Compiler Collective is a feature that is used to modify automatic dynamic
  shapes behavior by inferring if an input is dynamic based on whether
  its size varies across ranks.

## Profile-Guided Optimization (PGO)

Profile-Guided Optimization (PGO) enhances automatic dynamic by sharing profiling decisions across runs of your model. Specifically, it serializes all the choices made by automatic dynamic into a file on disk. You can then copy this file—or store it in a centralized metadata service like S3—and reuse it on other machines to ensure consistent behavior across environments.

For the purposes of the rest of this tutorial, you can use the following environmental variables to turn on PGO locally `TORCH_COMPILE_JOB_ID=1 TORCH_DYNAMO_AUTOMATIC_DYNAMIC_LOCAL_PGO=1`

(identifying-dynamic-elements-marked-by-pgo)=
### Identifying Dynamic Elements Marked by PGO

Use `tlparse` to find line numbers of interest and check for multiple values
seen for inputs.

To determine which elements are marked as dynamic by Profile-Guided Optimization (PGO),
follow these steps using `tlparse`:

1. In the `tlparse` output, identify the line number of the frame of interest. Example:

   ```{image} ../../../_static/img/dynamic_shapes/tlparse4_pgo.png
   ```

2. Open `local_code` using `put_local_code_state_` or `put_remote_code_state_` for the
   latest frame (for example, 6/1).

   Each `?` indicates that multiple values have been observed for this input.

   For instance, the following output shows that the input `L['m']` has been seen with
   multiple sizes at `size[0]`, but the stride has consistently been 1:

   ```
   /data/users/bobren/a/pytorch/r2.py:2:func:
   L['m']: fully dynamic scalar or tensor
   L['x']: tensor size=[?] stride=[1]
   L['y']: tensor size=[?] stride=[1]
   L['z']: tensor size=[?] stride=[1]
   ```

```{note}
If an element is marked as dynamic by PGO, it does not guarantee that it will remain dynamic in the graph. Specialization can revert it to a static state.
```

## Compiler Collective

Different ranks can communicate with each other to share observed sizes. In the second
iteration, automatic dynamic uses this information to determine which elements to mark
as dynamic based on inputs seen across all ranks. Check this [PR](https://github.com/pytorch/pytorch/pull/130935) for more details.
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
2. Mark them as dynamic using one of the {ref}`user_annotations`.

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

```{image} ../../../_static/img/dynamic_shapes/tlparse5_dynamic_shapes.png
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

```{image} ../../../_static/img/dynamic_shapes/tlparse6_size_related_recompilations.png
```

In the one below it is not, which indicates that dynamic shapes won't resolve it:

```{image} ../../../_static/img/dynamic_shapes/tlparse7_not_size_related_recompilations.png
:width: 500px
:align: center
```

2. **Compare Guards Files:** Ensure there are no guards on non-size-related
elementsthat exist in one graph but not the others.

3. **Early Check for Custom Triton Kernels:** Check if your model calls custom
Triton kernels with `tl.constexpr` arguments, as these are always
specialized. If your model receives different values for these arguments,
it could be a source of recompilation.


## **Identifying and Fixing Recompilation Causes**

1. **Is Something Not Marked Dynamic but Should Be?** Determine if an input was
marked dynamic and got specialized or was not marked dynamic at all. You can
identify this by:

    * Checking the Dynamo graph - look for `Sym(number)`. For example:

      ```
      Sym(256) vs Sym(s0)
      ```

    * Using dynamic logs:

      ```
      ["TORCH_LOGS=+dynamic"]
      create_symbol s2 = 2 for L['self']._modules['cle ...
      ```

    * Reviewing guards files. If a tensor size is dynamic, it will be indicated as `None`:

      ```
      TENSOR_MATCH:check_tensor(L['self'].x._parameters['weight']], Parameter, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=True, size=[None, None], stride=[None, 1])
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

    ```{image} ../../../_static/img/dynamic_shapes/tlparse8_compilation_metrics.png
    ```

    The log above indicates that `s0` is specialized to `33` due to the following code:

    ```
    `if self.x ==33` at example4.py line 16.
    ```

    * **+Dynamic Logs:** pass `["TORCH_LOGS=+dynamic"]`. Look for the first specialization, as once a variable is specialized, all dependent variables get specialized too.

    Example log:

    ```
    torch/fx/experimental/symbolic_shapes.py:6557] [0/2] eval Eq(s0, 33) [guard added] if self.x ==33:  # example4.py:16 in forward (_dynamo/variables/tensor.py:1242 in evaluate_expr), for more info run with TORCHDYNAMO_EXTENDED_DEBUG_GUARD_ADDED="Eq(s0, 33)"
    V0228 12:04:24.190000 2990033 torch/fx/experimental/symbolic_shapes.py:6000] [0/2] _update_var_to_range s0 = VR[33, 33] (update)
    ```

    The log above indicates that `s0` is specialized to `33` due to the following code:
    ```
    if self.x ==33. At example4.py like 16.
    ```
