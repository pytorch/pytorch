# mypy: allow-untyped-defs
import functools
import logging
import math
import os
import pathlib
from typing import Any, Optional, Union

from torch._inductor.ir import MultiTemplateBuffer
from torch._inductor.metrics import get_metric_table, is_metric_table_enabled
from torch.utils._ordered_set import OrderedSet
from .. import config
from ..codecache import code_hash, CodeCacheFuture, get_path, write_atomic
from ..runtime.benchmarking import benchmarker
from ..utils import cache_on_self, IndentedBuffer
from ..virtualized import V
from .common import TensorArg, WorkspaceArg


log = logging.getLogger(__name__)


class MultiKernelState:
    """
    Maintain state of multi-kernel compilation so we don't define duplicated
    multi-kernel for the same set of sub-kernels.

    V.graph.wrapper_code has a reference to MultiKernelState instance.
    """

    def __init__(self):
        self.subkernel_to_kernel_name = {}
        self.kernel_defs = IndentedBuffer()

    def define_kernel(
        self,
        kernels: list[Any],
        kernel_shape_keys: Optional[
            list[Union[None, tuple[tuple[int, ...], ...]]]
        ] = None,
    ) -> str:
        """
        Previously we name the multi kernel as "multi_kernel_{kernel_names[0]}".
        This has some minor issue.

        E.g. for persistent reduction https://gist.github.com/shunting314/39e7c00ff8bb2055942ed5a3255d61ca ,
        there are 2 flavors of non-persistent reduction:
          https://gist.github.com/shunting314/056d43d35907e87efb883970b35c17d4
        and
          https://gist.github.com/shunting314/02ee753b65c513c54e695626afe682bd

        The only different is cache eviction policy.

        We should name the multi-kernel differently in these 2 cases.

        kernels:
            A list of kernels
        kernel_shape_keys:
            Specified for size-hint multi-kernels.
            Each list element is a shape key, corresponding to the concrete input & output size hints each kernel was tuned for.
        """
        # Prevent circular import
        from ..select_algorithm import TritonTemplateKernel

        kernel_names = tuple(k.kernel_name for k in kernels)
        if kernel_names in self.subkernel_to_kernel_name:
            return self.subkernel_to_kernel_name[kernel_names]

        # name the multi kernel based on the first kernel
        multi_kernel_name = f"multi_kernel_{len(self.subkernel_to_kernel_name)}"
        self.subkernel_to_kernel_name[kernel_names] = multi_kernel_name

        if V.graph.cpp_wrapper and not config.triton.autotune_at_compile_time:
            # we should not generate any python code for multi-kernel during
            # the second pass of cpp-wrapper.
            return multi_kernel_name

        arg_index: dict[int, list[slice]] = {}
        _, call_args, _, arg_types = kernels[0].args.python_argdefs()
        if isinstance(kernels[0], TritonTemplateKernel) and isinstance(
            kernels[0].output_node, MultiTemplateBuffer
        ):
            for i, kernel in enumerate(kernels):
                additional_call_args, _ = kernel.additional_call_args_and_types()
                if i not in arg_index:
                    arg_index[i] = []
                arg_index[i].append(slice(0, len(call_args)))
                arg_index[i].append(
                    slice(
                        len(call_args) + i * len(additional_call_args),
                        len(call_args) + (i + 1) * len(additional_call_args),
                    )
                )
        else:
            kernels[0].add_numel_to_call_args(multi_kernel_name, call_args, arg_types)
            for i in range(len(kernels)):
                arg_index[i] = [slice(0, len(call_args))]

        keyed_by_sizes = kernel_shape_keys is not None
        buf = self.kernel_defs
        buf.writeline("")
        buf.writeline("arg_index = {")
        for key, slice_list in arg_index.items():
            slice_reprs = ", ".join(repr(s) for s in slice_list)
            buf.writeline(f"    {key}: [{slice_reprs}],")
        buf.writeline("}")

        if not keyed_by_sizes:  # no size hint keys, just call with list of kernels
            buf.writeline(
                f"{multi_kernel_name} = async_compile.multi_kernel({multi_kernel_name!r}, ["
            )
            with buf.indent():
                for name in kernel_names:
                    buf.writeline(f"{name},")
            buf.writeline("], arg_index=arg_index)")
        else:  # call with dict[size hint key, kernel]
            assert isinstance(kernels[0], TritonTemplateKernel)
            assert isinstance(kernel_shape_keys, list)
            assert len(kernels) == len(kernel_shape_keys)
            buf.writeline(
                f"{multi_kernel_name} = async_compile.size_hint_multi_kernel({multi_kernel_name!r}, {{"
            )
            with buf.indent():
                for shape_key, name in zip(kernel_shape_keys, kernel_names):
                    buf.writeline(f"{shape_key}: {name},")
            buf.writeline("}, arg_index=arg_index)")

        if config.triton.autotune_at_compile_time:
            V.graph.wrapper_code.src_to_kernel["\n".join(kernel_names)] = (
                multi_kernel_name
            )

        return multi_kernel_name


class MultiKernel:
    """
    This class maintains the compile time state for multi kernels.

    Assume we do codegen for a MultiKernel encapsulating kernel1 and kernel2.
    The generated definition for the multi-kernel will looks like:
    ```
    multi_kernel_kernel1 = MultiKernelCall(
        [kernel1, kernel2], multi_kernel_definition_code
    )
    ```

    Here is a concrete example: https://gist.github.com/shunting314/d9f3fb6bc6cee3dbae005825ca196d39
    """

    def __init__(self, kernels):
        assert len(kernels) >= 2

        self.kernels = kernels
        self.kernel_name = V.graph.wrapper_code.multi_kernel_state.define_kernel(
            kernels
        )

        # need this since some code in inductor check if the kernel object has an args
        # attribute to decide if it's a non-null kernel.
        self.args = object()

    @staticmethod
    def _merge_workspace_args(left: list[WorkspaceArg], right: list[WorkspaceArg]):
        if left == right:
            return left
        result = {x.inner_name: x for x in left}
        for arg in right:
            if arg.inner_name in result:
                result[arg.inner_name] = WorkspaceArg.maximum(
                    result[arg.inner_name], arg
                )
            else:
                result[arg.inner_name] = arg
        return [*result.values()]

    @staticmethod
    def merge_workspaces_inplace(kernels):
        if len(kernels) < 2:
            return
        # All kernels must share the same workspace
        workspace_args = functools.reduce(
            MultiKernel._merge_workspace_args,
            [kernel.args.workspace_args for kernel in kernels],
        )
        for kernel in kernels:
            kernel.args.workspace_args = workspace_args
        return workspace_args

    def call_kernel(self, kernel_name):
        """
        Collect the union of arguments from all subkernels as the arguments
        for the multi-kernel.
        """
        # Prevent circular import
        from ..select_algorithm import TritonTemplateKernel

        assert kernel_name == self.kernel_name
        V.graph.wrapper_code.write_triton_header_once()
        _, call_args, _, arg_types = self.kernels[0].args.python_argdefs()
        for kernel in self.kernels[1:]:
            _, other_call_args, _, other_arg_types = kernel.args.python_argdefs()
            assert call_args == other_call_args, (call_args, other_call_args)
            assert arg_types == other_arg_types

        if V.graph.cpp_wrapper and not config.triton.autotune_at_compile_time:
            # for the second pass of cpp-wrapper codegen, we should call
            # the fast kernel directly
            kernel_name = MultiKernelCall.lookup_choice(self.kernel_name)

        if isinstance(self.kernels[0], TritonTemplateKernel) and isinstance(
            self.kernels[0].output_node, MultiTemplateBuffer
        ):
            # For matmuls the grid arguments are passed in as additional arguments
            # to the kernel run method. These grids change based on the various
            # parameters of the matmul. So we need to pass each kernel's grid into
            # the multi call kernel.
            multi_call_args = call_args
            multi_call_arg_types = arg_types
            for kernel in self.kernels:
                additional_call_args, additional_arg_types = (
                    kernel.additional_call_args_and_types()
                )
                multi_call_args.extend(list(additional_call_args))
                multi_call_arg_types.extend(list(additional_arg_types))
        else:
            # numels for all subkernels should be the same. Use kernels[0] here
            self.kernels[0].add_numel_to_call_args(kernel_name, call_args, arg_types)
            multi_call_args = call_args
            multi_call_arg_types = arg_types

        for ws in self.kernels[0].args.workspace_args:
            V.graph.wrapper_code.generate_workspace_allocation(ws)

        if V.graph.cpp_wrapper:
            # We have already selected the best kernel at compile time
            # so we only have one set of call args. NB: this currently
            # doesn't work with MultiTemplateBuffer kernels. @bobrenjc93
            # will add it in a subsequent PR.
            V.graph.wrapper_code.generate_kernel_call(
                kernel_name, call_args, arg_types=arg_types
            )
        else:
            V.graph.wrapper_code.generate_kernel_call(
                kernel_name, multi_call_args, arg_types=multi_call_arg_types
            )

        for ws in reversed(self.kernels[0].args.workspace_args):
            V.graph.wrapper_code.generate_workspace_deallocation(ws)

    def codegen_nan_check(self):
        wrapper = V.graph.wrapper_code
        seen: OrderedSet[str] = OrderedSet()
        for k in self.kernels:
            _, call_args, precompile_args, _ = k.args.python_argdefs()
            for arg, precompile_arg in zip(call_args, precompile_args):
                if arg in seen:
                    continue
                seen.add(arg)
                if isinstance(precompile_arg, TensorArg):
                    line = f"assert not {arg}.isnan().any().item()"
                    wrapper.writeline(line)
                    line = f"assert not {arg}.isinf().any().item()"
                    wrapper.writeline(line)

    @property
    def removed_buffers(self):
        return OrderedSet.intersection(*[k.removed_buffers for k in self.kernels])

    @property
    def inplaced_to_remove(self):
        return OrderedSet.intersection(*[k.inplaced_to_remove for k in self.kernels])

    @property
    @cache_on_self
    def inplace_update_buffers(self):
        """
        Make sure all kernels have the same inplace update mappings.
        """
        for k in self.kernels[1:]:
            assert k.inplace_update_buffers == self.kernels[0].inplace_update_buffers
        return self.kernels[0].inplace_update_buffers

    def warn_mix_layout(self, kernel_name: str):
        pass


class MultiKernelCall:
    """
    This class is called at run time to actually run the kernel
    """

    def __init__(self, multi_kernel_name, kernels, arg_index):
        assert len(kernels) >= 1
        self._kernels = kernels
        self.multi_kernel_name = multi_kernel_name

        self.disable_cache = os.environ.get(
            "TORCHINDUCTOR_DISABLE_MULTI_KERNEL_CACHE"
        ) == "1" or is_metric_table_enabled("persistent_red_perf")

        self.picked_kernel = None
        self.arg_index = arg_index
        if config.triton.multi_kernel > 1:
            # manually force a subkernel to ease perf testing
            picked_by_config = config.triton.multi_kernel - 2
            assert picked_by_config < len(self._kernels)
            # pyrefly: ignore [bad-assignment]
            self.picked_kernel = picked_by_config
        elif not self.disable_cache:
            self.load_cache()

        self._recorded = False

    def cache_file_path(self):
        key = code_hash(
            ",".join(
                [
                    f"{k.fn.cache_key}{k.size_hints!r}{k.triton_meta!r}"
                    for k in self.kernels
                ]
            )
        )
        _, _, path = get_path(key, "picked_kernel")
        return pathlib.Path(path)

    def load_cache(self):
        assert self.picked_kernel is None
        path = self.cache_file_path()
        if path.exists():
            with path.open() as fd:
                # pyrefly: ignore [bad-assignment]
                self.picked_kernel = int(fd.read())
                # pyrefly: ignore [unsupported-operation]
                assert self.picked_kernel >= 0 and self.picked_kernel < len(
                    self._kernels
                )
                log.debug(
                    "Load picked kernel %d from cache file %s", self.picked_kernel, path
                )

    def store_cache(self):
        assert self.picked_kernel is not None
        path = self.cache_file_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        write_atomic(path, str(self.picked_kernel))
        log.debug("Store picked kernel %d to cache file %s", self.picked_kernel, path)

    @property
    def kernels(self):
        """
        Read results from future.

        This should be called after parallel compilation is done.
        In case you call this before compilation is done,
        it may slow down the parallel compilation.
        """
        for i, kernel in enumerate(self._kernels):
            if isinstance(kernel, CodeCacheFuture):
                self._kernels[i] = kernel.result()

        return self._kernels

    def benchmark_sub_kernels(self, *args, **kwargs):
        """
        Benchmark all the sub kernels and return the execution time
        (in milliseconds) for each of time.

        Unit test may mock this method to force a specific kernel to
        be picked.
        """

        def wrap_fn(kernel, index):
            def inner():
                filtered_args = self._get_filtered_args(args, index)
                args_clone, kwargs_clone = kernel.clone_args(*filtered_args, **kwargs)
                return kernel.run(*args_clone, **kwargs_clone)

            return inner

        return [
            benchmarker.benchmark(
                wrap_fn(kernel, index),
                # Currently the kernel type must be a CachingAutotuner
                device=kernel.device_props.type,
                rep=40,
            )
            for index, kernel in enumerate(self.kernels)
        ]

    def _get_filtered_args(self, args, index):
        """
        We pass in all arguments to all kernels into the MultiKernelCall
        so when invoking a particular kernel we need to filter to only the
        arguments for that specific kernel.
        """

        # This is sometimes invoked at runtime where V.graph is
        # a NullHandler
        if hasattr(V.graph, "cpp_wrapper") and V.graph.cpp_wrapper:
            # for cpp-wrapper, we should not filter the args since
            # we already have chosen a single kernel and arg set.
            return args
        return [item for s in self.arg_index[index] for item in args[s]]

    # record_choice and lookup_choice are helper functions for cpp-wrapper
    # codegen. The first pass use record_choice to keep the choice and
    # the second pass do lookup by calling lookup_choice.
    #
    # An alternative that reused the multi-kernel cache does not work well
    # since during codegen of the second pass, it's very hard to know the
    # path for the cache file. Also reading the cache file need do some IO
    # which can be slower.
    @staticmethod
    def record_choice(multi_kernel_name: str, picked_kernel_name: str):
        """
        Record the multi-kernel choice for cpp-wrapper after autotuning

        We should do nothing if this function is not called during codegen.
        """
        from torch._inductor.graph import GraphLowering

        if not isinstance(V.graph, GraphLowering):
            return

        if not V.graph.record_multi_kernel_choice:
            return

        V.graph.multi_kernel_to_choice[multi_kernel_name] = picked_kernel_name

    @staticmethod
    def lookup_choice(multi_kernel_name: str) -> str:
        # this should always been done during cpp-wrapper codegen
        assert (
            V.graph.record_multi_kernel_choice
            and multi_kernel_name in V.graph.multi_kernel_to_choice
        )
        # there should be no miss
        return V.graph.multi_kernel_to_choice[multi_kernel_name]

    def run(self, *args, **kwargs):
        if self.picked_kernel is None:
            timings = self.benchmark_sub_kernels(*args, **kwargs)
            self.picked_kernel = timings.index(min(timings))
            k0 = self.kernels[0]
            log.debug(
                "pick %dth sub-kernel in %s. Size hints %s. Reduction hint %s. Timings %s",
                self.picked_kernel,
                [k.inductor_meta.get("kernel_name") for k in self.kernels],
                k0.size_hints,
                k0.inductor_meta.get("reduction_hint"),
                timings,
            )
            get_metric_table("persistent_red_perf").add_row(
                functools.partial(self._metrics_table_row, timings)
            )

            if not self.disable_cache:
                self.store_cache()

        if not self._recorded:
            self._recorded = True
            picked_kernel_name = self.kernels[self.picked_kernel].inductor_meta.get(
                "kernel_name"
            )
            assert picked_kernel_name is not None
            self.record_choice(self.multi_kernel_name, picked_kernel_name)

        run = self.kernels[self.picked_kernel].run  # type: ignore[method-assign]
        filtered_args = self._get_filtered_args(args, self.picked_kernel)
        run(*filtered_args, **kwargs)

    def _metrics_table_row(self, timings):
        def get_kernel_path(k):
            return k.fn.fn.__code__.co_filename

        k0 = self.kernels[0]
        row = {
            "size_hints": k0.size_hints,
            "reduction_hint": k0.inductor_meta.get("reduction_hint"),
        }
        max_kernels = 4
        assert len(timings) <= max_kernels
        for i in range(max_kernels):
            if i < len(self.kernels):
                row[f"kernel{i}_path"] = get_kernel_path(self.kernels[i])
                row[f"kernel{i}_latency"] = timings[i]
            else:
                row[f"kernel{i}_path"] = ""
                row[f"kernel{i}_latency"] = ""
        return row


class SizeHintMultiKernel(MultiKernel):
    """
    Version of multi-kernel that generates kernels based on specified size hints.
    Currently only performs 1-d search over hints; doesn't perform combinatorial n-d search
    if n > 1 dynamic dimensions are specified.

    e.g. matmul([s0, s1], [s1, s2]) with size-hints [64, 256] only generates 2 kernels,
    based on tuning shapes ([64, 64], [64, 64]) and ([256, 256], [256, 256])
    """

    def __init__(self, kernels):
        assert isinstance(kernels, dict) and len(kernels) >= 1

        self.kernels, self.kernel_shape_keys = [], []
        for shape_key, kernel in kernels.items():
            self.kernels.append(kernel)
            self.kernel_shape_keys.append(shape_key)
        self.kernel_name = V.graph.wrapper_code.multi_kernel_state.define_kernel(
            self.kernels, self.kernel_shape_keys
        )

        # need this since some code in inductor check if the kernel object has an args
        # attribute to decide if it's a non-null kernel.
        self.args = object()


class SizeHintMultiKernelCall(MultiKernelCall):
    """
    Runtime class for size-hint multi-kernels.
    Instead of having a plain list of kernels to benchmark over, keys them by input & output shapes,
    and optionally perform shape-based selection. The pre-generated kernel is chosen based on the shape keys,
    with the heuristic being log2 l1 distance between the pre-generated / runtime input & output shapes.
    """

    def __init__(self, multi_kernel_name, kernels, arg_index):
        super().__init__(multi_kernel_name, list(kernels.values()), arg_index)
        self._kernel_hints = list(kernels.keys())

        # Caches results for unique shapes.
        self._shape_cache = {}

    def _get_shape_cache_key(self, *args, **kwargs):
        """
        Generate a cache key based on tensor shapes for shape-specialized dispatch.
        """
        shapes = []
        for arg in args:
            if hasattr(arg, "shape"):
                shapes.append(tuple(arg.shape))
        return tuple(shapes)

    def _get_cached_shape_choice(self, cache_key):
        """
        Get cached kernel choice for a specific shape.
        """
        return self._shape_cache.get(cache_key)

    def _cache_shape_choice(self, cache_key, kernel_idx):
        """
        Cache kernel choice for a specific shape.
        """
        self._shape_cache[cache_key] = kernel_idx

    def _dist_heuristic(self, k1, k2):
        """
        log2 L1 distance heuristic for kernel selection.
        """

        def dist(x, y):
            lx = math.log2(x) if x > 0 else -1
            ly = math.log2(y) if y > 0 else -1
            return abs(lx - ly)

        out = 0
        for s1, s2 in zip(k1, k2):
            out += sum(dist(x, y) for x, y in zip(s1, s2))
        return out

    def run(self, *args, **kwargs):
        cache_key = self._get_shape_cache_key(*args, **kwargs)
        cached_choice = self._get_cached_shape_choice(cache_key)
        if cached_choice is not None:
            self.picked_kernel = cached_choice
            log.debug(
                "using cached shape-specialized choice %dth sub-kernel in %s. Cache key: %s",
                self.picked_kernel,
                [k.inductor_meta.get("kernel_name") for k in self.kernels],
                cache_key,
            )
        else:
            self._select_kernel_by_shape(*args, **kwargs)

        if not self._recorded:
            self._recorded = True
            picked_kernel_name = self.kernels[self.picked_kernel].inductor_meta.get(
                "kernel_name"
            )
            assert picked_kernel_name is not None
            self.record_choice(self.multi_kernel_name, picked_kernel_name)

        run = self.kernels[self.picked_kernel].run  # type: ignore[method-assign]
        filtered_args = self._get_filtered_args(args, self.picked_kernel)
        run(*filtered_args, **kwargs)

    def _select_kernel_by_shape(self, *args, **kwargs):
        """
        Benchmark kernels for a particular shape and return the
        best kernel for this shape.
        """
        shape_key = self._get_shape_cache_key(*args, **kwargs)
        dists = [
            self._dist_heuristic(shape_key, key) if key is not None else 2**62
            for key in self._kernel_hints
        ]
        # pyrefly: ignore [bad-assignment]
        self.picked_kernel = dists.index(min(dists))
        self._cache_shape_choice(shape_key, self.picked_kernel)
