# mypy: allow-untyped-defs
import functools
import logging
import os
import pathlib

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

    def define_kernel(self, kernels):
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
        """
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

        buf = self.kernel_defs
        buf.writeline("")
        buf.writeline(
            f"{multi_kernel_name} = async_compile.multi_kernel({multi_kernel_name!r}, ["
        )
        with buf.indent():
            for name in kernel_names:
                buf.writeline(f"{name},")
        buf.writeline("])")

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

        # numels for all subkernels should be the same. Use kernels[0] here
        self.kernels[0].add_numel_to_call_args(kernel_name, call_args, arg_types)

        for ws in self.kernels[0].args.workspace_args:
            V.graph.wrapper_code.generate_workspace_allocation(ws)

        V.graph.wrapper_code.generate_kernel_call(
            kernel_name,
            call_args,
            arg_types=arg_types,
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

    def __init__(self, multi_kernel_name, kernels):
        assert len(kernels) >= 2
        self._kernels = kernels
        self.multi_kernel_name = multi_kernel_name

        self.disable_cache = os.environ.get(
            "TORCHINDUCTOR_DISABLE_MULTI_KERNEL_CACHE"
        ) == "1" or is_metric_table_enabled("persistent_red_perf")

        self.picked_kernel = None
        if config.triton.multi_kernel > 1:
            # manually force a subkernel to ease perf testing
            picked_by_config = config.triton.multi_kernel - 2
            assert picked_by_config < len(self._kernels)
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
                self.picked_kernel = int(fd.read())
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

        def wrap_fn(kernel):
            def inner():
                args_clone, kwargs_clone = kernel.clone_args(*args, **kwargs)
                return kernel.run(*args_clone, **kwargs_clone)

            return inner

        return [
            benchmarker.benchmark_gpu(wrap_fn(kernel), rep=40)
            for kernel in self.kernels
        ]

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
        self.run = self.kernels[self.picked_kernel].run  # type: ignore[method-assign]
        self.run(*args, **kwargs)

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
