import logging
import os
from typing import Any, Dict, List

from torch._inductor.metrics import get_metric_table, is_metric_table_enabled

from .. import config
from ..codecache import PyCodeCache, TritonFuture
from ..utils import cache_on_self, do_bench
from ..virtualized import V
from .common import TensorArg

log = logging.getLogger(__name__)


def get_kernel_argdefs(kernel):
    arg_defs, _, _ = kernel.args.python_argdefs()
    return arg_defs


def get_all_kernel_argdefs(kernels):
    argdefs_list = [get_kernel_argdefs(kernel) for kernel in kernels]
    all_argdefs: Dict[
        Any, None
    ] = {}  # use a dict rather than set to maintain insertion order
    for argdefs in argdefs_list:
        all_argdefs.update({arg: None for arg in argdefs})

    return list(all_argdefs.keys())


def get_numel_argdefs(kernel):
    numel_argdefs = []
    for tree in kernel.range_trees:
        if tree.prefix != "r" or kernel.inside_reduction:
            numel_argdefs.append(f"{tree.prefix}numel")

    return numel_argdefs


class MultiKernelState:
    """
    Maintain state of multi-kernel compilation so we don't define duplicated
    multi-kernel for the same set of sub-kernels.

    V.graph.wrapper_code has a reference to MultiKernelState instance.
    """

    def __init__(self):
        self.subkernel_to_kernel_name = {}

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

        wrapper = V.graph.wrapper_code

        kernel_call_def_code = "\n".join(
            [
                f"""
    def call{idx}(need_clone_args=False):
        args = [{', '.join(get_kernel_argdefs(kernels[idx]))}]
        if need_clone_args:
            args, _ = multi_kernel_call.kernels[{idx}].clone_args(*args)
        multi_kernel_call.kernels[{idx}].run(*args, {', '.join(get_numel_argdefs(kernels[idx]))}, grid=grid, stream=stream)
        """.format(
                    idx
                ).strip(
                    "\n"
                )
                for idx in range(len(kernels))
            ]
        )

        # add subkernel src code hashes to the multi-kernel source code so changing a
        # subkernel implementation will result in a differnt py file for
        # multi-kernel. This makes cache implementation straightforward since
        # we can decide cache file name based on multi-kernel py file name
        # directly.
        #
        # Without the hash added for subkernels, the cache file may be shared by
        # different subkernels which is incorrect.
        subkernel_hashes = "\n".join(
            f"# subkernel{i} code hash: {kernel.code_hash}"
            for i, kernel in enumerate(kernels)
        )

        src_code = f"""
{subkernel_hashes}
def run(multi_kernel_call, {', '.join(get_all_kernel_argdefs(kernels))}, {', '.join(get_numel_argdefs(kernels[0]))}, grid, stream):
{kernel_call_def_code}
    multi_kernel_call.run_with_argless_kernels([call0, call1])
        """  # noqa: B950 line too long
        wrapper.header.splice(
            f"""
        {multi_kernel_name} = MultiKernelCall([
            {", ".join(kernel_names)},
        ],
            '''
        """
        )
        wrapper.header.splice(src_code)
        wrapper.header.splice(
            """
            '''
        )
        """
        )

        return multi_kernel_name


class MultiKernel:
    """
    This class maintains the compile time state for multi kernels.

    Assume we do codegen for a MultiKernel encapsulating kernel1 and kernel2.
    The generated definition for the multi-kernel will looks like:
    ```
    multi_kernel_kernel1 = MultiKernelCall([kernel1, kernel2], multi_kernel_definition_code)
    ```

    Here is an concrete example: https://gist.github.com/shunting314/d9f3fb6bc6cee3dbae005825ca196d39
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

    def call_kernel(self, kernel_name):
        """
        Collect the union of arguments from all subkernels as the arguments
        for the multi-kernel.
        """
        assert kernel_name == self.kernel_name
        call_args_list = [kernel.get_call_args() for kernel in self.kernels]
        all_call_args: Dict[
            Any, None
        ] = {}  # use a dict rather than set to maintain insertion order
        for call_args in call_args_list:
            all_call_args.update({arg: None for arg in call_args})

        all_call_args = list(all_call_args.keys())
        grid: List[Any] = []

        # numels for all subkernels should be the same. Use kernels[0] here
        self.kernels[0].add_numel_to_call_args_and_grid(
            kernel_name, all_call_args, grid
        )
        grid = V.graph.wrapper_code.generate_default_grid(kernel_name, grid)

        V.graph.wrapper_code.generate_kernel_call(
            self.kernel_name,
            all_call_args,
            grid,
            V.graph.scheduler.current_device.index,
        )

    def codegen_nan_check(self):
        wrapper = V.graph.wrapper_code
        seen = set()
        for k in self.kernels:
            _, call_args, arg_types = k.args.python_argdefs()
            for arg, arg_type in zip(call_args, arg_types):
                if arg in seen:
                    continue
                seen.add(arg)
                if isinstance(arg_type, TensorArg):
                    line = f"assert not {arg}.isnan().any().item()"
                    wrapper.writeline(line)
                    line = f"assert not {arg}.isinf().any().item()"
                    wrapper.writeline(line)

    @property
    def removed_buffers(self):
        return set.intersection(*[k.removed_buffers for k in self.kernels])

    @property
    def inplaced_to_remove(self):
        return set.intersection(*[k.inplaced_to_remove for k in self.kernels])

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

    def __init__(self, kernels, src_code):
        assert len(kernels) >= 2
        self._kernels = kernels

        self._run = PyCodeCache.load(src_code).run
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

    def cache_file_path(self):
        py_file_path = self._run.__globals__["__file__"]
        return os.path.splitext(py_file_path)[0] + ".picked_kernel"

    def load_cache(self):
        assert self.picked_kernel is None
        path = self.cache_file_path()
        if os.path.exists(path):
            with open(path) as fd:
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
        with open(path, "w") as fd:
            fd.write(str(self.picked_kernel))
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
            if isinstance(kernel, TritonFuture):
                self._kernels[i] = kernel.result()

        return self._kernels

    def run(self, *args, **kwargs):
        self._run(self, *args, **kwargs)

    @staticmethod
    def benchmark_sub_kernels(kernel_calls):
        """
        Benchmark all the sub kernels and return the execution time
        (in milliseconds) for each of time.

        Unit test may mock this method to force a specific kernel to
        be picked.
        """
        return [
            do_bench(lambda: kernel_call(True), rep=40, fast_flush=True)
            for kernel_call in kernel_calls
        ]

    def run_with_argless_kernels(self, kernel_calls):
        if self.picked_kernel is None:
            timings = self.benchmark_sub_kernels(kernel_calls)
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

            def get_kernel_path(k):
                return k.fn.fn.__code__.co_filename

            get_metric_table("persistent_red_perf").add_row(
                lambda: {
                    "kernel1_name": get_kernel_path(self.kernels[0]),
                    "kernel2_name": get_kernel_path(self.kernels[1]),
                    "kernel1_latency": timings[0],
                    "kernel2_latency": timings[1],
                    "size_hints": k0.size_hints,
                    "reduction_hint": k0.inductor_meta.get("reduction_hint"),
                    "speedup": timings[1] / timings[0],
                }
            )
            if not self.disable_cache:
                self.store_cache()
        kernel_calls[self.picked_kernel]()
