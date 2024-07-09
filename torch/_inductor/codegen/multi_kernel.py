# mypy: allow-untyped-defs
import logging
import os
from typing import Any, List

from torch._inductor.metrics import get_metric_table, is_metric_table_enabled

from .. import config
from ..codecache import PyCodeCache, TritonFuture
from ..runtime.runtime_utils import do_bench_gpu
from ..utils import cache_on_self
from ..virtualized import V
from .common import TensorArg

log = logging.getLogger(__name__)


def get_kernel_argdefs(kernel):
    arg_defs, _, _, _ = kernel.args.python_argdefs()
    return arg_defs


def _get_all_args(args_list, arg_types_list=None):
    all_args = max(args_list, key=len)[:]
    arg_types = max(arg_types_list, key=len)[:] if arg_types_list is not None else None
    for args in args_list:
        assert set(args).issubset(set(all_args)), f"{args} v.s. {all_args}"

    return all_args, arg_types


def get_all_kernel_argdefs(kernels):
    """
    The logic here must match with `get_all_call_args`, except no need to get arg_types here
    """
    argdefs_list = [get_kernel_argdefs(kernel) for kernel in kernels]

    return _get_all_args(argdefs_list)[0]


def get_all_call_args(call_args_list, arg_types_list):
    """
    Passed in the call_args for each subkernel and return the call_args for the
    combined multi-kernel.

    Note an algorithm as follows does not always work:
    ```
        all_call_args: Dict[
            Any, None
        ] = {}  # use a dict rather than set to maintain insertion order
        for call_args in call_args_list:
            all_call_args.update({arg: None for arg in call_args})

        all_call_args = list(all_call_args.keys())
    ```
    It will fail if any kernel has the same argument passed in multiple times.
    Check test_pass_same_arg_multi_times in test_multi_kernel.py

    Instead, we pick the longest call args and assert that other call args are
    a subset of it.
    """
    return _get_all_args(call_args_list, arg_types_list)


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

        if V.graph.cpp_wrapper:
            # we should not generate any python code for multi-kernel during
            # the second pass of cpp-wrapper.
            return multi_kernel_name

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
        # subkernel implementation will result in a different py file for
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
    multi_kernel_call.run_with_argless_kernels([call0, call1])"""  # noqa: B950 line too long

        multi_kernel_compile = f"""
{multi_kernel_name} = async_compile.multi_kernel({multi_kernel_name!r}, [
    {", ".join(kernel_names)},
],
'''
{src_code}
'''
)"""
        wrapper.header.splice(multi_kernel_compile)
        if config.triton.autotune_at_compile_time:
            wrapper.kernel_autotune_defs.splice(multi_kernel_compile)

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
        V.graph.wrapper_code.write_triton_header_once()
        call_args_list = []
        arg_types_list = []
        for kernel in self.kernels:
            _, call_args, _, arg_types = kernel.args.python_argdefs()
            call_args_list.append(call_args)
            arg_types_list.append(arg_types)

        all_call_args, arg_types = get_all_call_args(call_args_list, arg_types_list)
        grid: List[Any] = []

        if V.graph.cpp_wrapper:
            # for the second pass of cpp-wrapper codegen, we should call
            # the fast kernel directly
            picked_kernel = MultiKernelCall.lookup_choice(kernel_name)
            kernel_name = self.kernels[picked_kernel].kernel_name
            final_call_args = call_args_list[picked_kernel]
            arg_types = arg_types_list[picked_kernel]
        else:
            final_call_args = all_call_args

        # numels for all subkernels should be the same. Use kernels[0] here
        self.kernels[0].add_numel_to_call_args_and_grid(
            kernel_name, final_call_args, arg_types, grid
        )

        grid = V.graph.wrapper_code.generate_default_grid(kernel_name, grid)
        V.graph.wrapper_code.generate_kernel_call(
            kernel_name,
            final_call_args,
            grid,
            arg_types=arg_types,
        )

    def codegen_nan_check(self):
        wrapper = V.graph.wrapper_code
        seen = set()
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

    def __init__(self, multi_kernel_name, kernels, src_code):
        assert len(kernels) >= 2
        self._kernels = kernels
        self.multi_kernel_name = multi_kernel_name

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

        self._recorded = False

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
            do_bench_gpu(lambda: kernel_call(True), rep=40, fast_flush=True)
            for kernel_call in kernel_calls
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
    def record_choice(multi_kernel_name, choice):
        """
        Record the multi-kernel choice for cpp-wrapper first pass codegen
        for the second pass.

        We should do nothing if this function is not called during codegen.
        """
        from torch._inductor.graph import GraphLowering

        if not isinstance(V.graph, GraphLowering):
            return

        if not V.graph.record_multi_kernel_choice:
            return

        V.graph.multi_kernel_to_choice[multi_kernel_name] = choice

    @staticmethod
    def lookup_choice(multi_kernel_name):
        # this should always been done during cpp-wrapper codegen
        assert V.graph.record_multi_kernel_choice
        # there should be no miss
        return V.graph.multi_kernel_to_choice[multi_kernel_name]

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

        if not self._recorded:
            self._recorded = True
            self.record_choice(self.multi_kernel_name, self.picked_kernel)
        kernel_calls[self.picked_kernel]()
