import logging
import os

from .. import config
from ..codecache import PyCodeCache, TritonFuture
from ..utils import do_bench
from ..virtualized import V

log = logging.getLogger(__name__)


def get_kernel_argdefs(kernel):
    arg_defs, _, _ = kernel.args.python_argdefs()
    return arg_defs


def get_all_kernel_argdefs(kernels):
    argdefs_list = [get_kernel_argdefs(kernel) for kernel in kernels]
    all_argdefs = {}  # use a dict rather than set to maintain insertion order
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

        # this is mainly for sanity check
        self.used_names = set()

    def define_kernel(self, kernels):
        kernel_names = tuple(k.kernel_name for k in kernels)
        if kernel_names in self.subkernel_to_kernel_name:
            return self.subkernel_to_kernel_name[kernel_names]

        # name the multi kernel based on the first kernel
        multi_kernel_name = f"multi_kernel_{kernel_names[0]}"
        assert multi_kernel_name not in self.used_names
        self.used_names.add(multi_kernel_name)
        self.subkernel_to_kernel_name[kernel_names] = multi_kernel_name

        wrapper = V.graph.wrapper_code

        kernel_call_def_code = "\n".join(
            [
                f"""
    def call{idx}(need_clone_args=False):
        args = [{', '.join(get_kernel_argdefs(kernels[idx]))}]
        if need_clone_args:
            args = multi_kernel_call.kernels[{idx}].clone_args(*args)
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
    """

    def __init__(self, kernels):
        assert len(kernels) >= 2

        self.kernels = kernels
        self.kernel_name = V.graph.wrapper_code.multi_kernel_state.define_kernel(
            kernels
        )

    def call_kernel(self):
        """
        Collect the union of arguments from all subkernels as the arguments
        for the multi-kernel.
        """
        call_args_list = [kernel.get_call_args() for kernel in self.kernels]
        all_call_args = {}  # use a dict rather than set to maintain insertion order
        for call_args in call_args_list:
            all_call_args.update({arg: None for arg in call_args})

        all_call_args = list(all_call_args.keys())
        grid = []

        # numels for all subkernels should be the same. Use kernels[0] here
        self.kernels[0].add_numel_to_call_args_and_grid(all_call_args, grid)

        V.graph.wrapper_code.generate_kernel_call(
            self.kernel_name,
            all_call_args,
            grid,
            V.graph.scheduler.current_device.index,
        )


class MultiKernelCall:
    """
    This class is called at run time to actually run the kernel
    """

    def __init__(self, kernels, src_code):
        assert len(kernels) >= 2
        self._kernels = kernels

        self.picked_kernel = None
        if config.triton.multi_kernel > 1:
            # manually force a subkernel to ease perf testing
            self.picked_by_config = config.triton.multi_kernel - 2
            assert self.picked_by_config < len(self._kernels)
            self.picked_kernel = self.picked_by_config

        self._run = PyCodeCache.load(src_code).run
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

    def run_with_argless_kernels(self, kernel_calls):
        if self.picked_kernel is None:
            timings = [
                do_bench(lambda: kernel_call(True), rep=40, fast_flush=True)
                for kernel_call in kernel_calls
            ]
            self.picked_kernel = timings.index(min(timings))
            log.debug(
                "pick %dth sub-kernel in %s. Timings %s",
                self.picked_kernel,
                [k.fn.__name__ for k in self.kernels],
                timings,
            )
            self.store_cache()
        kernel_calls[self.picked_kernel]()
