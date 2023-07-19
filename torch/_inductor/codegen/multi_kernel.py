import builtins

import sympy

from ..codecache import TritonFuture
from ..utils import do_bench
from ..virtualized import V


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
        kernels = tuple(k.kernel_name for k in kernels)
        if kernels in self.subkernel_to_kernel_name:
            return self.subkernel_to_kernel_name[kernels]

        # name the multi kernel based on the first kernel
        multi_kernel_name = f"multi_kernel_{kernels[0]}"
        assert multi_kernel_name not in self.used_names
        self.used_names.add(multi_kernel_name)
        self.subkernel_to_kernel_name[kernels] = multi_kernel_name

        wrapper = V.graph.wrapper_code
        wrapper.header.splice(
            f"""
        {multi_kernel_name} = MultiKernelCall([
            {", ".join(kernels)},
        ])
        """
        )

        return multi_kernel_name


class MultiKernel:
    """
    This class maintains the compile time state for multi kernels.

    Assume we do codegen for a MultiKernel encapsulating kernel1 and kernel2.
    The generated definition for the multi-kernel will looks like:
    ```
    multi_kernel_kernel1 = MultiKernelCall([kernel1, kernel2])
    ```
    """

    def __init__(self, kernels):
        assert len(kernels) >= 2

        self.kernels = kernels
        self.kernel_name = V.graph.wrapper_code.multi_kernel_state.define_kernel(
            kernels
        )

    def get_call_args(self, kernel):
        _, call_args, _ = kernel.args.python_argdefs()
        # dynamo wraps unspec variable as 0d CPU tensor, need convert to scalar
        for i in range(len(call_args)):
            if V.graph.is_unspec_arg(call_args[i]):
                call_args[i] = call_args[i] + ".item()"
        return call_args

    def call_kernel(self):
        """
        Collect the union of arguments from all subkernels as the arguments
        for the multi-kernel.
        """
        call_args_list = [self.get_call_args(kernel) for kernel in self.kernels]
        call_args = call_args_list[0]
        for other_call_args in call_args_list[1:]:
            assert set(call_args) == set(
                other_call_args
            ), f"call_args: {call_args}, other call args: {other_call_args}"

        # TODO dedup the code with TritonKernel class
        grid = []
        for tree in self.kernels[0].range_trees:
            if isinstance(tree.numel, (sympy.Integer, sympy.Symbol)):
                expr = tree.numel
            else:
                expr = V.graph.wrapper_code.generate_numel_expr(name, tree)

            if tree.prefix != "r" or self.kernels[0].inside_reduction:
                call_args.append(expr)
            if tree.prefix != "r":
                grid.append(expr)

        V.graph.wrapper_code.generate_kernel_call(
            self.kernel_name,
            call_args,
            grid,
            V.graph.scheduler.current_device.index,
        )


class MultiKernelCall:
    """
    This class is called at run time to actually run the kernel

    TODO: we could add cache for the choices piced by the MultiKernelCall
    """

    def __init__(self, kernels):
        assert len(kernels) >= 2
        self._kernels = kernels

        self.picked_kernel = None

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

    def bench(self, kernel, *args, **kwargs):
        def kernel_call():
            cloned_args = kernel.clone_args(*args)
            kernel.run(*cloned_args, **kwargs)

        return do_bench(kernel_call, rep=40, fast_flush=True)

    def run(self, *args, **kwargs):
        if self.picked_kernel is None:
            timings = {
                kernel: self.bench(kernel, *args, **kwargs) for kernel in self.kernels
            }
            self.picked_kernel = builtins.min(timings, key=timings.get)
        self.picked_kernel.run(*args, **kwargs)
