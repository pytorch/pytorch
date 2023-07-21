import builtins

import sympy

from ..codecache import PyCodeCache, TritonFuture
from ..utils import do_bench
from ..virtualized import V


# TODO rename
def get_kernel_call_args(kernel):
    return kernel.args.python_argdefs()[0]  # TODO
    _, call_args, _ = kernel.args.python_argdefs()
    # dynamo wraps unspec variable as 0d CPU tensor, need convert to scalar
    for i in range(len(call_args)):
        if V.graph.is_unspec_arg(call_args[i]):
            call_args[i] = call_args[i] + ".item()"
    return call_args


def get_all_kernel_call_args(kernels):
    call_args_list = [get_kernel_call_args(kernel) for kernel in kernels]
    all_call_args = {}  # use a dict rather than set to maintain insertion order
    for call_args in call_args_list:
        all_call_args.update({arg: None for arg in call_args})

    return list(all_call_args.keys())


def get_extra_arg_names(kernel):
    extra_arg_names = []
    for tree in kernel.range_trees:
        if tree.prefix != "r" or kernel.inside_reduction:
            extra_arg_names.append(f"{tree.prefix}numel")

    # extra_arg_names.extend(["grid", "stream"])
    return extra_arg_names


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
        all_call_args = get_all_kernel_call_args(kernels)
        src_code = f"""
def run(multi_kernel, {', '.join(get_all_kernel_call_args(kernels))}, {', '.join(get_extra_arg_names(kernels[0]))}, grid, stream):
    def call0():
        # TODO: clone the args if doing the benchmarking
        multi_kernel.kernels[0].run({', '.join(get_kernel_call_args(kernels[0]))}, {', '.join(get_extra_arg_names(kernels[0]))}, grid=grid, stream=stream)
    def call1():
        multi_kernel.kernels[1].run({', '.join(get_kernel_call_args(kernels[1]))}, {', '.join(get_extra_arg_names(kernels[1]))}, grid=grid, stream=stream)
    multi_kernel.run_with_argless_kernels([call0, call1])
        """
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
        # TODO: should use arg name rather than call arg?
        call_args_list = [self.get_call_args(kernel) for kernel in self.kernels]
        all_call_args = {}  # use a dict rather than set to maintain insertion order
        for call_args in call_args_list:
            all_call_args.update({arg: None for arg in call_args})

        all_call_args = list(all_call_args.keys())

        # TODO dedup the code with TritonKernel class
        grid = []
        for tree in self.kernels[0].range_trees:
            if isinstance(tree.numel, (sympy.Integer, sympy.Symbol)):
                expr = tree.numel
            else:
                expr = V.graph.wrapper_code.generate_numel_expr(name, tree)

            if tree.prefix != "r" or self.kernels[0].inside_reduction:
                all_call_args.append(expr)
            if tree.prefix != "r":
                grid.append(expr)

        V.graph.wrapper_code.generate_kernel_call(
            self.kernel_name,
            all_call_args,
            grid,
            V.graph.scheduler.current_device.index,
        )


class MultiKernelCall:
    """
    This class is called at run time to actually run the kernel

    TODO: we could add cache for the choices piced by the MultiKernelCall
    """

    def __init__(self, kernels, src_code):
        assert len(kernels) >= 2
        self._kernels = kernels

        self.picked_kernel = None

        self._run = PyCodeCache.load(src_code).run

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
        self._run(self, *args, **kwargs)

    def run_with_argless_kernels(self, kernel_calls):
        if self.picked_kernel is None:
            timings = [
                do_bench(kernel_call, rep=40, fast_flush=True) for kernel_call in kernel_calls 
            ]
            self.picked_kernel = timings.index(min(timings))
        kernel_calls[self.picked_kernel]()

    def old_run(self, *args, **kwargs):
        if self.picked_kernel is None:
            timings = {
                kernel: self.bench(kernel, *args, **kwargs) for kernel in self.kernels
            }
            self.picked_kernel = builtins.min(timings, key=timings.get)
        self.picked_kernel.run(*args, **kwargs)
