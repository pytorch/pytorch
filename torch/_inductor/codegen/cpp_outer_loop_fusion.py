# mypy: allow-untyped-defs

from .cpp import CppKernel, LoopNest, ParallelDepth


class OuterLoopFusedKernel(CppKernel):
    def __init__(self, kernel_group):
        super().__init__(kernel_group.args, kernel_group.ws.num_threads)
        self.inner: list[LoopNest] = []

    def decide_parallel_depth(self, max_parallel_depth, threads):
        kernels_parallel_depth = []
        nested_kernels: list[CppKernel] = [
            loop_nest.get_kernel() for loop_nest in self.inner
        ]
        # TODO(leslie-fang-intel): only enable parallel within all outer loop levels.
        for kernel in nested_kernels:
            # For any ScalarKernel, VecKernel, or Tile2DKernel,
            # they should all have the same call_ranges
            call_ranges = kernel.call_ranges
            assert call_ranges is not None
            kernels_parallel_depth.append(
                kernel.decide_parallel_depth(
                    ParallelDepth(
                        parallel_depth=(
                            len(call_ranges) - max_parallel_depth.start_depth
                        ),
                        start_depth=max_parallel_depth.start_depth,
                    ),
                    threads,
                ).parallel_depth
            )
        return ParallelDepth(
            parallel_depth=min(
                max_parallel_depth.parallel_depth, max(kernels_parallel_depth)
            ),
            start_depth=max_parallel_depth.start_depth,
        )
