"""Python wrapper codegen for output that is meant to be read and hand-edited.

Inductor's normal python wrapper is written to be loaded by inductor. This variant is
written to be opened by a person (or an agent) who wants to retune the generated kernel
in place: it emits Triton kernels as ordinary module-level code rather than as source
strings handed to ``AsyncCompile``. See ``torch.compiler.export_python``, which is the
consumer.

The tradeoffs are deliberate and are the reason this is opt-in: a kernel defined at
module level compiles serially, in process, on its first launch, instead of fanning out
to the compile worker pool. And every hoisted kernel names itself by the wrapper's
``__file__``, so they all share one autotune-cache key; that cache is effectively off in
this mode (its configs_hash check keeps a wrong config from being applied).
"""

import re
from typing_extensions import override

import torch._inductor.config as config
from torch.utils._ordered_set import OrderedSet

from .. import ir
from .wrapper import PythonWrapperCodegen, SubgraphPythonWrapperCodegen


class ReadablePythonWrapperCodegen(PythonWrapperCodegen):
    """Emit kernels as code rather than as strings passed to AsyncCompile."""

    async_compiles_triton_kernels = False

    @override
    def emit_triton_kernel_definition(
        self,
        kernel_name: str,
        subs_name: str,
        src_code: str,
        device_type: str,
        metadata: str | None = None,
    ) -> None:
        # src_code is already a complete module: the triton imports, the
        # @triton_heuristics.* decorator that builds the CachingAutotuner, and the
        # @triton.jit def. Spliced at module level it binds kernel_name to the same
        # object async_compile.triton would have returned, so the launch site
        # (KERNEL.run(...)) is unchanged.
        # The provenance comment leads with "# kernel path: /tmp/torchinductor_.../x.py",
        # which is where the kernel WOULD have been compiled from. It is defined right
        # here instead, and pointing a reader at a cache file is the exact confusion this
        # mode exists to remove. The source-op mapping below it is worth keeping.
        if metadata:
            metadata = "\n".join(
                line
                for line in metadata.splitlines()
                if not line.startswith("# kernel path:")
            )
        # @triton.jit helpers are numbered per kernel and named by op sequence, so two
        # kernels can define the same helper name with different bodies; in one shared
        # namespace the later def would win for both. Make them kernel-unique.
        helpers = re.findall(r"^def (_triton_helper_fn\w*)\(", src_code, re.MULTILINE)
        for helper in OrderedSet(helpers):
            src_code = re.sub(rf"\b{helper}\b", f"{helper}_{kernel_name}", src_code)
        self.define_kernel(
            kernel_name,
            src_code,
            metadata,
            standalone=True,
            # The compile-time autotune block execs its kernels instead of emitting
            # them, and a module-level kernel there has no __file__ to name itself by,
            # so that block keeps the AsyncCompile form.
            autotune_body=(
                self.async_compile_triton_body(subs_name, src_code, device_type)
                if config.triton.autotune_at_compile_time
                else None
            ),
        )

    @override
    @staticmethod
    def create(
        is_subgraph: bool,
        subgraph_name: str | None,
        parent_wrapper: PythonWrapperCodegen | None,
        partition_signatures: ir.GraphPartitionSignature | None = None,
    ) -> PythonWrapperCodegen:
        if is_subgraph:
            if subgraph_name is None:
                raise AssertionError("expected subgraph_name to be set")
            if parent_wrapper is None:
                raise AssertionError("expected parent_wrapper to be set")
            # graph_partition is on by default in OSS, so partition subgraphs are the
            # common case rather than an exotic one -- delegating them to the stock
            # subgraph wrapper would leave their kernels stringified. Compose instead,
            # with this class first so its kernel emission wins.
            return _ReadableSubgraphPythonWrapperCodegen(
                subgraph_name, parent_wrapper, partition_signatures
            )
        return ReadablePythonWrapperCodegen()


class _ReadableSubgraphPythonWrapperCodegen(
    ReadablePythonWrapperCodegen, SubgraphPythonWrapperCodegen
):
    """Subgraph wrapper that keeps readable kernel emission.

    MRO puts ReadablePythonWrapperCodegen first, so kernels stay unstringified while the
    subgraph overrides (no duplicate header, no benchmark harness) still apply.
    """


def readable_wrapper_requested() -> bool:
    """Whether this compile asked for readable output, and can actually have it."""
    if not config.readable_wrapper:
        return False
    if not config.triton.unique_kernel_names:
        # Every kernel would be named `triton_`, so hoisting them to module level makes
        # all but the last unreachable -- an artifact that runs and is wrong.
        raise RuntimeError(
            "torch._inductor.config.readable_wrapper defines kernels at module level, "
            "which requires triton.unique_kernel_names so they do not shadow each "
            "other. Enable unique_kernel_names or disable readable_wrapper."
        )
    for flag in ("benchmark_kernel", "benchmark_combo_kernel"):
        if getattr(config, flag):
            # These append get_args()/call()/__main__ to each kernel's source; at module
            # level those collide with each other and with the wrapper's own call.
            raise RuntimeError(
                f"torch._inductor.config.readable_wrapper is incompatible with {flag}, "
                "which appends a get_args()/call()/__main__ harness to every kernel; "
                "defined at module level those collide."
            )
    return True
