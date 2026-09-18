"""Python wrapper codegen for output that is meant to be read and hand-edited.

Inductor's normal python wrapper is written to be loaded by inductor. This variant is
written to be opened by a person (or an agent) who wants to retune the generated kernel
in place: it emits Triton kernels as ordinary module-level code rather than as source
strings handed to ``AsyncCompile``, and it drops the ``AsyncCompile`` lifecycle when no
backend still needs it. See ``torch.compiler.export_python``, which is the consumer.

The tradeoff is deliberate and is the reason this is opt-in: a kernel defined at module
level compiles serially, in process, on its first launch, instead of fanning out to the
compile worker pool.
"""

import re
from typing_extensions import override

import torch._inductor.config as config
from torch.utils._indented_buffer import DeferredLineBase, IndentedBuffer

from .. import ir
from ..runtime import triton_heuristics
from ..utils import cache_on_self
from ..virtualized import V
from .wrapper import PythonWrapperCodegen, SubgraphPythonWrapperCodegen


# Emitted whatever the analysis says. `torch` is used by essentially every graph and is
# what the rest of the preamble is written in terms of, so dropping it could only ever
# be wrong.
_ALWAYS_EMIT = frozenset({"torch"})


class _LineIfAsyncCompileUsed(DeferredLineBase):
    """A line that survives assembly only if some kernel actually bound via AsyncCompile.

    Whether the emitted module needs an ``AsyncCompile`` is not known when the preamble
    is written -- ``write_header`` runs from ``__init__``, before a single kernel has
    been defined -- so the decision is deferred to ``getvalue()``, which runs after
    every kernel definition has been replayed.
    """

    def __init__(self, line: str, wrapper: PythonWrapperCodegen) -> None:
        super().__init__(line)
        self.wrapper = wrapper

    def __call__(self) -> str | None:
        return self.line if self.wrapper.uses_async_compile else None

    def _new_line(self, line: str) -> "_LineIfAsyncCompileUsed":
        return _LineIfAsyncCompileUsed(line, self.wrapper)


class _LineIfNamesUsed(DeferredLineBase):
    """A preamble line that survives assembly only if the module uses what it binds.

    Which bindings a graph needs is not known when the preamble is written, so the
    question is asked at ``getvalue()`` time, against the finished module. Asking then
    rather than at each use site is what makes this total: inductor emits these names
    from many places, several by interpolation (``empty_strided_{device}``) or via
    ``repr()`` (``device(...)``, ``inf``, ``nan``), which no registry of use sites would
    catch.
    """

    def __init__(
        self, line: str, names: tuple[str, ...], wrapper: "ReadablePythonWrapperCodegen"
    ) -> None:
        super().__init__(line)
        self.names = names
        self.wrapper = wrapper

    def __call__(self) -> str | None:
        if self.wrapper.scanning_for_uses:
            # Reached while building the text to scan. Every preamble line reads as
            # absent there, which is what makes `X = mod.X` not count as a use of X.
            return None
        return self.line if self.wrapper.preamble_names_used(self.names) else None

    def _new_line(self, line: str) -> "_LineIfNamesUsed":
        return _LineIfNamesUsed(line, self.names, self.wrapper)


class ReadablePythonWrapperCodegen(PythonWrapperCodegen):
    """Emit kernels as code rather than as strings passed to AsyncCompile."""

    async_compiles_triton_kernels = False

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.scanning_for_uses = False
        self._scan_text: str | None = None
        self._kernel_texts: list[str] = []

    @override
    def _define_kernel_helper(self, kernel_name, kernel_body, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        # A kernel is a self-contained module -- that is why the default path can exec
        # each one in its own namespace -- so it never reads a wrapper binding. Record
        # it so the usage scan can leave it out: its own `math as tl_math` import and
        # its `'device': 0` metadata are not uses of the wrapper's `math` or `device`.
        self._kernel_texts.append(kernel_body)
        super()._define_kernel_helper(kernel_name, kernel_body, *args, **kwargs)

    @override
    def write_preamble_line(
        self, buf: IndentedBuffer, names: tuple[str, ...], line: str
    ) -> None:
        if any(name in _ALWAYS_EMIT for name in names):
            buf.writeline(line)
            return
        buf.writeline(_LineIfNamesUsed(line, names, self))

    def preamble_names_used(self, names: tuple[str, ...]) -> bool:
        if self._scan_text is None:
            self.scanning_for_uses = True
            try:
                # Every buffer that can hold a use, including header, which by now also
                # holds the kernel definitions replayed into it.
                text = "\n".join(
                    buf.getrawvalue()
                    for buf in (
                        self.imports,
                        self.header,
                        self.subgraph_definitions,
                        self.prefix,
                        self.wrapper_call,
                        self.suffix,
                        self.kernel_declarations,
                    )
                )
                for kernel_text in self._kernel_texts:
                    text = text.replace(kernel_text, "")
                # Inductor emits a provenance comment per kernel naming every source op
                # ("Original ATen: [aten.mul, ...]"), so a comment-blind scan reads
                # `aten` as used by every graph. A name mentioned in a comment is not a
                # use.
                self._scan_text = "\n".join(
                    line
                    for line in text.splitlines()
                    if not line.lstrip().startswith("#")
                )
            finally:
                self.scanning_for_uses = False
        return any(
            re.search(rf"\b{re.escape(name)}\b", self._scan_text) for name in names
        )

    @override
    @cache_on_self
    def write_triton_header_once(self) -> None:
        if config.triton.autotune_at_compile_time or V.graph.cpp_wrapper:
            # Those paths splice into buffers this mode does not prune; leave them be.
            super().write_triton_header_once()
            return
        # `import triton` / `import triton.language as tl` are duplicated by every
        # hoisted kernel's own import block, and `start_graph`/`end_graph` are only used
        # under profile_bandwidth. Emit each only if the wrapper's own code wants it.
        for names, line in (
            (("triton",), "import triton"),
            (("tl",), "import triton.language as tl"),
            (
                ("start_graph", "end_graph"),
                f"from {triton_heuristics.__name__} import start_graph, end_graph",
            ),
        ):
            self.write_preamble_line(self.imports, names, line)
        self.imports.writeline(
            V.graph.device_ops.import_get_raw_stream_as("get_raw_stream")
        )

    @override
    def add_benchmark_harness(self, output: IndentedBuffer) -> None:
        # The harness (get_args/benchmark_compiled_module/__main__) is a runnable
        # benchmark script bolted onto the module. This mode emits a module to be read
        # and edited, and the harness is also written into the output buffer after the
        # preamble has been scanned, so it could reference a binding already dropped.
        return

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
    def write_async_compile_binding(self) -> None:
        self.header.writeline(
            _LineIfAsyncCompileUsed("async_compile = AsyncCompile()", self)
        )

    @override
    def write_async_compile_wait(self) -> None:
        self.prefix.writeline("")
        self.prefix.writeline(
            _LineIfAsyncCompileUsed("async_compile.wait(globals())", self)
        )
        self.prefix.writeline(_LineIfAsyncCompileUsed("del async_compile", self))

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
    if config.benchmark_kernel:
        # benchmark_kernel appends get_args()/call()/__main__ to each kernel's source;
        # at module level those collide with each other and with the wrapper's own call.
        raise RuntimeError(
            "torch._inductor.config.readable_wrapper is incompatible with "
            "benchmark_kernel, which appends a get_args()/call()/__main__ harness to "
            "every kernel; defined at module level those collide."
        )
    return True
