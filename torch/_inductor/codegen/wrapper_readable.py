"""Python wrapper codegen for output that is meant to be read and hand-edited.

Inductor's normal python wrapper is written to be loaded by inductor. This variant is
written to be opened by a person (or an agent) who wants to retune the generated kernel
in place: it emits Triton kernels as ordinary module-level code rather than as source
strings handed to ``AsyncCompile``, and it emits only the preamble lines (including the
``AsyncCompile`` lifecycle) that the finished module uses. See
``torch.compiler.export_python``, which is the consumer.

The tradeoffs are deliberate and are the reason this is opt-in: a kernel defined at
module level compiles serially, in process, on its first launch, instead of fanning out
to the compile worker pool. And every hoisted kernel names itself by the wrapper's
``__file__``, so they all share one autotune-cache key; that cache is effectively off in
this mode (its configs_hash check keeps a wrong config from being applied).
"""

import re
from typing_extensions import override

import torch._inductor.config as config
from torch.utils._indented_buffer import DeferredLineBase, IndentedBuffer
from torch.utils._ordered_set import OrderedSet

from .. import ir
from .wrapper import PythonWrapperCodegen, SubgraphPythonWrapperCodegen


# Emitted whatever the analysis says. `torch` is used by essentially every graph and is
# what the rest of the preamble is written in terms of, so dropping it could only ever
# be wrong.
_ALWAYS_EMIT = ("torch",)


class _LineIfNamesUsed(DeferredLineBase):
    """A preamble line that survives assembly only if the module uses what it is for.

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
        self._used_names: OrderedSet[str] | None = None
        self._kernel_texts: list[str] = []

    @override
    def _define_kernel_helper(self, *args, standalone: bool = False, **kwargs) -> None:  # type: ignore[no-untyped-def]
        header_lines = self.header.get_lines_ref()
        start = len(header_lines)
        super()._define_kernel_helper(*args, standalone=standalone, **kwargs)
        if not standalone:
            # `X = async_compile.cpp_pybinding(...)` and friends are wrapper code: they
            # are what keep async_compile alive.
            return
        # A hoisted kernel is a self-contained module -- that is why the default path
        # can exec each one in its own namespace -- so it never reads a wrapper binding.
        # Record it so the usage scan can leave it out: its own `math as tl_math` import
        # and its `'device': 0` metadata are not uses of the wrapper's `math` or
        # `device`. Recorded as written into header, i.e. after splice has reformatted
        # it, so that it matches the assembled text.
        kernel = IndentedBuffer()
        kernel.writelines(header_lines[start:])
        self._kernel_texts.append(kernel.getrawvalue())

    @override
    def write_preamble_line(
        self, buf: IndentedBuffer, names: tuple[str, ...], line: str
    ) -> None:
        if any(name in _ALWAYS_EMIT for name in names):
            buf.writeline(line)
            return
        buf.writeline(_LineIfNamesUsed(line, names, self))

    def preamble_names_used(self, names: tuple[str, ...]) -> bool:
        if self._used_names is None:
            self.scanning_for_uses = True
            try:
                # Every buffer that can hold a use, including header, which by now also
                # holds the kernel definitions replayed into it, and
                # subgraph_definitions, which holds each subgraph's finished text.
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
                self._used_names = OrderedSet(
                    word
                    for line in text.splitlines()
                    if not line.lstrip().startswith("#")
                    for word in re.findall(r"\w+", line)
                )
            finally:
                self.scanning_for_uses = False
        return not self._used_names.isdisjoint(names)

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
    subgraph overrides (no header, triton imports and the AsyncCompile wait left to the
    root, no benchmark harness) still apply: the readable class overrides none of those.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        root = self.get_root_graph()
        if not isinstance(root, ReadablePythonWrapperCodegen):
            raise AssertionError(f"expected a readable root wrapper, got {type(root)}")
        # This subgraph's text is spliced into the root module, so the root's usage
        # scan has to leave out this subgraph's kernels too.
        self._kernel_texts = root._kernel_texts


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
    if config.profile_bandwidth_output:
        # profile_bandwidth_output runs the module's benchmark harness, which this mode
        # does not emit.
        raise RuntimeError(
            "torch._inductor.config.readable_wrapper is incompatible with "
            "profile_bandwidth_output, which runs the benchmark harness that "
            "readable_wrapper leaves out of the module."
        )
    return True
