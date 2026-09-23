"""Python wrapper codegen for output that is meant to be read and hand-edited.

Inductor's normal python wrapper is written to be loaded by inductor. This variant is
written to be opened by a person (or an agent) who wants to retune the generated kernel
in place: it emits Triton kernels as ordinary module-level code rather than as source
strings handed to ``AsyncCompile``, and it emits only the preamble lines (including the
``AsyncCompile`` lifecycle) that the finished module uses. See
``torch.compiler.export_python``, which is the consumer.

A kernel never autotunes at runtime, because the config a kernel launches with decides
its numerics (a reduction's block sizes set its summation order) and so has to be fixed
by the artifact. Triton kernels therefore require ``triton.autotune_at_compile_time``,
which ``compile_fx_inner`` turns on when it is unset: each hoisted kernel is pinned, through ``triton_heuristics.fixed_config``, to the config
that tuning chose, and those configs are listed in ``KERNEL_CONFIGS`` at the top of the
module, where they can be edited. For the same reason ``triton.multi_kernel`` and
user-defined kernels autotuned over several configs are refused.

The tradeoff that makes this opt-in: a kernel defined at module level compiles serially,
in process, on its first launch, instead of fanning out to the compile worker pool.

Only Triton kernels inductor generates are hoisted. A user-defined ``@triton.jit`` kernel
is still emitted as a source string passed to ``async_compile.triton``, and so are the
kernels of other backends (C++, MPS, Halide, Pallas).
"""

import ast
import re
from typing import Any
from typing_extensions import override

import torch._inductor.config as config
from torch.utils._indented_buffer import DeferredLineBase, IndentedBuffer
from torch.utils._ordered_set import OrderedSet

from .. import ir
from ..runtime.triton_heuristics import config_to_dict
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


class _KernelConfigs(DeferredLineBase):
    """``KERNEL_CONFIGS``: the config compile-time autotuning chose for each hoisted kernel.

    The kernels below read it from their decorators, so it is written ahead of them, but
    its contents are only known once the compile-time autotune block has run, after every
    kernel (subgraphs' included) has been emitted.
    """

    def __init__(self, wrapper: "ReadablePythonWrapperCodegen") -> None:
        super().__init__("")
        self.wrapper = wrapper

    def __call__(self) -> str | None:
        configs = self.wrapper.kernel_configs
        if self.wrapper.scanning_for_uses or not configs:
            return None
        rows = "".join(f"    {name!r}: {cfg!r},\n" for name, cfg in configs.items())
        return (
            "# The launch config of each kernel below, chosen by autotuning at compile time.\n"
            "# Kernels launch with exactly these and never retune at runtime.\n"
            f"KERNEL_CONFIGS = {{\n{rows}}}"
        )

    def _new_line(self, line: str) -> "_KernelConfigs":
        return _KernelConfigs(self.wrapper)


def _pin_to_tuned_config(src_code: str, kernel_name: str) -> str | None:
    """Replace the kernel's heuristics decorator with a fixed_config reading KERNEL_CONFIGS.

    Returns None for a template kernel, which is built with its one config and never
    tuned again, so it has nothing to pin.

    A FIXED autotuner has one config and is exempt from coordinate descent and from
    dynamic RBLOCK scaling, so its first launch compiles that config and runs it.
    """
    defs = [
        node
        for node in ast.parse(src_code).body
        if isinstance(node, ast.FunctionDef) and node.name == kernel_name
    ]
    if len(defs) != 1:
        raise AssertionError(f"expected one def of {kernel_name}, found {len(defs)}")
    (fn,) = defs
    decorator = fn.decorator_list[0]
    if not (
        isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Attribute)
        and decorator.func.attr != "fixed_config"
    ):
        raise AssertionError(f"unexpected decorator on {kernel_name}")
    if decorator.func.attr == "template":
        return None
    kwargs = {kw.arg: kw.value for kw in decorator.keywords}
    inductor_meta = ast.literal_eval(kwargs["inductor_meta"])
    # Sequential combo-kernel tuning would retune the pinned config on first launch,
    # coordinate_descent_tuning would have cached_autotune consult the autotune cache,
    # whose best config for this file would replace the pinned one, and
    # incremental_autotune installs a tuning plugin (get_caching_autotuner_plugins).
    for key in (
        "combo_tuning_groups",
        "coordinate_descent_tuning",
        "incremental_autotune",
    ):
        inductor_meta.pop(key, None)
    triton_meta = ast.get_source_segment(src_code, kwargs["triton_meta"])
    lines = src_code.splitlines()
    lines[decorator.lineno - 1 : decorator.end_lineno] = [
        "@triton_heuristics.fixed_config(",
        f"    config=KERNEL_CONFIGS[{kernel_name!r}],",
        "    filename=__file__,",
        f"    triton_meta={triton_meta},",
        f"    inductor_meta={inductor_meta!r},",
        ")",
    ]
    return "\n".join(lines)


class ReadablePythonWrapperCodegen(PythonWrapperCodegen):
    """Emit kernels as code rather than as strings passed to AsyncCompile."""

    async_compiles_triton_kernels = False

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.scanning_for_uses = False
        self._used_names: OrderedSet[str] | None = None
        self._kernel_texts: list[str] = []
        # Filled in by generate_and_run_autotune_block; None until then.
        self.kernel_configs: dict[str, dict[str, Any] | None] = {}
        if not isinstance(self, SubgraphPythonWrapperCodegen):
            self.header.writeline(_KernelConfigs(self))

    @override
    def _define_kernel_helper(
        self,
        kernel_name: str,
        kernel_body: str,
        metadata: str | None = None,
        gpu: bool = True,
        cpp_definition: str | None = None,
        standalone: bool = False,
        autotune_body: str | None = None,
    ) -> None:
        header_lines = self.header.get_lines_ref()
        start = len(header_lines)
        super()._define_kernel_helper(
            kernel_name,
            kernel_body,
            metadata,
            gpu,
            cpp_definition,
            standalone,
            autotune_body,
        )
        if not standalone or len(header_lines) == start:
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
                # generate() also writes a few lines straight into the assembled result
                # (generate_after_suffix and friends); none of them may name a preamble
                # binding, or the emitted module raises NameError.
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
                # `aten` as used by every graph, so whole-line comments are skipped.
                # Trailing comments and string literals still count, which can only
                # keep a line, never drop a needed one.
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
    def generate_and_run_autotune_block(self) -> dict[str, Any] | None:
        scope = super().generate_and_run_autotune_block()
        if scope is None:
            # A subgraph: its kernels are tuned, and recorded, by the root.
            return None
        for name in self.kernel_configs:
            autotuner = scope.get(name)
            if autotuner is None or len(autotuner.launchers) != 1:
                raise RuntimeError(
                    f"torch._inductor.config.readable_wrapper: kernel {name} was not "
                    "autotuned down to one config at compile time, so the emitted "
                    "module would have to tune it at runtime."
                )
            self.kernel_configs[name] = config_to_dict(autotuner.launchers[0].config)
        return scope

    @override
    def define_user_defined_triton_kernel(  # type: ignore[override]
        self, kernel: Any, configs: list[Any], *args: Any, **kwargs: Any
    ) -> Any:
        if len(configs) > 1:
            # The kernel is still a source string for AsyncCompile, whose autotuner
            # would benchmark these configs on first launch.
            raise RuntimeError(
                "torch._inductor.config.readable_wrapper does not tune at runtime, but "
                f"user-defined Triton kernel {kernel.__name__} is autotuned over "
                f"{len(configs)} configs; give it a single config."
            )
        return super().define_user_defined_triton_kernel(
            kernel, configs, *args, **kwargs
        )

    @override
    def emit_triton_kernel_definition(
        self,
        kernel_name: str,
        subs_name: str,
        src_code: str,
        device_type: str,
        metadata: str | None = None,
    ) -> None:
        if not config.triton.autotune_at_compile_time:
            raise RuntimeError(
                "torch._inductor.config.readable_wrapper pins each Triton kernel to the "
                "config chosen by autotuning at compile time and requires "
                "triton.autotune_at_compile_time; enable it."
            )
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
        # Kernels define module-level @triton.jit helpers under names that are only
        # unique per kernel (scan combine_fns, flex attention's forward_inner, ...), so
        # two kernels can define the same name with different bodies; in one shared
        # namespace the later def would win for both. Make them kernel-unique.
        helpers = re.findall(r"^def (\w+)\(", src_code, re.MULTILINE)
        for helper in OrderedSet(helpers) - OrderedSet([kernel_name, subs_name]):
            src_code = re.sub(rf"\b{helper}\b", f"{helper}_{kernel_name}", src_code)
        pinned = _pin_to_tuned_config(src_code, subs_name)
        if pinned is not None:
            self.kernel_configs[subs_name] = None
        self.define_kernel(
            kernel_name,
            src_code if pinned is None else pinned,
            metadata,
            standalone=True,
            # The compile-time autotune block execs its kernels instead of emitting
            # them, and a module-level kernel there has no __file__ to name itself by,
            # so that block keeps the AsyncCompile form with the original heuristics.
            # It runs at compile time only and is not carried in the emitted module.
            autotune_body=self.async_compile_triton_body(
                subs_name, src_code, device_type
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
    root) still apply: the readable class overrides none of those. Both drop the
    benchmark harness.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        root = self.get_root_graph()
        if not isinstance(root, ReadablePythonWrapperCodegen):
            raise AssertionError(f"expected a readable root wrapper, got {type(root)}")
        # This subgraph's text is spliced into the root module, so the root's usage
        # scan has to leave out this subgraph's kernels too.
        self._kernel_texts = root._kernel_texts
        self.kernel_configs = root.kernel_configs


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
    if config.triton.multi_kernel:
        # MultiKernelCall benchmarks its candidate kernels on first launch and keeps the
        # fastest, which is choosing a kernel at runtime.
        raise RuntimeError(
            "torch._inductor.config.readable_wrapper does not choose kernels at runtime "
            "and is incompatible with triton.multi_kernel, which benchmarks its "
            "candidates on first launch."
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


def select_wrapper_codegen(
    device: str,
    registered: type[PythonWrapperCodegen],
    cpp_wrapper: bool,
    fx_wrapper: bool,
) -> type[PythonWrapperCodegen]:
    """The wrapper class this compile uses: the device's registered one, or the readable one.

    readable_wrapper is a per-compile config, so it is resolved here, at the compile's
    own call site, and never by get_wrapper_codegen_for_device, which also serves as the
    accessor for what a device has registered.
    """
    if config.readable_wrapper and (cpp_wrapper or fx_wrapper):
        raise RuntimeError(
            "torch._inductor.config.readable_wrapper emits a python wrapper and is "
            "incompatible with cpp_wrapper and fx_wrapper; disable one of them."
        )
    if not readable_wrapper_requested():
        return registered
    if registered is not PythonWrapperCodegen:
        # An out-of-tree python wrapper (or PythonWrapperMtia) cannot be replaced by one
        # that knows nothing about that backend, and keeping it would silently ignore
        # the flag.
        raise RuntimeError(
            "torch._inductor.config.readable_wrapper replaces the stock python wrapper, "
            f"but {device} registers {registered.__name__}; disable readable_wrapper "
            "for this device."
        )
    return ReadablePythonWrapperCodegen
