"""Toolchain abstraction for native-AOT kernel export and launcher codegen.

One ``Toolchain`` subclass per DSL/compiler. Everything per-kind lives
here -- how a builder dict is validated and compiled into artifacts, what
those artifacts are (extensions, link inputs), and how the C++ launcher
marshals at::Tensor arguments into the kernel's ABI -- so adding a DSL
means adding one class and registering it, with export.py, gen_aot_lib.py
and the CMake project untouched.

The cross-toolchain contract, in order of flow:

  1. An op's builder module (package-imported by export.py) exposes
     ``build(spec) -> dict``. The dict's ``kind`` selects the toolchain;
     ``prefix`` names the artifacts; ``REQUIRED_BUILD_KEYS`` lists what
     else the toolchain needs (validated up front so a bad builder fails
     at export with a message, not a KeyError mid-compile).
  2. ``export(build_result, out_dir)`` runs the compiler and writes
     ``<prefix>.<artifact_exts>``. It returns the marshalling metadata to
     persist in the ``<prefix>.json`` sidecar -- the ONLY channel from
     export-time knowledge to launcher codegen.
  3. ``gen_launcher(sidecar)`` emits C++ with the toolchain-independent
     signature every declaration's ``cpp_launch()`` programs against:

         void launch_<prefix>(const at::Tensor&..., <scalars>..., c10::Stream)

     Everything above the launcher (guard chain, cond, DispatchStub
     registration) is toolchain-blind. The stream crosses that boundary
     as a device-agnostic c10::Stream, so the shared contract carries no
     CUDA type; each launcher body narrows it to the raw handle its own
     C ABI needs (cute_dsl_*_wrapper, the Triton entry point and
     cuLaunchKernel all take cudaStream_t/CUstream). Declarations pass
     at::cuda::getCurrentCUDAStream(), which converts implicitly.

Properties consumed by the driver scripts:
  * ``artifact_exts``: extensions written next to the sidecar; used to
    spot artifacts left with no sidecar (export._check_no_orphan_artifacts).
    The idempotency skip keys on the sidecar itself, not on these.
  * ``link_exts``: which of ``artifact_exts`` go to the LINKER. The
    generator names exactly these files, for the sidecars that survived
    the arch tie-break, in the CMake it emits -- exact paths rather than a
    glob, so an artifact no launcher references cannot ride along into
    libtorch_cuda. Empty for a kind that embeds its artifact in the
    generated source instead (Triton's cubin bytes); it must be a subset
    of ``artifact_exts``, which _assert_link_exts_are_exportable checks
    at import -- along with the attribute being declared at all, since the
    inherited default cannot be told from a deliberate empty.
  * ``launcher_includes``: per-kind includes for the generated .cpp.
  * ``kernel_includes(sidecar)``: per-kernel includes for that same file,
    for toolchains whose export writes a header (CuTeDSL's ABI struct).
  * ``NARROWS_SHAPES_TO_INT32``: the exported ABI takes i32 extents, so
    gen_aot_lib emits a stub gate declining dims past INT32_MAX.
  * ``ARCH_ENV_VAR``: env var this kind reads when no arch is passed
    explicitly. export REFUSES it rather than resolving it: it answers for
    one kind, so with two kinds registered a tree named for its value held
    sidecars recording the detected arch. Pass --arch instead.

Export runs as stage 2 of the two-stage build (build torch -> build
the AOT lib), so torch is always importable during export.
"""

from __future__ import annotations

import os
import re


class Toolchain:
    kind: str = ""
    artifact_exts: tuple[str, ...] = ()
    # None means NOT DECLARED, which is refused: () is a real value (a kind whose
    # launcher embeds the artifact instead of linking it), so inheriting a default
    # must not be able to pass for that.
    link_exts: tuple[str, ...] | None = None
    launcher_includes: tuple[str, ...] = ()

    # Torch build backends this kind can emit kernels for, as the names
    # torch reports: "cuda" or "rocm". A build whose backend is not listed
    # never asks for this toolchain, so its runtime being absent there is
    # EXPECTED, not a skip worth reporting -- that is what keeps a ROCm
    # build from complaining about missing CuTeDSL/Triton wheels while a
    # CUDA build still fails loudly if a declared kernel cannot be built.
    # Declared per-kind rather than inferred so a future ROCm DSL (FlyDSL)
    # is a new class with BACKENDS = ("rocm",) and no gate to rewrite.
    BACKENDS: tuple[str, ...] = ("cuda",)

    # Importable modules this kind needs to COMPILE a kernel. Absence is
    # FATAL once a declaration targeting this build's backend reaches
    # export: its kernels were asked for, and exporting only some of them
    # ships a wheel that silently underperforms. Build without the DSL
    # wheels via TORCH_NATIVE_AOT=0 instead (see build_stage2.should_run
    # and test_missing_runtime_is_fatal_not_skipped). Nothing at RUNTIME
    # needs these -- the exported artifacts are self-contained.
    REQUIRED_RUNTIMES: tuple[str, ...] = ()

    # Distribution names whose versions define this kind's COMPILER, recorded per
    # sidecar and compared on the next run: the DSL version appears in no file the
    # source closure hashes, so an upgraded wheel otherwise invalidates nothing and
    # the tree mixes compilers. DISTRIBUTION names, not REQUIRED_RUNTIMES' module
    # names, keep the lookup metadata-only -- importing cutlass for __version__
    # would cost the skip path an MLIR import every run.
    RUNTIME_DISTS: tuple[str, ...] = ()

    # True when this kind's exported ABI carries int32_t shape slots, so the
    # generated stub must decline a dim past INT32_MAX (_int32_size_gate). A
    # property of the exported ABI, hence per-kind: Triton takes its scalar widths
    # from the kernel's own signature.
    NARROWS_SHAPES_TO_INT32: bool = False

    # Env var this kind falls back to when no explicit arch is given. The
    # sidecar records the arch it resolves to, so a run that changes only
    # this variable is not mistaken for one that already exported (see
    # export._effective_arch).
    ARCH_ENV_VAR: str | None = None

    REQUIRED_BUILD_KEYS: tuple[str, ...] = ()

    @classmethod
    def missing_runtimes(cls) -> list[str]:
        import importlib.util

        return [m for m in cls.REQUIRED_RUNTIMES if importlib.util.find_spec(m) is None]

    @classmethod
    def serves_backend(cls, backend: str) -> bool:
        return backend in cls.BACKENDS

    def validate_build_result(self, b: dict) -> None:
        missing = [k for k in ("prefix", *self.REQUIRED_BUILD_KEYS) if k not in b]
        if missing:
            raise RuntimeError(
                f"{self.kind} builder result for "
                f"{b.get('prefix', '<unnamed>')} is missing keys: {missing}"
            )

    def export(self, b: dict, out_dir: str, arch: str | None = None) -> dict:
        """Compile one spec point; return sidecar marshalling metadata.

        ``arch`` is an sm string ("sm_90a") or None for detect-from-
        device. With an explicit arch no toolchain touches the CUDA
        driver, so export runs on GPU-less machines, and the arch is
        per-compile state rather than per-process: CuTeDSL passes
        --gpu-arch, Triton kinds install a fixed GPUTarget driver. One
        process may therefore export for several arches."""
        raise NotImplementedError

    def gen_launcher(self, sidecar: dict) -> str:
        """Emit the launch_<prefix>() helper for one sidecar."""
        raise NotImplementedError

    def kernel_includes(self, sidecar: dict) -> list[str]:
        """Per-KERNEL includes for the generated .cpp, e.g. the ABI header
        export wrote. launcher_includes is the per-KIND counterpart, and
        most toolchains need only that."""
        return []

    def validate_abi(self, sidecar: dict) -> None:
        """Refuse an exported ABI this kind's launcher would marshal wrongly.

        Called by the generator for every sidecar it is about to emit a
        launcher for. The launcher is a fixed template while the exported ABI
        comes from the DSL, so a width the template does not match is a silent
        wrong value at runtime rather than a compile error. Default: nothing to
        check."""


class CuteDslToolchain(Toolchain):
    """cute.compile + export_to_c: a .o kernel object plus a header of
    per-tensor ABI structs ({data, dynamic shape/stride slots}); module
    (cubin) load is explicit and eager across devices."""

    kind = "cutedsl"
    artifact_exts = (".o", ".h")
    # The .h feeds the compiler; only the .o reaches the linker.
    link_exts = (".o",)
    launcher_includes = ()  # per-kernel header, included by prefix below

    # export_to_c emits `int32_t dynamic_shapes[]`, so the generated stub
    # must decline dims that do not fit (see gen_aot_lib._int32_size_gate).
    NARROWS_SHAPES_TO_INT32 = True

    ARCH_ENV_VAR = "CUTE_DSL_ARCH"

    # tvm_ffi: the JIT wrappers pass --enable-tvm-ffi, and cutlass imports
    # it during compile even though the exported ABI does not use it.
    REQUIRED_RUNTIMES = ("cutlass", "tvm_ffi")
    RUNTIME_DISTS = ("nvidia-cutlass-dsl", "apache-tvm-ffi")
    REQUIRED_BUILD_KEYS = ("fn", "fake_args", "tensor_args")

    # Rendered into the generated file's anonymous namespace, so the module
    # handle, the once_flag and launch_ itself all get internal linkage --
    # nothing here is part of libtorch_cuda's exported surface.
    LAUNCHER_TMPL = """\
{prefix}_Kernel_Module_t {prefix}_module;
c10::once_flag {prefix}_loaded;

void launch_{prefix}({tparams}, c10::Stream stream) {{
  c10::call_once({prefix}_loaded, [] {{ {prefix}_Kernel_Module_Load(&{prefix}_module); }});
{fills}
  int32_t rc = cute_dsl_{prefix}_wrapper(&{prefix}_module, {call_args},
                                         c10::cuda::CUDAStream(stream).stream());
  TORCH_CHECK(rc == 0, "{prefix} launch failed with code ", rc);
}}
"""

    _warmed_up = False

    @classmethod
    def _warm_up_exporter(cls) -> None:
        """Build one JIT engine in this process so export_to_c works for
        ANY --gpu-arch.

        export_to_c ultimately calls export_module_to_bytes(...,
        enable_pic=True), which needs the LLVM machinery that only gets
        initialized when the DSL creates a JIT engine. dsl.py creates one
        when `num_kernels == 0 or compile_gpu_arch == envar.arch`, so
        without this a cross-arch export is the FIRST compile in a fresh
        process and dies with "Failed to dump object file with PIC
        relocation" -- while the very same call succeeds if any
        engine-creating compile ran before it.

        A kernel-free @cute.jit takes the num_kernels == 0 branch, so it
        initializes the engine for any target: ~0.12s, once per process,
        no CUDA device needed. That is what lets one worker export for
        any mix of arches.
        """
        if cls._warmed_up:
            return
        # Defined in a helper module, not here: this file has `from
        # __future__ import annotations`, which turns the jit function's
        # parameter annotation into a string the DSL cannot resolve
        # ("NameError: name 'Float32' is not defined").
        from tools.native_aot.cutedsl_warmup import warm_up

        warm_up()
        cls._warmed_up = True

    def export(self, b: dict, out_dir: str, arch: str | None = None) -> dict:
        import cutlass.cute as cute

        # Optional compile options (e.g. --enable-assertions). Must match
        # what the op's JIT wrapper passes (minus --enable-tvm-ffi, which
        # would change the exported ABI) or the two routes' SASS diverges.
        opts = b.get("options")
        if arch:
            # --gpu-arch is authoritative: dsl.py prefers
            # compile_options.gpu_arch over the CUTE_DSL_ARCH env var, so
            # one process can export for several arches.
            opts = f"{opts} --gpu-arch {arch}" if opts else f"--gpu-arch {arch}"
            self._warm_up_exporter()
        compiled = cute.compile(
            b["fn"], *b["fake_args"], **({"options": opts} if opts else {})
        )
        compiled.export_to_c(
            file_path=out_dir, file_name=b["prefix"], function_prefix=b["prefix"]
        )
        sidecar = {"tensor_args": b["tensor_args"]}
        if b.get("scalar_args"):
            sidecar["scalar_args"] = b["scalar_args"]
        return sidecar

    def kernel_includes(self, sidecar: dict) -> list[str]:
        # _include_dir is the path from the generated source to this artifact's
        # tree (source at <root>/<op>/, kernels at <root>/<arch>/<op>/). Joined
        # with "/", not os.path.join: an #include takes forward slashes on every
        # platform.
        rel = sidecar.get("_include_dir", "")
        name = f"{sidecar['prefix']}.h"
        return [f'#include "{rel + "/" + name if rel else name}"']

    # One struct per tensor argument, PARSED rather than pattern-matched: every
    # text shortcut here accepted a truncating or out-of-bounds launcher. A bare
    # find() of the type name matched inside a LONGER argument's name (mA inside
    # mA_transposed), and re.search for the member took the first textual match, so
    # a commented-out `int64_t dynamic_strides[1];` stood in for the real int32_t
    # one. Hence: comments stripped, bodies captured brace-free (one that could run
    # past its terminator swallowed the next struct), members matched AS A WHOLE
    # between semicolons.
    _ABI_COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*", re.DOTALL)
    # An optional struct TAG (`typedef struct Tag {`) and optional attributes
    # before the name are both ordinary C for the same declaration.
    _ABI_STRUCT = re.compile(
        r"typedef\s+struct\s+(?:\w+\s*)?\{(?P<body>[^{}]*)\}"
        r"\s*(?:__attribute__\s*\(\(.*?\)\)\s*|alignas\s*\([^)]*\)\s*)*"
        r"(?P<name>\w+)\s*;",
        re.DOTALL,
    )
    _ABI_MEMBER = re.compile(
        r"\s*(?P<type>[A-Za-z_][\w:]*(?:\s+[A-Za-z_][\w:]*)*)"
        r"\s+(?P<field>dynamic_strides|dynamic_shapes)\s*\[\s*(?P<bound>[^\]]*?)\s*\]\s*$",
        re.DOTALL,
    )
    # An allowlist rather than "not int32_t": an unknown spelling or typedef alias
    # is refused LOUDLY, where assuming 64-bit restores the silent truncation this
    # check exists to prevent.
    _ABI_INT64 = frozenset({"int64_t", "std::int64_t", "long long", "signed long long"})

    def validate_abi(self, sidecar: dict) -> None:
        """Refuse a header whose stride slots are not int64, or whose slot counts
        do not EQUAL what the sidecar claims.

        The launcher assigns aten's int64 strides straight across, and the declared
        width is a PER-ARGUMENT property (use_32bit_stride is a per-argument kwarg),
        so int32 slots truncate through a plain implicit conversion: no warning, no
        error, a wrong stride. Shapes are cast explicitly behind the size gate, so
        only their COUNT is checked.

        EQUALITY in both directions, because the array bound and the sidecar list
        are independent statements about one number -- the bound from the DSL's fake
        args, the list hand-written in the builder -- and the launcher indexes from
        the sidecar:
          * claiming MORE stores past the end of the struct, and torch compiles with
            -Wno-array-bounds, so nothing warns (ASan reports the overflow where the
            compiler is silent);
          * claiming FEWER leaves slots of an UNINITIALIZED local unwritten.

        Anything this parser cannot read unambiguously is REFUSED, including a
        tensor claiming no slots: skipping there is how the under-claim direction
        stayed open, "claims nothing" being exactly the state that leaves every
        declared slot unwritten.

        Read from the header, so no schema change is needed, and skipped only when
        the header is ABSENT (unit fixtures); generation checks it is on disk."""
        path = os.path.join(sidecar.get("_dir") or "", f"{sidecar['prefix']}.h")
        try:
            # utf-8 with replacement, not the ambient locale: a valid non-ASCII
            # header raised UnicodeDecodeError under LC_ALL=C.
            with open(path, encoding="utf-8", errors="replace") as f:
                header = f.read()
        except FileNotFoundError:
            # ABSENT, which is not the same as unreadable: OSError is deliberately
            # not caught, so a permission error or a directory in its place still
            # raises. Nothing that ships gets here -- export re-exports a point whose
            # header is missing, and generation refuses a sidecar whose artifacts are
            # not on disk -- so this is the unit-fixture path, where a sidecar dict
            # stands in for a tree that was never written.
            return
        header = self._ABI_COMMENT.sub(" ", header)
        prefix = sidecar["prefix"]
        structs: dict[str, list[str]] = {}
        for m in self._ABI_STRUCT.finditer(header):
            structs.setdefault(m.group("name"), []).append(m.group("body"))

        targs = sidecar.get("tensor_args", [])
        if not isinstance(targs, list):
            raise RuntimeError(
                f"{path}: this sidecar's tensor_args is {type(targs).__name__}, not "
                f"a list, so it cannot describe the kernel's ABI. Re-export this "
                f"point."
            )
        for a in targs:
            if not isinstance(a, dict) or not isinstance(a.get("name"), str):
                raise RuntimeError(
                    f"{path}: a tensor_args entry is {a!r}, which names no tensor. "
                    f"The launcher fills one ABI struct per entry; re-export this "
                    f"point, and check the builder's tensor_args."
                )
            name = a["name"]
            tname = f"{prefix}_Tensor_{name}_t"
            # Two names for one thing, and both are external: the DSL's C header
            # declares dynamic_shapes, while the sidecar records dynamic_sizes after
            # aten. Kept as an explicit (header member, sidecar key) pair so every
            # message below can name the side it is about -- one that used the header
            # spelling while telling the reader to fix the builder named a sidecar key
            # that does not exist.
            claims = {
                "dynamic_strides": ("dynamic_strides", a.get("dynamic_strides") or []),
                "dynamic_shapes": ("dynamic_sizes", a.get("dynamic_sizes") or []),
            }
            for key, slots in claims.values():
                if not isinstance(slots, list):
                    raise RuntimeError(
                        f"{path}: {name}'s {key} is {slots!r}, not a list of "
                        f"dims. The launcher emits one assignment per element; fix "
                        f"the builder and re-export this point."
                    )
            found = structs.get(tname, [])
            if not found:
                raise RuntimeError(
                    f"{path}: no `typedef struct {{...}} {tname};` this parser can "
                    f"read, and the generated launcher declares that exact type. "
                    f"Either the header is not the one for this sidecar, or the DSL "
                    f"changed its C header shape -- re-export this point, and if "
                    f"the shape changed, update validate_abi. (Refused rather than "
                    f"skipped: an unreadable struct hides both a truncating width "
                    f"and a slot-count mismatch.)"
                )
            if len(found) > 1:
                raise RuntimeError(
                    f"{path}: {tname} is declared {len(found)} times, so which "
                    f"widths the compiler sees depends on the preprocessor. "
                    f"Re-export this point; if the DSL now emits conditional ABI "
                    f"variants, update validate_abi to pick the right one."
                )
            # Keyed by field. Parsing each declaration WHOLE is what stops a
            # neighbouring member standing in for it.
            declared: dict[str, tuple[str, str]] = {}
            for decl_text in found[0].split(";"):
                m = self._ABI_MEMBER.match(decl_text)
                if not m:
                    continue
                field = m.group("field")
                if field in declared:
                    raise RuntimeError(
                        f"{path}: {tname} appears to declare {field} twice, which "
                        f"is not one struct -- this parser is reading text from "
                        f"more than one declaration. Re-export this point, and if "
                        f"the DSL changed its C header shape, update validate_abi."
                    )
                declared[field] = (
                    " ".join(m.group("type").split()),
                    m.group("bound"),
                )
            for field, (key, slots) in claims.items():
                if field not in declared:
                    # MENTIONED but unread is not the same as absent: a spelling
                    # this parser cannot classify (a comma-separated declarator,
                    # an attribute between the type and the name) otherwise counted
                    # as zero slots, so a sidecar claiming zero passed while the
                    # struct declared some -- the launcher then leaves those slots
                    # of an uninitialized local unwritten. Fails closed here for the
                    # same reason the struct level does.
                    if field in found[0]:
                        raise RuntimeError(
                            f"{path}: {tname} contains text mentioning {field} that "
                            f"this parser could not read as a declaration, so its "
                            f"slot count cannot be compared with the {len(slots)} "
                            f"the sidecar claims. Re-export this point; if the DSL "
                            f"changed its C header shape, update validate_abi."
                        )
                    # The DSL omits the member at zero slots, so absent means
                    # zero -- which still has to equal the sidecar's count.
                    if slots:
                        raise RuntimeError(
                            f"{path}: {name} declares no {field} this parser can "
                            f"read, but the sidecar's {key} claims {len(slots)}. "
                            f"The launcher would assign to a member that does not "
                            f"exist. Make the builder's {key!r} list match the "
                            f"dims its fake args mark dynamic, and re-export."
                        )
                    continue
                ctype, bound = declared[field]
                if field == "dynamic_strides" and ctype not in self._ABI_INT64:
                    raise RuntimeError(
                        f"{path}: the launcher assigns aten's int64 strides "
                        f"straight into {name}'s {field}, so they must be declared "
                        f"64-bit -- this header declares `{ctype}`. Either this "
                        f"argument's stride symbols are 32-bit (truncation, "
                        f"silent) or the exported header changed shape. Mark them "
                        f"64-bit (cute.sym_int64, and do not pass "
                        f"use_32bit_stride=True for this argument) and re-export; "
                        f"if `{ctype}` IS a 64-bit spelling, add it to _ABI_INT64."
                    )
                # C reads a leading-zero literal as OCTAL -- [010] is 8 where
                # int() gives 10 -- and comparing those two counts is this check's
                # entire job, so refuse rather than parse it.
                if re.fullmatch(r"0[0-9]+[uUlL]*", bound):
                    raise RuntimeError(
                        f"{path}: {name}'s {field} bound `{bound}` has a leading "
                        f"zero, which C reads as octal, so the count this parser "
                        f"would compare is not the array's size. Re-export this "
                        f"point; if the DSL now emits octal bounds, teach "
                        f"validate_abi to read them."
                    )
                # A C bound may carry u/U/l/L suffixes; length-bounded so a
                # pathological literal cannot raise out of int().
                digits = re.fullmatch(r"(?P<n>[0-9]{1,6})[uUlL]*", bound)
                if not digits:
                    raise RuntimeError(
                        f"{path}: {name}'s {field} is declared with the bound "
                        f"`{bound}`, which is not a literal count, so it cannot be "
                        f"compared with the {len(slots)} slot(s) the sidecar's "
                        f"{key} claims. Re-export this point; if the DSL now emits "
                        f"computed bounds, update validate_abi."
                    )
                if int(digits.group("n")) != len(slots):
                    raise RuntimeError(
                        f"{path}: the sidecar's {key} claims {len(slots)} slot(s) "
                        f"for {name} but the header declares {field}[{bound}]. The "
                        f"launcher fills exactly the slots the sidecar lists, into "
                        f"an uninitialized struct, so a mismatch either stores past "
                        f"the end of that array or leaves the kernel reading an "
                        f"indeterminate value -- and torch builds with "
                        f"-Wno-array-bounds, so nothing warns. Make the builder's "
                        f"{key!r} list match the dims its fake args mark dynamic, "
                        f"and re-export this point."
                    )

    def gen_launcher(self, sidecar: dict) -> str:
        prefix = sidecar["prefix"]
        targs = sidecar["tensor_args"]
        sargs = sidecar.get("scalar_args", [])
        fills = []
        for a in targs:
            n = a["name"]
            fills.append(f"  {prefix}_Tensor_{n}_t {n}_s;")
            if a.get("read_only"):
                # const_data_ptr: a mutable data_ptr() would materialize
                # copy-on-write inputs (PAIN_POINTS P15). The ABI struct
                # field is void*, hence the const_cast; the kernel only
                # reads through it (declaration's read_only promise).
                fills.append(f"  {n}_s.data = const_cast<void*>({n}.const_data_ptr());")
            else:
                fills.append(f"  {n}_s.data = {n}.mutable_data_ptr();")
            for slot, dim in enumerate(a.get("dynamic_sizes", [])):
                fills.append(
                    f"  {n}_s.dynamic_shapes[{slot}] = static_cast<int32_t>({n}.size({dim}));"
                )
            for slot, dim in enumerate(a.get("dynamic_strides", [])):
                fills.append(f"  {n}_s.dynamic_strides[{slot}] = {n}.stride({dim});")
        params = [f"const at::Tensor& {a['name']}" for a in targs]
        params += [f"{a['ctype']} {a['name']}" for a in sargs]
        # Wrapper argument order matches the exported signature: tensor
        # structs first, then scalars (by value), then the stream.
        call_args = [f"&{a['name']}_s" for a in targs] + [a["name"] for a in sargs]
        return self.LAUNCHER_TMPL.format(
            prefix=prefix,
            tparams=", ".join(params),
            fills="\n".join(fills),
            call_args=", ".join(call_args),
        )


TOOLCHAINS: dict[str, Toolchain] = {tc.kind: tc for tc in (CuteDslToolchain(),)}


def _assert_link_exts_are_exportable(registry: dict[str, Toolchain]) -> None:
    """Every link input a kind names must be something it also exports.

    Generation iterates artifact_exts and links `if ext in link_exts`, so an ext
    the kind cannot produce is silent: it contributes no link input, nothing
    passes --no-undefined, torch_cuda links green, and the first call fails on an
    undefined symbol. Checked at import so a new toolchain cannot ship it.

    Omitting link_exts entirely is the same silence, and the subset rule cannot see
    it -- the inherited default is a subset of everything -- so it is refused
    separately."""
    for tc in registry.values():
        if tc.link_exts is None:
            raise RuntimeError(
                f"toolchain {tc.kind}: link_exts is not declared. Name the "
                f"extensions that reach the linker, or () if this kind embeds its "
                f"artifact in the generated source; inheriting the default would "
                f"link nothing, which is the same as not shipping the kernels"
            )
        extra = sorted(set(tc.link_exts) - set(tc.artifact_exts))
        if extra:
            raise RuntimeError(
                f"toolchain {tc.kind}: link_exts {sorted(tc.link_exts)} is not a "
                f"subset of artifact_exts {sorted(tc.artifact_exts)}, so {extra} "
                f"can never be exported and its kernels would not be linked"
            )


_assert_link_exts_are_exportable(TOOLCHAINS)


def get_toolchain(kind: str) -> Toolchain:
    if kind not in TOOLCHAINS:
        raise RuntimeError(
            f"unknown toolchain kind {kind!r}; known: {sorted(TOOLCHAINS)}"
        )
    return TOOLCHAINS[kind]


def all_artifact_exts() -> set[str]:
    """Every extension some toolchain writes beside a sidecar.

    ONE notion of "kernel artifact" for both sweeps that hunt undescribed files
    (export's orphan check and generation's no-declaration check), which computed
    separately could disagree about a new toolchain and leave one sweep blind."""
    return {e for tc in TOOLCHAINS.values() for e in tc.artifact_exts}


def for_backend(backend: str) -> dict[str, Toolchain]:
    """The toolchains that can emit kernels for this torch build backend
    ("cuda" or "rocm"), per each kind's BACKENDS."""
    return {k: tc for k, tc in TOOLCHAINS.items() if tc.serves_backend(backend)}
