"""Per-DSL toolchains for native-AOT kernel export and launcher codegen.

One ``Toolchain`` subclass per DSL or compiler holds everything kind-specific: how a
builder dict is validated and compiled into artifacts, what those artifacts are, and
how the generated C++ launcher marshals at::Tensor arguments into the kernel's ABI.
Adding a DSL means adding one class and registering it below.

The contract, in order of flow:

  1. An op's builder module exposes ``build(spec) -> dict``. The dict's ``kind``
     selects the toolchain, ``prefix`` names the artifacts, and the kind's
     ``REQUIRED_BUILD_KEYS`` say what else it needs.
  2. ``export(build_result, out_dir)`` compiles, writes
     ``<prefix>.<artifact_exts>``, and returns the marshalling metadata to store in
     the ``<prefix>.json`` sidecar -- the only channel from export-time knowledge to
     launcher codegen.
  3. ``gen_launcher(sidecar)`` emits C++ with one signature for every kind:

         void launch_<prefix>(const at::Tensor&..., <scalars>..., c10::Stream)

     Everything above the launcher is kind-blind, so the shared contract names no
     CUDA type; each launcher narrows the stream to the handle its own C ABI takes.

Class attributes the driver scripts read:

  * ``artifact_exts``: extensions written beside the sidecar.
  * ``link_exts``: which of those reach the linker. Empty for a kind that embeds its
    artifact in the generated source; None means not declared, refused at import.
  * ``launcher_includes``, ``kernel_includes(sidecar)``: includes for the generated
    .cpp, per kind and per kernel.
  * ``NARROWS_SHAPES_TO_INT32``: this kind's ABI takes int32 extents, so generation
    emits a gate declining dims past INT32_MAX.
  * ``ARCH_ENV_VAR``: the env var this kind reads for its arch. export refuses it
    and asks for --arch, since one kind's variable cannot answer for a tree holding
    several kinds' artifacts.

Export runs in stage 2, after torch is built, so a builder module may import torch.
"""

from __future__ import annotations

import os
import re


class Toolchain:
    kind: str = ""
    artifact_exts: tuple[str, ...] = ()
    # None means not declared. () is a real value -- a kind whose launcher embeds the
    # artifact rather than linking it -- so an inherited default must not pass for it.
    link_exts: tuple[str, ...] | None = None
    launcher_includes: tuple[str, ...] = ()

    # Torch build backends this kind can emit kernels for, spelled as torch reports
    # them ("cuda", "rocm"). A build on another backend never asks for this
    # toolchain, so its runtime being absent there is expected rather than a skip
    # worth reporting.
    BACKENDS: tuple[str, ...] = ("cuda",)

    # Importable modules needed to COMPILE a kernel of this kind. Absent once a
    # declaration for this backend reaches export, they are fatal: exporting only
    # some of an op's kernels ships a wheel that silently underperforms. Build
    # without the DSL wheels with TORCH_NATIVE_AOT=0. Nothing at runtime needs
    # these -- the exported artifacts are self-contained.
    REQUIRED_RUNTIMES: tuple[str, ...] = ()

    # Distributions whose versions define this kind's compiler, recorded per sidecar
    # and compared on the next run: the DSL version appears in no file the source
    # closure hashes, so without this an upgraded wheel invalidates nothing and the
    # tree mixes compilers. Distribution names rather than module names keep the
    # lookup metadata-only, with no MLIR import on the skip path.
    RUNTIME_DISTS: tuple[str, ...] = ()

    # True when this kind's exported ABI carries int32_t shape slots. A property of
    # the ABI, hence per-kind: Triton takes its scalar widths from the kernel's own
    # signature.
    NARROWS_SHAPES_TO_INT32: bool = False

    # Env var this kind falls back to when no arch is given explicitly. The sidecar
    # records the arch it resolved to, so changing only this variable is not
    # mistaken for a tree that already exported.
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

        ``arch`` is an sm string ("sm_90a"), or None to detect from the local
        device. Given one explicitly, no toolchain touches the CUDA driver, so
        export runs on GPU-less machines and one process can export for several
        arches."""
        raise NotImplementedError

    def gen_launcher(self, sidecar: dict) -> str:
        """Emit the launch_<prefix>() helper for one sidecar."""
        raise NotImplementedError

    def kernel_includes(self, sidecar: dict) -> list[str]:
        """Per-kernel includes for the generated .cpp, such as an ABI header export
        wrote. ``launcher_includes`` is the per-kind counterpart."""
        return []

    def validate_abi(self, sidecar: dict) -> None:
        """Refuse an exported ABI this kind's launcher would marshal wrongly.

        The launcher is a fixed template while the ABI comes from the DSL, so a
        width the template does not match is a wrong value at runtime rather than a
        compile error. Default: nothing to check."""


class CuteDslToolchain(Toolchain):
    """cute.compile + export_to_c: a .o kernel object plus a header of per-tensor
    ABI structs ({data, dynamic shape/stride slots}). Module load is explicit and
    eager across devices."""

    kind = "cutedsl"
    artifact_exts = (".o", ".h")
    # The .h feeds the compiler; only the .o reaches the linker.
    link_exts = (".o",)
    launcher_includes = ()  # per-kernel header, included by prefix below

    # export_to_c emits `int32_t dynamic_shapes[]`.
    NARROWS_SHAPES_TO_INT32 = True

    ARCH_ENV_VAR = "CUTE_DSL_ARCH"

    # tvm_ffi: the JIT wrappers pass --enable-tvm-ffi, and cutlass imports it during
    # compile even though the exported ABI does not use it.
    REQUIRED_RUNTIMES = ("cutlass", "tvm_ffi")
    RUNTIME_DISTS = ("nvidia-cutlass-dsl", "apache-tvm-ffi")
    REQUIRED_BUILD_KEYS = ("fn", "fake_args", "tensor_args")

    # Rendered into the generated file's anonymous namespace, so the module handle,
    # the once_flag and launch_ itself all get internal linkage.
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
        """Build one JIT engine in this process so export_to_c works for any
        --gpu-arch.

        export_to_c needs LLVM machinery that the DSL initializes only when it
        creates a JIT engine, which it does for a compile matching the ambient arch.
        Without this, a cross-arch export as the first compile in a process fails
        with "Failed to dump object file with PIC relocation". A kernel-free
        @cute.jit initializes the engine for any target in ~0.12s, once per process,
        with no CUDA device needed."""
        if cls._warmed_up:
            return
        # A helper module, because this file's `from __future__ import annotations`
        # would stringify the jit function's annotation past the DSL's reach.
        from tools.native_aot.cutedsl_warmup import warm_up

        warm_up()
        cls._warmed_up = True

    def export(self, b: dict, out_dir: str, arch: str | None = None) -> dict:
        import cutlass.cute as cute

        # Optional compile options (e.g. --enable-assertions). These must match what
        # the op's JIT wrapper passes, minus --enable-tvm-ffi, which would change
        # the exported ABI; otherwise the two routes' SASS diverges.
        opts = b.get("options")
        if arch:
            # --gpu-arch outranks the CUTE_DSL_ARCH env var, so one process can
            # export for several arches.
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
        # _include_dir is the path from the generated source to this artifact's tree
        # (source at <root>/<op>/, kernels at <root>/<arch>/<op>/). Joined with "/",
        # not os.path.join: an #include takes forward slashes everywhere.
        rel = sidecar.get("_include_dir", "")
        name = f"{sidecar['prefix']}.h"
        return [f'#include "{rel + "/" + name if rel else name}"']

    # One struct per tensor argument, parsed rather than pattern-matched: comments
    # are stripped, bodies captured brace-free so one cannot run past its
    # terminator, and members matched as a whole between semicolons. A looser match
    # reads a neighbouring member's width as this one's.
    _ABI_COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*", re.DOTALL)
    # A struct tag (`typedef struct Tag {`) and attributes before the name are both
    # ordinary C for the same declaration.
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
    # An allowlist rather than "not int32_t": an unknown spelling or typedef alias is
    # refused loudly, where assuming 64-bit would restore the silent truncation.
    _ABI_INT64 = frozenset({"int64_t", "std::int64_t", "long long", "signed long long"})

    def validate_abi(self, sidecar: dict) -> None:
        """Refuse a header whose stride slots are not int64, or whose slot counts do
        not equal what the sidecar claims.

        The launcher assigns aten's int64 strides straight across, and the declared
        width is a per-argument property, so int32 slots truncate through an
        implicit conversion. Shapes are cast explicitly behind the size gate, so
        only their count is checked.

        Counts must match in both directions: the array bound and the sidecar list
        are independent statements about one number, and the launcher indexes from
        the sidecar. Claiming more stores past the end of the struct (torch builds
        with -Wno-array-bounds, so nothing warns); claiming fewer leaves slots of an
        uninitialized local unwritten. Anything this parser cannot read
        unambiguously is refused, including a tensor claiming no slots.

        Read from the header, so no schema change is needed, and skipped only when
        the header is absent; generation checks it is on disk."""
        path = os.path.join(sidecar.get("_dir") or "", f"{sidecar['prefix']}.h")
        try:
            # utf-8 with replacement rather than the ambient locale, which raises on
            # a valid non-ASCII header under LC_ALL=C.
            with open(path, encoding="utf-8", errors="replace") as f:
                header = f.read()
        except FileNotFoundError:
            # Absent, which is not unreadable: OSError is deliberately not caught, so
            # a permission error still raises, and a header that IS there but
            # unparsable is refused below. Returning is not an acceptance -- the
            # production caller is generation, which refuses a sidecar whose
            # artifacts are gone before emitting any source.
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
            # Two names for one thing, both external: the DSL's C header declares
            # dynamic_shapes, the sidecar records dynamic_sizes after aten. Kept as
            # an explicit (header member, sidecar key) pair so each message below can
            # name the side it is about.
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
                    f"the shape changed, update validate_abi."
                )
            if len(found) > 1:
                raise RuntimeError(
                    f"{path}: {tname} is declared {len(found)} times, so which "
                    f"widths the compiler sees depends on the preprocessor. "
                    f"Re-export this point; if the DSL now emits conditional ABI "
                    f"variants, update validate_abi to pick the right one."
                )
            # Keyed by field, each declaration parsed whole.
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
                    # Mentioned but unread is not absent: a spelling this parser
                    # cannot classify would otherwise count as zero slots and pass a
                    # sidecar claiming zero, leaving the struct's real slots
                    # unwritten. Fails closed, as the struct level does.
                    if field in found[0]:
                        raise RuntimeError(
                            f"{path}: {tname} contains text mentioning {field} that "
                            f"this parser could not read as a declaration, so its "
                            f"slot count cannot be compared with the {len(slots)} "
                            f"the sidecar claims. Re-export this point; if the DSL "
                            f"changed its C header shape, update validate_abi."
                        )
                    # The DSL omits the member at zero slots, so absent means zero --
                    # which still has to equal the sidecar's count.
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
                        f"64-bit -- this header declares `{ctype}`. Mark this "
                        f"argument's strides 64-bit (cute.sym_int64, without "
                        f"use_32bit_stride=True) and re-export; if `{ctype}` is "
                        f"itself a 64-bit spelling, add it to _ABI_INT64."
                    )
                # C reads a leading-zero literal as octal -- [010] is 8, not 10 --
                # and comparing counts is this check's whole job.
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
                        f"launcher fills the slots the sidecar lists into an "
                        f"uninitialized struct, so a mismatch stores past the end "
                        f"of that array or leaves it unwritten. Make the builder's "
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
                # copy-on-write inputs. The ABI field is void*, hence the
                # const_cast; the kernel only reads through it.
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
        # Matches the exported signature: tensor structs, then scalars by value,
        # then the stream.
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

    Generation links `if ext in link_exts`, so an ext the kind cannot produce
    contributes no link input: torch_cuda links green and the first call fails on an
    undefined symbol. Omitting link_exts entirely is the same silence, and the
    subset rule cannot catch it, so it is refused separately."""
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

    One notion of "kernel artifact" for both sweeps that hunt undescribed files --
    export's orphan check and generation's no-declaration check -- which computed
    separately could disagree about a new toolchain."""
    return {e for tc in TOOLCHAINS.values() for e in tc.artifact_exts}


def for_backend(backend: str) -> dict[str, Toolchain]:
    """The toolchains that can emit kernels for this torch build backend
    ("cuda" or "rocm"), per each kind's BACKENDS."""
    return {k: tc for k, tc in TOOLCHAINS.items() if tc.serves_backend(backend)}
