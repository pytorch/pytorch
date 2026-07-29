import functools
import os
from functools import cached_property
from typing import Any
from typing_extensions import Unpack

from ..utils import is_rocm
from .triton_compat import ASTSource, CompiledKernel, knobs as triton_knobs
from .triton_helpers import get_constexprs


@functools.lru_cache(None)
def _tma_arg_helpers():
    """Cached (make_arg, TensorDescriptor) for host-side TMA arg expansion.
    make_arg(arg, metadata) -> [CUtensorMap, *shape, *strides]; the nvidia
    backend's make_tensordesc_arg ignores its third arg, so we pass None."""
    from triton.backends.nvidia.driver import make_tensordesc_arg
    from triton.tools.tensor_descriptor import TensorDescriptor

    def make_arg(arg, metadata):
        return make_tensordesc_arg(arg, metadata, None)

    return make_arg, TensorDescriptor


class StaticallyLaunchedTritonKernel:
    """
    Parses the metadata of a CompiledKernel from Triton into a structure that can
    launch the cuda kernel directly. Only works for triton kernels compiled to cubin.

    Doing this avoids C++ codegen and compilation during compile, since we can use a
    statically compiled library to launch the kernel. To avoid mallocing for the arguments,
    we have a launcher for different numbers of arguments up to a max. StaticCudaLauncher
    only supports # of arguments up until 10 for now.

    Workflow:
    Compile time:
    1. Compile a kernel with triton and get a CompiledKernel
    2. Instantiate kernel = StaticallyLaunchedTritonKernel(triton_kernel)
    3. Write to a cubin file: kernel.write_cubin_to_file(filepath)
    4. Call kernel.load_kernel() (CUDA should be initialized by this point) to load the cubin
    Runtime:
    5. Call kernel.run(grid, stream, args) to launch the kernel

    Note that after step 3, StaticallyLaunchedTritonKernel is fully pickleable/serializable.
    This allows it to be cached by FXGraphCache/TritonBundler, as well as sent from the worker
    to the parent process in inductor.

    There are two main versions of triton that we wish to support: 3.3 and 3.2. Triton makes considerable changes
    to how it handles constants in 3.3, so there's some special logic necessary to handle both versions.
    """

    @cached_property
    def C_impl(self):
        raise NotImplementedError

    def __init__(self, kernel: CompiledKernel) -> None:
        # pyrefly: ignore [missing-attribute]
        self.name = kernel.src.fn.__name__
        # pyrefly: ignore [missing-attribute]
        self.cubin_path = kernel._cubin_path

        # Used by torch.compile to filter constants in older triton versions
        # pyrefly: ignore [missing-attribute]
        self.arg_names = kernel.src.fn.arg_names

        # Const exprs that are declared by the triton kernel directly
        # Used to generate the kernel launcher's def args
        # pyrefly: ignore [missing-attribute]
        self.declared_constexprs = get_constexprs(kernel.src.fn)

        # pyrefly: ignore [missing-attribute]
        self.hash = kernel.hash

        if triton_knobs is None:
            # pyrefly: ignore [missing-attribute]
            launch_enter = kernel.__class__.launch_enter_hook
            # pyrefly: ignore [missing-attribute]
            launch_exit = kernel.__class__.launch_exit_hook
        else:
            launch_enter = triton_knobs.runtime.launch_enter_hook
            launch_exit = triton_knobs.runtime.launch_exit_hook

        def hook_is_empty(hook: Any) -> bool:
            if hook is None:
                return True
            if (
                triton_knobs
                and (HookChain := getattr(triton_knobs, "HookChain", None)) is not None
                and isinstance(hook, HookChain)
            ):
                # Support hooks after https://github.com/triton-lang/triton/pull/7866
                return len(hook.calls) == 0
            return False

        if not hook_is_empty(launch_enter) or not hook_is_empty(launch_exit):
            raise NotImplementedError(
                "We don't support launch enter or launch exit hooks"
            )
        # pyrefly: ignore [missing-attribute]
        self.num_warps = kernel.metadata.num_warps
        self.shared = (
            # pyrefly: ignore [missing-attribute]
            kernel.shared if hasattr(kernel, "shared") else kernel.metadata.shared
        )

        def needs_scratch_arg(scratch_name: str, param_name: str) -> bool:
            # pyrefly: ignore [missing-attribute]
            if hasattr(kernel.metadata, param_name):
                # pyrefly: ignore [missing-attribute]
                if getattr(kernel.metadata, param_name) > 0:
                    raise NotImplementedError(
                        f"{scratch_name} scratch not yet supported"
                    )
                return True
            return False

        # Newer triton versions pass an extra global scratch parameter to the compiled cuda kernel.
        # Inductor never uses this field or enables it, but we still have to pass
        # an extra None into the set of params if its enabled
        self.has_global_scratch = needs_scratch_arg("Global", "global_scratch_size")
        # same situation for profile scratch - triton-lang/triton#7258
        self.has_profile_scratch = needs_scratch_arg("Profile", "profile_scratch_size")

        # pyrefly: ignore [missing-attribute]
        self.tensordesc_meta = getattr(kernel.metadata, "tensordesc_meta", None)
        self._has_tensordesc = False
        # pyrefly: ignore [missing-attribute]
        self.arg_tys = self.arg_ty_from_signature(kernel.src)
        self.function: int | None = None  # Loaded by load_kernel(on the parent process)
        self.module: int | None = None  # Owns the HIP/CUDA module loaded for function
        num_ctas = 1
        if hasattr(kernel, "num_ctas"):
            num_ctas = kernel.num_ctas
        elif hasattr(kernel, "metadata"):
            num_ctas = kernel.metadata.num_ctas

        if num_ctas != 1:
            raise NotImplementedError(
                "Static cuda launcher only supports num_ctas == 1"
            )

    def reload_cubin_from_raw(self, filepath: str) -> str:
        """
        If the cubin file triton generated gets deleted under us, we can
        reload it from the raw cubin file.
        """
        if self.cubin_path is None:
            if self.cubin_raw is None:
                raise AssertionError("cubin_raw must be set when cubin_path is None")
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "wb") as f:
                f.write(self.cubin_raw)
                self.cubin_path = filepath  # pyre-ignore
        return self.cubin_path

    def load_kernel(self, device: int) -> None:
        if self.function is not None:
            return

        if not hasattr(self, "cubin_path"):
            raise AssertionError("cubin_path attribute not set before load_kernel")
        if self.cubin_path is None:
            raise AssertionError("cubin_path must not be None before load_kernel")
        (self.module, self.function, self.n_regs, self.n_spills) = (
            self.C_impl._load_kernel(self.cubin_path, self.name, self.shared, device)
        )
        # Don't need the cubin path anymore now that we've loaded
        self.cubin_path = None
        self.cubin_raw = None

    def close(self) -> None:
        mod = self.module
        if mod is None:
            return
        # Clear Python-visible handles first so repeated cleanup is harmless even if
        # the driver reports an error while unloading.
        self.module = None
        self.function = None
        self.C_impl._unload_kernel(mod)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    @functools.lru_cache
    def type_mappings() -> dict[str, str]:
        return {
            "i1": "i",
            "i8": "b",
            "i16": "h",
            "i32": "i",
            "i64": "l",
            "u1": "I",
            "u8": "B",
            "u16": "H",
            "u32": "I",
            "u64": "K",
            "fp16": "f",
            "bf16": "f",
            "fp32": "f",
            "f32": "f",
            "fp64": "d",
            # TODO handle nvTmaDesc/CUtensormap
        }

    def extract_type(self, ty: str) -> str:
        """
        Takes a triton type from CompiledKernel.signature and
        converts it into a single char encoding. _StaticCudaLauncher
        will switch on this char to figure out what type the underlying
        value should be passed to the triton kernel as.
        """
        if ty[0] == "*":
            return "O"
        elif ty == "nvTmaDesc" or ty.startswith("tensordesc<"):
            raise NotImplementedError("TMA descriptor kernels are not yet supported")
        return StaticallyLaunchedTritonKernel.type_mappings()[ty]

    def _expand_tensordesc_type(self, ty: str) -> str:
        """Expand a tensordesc<dtype[shape]> signature entry into the per-arg
        type chars triton lowers it to (mirrors nvidia driver expand_signature):
        nvTmaDesc path -> CUtensorMap + shape(i32) + strides(i64); host-decomposed
        path -> base ptr + shape/strides(i64) + 2 flags + shape(i32) + strides(i64).

        This mirrors the string parse in triton's own expand_signature: the only
        thing we need is the descriptor's rank, which lives in the shape list of
        the serialized "tensordesc<dtype[d0, d1, ...]>" form. We parse that string
        here rather than importing expand_signature because that helper's argument
        signature differs across triton versions (a 2-arg (signature, meta) form
        in the fb-triton nvidia driver vs a 3-arg (signature, meta,
        descriptor_type) form as backends converge on a shared helper), so
        importing it would break across triton bumps; the serialized string
        format is stable.
        TODO: switch to triton's expand_signature once the versions converge on a
        single argument signature.
        """
        import re

        # Same regex triton's expand_signature uses. We only need the rank, so
        # the captured dtype group is unused (kept to match triton's pattern).
        match = re.match(r"tensordesc<([^\[>]*)\[([^\]]*)\]", ty)
        if match is None:
            raise NotImplementedError(f"Could not parse tensordesc type: {ty}")
        ndim = match.group(2).count(",") + 1
        meta = self.tensordesc_meta
        idx = self._tensordesc_idx
        self._tensordesc_idx += 1
        per_desc_meta = meta[idx] if meta else None
        if per_desc_meta is None:
            chars = ["O"] + ["l"] * (2 * ndim) + ["i", "i"]
        else:
            chars = ["M"]
        chars += ["i"] * ndim + ["l"] * ndim
        return "".join(chars)

    def arg_ty_from_signature(self, src: ASTSource) -> str:
        def index_key(i: Any) -> int:
            if isinstance(i, str):
                # pyrefly: ignore [missing-attribute]
                return src.fn.arg_names.index(i)
            elif isinstance(i, tuple):
                # In triton 3.3, src.fn.constants has tuples as a key
                return i[0]
            else:
                return i

        # pyrefly: ignore [missing-attribute]
        signature = {index_key(key): value for key, value in src.signature.items()}
        # Triton uses these as the main way to filter out constants passed to their cubin
        constants = [index_key(key) for key in getattr(src, "constants", dict())]
        # This value is always a superset of kernel.fn.constexprs: kernel.fn.constexprs are
        # constants declared by the triton kernel directly, whereas this list can have
        # constants that are unused by the triton kernel that triton figured out during
        # compilation.
        self.full_constexprs = constants
        # Despite requiring them to be passed in, the triton CUDA launcher
        # completely ignores the constexprs passed into it when generating code.
        # So we can ignore them here too
        params = []
        self._tensordesc_idx = 0

        for i in sorted(signature.keys()):
            ty = signature[i]
            # In newer triton versions, constants are passed in to signature with type `constexpr`
            # In older triton versions, there can be constants in src.constants that are not `constexpr` in signature
            # so we check both here
            if ty == "constexpr" or i in constants:
                pass
            elif isinstance(ty, str) and ty.startswith("tensordesc<"):
                self._has_tensordesc = True
                params.append(self._expand_tensordesc_type(ty))
            else:
                # pyrefly: ignore [bad-argument-type]
                params.append(self.extract_type(ty))
        return "".join(params)

    def __getstate__(self) -> dict[str, Any]:
        # Remove objects that are no longer valid for pickling
        state = self.__dict__.copy()
        state["function"] = None
        state["module"] = None
        # Cubin paths aren't consistent across processes, so we clear
        # and reload them.
        state["cubin_path"] = None
        return state

    def _expand_tma_args(self, args: tuple[object, ...]) -> tuple[object, ...]:
        """Expand host-side TMA TensorDescriptor args into the flat kernel params
        (CUtensorMap + shape + strides) so they match the expanded type string."""
        make_tensordesc_arg, TensorDescriptor = _tma_arg_helpers()

        meta = self.tensordesc_meta
        out: list[object] = []
        td_idx = 0
        for arg in args:
            if isinstance(arg, TensorDescriptor):
                per_desc_meta = meta[td_idx] if meta else None
                td_idx += 1
                out.extend(make_tensordesc_arg(arg, per_desc_meta))
            else:
                out.append(arg)
        return tuple(out)

    def run(
        self,
        grid_x: int,
        grid_y: int,
        grid_z: int,
        stream: int,
        *args: Unpack[tuple[object, ...]],
    ) -> None:
        """Actually run the kernel at runtime. This function is the hot codepath."""

        # Assert load_kernel() has been called and args match
        if self.function is None:
            raise AssertionError("load_kernel() must be called before run()")

        # TODO: actually, if the args *don't* match, we probably should
        # throw an exception. But if inductor is the only one calling this
        # thing, it should always match.
        # Get rid of constants before passing to cubin launcher

        if self._has_tensordesc:
            args = self._expand_tma_args(args)

        arg_tys = self.arg_tys

        if is_rocm():
            # ROCm/HIP kernel ABI: The Triton HIP backend ALWAYS includes both
            # global_scratch and profile_scratch parameters in the kernel signature,
            # even when the kernel doesn't use them (i.e., when has_*_scratch is False).
            #
            # This differs fundamentally from CUDA, where these parameters are only
            # present in the signature if the corresponding has_*_scratch flag is True.
            #
            # The flags indicate whether memory will be allocated/used:
            # - has_global_scratch: Whether global scratch workspace is needed
            # - has_profile_scratch: Whether profiling instrumentation is enabled
            #
            # However, regardless of flag values, we MUST always pass both parameters
            # to match the HIP kernel ABI. Passing None is safe:
            #
            # - If scratch is not needed (has_*_scratch=False or scratch_size=0):
            #   The None becomes nullptr, which the kernel never dereferences
            #
            # - If scratch is needed (has_*_scratch=True and scratch_size>0):
            #   The None becomes nullptr initially, but the HIP runtime intercepts
            #   the kernel launch, allocates the required scratch memory based on
            #   kernel metadata, and replaces the nullptr with a valid pointer before
            #   the kernel actually executes
            #
            # Not passing both parameters causes segmentation faults because the kernel
            # expects them at specific positions in the argument array.
            arg_tys = arg_tys + "OO"
            args = (*args, None, None)

        else:
            for has_scratch in [self.has_global_scratch, self.has_profile_scratch]:
                if has_scratch:
                    arg_tys = arg_tys + "O"
                    args = (*args, None)
        # pyrefly: ignore [bad-argument-type]
        if len(args) != len(arg_tys):
            raise AssertionError(f"Expected {len(arg_tys)} args, got {len(args)}")

        # TODO: can handle grid functions here or in C++, so
        # that we don't need the grid handler above.
        self.C_impl._launch_kernel(
            self.function,
            grid_x,
            grid_y,
            grid_z,
            self.num_warps,
            self.shared,
            arg_tys,
            args,
            stream,
        )


class StaticallyLaunchedCudaKernel(StaticallyLaunchedTritonKernel):
    @cached_property
    def C_impl(self):
        from torch._C import _StaticCudaLauncher

        return _StaticCudaLauncher

    def __init__(self, kernel: CompiledKernel) -> None:
        # pyrefly: ignore [missing-attribute]
        if "hsaco" in kernel.asm:
            # pyrefly: ignore [missing-attribute]
            self.cubin_raw = kernel.asm["hsaco"]

        # pyrefly: ignore [missing-attribute]
        elif "cubin" in kernel.asm:
            # pyrefly: ignore [missing-attribute]
            self.cubin_raw = kernel.asm["cubin"]
        else:
            raise RuntimeError(
                "Expected either 'hsaco' (ROCm) or 'cubin' (CUDA) in kernel.asm"
            )
        super().__init__(kernel)


class StaticallyLaunchedXpuKernel(StaticallyLaunchedTritonKernel):
    @cached_property
    def C_impl(self):
        from torch._C import _StaticXpuLauncher

        return _StaticXpuLauncher

    def __init__(self, kernel: CompiledKernel) -> None:
        # pyrefly: ignore [missing-attribute]
        self.cubin_raw = kernel.asm.get("zebin", None)
        super().__init__(kernel)

    def load_kernel(self, device: int) -> None:
        if self.function is not None:
            return

        if not hasattr(self, "cubin_path"):
            raise AssertionError("expected cubin_path attribute to be set")
        if self.cubin_path is None:
            raise AssertionError("expected cubin_path to not be None")
        # The XPU static launcher returns a PyCapsule for the loaded SYCL kernel,
        # not a separate module/function pair like the CUDA/HIP launcher.
        (self.function, self.n_regs, self.n_spills) = self.C_impl._load_kernel(
            self.cubin_path, self.name, self.shared, device
        )
        self.module = None
        self.cubin_path = None
        self.cubin_raw = None

    def close(self) -> None:
        self.module = None
        # Drop the PyCapsule reference so its destructor can release sycl::kernel.
        self.function = None


def statically_launched_kernel_by_device(
    kernel: CompiledKernel, device_type: str = "cuda"
) -> StaticallyLaunchedTritonKernel:
    if device_type in ("cuda", "hip"):
        return StaticallyLaunchedCudaKernel(kernel)
    elif device_type == "xpu":
        return StaticallyLaunchedXpuKernel(kernel)
    else:
        raise NotImplementedError(
            f"Device type {device_type} is not supported for static launcher"
        )
