import functools
from typing import Any, Optional
from typing_extensions import Unpack

from .triton_compat import ASTSource, CompiledKernel, knobs


class StaticallyLaunchedCudaKernel:
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
    2. Instantiate kernel = StaticallyLaunchedCudaKernel(triton_kernel)
    3. Write to a cubin file: kernel.write_cubin_to_file(filepath)
    4. Call kernel.load_kernel() (CUDA should be initialized by this point) to load the cubin
    Runtime:
    5. Call kernel.run(grid, stream, args) to launch the kernel

    Note that after step 3, StaticallyLaunchedCudaKernel is fully pickleable/serializable.
    This allows it to be cached by FXGraphCache/TritonBundler, as well as sent from the worker
    to the parent process in inductor.

    There are two main versions of triton that we wish to support: 3.3 and 3.2. Triton makes considerable changes
    to how it handles constants in 3.3, so there's some special logic necessary to handle both versions.
    """

    def __init__(self, kernel: CompiledKernel) -> None:
        self.name = kernel.src.fn.__name__
        self.cubin_path = kernel._cubin_path

        # Used by torch.compile to filter constants in older triton versions
        self.arg_names = kernel.src.fn.arg_names

        # Const exprs that are declared by the triton kernel directly
        # Used to generate the kernel launcher's def args
        self.declared_constexprs = kernel.src.fn.constexprs

        self.hash = kernel.hash

        if knobs is None:
            launch_enter = kernel.__class__.launch_enter_hook
            launch_exit = kernel.__class__.launch_exit_hook
        else:
            launch_enter = knobs.runtime.launch_enter_hook
            launch_exit = knobs.runtime.launch_exit_hook

        if launch_enter is not None or launch_exit is not None:
            raise NotImplementedError(
                "We don't support launch enter or launch exit hooks"
            )
        self.num_warps = kernel.metadata.num_warps
        self.shared = (
            kernel.shared if hasattr(kernel, "shared") else kernel.metadata.shared
        )

        # Newer triton versions pass an extra global scratch parameter to the compiled cuda kernel.
        # Inductor never uses this field or enables it, but we still have to pass
        # an extra None into the set of params if its enabled
        if hasattr(kernel.metadata, "global_scratch_size"):
            if kernel.metadata.global_scratch_size > 0:
                raise NotImplementedError("Global scratch not yet supported")
            else:
                self.has_global_scratch = True
        else:
            self.has_global_scratch = False

        self.arg_tys = self.arg_ty_from_signature(kernel.src)
        self.function: Optional[int] = (
            None  # Loaded by load_kernel(on the parent process)
        )
        num_ctas = 1
        if hasattr(kernel, "num_ctas"):
            num_ctas = kernel.num_ctas
        elif hasattr(kernel, "metadata"):
            num_ctas = kernel.metadata.num_ctas

        if num_ctas != 1:
            raise NotImplementedError(
                "Static cuda launcher only supports num_ctas == 1"
            )

    def load_kernel(self, device: int) -> None:
        from torch._C import _StaticCudaLauncher

        if self.function is not None:
            return

        assert hasattr(self, "cubin_path")
        assert self.cubin_path is not None
        (self.function, self.n_regs, self.n_spills) = _StaticCudaLauncher._load_kernel(
            self.cubin_path, self.name, self.shared, device
        )
        # Don't need the cubin path anymore now that we've loaded
        self.cubin_path = None

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
        elif ty == "nvTmaDesc":
            raise NotImplementedError("nvTmaDesc kernels are not yet supported")
        return StaticallyLaunchedCudaKernel.type_mappings()[ty]

    def arg_ty_from_signature(self, src: ASTSource) -> str:
        def index_key(i: Any) -> int:
            if isinstance(i, str):
                return src.fn.arg_names.index(i)
            elif isinstance(i, tuple):
                # In triton 3.3, src.fn.constants has tuples as a key
                return i[0]
            else:
                return i

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

        for i in sorted(signature.keys()):
            ty = signature[i]
            # In newer triton versions, constants are passed in to signature with type `constexpr`
            # In older triton versions, there can be constants in src.constants that are not `constexpr` in signature
            # so we check both here
            if ty == "constexpr" or i in constants:
                pass
            else:
                params.append(self.extract_type(ty))
        return "".join(params)

    def __getstate__(self) -> dict[str, Any]:
        # Remove objects that are no longer valid for pickling
        state = self.__dict__.copy()
        state["function"] = None
        # Cubin paths aren't consistent across processes, so we clear
        # and reload them.
        state["cubin_path"] = None
        return state

    def run(
        self,
        grid_x: int,
        grid_y: int,
        grid_z: int,
        stream: int,
        *args: Unpack[tuple[object, ...]],
    ) -> None:
        """Actually run the kernel at runtime. This function is the hot codepath."""
        from torch._C import _StaticCudaLauncher

        # Assert load_kernel() has been called and args match
        assert self.function is not None

        # TODO: actually, if the args *don't* match, we probably should
        # throw an exception. But if inductor is the only one calling this
        # thing, it should always match.
        # Get rid of constants before passing to cubin launcher

        # Add a None if triton wants an extra parameter to the cubin
        if self.has_global_scratch:
            arg_tys = self.arg_tys + "O"
            args = (*args, None)
        else:
            arg_tys = self.arg_tys
        assert len(args) == len(arg_tys)

        # TODO: can handle grid functions here or in C++, so
        # that we don't need the grid handler above.
        _StaticCudaLauncher._launch_kernel(
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
