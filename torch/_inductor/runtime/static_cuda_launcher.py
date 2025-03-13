import functools
from typing import Any, Optional
from typing_extensions import Unpack

from torch.utils._ordered_set import OrderedSet

from .triton_compat import ASTSource, CompiledKernel


MAX_SHARED_MEMORY = 49152


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
    """

    def __init__(self, kernel: CompiledKernel) -> None:
        # To be used later when hooking up with torch.compile:
        # inductor knows where the cubin file should be from triton,
        # so won't need to write to a tmp file directly.
        if hasattr(kernel, "_cubin_path"):
            self.cubin_path = kernel._cubin_path
        else:
            self.cubin = kernel.asm["cubin"]

        # TODO: is this right?
        self.name = kernel.src.fn.__name__
        self.hash = kernel.hash
        if (
            kernel.__class__.launch_enter_hook is not None
            or kernel.__class__.launch_exit_hook is not None
        ):
            raise NotImplementedError(
                "We don't support launch enter or launch exit hooks"
            )
        self.num_warps = kernel.metadata.num_warps
        self.shared = (
            kernel.shared if hasattr(kernel, "shared") else kernel.metadata.shared
        )
        # When shared memory > 48 KB, triton allocates CUDA memory via both static and dynamic
        # memory allocation, which gets really complicated. We'll handle it later.
        # See triton/third-party/nvidia/driver.c in loadBinary
        if self.shared > MAX_SHARED_MEMORY:
            raise NotImplementedError(
                "Shared memory size > 48KB requires special triton handling"
            )

        # Newer triton versions pass an extra global scratch parameter to the compiled cuda kernel.
        # Inductor never uses this field or enables it, but we still have to pass an extra None
        # into the set of params if its enabled
        if hasattr(kernel.metadata, "global_scratch_size"):
            if kernel.metadata.global_scratch_size > 0:
                raise NotImplementedError("Global scratch not yet supported")
            else:
                self.has_global_scratch = True
        else:
            self.has_global_scratch = False

        self.arg_tys, self.constant_idxs = self.arg_ty_from_signature(kernel.src)
        self.function: Optional[int] = (
            None  # Loaded by load_kernel(on the parent process)
        )
        num_args = len(self.arg_tys)
        num_ctas = 1
        if hasattr(kernel, "num_ctas"):
            num_ctas = kernel.num_ctas
        elif hasattr(kernel, "metadata"):
            num_ctas = kernel.metadata.num_ctas

        if num_ctas != 1:
            raise NotImplementedError(
                "Static cuda launcher only supports num_ctas == 1"
            )

        if num_args > 25 or num_args == 0:
            raise NotImplementedError(
                "No static cuda launcher available for %d arguments", num_args
            )

    def load_kernel(self) -> None:
        from torch._C import _StaticCudaLauncher

        assert hasattr(self, "cubin_path")
        if self.function is not None:
            return
        (self.function, self.n_regs, self.n_spills) = _StaticCudaLauncher._load_kernel(
            self.cubin_path, self.name, self.shared
        )

    def write_cubin_to_file(self, filepath: str) -> None:
        """
        Only used for tests where we don't have a cubin path.
        """
        if hasattr(self, "cubin_path"):
            return
        # Just used by tests for now.
        # TODO: derive cubin_path from wherever triton stores the cubin file on disk.
        with open(filepath, "wb") as f:
            f.write(self.cubin)
            del self.cubin
        self.cubin_path = filepath

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

    def arg_ty_from_signature(self, src: ASTSource) -> tuple[str, OrderedSet[int]]:
        def index_key(i: Any) -> int:
            return src.fn.arg_names.index(i) if isinstance(i, str) else i

        signature = {index_key(key): value for key, value in src.signature.items()}
        constants = [index_key(key) for key in getattr(src, "constants", dict())]
        # Despite requiring them to be passed in, the triton CUDA launcher
        # completely ignores the constexprs passed into it when generating code.
        # So we can ignore them here too
        params = []

        constant_idxs: OrderedSet[int] = OrderedSet()
        for i in sorted(signature.keys()):
            ty = signature[i]
            # In newer triton versions, constants are passed in to signature with type `constexpr`
            # In older triton versions, there can be constants in src.constants that are not `constexpr` in signature
            # so we check both here
            if ty == "constexpr" or i in constants:
                constant_idxs.add(i)
            else:
                params.append(self.extract_type(ty))
        return "".join(params), constant_idxs

    def run(
        self, grid: tuple[int, ...], stream: int, *args: Unpack[tuple[object, ...]]
    ) -> None:
        """Actually run the kernel at runtime. This function is the hot codepath."""
        from torch._C import _StaticCudaLauncher

        # Assert load_kernel() has been called and args match
        assert self.function is not None

        # TODO: actually, if the args *don't* match, we probably should
        # throw an exception. But if inductor is the only one calling this
        # thing, it should always match.
        # Get rid of constants before passing to cubin launcher

        # TODO: is this (and the check below) slow to do at runtime? The thing is,
        # we already spend the time in CachingAutotuner.launch() to massage the arguments
        # properly anyways so this isn't exactly slower than that...
        args = tuple(args[i] for i in range(len(args)) if i not in self.constant_idxs)

        # Add a None if triton wants an extra parameter to the cubin
        if self.has_global_scratch:
            arg_tys = self.arg_tys + "O"
            args = (*args, None)
        else:
            arg_tys = self.arg_tys

        assert len(args) == len(arg_tys)

        # TODO: can handle grid functions here or in C++, so
        # that we don't need the grid handler above.
        grid_x = grid[0]
        grid_y = grid[1] if len(grid) > 1 else 1
        grid_z = grid[2] if len(grid) > 2 else 1
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
