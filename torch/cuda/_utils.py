import ctypes
import os
import sys
from typing import Any

import torch

# The _get_device_index has been moved to torch.utils._get_device_index
from torch._utils import _get_device_index as _torch_get_device_index


def _get_nvrtc_version(cuda_version: int) -> str:
    # Follows same logic as LazyNVRTC.cpp getLibVersion()
    major = cuda_version // 1000
    minor = (cuda_version // 10) % 10

    if sys.platform == "win32":
        if major < 11 or (major == 11 and minor < 3):
            return f"{major}{minor}"
        elif major == 11:
            return "112"
        else:
            return f"{major}0"
    else:
        if major < 11 or (major == 11 and minor < 3):
            return f"{major}.{minor}"
        elif major == 11:
            return "11.2"
        else:
            return str(major)


# Load CUDA driver and NVRTC
def _get_cuda_library():
    if sys.platform == "win32":
        return ctypes.CDLL("nvcuda.dll")
    else:  # Unix-based systems
        return ctypes.CDLL("libcuda.so.1")


def _get_nvrtc_library():
    # Get NVRTC version based on CUDA runtime version
    from torch.cuda import cudart

    cuda_runtime_version = cudart().getVersion()
    version = _get_nvrtc_version(cuda_runtime_version)

    if sys.platform == "win32":
        # Windows: nvrtc64_XY_0.dll or nvrtc64_X0_0.dll
        lib_name = f"nvrtc64_{version}_0.dll"
        return ctypes.CDLL(lib_name)
    else:
        # Unix-based systems
        # Linux: libnvrtc.so.X.Y or libnvrtc.so.X
        lib_paths = [
            f"libnvrtc.so.{version}",
            os.path.join(
                os.environ.get("CUDA_HOME", ""), f"lib64/libnvrtc.so.{version}"
            ),
            "/usr/local/cuda/lib64/libnvrtc.so",
        ]

        for path in lib_paths:
            try:
                return ctypes.CDLL(path)
            except OSError:
                continue

        raise RuntimeError(
            "Could not find libnvrtc.so. Please make sure CUDA is installed."
        )


def _nvrtc_compile(
    kernel_source: str,
    kernel_name: str,
    compute_capability: str = None,
    header_code: str = "",
    cuda_include_dirs: list = None,
    nvcc_options: list = None,
):
    """
    Compiles a CUDA kernel using NVRTC and returns the PTX code.

    Args:
        kernel_source (str): The CUDA kernel source code as a string
        kernel_name (str): The name of the kernel function to compile
        compute_capability (str, optional): The compute capability to target (e.g., "86").
                                           If None, will detect from current device.
        header_code (str, optional): Additional header code to prepend to the kernel source
        cuda_include_dirs (list, optional): List of directories containing CUDA headers
        nvcc_options (list, optional): Additional options to pass to NVRTC

    Returns:
        str: The compiled PTX code
    """
    # Ensure CUDA is initialized
    from torch.cuda import _lazy_init

    _lazy_init()
    from torch.cuda._utils import _get_nvrtc_library

    # Load NVRTC library
    libnvrtc = _get_nvrtc_library()

    # NVRTC constants
    NVRTC_SUCCESS = 0

    # Helper: check NVRTC errors
    def check_nvrtc(result):
        if result != NVRTC_SUCCESS:
            err_str = ctypes.c_char_p()
            libnvrtc.nvrtcGetErrorString(result, ctypes.byref(err_str))
            raise RuntimeError(f"NVRTC error: {err_str.value.decode()}")

    # Add 'extern "C"' if not already present to ensure C linkage
    if not kernel_source.strip().startswith('extern "C"'):
        kernel_source = f'extern "C" {kernel_source}'

    # Combine header code and kernel source
    if header_code:
        full_source = header_code + "\n" + kernel_source
    else:
        full_source = kernel_source

    # Convert source to bytes
    source_bytes = full_source.encode("utf-8")

    # Get compute capability if not provided
    if compute_capability is None:
        props = get_device_properties(current_device())
        compute_capability = f"{props.major}{props.minor}"

    # Prepare compilation options
    options = []
    options.append(f"--gpu-architecture=sm_{compute_capability}".encode())

    # Add custom include directories
    if cuda_include_dirs:
        for directory in cuda_include_dirs:
            options.append(f"-I{directory}".encode())

    # Add custom NVCC options
    if nvcc_options:
        for option in nvcc_options:
            options.append(option.encode("utf-8"))

            # TODO: Should we refactor flags into a common place?
            from torch.utils.cpp_extension import COMMON_NVCC_FLAGS

            options.extend(COMMON_NVCC_FLAGS)

    # Convert options to C array
    num_options = len(options)
    options_array = (ctypes.c_char_p * num_options)(*options)

    # Create program
    prog = ctypes.c_void_p()
    check_nvrtc(
        libnvrtc.nvrtcCreateProgram(
            ctypes.byref(prog),
            source_bytes,
            f"{kernel_name}.cu".encode(),
            0,
            None,
            None,
        )
    )

    # Compile program
    res = libnvrtc.nvrtcCompileProgram(prog, num_options, options_array)

    # Handle compilation errors
    if res != NVRTC_SUCCESS:
        # Get log
        log_size = ctypes.c_size_t()
        libnvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(log_size))
        log = ctypes.create_string_buffer(log_size.value)
        libnvrtc.nvrtcGetProgramLog(prog, log)
        raise RuntimeError(f"Kernel compilation failed:\n{log.value.decode()}")

    # Get PTX
    ptx_size = ctypes.c_size_t()
    check_nvrtc(libnvrtc.nvrtcGetPTXSize(prog, ctypes.byref(ptx_size)))
    ptx = ctypes.create_string_buffer(ptx_size.value)
    check_nvrtc(libnvrtc.nvrtcGetPTX(prog, ptx))
    libnvrtc.nvrtcDestroyProgram(ctypes.byref(prog))

    return ptx.value


class _CudaKernel:
    """
    Represents a compiled CUDA kernel that can be called with PyTorch tensors.
    """

    def __init__(self, func, module):
        self.func = func
        self.module = module

    def __call__(
        self, grid=(1, 1, 1), block=(1, 1, 1), args=None, shared_mem=0, stream=None
    ):
        """
        Call the compiled CUDA kernel

        Args:
            grid (tuple): Grid dimensions (grid_x, grid_y, grid_z)
            block (tuple): Block dimensions (block_x, block_y, block_z)
            args (list): List of arguments to pass to the kernel.
                         PyTorch tensor arguments will be automatically converted to pointers.
            shared_mem (int): Shared memory size in bytes
            stream (torch.cuda.Stream): CUDA stream to use. If None, uses current stream.
        """
        from torch.cuda._utils import _get_cuda_library

        libcuda = _get_cuda_library()
        CUDA_SUCCESS = 0

        # Helper: check CUDA errors
        def check_cuda(result):
            if result != CUDA_SUCCESS:
                err_str = ctypes.c_char_p()
                libcuda.cuGetErrorString(result, ctypes.byref(err_str))
                raise RuntimeError(f"CUDA error: {err_str.value.decode()}")

        if not args:
            args = []

        # Process arguments and convert tensors to pointers
        processed_args = []
        c_args = []

        for arg in args:
            if isinstance(arg, torch.Tensor):
                if not arg.is_cuda:
                    raise ValueError("All tensor arguments must be CUDA tensors")
                # Get pointer to tensor data
                ptr = ctypes.c_void_p(arg.data_ptr())
                processed_args.append(ptr)
                c_args.append(ctypes.byref(ptr))
            elif isinstance(arg, int):
                # Convert integers to C int
                c_int = ctypes.c_int(arg)
                processed_args.append(c_int)
                c_args.append(ctypes.byref(c_int))
            elif isinstance(arg, float):
                # Convert floats to C float
                c_float = ctypes.c_float(arg)
                processed_args.append(c_float)
                c_args.append(ctypes.byref(c_float))
            else:
                raise TypeError(f"Unsupported argument type: {type(arg)}")

        # Convert to array of void pointers
        c_args_array = (ctypes.c_void_p * len(c_args))()
        for i, arg in enumerate(c_args):
            c_args_array[i] = ctypes.cast(arg, ctypes.c_void_p)

        # Get the stream
        if stream is None:
            from torch.cuda import current_stream

            stream = current_stream()

        # Launch the kernel with the current stream
        with stream:
            check_cuda(
                libcuda.cuLaunchKernel(
                    self.func,
                    grid[0],
                    grid[1],
                    grid[2],
                    block[0],
                    block[1],
                    block[2],
                    shared_mem,
                    None,
                    c_args_array,
                    None,
                )
            )


def _cuda_load_module(ptx, kernel_names=None):
    """
    Loads a CUDA module from PTX code and returns a module object that can access kernels.

    Args:
        ptx (bytes or str): The PTX code to load
        kernel_names (list, optional): List of kernel names to extract from the module.
                                      If None, will return a module object with __getattr__.

    Returns:
        object: If kernel_names is None, returns a module object with __getattr__ to access kernels.
               If kernel_names is provided, returns a dict mapping kernel names to _CudaKernel objects.
    """
    # Ensure CUDA is initialized
    from torch.cuda import _lazy_init

    _lazy_init()
    from torch.cuda._utils import _get_cuda_library

    # Load CUDA driver library
    libcuda = _get_cuda_library()

    # CUDA constants
    CUDA_SUCCESS = 0

    # Helper: check CUDA errors
    def check_cuda(result):
        if result != CUDA_SUCCESS:
            err_str = ctypes.c_char_p()
            libcuda.cuGetErrorString(result, ctypes.byref(err_str))
            raise RuntimeError(f"CUDA error: {err_str.value.decode()}")

    # Convert PTX to bytes if it's a string
    if isinstance(ptx, str):
        ptx = ptx.encode("utf-8")

    # Load PTX module
    module = ctypes.c_void_p()
    from torch.cuda import current_stream

    with current_stream():
        check_cuda(libcuda.cuModuleLoadData(ctypes.byref(module), ptx))

    if kernel_names:
        # Return specific kernels
        kernels = {}
        for name in kernel_names:
            func = ctypes.c_void_p()
            check_cuda(
                libcuda.cuModuleGetFunction(
                    ctypes.byref(func), module, name.encode("utf-8")
                )
            )
            kernels[name] = _CudaKernel(func, module)
        return kernels
    else:
        # Create a module-like object with __getattr__
        class CudaModule:
            def __init__(self, module):
                self._module = module
                self._kernels = {}

            def __getattr__(self, name):
                if name in self._kernels:
                    return self._kernels[name]

                func = ctypes.c_void_p()
                try:
                    check_cuda(
                        libcuda.cuModuleGetFunction(
                            ctypes.byref(func), self._module, name.encode("utf-8")
                        )
                    )
                    kernel = _CudaKernel(func, self._module)
                    self._kernels[name] = kernel
                    return kernel
                except RuntimeError:
                    raise AttributeError(f"No kernel named '{name}' in this module")

        return CudaModule(module)


def _get_device_index(
    device: Any, optional: bool = False, allow_cpu: bool = False
) -> int:
    r"""Get the device index from :attr:`device`, which can be a torch.device object, a Python integer, or ``None``.

    If :attr:`device` is a torch.device object, returns the device index if it
    is a CUDA device. Note that for a CUDA device without a specified index,
    i.e., ``torch.device('cuda')``, this will return the current default CUDA
    device if :attr:`optional` is ``True``. If :attr:`allow_cpu` is ``True``,
    CPU devices will be accepted and ``-1`` will be returned in this case.

    If :attr:`device` is a Python integer, it is returned as is.

    If :attr:`device` is ``None``, this will return the current default CUDA
    device if :attr:`optional` is ``True``.
    """
    if isinstance(device, int):
        return device
    if isinstance(device, str):
        device = torch.device(device)
    if isinstance(device, torch.device):
        if allow_cpu:
            if device.type not in ["cuda", "cpu"]:
                raise ValueError(f"Expected a cuda or cpu device, but got: {device}")
        elif device.type != "cuda":
            raise ValueError(f"Expected a cuda device, but got: {device}")
    if not torch.jit.is_scripting():
        if isinstance(device, torch.cuda.device):
            return device.idx
    return _torch_get_device_index(device, optional, allow_cpu)
