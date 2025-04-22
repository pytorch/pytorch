from typing import Optional, List, Tuple
import os
import sys
import ctypes
import torch

def get_nvrtc_and_cuda_libs() -> Tuple[ctypes.CDLL, ctypes.CDLL]:
    """
    Get the NVRTC and CUDA libraries for runtime compilation.
    
    Returns:
        Tuple[ctypes.CDLL, ctypes.CDLL]: A tuple containing (libnvrtc, libcuda)
        
    Raises:
        RuntimeError: If the libraries cannot be found or loaded
    """
    if sys.platform == 'win32':
        from torch.cuda import cudart
        libnvrtc_name = 'nvrtc64_%d_0.dll' % (cudart().cudaRuntimeGetVersion() // 1000)
        try:
            libnvrtc = ctypes.CDLL(libnvrtc_name)
            libcuda = ctypes.CDLL('nvcuda.dll')
        except OSError as e:
            raise RuntimeError(f"Could not load NVRTC or CUDA libraries on Windows: {e}")
    else:  # Unix-based systems
        libnvrtc_paths = [
            'libnvrtc.so',
            os.path.join(os.environ.get('CUDA_HOME', ''), 'lib64/libnvrtc.so'),
            '/usr/local/cuda/lib64/libnvrtc.so',
        ]
        
        libnvrtc = None
        for path in libnvrtc_paths:
            try:
                libnvrtc = ctypes.CDLL(path)
                break
            except OSError:
                continue
                
        if libnvrtc is None:
            raise RuntimeError("Could not find libnvrtc.so. Please make sure CUDA is installed.")
            
        try:
            libcuda = ctypes.CDLL('libcuda.so')
        except OSError as e:
            raise RuntimeError(f"Could not load CUDA driver library: {e}")
    
    return libnvrtc, libcuda

def check_nvrtc_error(result: int, libnvrtc: ctypes.CDLL) -> None:
    """
    Check NVRTC error code and raise an exception if there was an error.
    
    Args:
        result (int): The NVRTC error code to check
        libnvrtc (ctypes.CDLL): The loaded NVRTC library
        
    Raises:
        RuntimeError: If the error code indicates an error occurred
    """
    NVRTC_SUCCESS = 0
    if result != NVRTC_SUCCESS:
        err_str = ctypes.c_char_p()
        libnvrtc.nvrtcGetErrorString(result, ctypes.byref(err_str))
        raise RuntimeError(f'NVRTC error: {err_str.value.decode()}')

def check_cuda_error(result: int, libcuda: ctypes.CDLL) -> None:
    """
    Check CUDA driver error code and raise an exception if there was an error.
    
    Args:
        result (int): The CUDA error code to check
        libcuda (ctypes.CDLL): The loaded CUDA library
        
    Raises:
        RuntimeError: If the error code indicates an error occurred
    """
    CUDA_SUCCESS = 0
    if result != CUDA_SUCCESS:
        err_str = ctypes.c_char_p()
        libcuda.cuGetErrorString(result, ctypes.byref(err_str))
        raise RuntimeError(f'CUDA error: {err_str.value.decode()}')

def compile_cuda_source(
    source: str,
    kernel_name: str,
    compute_capability: Optional[str] = None,
    header_code: str = "",
    cuda_include_dirs: Optional[List[str]] = None,
    nvcc_options: Optional[List[str]] = None
) -> bytes:
    """
    Compiles CUDA source code to PTX using NVRTC.
    
    Args:
        source (str): The CUDA kernel source code
        kernel_name (str): The name of the kernel function
        compute_capability (str, optional): The compute capability to target (e.g., "86").
                                           If None, will detect from current device.
        header_code (str, optional): Additional header code to prepend to the kernel source
        cuda_include_dirs (list, optional): List of directories containing CUDA headers
        nvcc_options (list, optional): Additional options to pass to NVRTC
        
    Returns:
        bytes: The compiled PTX code
        
    Raises:
        RuntimeError: If compilation fails or required libraries cannot be loaded
    """
    # Ensure CUDA is initialized
    torch.cuda._lazy_init()
    
    # Load NVRTC and CUDA libraries
    libnvrtc, libcuda = get_nvrtc_and_cuda_libs()
    
    # Add 'extern "C"' if not already present to ensure C linkage
    if 'extern "C"' not in source:
        source = f'extern "C" {source}'
    
    # Combine header code and kernel source
    if header_code:
        full_source = header_code + "\n" + source
    else:
        full_source = source
    
    # Convert source to bytes
    source_bytes = full_source.encode('utf-8')
    
    # Get compute capability if not provided
    if compute_capability is None:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        compute_capability = f"{props.major}{props.minor}"
    
    # Prepare compilation options
    options = []
    options.append(f"--gpu-architecture=sm_{compute_capability}".encode('utf-8'))
    
    # Add custom include directories
    if cuda_include_dirs:
        for directory in cuda_include_dirs:
            options.append(f"-I{directory}".encode('utf-8'))
    
    # Add custom NVCC options
    if nvcc_options:
        for option in nvcc_options:
            options.append(option.encode('utf-8'))
    
    # Convert options to C array
    num_options = len(options)
    options_array = (ctypes.c_char_p * num_options)(*options)
    
    # Create program
    prog = ctypes.c_void_p()
    check_nvrtc_error(libnvrtc.nvrtcCreateProgram(
        ctypes.byref(prog),
        source_bytes,
        f"{kernel_name}.cu".encode('utf-8'),
        0, None, None
    ), libnvrtc)
    
    # Compile program
    res = libnvrtc.nvrtcCompileProgram(prog, num_options, options_array)
    
    # Handle compilation errors
    if res != 0:  # NVRTC_SUCCESS = 0
        # Get log
        log_size = ctypes.c_size_t()
        libnvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(log_size))
        log = ctypes.create_string_buffer(log_size.value)
        libnvrtc.nvrtcGetProgramLog(prog, log)
        raise RuntimeError(f"Kernel compilation failed:\n{log.value.decode()}")
    
    # Get PTX
    ptx_size = ctypes.c_size_t()
    check_nvrtc_error(libnvrtc.nvrtcGetPTXSize(prog, ctypes.byref(ptx_size)), libnvrtc)
    ptx = ctypes.create_string_buffer(ptx_size.value)
    check_nvrtc_error(libnvrtc.nvrtcGetPTX(prog, ptx), libnvrtc)
    libnvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
    
    return ptx.raw

class CudaKernel:
    """
    A callable wrapper for a CUDA kernel function.
    
    This class wraps a CUDA kernel function and provides a Python interface to call it
    with PyTorch tensor arguments.
    """
    
    def __init__(self, func: ctypes.c_void_p, libcuda: ctypes.CDLL):
        """
        Initialize the CUDA kernel.
        
        Args:
            func (ctypes.c_void_p): Pointer to the CUDA kernel function
            libcuda (ctypes.CDLL): The loaded CUDA library
        """
        self.func = func
        self.libcuda = libcuda
    
    def __call__(
        self,
        grid: Tuple[int, int, int] = (1, 1, 1),
        block: Tuple[int, int, int] = (1, 1, 1),
        args = None,
        shared_mem: int = 0,
        stream = None
    ) -> None:
        """
        Call the compiled CUDA kernel.
        
        Args:
            grid (tuple): Grid dimensions (grid_x, grid_y, grid_z)
            block (tuple): Block dimensions (block_x, block_y, block_z)
            args (list): List of arguments to pass to the kernel. 
                         PyTorch tensor arguments will be automatically converted to pointers.
            shared_mem (int): Shared memory size in bytes
            stream (torch.cuda.Stream): CUDA stream to use. If None, uses current stream.
        """
        if args is None:
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
            stream = torch.cuda.current_stream()
        
        # Launch the kernel with the current stream
        with stream:
            check_cuda_error(self.libcuda.cuLaunchKernel(
                self.func,
                grid[0], grid[1], grid[2],
                block[0], block[1], block[2],
                shared_mem, None,
                c_args_array, None
            ), self.libcuda)

def load_cuda_module(ptx: bytes, kernel_name: str) -> CudaKernel:
    """
    Loads a PTX module and returns a callable kernel function.
    
    Args:
        ptx (bytes): The compiled PTX code
        kernel_name (str): Name of the kernel function to load
        
    Returns:
        CudaKernel: A callable wrapper for the CUDA kernel
        
    Raises:
        RuntimeError: If module loading fails or the function cannot be found
    """
    # Load CUDA library
    _, libcuda = get_nvrtc_and_cuda_libs()
    
    # Convert kernel name to bytes
    kernel_name_bytes = kernel_name.encode('utf-8')
    
    # Load PTX module
    module = ctypes.c_void_p()
    with torch.cuda.current_stream():
        check_cuda_error(libcuda.cuModuleLoadData(ctypes.byref(module), ptx), libcuda)
    
    # Get function from module
    func = ctypes.c_void_p()
    check_cuda_error(libcuda.cuModuleGetFunction(
        ctypes.byref(func), module, kernel_name_bytes
    ), libcuda)
    
    # Create and return the callable kernel wrapper
    return CudaKernel(func, libcuda)