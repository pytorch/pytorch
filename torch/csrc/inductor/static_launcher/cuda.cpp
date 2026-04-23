#if defined(USE_CUDA)

#include <ATen/Context.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/inductor/static_launcher/cuda.h>
#include <cstdint>

#include <torch/csrc/utils/python_numbers.h>
#include <filesystem>
#include <optional>

#if defined(USE_ROCM)
#include <hip/hip_runtime_api.h>
#endif

/**
  Implements a static launcher for triton compiled CUDA kernels.
  Given a path to a cubin file, a function name, and some metadata,
  this class loads and launches the cubin.

  Doing this avoids C++ codegen and compilation during compile, since we can
  use a statically compiled library to launch the kernel. To avoid mallocing
  for the arguments, we have a launcher for different numbers of arguments up
  to a max. StaticCudaLauncher only supports # of arguments up until 10 for
  now.

  Note that we allocate 8 bytes per argument, no matter the types of each
  argument, since we don't know ahead of time what the types of each argument
  passed to the triton kernel are. This may take slightly more memory on the
  stack, and will require some benchmarking. However, since the vast majority
  of triton kernels have less than 10 args, this seems unlikely to be
  expensive.

  This launcher is paired with StaticallyLaunchedCudaKernel in
  triton_heuristics.py.

  TODO:
  - Handle CutensorMap, NvtmDesc
  - Handle launch_enter and launch_exit hooks (in python maybe?)
 */

// Use ATen/NVRTC.h to gain access to the CUDA driver API.
// This function is only called when CUDA is enabled, and only called to load
// and launch triton compiled CUDA kernels, so CUDA should always be
// initialized.
namespace {
const at::cuda::NVRTC& nvrtc() {
  return at::globalContext().getNVRTC();
}

// 120 max args + 1 for global scratch size
#define MAX_ARGS 121

CUdeviceptr getPointer(PyObject* obj) {
  CUdeviceptr data_ptr = 0;

  if (THPUtils_checkLong(obj)) {
#if defined(USE_ROCM)
    data_ptr = reinterpret_cast<hipDeviceptr_t>(THPUtils_unpackUInt64(obj));
#else
    data_ptr = THPUtils_unpackUInt64(obj);
#endif

    return data_ptr;
  }
  if (Py_IsNone(obj)) {
    // valid nullptr
    return data_ptr;
  }
  auto ptr = THPObjectPtr{PyObject_GetAttrString(obj, "data_ptr")};
  TORCH_CHECK(
      ptr != nullptr,
      "Pointer argument must be either uint64 or have data_ptr method")
  auto empty_tuple = THPObjectPtr{PyTuple_New(0)};
  auto ret = THPObjectPtr{PyObject_Call(ptr, empty_tuple, nullptr)};
  TORCH_CHECK(
      THPUtils_checkLong(ret),
      "data_ptr method of Pointer object must return 64-bit int");

#if defined(USE_ROCM)
  data_ptr = reinterpret_cast<hipDeviceptr_t>(THPUtils_unpackUInt64(ret));
#else
  data_ptr = THPUtils_unpackUInt64(ret);
#endif

  if (!data_ptr)
    return data_ptr;

  CUdeviceptr dev_ptr = 0;
#if defined(USE_ROCM)
  AT_CUDA_DRIVER_CHECK(hipPointerGetAttribute(
      &dev_ptr, HIP_POINTER_ATTRIBUTE_DEVICE_POINTER, data_ptr));
#else
  AT_CUDA_DRIVER_CHECK(nvrtc().cuPointerGetAttribute(
      &dev_ptr, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, data_ptr));
#endif

  return dev_ptr;
}

#define SHARED_MEM_STATIC_MAX 49152 // 48 KB

CUfunction loadKernel(
    std::string filePath,
    const std::string& funcName,
    uint32_t sharedMemBytes,
    CUdevice device,
    const std::optional<std::string>& cubinDir = std::nullopt) {
  if (cubinDir) {
    std::filesystem::path p1{*cubinDir};
    std::filesystem::path p2{filePath};
    filePath = (p1 / p2.filename()).string();
  }
  CUmodule mod = nullptr;
  CUfunction func = nullptr;

#if defined(USE_ROCM)
  AT_CUDA_DRIVER_CHECK(hipModuleLoad(&mod, filePath.c_str()));
  AT_CUDA_DRIVER_CHECK(hipModuleGetFunction(&func, mod, funcName.c_str()));
  int shared_optin = 0;
  AT_CUDA_DRIVER_CHECK(hipDeviceGetAttribute(
      &shared_optin, hipDeviceAttributeMaxSharedMemoryPerBlock, device));

#else
  AT_CUDA_DRIVER_CHECK(nvrtc().cuModuleLoad(&mod, filePath.c_str()));
  AT_CUDA_DRIVER_CHECK(
      nvrtc().cuModuleGetFunction(&func, mod, funcName.c_str()));
  int shared_optin = 0;
  AT_CUDA_DRIVER_CHECK(nvrtc().cuDeviceGetAttribute(
      &shared_optin,
      CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
      device));

#endif

  // Shared memory logic from triton/third-party/nvidia/backend/driver.c
  // If we're using more than 48 KB of shared memory, and we have
  // access to more than 48 KB of shared memory on the device,
  // we set maximum dynamic shared memory to the difference between
  // the static shared memory and total max shared memory allowed on the device.
  // This prevents us from setting shared memory above the maximum

  // TODO: Unify the CUDA and ROCm shared memory checks. Currently using <= for
  // ROCm and < for CUDA because ROCm hits the boundary case more often.
#if defined(USE_ROCM)
  TORCH_CHECK_WITH(
      OutOfMemoryError,
      sharedMemBytes <= static_cast<uint32_t>(shared_optin),
      "out of resource: ",
      funcName,
      " Required: ",
      sharedMemBytes,
      " Hardware limit:",
      shared_optin,
      " Reducing block sizes or `num_stages` may help.");
#else
  TORCH_CHECK_WITH(
      OutOfMemoryError,
      sharedMemBytes < static_cast<uint32_t>(shared_optin),
      "out of resource: ",
      funcName,
      " Required: ",
      sharedMemBytes,
      " Hardware limit:",
      shared_optin,
      " Reducing block sizes or `num_stages` may help.");
#endif

  if (sharedMemBytes > SHARED_MEM_STATIC_MAX &&
      shared_optin > SHARED_MEM_STATIC_MAX) {
#if defined(USE_ROCM)
    AT_CUDA_DRIVER_CHECK(hipFuncSetCacheConfig(func, hipFuncCachePreferShared));
    int shared_total = 0, shared_static = 0;
    AT_CUDA_DRIVER_CHECK(hipDeviceGetAttribute(
        &shared_total,
        hipDeviceAttributeMaxSharedMemoryPerMultiprocessor,
        device));
    AT_CUDA_DRIVER_CHECK(hipFuncGetAttribute(
        &shared_static, HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func));
    AT_CUDA_DRIVER_CHECK(hipFuncSetAttribute(
        func,
        hipFuncAttributeMaxDynamicSharedMemorySize,
        shared_optin - shared_static));

#else
    AT_CUDA_DRIVER_CHECK(
        nvrtc().cuFuncSetCacheConfig(func, CU_FUNC_CACHE_PREFER_SHARED));
    int shared_total = 0, shared_static = 0;
    AT_CUDA_DRIVER_CHECK(nvrtc().cuDeviceGetAttribute(
        &shared_total,
        CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
        device));
    AT_CUDA_DRIVER_CHECK(nvrtc().cuFuncGetAttribute(
        &shared_static, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func));
    AT_CUDA_DRIVER_CHECK(nvrtc().cuFuncSetAttribute(
        func,
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        shared_optin - shared_static));
#endif
  }
  return func;
}

inline void launchKernel(
    CUfunction func,
    uint32_t gridX,
    uint32_t gridY,
    uint32_t gridZ,
    uint32_t numWarps,
    uint32_t sharedMemBytes,
    void** args,
    cudaStream_t stream) {
  // cta_args is always 1 for inductor generated triton kernels,
  // so we don't need to figure out grid dimension here
#if defined(USE_ROCM)
  int device = 0;
  AT_CUDA_DRIVER_CHECK(hipGetDevice(&device));
  int warp_size = 0;
  AT_CUDA_DRIVER_CHECK(
      hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, device));

  AT_CUDA_DRIVER_CHECK(hipModuleLaunchKernel(
      func,
      gridX,
      gridY,
      gridZ,
      warp_size * numWarps, // blockDim.x
      1, // blockDim.y
      1, // blockDim.z
      sharedMemBytes,
      stream,
      args,
      nullptr));

#else
  AT_CUDA_DRIVER_CHECK(nvrtc().cuLaunchKernel(
      func,
      gridX,
      gridY,
      gridZ,
      32 * numWarps, // blockDim.x
      1, // blockDim.y
      1, // blockDim.z
      sharedMemBytes,
      stream,
      args,
      nullptr));
#endif
}

template <typename FINAL, typename F>
void convertType(F converter, const char* name, void* slot, PyObject* item) {
  auto temp = converter(item);
  if (PyErr_Occurred()) {
    std::string msg = "Failed to convert argument to ";
    msg += name;
    TORCH_CHECK(false, msg);
  }
  *reinterpret_cast<FINAL*>(slot) = static_cast<FINAL>(temp);
}

/**
  Given a list of args and their types (in a string), along with two stack
  allocated arrays, puts each argument arg_{i} into argStorage[i], and a
  pointer to the argument in kernelArgs[i]. We then can pass `kernelArgs`
  directly to launchKernel. Note that some args can be less than 8 bytes, but
  we'll still allocate 8 bytes on the stack for them.

  * TODO: Need to handle NvtmDesc here.
*/
void parseKernelArgs(
    PyObject* varArgs,
    const char* argTypes,
    uint64_t* argStorage,
    void** kernelArgs) {
  int numKernelArgs = static_cast<int>(std::strlen(argTypes));
  TORCH_CHECK(
      PyTuple_Check(varArgs), "Kernel arguments must be provided as a tuple");
  TORCH_CHECK(
      PyTuple_Size(varArgs) == static_cast<Py_ssize_t>(numKernelArgs),
      "Mismatch between number of argument types and provided arguments");

  for (int i = 0; i < numKernelArgs; ++i) {
    // Get pointer to the ith 8-byte slot.
    void* slot = static_cast<void*>(&argStorage[i]);
    PyObject* item = PyTuple_GetItem(varArgs, i);
    char typeChar = argTypes[i];
    switch (typeChar) {
      case 'b':
        convertType<int8_t>(THPUtils_unpackInt, "int8", slot, item);
        break;
      case 'h':
        convertType<int16_t>(THPUtils_unpackInt, "int16", slot, item);
        break;
      case 'i':
        convertType<int32_t>(THPUtils_unpackLong, "int32", slot, item);
        break;
      case 'l':
        convertType<int64_t>(THPUtils_unpackLong, "int64", slot, item);
        break;
      case 'B':
        convertType<uint8_t>(THPUtils_unpackUInt32, "uint8", slot, item);
        break;
      case 'H':
        convertType<uint16_t>(THPUtils_unpackUInt32, "uint16", slot, item);
        break;
      case 'I':
        convertType<uint32_t>(THPUtils_unpackUInt32, "uint32", slot, item);
        break;
      case 'K':
        convertType<uint64_t>(THPUtils_unpackUInt64, "uint64", slot, item);
        break;
      case 'f':
        convertType<float>(THPUtils_unpackDouble, "float", slot, item);
        break;
      case 'd':
        convertType<double>(THPUtils_unpackDouble, "double", slot, item);
        break;
      case 'O': { // pointer; using helper getPointer() (which may call
                  // data_ptr() if needed)
        CUdeviceptr ptr = getPointer(item);
        *reinterpret_cast<CUdeviceptr*>(slot) = ptr;
        break;
      }
      default:
        TORCH_CHECK(false, "Unknown type passed in: ", typeChar);
    }
    // Save the pointer to this slot.
    kernelArgs[i] = slot;
  }
}

/* Load the CUDA kernel into memory (called during torch.compile), and
  return a pointer to it (along with nregs and nspills).
  Called in python as:
  (function, n_regs, n_spills) = load_kernel(cubin_path, func_name,
  sharedMemBytes)
*/
PyObject* load_kernel(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  const char* filePath = nullptr;
  const char* funcName = nullptr;
  int sharedMemBytes = 0;
  int n_regs = 0;
  int n_spills = 0;
  int device_ptr = 0;
  if (!PyArg_ParseTuple(
          args, "ssii", &filePath, &funcName, &sharedMemBytes, &device_ptr)) {
    return nullptr;
  }
  CUdevice device = static_cast<CUdevice>(device_ptr); // NOLINT

  // Ensure CUDA context is initialized before loading kernel
  CUcontext pctx = nullptr;

#if defined(USE_ROCM)
  AT_CUDA_DRIVER_CHECK(hipCtxGetCurrent(&pctx));
  if (!pctx) {
    AT_CUDA_DRIVER_CHECK(hipDevicePrimaryCtxRetain(&pctx, device));
    AT_CUDA_DRIVER_CHECK(hipCtxSetCurrent(pctx));
  }
#else
  AT_CUDA_DRIVER_CHECK(nvrtc().cuCtxGetCurrent(&pctx));
  if (!pctx) {
    AT_CUDA_DRIVER_CHECK(nvrtc().cuDevicePrimaryCtxRetain(&pctx, device));
    AT_CUDA_DRIVER_CHECK(nvrtc().cuCtxSetCurrent(pctx));
  }
#endif

  CUfunction func = nullptr;
  func = loadKernel(filePath, funcName, sharedMemBytes, device);

#if defined(USE_ROCM)
  AT_CUDA_DRIVER_CHECK(
      hipFuncGetAttribute(&n_regs, HIP_FUNC_ATTRIBUTE_NUM_REGS, func));
  AT_CUDA_DRIVER_CHECK(hipFuncGetAttribute(
      &n_spills, HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, func));

#else
  AT_CUDA_DRIVER_CHECK(
      nvrtc().cuFuncGetAttribute(&n_regs, CU_FUNC_ATTRIBUTE_NUM_REGS, func));
  AT_CUDA_DRIVER_CHECK(nvrtc().cuFuncGetAttribute(
      &n_spills, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, func));

#endif
  n_spills /= 4;
  // Return a tuple of CUFunction, n_regs, n_spills
  return Py_BuildValue(
      "(Kii)", reinterpret_cast<uint64_t>(func), n_regs, n_spills);
  END_HANDLE_TH_ERRORS
}

PyObject* launch_kernel_inner(
    CUfunction func,
    int gridX,
    int gridY,
    int gridZ,
    int numWarps,
    int sharedMemBytes,
    const char* argTypes,
    PyObject* varArgs,
    cudaStream_t cudaStream) {
  // Launch the kernel
  // Prepare the arguments for the kernel
  // We allocate 8 bytes per argument on the stack. We then allocate 8 more
  // bytes to point to each 8 byte slot in argStorage, and pass that array of
  // pointers to launchKernel.
  std::array<uint64_t, MAX_ARGS> argStorage = {};
  std::array<void*, MAX_ARGS> kernelArgs = {};
  parseKernelArgs(varArgs, argTypes, argStorage.data(), kernelArgs.data());
  launchKernel(
      func,
      gridX,
      gridY,
      gridZ,
      numWarps,
      sharedMemBytes,
      kernelArgs.data(),
      cudaStream);
  Py_RETURN_NONE;
}

PyObject* launch_kernel_slow(
    CUfunction func,
    int gridX,
    int gridY,
    int gridZ,
    int numWarps,
    int sharedMemBytes,
    const char* argTypes,
    PyObject* varArgs,
    cudaStream_t cudaStream) {
  /* For the slow case, allocate memory on the stack instead of the heap */
  size_t numArgs = std::strlen(argTypes);
  std::vector<uint64_t> argStorage(numArgs);
  std::vector<void*> kernelArgs(numArgs);

  parseKernelArgs(varArgs, argTypes, argStorage.data(), kernelArgs.data());

  launchKernel(
      func,
      gridX,
      gridY,
      gridZ,
      numWarps,
      sharedMemBytes,
      kernelArgs.data(),
      cudaStream);
  Py_RETURN_NONE;
}

/**
*  Main entrypoint function called at runtime; called like this in python land:
    launcher(
      function, # CUfunction returned by load_kernel()
      grid_x,
      grid_y,
      grid_z,
      num_warps,
      shared,
      arg_tys, # e.g. "bO" for (int8_t, uint64_t)
      args, # tuple of arguments passed to the kernel
      stream,
  )
*
*/
PyObject* launch_kernel(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  // Pointer to CUfunction generated by load_kernel()
  uint64_t func_ptr = 0;
  int gridX = 0, gridY = 0, gridZ = 0, numWarps = 0, sharedMemBytes = 0;
  // stream here should be the raw stream gotten from
  // device_interface.get_raw_stream()
  uint64_t stream = 0;
  const char* argTypes = nullptr;
  PyObject* varArgs = nullptr;
  // Parse the fixed arguments and the format string
  if (!PyArg_ParseTuple(
          args,
          "KiiiiisOK",
          &func_ptr,
          &gridX,
          &gridY,
          &gridZ,
          &numWarps,
          &sharedMemBytes,
          &argTypes,
          &varArgs,
          &stream)) {
    return nullptr;
  }
  if (gridX * gridY * gridZ <= 0) {
    // No need to do any work if we're outside of grid bounds
    Py_RETURN_NONE;
  }
  CUcontext pctx = nullptr;
#if defined(USE_ROCM)
  AT_CUDA_DRIVER_CHECK(hipCtxGetCurrent(&pctx));
#else
  AT_CUDA_DRIVER_CHECK(nvrtc().cuCtxGetCurrent(&pctx));
#endif

  if (!pctx) {
    // Ensure device context exists
    CUdevice device = 0;
#if defined(USE_ROCM)
    AT_CUDA_DRIVER_CHECK(hipDeviceGet(&device, 0));
    AT_CUDA_DRIVER_CHECK(hipDevicePrimaryCtxRetain(&pctx, device));
    AT_CUDA_DRIVER_CHECK(hipCtxSetCurrent(pctx));
#else
    AT_CUDA_DRIVER_CHECK(nvrtc().cuDeviceGet(&device, 0));
    AT_CUDA_DRIVER_CHECK(nvrtc().cuDevicePrimaryCtxRetain(&pctx, device));
    AT_CUDA_DRIVER_CHECK(nvrtc().cuCtxSetCurrent(pctx));

#endif
  }
  CUfunction func = reinterpret_cast<CUfunction>(func_ptr); // NOLINT
  cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(stream); // NOLINT
  auto num_args = std::strlen(argTypes);
  // Kernels with no arguments should just pass nullptr to cuLaunchKernel
  if (num_args == 0) {
    launchKernel(
        func,
        gridX,
        gridY,
        gridZ,
        numWarps,
        sharedMemBytes,
        nullptr,
        cudaStream);
    Py_RETURN_NONE;
  } else if (num_args <= MAX_ARGS) {
    return launch_kernel_inner(
        func,
        gridX,
        gridY,
        gridZ,
        numWarps,
        sharedMemBytes,
        argTypes,
        varArgs,
        cudaStream);
  } else {
    return launch_kernel_slow(
        func,
        gridX,
        gridY,
        gridZ,
        numWarps,
        sharedMemBytes,
        argTypes,
        varArgs,
        cudaStream);
  }
  END_HANDLE_TH_ERRORS
}

std::array<PyMethodDef, 2> StaticCudaLauncherMethods = {
    PyMethodDef{
        "_launch_kernel",
        launch_kernel,
        METH_VARARGS,
        "Statically launch triton compiled CUDA kernels"},
    PyMethodDef{
        "_load_kernel",
        load_kernel,
        METH_VARARGS,
        "Load CUDA kernel from cubin file"}};

// Define a minimal type for StaticCudaLauncher.
// We don't implement __new__ or __init__ because we're using it only as a
// container for static methods.
PyTypeObject StaticCudaLauncherType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "torch._C._StaticCudaLauncher", // tp_name
    sizeof(PyObject), // tp_basicsize
    0, // tp_itemsize
    nullptr, // tp_dealloc
    0, // tp_print (deprecated)
    nullptr, // tp_getattr
    nullptr, // tp_setattr
    nullptr, // tp_reserved
    nullptr, // tp_repr
    nullptr, // tp_as_number
    nullptr, // tp_as_sequence
    nullptr, // tp_as_mapping
    nullptr, // tp_hash
    nullptr, // tp_call
    nullptr, // tp_str
    nullptr, // tp_getattro
    nullptr, // tp_setattro
    nullptr, // tp_as_buffer
    Py_TPFLAGS_DEFAULT,
    "Statically defined launchers for triton compiled CUDA kernels", // tp_doc
    nullptr, // tp_traverse
    nullptr, // tp_clear
    nullptr, // tp_richcompare
    0, // tp_weaklistoffset
    nullptr, // tp_iter
    nullptr, // tp_iternext
    nullptr, // tp_methods
    nullptr, // tp_members
    nullptr, // tp_getset
    nullptr, // tp_base
    nullptr, // tp_dict (automatically allocated)
    nullptr, // tp_descr_get
    nullptr, // tp_descr_set
    0, // tp_dictoffset
    nullptr, // tp_init
    nullptr, // tp_alloc
    nullptr, // tp_new
};
// ---------------------------------------------------------------------------
// getPointerFast: extract device pointer WITHOUT cuPointerGetAttribute.
// Uses THPVariable_Unpack (direct C++ field access) + at::Tensor::data_ptr()
// instead of going through Python data_ptr() method.
// No THPVariable_Check or tensor.defined() guard: in the _FastCudaLauncher
// path all 'O'-typed args are guaranteed to be tensors by inductor codegen.
// ---------------------------------------------------------------------------
inline CUdeviceptr getPointerFast(PyObject* obj) {
  if (THPUtils_checkLong(obj)) {
#if defined(USE_ROCM)
    return reinterpret_cast<hipDeviceptr_t>(THPUtils_unpackUInt64(obj));
#else
    return THPUtils_unpackUInt64(obj);
#endif
  }
  if (obj == Py_None) {
    return 0;
  }
  // Fast type check: inductor-generated tensors are always exact THPVariable
  // (or Parameter). Use CheckExact — a single Py_TYPE pointer comparison —
  // instead of THPVariable_Check which also calls PyObject_IsInstance.
  if (C10_LIKELY(THPVariable_CheckExact(obj))) {
    auto tensor = THPVariable_Unpack(obj);
    TORCH_CHECK(
        tensor.defined(),
        "_FastCudaLauncher: received undefined tensor argument");
    return reinterpret_cast<CUdeviceptr>(tensor.data_ptr());
  }
  // Slow fallback for non-tensor objects with a data_ptr() method
  // (e.g. tensor-like wrappers / proxy objects).
  PyObject* data_ptr_method = PyObject_GetAttrString(obj, "data_ptr");
  TORCH_CHECK(
      data_ptr_method,
      "_FastCudaLauncher: expected tensor or object with data_ptr() method");
  PyObject* ret = PyObject_CallNoArgs(data_ptr_method);
  Py_DECREF(data_ptr_method);
  TORCH_CHECK(ret, "_FastCudaLauncher: data_ptr() call failed");
  auto raw = THPUtils_unpackUInt64(ret);
  Py_DECREF(ret);
  // C-style cast: static_cast on CUDA (uint64→unsigned long long),
  // reinterpret_cast on HIP (uint64→void*).
  return (CUdeviceptr)raw;
}

// ---------------------------------------------------------------------------
// _FastCudaLauncher: pre-bound callable that uses vectorcall (PEP 590)
// to launch a triton kernel with minimal overhead.
//
// Pre-binds: CUfunction, numWarps, sharedMemBytes, argTypes, scratch slots.
// Per-call:  only grid, stream, and kernel args are passed.
// Skips:     cuCtxGetCurrent, cuPointerGetAttribute, PyArg_ParseTuple for
//            kernel metadata.
// ---------------------------------------------------------------------------
struct FastCudaLauncherObject {
  PyObject_HEAD
  vectorcallfunc vectorcall;
  CUfunction func;
  uint32_t numWarps;
  uint32_t sharedMemBytes;
  int numKernelArgs; // args passed from Python
  int numTotalArgs; // numKernelArgs + nScratch
  char argTypes[MAX_ARGS + 1]; // null-terminated
  // Thread safety: argStorage/kernelArgs are shared across calls but safe
  // because the GIL is held throughout fast_launcher_vectorcall (no
  // Py_BEGIN_ALLOW_THREADS).  cuLaunchKernel copies arg values from the
  // kernelArgs pointers synchronously before returning, so by the time the
  // GIL could be released the driver already has its own copy.
  // TODO(T000000): Not safe under free-threaded Python (PEP 703, nogil).
  // If two threads call the same instance concurrently without the GIL,
  // they will corrupt argStorage/kernelArgs.  Revisit when nogil is stable.
  uint64_t argStorage[MAX_ARGS];
  void* kernelArgs[MAX_ARGS];
};

static PyObject* fast_launcher_vectorcall(
    PyObject* callable,
    PyObject* const* args,
    size_t nargsf,
    PyObject* kwnames);

static PyObject* FastCudaLauncher_new(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwds) {
  HANDLE_TH_ERRORS
  auto* self =
      reinterpret_cast<FastCudaLauncherObject*>(type->tp_alloc(type, 0));
  if (!self)
    return nullptr;

  uint64_t func_ptr = 0;
  int numWarps = 0, shared = 0, nScratch = 0;
  const char* argTypes = nullptr;
  if (!PyArg_ParseTuple(
          args, "Kiisi", &func_ptr, &numWarps, &shared, &argTypes, &nScratch)) {
    Py_DECREF(self);
    return nullptr;
  }

  self->func = reinterpret_cast<CUfunction>(func_ptr); // NOLINT
  self->numWarps = static_cast<uint32_t>(numWarps);
  self->sharedMemBytes = static_cast<uint32_t>(shared);

  int nKernel = static_cast<int>(std::strlen(argTypes));
  int nTotal = nKernel + nScratch;
  if (nTotal > MAX_ARGS) {
    Py_DECREF(self);
    PyErr_Format(
        PyExc_ValueError,
        "_FastCudaLauncher: too many arguments (%d > %d)",
        nTotal,
        MAX_ARGS);
    return nullptr;
  }

  self->numKernelArgs = nKernel;
  self->numTotalArgs = nTotal;
  std::memcpy(self->argTypes, argTypes, nKernel);
  // Scratch slots are pointer type ('O')
  for (int i = nKernel; i < nTotal; ++i) {
    self->argTypes[i] = 'O';
  }
  self->argTypes[nTotal] = '\0';

  // Pre-compute kernelArgs pointers and zero all storage.
  std::memset(self->argStorage, 0, sizeof(self->argStorage));
  for (int i = 0; i < nTotal; ++i) {
    self->kernelArgs[i] = &self->argStorage[i];
  }

  // Set vectorcall function pointer.
  self->vectorcall = fast_launcher_vectorcall;

  return reinterpret_cast<PyObject*>(self);
  END_HANDLE_TH_ERRORS
}

static void FastCudaLauncher_dealloc(PyObject* self) {
  Py_TYPE(self)->tp_free(self);
}

// The hot path — called every kernel launch via vectorcall.
// args layout: [grid_x, grid_y, grid_z, stream, kernel_arg0, ...]
static PyObject* fast_launcher_vectorcall(
    PyObject* callable,
    PyObject* const* args,
    size_t nargsf,
    PyObject* kwnames) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      kwnames == nullptr,
      "_FastCudaLauncher: keyword arguments are not supported");
  auto* self = reinterpret_cast<FastCudaLauncherObject*>(callable);
  Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);

  // We expect at least 4 fixed args (grid_x, grid_y, grid_z, stream)
  // plus numKernelArgs kernel arguments.
  TORCH_CHECK(
      nargs >= 4 + self->numKernelArgs,
      "_FastCudaLauncher: expected ",
      4 + self->numKernelArgs,
      " args, got ",
      nargs);

  // Grid and stream are always integers from inductor codegen.
  // Debug-only guard; zero cost in release/opt builds.
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      PyLong_Check(args[0]) && PyLong_Check(args[1]) && PyLong_Check(args[2]),
      "_FastCudaLauncher: grid dimensions must be integers");
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      PyLong_Check(args[3]), "_FastCudaLauncher: stream must be an integer");

  int gridX = static_cast<int>(THPUtils_unpackLong(args[0]));
  int gridY = static_cast<int>(THPUtils_unpackLong(args[1]));
  int gridZ = static_cast<int>(THPUtils_unpackLong(args[2]));

  if (gridX <= 0 || gridY <= 0 || gridZ <= 0) {
    Py_RETURN_NONE;
  }

  uint64_t stream = THPUtils_unpackUInt64(args[3]);

  // Pack kernel args from Python into pre-allocated argStorage.
  for (int i = 0; i < self->numKernelArgs; ++i) {
    PyObject* item = args[4 + i];
    void* slot = static_cast<void*>(&self->argStorage[i]);
    char typeChar = self->argTypes[i];
    switch (typeChar) {
      case 'O': {
        *reinterpret_cast<CUdeviceptr*>(slot) = getPointerFast(item);
        break;
      }
      case 'b':
        convertType<int8_t>(THPUtils_unpackInt, "int8", slot, item);
        break;
      case 'h':
        convertType<int16_t>(THPUtils_unpackInt, "int16", slot, item);
        break;
      case 'i':
        convertType<int32_t>(THPUtils_unpackLong, "int32", slot, item);
        break;
      case 'l':
        convertType<int64_t>(THPUtils_unpackLong, "int64", slot, item);
        break;
      case 'B':
        convertType<uint8_t>(THPUtils_unpackUInt32, "uint8", slot, item);
        break;
      case 'H':
        convertType<uint16_t>(THPUtils_unpackUInt32, "uint16", slot, item);
        break;
      case 'I':
        convertType<uint32_t>(THPUtils_unpackUInt32, "uint32", slot, item);
        break;
      case 'K':
        convertType<uint64_t>(THPUtils_unpackUInt64, "uint64", slot, item);
        break;
      case 'f':
        convertType<float>(THPUtils_unpackDouble, "float", slot, item);
        break;
      case 'd':
        convertType<double>(THPUtils_unpackDouble, "double", slot, item);
        break;
      default:
        TORCH_CHECK(
            false, "_FastCudaLauncher: unknown arg type '", typeChar, "'");
    }
  }
  // Scratch slots already zeroed at construction time and stay zero.
  // Invariant: inductor codegen always passes None for scratch args, so the
  // pre-zeroed nullptr values remain valid across launches.  If scratch
  // semantics ever require non-null per-launch values, re-zero here.

  launchKernel(
      self->func,
      static_cast<uint32_t>(gridX),
      static_cast<uint32_t>(gridY),
      static_cast<uint32_t>(gridZ),
      self->numWarps,
      self->sharedMemBytes,
      self->kernelArgs,
      reinterpret_cast<cudaStream_t>(stream)); // NOLINT

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyTypeObject FastCudaLauncherType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "torch._C._FastCudaLauncher", // tp_name
    sizeof(FastCudaLauncherObject), // tp_basicsize
    0, // tp_itemsize
    FastCudaLauncher_dealloc, // tp_dealloc
    offsetof(FastCudaLauncherObject, vectorcall), // tp_vectorcall_offset
    nullptr, // tp_getattr
    nullptr, // tp_setattr
    nullptr, // tp_reserved
    nullptr, // tp_repr
    nullptr, // tp_as_number
    nullptr, // tp_as_sequence
    nullptr, // tp_as_mapping
    nullptr, // tp_hash
    PyVectorcall_Call, // tp_call — fallback for non-vectorcall callers
    nullptr, // tp_str
    nullptr, // tp_getattro
    nullptr, // tp_setattro
    nullptr, // tp_as_buffer
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_VECTORCALL,
    "Pre-bound fast launcher for triton CUDA kernels (vectorcall)", // tp_doc
    nullptr, // tp_traverse
    nullptr, // tp_clear
    nullptr, // tp_richcompare
    0, // tp_weaklistoffset
    nullptr, // tp_iter
    nullptr, // tp_iternext
    nullptr, // tp_methods
    nullptr, // tp_members
    nullptr, // tp_getset
    nullptr, // tp_base
    nullptr, // tp_dict
    nullptr, // tp_descr_get
    nullptr, // tp_descr_set
    0, // tp_dictoffset
    nullptr, // tp_init
    nullptr, // tp_alloc
    FastCudaLauncher_new, // tp_new
};

} // anonymous namespace
// Module initialization: add StaticCudaLauncher to the module with our static
// methods.
bool StaticCudaLauncher_init(PyObject* module) {
  if (PyType_Ready(&StaticCudaLauncherType) < 0) {
    return false;
  }
  // Add our static methods to the type's dictionary.
  PyObject* dict = StaticCudaLauncherType.tp_dict;
  for (auto& def : StaticCudaLauncherMethods) {
    PyObject* func = PyCFunction_New(&def, nullptr);
    if (!func) {
      return false;
    }
    PyObject* static_method = PyStaticMethod_New(func);
    Py_DECREF(func);
    if (PyDict_SetItemString(dict, def.ml_name, static_method) < 0) {
      Py_DECREF(static_method);
      return false;
    }
    Py_DECREF(static_method);
  }
  Py_INCREF(&StaticCudaLauncherType);
  if (PyModule_AddObject(
          module, "_StaticCudaLauncher", (PyObject*)&StaticCudaLauncherType) <
      0) {
    Py_DECREF(&StaticCudaLauncherType);
    return false;
  }
  return true;
}

bool FastCudaLauncher_init(PyObject* module) {
  if (PyType_Ready(&FastCudaLauncherType) < 0) {
    return false;
  }
  Py_INCREF(&FastCudaLauncherType);
  if (PyModule_AddObject(
          module, "_FastCudaLauncher", (PyObject*)&FastCudaLauncherType) < 0) {
    Py_DECREF(&FastCudaLauncherType);
    return false;
  }
  return true;
}
#endif
