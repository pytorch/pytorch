
#ifdef USE_CUDA
// TODO what are the right imports to get access to CUDA drivers?
#include <torch/csrc/utils/pythoncapi_compat.h>

#include <ATen/Context.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/inductor/static_cuda_launcher.h>
#include <cstdint>
#include <stdexcept>

#include <torch/csrc/utils/python_numbers.h>
#include <filesystem>
#include <optional>
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

static const at::cuda::NVRTC& nvrtc() {
  return at::globalContext().getNVRTC();
}

static CUdeviceptr getPointer(PyObject* obj) {
  CUdeviceptr data_ptr = 0;
  if (THPUtils_checkLong(obj)) {
    data_ptr = THPUtils_unpackUInt64(obj);
    return data_ptr;
  }
  if (obj == Py_None) {
    // valid nullptr
    return data_ptr;
  }
  PyObject* ptr = PyObject_GetAttrString(obj, "data_ptr");
  if (ptr) {
    PyObject* empty_tuple = PyTuple_New(0);
    PyObject* ret = PyObject_Call(ptr, empty_tuple, nullptr);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    if (!THPUtils_checkLong(ret)) {
      throw std::runtime_error(
          "data_ptr method of Pointer object must return 64-bit int");
    }
    data_ptr = THPUtils_unpackUInt64(ret);
    if (!data_ptr)
      return data_ptr;

    CUdeviceptr dev_ptr = 0;
    AT_CUDA_DRIVER_CHECK(nvrtc().cuPointerGetAttribute(
        &dev_ptr, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, data_ptr));
    Py_DECREF(ret);
    return dev_ptr;
  }
  throw std::runtime_error(
      "Pointer argument must be either uint64 or have data_ptr method");
}

static inline CUfunction loadKernel(
    std::string filePath,
    const std::string& funcName,
    uint32_t sharedMemBytes,
    const std::optional<std::string>& cubinDir = std::nullopt) {
  if (cubinDir) {
    std::filesystem::path p1{*cubinDir};
    std::filesystem::path p2{filePath};
    filePath = (p1 / p2.filename()).string();
  }

  CUmodule mod = nullptr;
  CUfunction func = nullptr;
  AT_CUDA_DRIVER_CHECK(nvrtc().cuModuleLoad(&mod, filePath.c_str()));
  AT_CUDA_DRIVER_CHECK(
      nvrtc().cuModuleGetFunction(&func, mod, funcName.c_str()));
  if (sharedMemBytes > 0) {
    AT_CUDA_DRIVER_CHECK(nvrtc().cuFuncSetAttribute(
        func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, sharedMemBytes));
  }
  return func;
}

template <size_t NUM_ARGS>
static inline void launchKernel(
    CUfunction func,
    uint32_t gridX,
    uint32_t gridY,
    uint32_t gridZ,
    uint32_t numWarps,
    uint32_t sharedMemBytes,
    std::array<void*, NUM_ARGS>& args,
    cudaStream_t stream) {
  AT_CUDA_DRIVER_CHECK(nvrtc().cuLaunchKernel(
      func,
      gridX,
      gridY,
      gridZ,
      32 * numWarps,
      1,
      1,
      sharedMemBytes,
      stream,
      args.data(),
      nullptr));
}

/**
   Given a list of args and their types (in a string), along with two stack
   allocated arrays, puts each argument arg_{i} into argStorage[i], and a
   pointer to the argument in kernelArgs[i]. We then can pass `kernelArgs`
   directly to launchKernel. Note that some args can be less than 8 bytes, but
   we'll still allocate 8 bytes on the stack for them.
 */
template <size_t NUM_ARGS>
void parseKernelArgs(
    PyObject* varArgs,
    const char* argTypes,
    std::array<uint64_t, NUM_ARGS>& argStorage,
    std::array<void*, NUM_ARGS>& kernelArgs) {
  int numKernelArgs = static_cast<int>(std::strlen(argTypes));
  if (!PyTuple_Check(varArgs)) {
    throw std::runtime_error("Kernel arguments must be provided as a tuple");
  }
  if (PyTuple_Size(varArgs) != static_cast<Py_ssize_t>(numKernelArgs)) {
    throw std::runtime_error(
        "Mismatch between number of argument types and provided arguments");
  }
  for (int i = 0; i < numKernelArgs; ++i) {
    // Get pointer to the ith 8-byte slot.
    void* slot = static_cast<void*>(&argStorage[i]);
    PyObject* item = PyTuple_GetItem(varArgs, i);
    char typeChar = argTypes[i];

    switch (typeChar) {
      case 'b': { // (int8_t)
        long temp = THPUtils_unpackInt(item);
        if (PyErr_Occurred()) {
          throw std::runtime_error("Failed to convert argument to int8");
        }
        *reinterpret_cast<int8_t*>(slot) = static_cast<int8_t>(temp);
        break;
      }
      case 'h': { // (int16_t)
        long temp = THPUtils_unpackInt(item);
        if (PyErr_Occurred()) {
          throw std::runtime_error("Failed to convert argument to int16");
        }
        *reinterpret_cast<int16_t*>(slot) = static_cast<int16_t>(temp);
        break;
      }
      case 'i': { // (int32_t)
        long temp = THPUtils_unpackLong(item);
        if (PyErr_Occurred()) {
          throw std::runtime_error("Failed to convert argument to int32");
        }
        *reinterpret_cast<int32_t*>(slot) = static_cast<int32_t>(temp);
        break;
      }
      case 'l': { // (int64_t)
        long long temp = THPUtils_unpackLong(item);
        if (PyErr_Occurred()) {
          throw std::runtime_error("Failed to convert argument to int64");
        }
        *reinterpret_cast<int64_t*>(slot) = static_cast<int64_t>(temp);
        break;
      }
      case 'B': { // (uint8_t)
        unsigned long temp = THPUtils_unpackUInt32(item);
        if (PyErr_Occurred()) {
          throw std::runtime_error("Failed to convert argument to uint8");
        }
        *reinterpret_cast<uint8_t*>(slot) = static_cast<uint8_t>(temp);
        break;
      }
      case 'H': { // (uint16_t)
        unsigned long temp = THPUtils_unpackUInt32(item);
        if (PyErr_Occurred()) {
          throw std::runtime_error("Failed to convert argument to uint16");
        }
        *reinterpret_cast<uint16_t*>(slot) = static_cast<uint16_t>(temp);
        break;
      }
      case 'I': { // (uint32_t)
        unsigned long temp = THPUtils_unpackUInt32(item);
        if (PyErr_Occurred()) {
          throw std::runtime_error("Failed to convert argument to uint32");
        }
        *reinterpret_cast<uint32_t*>(slot) = static_cast<uint32_t>(temp);
        break;
      }
      case 'K': { // (uint64_t)
        unsigned long long temp = THPUtils_unpackUInt64(item);
        if (PyErr_Occurred()) {
          throw std::runtime_error("Failed to convert argument to uint64");
        }
        *reinterpret_cast<uint64_t*>(slot) = static_cast<uint64_t>(temp);
        break;
      }
      case 'f': { // float (fp16, bf16, fp32, f32)
        double temp = THPUtils_unpackDouble(item);
        if (PyErr_Occurred()) {
          throw std::runtime_error("Failed to convert argument to float");
        }
        *reinterpret_cast<float*>(slot) = static_cast<float>(temp);
        break;
      }
      case 'd': { // double (64-bit float; fp64)
        double temp = THPUtils_unpackDouble(item);
        if (PyErr_Occurred()) {
          throw std::runtime_error("Failed to convert argument to double");
        }
        *reinterpret_cast<double*>(slot) = temp;
        break;
      }
      case 'O': { // pointer; using helper getPointer() (which may call
                  // data_ptr() if needed)
        CUdeviceptr ptr = getPointer(item);
        *reinterpret_cast<CUdeviceptr*>(slot) = ptr;
        break;
      }
      default:
        throw std::runtime_error(
            std::string("Unsupported argument type: ") + typeChar);
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
static PyObject* load_kernel(PyObject* self, PyObject* args) {
  const char* filePath = nullptr;
  const char* funcName = nullptr;
  int sharedMemBytes = 0;
  int n_regs = 0;
  int n_spills = 0;
  if (!PyArg_ParseTuple(args, "ssi", &filePath, &funcName, &sharedMemBytes)) {
    return nullptr;
  }
  CUfunction func = nullptr;
  try {
    func = loadKernel(filePath, funcName, sharedMemBytes);
    // Taken from triton/nvidia/backend/driver.c
    AT_CUDA_DRIVER_CHECK(
        nvrtc().cuFuncGetAttribute(&n_regs, CU_FUNC_ATTRIBUTE_NUM_REGS, func));
    AT_CUDA_DRIVER_CHECK(nvrtc().cuFuncGetAttribute(
        &n_spills, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, func));
    n_spills /= 4;
  } catch (const std::runtime_error& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }
  // Return a tuple of CUFunction, n_regs, n_spills
  return Py_BuildValue(
      "(Kii)", reinterpret_cast<uint64_t>(func), n_regs, n_spills);
}

template <size_t NUM_ARGS>
static PyObject* launch_kernel_inner(
    CUfunction func,
    int gridX,
    int gridY,
    int gridZ,
    int numWarps,
    int sharedMemBytes,
    const char* argTypes,
    PyObject* varArgs,
    cudaStream_t cudaStream) {
  try {
    // Launch the kernel
    // Prepare the arguments for the kernel
    // We allocate 8 bytes per argument on the stack. We then allocate 8 more
    // bytes to point to each 8 byte slot in argStorage, and pass that array of
    // pointers to launchKernel.
    std::array<uint64_t, NUM_ARGS> argStorage = {};
    std::array<void*, NUM_ARGS> kernelArgs = {};
    parseKernelArgs(varArgs, argTypes, argStorage, kernelArgs);

    launchKernel(
        func,
        gridX,
        gridY,
        gridZ,
        numWarps,
        sharedMemBytes,
        kernelArgs,
        cudaStream);
  } catch (const std::runtime_error& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }
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
static PyObject* launch_kernel(PyObject* self, PyObject* args) {
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
          "KiiiiisOl",
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
  CUfunction func = reinterpret_cast<CUfunction>(func_ptr); // NOLINT
  cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(stream); // NOLINT
  auto num_args = std::strlen(argTypes);
  switch (num_args) {
    case 1:
      return launch_kernel_inner<1>(
          func,
          gridX,
          gridY,
          gridZ,
          numWarps,
          sharedMemBytes,
          argTypes,
          varArgs,
          cudaStream);
    case 2:
      return launch_kernel_inner<2>(
          func,
          gridX,
          gridY,
          gridZ,
          numWarps,
          sharedMemBytes,
          argTypes,
          varArgs,
          cudaStream);
    case 3:
      return launch_kernel_inner<3>(
          func,
          gridX,
          gridY,
          gridZ,
          numWarps,
          sharedMemBytes,
          argTypes,
          varArgs,
          cudaStream);
    case 4:
      return launch_kernel_inner<4>(
          func,
          gridX,
          gridY,
          gridZ,
          numWarps,
          sharedMemBytes,
          argTypes,
          varArgs,
          cudaStream);
    case 5:
      return launch_kernel_inner<5>(
          func,
          gridX,
          gridY,
          gridZ,
          numWarps,
          sharedMemBytes,
          argTypes,
          varArgs,
          cudaStream);
    case 6:
      return launch_kernel_inner<6>(
          func,
          gridX,
          gridY,
          gridZ,
          numWarps,
          sharedMemBytes,
          argTypes,
          varArgs,
          cudaStream);
    case 7:
      return launch_kernel_inner<7>(
          func,
          gridX,
          gridY,
          gridZ,
          numWarps,
          sharedMemBytes,
          argTypes,
          varArgs,
          cudaStream);
    case 8:
      return launch_kernel_inner<8>(
          func,
          gridX,
          gridY,
          gridZ,
          numWarps,
          sharedMemBytes,
          argTypes,
          varArgs,
          cudaStream);
    case 9:
      return launch_kernel_inner<9>(
          func,
          gridX,
          gridY,
          gridZ,
          numWarps,
          sharedMemBytes,
          argTypes,
          varArgs,
          cudaStream);
    case 10:
      return launch_kernel_inner<10>(
          func,
          gridX,
          gridY,
          gridZ,
          numWarps,
          sharedMemBytes,
          argTypes,
          varArgs,
          cudaStream);
    default:
      throw std::runtime_error(
          "Unsupported number of arguments; must be between 1 and 10");
  }
}

static std::array<PyMethodDef, 2> StaticCudaLauncherMethods = {
    PyMethodDef{
        "_launch_kernel",
        (PyCFunction)launch_kernel,
        METH_VARARGS,
        "Cuda Launcher with up to 10 args"},
    PyMethodDef{
        "_load_kernel",
        (PyCFunction)load_kernel,
        METH_VARARGS,
        "Load CUDA kernel from cubin file"}};

// Define a minimal type for StaticCudaLauncher.
// We don't implement __new__ or __init__ because we're using it only as a
// container for static methods.
static PyTypeObject StaticCudaLauncherType = {
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
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_DISALLOW_INSTANTIATION,
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

// Module initialization: add StaticCudaLauncher to the module with our static
// methods.
bool StaticCudaLauncher_init(PyObject* module) {
  if (PyType_Ready(&StaticCudaLauncherType) < 0) {
    return false;
  }
  // Add our static methods to the type's dictionary.
  PyObject* dict = StaticCudaLauncherType.tp_dict;

  for (auto def = StaticCudaLauncherMethods.begin();
       def != StaticCudaLauncherMethods.end();
       def++) {
    PyObject* func = PyCFunction_New(def, nullptr); // &*
    if (!func) {
      return false;
    }
    PyObject* static_method = PyStaticMethod_New(func);
    Py_DECREF(func);
    if (PyDict_SetItemString(dict, def->ml_name, static_method) < 0) {
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
#endif
