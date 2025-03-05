#include <torch/csrc/inductor/static_cuda_launcher.h>
#include <torch/csrc/utils/pythoncapi_compat.h>
#include <cstdint>
#include <stdexcept>

#include <torch/csrc/inductor/cpp_wrapper/device_internal/cuda.h>
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
 */

#define CUDA_DRIVER_CHECK(EXPR)                                   \
  do {                                                            \
    CUresult code = EXPR;                                         \
    const char* msg;                                              \
    CUresult code_get_error = cuGetErrorString(code, &msg);       \
    if (code_get_error != CUDA_SUCCESS) {                         \
      throw std::runtime_error(                                   \
          std::string("CUDA driver error: ") +                    \
          std::string("invalid error code!"));                    \
    }                                                             \
    if (code != CUDA_SUCCESS) {                                   \
      throw std::runtime_error(                                   \
          std::string("CUDA driver error: ") + std::string(msg)); \
    }                                                             \
  } while (0);

static inline CUdeviceptr getPointer(PyObject* obj, int idx) {
  CUdeviceptr data_ptr = 0;
  if (PyLong_Check(obj)) {
    data_ptr = PyLong_AsUnsignedLongLong(obj);
    return data_ptr;
  }
  if (obj == Py_None) {
    // valid nullptr
    return data_ptr;
  }
  PyObject* ptr = PyObject_GetAttrString(obj, "data_ptr");
  if (ptr) {
    PyObject* empty_tuple = PyTuple_New(0);
    PyObject* ret = PyObject_Call(ptr, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    if (!PyLong_Check(ret)) {
      throw std::runtime_error(
          "data_ptr method of Pointer object must return 64-bit int");
    }
    data_ptr = PyLong_AsUnsignedLongLong(ret);
    if (!data_ptr)
      return data_ptr;

    CUdeviceptr dev_ptr;
    CUDA_DRIVER_CHECK(cuPointerGetAttribute(
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

  CUmodule mod;
  CUfunction func;
  CUDA_DRIVER_CHECK(cuModuleLoad(&mod, filePath.c_str()));
  CUDA_DRIVER_CHECK(cuModuleGetFunction(&func, mod, funcName.c_str()));
  if (sharedMemBytes > 0) {
    CUDA_DRIVER_CHECK(cuFuncSetAttribute(
        func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, sharedMemBytes))
  }
  return func;
}

static inline void launchKernel(
    CUfunction func,
    uint32_t gridX,
    uint32_t gridY,
    uint32_t gridZ,
    uint32_t numWarps,
    uint32_t sharedMemBytes,
    void* args[],
    cudaStream_t stream) {
  CUDA_DRIVER_CHECK(cuLaunchKernel(
      func,
      gridX,
      gridY,
      gridZ,
      32 * numWarps,
      1,
      1,
      sharedMemBytes,
      stream,
      args,
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
    uint64_t argStorage[NUM_ARGS],
    void* kernelArgs[NUM_ARGS]) {
  size_t numKernelArgs = std::strlen(argTypes);
  if (!PyTuple_Check(varArgs)) {
    throw std::runtime_error("Kernel arguments must be provided as a tuple");
  }
  if (PyTuple_Size(varArgs) != static_cast<Py_ssize_t>(numKernelArgs)) {
    throw std::runtime_error(
        "Mismatch between number of argument types and provided arguments");
  }
  for (size_t i = 0; i < numKernelArgs; ++i) {
    // Get pointer to the ith 8-byte slot.
    void* slot = static_cast<void*>(&argStorage[i]);
    PyObject* item = PyTuple_GetItem(varArgs, i);
    char typeChar = argTypes[i];

    switch (typeChar) {
      case 'b': { // (int8_t)
        long temp = PyLong_AsLong(item);
        if (PyErr_Occurred()) {
          throw std::runtime_error("Failed to convert argument to int8");
        }
        *reinterpret_cast<int8_t*>(slot) = static_cast<int8_t>(temp);
        break;
      }
      case 'h': { // (int16_t)
        long temp = PyLong_AsLong(item);
        if (PyErr_Occurred()) {
          throw std::runtime_error("Failed to convert argument to int16");
        }
        *reinterpret_cast<int16_t*>(slot) = static_cast<int16_t>(temp);
        break;
      }
      case 'i': { // (int32_t)
        long temp = PyLong_AsLong(item);
        if (PyErr_Occurred()) {
          throw std::runtime_error("Failed to convert argument to int32");
        }
        *reinterpret_cast<int32_t*>(slot) = static_cast<int32_t>(temp);
        break;
      }
      case 'l': { // (int64_t)
        long long temp = PyLong_AsLongLong(item);
        if (PyErr_Occurred()) {
          throw std::runtime_error("Failed to convert argument to int64");
        }
        *reinterpret_cast<int64_t*>(slot) = static_cast<int64_t>(temp);
        break;
      }
      case 'B': { // (uint8_t)
        unsigned long temp = PyLong_AsUnsignedLong(item);
        if (PyErr_Occurred()) {
          throw std::runtime_error("Failed to convert argument to uint8");
        }
        *reinterpret_cast<uint8_t*>(slot) = static_cast<uint8_t>(temp);
        break;
      }
      case 'H': { // (uint16_t)
        unsigned long temp = PyLong_AsUnsignedLong(item);
        if (PyErr_Occurred()) {
          throw std::runtime_error("Failed to convert argument to uint16");
        }
        *reinterpret_cast<uint16_t*>(slot) = static_cast<uint16_t>(temp);
        break;
      }
      case 'I': { // (uint32_t)
        unsigned long temp = PyLong_AsUnsignedLong(item);
        if (PyErr_Occurred()) {
          throw std::runtime_error("Failed to convert argument to uint32");
        }
        *reinterpret_cast<uint32_t*>(slot) = static_cast<uint32_t>(temp);
        break;
      }
      case 'K': { // (uint64_t)
        unsigned long long temp = PyLong_AsUnsignedLongLong(item);
        if (PyErr_Occurred()) {
          throw std::runtime_error("Failed to convert argument to uint64");
        }
        *reinterpret_cast<uint64_t*>(slot) = static_cast<uint64_t>(temp);
        break;
      }
      case 'f': { // float (fp16, bf16, fp32, f32)
        double temp = PyFloat_AsDouble(item);
        if (PyErr_Occurred()) {
          throw std::runtime_error("Failed to convert argument to float");
        }
        *reinterpret_cast<float*>(slot) = static_cast<float>(temp);
        break;
      }
      case 'd': { // double (64-bit float; fp64)
        double temp = PyFloat_AsDouble(item);
        if (PyErr_Occurred()) {
          throw std::runtime_error("Failed to convert argument to double");
        }
        *reinterpret_cast<double*>(slot) = temp;
        break;
      }
      case 'O': { // pointer; using helper getPointer() (which may call
                  // data_ptr() if needed)
        CUdeviceptr ptr = getPointer(item, i);
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

/**
 *  Main entrypoint function; called like this in python land:
    launcher(
      cubin_path, # File path of cubin file
      name, # Name of triton kernel
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
template <size_t NUM_ARGS>
static PyObject* launch_kernel(PyObject* self, PyObject* args) {
  const char* filePath;
  const char* funcName;
  int gridX, gridY, gridZ, numWarps, sharedMemBytes;
  // stream here should be the raw stream gotten from
  // device_interface.get_raw_stream()
  uint64_t stream;
  const char* argTypes;
  PyObject* varArgs;
  // Parse the fixed arguments and the format string
  if (!PyArg_ParseTuple(
          args,
          "ssiiiiisOl",
          &filePath,
          &funcName,
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
  CUfunction func;
  try {
    func = loadKernel(filePath, funcName, sharedMemBytes);
  } catch (const std::runtime_error& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }
  cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(stream);
  try {
    // Launch the kernel
    // Prepare the arguments for the kernel
    // We allocate 8 bytes per argument on the stack. We then allocate 8 more
    // bytes to point to each 8 byte slot in argStorage, and pass that array of
    // pointers to launchKernel.
    std::array<uint64_t, NUM_ARGS> argStorage;
    std::array<void*, NUM_ARGS> kernelArgs;
    parseKernelArgs<NUM_ARGS>(
        varArgs, argTypes, argStorage.data(), kernelArgs.data());

    launchKernel(
        func,
        gridX,
        gridY,
        gridZ,
        numWarps,
        sharedMemBytes,
        kernelArgs.data(),
        cudaStream);
  } catch (const std::runtime_error& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }
  Py_RETURN_NONE;
}

static PyMethodDef StaticCudaLauncherMethods[] = {
    {"_launch_cuda_kernel_1",
     (PyCFunction)launch_kernel<1>,
     METH_VARARGS,
     "Cuda Launcher with 1 arg"},
    {"_launch_cuda_kernel_2",
     (PyCFunction)launch_kernel<2>,
     METH_VARARGS,
     "Cuda Launcher with 2 args"},
    {"_launch_cuda_kernel_3",
     (PyCFunction)launch_kernel<3>,
     METH_VARARGS,
     "Cuda Launcher with 3 args"},
    {"_launch_cuda_kernel_4",
     (PyCFunction)launch_kernel<4>,
     METH_VARARGS,
     "Cuda Launcher with 4 args"},
    {"_launch_cuda_kernel_5",
     (PyCFunction)launch_kernel<5>,
     METH_VARARGS,
     "Cuda Launcher with 5 args"},
    {"_launch_cuda_kernel_6",
     (PyCFunction)launch_kernel<6>,
     METH_VARARGS,
     "Cuda Launcher with 6 args"},
    {"_launch_cuda_kernel_7",
     (PyCFunction)launch_kernel<7>,
     METH_VARARGS,
     "Cuda Launcher with 7 args"},
    {"_launch_cuda_kernel_8",
     (PyCFunction)launch_kernel<8>,
     METH_VARARGS,
     "Cuda Launcher with 8 args"},
    {"_launch_cuda_kernel_9",
     (PyCFunction)launch_kernel<9>,
     METH_VARARGS,
     "Cuda Launcher with 9 args"},
    {"_launch_cuda_kernel_10",
     (PyCFunction)launch_kernel<10>,
     METH_VARARGS,
     "Cuda Launcher with 10 args"},
    {nullptr, nullptr, 0, nullptr} // Sentinel
};

// Define a minimal type for StaticCudaLauncher.
// We don't implement __new__ or __init__ because we're using it only as a
// container for static methods.
static PyTypeObject StaticCudaLauncherType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "torch._C._StaticCudaLauncher", // tp_name
    sizeof(PyObject), // tp_basicsize
    0, // tp_itemsize
    0, // tp_dealloc
    0, // tp_print (deprecated)
    0, // tp_getattr
    0, // tp_setattr
    0, // tp_reserved
    0, // tp_repr
    0, // tp_as_number
    0, // tp_as_sequence
    0, // tp_as_mapping
    0, // tp_hash
    0, // tp_call
    0, // tp_str
    0, // tp_getattro
    0, // tp_setattro
    0, // tp_as_buffer
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_DISALLOW_INSTANTIATION,
    "Statically defined launchers for triton compiled CUDA kernels", // tp_doc
    0, // tp_traverse
    0, // tp_clear
    0, // tp_richcompare
    0, // tp_weaklistoffset
    0, // tp_iter
    0, // tp_iternext
    nullptr, // tp_methods
    0, // tp_members
    0, // tp_getset
    0, // tp_base
    0, // tp_dict (automatically allocated)
    0, // tp_descr_get
    0, // tp_descr_set
    0, // tp_dictoffset
    0, // tp_init
    0, // tp_alloc
    0, // tp_new
};

// Module initialization: add StaticCudaLauncher to the module with our static
// methods.
bool StaticCudaLauncher_init(PyObject* module) {
  if (PyType_Ready(&StaticCudaLauncherType) < 0) {
    return false;
  }
  // Add our static methods to the type's dictionary.
  PyObject* dict = StaticCudaLauncherType.tp_dict;
  for (PyMethodDef* def = StaticCudaLauncherMethods; def->ml_name != nullptr;
       def++) {
    PyObject* func = PyCFunction_New(def, nullptr);
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
