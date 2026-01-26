/**
 * This file follows the API design of "static_cuda_launcher.cpp" and copied
 * parts of the code.
 * TODO: Extract the parts shared with static_cuda_launcher.cpp and unify to a
 * static_triton_launcher.h
 */

#if defined(USE_XPU)
// TODO: enable on Windows.
#ifndef _WIN32
#include <torch/csrc/utils/pythoncapi_compat.h>

#include <ATen/Context.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <fmt/format.h>
#include <torch/csrc/inductor/static_launcher/xpu.h>
#include <cstdint>
#include <stdexcept>

#include <torch/csrc/utils/python_numbers.h>
#include <filesystem>
#include <fstream>
#include <optional>

#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>

#define ZE_CHECK(status)                                                  \
  {                                                                       \
    if (status != ZE_RESULT_SUCCESS) {                                    \
      std::stringstream ss;                                               \
      ss << "L0 runtime error: " << std::hex << std::uppercase << status; \
      throw std::runtime_error(ss.str());                                 \
    }                                                                     \
  }

namespace {

/**
 * For num_args <= MAX_ARGS, we use static allocated memory for better
 * performance. And for num_args > MAX_ARGS, we use dynamic allocated heap
 * memory. Here we use 120 as MAX_ARGS following static_cuda_launcher.cpp which
 * has been tuned before.
 */
// 120 max args + 1 for global scratch size
#define MAX_ARGS 121
typedef void* syclDevicePtr_t;

syclDevicePtr_t getPointer(
    PyObject* obj,
    int idx,
    const sycl::queue* queuePtr) {
  syclDevicePtr_t data_ptr = 0;

  if (THPUtils_checkLong(obj)) {
    data_ptr = reinterpret_cast<syclDevicePtr_t>(THPUtils_unpackUInt64(obj));

    return data_ptr;
  }
  if (obj == Py_None) {
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

  data_ptr = reinterpret_cast<syclDevicePtr_t>(THPUtils_unpackUInt64(ret));

  if (!data_ptr)
    return data_ptr;

  auto context = queuePtr->get_context();
  auto handle = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(context);
  ze_memory_allocation_properties_t prop;
  prop.stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
  prop.pNext = nullptr;
  auto res = zeMemGetAllocProperties(
      (ze_context_handle_t)handle, data_ptr, &prop, nullptr);

  TORCH_CHECK(
      res == ZE_RESULT_SUCCESS,
      fmt::format(
          "Failed to get memory properties for pointer argument at {}-th argument, err={}",
          idx,
          static_cast<int>(res)));

  TORCH_CHECK(
      prop.type == ZE_MEMORY_TYPE_DEVICE,
      fmt::format(
          "Pointer argument doesn't reference XPU device memory at {}-th argument, err={}",
          idx,
          static_cast<int>(res)));

  return data_ptr;
}

// TODO: unify and reuse with static_cuda_launcher.cpp
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
*/
void parseKernelArgs(
    PyObject* varArgs,
    const char* argTypes,
    uint64_t* argStorage,
    void** kernelArgs,
    const sycl::queue* queuePtr) {
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
        syclDevicePtr_t ptr = getPointer(item, i, queuePtr);
        *reinterpret_cast<syclDevicePtr_t*>(slot) = ptr;
        break;
      }
      default:
        TORCH_CHECK(false, "Unknown type passed in: ", typeChar);
    }
    // Save the pointer to this slot.
    kernelArgs[i] = slot;
  }
}

inline ze_module_handle_t _createModule(
    const uint8_t* binaryPtr,
    size_t binarySize,
    const int device_idx) {
  auto& syclDevice = c10::xpu::get_raw_device(device_idx);
  auto& syclContext = c10::xpu::get_device_context();
  auto device =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(syclDevice);
  auto context =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(syclContext);

  const char* buildFlags = "";
  const ze_module_format_t format = ZE_MODULE_FORMAT_NATIVE;
  ze_module_desc_t moduleDescription = {};
  moduleDescription.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
  moduleDescription.format = format;
  moduleDescription.inputSize = binarySize;
  moduleDescription.pInputModule = (uint8_t*)binaryPtr;
  moduleDescription.pBuildFlags = buildFlags;
  ze_module_build_log_handle_t buildLog = nullptr;
  ze_module_handle_t module = nullptr;
  auto error_no =
      zeModuleCreate(context, device, &moduleDescription, &module, &buildLog);

  if (error_no != ZE_RESULT_SUCCESS) {
    size_t szLog = 0;
    ZE_CHECK(zeModuleBuildLogGetString(buildLog, &szLog, nullptr));
    std::vector<char> log(szLog);
    ZE_CHECK(zeModuleBuildLogGetString(buildLog, &szLog, log.data()));
    std::cerr << "L0 build module failed. Log: " << log.data() << std::endl;
  }
  if (buildLog) {
    ZE_CHECK(zeModuleBuildLogDestroy(buildLog));
  }
  ZE_CHECK(error_no);
  return module;
}

inline sycl::kernel* _createKernel(
    ze_module_handle_t module,
    const char* kernelName,
    uint32_t* nSpillsPtr = nullptr) {
  assert(module);
  assert(kernelName);
  ze_kernel_handle_t kernel = nullptr;
  ze_kernel_desc_t kernelDescription = {};
  kernelDescription.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
  kernelDescription.pNext = nullptr;
  kernelDescription.flags = ZE_KERNEL_FLAG_FORCE_RESIDENCY;
  kernelDescription.pKernelName = kernelName;
  ZE_CHECK(zeKernelCreate(module, &kernelDescription, &kernel));
  if (nSpillsPtr) {
    ze_kernel_properties_t props;
    props.stype = ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES;
    props.pNext = nullptr;
    ZE_CHECK(zeKernelGetProperties(kernel, &props));
    *nSpillsPtr = props.spillMemSize;
  }
  auto& syclContext = c10::xpu::get_device_context();
  auto mod = sycl::make_kernel_bundle<
      sycl::backend::ext_oneapi_level_zero,
      sycl::bundle_state::executable>(
      {module, sycl::ext::oneapi::level_zero::ownership::transfer},
      syclContext);
  auto fun =
      new sycl::kernel(sycl::make_kernel<sycl::backend::ext_oneapi_level_zero>(
          {mod, kernel, sycl::ext::oneapi::level_zero::ownership::transfer},
          syclContext));
  return fun;
}

sycl::kernel* loadKernel(
    const char* filePath,
    const char* funcName,
    uint32_t sharedMemBytes,
    uint32_t* nSpillsPtr,
    int device_idx) {
  std::ifstream IFS(filePath, std::ios::binary);
  std::ostringstream OSS;
  OSS << IFS.rdbuf();
  std::string data(OSS.str());
  auto mod = _createModule(
      reinterpret_cast<const uint8_t*>(data.c_str()), data.size(), device_idx);

  return _createKernel(mod, funcName, nSpillsPtr);
}

void launchKernel(
    sycl::kernel* kernelPtr,
    uint32_t gridX,
    uint32_t gridY,
    uint32_t gridZ,
    uint32_t numWarps,
    uint32_t sharedMemBytes,
    void** params,
    sycl::queue* queuePtr) {
  uint32_t threadsPerWarp = kernelPtr->get_info<
      sycl::info::kernel_device_specific::compile_sub_group_size>(
      queuePtr->get_device());
  if (threadsPerWarp == 0) {
    threadsPerWarp = 32; // default to 32 if not set
  }
  std::string kernelName =
      kernelPtr->get_info<sycl::info::kernel::function_name>();
  uint32_t numParams = kernelPtr->get_info<sycl::info::kernel::num_args>();
  size_t globalRangeX = gridX * threadsPerWarp * numWarps;
  size_t globalRangeY = gridY;
  size_t globalRangeZ = gridZ;
  size_t localRangeX = numWarps * threadsPerWarp;
  size_t localRangeY = 1;
  size_t localRangeZ = 1;
  sycl::range<3> globalRange(globalRangeZ, globalRangeY, globalRangeX);
  sycl::range<3> localRange(localRangeZ, localRangeY, localRangeX);
  sycl::nd_range<3> parallelWorkSize(globalRange, localRange);
  if (sharedMemBytes > 0) {
    // numParams from sycl info  = user provided args + sharedMemoryBuffer
    numParams -= 1;
  }
  // Submit the imported kernel.
  auto cgf = [&](sycl::handler& cgh) {
    for (uint32_t i = 0; i < numParams; ++i) {
      cgh.set_arg(i, *(static_cast<void**>(params[i])));
    }

    if (sharedMemBytes > 0) {
      using share_mem_t = sycl::local_accessor<int8_t, 1>;
      share_mem_t localBuffer = share_mem_t(sharedMemBytes, cgh);
      cgh.set_arg(numParams, localBuffer);
      cgh.parallel_for(parallelWorkSize, *kernelPtr);
    } else {
      cgh.parallel_for(parallelWorkSize, *kernelPtr);
    }
  };
  auto event = queuePtr->submit(cgf);
}

/* Load the kernel into memory (called during torch.compile), and
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
  int device = 0;
  if (!PyArg_ParseTuple(
          args, "ssii", &filePath, &funcName, &sharedMemBytes, &device)) {
    return nullptr;
  }
  // Level-zero does not support get n_regs, so we return 0 here.
  uint32_t n_regs = 0;
  uint32_t n_spills = 0;
  sycl::kernel* func =
      loadKernel(filePath, funcName, sharedMemBytes, &n_spills, device);

  PyObject* kernel_py = PyCapsule_New(
      reinterpret_cast<void*>(func), "sycl_kernel", [](PyObject* cap) {
        void* ptr = PyCapsule_GetPointer(cap, "sycl_kernel");
        delete reinterpret_cast<sycl::kernel*>(ptr);
      });

  return Py_BuildValue("(Oii)", kernel_py, n_regs, n_spills);
  END_HANDLE_TH_ERRORS
}

PyObject* launch_kernel_inner(
    sycl::kernel* func,
    int gridX,
    int gridY,
    int gridZ,
    int numWarps,
    int sharedMemBytes,
    const char* argTypes,
    PyObject* varArgs,
    sycl::queue* queuePtr) {
  // Launch the kernel
  // Prepare the arguments for the kernel
  // We allocate 8 bytes per argument on the stack. We then allocate 8 more
  // bytes to point to each 8 byte slot in argStorage, and pass that array of
  // pointers to launchKernel.
  std::array<uint64_t, MAX_ARGS> argStorage = {};
  std::array<void*, MAX_ARGS> kernelArgs = {};
  parseKernelArgs(
      varArgs, argTypes, argStorage.data(), kernelArgs.data(), queuePtr);
  launchKernel(
      func,
      gridX,
      gridY,
      gridZ,
      numWarps,
      sharedMemBytes,
      kernelArgs.data(),
      queuePtr);

  Py_RETURN_NONE;
}

PyObject* launch_kernel_slow(
    sycl::kernel* func,
    int gridX,
    int gridY,
    int gridZ,
    int numWarps,
    int sharedMemBytes,
    const char* argTypes,
    PyObject* varArgs,
    sycl::queue* queuePtr) {
  /* For the slow case, allocate memory on the stack instead of the heap */
  size_t numArgs = std::strlen(argTypes);
  std::vector<uint64_t> argStorage(numArgs);
  std::vector<void*> kernelArgs(numArgs);

  parseKernelArgs(
      varArgs, argTypes, argStorage.data(), kernelArgs.data(), queuePtr);

  launchKernel(
      func,
      gridX,
      gridY,
      gridZ,
      numWarps,
      sharedMemBytes,
      kernelArgs.data(),
      queuePtr);
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
  PyObject* kernel_py = nullptr;
  int gridX = 0, gridY = 0, gridZ = 0, numWarps = 0, sharedMemBytes = 0;
  // stream here should be the raw stream gotten from
  // device_interface.get_raw_stream()
  uint64_t stream = 0;
  const char* argTypes = nullptr;
  PyObject* varArgs = nullptr;
  // Parse the fixed arguments and the format string
  if (!PyArg_ParseTuple(
          args,
          "OiiiiisOK",
          &kernel_py,
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

  sycl::kernel* func = reinterpret_cast<sycl::kernel*>(
      PyCapsule_GetPointer(kernel_py, "sycl_kernel")); // NOLINT
  sycl::queue* queuePtr = reinterpret_cast<sycl::queue*>(stream); // NOLINT
  auto num_args = std::strlen(argTypes);
  // Kernels with no arguments should just pass nullptr to cuLaunchKernel
  if (num_args == 0) {
    launchKernel(
        func, gridX, gridY, gridZ, numWarps, sharedMemBytes, nullptr, queuePtr);
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
        queuePtr);
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
        queuePtr);
  }
  END_HANDLE_TH_ERRORS
}

std::array<PyMethodDef, 2> StaticXpuLauncherMethods = {
    PyMethodDef{
        "_launch_kernel",
        launch_kernel,
        METH_VARARGS,
        "Statically launch triton compiled XPU kernels"},
    PyMethodDef{
        "_load_kernel",
        load_kernel,
        METH_VARARGS,
        "Load XPU kernel from zebin file"}};

// Define a minimal type for StaticXpuLauncher.
// We don't implement __new__ or __init__ because we're using it only as a
// container for static methods.
PyTypeObject StaticXpuLauncherType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "torch._C._StaticXpuLauncher", // tp_name
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
    "Statically defined launchers for triton compiled kernels", // tp_doc
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
} // anonymous namespace
// Module initialization: add StaticXpuLauncher to the module with our static
// methods.
bool StaticXpuLauncher_init(PyObject* module) {
  if (PyType_Ready(&StaticXpuLauncherType) < 0) {
    return false;
  }
  // Add our static methods to the type's dictionary.
  PyObject* dict = StaticXpuLauncherType.tp_dict;
  for (auto& def : StaticXpuLauncherMethods) {
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
  Py_INCREF(&StaticXpuLauncherType);
  if (PyModule_AddObject(
          module, "_StaticXpuLauncher", (PyObject*)&StaticXpuLauncherType) <
      0) {
    Py_DECREF(&StaticXpuLauncherType);
    return false;
  }
  return true;
}
#endif
#endif
