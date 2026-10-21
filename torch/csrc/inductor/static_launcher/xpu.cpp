/**
 * This file follows the API design of "static_launcher/cuda.cpp" and copied
 * parts of the code.
 * TODO: Extract the parts shared with static_launcher/cuda.cpp and unify to a
 * common static_triton_launcher.h
 */

#if defined(USE_XPU)
// TODO: enable on Windows.
#ifndef _WIN32
#include <fmt/format.h>
#include <array>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <type_traits>

#include <torch/csrc/utils/pythoncapi_compat.h>

#include <ATen/Context.h>
#include <ATen/xpu/level_zero_stub/ATenLevelZero.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <torch/csrc/inductor/static_launcher/common.h>
#include <torch/csrc/inductor/static_launcher/xpu.h>
#include <cstdint>
#include <cstring>

#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>
#include <torch/csrc/utils/python_numbers.h>

#define ZE_CHECK(status)                                                  \
  {                                                                       \
    if (status != ZE_RESULT_SUCCESS) {                                    \
      std::stringstream ss;                                               \
      ss << "L0 runtime error: " << std::hex << std::uppercase << status; \
      TORCH_CHECK(false, std::move(ss).str());                            \
    }                                                                     \
  }

namespace {
const at::xpu::LevelZero& ze() {
  return at::globalContext().getLevelZero();
}
/**
 * For num_args <= MAX_ARGS, we use static allocated memory for better
 * performance. And for num_args > MAX_ARGS, we use dynamic allocated heap
 * memory. Here we use 120 as MAX_ARGS following static_cuda_launcher.cpp which
 * has been tuned before.
 */
// 120 max args + 1 for global scratch size
#define MAX_ARGS 121
typedef void* syclDevicePtr_t;

/**
 * A loaded kernel and the launch properties that are fixed once it is loaded:
 * the SIMD width it was compiled for (which with num_warps gives the work-group
 * size) and its argument count.
 */
struct LoadedKernel {
  sycl::kernel kernel;
  uint32_t threadsPerWarp = 0;
  uint32_t numArgs = 0;
};

syclDevicePtr_t callDataPtrMethod(PyObject* obj) {
  auto ptr = THPObjectPtr{PyObject_GetAttrString(obj, "data_ptr")};
  TORCH_CHECK(
      ptr != nullptr,
      "Pointer argument must be either uint64 or have data_ptr method")
  auto empty_tuple = THPObjectPtr{PyTuple_New(0)};
  auto ret = THPObjectPtr{PyObject_Call(ptr, empty_tuple, nullptr)};
  TORCH_CHECK(
      THPUtils_checkLong(ret),
      "data_ptr method of Pointer object must return 64-bit int");

  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  return reinterpret_cast<syclDevicePtr_t>(THPUtils_unpackUInt64(ret));
}

syclDevicePtr_t getPointer(
    PyObject* obj,
    int idx,
    const sycl::queue* queuePtr) {
  if (THPUtils_checkLong(obj)) {
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    return reinterpret_cast<syclDevicePtr_t>(THPUtils_unpackUInt64(obj));
  }
  if (Py_IsNone(obj)) {
    // valid nullptr
    return nullptr;
  }
  syclDevicePtr_t data_ptr = callDataPtrMethod(obj);
  if (!data_ptr)
    return data_ptr;

  auto context = queuePtr->get_context();
  auto handle = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(context);
  ze_memory_allocation_properties_t prop;
  prop.stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
  prop.pNext = nullptr;
  auto res = ze().zeMemGetAllocProperties(
      (ze_context_handle_t)handle, data_ptr, &prop, nullptr);

  TORCH_CHECK(
      res == ZE_RESULT_SUCCESS,
      fmt::format(
          "Failed to get memory properties for pointer argument at {}-th argument, err={}",
          idx,
          static_cast<int>(res)));

  TORCH_CHECK(
      prop.type != ZE_MEMORY_TYPE_UNKNOWN,
      fmt::format(
          "Pointer argument references unknown type of memory at {}-th argument, err={}",
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
  Puts argument `item`, of the type `typeChar` names, into the 8-byte `slot`.
  Note that some args can be less than 8 bytes, but we'll still allocate 8
  bytes for them. `getPtr` resolves a pointer argument to a device pointer, so
  that a caller can choose how much it validates.
*/
template <typename PointerGetter>
void packKernelArg(
    char typeChar,
    void* slot,
    PyObject* item,
    int idx,
    const PointerGetter& getPtr) {
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
    case 'e':
      convertType<uint16_t>(
          torch::inductor::static_launcher::unpackTritonFp16,
          "float16",
          slot,
          item);
      break;
    case 'y':
      convertType<uint16_t>(
          torch::inductor::static_launcher::unpackTritonBf16,
          "bfloat16",
          slot,
          item);
      break;
    case 'f':
      convertType<float>(THPUtils_unpackDouble, "float", slot, item);
      break;
    case 'd':
      convertType<double>(THPUtils_unpackDouble, "double", slot, item);
      break;
    case 'O': // pointer
      *reinterpret_cast<syclDevicePtr_t*>(slot) = getPtr(item, idx);
      break;
    default:
      TORCH_CHECK(false, "Unknown type passed in: ", typeChar);
  }
}

void parseKernelArgs(
    PyObject* varArgs,
    const char* argTypes,
    uint64_t* argStorage,
    const sycl::queue* queuePtr) {
  int numKernelArgs = static_cast<int>(std::strlen(argTypes));
  TORCH_CHECK(
      PyTuple_Check(varArgs), "Kernel arguments must be provided as a tuple");
  TORCH_CHECK(
      PyTuple_Size(varArgs) == static_cast<Py_ssize_t>(numKernelArgs),
      "Mismatch between number of argument types and provided arguments");

  auto getPtr = [queuePtr](PyObject* item, int idx) {
    return getPointer(item, idx, queuePtr);
  };
  for (int i = 0; i < numKernelArgs; ++i) {
    packKernelArg(
        argTypes[i], &argStorage[i], PyTuple_GetItem(varArgs, i), i, getPtr);
  }
}

// A Level Zero handle owns its resource until make_kernel_bundle/make_kernel
// transfers it to SYCL. The ZE_CHECKs in between throw, and nothing else holds
// the handle, so the load path keeps one of these over that window.
struct ZeModuleDeleter {
  void operator()(ze_module_handle_t module) const {
    ze().zeModuleDestroy(module);
  }
};
struct ZeKernelDeleter {
  void operator()(ze_kernel_handle_t kernel) const {
    ze().zeKernelDestroy(kernel);
  }
};
using ZeModulePtr =
    std::unique_ptr<std::remove_pointer_t<ze_module_handle_t>, ZeModuleDeleter>;
using ZeKernelPtr =
    std::unique_ptr<std::remove_pointer_t<ze_kernel_handle_t>, ZeKernelDeleter>;

inline ZeModulePtr _createModule(
    const uint8_t* binaryPtr,
    size_t binarySize,
    const int device_idx) {
  auto& syclDevice =
      c10::xpu::get_raw_device(static_cast<c10::DeviceIndex>(device_idx));
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
  moduleDescription.pInputModule = binaryPtr;
  moduleDescription.pBuildFlags = buildFlags;
  ze_module_build_log_handle_t buildLog = nullptr;
  ze_module_handle_t rawModule = nullptr;
  ze_result_t error_no = ze().zeModuleCreate(
      context, device, &moduleDescription, &rawModule, &buildLog);
  ZeModulePtr module(rawModule);

  if (error_no != ZE_RESULT_SUCCESS) {
    // Retrieve the build log on a best-effort basis; failures here must not
    // mask the real build error reported via ZE_CHECK(error_no) below, and
    // must not skip the buildLog cleanup that follows.
    size_t szLog = 0;
    std::string strLog;
    if (ze().zeModuleBuildLogGetString(buildLog, &szLog, nullptr) ==
            ZE_RESULT_SUCCESS &&
        szLog > 0) {
      strLog.resize(szLog);
      ze().zeModuleBuildLogGetString(buildLog, &szLog, strLog.data());
    }
    std::cerr << "L0 build module failed. Log: " << strLog.c_str() << '\n';
  }
  if (buildLog) {
    ZE_CHECK(ze().zeModuleBuildLogDestroy(buildLog));
  }
  ZE_CHECK(error_no);
  return module;
}

inline LoadedKernel _createKernel(
    ZeModulePtr module,
    const char* kernelName,
    uint32_t* nSpillsPtr) {
  assert(module);
  assert(kernelName);
  auto& syclContext = c10::xpu::get_device_context();
  // Hand the module to SYCL before creating a kernel from it. zeModuleDestroy
  // requires every kernel handle created from the module to be destroyed first,
  // so the bundle has to be declared ahead of the kernel guard to be torn down
  // after it when anything below throws.
  ze_module_handle_t rawModule = module.get();
  auto mod = sycl::make_kernel_bundle<
      sycl::backend::ext_oneapi_level_zero,
      sycl::bundle_state::executable>(
      {rawModule, sycl::ext::oneapi::level_zero::ownership::transfer},
      syclContext);
  module.release();

  ze_kernel_handle_t rawKernel = nullptr;
  ze_kernel_desc_t kernelDescription = {};
  kernelDescription.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
  kernelDescription.pNext = nullptr;
  kernelDescription.flags = ZE_KERNEL_FLAG_FORCE_RESIDENCY;
  kernelDescription.pKernelName = kernelName;
  ZE_CHECK(ze().zeKernelCreate(rawModule, &kernelDescription, &rawKernel));
  ZeKernelPtr kernel(rawKernel);
  ze_kernel_properties_t props{};
  props.stype = ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES;
  props.pNext = nullptr;
  ZE_CHECK(ze().zeKernelGetProperties(kernel.get(), &props));
  // requiredSubgroupSize is the intel_reqd_sub_group_size attribute triton
  // always emits, and is exactly what the per-launch
  // sycl::info::kernel_device_specific::compile_sub_group_size query it
  // replaces returned, down to the default of 32 for a kernel that declares no
  // required size.
  const uint32_t requiredSimd = props.requiredSubgroupSize;
  const uint32_t threadsPerWarp = requiredSimd != 0 ? requiredSimd : 32;
  // Level Zero reports spillMemSize in bytes per hardware thread, while
  // CUDA/HIP report n_spills as LOCAL_SIZE_BYTES / 4, i.e. dword-equivalents
  // per lane. Inductor's spill thresholds are calibrated on the latter, so
  // normalize the same way the Intel triton driver does; otherwise the
  // statically launched kernel reports a value 4 * SIMD width too large.
  // maxSubgroupSize is only a best-effort stand-in here: Level Zero documents
  // it as a maximum, not as the compiled width. Truncating division matches
  // CUDA/HIP, so a scratch allocation below one dword per lane reports 0. With
  // an unknown SIMD width, report the raw bytes: over-reporting beats hiding an
  // allocation.
  const uint32_t spillSimd =
      requiredSimd != 0 ? requiredSimd : props.maxSubgroupSize;
  *nSpillsPtr = spillSimd != 0 ? props.spillMemSize / (4 * spillSimd)
                               : props.spillMemSize;
  auto& syclContext = c10::xpu::get_device_context();
  auto mod = sycl::make_kernel_bundle<
      sycl::backend::ext_oneapi_level_zero,
      sycl::bundle_state::executable>(
      {module.get(), sycl::ext::oneapi::level_zero::ownership::transfer},
      syclContext);
  module.release();
  auto fun = sycl::make_kernel<sycl::backend::ext_oneapi_level_zero>(
      {mod, kernel.get(), sycl::ext::oneapi::level_zero::ownership::transfer},
      syclContext);
  kernel.release();
  // numKernelArgs is the same value sycl::info::kernel::num_args reports.
  return LoadedKernel{std::move(fun), threadsPerWarp, props.numKernelArgs};
}

std::unique_ptr<LoadedKernel> loadKernel(
    const char* filePath,
    const char* funcName,
    uint32_t sharedMemBytes,
    uint32_t* nSpillsPtr,
    int device_idx) {
  std::ifstream IFS(filePath, std::ios::binary);
  std::ostringstream OSS;
  OSS << IFS.rdbuf();
  std::string data(std::move(OSS).str());
  auto mod = _createModule(
      reinterpret_cast<const uint8_t*>(data.c_str()), data.size(), device_idx);

  return std::make_unique<LoadedKernel>(
      _createKernel(std::move(mod), funcName, nSpillsPtr));
}

template <typename T>
void setKernelArg(sycl::handler& cgh, int index, const void* argStorage) {
  T value{};
  std::memcpy(&value, argStorage, sizeof(value));
  cgh.set_arg(index, value);
}

void setKernelArg(
    sycl::handler& cgh,
    int index,
    char typeChar,
    const void* argStorage) {
  switch (typeChar) {
    case 'b':
      setKernelArg<int8_t>(cgh, index, argStorage);
      break;
    case 'h':
      setKernelArg<int16_t>(cgh, index, argStorage);
      break;
    case 'i':
      setKernelArg<int32_t>(cgh, index, argStorage);
      break;
    case 'l':
      setKernelArg<int64_t>(cgh, index, argStorage);
      break;
    case 'B':
      setKernelArg<uint8_t>(cgh, index, argStorage);
      break;
    case 'H':
    case 'e':
    case 'y':
      setKernelArg<uint16_t>(cgh, index, argStorage);
      break;
    case 'I':
    case 'f':
      setKernelArg<uint32_t>(cgh, index, argStorage);
      break;
    case 'K':
    case 'd':
      setKernelArg<uint64_t>(cgh, index, argStorage);
      break;
    case 'O':
      setKernelArg<syclDevicePtr_t>(cgh, index, argStorage);
      break;
    default:
      TORCH_CHECK(false, "Unknown type passed in: ", typeChar);
  }
}

/**
  Submits the kernel with the arguments packed in `argStorage`. `numParams` is
  the number of those arguments, and `bindLocal` says whether the kernel takes a
  trailing dynamic shared memory argument.

  Neither is checked here, so a caller must guarantee:
    - numParams == strlen(argTypes), and argStorage has numParams slots. It may
      only be null when numParams is 0.
    - bindLocal only if the kernel's own argument count is numParams + 1.
      Otherwise set_arg(numParams, ...) binds an argument it does not have.
  The kernel argument count check in launchKernel() is what establishes the
  second one today.

  `argTypes` sits between the two counts, and `bindLocal` after the queue, so
  that swapping either for its neighbour does not compile.
*/
void submitKernel(
    const LoadedKernel* loadedKernel,
    uint32_t gridX,
    uint32_t gridY,
    uint32_t gridZ,
    uint32_t numWarps,
    uint32_t sharedMemBytes,
    const char* argTypes,
    uint32_t numParams,
    const uint64_t* argStorage,
    sycl::queue* queuePtr,
    bool bindLocal) {
  size_t localRangeX =
      static_cast<size_t>(numWarps) * loadedKernel->threadsPerWarp;
  sycl::range<3> globalRange(gridZ, gridY, gridX * localRangeX);
  sycl::range<3> localRange(1, 1, localRangeX);
  sycl::nd_range<3> parallelWorkSize(globalRange, localRange);
  auto cgf = [&](sycl::handler& cgh) {
    for (uint32_t i = 0; i < numParams; ++i) {
      setKernelArg(cgh, static_cast<int>(i), argTypes[i], &argStorage[i]);
    }

    if (bindLocal) {
      using share_mem_t = sycl::local_accessor<int8_t, 1>;
      share_mem_t localBuffer = share_mem_t(sharedMemBytes, cgh);
      cgh.set_arg(static_cast<int>(numParams), localBuffer);
      cgh.parallel_for(parallelWorkSize, loadedKernel->kernel);
    } else {
      cgh.parallel_for(parallelWorkSize, loadedKernel->kernel);
    }
  };
  queuePtr->submit(cgf);
}

void launchKernel(
    const LoadedKernel* loadedKernel,
    uint32_t gridX,
    uint32_t gridY,
    uint32_t gridZ,
    uint32_t numWarps,
    uint32_t sharedMemBytes,
    const char* argTypes,
    const uint64_t* argStorage,
    sycl::queue* queuePtr) {
  const uint32_t kernelNumArgs = loadedKernel->numArgs;
  const uint32_t numParams = static_cast<uint32_t>(std::strlen(argTypes));
  const bool bindLocal = sharedMemBytes > 0 && kernelNumArgs == numParams + 1;
  TORCH_CHECK(
      kernelNumArgs == numParams + (bindLocal ? 1 : 0),
      "Kernel argument count mismatch: kernel expects ",
      kernelNumArgs,
      " arguments, launcher has ",
      numParams,
      " scalar/pointer arguments");
  submitKernel(
      loadedKernel,
      gridX,
      gridY,
      gridZ,
      numWarps,
      sharedMemBytes,
      argTypes,
      numParams,
      argStorage,
      queuePtr,
      bindLocal);
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
  std::unique_ptr<LoadedKernel> func =
      loadKernel(filePath, funcName, sharedMemBytes, &n_spills, device);

  PyObject* kernel_py = PyCapsule_New(
      reinterpret_cast<void*>(func.get()),
      "static_xpu_kernel",
      [](PyObject* cap) {
        void* ptr = PyCapsule_GetPointer(cap, "static_xpu_kernel");
        delete reinterpret_cast<LoadedKernel*>(ptr);
      });
  if (kernel_py == nullptr) {
    return nullptr;
  }
  func.release();

  // "N": the tuple takes ownership of the capsule reference.
  return Py_BuildValue("(Nii)", kernel_py, n_regs, n_spills);
  END_HANDLE_TH_ERRORS
}

PyObject* launch_kernel_inner(
    const LoadedKernel* func,
    int gridX,
    int gridY,
    int gridZ,
    int numWarps,
    int sharedMemBytes,
    const char* argTypes,
    PyObject* varArgs,
    sycl::queue* queuePtr) {
  // Launch the kernel
  // Prepare the arguments for the kernel: 8 bytes per argument on the stack,
  // which is what launchKernel reads them from.
  std::array<uint64_t, MAX_ARGS> argStorage = {};
  parseKernelArgs(varArgs, argTypes, argStorage.data(), queuePtr);
  launchKernel(
      func,
      gridX,
      gridY,
      gridZ,
      numWarps,
      sharedMemBytes,
      argTypes,
      argStorage.data(),
      queuePtr);

  Py_RETURN_NONE;
}

PyObject* launch_kernel_slow(
    const LoadedKernel* func,
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

  parseKernelArgs(varArgs, argTypes, argStorage.data(), queuePtr);

  launchKernel(
      func,
      gridX,
      gridY,
      gridZ,
      numWarps,
      sharedMemBytes,
      argTypes,
      argStorage.data(),
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

  const LoadedKernel* func = reinterpret_cast<const LoadedKernel*>(
      PyCapsule_GetPointer(kernel_py, "static_xpu_kernel")); // NOLINT
  if (func == nullptr) {
    // PyCapsule_GetPointer set a Python error; report it as a TORCH_CHECK
    // failure rather than dereferencing null below.
    PyErr_Clear();
    TORCH_CHECK(false, "_launch_kernel was not passed a loaded XPU kernel");
  }
  sycl::queue* queuePtr = reinterpret_cast<sycl::queue*>(stream); // NOLINT
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
        argTypes,
        nullptr,
        queuePtr);
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
  if (PyModule_AddType(module, &StaticXpuLauncherType) < 0) {
    return false;
  }
  return true;
}
#endif
#endif
