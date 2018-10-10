#include "caffe2/core/hip/common_hip.h"
#include "caffe2/core/asan.h"

#include <atomic>
#include <cstdlib>
#include <sstream>

#include "caffe2/core/init.h"
#include "caffe2/core/logging.h"

C10_DEFINE_bool(
    caffe2_hip_full_device_control,
    false,
    "If true, assume all the hipSetDevice and hipGetDevice calls will be "
    "controlled by Caffe2, and non-Caffe2 code will ensure that the entry and "
    "exit point has the same cuda device. Under the hood, Caffe2 will use "
    "thread local variables to cache the device, in order to speed up set and "
    "get device calls. This is an experimental feature that may have non "
    "trivial side effects, so use it with care and only enable it if you are "
    "absolutely sure. Also, this flag should not be changed after the program "
    "initializes.");

namespace caffe2 {

int NumHipDevices()
{
    if(getenv("CAFFE2_DEBUG_HIP_INIT_ORDER"))
    {
        static bool first = true;
        if(first)
        {
            first = false;
            std::cerr << "DEBUG: caffe2::NumHipDevices() invoked for the first time" << std::endl;
        }
    }
    static int count = -1;
    if(count < 0)
    {
        auto err = hipGetDeviceCount(&count);
        switch(err)
        {
        case hipSuccess:
            // Everything is good.
            break;
        case hipErrorNoDevice: count = 0; break;
        case hipErrorInsufficientDriver:
            LOG(WARNING) << "Insufficient HIP driver. Cannot use HIP.";
            count = 0;
            break;
        case hipErrorInitializationError:
            LOG(WARNING) << "HIP driver initialization failed, you might not "
                            "have a HIP gpu.";
            count = 0;
            break;
        case hipErrorUnknown:
            LOG(ERROR) << "Found an unknown error - this may be due to an "
                          "incorrectly set up environment, e.g. changing env "
                          "variable HIP_VISIBLE_DEVICES after program start. "
                          "I will set the available devices to be zero.";
            count = 0;
            break;
        case hipErrorMemoryAllocation:
#if CAFFE2_ASAN_ENABLED
            // In ASAN mode, we know that a hipErrorMemoryAllocation error will
            // pop up.
            LOG(ERROR) << "It is known that HIP does not work well with ASAN. As "
                          "a result we will simply shut down HIP support. If you "
                          "would like to use GPUs, turn off ASAN.";
            count = 0;
            break;
#else  // CAFFE2_ASAN_ENABLED
            // If we are not in ASAN mode and we get hipErrorMemoryAllocation,
            // this means that something is wrong before NumCudaDevices() call.
            LOG(FATAL) << "Unexpected error from hipGetDeviceCount(). Did you run "
                          "some HIP functions before calling NumHipDevices() "
                          "that might have already set an error? Error: "
                       << err;
            break;
#endif // CAFFE2_ASAN_ENABLED
        default:
            LOG(FATAL) << "Unexpected error from hipGetDeviceCount(). Did you run "
                          "some HIP functions before calling NumHipDevices() "
                          "that might have already set an error? Error: "
                       << err;
        }
    }
    return count;
}

namespace {
int gDefaultGPUID = 0;
// Only used when c10::FLAGS_caffe2_hip_full_device_control is set true.
thread_local int gCurrentDevice = -1;
} // namespace

void SetDefaultGPUID(const int deviceid)
{
    CAFFE_ENFORCE_LT(deviceid,
                     NumHipDevices(),
                     "The default gpu id should be smaller than the number of gpus "
                     "on this machine: ",
                     deviceid,
                     " vs ",
                     NumHipDevices());
    gDefaultGPUID = deviceid;
}

int GetDefaultGPUID() { return gDefaultGPUID; }

int CaffeHipGetDevice()
{
  if (c10::FLAGS_caffe2_hip_full_device_control) {
    if (gCurrentDevice < 0) {
      HIP_ENFORCE(hipGetDevice(&gCurrentDevice));
    }
    return gCurrentDevice;
  } else {
    int gpu_id = 0;
    HIP_ENFORCE(hipGetDevice(&gpu_id));
    return gpu_id;
  }
}

void CaffeHipSetDevice(const int id)
{
  if (c10::FLAGS_caffe2_hip_full_device_control) {
    if (gCurrentDevice != id) {
      HIP_ENFORCE(hipSetDevice(id));
    }
    gCurrentDevice = id;
  } else {
    HIP_ENFORCE(hipSetDevice(id));
  }
}

int GetGPUIDForPointer(const void* ptr)
{
    hipPointerAttribute_t attr;
    hipError_t err = hipPointerGetAttributes(&attr, ptr);

    if(err == hipErrorInvalidValue)
    {
        // Occurs when the pointer is in the CPU address space that is
        // unmanaged by HIP; make sure the last error state is cleared,
        // since it is persistent
        err = hipGetLastError();
        CHECK(err == hipErrorInvalidValue);
        return -1;
    }

    // Otherwise, there must be no error
    HIP_ENFORCE(err);

    if(attr.memoryType == hipMemoryTypeHost)
    {
        return -1;
    }

    return attr.device;
}

struct HipDevicePropWrapper
{
    HipDevicePropWrapper() : props(NumHipDevices())
    {
        for(int i = 0; i < NumHipDevices(); ++i)
        {
            HIP_ENFORCE(hipGetDeviceProperties(&props[i], i));
        }
    }

    vector<hipDeviceProp_t> props;
};

const hipDeviceProp_t& GetDeviceProperty(const int deviceid)
{
    // According to C++11 standard section 6.7, static local variable init is
    // thread safe. See
    //   https://stackoverflow.com/questions/8102125/is-local-static-variable-initialization-thread-safe-in-c11
    // for details.
    static HipDevicePropWrapper props;
    CAFFE_ENFORCE_LT(deviceid,
                     NumHipDevices(),
                     "The gpu id should be smaller than the number of gpus ",
                     "on this machine: ",
                     deviceid,
                     " vs ",
                     NumHipDevices());
    return props.props[deviceid];
}

void DeviceQuery(const int device)
{
    const hipDeviceProp_t& prop = GetDeviceProperty(device);
    std::stringstream ss;
    ss << std::endl;
    ss << "Device id:                     " << device << std::endl;
    ss << "Major revision number:         " << prop.major << std::endl;
    ss << "Minor revision number:         " << prop.minor << std::endl;
    ss << "Name:                          " << prop.name << std::endl;
    ss << "Total global memory:           " << prop.totalGlobalMem << std::endl;
    ss << "Total shared memory per block: " << prop.sharedMemPerBlock << std::endl;
    ss << "Total registers per block:     " << prop.regsPerBlock << std::endl;
    ss << "Warp size:                     " << prop.warpSize << std::endl;
    //  ss << "Maximum memory pitch:          " << prop.memPitch << std::endl;
    ss << "Maximum threads per block:     " << prop.maxThreadsPerBlock << std::endl;
    ss << "Maximum dimension of block:    " << prop.maxThreadsDim[0] << ", "
       << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << std::endl;
    ss << "Maximum dimension of grid:     " << prop.maxGridSize[0] << ", " << prop.maxGridSize[1]
       << ", " << prop.maxGridSize[2] << std::endl;
    ss << "Clock rate:                    " << prop.clockRate << std::endl;
    ss << "Total constant memory:         " << prop.totalConstMem << std::endl;
    //  ss << "Texture alignment:             " << prop.textureAlignment << std::endl;
    //  ss << "Concurrent copy and execution: "
    //     << (prop.deviceOverlap ? "Yes" : "No") << std::endl;
    ss << "Number of multiprocessors:     " << prop.multiProcessorCount << std::endl;
    //  ss << "Kernel execution timeout:      "
    //     << (prop.kernelExecTimeoutEnabled ? "Yes" : "No") << std::endl;
    LOG(INFO) << ss.str();
    return;
}

bool GetHipPeerAccessPattern(vector<vector<bool>>* pattern)
{
    int gpu_count;
    if(hipGetDeviceCount(&gpu_count) != hipSuccess)
        return false;
    pattern->clear();
    pattern->resize(gpu_count, vector<bool>(gpu_count, false));
    for(int i = 0; i < gpu_count; ++i)
    {
        for(int j = 0; j < gpu_count; ++j)
        {
            int can_access = true;
            if(i != j)
            {
                if(hipDeviceCanAccessPeer(&can_access, i, j) != hipSuccess)
                {
                    return false;
                }
            }
            (*pattern)[i][j] = static_cast<bool>(can_access);
        }
    }
    return true;
}

const char* rocblasGetErrorString(rocblas_status error)
{
    switch(error)
    {
    case rocblas_status_success: return "rocblas_status_success";
    case rocblas_status_invalid_handle: return "rocblas_status_invalid_handle";
    case rocblas_status_not_implemented: return "rocblas_status_not_implemented";
    case rocblas_status_invalid_pointer: return "rocblas_status_invalid_pointer";
    case rocblas_status_invalid_size: return "rocblas_status_invalid_size";
    case rocblas_status_memory_error: return "rocblas_status_memory_error";
    case rocblas_status_internal_error: return "rocblas_status_internal_error";
    }
    // To suppress compiler warning.
    return "Unrecognized rocblas error string";
}

const char* hiprandGetErrorString(hiprandStatus_t error)
{
    switch(error)
    {
    case HIPRAND_STATUS_SUCCESS: return "HIPRAND_STATUS_SUCCESS";
    case HIPRAND_STATUS_VERSION_MISMATCH: return "HIPRAND_STATUS_VERSION_MISMATCH";
    case HIPRAND_STATUS_NOT_INITIALIZED: return "HIPRAND_STATUS_NOT_INITIALIZED";
    case HIPRAND_STATUS_ALLOCATION_FAILED: return "HIPRAND_STATUS_ALLOCATION_FAILED";
    case HIPRAND_STATUS_TYPE_ERROR: return "HIPRAND_STATUS_TYPE_ERROR";
    case HIPRAND_STATUS_OUT_OF_RANGE: return "HIPRAND_STATUS_OUT_OF_RANGE";
    case HIPRAND_STATUS_LENGTH_NOT_MULTIPLE: return "HIPRAND_STATUS_LENGTH_NOT_MULTIPLE";
    case HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED:
        return "HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED";
    case HIPRAND_STATUS_LAUNCH_FAILURE: return "HIPRAND_STATUS_LAUNCH_FAILURE";
    case HIPRAND_STATUS_PREEXISTING_FAILURE: return "HIPRAND_STATUS_PREEXISTING_FAILURE";
    case HIPRAND_STATUS_INITIALIZATION_FAILED: return "HIPRAND_STATUS_INITIALIZATION_FAILED";
    case HIPRAND_STATUS_ARCH_MISMATCH: return "HIPRAND_STATUS_ARCH_MISMATCH";
    case HIPRAND_STATUS_INTERNAL_ERROR: return "HIPRAND_STATUS_INTERNAL_ERROR";
    case HIPRAND_STATUS_NOT_IMPLEMENTED: return "HIPRAND_STATUS_NOT_IMPLEMENTED";
    }
    // To suppress compiler warning.
    return "Unrecognized HIPRAND error string";
}

// Turn on the flag g_caffe2_has_hip_linked to true for HasHipRuntime()
// function.
extern bool g_caffe2_has_hip_linked;
namespace {
class HipRuntimeFlagFlipper
{
    public:
     HipRuntimeFlagFlipper() {
       internal::SetHipRuntimeFlag();
     }
};
static HipRuntimeFlagFlipper g_flipper;
} // namespace

} // namespace caffe2
