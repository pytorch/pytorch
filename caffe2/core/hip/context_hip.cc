#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <string>
#include <unordered_map>

#include "cub/util_allocator.cuh"
#include "caffe2/core/asan.h"
#include "caffe2/core/hip/common_miopen.h"
#include "caffe2/core/hip/context_hip.h"
#include "caffe2/core/init.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/tensor.h"
#include "caffe2/utils/string_utils.h"

CAFFE2_DEFINE_string(caffe2_hip_memory_pool,
                     "",
                     "Sets the memory pool used by caffe2. Possible values are "
                     "none, cnmen and cub.");

// For description of CUB caching allocator configuration, see
// https://nvlabs.github.io/cub/structcub_1_1_caching_device_allocator.html
CAFFE2_DEFINE_int(caffe2_cub_bin_growth,
                  8,
                  "If using cub as the memory allocator, sets the growth of bins "
                  "used by the cub pool.");
CAFFE2_DEFINE_int(caffe2_cub_min_bin,
                  3,
                  "If using cub as the memory allocator, sets the min number of "
                  "bins.");
CAFFE2_DEFINE_int(caffe2_cub_max_bin,
                  10,
                  "If using cub as the memory allocator, sets the max number of "
                  "bins.");
CAFFE2_DEFINE_int(caffe2_cub_max_managed_mb,
                  10 * 1024,
                  "If using cub as the memory allocators, sets the maximum amount "
                  "of memory managed in gigabytes");
CAFFE2_DEFINE_bool(caffe2_cub_print_allocation_events,
                   false,
                   "If true CachingDeviceAllocator will print allocation and deallocation "
                   "events to stdout.");

CAFFE2_DEFINE_bool(caffe2_gpu_memory_tracking,
                   false,
                   "If set, logs changes in GPU memory allocations");
CAFFE2_DEFINE_int(caffe2_gpu_memory_report_interval_mb,
                  128,
                  "The threshold in MB on how frequently to report memory changes");

namespace caffe2 {

CAFFE_KNOWN_TYPE(Tensor<HIPContext>);

thread_local ThreadLocalHIPObjects HIPContext::hip_objects_;

// TODO(jiayq): these variables shouldn't be currently accessed during static
// initialization. We should consider moving them to a Mayer's singleton to
// be totally safe against SIOF.

// Static global variables for setting up the memory pool.
HipMemoryPoolType g_hip_memory_pool_type;

// For cub allocator
unique_ptr<cub::CachingDeviceAllocator> g_cub_allocator;
// an unordered map that holds the map from the cuda memory pointer to the
// device id that it is allocated from. This is used in the cuda memory pool
// cases, where we need the device id to carry out the deletion.
// Note(jiayq): an alternate approach is to use cudaGetPointerAttributes, but
// that is usually quite slow. We might want to benchmark the speed difference
// though.
// Note(jiayq): another alternate approach is to augment the Tensor class that
// would allow one to record the device id. However, this does not address any
// non-tensor allocation and deallocation.
// Ideally, a memory pool should already have the device id information, as
// long as we are using UVA (as of CUDA 5 and later) so the addresses are
// unique.
static std::unordered_map<void*, uint8_t> g_hip_device_affiliation;

// Data structures for optional memory tracking. Access to these structures
// is garded by the HIPContext::mutex.
static std::unordered_map<void*, long> g_size_map;
static std::vector<long> g_total_by_gpu_map(CAFFE2_COMPILE_TIME_MAX_HIP_GPUS, 0);
static std::vector<long> g_max_by_gpu_map(CAFFE2_COMPILE_TIME_MAX_HIP_GPUS, 0);

static long g_total_mem = 0;
static long g_last_rep  = 0;

HipMemoryPoolType GetHipMemoryPoolType() { return g_hip_memory_pool_type; }

vector<TIndex>
GetHipTensorInfo(const void* c, bool* shares_data, size_t* capacity, DeviceOption* device)
{
    vector<TIndex> dims          = GetTensorInfo<HIPContext>(c, shares_data, capacity, device);
    const Tensor<HIPContext>* tc = static_cast<const Tensor<HIPContext>*>(c);
    device->set_device_type(HIP);
    device->set_hip_gpu_id(GetGPUIDForPointer(tc->raw_data()));
    return dims;
}

///////////////////////////////////////////////////////////////////////////////
// A wrapper to allow us to lazily initialize all HIP environments that Caffe
// uses. This gets done the first time a caffe2::HIPContext::New() gets called
// which is probably the decisive indication that this caffe2 run is going to
// use GPUs. We avoid HIP initialization with core/init.h functionalities so
// that we have minimal resource impact in case we will need to run multiple
// caffe2 instances on a GPU machine.
///////////////////////////////////////////////////////////////////////////////

static void Caffe2InitializeHip()
{
    // If the current run does not have any HIP devices, do nothing.
    if(!HasHipGPU())
    {
        VLOG(1) << "No HIP gpu present. Skipping.";
        return;
    }
    // Check if the number of GPUs matches the expected compile-time max number
    // of GPUs.
    CAFFE_ENFORCE_LE(NumHipDevices(),
                     CAFFE2_COMPILE_TIME_MAX_HIP_GPUS,
                     "Number of HIP devices on the machine is larger than the compiled "
                     "max number of gpus expected (",
                     CAFFE2_COMPILE_TIME_MAX_HIP_GPUS,
                     "). Increase that and recompile the caffe binary.");

    for(int i = 0; i < NumHipDevices(); ++i)
    {
        DeviceGuard g(i);
        // Enable peer access.
        const int peer_group = i / CAFFE2_HIP_MAX_PEER_SIZE;
        const int peer_start = peer_group * CAFFE2_HIP_MAX_PEER_SIZE;
        const int peer_end = std::min(NumHipDevices(), (peer_group + 1) * CAFFE2_HIP_MAX_PEER_SIZE);
        VLOG(1) << "Enabling peer access within group #" << peer_group << ", from gpuid "
                << peer_start << " to " << peer_end - 1 << ", for gpuid " << i << ".";

        for(int j = peer_start; j < peer_end; ++j)
        {
            if(i == j)
                continue;
            int can_access;
            HIP_ENFORCE(hipDeviceCanAccessPeer(&can_access, i, j));
            if(can_access)
            {
                VLOG(1) << "Enabling peer access from " << i << " to " << j;
                // Note: just for future reference, the 0 here is not a gpu id, it is
                // a reserved flag for hipDeviceEnablePeerAccess that should always be
                // zero currently.
                HIP_ENFORCE(hipDeviceEnablePeerAccess(j, 0));
            }
        }
    }

    RegisterTypeCallFunction(TypeMeta::Id<Tensor<HIPContext>>(), GetTensorType<HIPContext>);

    RegisterTensorInfoFunction(TypeMeta::Id<Tensor<HIPContext>>(), GetHipTensorInfo);

    // CheckMiOpenVersions();
}

static void SetUpCub()
{
    VLOG(1) << "Setting up cub memory pool.";
    // Sets up the cub memory pool
    try
    {
        g_cub_allocator.reset(
            new cub::CachingDeviceAllocator(FLAGS_caffe2_cub_bin_growth,
                                            FLAGS_caffe2_cub_min_bin,
                                            FLAGS_caffe2_cub_max_bin,
                                            size_t(FLAGS_caffe2_cub_max_managed_mb) * 1024L * 1024L,
                                            false,
                                            FLAGS_caffe2_cub_print_allocation_events));
    }
    catch(...)
    {
        CAFFE_THROW("Some error happened at cub initialization.");
    }
    VLOG(1) << "Done setting up cub memory pool.";
}

static void Caffe2SetHIPMemoryPool()
{
    if(FLAGS_caffe2_hip_memory_pool == "" || FLAGS_caffe2_hip_memory_pool == "none")
    {
        g_hip_memory_pool_type = HipMemoryPoolType::NONE;
    }
    else if(FLAGS_caffe2_hip_memory_pool == "cnmem")
    {
        CAFFE_THROW("CNMEM is no longer used by Caffe2. Use cub instead. "
                    "This error message may go away in the future.");
    }
    else if(FLAGS_caffe2_hip_memory_pool == "cub")
    {
        // Sets up cub.
        g_hip_memory_pool_type = HipMemoryPoolType::CUB;
        SetUpCub();
    }
    else
    {
        CAFFE_THROW("Unrecognized HIP memory pool type: ", FLAGS_caffe2_hip_memory_pool);
    }
}

// An initialization function that sets the CPU side to use pinned cpu
// allocator.
void Caffe2UsePinnedCPUAllocator()
{
#if CAFFE2_ASAN_ENABLED
    // Note(jiayq): for more details, see
    //     https://github.com/google/sanitizers/issues/629
    LOG(WARNING) << "There are known issues between address sanitizer and "
                    "hipHostMalloc. As a result, caffe2 will not enable pinned "
                    "memory allocation in asan mode. If you are expecting any "
                    "behavior that depends on asan, be advised that it is not "
                    "turned on.";
#else
    if(!HasHipGPU())
    {
        VLOG(1) << "No GPU present. I won't use pinned allocator then.";
        return;
    }
    VLOG(1) << "Caffe2 gpu: setting CPUAllocator to PinnedCPUAllocator.";
    SetCPUAllocator(new PinnedCPUAllocator());
#endif
}

// Caffe2HipInitializerHelper is a minimal struct whose sole purpose is to
// detect the first hint that this Caffe2 run is going to use GPU: either
// HIPContext is initialized or HIPContext::New is called. It then runs
// all the related cuda initialization functions.
namespace {
struct Caffe2HipInitializerHelper
{
    Caffe2HipInitializerHelper()
    {
        // We cannot use bool because nvcc changes bool to __nv_bool which does
        // not have a std::atomic instantiation.
        static std::atomic<char> first_call(1);
        if(first_call.fetch_and((char)0))
        {
            Caffe2InitializeHip();
            Caffe2SetHIPMemoryPool();
            Caffe2UsePinnedCPUAllocator();
        }
    }
};
} // namespace

/**
 * A utility function to rectify the gpu id. If the context specifies the
 * gpu id to be -1, it means that we will just use the current gpu id when
 * the function is being called.
 */
static inline int RectifyGPUID(const int gpu_id) {
  return gpu_id == -1 ? CaffeHipGetDevice() : gpu_id;
}

HIPContext::HIPContext(const int gpu_id)
    : gpu_id_(RectifyGPUID(gpu_id)), random_seed_(RandomNumberSeed()) {
  static Caffe2HipInitializerHelper g_hip_initializer_;
}

HIPContext::HIPContext(const DeviceOption& option)
    : gpu_id_(
          option.has_hip_gpu_id() ? RectifyGPUID(option.hip_gpu_id())
                                  : CaffeHipGetDevice()),
      random_seed_(
          option.has_random_seed() ? option.random_seed()
                                   : RandomNumberSeed()) {
  static Caffe2HipInitializerHelper g_hip_initializer_;
  DCHECK_EQ(option.device_type(), HIP);
}

// shared mutex to lock out alloc / free during NCCL launches
std::mutex& HIPContext::mutex()
{
    static std::mutex m;
    return m;
}

std::vector<long> HIPContext::TotalMemoryByGpu()
{
    std::lock_guard<std::mutex> lock(HIPContext::mutex());
    CAFFE_ENFORCE(FLAGS_caffe2_gpu_memory_tracking,
                  "Pass --caffe2_gpu_memory_tracking to enable memory stats");
    return g_total_by_gpu_map;
}

std::vector<long> HIPContext::MaxMemoryByGpu()
{
    std::lock_guard<std::mutex> lock(HIPContext::mutex());
    CAFFE_ENFORCE(FLAGS_caffe2_gpu_memory_tracking,
                  "Pass --caffe2_gpu_memory_tracking to enable memory stats");
    return g_max_by_gpu_map;
}

namespace {
void TrackMemoryAlloc(size_t nbytes)
{
    int this_gpu = CaffeHipGetDevice();
    g_total_by_gpu_map[this_gpu] += nbytes;
    g_max_by_gpu_map[this_gpu] = std::max(g_max_by_gpu_map[this_gpu], g_total_by_gpu_map[this_gpu]);
    g_total_mem += nbytes;
    if(g_total_mem - g_last_rep > FLAGS_caffe2_gpu_memory_report_interval_mb * 1024 * 1024)
    {
        for(int gpu = 0; gpu < g_total_by_gpu_map.size(); gpu++)
        {
            long t     = g_total_by_gpu_map[gpu];
            long max_t = g_max_by_gpu_map[gpu];
            if(max_t > 0)
            {
                if(max_t != t)
                {
                    LOG(INFO) << "GPU " << gpu << ": " << t / 1024 / 1024 << " MB"
                              << " (max: " << max_t / 1024 / 1024 << " MB)";
                }
                else
                {
                    LOG(INFO) << "GPU " << gpu << ": " << t / 1024 / 1024 << " MB";
                }
            }
        }
        LOG(INFO) << "Total: " << g_total_mem / 1024 / 1024 << " MB";
        g_last_rep = g_total_mem;
    }
}
}

std::pair<void*, MemoryDeleter> HIPContext::New(size_t nbytes)
{
    // Lock the mutex
    std::lock_guard<std::mutex> lock(HIPContext::mutex());
    // A one-time caffe2 cuda initializer.
    static Caffe2HipInitializerHelper g_hip_initializer_;
    void* ptr = nullptr;

    if(FLAGS_caffe2_gpu_memory_tracking)
    {
        TrackMemoryAlloc(nbytes);
    }
    switch(g_hip_memory_pool_type)
    {
    case HipMemoryPoolType::NONE:
        HIP_ENFORCE(hipMalloc(&ptr, nbytes));
        if(FLAGS_caffe2_gpu_memory_tracking)
        {
            g_size_map[ptr]               = nbytes;
            g_hip_device_affiliation[ptr] = CaffeHipGetDevice();
        }
        return {ptr, Delete};
    case HipMemoryPoolType::CUB:
        HIP_ENFORCE(g_cub_allocator->DeviceAllocate(&ptr, nbytes));
        g_hip_device_affiliation[ptr] = CaffeHipGetDevice();
        VLOG(2) << "CUB allocating pointer " << ptr << " on device " << CaffeHipGetDevice();
        if(FLAGS_caffe2_gpu_memory_tracking)
        {
            g_size_map[ptr] = nbytes;
        }
        return {ptr, Delete};
    }
    return {nullptr, Delete};
}

void HIPContext::Delete(void* ptr)
{
    // lock the mutex
    std::lock_guard<std::mutex> lock(HIPContext::mutex());

    if(FLAGS_caffe2_gpu_memory_tracking)
    {
        auto sz_it = g_size_map.find(ptr);
        DCHECK(sz_it != g_size_map.end());
        auto aff_it = g_hip_device_affiliation.find(ptr);
        DCHECK(aff_it != g_hip_device_affiliation.end());
        g_total_mem -= sz_it->second;
        g_total_by_gpu_map[aff_it->second] -= sz_it->second;
        g_size_map.erase(sz_it);
    }

    switch(g_hip_memory_pool_type)
    {
    case HipMemoryPoolType::NONE:
    {
        // If memory pool is not set up, use simple hipFree.
        hipError_t error = hipFree(ptr);
        // For some reason, in Python runtime we sometimes delete a data pointer
        // after the cuda runtime exits - this is odd but is probably caused by
        // a static workspace that pycaffe2 uses, and the destruction got
        // entangled in some race condition. Anyway, since cuda runtime is exiting
        // anyway, we will not need to worry about memory leak, so we basically
        // ignore it. This is definitely not ideal but works for now.
        if(error != hipSuccess)
        {
            LOG(FATAL) << "Error at: " << __FILE__ << ":" << __LINE__ << ": "
                       << hipGetErrorString(error);
        }

        if(FLAGS_caffe2_gpu_memory_tracking)
        {
            g_hip_device_affiliation.erase(g_hip_device_affiliation.find(ptr));
        }

        break;
    }
    case HipMemoryPoolType::CUB:
    {
        auto it = g_hip_device_affiliation.find(ptr);
        DCHECK(it != g_hip_device_affiliation.end());
        VLOG(2) << "CUB freeing pointer " << ptr << " on device " << it->second;
        HIP_ENFORCE(g_cub_allocator->DeviceFree(it->second, ptr));
        g_hip_device_affiliation.erase(it);
        break;
    }
    }
}

} // namespace caffe2
