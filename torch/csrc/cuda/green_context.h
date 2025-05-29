#pragma once
#include <c10/cuda/driver_api.h>
#include <cuda.h>
#include <memory>
#include <stdexcept>
#include <vector>

class GreenContext {
public:
    GreenContext(int device_id, unsigned int num_sms) {
        CUdevice device;
        C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuDeviceGet_(&device, device_id));

        // Get device resources
        CUdevResource device_resource;
        C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuDeviceGetDevResource_(device, &device_resource, CU_DEV_RESOURCE_TYPE_SM));

        // Split resources
        std::vector<CUdevResource> result(1);
        auto result_data = result.data();
        unsigned int nb_groups = 1;
        CUdevResource remaining;
        
        C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuDevSmResourceSplitByCount_(
            result_data,
            &nb_groups,
            &device_resource,
            &remaining,
            0,  // default flags
            num_sms
        ));

        if (nb_groups != 1) {
            throw std::runtime_error("Failed to create single resource group");
        }

        // Generate resource descriptor
        CUdevResourceDesc desc;
        C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuDevResourceGenerateDesc_(&desc, result_data, 1));

        // Create green context
        C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuGreenCtxCreate_(&green_ctx_, desc, device, CU_GREEN_CTX_DEFAULT_STREAM));

        // Convert to regular context
        C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuCtxFromGreenCtx_(&context_, green_ctx_));
        TORCH_INTERNAL_ASSERT(context_, "Green ctx conversion to regular ctx failed!");
        TORCH_WARN("FINISH CONVERT");
    }

    static std::unique_ptr<GreenContext> create(int device_id, unsigned int num_sms) {
        return std::make_unique<GreenContext>(device_id, num_sms);
    }

    // Delete copy constructor and assignment
    GreenContext(const GreenContext&) = delete;
    GreenContext& operator=(const GreenContext&) = delete;

    // Implement move operations
    GreenContext(GreenContext&& other) noexcept 
        : green_ctx_(other.green_ctx_)
        , context_(other.context_)
        , parent_stream_(other.parent_stream_) {
        other.green_ctx_ = nullptr;
        other.context_ = nullptr;
        other.parent_stream_ = NULL;
    }

    GreenContext& operator=(GreenContext&& other) noexcept {
        if (this != &other) {
            // Clean up current resources
            if (green_ctx_) {
                CUresult result = cuGreenCtxDestroy(green_ctx_);
                if (result != CUDA_SUCCESS) {
                    fprintf(stderr, "Failed to destroy green context during move: %d\n", result);
                }
            }
            
            // Take ownership of other's resources
            green_ctx_ = other.green_ctx_;
            context_ = other.context_;
            parent_stream_ = other.parent_stream_;
            
            // Null out other's pointers
            other.green_ctx_ = nullptr;
            other.context_ = nullptr;
            other.parent_stream_ = NULL;
        }
        return *this;
    }

    ~GreenContext() noexcept {
        C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuGreenCtxDestroy_(green_ctx_));
    }

    // Get the underlying CUDA context
    CUcontext getContext() const { return context_; }
    
    // Get the underlying green context
    CUgreenCtx getGreenContext() const { return green_ctx_; }

    // Make this context current
    void makeCurrent() {
        auto current_stream = c10::cuda::getCurrentCUDAStream();
        parent_stream_ = current_stream.stream();
        cudaEvent_t ev;
        C10_CUDA_CHECK(cudaEventCreate(&ev));
        C10_CUDA_CHECK(cudaEventRecord(ev, current_stream));
        TORCH_WARN("STARTING PUSH");
        CUcontext current = nullptr;
        uint32_t version;
        C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuCtxGetCurrent_(&current));
        if (!current) {
          C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuCtxSetCurrent_(context_));
        } else {

          C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuCtxPushCurrent_(context_));
        }
        TORCH_WARN("FINISH PUSH");
        // currently hardcoes the new green context to use the default stream
        // TODO(eqy): consider creating a new stream if e.g., it allows interop
        // with CUDA Graph captures etc.
        C10_CUDA_CHECK(cudaStreamWaitEvent(NULL, ev, 0));
        c10::cuda::setCurrentCUDAStream(c10::cuda::getDefaultCUDAStream());
        C10_CUDA_CHECK(cudaEventDestroy(ev));
        // C10_CUDA_CHECK(cudaStreamWaitEvent(
    }

    void popCurrent() {
        // see above note about stream being hardcoded to the default stream
        cudaEvent_t ev;
        C10_CUDA_CHECK(cudaEventCreate(&ev));
        C10_CUDA_CHECK(cudaEventRecord(ev, NULL));
        CUcontext popped;
        C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuCtxPopCurrent_(&popped));
        TORCH_INTERNAL_ASSERT(popped == context_, "expected popped context to be the current ctx");
        C10_CUDA_CHECK(cudaStreamWaitEvent(parent_stream_, ev, 0));
        C10_CUDA_CHECK(cudaEventDestroy(ev));
    }

private:
    CUgreenCtx green_ctx_ = nullptr;
    CUcontext context_ = nullptr;
    cudaStream_t parent_stream_ = nullptr;
};
