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
        , context_(other.context_) {
        other.green_ctx_ = nullptr;
        other.context_ = nullptr;
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
            
            // Null out other's pointers
            other.green_ctx_ = nullptr;
            other.context_ = nullptr;
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
        C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuCtxSetCurrent_(context_));
    }

private:
    CUgreenCtx green_ctx_ = nullptr;
    CUcontext context_ = nullptr;
};
