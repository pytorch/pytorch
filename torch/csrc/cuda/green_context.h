#pragma once
#include <cuda.h>
#include <memory>
#include <stdexcept>
#include <vector>

class CudaError : public std::runtime_error {
public:
    explicit CudaError(CUresult error, const char* call) 
        : std::runtime_error(formatError(error, call)) {}

private:
    static std::string formatError(CUresult error, const char* call) {
        const char* str;
        cuGetErrorString(error, &str);
        return std::string(call) + " failed with error " + str;
    }
};

#define CUDA_CHECK(call) do { \
    CUresult result = call; \
    if (result != CUDA_SUCCESS) { \
        throw CudaError(result, #call); \
    } \
} while(0)

class GreenContext {
public:
    GreenContext(int device_id, unsigned int num_sms) {
        CUdevice device;
        CUDA_CHECK(cuDeviceGet(&device, device_id));

        // Get device resources
        CUdevResource device_resource;
        CUDA_CHECK(cuDeviceGetDevResource(device, &device_resource, CU_DEV_RESOURCE_TYPE_SM));

        // Split resources
        std::vector<CUdevResource> result(1);
        auto result_data = result.data();
        unsigned int nb_groups = 1;
        CUdevResource remaining;
        
        CUDA_CHECK(cuDevSmResourceSplitByCount(
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
        CUDA_CHECK(cuDevResourceGenerateDesc(&desc, result_data, 1));

        // Create green context
        CUDA_CHECK(cuGreenCtxCreate(&green_ctx_, desc, device, CU_GREEN_CTX_DEFAULT_STREAM));

        // Convert to regular context
        CUDA_CHECK(cuCtxFromGreenCtx(&context_, green_ctx_));
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
        if (green_ctx_) {
            CUresult result = cuGreenCtxDestroy(green_ctx_);
            if (result != CUDA_SUCCESS) {
                fprintf(stderr, "Failed to destroy green context: %d\n", result);
            }
        }
    }

    // Get the underlying CUDA context
    CUcontext getContext() const { return context_; }
    
    // Get the underlying green context
    CUgreenCtx getGreenContext() const { return green_ctx_; }

    // Make this context current
    void makeCurrent() {
        CUDA_CHECK(cuCtxSetCurrent(context_));
    }

private:
    CUgreenCtx green_ctx_ = nullptr;
    CUcontext context_ = nullptr;
};
