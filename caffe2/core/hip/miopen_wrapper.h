// Copyright 2004-present Facebook. All Rights Reserved.
#ifndef CAFFE2_CORE_MIOPEN_WRAPPERS_H_
#define CAFFE2_CORE_MIOPEN_WRAPPERS_H_

#include "caffe2/core/hip/common_miopen.h"
#include "caffe2/core/hip/context_gpu.h"

#include <c10/hip/HIPGuard.h>

namespace caffe2 {

class MIOPENWrapper;

/**
 * MIOPENWorkspace is a wrapper around a raw cuda pointer that holds the miopen
 * scratch space. This struct is meant to be only used in MIOPENWrapper to
 * provide a program-wide scratch space for MIOPEN. The reason behind it is that
 * miopen function calls are usually very efficient, hence one probably does not
 * want to run multiple miopen calls at the same time. As a result, one should
 * not need more than one miopen workspace per device.
 */
struct MIOPENWorkspace
{
    ~MIOPENWorkspace() noexcept {}

    void* get(size_t nbytes)
    {
        if(nbytes_ < nbytes)
        {
            reset();
            data_ = HIPContext::New(nbytes);
            nbytes_               = nbytes;
        }
        CAFFE_ENFORCE_GE(nbytes_, nbytes);
        return data_.get();
    }

    void reset()
    {
      data_.clear();
      nbytes_ = 0;
    }

    private:
     at::DataPtr data_;
     size_t nbytes_{0};
};

// MIOPENState is the owner of the MIOPENWorkspace, and serializes all
// executions of operations that use the state onto it's own stream
// (so multiple Net workers can reuse the same workspace from
// different threads and HIP streams).
class MIOPENState
{
    public:
    explicit MIOPENState(size_t gpu_id) : gpu_id_(gpu_id)
    {
        HIPGuard g(gpu_id_);
        MIOPEN_ENFORCE(miopenCreate(&miopen_handle_));
        HIP_ENFORCE(hipEventCreate(&before_));
        HIP_ENFORCE(hipEventCreate(&after_));
        HIP_ENFORCE(hipStreamCreate(&stream_));
        MIOPEN_ENFORCE(miopenSetStream(miopen_handle_, stream_));
    }

    ~MIOPENState() noexcept
    {
        HIPGuard g(gpu_id_);
        MIOPEN_CHECK(miopenDestroy(miopen_handle_));
        HIP_CHECK(hipStreamDestroy(stream_));
        HIP_CHECK(hipEventDestroy(after_));
        HIP_CHECK(hipEventDestroy(before_));
    }

    miopenHandle_t& miopen_handle() { return miopen_handle_; }

    MIOPENWorkspace& workspace() { return workspace_; }

    template <typename F>
    void execute(hipStream_t stream, F&& f)
    {
        HIP_ENFORCE(hipEventRecord(before_, stream));
        HIP_ENFORCE(hipStreamWaitEvent(stream_, before_, 0));
        f(this);
        HIP_ENFORCE(hipEventRecord(after_, stream_));
        HIP_ENFORCE(hipStreamWaitEvent(stream, after_, 0));
    }

    private:
    miopenHandle_t miopen_handle_{nullptr};
    hipEvent_t before_{nullptr};
    hipEvent_t after_{nullptr};
    hipStream_t stream_{nullptr};
    MIOPENWorkspace workspace_;
    size_t gpu_id_{0};
    C10_DISABLE_COPY_AND_ASSIGN(MIOPENState);
};

/**
 * MIOPENWrapper is a class that wraps the miopen handles and miopen workspaces.
 *
 * The wrapper ensures that for each thread and each gpu, there is one
 * identical miopen handle, which is also associated with the thread-local
 * per-device hip stream. The wrapper also hosts the device-specific miopen
 * workspace (scratch space for some miopen functions).
 *
 */
class MIOPENWrapper
{
    public:
    /**
     * Creates a miopen wrapper associated with a HIPContext object. Note that
     * the HIPContext object should outlive the MIOPENWrapper.
     */
    explicit MIOPENWrapper(HIPContext* context) : context_(context) {}

    /**
     * Returns the inline miopen handle that executes on the current
     * thread's hip_stream.
     */
    miopenHandle_t inline_miopen_handle() { return context_->miopen_handle(); }

    // Executes the closure F on the MIOPENState associated with state_idx
    template <typename F>
    void with_miopen_state(size_t state_idx, F&& f)
    {
        CAFFE_ENFORCE(state_idx < CAFFE2_COMPILE_TIME_MAX_MIOPEN_STATES, "Invalid state_idx");
        auto& sync_state = miopen_states()[context_->device_id()][state_idx];

        HIPGuard dg(context_->device_id());

        // We need to serialize execution on the MIOPENState as we can't
        // allow multiple threads to race through the cudaEventRecord
        // calls (so a worker thread might wait on another worker thread's
        // execution)
        std::lock_guard<std::mutex> g(sync_state.mutex);
        if(!sync_state.state.get())
        {
          sync_state.state.reset(new MIOPENState(context_->device_id()));
        }
        CHECK_NOTNULL(sync_state.state.get())->execute(context_->hip_stream(), f);
    }

    protected:
    // Pointer to an external cuda context that the miopen wrapper will use.
    HIPContext* context_;

    static constexpr size_t CAFFE2_COMPILE_TIME_MAX_MIOPEN_STATES = 4;

    struct SyncedMIOPENState
    {
        std::mutex mutex;
        std::unique_ptr<MIOPENState> state;
    };

    using PerGPUMIOPENStates = std::array<
        std::array<SyncedMIOPENState, CAFFE2_COMPILE_TIME_MAX_MIOPEN_STATES>,
        C10_COMPILE_TIME_MAX_GPUS>;
    static PerGPUMIOPENStates& miopen_states();

    C10_DISABLE_COPY_AND_ASSIGN(MIOPENWrapper);
};

}; // namespace caffe2

#endif
