// cuda-command-queue.h
#pragma once
#include "cuda-base.h"
#include "cuda-device.h"
#include "cuda-helper-functions.h"
#include "cuda-pipeline-state.h"
#include "cuda-shader-object.h"

namespace gfx
{
#ifdef GFX_ENABLE_CUDA
using namespace Slang;

namespace cuda
{

class CommandQueueImpl : public ICommandQueue, public ComObject
{
public:
    SLANG_COM_OBJECT_IUNKNOWN_ALL
    ICommandQueue* getInterface(const Guid& guid);

    RefPtr<ComputePipelineStateImpl> currentPipeline;
    RefPtr<RootShaderObjectImpl> currentRootObject;
    RefPtr<DeviceImpl> renderer;
    CUstream stream;
    Desc m_desc;

    void init(DeviceImpl* inRenderer);
    ~CommandQueueImpl();

    virtual SLANG_NO_THROW const Desc& SLANG_MCALL getDesc() override { return m_desc; }

    virtual SLANG_NO_THROW void SLANG_MCALL executeCommandBuffers(
        GfxCount count,
        ICommandBuffer* const* commandBuffers,
        IFence* fence,
        uint64_t valueToSignal) override;

    virtual SLANG_NO_THROW void SLANG_MCALL waitOnHost() override;

    virtual SLANG_NO_THROW Result SLANG_MCALL
    waitForFenceValuesOnDevice(GfxCount fenceCount, IFence** fences, uint64_t* waitValues) override;

    virtual SLANG_NO_THROW Result SLANG_MCALL getNativeHandle(InteropHandle* outHandle) override;

    void setPipelineState(IPipelineState* state);

    Result bindRootShaderObject(IShaderObject* object);

    void dispatchCompute(int x, int y, int z);

    void copyBuffer(
        IBufferResource* dst,
        size_t dstOffset,
        IBufferResource* src,
        size_t srcOffset,
        size_t size);

    void uploadBufferData(IBufferResource* dst, size_t offset, size_t size, void* data);

    void writeTimestamp(IQueryPool* pool, SlangInt index);

    void execute(CommandBufferImpl* commandBuffer);
};

} // namespace cuda
#endif
} // namespace gfx
