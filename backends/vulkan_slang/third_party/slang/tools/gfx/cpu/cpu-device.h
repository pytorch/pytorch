// cpu-device.h
#pragma once
#include "cpu-base.h"
#include "cpu-pipeline-state.h"
#include "cpu-shader-object.h"

namespace gfx
{
using namespace Slang;

namespace cpu
{

class DeviceImpl : public ImmediateComputeDeviceBase
{
public:
    ~DeviceImpl();

    virtual SLANG_NO_THROW SlangResult SLANG_MCALL initialize(const Desc& desc) override;

    virtual SLANG_NO_THROW Result SLANG_MCALL createTextureResource(
        const ITextureResource::Desc& desc,
        const ITextureResource::SubresourceData* initData,
        ITextureResource** outResource) override;

    virtual SLANG_NO_THROW Result SLANG_MCALL createBufferResource(
        const IBufferResource::Desc& descIn,
        const void* initData,
        IBufferResource** outResource) override;

    virtual SLANG_NO_THROW Result SLANG_MCALL createTextureView(
        ITextureResource* inTexture,
        IResourceView::Desc const& desc,
        IResourceView** outView) override;

    virtual SLANG_NO_THROW Result SLANG_MCALL createBufferView(
        IBufferResource* inBuffer,
        IBufferResource* counterBuffer,
        IResourceView::Desc const& desc,
        IResourceView** outView) override;

    virtual Result createShaderObjectLayout(
        slang::ISession* session,
        slang::TypeLayoutReflection* typeLayout,
        ShaderObjectLayoutBase** outLayout) override;

    virtual Result createShaderObject(ShaderObjectLayoutBase* layout, IShaderObject** outObject)
        override;

    virtual Result createMutableShaderObject(
        ShaderObjectLayoutBase* layout,
        IShaderObject** outObject) override;

    virtual Result createRootShaderObject(IShaderProgram* program, ShaderObjectBase** outObject)
        override;

    virtual SLANG_NO_THROW Result SLANG_MCALL createProgram(
        const IShaderProgram::Desc& desc,
        IShaderProgram** outProgram,
        ISlangBlob** outDiagnosticBlob) override;

    virtual SLANG_NO_THROW Result SLANG_MCALL createComputePipelineState(
        const ComputePipelineStateDesc& desc,
        IPipelineState** outState) override;

    virtual SLANG_NO_THROW Result SLANG_MCALL
    createQueryPool(const IQueryPool::Desc& desc, IQueryPool** outPool) override;

    virtual void writeTimestamp(IQueryPool* pool, GfxIndex index) override;

    virtual SLANG_NO_THROW const DeviceInfo& SLANG_MCALL getDeviceInfo() const override;

    virtual SLANG_NO_THROW Result SLANG_MCALL
    createSamplerState(ISamplerState::Desc const& desc, ISamplerState** outSampler) override;

    virtual void submitGpuWork() override {}
    virtual void waitForGpu() override {}
    virtual void* map(IBufferResource* buffer, MapFlavor flavor) override;
    virtual void unmap(IBufferResource* buffer, size_t offsetWritten, size_t sizeWritten) override;

private:
    RefPtr<PipelineStateImpl> m_currentPipeline = nullptr;
    RefPtr<RootShaderObjectImpl> m_currentRootObject = nullptr;
    DeviceInfo m_info;

    virtual void setPipelineState(IPipelineState* state) override;

    virtual void bindRootShaderObject(IShaderObject* object) override;

    virtual void dispatchCompute(int x, int y, int z) override;

    virtual void copyBuffer(
        IBufferResource* dst,
        size_t dstOffset,
        IBufferResource* src,
        size_t srcOffset,
        size_t size) override;
};

} // namespace cpu

Result SLANG_MCALL createCPUDevice(const IDevice::Desc* desc, IDevice** outDevice);

} // namespace gfx
