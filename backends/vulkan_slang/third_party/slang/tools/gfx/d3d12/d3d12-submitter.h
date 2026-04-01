// d3d12-submitter.h
#pragma once

#include "d3d12-base.h"

namespace gfx
{
namespace d3d12
{

using namespace Slang;

struct Submitter
{
    virtual void setRootConstantBufferView(
        int index,
        D3D12_GPU_VIRTUAL_ADDRESS gpuBufferLocation) = 0;
    virtual void setRootUAV(int index, D3D12_GPU_VIRTUAL_ADDRESS gpuBufferLocation) = 0;
    virtual void setRootSRV(int index, D3D12_GPU_VIRTUAL_ADDRESS gpuBufferLocation) = 0;
    virtual void setRootDescriptorTable(int index, D3D12_GPU_DESCRIPTOR_HANDLE BaseDescriptor) = 0;
    virtual void setRootSignature(ID3D12RootSignature* rootSignature) = 0;
    virtual void setRootConstants(
        Index rootParamIndex,
        Index dstOffsetIn32BitValues,
        Index countOf32BitValues,
        void const* srcData) = 0;
    virtual void setPipelineState(PipelineStateBase* pipelineState) = 0;
};

struct GraphicsSubmitter : public Submitter
{
    virtual void setRootConstantBufferView(int index, D3D12_GPU_VIRTUAL_ADDRESS gpuBufferLocation)
        override;
    virtual void setRootUAV(int index, D3D12_GPU_VIRTUAL_ADDRESS gpuBufferLocation) override;
    virtual void setRootSRV(int index, D3D12_GPU_VIRTUAL_ADDRESS gpuBufferLocation) override;
    virtual void setRootDescriptorTable(int index, D3D12_GPU_DESCRIPTOR_HANDLE baseDescriptor)
        override;
    virtual void setRootSignature(ID3D12RootSignature* rootSignature) override;
    virtual void setRootConstants(
        Index rootParamIndex,
        Index dstOffsetIn32BitValues,
        Index countOf32BitValues,
        void const* srcData) override;
    virtual void setPipelineState(PipelineStateBase* pipeline) override;

    GraphicsSubmitter(ID3D12GraphicsCommandList* commandList)
        : m_commandList(commandList)
    {
    }

    ID3D12GraphicsCommandList* m_commandList;
};

struct ComputeSubmitter : public Submitter
{
    virtual void setRootConstantBufferView(int index, D3D12_GPU_VIRTUAL_ADDRESS gpuBufferLocation)
        override;
    virtual void setRootUAV(int index, D3D12_GPU_VIRTUAL_ADDRESS gpuBufferLocation) override;
    virtual void setRootSRV(int index, D3D12_GPU_VIRTUAL_ADDRESS gpuBufferLocation) override;
    virtual void setRootDescriptorTable(int index, D3D12_GPU_DESCRIPTOR_HANDLE baseDescriptor)
        override;
    virtual void setRootSignature(ID3D12RootSignature* rootSignature) override;
    virtual void setRootConstants(
        Index rootParamIndex,
        Index dstOffsetIn32BitValues,
        Index countOf32BitValues,
        void const* srcData) override;
    virtual void setPipelineState(PipelineStateBase* pipeline) override;
    ComputeSubmitter(ID3D12GraphicsCommandList* commandList)
        : m_commandList(commandList)
    {
    }

    ID3D12GraphicsCommandList* m_commandList;
};

} // namespace d3d12
} // namespace gfx
