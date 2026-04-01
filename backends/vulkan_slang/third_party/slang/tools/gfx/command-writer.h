#pragma once

#include "core/slang-basic.h"
#include "renderer-shared.h"
#include "slang-com-ptr.h"
#include "slang-gfx.h"

namespace gfx
{

enum class CommandName
{
    SetPipelineState,
    BindRootShaderObject,
    SetFramebuffer,
    ClearFrame,
    SetViewports,
    SetScissorRects,
    SetPrimitiveTopology,
    SetVertexBuffers,
    SetIndexBuffer,
    Draw,
    DrawIndexed,
    DrawInstanced,
    DrawIndexedInstanced,
    SetStencilReference,
    DispatchCompute,
    UploadBufferData,
    CopyBuffer,
    WriteTimestamp,
};

const uint8_t kMaxCommandOperands = 5;

struct Command
{
    CommandName name;
    uint32_t operands[kMaxCommandOperands];
    Command() = default;
    Command(CommandName inName, uint32_t op)
        : name(inName)
    {
        operands[0] = op;
    }
    Command(CommandName inName, uint32_t op1, uint32_t op2)
        : name(inName)
    {
        operands[0] = op1;
        operands[1] = op2;
    }
    Command(CommandName inName, uint32_t op1, uint32_t op2, uint32_t op3)
        : name(inName)
    {
        operands[0] = op1;
        operands[1] = op2;
        operands[2] = op3;
    }
    Command(CommandName inName, uint32_t op1, uint32_t op2, uint32_t op3, uint32_t op4)
        : name(inName)
    {
        operands[0] = op1;
        operands[1] = op2;
        operands[2] = op3;
        operands[3] = op4;
    }
    Command(
        CommandName inName,
        uint32_t op1,
        uint32_t op2,
        uint32_t op3,
        uint32_t op4,
        uint32_t op5)
        : name(inName)
    {
        operands[0] = op1;
        operands[1] = op2;
        operands[2] = op3;
        operands[3] = op4;
        operands[4] = op5;
    }
};

class CommandWriter
{
public:
    Slang::List<Command> m_commands;
    Slang::List<Slang::RefPtr<Slang::RefObject>> m_objects;
    Slang::List<uint8_t> m_data;
    bool m_hasWriteTimestamps = false;

public:
    void clear()
    {
        m_commands.clear();
        for (auto& obj : m_objects)
            obj = nullptr;
        m_objects.clear();
        m_data.clear();
        m_hasWriteTimestamps = false;
    }

    // Copies user data into `m_data` buffer and returns the offset to retrieve the data.
    Offset encodeData(const void* data, Size size)
    {
        Offset offset = (Offset)m_data.getCount();
        m_data.setCount(m_data.getCount() + size);
        memcpy(m_data.getBuffer() + offset, data, size);
        return offset;
    }

    Offset encodeObject(Slang::RefObject* obj)
    {
        Offset offset = (Offset)m_objects.getCount();
        m_objects.add(obj);
        return offset;
    }

    template<typename T>
    T* getObject(uint32_t offset)
    {
        return static_cast<T*>(m_objects[offset].Ptr());
    }

    template<typename T>
    T* getData(Offset offset)
    {
        return reinterpret_cast<T*>(m_data.getBuffer() + offset);
    }

    void setPipelineState(IPipelineState* state)
    {
        auto offset = encodeObject(static_cast<PipelineStateBase*>(state));
        m_commands.add(Command(CommandName::SetPipelineState, (uint32_t)offset));
    }

    void bindRootShaderObject(IShaderObject* object)
    {
        auto rootOffset = encodeObject(static_cast<ShaderObjectBase*>(object));
        m_commands.add(Command(CommandName::BindRootShaderObject, (uint32_t)rootOffset));
    }

    void uploadBufferData(IBufferResource* buffer, Offset offset, Size size, void* data)
    {
        auto bufferOffset = encodeObject(static_cast<BufferResource*>(buffer));
        auto dataOffset = encodeData(data, size);
        m_commands.add(Command(
            CommandName::UploadBufferData,
            (uint32_t)bufferOffset,
            (uint32_t)offset,
            (uint32_t)size,
            (uint32_t)dataOffset));
    }

    void copyBuffer(
        IBufferResource* dst,
        Offset dstOffset,
        IBufferResource* src,
        Offset srcOffset,
        Size size)
    {
        auto dstBuffer = encodeObject(static_cast<BufferResource*>(dst));
        auto srcBuffer = encodeObject(static_cast<BufferResource*>(src));
        m_commands.add(Command(
            CommandName::CopyBuffer,
            (uint32_t)dstBuffer,
            (uint32_t)dstOffset,
            (uint32_t)srcBuffer,
            (uint32_t)srcOffset,
            (uint32_t)size));
    }

    void setFramebuffer(IFramebuffer* frameBuffer)
    {
        auto framebufferOffset = encodeObject(static_cast<FramebufferBase*>(frameBuffer));
        m_commands.add(Command(CommandName::SetFramebuffer, (uint32_t)framebufferOffset));
    }

    void clearFrame(uint32_t colorBufferMask, bool clearDepth, bool clearStencil)
    {
        m_commands.add(Command(
            CommandName::ClearFrame,
            colorBufferMask,
            clearDepth ? 1 : 0,
            clearStencil ? 1 : 0));
    }

    void setViewports(GfxCount count, const Viewport* viewports)
    {
        auto offset = encodeData(viewports, sizeof(Viewport) * count);
        m_commands.add(Command(CommandName::SetViewports, (uint32_t)count, (uint32_t)offset));
    }

    void setScissorRects(GfxCount count, const ScissorRect* scissors)
    {
        auto offset = encodeData(scissors, sizeof(ScissorRect) * count);
        m_commands.add(Command(CommandName::SetScissorRects, (uint32_t)count, (uint32_t)offset));
    }

    void setPrimitiveTopology(PrimitiveTopology topology)
    {
        m_commands.add(Command(CommandName::SetPrimitiveTopology, (uint32_t)topology));
    }

    void setVertexBuffers(
        GfxIndex startSlot,
        GfxCount slotCount,
        IBufferResource* const* buffers,
        const Offset* offsets)
    {
        Offset bufferOffset = 0;
        for (GfxCount i = 0; i < slotCount; i++)
        {
            auto offset = encodeObject(static_cast<BufferResource*>(buffers[i]));
            if (i == 0)
                bufferOffset = offset;
        }
        auto offsetsOffset = encodeData(offsets, sizeof(Size) * slotCount);
        m_commands.add(Command(
            CommandName::SetVertexBuffers,
            (uint32_t)startSlot,
            (uint32_t)slotCount,
            (uint32_t)bufferOffset,
            (uint32_t)offsetsOffset));
    }

    void setIndexBuffer(IBufferResource* buffer, Format indexFormat, Offset offset)
    {
        auto bufferOffset = encodeObject(static_cast<BufferResource*>(buffer));
        m_commands.add(Command(
            CommandName::SetIndexBuffer,
            (uint32_t)bufferOffset,
            (uint32_t)indexFormat,
            (uint32_t)offset));
    }

    void draw(GfxCount vertexCount, GfxIndex startVertex)
    {
        m_commands.add(Command(CommandName::Draw, (uint32_t)vertexCount, (uint32_t)startVertex));
    }

    void drawIndexed(GfxCount indexCount, GfxIndex startIndex, GfxIndex baseVertex)
    {
        m_commands.add(Command(
            CommandName::DrawIndexed,
            (uint32_t)indexCount,
            (uint32_t)startIndex,
            (uint32_t)baseVertex));
    }

    void drawInstanced(
        GfxCount vertexCount,
        GfxCount instanceCount,
        GfxIndex startVertex,
        GfxIndex startInstanceLocation)
    {
        m_commands.add(Command(
            CommandName::DrawInstanced,
            (uint32_t)vertexCount,
            (uint32_t)instanceCount,
            (uint32_t)startVertex,
            (uint32_t)startInstanceLocation));
    }

    void drawIndexedInstanced(
        GfxCount indexCount,
        GfxCount instanceCount,
        GfxIndex startIndexLocation,
        GfxIndex baseVertexLocation,
        GfxIndex startInstanceLocation)
    {
        m_commands.add(Command(
            CommandName::DrawIndexedInstanced,
            (uint32_t)indexCount,
            (uint32_t)instanceCount,
            (uint32_t)startIndexLocation,
            (uint32_t)baseVertexLocation,
            (uint32_t)startInstanceLocation));
    }

    void setStencilReference(uint32_t referenceValue)
    {
        m_commands.add(Command(CommandName::SetStencilReference, referenceValue));
    }

    void dispatchCompute(int x, int y, int z)
    {
        m_commands.add(
            Command(CommandName::DispatchCompute, (uint32_t)x, (uint32_t)y, (uint32_t)z));
    }

    void writeTimestamp(IQueryPool* pool, GfxIndex index)
    {
        auto poolOffset = encodeObject(static_cast<QueryPoolBase*>(pool));
        m_commands.add(Command(CommandName::WriteTimestamp, (uint32_t)poolOffset, (uint32_t)index));
        m_hasWriteTimestamps = true;
    }
};
} // namespace gfx
