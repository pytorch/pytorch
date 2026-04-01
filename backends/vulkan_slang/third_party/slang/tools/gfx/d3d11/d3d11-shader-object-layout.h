// d3d11-shader-object-layout.h
#pragma once

#include "d3d11-base.h"
#include "d3d11-helper-functions.h"

namespace gfx
{

using namespace Slang;

namespace d3d11
{

class ShaderObjectLayoutImpl : public ShaderObjectLayoutBase
{
public:
    // A shader object comprises three main kinds of state:
    //
    // * Zero or more bytes of ordinary ("uniform") data
    // * Zero or more *bindings* for textures, buffers, and samplers
    // * Zero or more *sub-objects* representing nested parameter blocks, etc.
    //
    // A shader object *layout* stores information that can be used to
    // organize these different kinds of state and optimize access to them.
    //
    // For example, both texture/buffer/sampler bindings and sub-objects
    // are organized into logical *binding ranges* by the Slang reflection
    // API, and a shader object layout will store information about those
    // ranges in a form that is usable for the D3D11 API:

    /// Information about a logical binding range as reported by Slang reflection
    struct BindingRangeInfo
    {
        /// The type of bindings in this range
        slang::BindingType bindingType;

        /// The number of bindings in this range
        Index count;

        /// The starting index for this range in the appropriate "flat" array in a shader object.
        /// E.g., for a shader resource view range, this would be an index into the `m_srvs` array.
        Index baseIndex;

        /// The offset of this binding range from the start of the sub-object
        /// in terms of whatever D3D11 register class it consumes. E.g., for
        /// a `Texture2D` binding range this will represent an offset in
        /// `t` registers.
        ///
        uint32_t registerOffset;

        /// An index into the sub-object array if this binding range is treated
        /// as a sub-object.
        Index subObjectIndex;

        /// Is this binding range specializable, e.g. an existential value or ParameterBlock<IFoo>.
        bool isSpecializable;
    };

    // Sometimes we just want to iterate over the ranges that represent
    // sub-objects while skipping over the others, because sub-object
    // ranges often require extra handling or more state.
    //
    // For that reason we also store pre-computed information about each
    // sub-object range.

    /// Offset information for a sub-object range
    struct SubObjectRangeOffset : BindingOffset
    {
        SubObjectRangeOffset() {}

        SubObjectRangeOffset(slang::VariableLayoutReflection* varLayout);

        /// The offset for "pending" ordinary data related to this range
        uint32_t pendingOrdinaryData = 0;
    };

    /// Stride information for a sub-object range
    struct SubObjectRangeStride : BindingOffset
    {
        SubObjectRangeStride() {}

        SubObjectRangeStride(slang::TypeLayoutReflection* typeLayout);

        /// The strid for "pending" ordinary data related to this range
        uint32_t pendingOrdinaryData = 0;
    };

    /// Information about a logical binding range as reported by Slang reflection
    struct SubObjectRangeInfo
    {
        /// The index of the binding range that corresponds to this sub-object range
        Index bindingRangeIndex;

        /// The layout expected for objects bound to this range (if known)
        RefPtr<ShaderObjectLayoutImpl> layout;

        /// The offset to use when binding the first object in this range
        SubObjectRangeOffset offset;

        /// Stride between consecutive objects in this range
        SubObjectRangeStride stride;
    };

    struct Builder
    {
    public:
        Builder(RendererBase* renderer, slang::ISession* session)
            : m_renderer(renderer), m_session(session)
        {
        }

        RendererBase* m_renderer;
        slang::ISession* m_session;
        slang::TypeLayoutReflection* m_elementTypeLayout;

        List<BindingRangeInfo> m_bindingRanges;
        List<SubObjectRangeInfo> m_subObjectRanges;

        /// The indices of the binding ranges that represent SRVs
        List<Index> m_srvRanges;

        /// The indices of the binding ranges that represent UAVs
        List<Index> m_uavRanges;

        /// The indices of the binding ranges that represent samplers
        List<Index> m_samplerRanges;

        Index m_srvCount = 0;
        Index m_samplerCount = 0;
        Index m_uavCount = 0;
        Index m_subObjectCount = 0;

        uint32_t m_totalOrdinaryDataSize = 0;

        /// The container type of this shader object. When `m_containerType` is
        /// `StructuredBuffer` or `UnsizedArray`, this shader object represents a collection
        /// instead of a single object.
        ShaderObjectContainerType m_containerType = ShaderObjectContainerType::None;

        Result setElementTypeLayout(slang::TypeLayoutReflection* typeLayout);
        SlangResult build(ShaderObjectLayoutImpl** outLayout);
    };

    static Result createForElementType(
        RendererBase* renderer,
        slang::ISession* session,
        slang::TypeLayoutReflection* elementType,
        ShaderObjectLayoutImpl** outLayout);

    List<BindingRangeInfo> const& getBindingRanges() { return m_bindingRanges; }

    Index getBindingRangeCount() { return m_bindingRanges.getCount(); }

    BindingRangeInfo const& getBindingRange(Index index) { return m_bindingRanges[index]; }

    Index getSRVCount() { return m_srvCount; }
    Index getSamplerCount() { return m_samplerCount; }
    Index getUAVCount() { return m_uavCount; }
    Index getSubObjectCount() { return m_subObjectCount; }
    Index getVaryingOutputCount() { return m_varyingOutputCount; }

    SubObjectRangeInfo const& getSubObjectRange(Index index) { return m_subObjectRanges[index]; }
    List<SubObjectRangeInfo> const& getSubObjectRanges() { return m_subObjectRanges; }

    RendererBase* getRenderer() { return m_renderer; }

    slang::TypeReflection* getType() { return m_elementTypeLayout->getType(); }

    /// Get the indices that represent all the SRV ranges in this type
    List<Index> const& getSRVRanges() const { return m_srvRanges; }

    /// Get the indices that reprsent all the UAV ranges in this type
    List<Index> const& getUAVRanges() const { return m_uavRanges; }

    /// Get the indices that represnet all the sampler ranges in this type
    List<Index> const& getSamplerRanges() const { return m_samplerRanges; }

    uint32_t getTotalOrdinaryDataSize() const { return m_totalOrdinaryDataSize; }

protected:
    Result _init(Builder const* builder);

    List<BindingRangeInfo> m_bindingRanges;
    List<Index> m_srvRanges;
    List<Index> m_uavRanges;
    List<Index> m_samplerRanges;
    Index m_srvCount = 0;
    Index m_samplerCount = 0;
    Index m_uavCount = 0;
    Index m_subObjectCount = 0;
    Index m_varyingInputCount = 0;
    Index m_varyingOutputCount = 0;
    uint32_t m_totalOrdinaryDataSize = 0;
    List<SubObjectRangeInfo> m_subObjectRanges;
};

class RootShaderObjectLayoutImpl : public ShaderObjectLayoutImpl
{
    typedef ShaderObjectLayoutImpl Super;

public:
    struct EntryPointInfo
    {
        RefPtr<ShaderObjectLayoutImpl> layout;

        /// The offset for this entry point's parameters, relative to the starting offset for the
        /// program
        BindingOffset offset;
    };

    struct Builder : Super::Builder
    {
        Builder(
            RendererBase* renderer,
            slang::IComponentType* program,
            slang::ProgramLayout* programLayout)
            : Super::Builder(renderer, program->getSession())
            , m_program(program)
            , m_programLayout(programLayout)
        {
        }

        Result build(RootShaderObjectLayoutImpl** outLayout);
        void addGlobalParams(slang::VariableLayoutReflection* globalsLayout);
        void addEntryPoint(
            SlangStage stage,
            ShaderObjectLayoutImpl* entryPointLayout,
            slang::EntryPointLayout* slangEntryPoint);

        slang::IComponentType* m_program;
        slang::ProgramLayout* m_programLayout;
        List<EntryPointInfo> m_entryPoints;
        SimpleBindingOffset m_pendingDataOffset;
    };

    EntryPointInfo& getEntryPoint(Index index) { return m_entryPoints[index]; }

    List<EntryPointInfo>& getEntryPoints() { return m_entryPoints; }

    static Result create(
        RendererBase* renderer,
        slang::IComponentType* program,
        slang::ProgramLayout* programLayout,
        RootShaderObjectLayoutImpl** outLayout);

    slang::IComponentType* getSlangProgram() const { return m_program; }
    slang::ProgramLayout* getSlangProgramLayout() const { return m_programLayout; }

    /// Get the offset at which "pending" shader parameters for this program start
    SimpleBindingOffset const& getPendingDataOffset() const { return m_pendingDataOffset; }

protected:
    Result _init(Builder const* builder);

    ComPtr<slang::IComponentType> m_program;
    slang::ProgramLayout* m_programLayout = nullptr;

    List<EntryPointInfo> m_entryPoints;
    SimpleBindingOffset m_pendingDataOffset;
};

} // namespace d3d11
} // namespace gfx
