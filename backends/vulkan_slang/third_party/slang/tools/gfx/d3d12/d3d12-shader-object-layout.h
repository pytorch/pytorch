// d3d12-shader-object-layout.h
#pragma once

#include "d3d12-base.h"

namespace gfx
{
namespace d3d12
{

using namespace Slang;

/// A representation of the offset at which to bind a shader parameter or sub-object
struct BindingOffset
{
    // Note: When we actually bind a shader object to the pipeline we do not care about
    // HLSL-specific notions like `t` registers and `space`s. Those concepts are all
    // mediated by the root signature.
    //
    // Instead, we need to consider the offsets at which the object will be bound
    // into the actual D3D12 API state, which consists of the index of the current
    // root parameter to bind from, as well as indices into the current descriptor
    // tables (for resource views and samplers).

    uint32_t rootParam = 0;
    uint32_t resource = 0;
    uint32_t sampler = 0;

    void operator+=(BindingOffset const& offset)
    {
        rootParam += offset.rootParam;
        resource += offset.resource;
        sampler += offset.sampler;
    }
};

// Provides information on how binding ranges are stored in descriptor tables for
// a shader object.
// We allocate one CPU descriptor table for each descriptor heap type for the shader
// object. In `ShaderObjectLayoutImpl`, we store the offset into the descriptor tables
// for each binding, so we know where to write the descriptor when the user sets
// a resource or sampler binding.
class ShaderObjectLayoutImpl : public ShaderObjectLayoutBase
{
public:
    /// Information about a single logical binding range
    struct BindingRangeInfo
    {
        // Some of the information we store on binding ranges is redundant with
        // the information that Slang's reflection information stores, but having
        // it here can make the code more compact and obvious.

        /// The type of binding in this range.
        slang::BindingType bindingType;

        /// The shape of the resource
        SlangResourceShape resourceShape;

        /// The number of distinct bindings in this range.
        uint32_t count;

        /// A "flat" index for this range in whatever array provides backing storage for it
        uint32_t baseIndex;

        /// An index into the sub-object array if this binding range is treated
        /// as a sub-object.
        uint32_t subObjectIndex;

        /// The stride of a structured buffer.
        uint32_t bufferElementStride;

        bool isRootParameter;

        /// Is this binding range represent a specialization point, such as an existential value, or
        /// a `ParameterBlock<IFoo>`.
        bool isSpecializable;
    };

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

    /// Information about a sub-objecrt range
    struct SubObjectRangeInfo
    {
        /// The index of the binding range corresponding to this sub-object range
        Index bindingRangeIndex = 0;

        /// Layout information for the type of sub-object expected to be bound, if known
        RefPtr<ShaderObjectLayoutImpl> layout;

        /// The offset to use when binding the first object in this range
        SubObjectRangeOffset offset;

        /// Stride between consecutive objects in this range
        SubObjectRangeStride stride;
    };

    struct RootParameterInfo
    {
        IResourceView::Type type;
    };

    static bool isBindingRangeRootParameter(
        SlangSession* globalSession,
        const char* rootParameterAttributeName,
        slang::TypeLayoutReflection* typeLayout,
        Index bindingRangeIndex);

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
        List<RootParameterInfo> m_rootParamsInfo;

        /// The number of sub-objects (not just sub-object *ranges*) stored in instances of this
        /// layout
        uint32_t m_subObjectCount = 0;

        /// Counters for the number of root parameters, resources, and samplers in this object
        /// itself
        BindingOffset m_ownCounts;

        /// Counters for the number of root parameters, resources, and sampler in this object
        /// and transitive sub-objects
        BindingOffset m_totalCounts;

        /// The number of root parameter consumed by (transitive) sub-objects
        uint32_t m_childRootParameterCount = 0;

        /// The total size in bytes of the ordinary data for this object and transitive
        /// sub-object.
        uint32_t m_totalOrdinaryDataSize = 0;

        /// The container type of this shader object. When `m_containerType` is
        /// `StructuredBuffer` or `UnsizedArray`, this shader object represents a collection
        /// instead of a single object.
        ShaderObjectContainerType m_containerType = ShaderObjectContainerType::None;

        Result setElementTypeLayout(slang::TypeLayoutReflection* typeLayout);

        Result build(ShaderObjectLayoutImpl** outLayout);
    };

    static Result createForElementType(
        RendererBase* renderer,
        slang::ISession* session,
        slang::TypeLayoutReflection* elementType,
        ShaderObjectLayoutImpl** outLayout);

    List<BindingRangeInfo> const& getBindingRanges() { return m_bindingRanges; }

    Index getBindingRangeCount() { return m_bindingRanges.getCount(); }

    BindingRangeInfo const& getBindingRange(Index index) { return m_bindingRanges[index]; }

    uint32_t getResourceSlotCount() { return m_ownCounts.resource; }
    uint32_t getSamplerSlotCount() { return m_ownCounts.sampler; }
    Index getSubObjectSlotCount() { return m_subObjectCount; }
    Index getSubObjectCount() { return m_subObjectCount; }

    uint32_t getTotalResourceDescriptorCount() { return m_totalCounts.resource; }
    uint32_t getTotalSamplerDescriptorCount() { return m_totalCounts.sampler; }

    uint32_t getOrdinaryDataBufferCount() { return m_totalOrdinaryDataSize ? 1 : 0; }
    bool hasOrdinaryDataBuffer() { return m_totalOrdinaryDataSize != 0; }

    uint32_t getTotalResourceDescriptorCountWithoutOrdinaryDataBuffer()
    {
        return m_totalCounts.resource - getOrdinaryDataBufferCount();
    }

    uint32_t getOwnUserRootParameterCount() { return (uint32_t)m_rootParamsInfo.getCount(); }
    uint32_t getTotalRootTableParameterCount() { return m_totalCounts.rootParam; }
    uint32_t getChildRootParameterCount() { return m_childRootParameterCount; }

    uint32_t getTotalOrdinaryDataSize() const { return m_totalOrdinaryDataSize; }

    SubObjectRangeInfo const& getSubObjectRange(Index index) { return m_subObjectRanges[index]; }
    List<SubObjectRangeInfo> const& getSubObjectRanges() { return m_subObjectRanges; }

    RendererBase* getRenderer() { return m_renderer; }

    slang::TypeReflection* getType() { return m_elementTypeLayout->getType(); }

    const RootParameterInfo& getRootParameterInfo(Index index) { return m_rootParamsInfo[index]; }

protected:
    Result init(Builder* builder);

    List<BindingRangeInfo> m_bindingRanges;
    List<SubObjectRangeInfo> m_subObjectRanges;
    List<RootParameterInfo> m_rootParamsInfo;

    BindingOffset m_ownCounts;
    BindingOffset m_totalCounts;

    uint32_t m_subObjectCount = 0;
    uint32_t m_childRootParameterCount = 0;

    uint32_t m_totalOrdinaryDataSize = 0;
};

class RootShaderObjectLayoutImpl : public ShaderObjectLayoutImpl
{
    typedef ShaderObjectLayoutImpl Super;

public:
    struct EntryPointInfo
    {
        RefPtr<ShaderObjectLayoutImpl> layout;
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

        void addEntryPoint(SlangStage stage, ShaderObjectLayoutImpl* entryPointLayout);

        slang::IComponentType* m_program;
        slang::ProgramLayout* m_programLayout;
        List<EntryPointInfo> m_entryPoints;
    };

    EntryPointInfo& getEntryPoint(Index index) { return m_entryPoints[index]; }

    List<EntryPointInfo>& getEntryPoints() { return m_entryPoints; }

    struct DescriptorSetLayout
    {
        List<D3D12_DESCRIPTOR_RANGE1> m_resourceRanges;
        List<D3D12_DESCRIPTOR_RANGE1> m_samplerRanges;
        uint32_t m_resourceCount = 0;
        uint32_t m_samplerCount = 0;
    };

    struct RootSignatureDescBuilder
    {
        DeviceImpl* m_device;

        RootSignatureDescBuilder(DeviceImpl* device)
            : m_device(device)
        {
        }

        // We will use one descriptor set for the global scope and one additional
        // descriptor set for each `ParameterBlock` binding range in the shader object
        // hierarchy, regardless of the shader's `space` indices.
        List<DescriptorSetLayout> m_descriptorSets;
        List<D3D12_ROOT_PARAMETER1> m_rootParameters;
        List<D3D12_ROOT_PARAMETER1> m_rootDescTableParameters;

        D3D12_ROOT_SIGNATURE_DESC1 m_rootSignatureDesc = {};

        static Result translateDescriptorRangeType(
            slang::BindingType c,
            D3D12_DESCRIPTOR_RANGE_TYPE* outType);

        /// Stores offset information to apply to the reflected register/space for a descriptor
        /// range.
        ///
        struct BindingRegisterOffset
        {
            uint32_t spaceOffset = 0; // The `space` index as specified in shader.

            enum
            {
                kRangeTypeCount = 4
            };

            /// An offset to apply for each D3D12 register class, as given
            /// by a `D3D12_DESCRIPTOR_RANGE_TYPE`.
            ///
            /// Note that the `D3D12_DESCRIPTOR_RANGE_TYPE` enumeration has
            /// values between 0 and 3, inclusive.
            ///
            uint32_t offsetForRangeType[kRangeTypeCount] = {0, 0, 0, 0};

            uint32_t& operator[](D3D12_DESCRIPTOR_RANGE_TYPE type)
            {
                return offsetForRangeType[int(type)];
            }

            uint32_t operator[](D3D12_DESCRIPTOR_RANGE_TYPE type) const
            {
                return offsetForRangeType[int(type)];
            }

            BindingRegisterOffset() {}

            BindingRegisterOffset(slang::VariableLayoutReflection* varLayout)
            {
                if (varLayout)
                {
                    spaceOffset = (UINT)varLayout->getOffset(
                        SLANG_PARAMETER_CATEGORY_SUB_ELEMENT_REGISTER_SPACE);
                    offsetForRangeType[D3D12_DESCRIPTOR_RANGE_TYPE_CBV] =
                        (UINT)varLayout->getOffset(SLANG_PARAMETER_CATEGORY_CONSTANT_BUFFER);
                    offsetForRangeType[D3D12_DESCRIPTOR_RANGE_TYPE_SRV] =
                        (UINT)varLayout->getOffset(SLANG_PARAMETER_CATEGORY_SHADER_RESOURCE);
                    offsetForRangeType[D3D12_DESCRIPTOR_RANGE_TYPE_UAV] =
                        (UINT)varLayout->getOffset(SLANG_PARAMETER_CATEGORY_UNORDERED_ACCESS);
                    offsetForRangeType[D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER] =
                        (UINT)varLayout->getOffset(SLANG_PARAMETER_CATEGORY_SAMPLER_STATE);
                }
            }

            void operator+=(BindingRegisterOffset const& other)
            {
                spaceOffset += other.spaceOffset;
                for (int i = 0; i < kRangeTypeCount; ++i)
                {
                    offsetForRangeType[i] += other.offsetForRangeType[i];
                }
            }
        };

        struct BindingRegisterOffsetPair
        {
            BindingRegisterOffset primary;
            BindingRegisterOffset pending;

            BindingRegisterOffsetPair() {}

            BindingRegisterOffsetPair(slang::VariableLayoutReflection* varLayout)
                : primary(varLayout), pending(varLayout->getPendingDataLayout())
            {
            }

            void operator+=(BindingRegisterOffsetPair const& other)
            {
                primary += other.primary;
                pending += other.pending;
            }
        };
        /// Add a new descriptor set to the layout being computed.
        ///
        /// Note that a "descriptor set" in the layout may amount to
        /// zero, one, or two different descriptor *tables* in the
        /// final D3D12 root signature. Each descriptor set may
        /// contain zero or more view ranges (CBV/SRV/UAV) and zero
        /// or more sampler ranges. It maps to a view descriptor table
        /// if the number of view ranges is non-zero and to a sampler
        /// descriptor table if the number of sampler ranges is non-zero.
        ///
        uint32_t addDescriptorSet();

        Result addDescriptorRange(
            Index physicalDescriptorSetIndex,
            D3D12_DESCRIPTOR_RANGE_TYPE rangeType,
            UINT registerIndex,
            UINT spaceIndex,
            UINT count,
            bool isRootParameter);
        /// Add one descriptor range as specified in Slang reflection information to the layout.
        ///
        /// The layout information is taken from `typeLayout` for the descriptor
        /// range with the given `descriptorRangeIndex` within the logical
        /// descriptor set (reflected by Slang) with the given `logicalDescriptorSetIndex`.
        ///
        /// The `physicalDescriptorSetIndex` is the index in the `m_descriptorSets` array of
        /// the descriptor set that the range should be added to.
        ///
        /// The `offset` encodes information about space and/or register offsets that
        /// should be applied to descrptor ranges.
        ///
        /// This operation can fail if the given descriptor range encodes a range that
        /// doesn't map to anything directly supported by D3D12. Higher-level routines
        /// will often want to ignore such failures.
        ///
        Result addDescriptorRange(
            slang::TypeLayoutReflection* typeLayout,
            Index physicalDescriptorSetIndex,
            BindingRegisterOffset const& containerOffset,
            BindingRegisterOffset const& elementOffset,
            Index logicalDescriptorSetIndex,
            Index descriptorRangeIndex,
            bool isRootParameter);

        /// Add one binding range to the computed layout.
        ///
        /// The layout information is taken from `typeLayout` for the binding
        /// range with the given `bindingRangeIndex`.
        ///
        /// The `physicalDescriptorSetIndex` is the index in the `m_descriptorSets` array of
        /// the descriptor set that the range should be added to.
        ///
        /// The `offset` encodes information about space and/or register offsets that
        /// should be applied to descrptor ranges.
        ///
        /// Note that a single binding range may encompass zero or more descriptor ranges.
        ///
        void addBindingRange(
            slang::TypeLayoutReflection* typeLayout,
            Index physicalDescriptorSetIndex,
            BindingRegisterOffset const& containerOffset,
            BindingRegisterOffset const& elementOffset,
            Index bindingRangeIndex);

        void addAsValue(
            slang::VariableLayoutReflection* varLayout,
            Index physicalDescriptorSetIndex);

        /// Add binding ranges and parameter blocks to the root signature.
        ///
        /// The layout information is taken from `typeLayout` which should
        /// be a layout for either a program or an entry point.
        ///
        /// The `physicalDescriptorSetIndex` is the index in the `m_descriptorSets` array of
        /// the descriptor set that binding ranges not belonging to nested
        /// parameter blocks should be added to.
        ///
        /// The `offset` encodes information about space and/or register offsets that
        /// should be applied to descrptor ranges.
        ///
        void addAsConstantBuffer(
            slang::TypeLayoutReflection* typeLayout,
            Index physicalDescriptorSetIndex,
            BindingRegisterOffsetPair containerOffset,
            BindingRegisterOffsetPair elementOffset);

        void addAsValue(
            slang::TypeLayoutReflection* typeLayout,
            Index physicalDescriptorSetIndex,
            BindingRegisterOffsetPair containerOffset,
            BindingRegisterOffsetPair elementOffset);

        D3D12_ROOT_SIGNATURE_DESC1& build();
    };

    static Result createRootSignatureFromSlang(
        DeviceImpl* device,
        RootShaderObjectLayoutImpl* rootLayout,
        slang::IComponentType* program,
        ID3D12RootSignature** outRootSignature,
        ID3DBlob** outError);

    static Result create(
        DeviceImpl* device,
        slang::IComponentType* program,
        slang::ProgramLayout* programLayout,
        RootShaderObjectLayoutImpl** outLayout,
        ID3DBlob** outError);

    slang::IComponentType* getSlangProgram() const { return m_program; }
    slang::ProgramLayout* getSlangProgramLayout() const { return m_programLayout; }

protected:
    Result init(Builder* builder);

    ComPtr<slang::IComponentType> m_program;
    slang::ProgramLayout* m_programLayout = nullptr;

    List<EntryPointInfo> m_entryPoints;

public:
    ComPtr<ID3D12RootSignature> m_rootSignature;
};

} // namespace d3d12
} // namespace gfx
