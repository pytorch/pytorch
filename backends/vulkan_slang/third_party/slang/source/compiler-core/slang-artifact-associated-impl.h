// slang-artifact-associated-impl.h
#ifndef SLANG_ARTIFACT_ASSOCIATED_IMPL_H
#define SLANG_ARTIFACT_ASSOCIATED_IMPL_H

#include "../core/slang-com-object.h"
#include "../core/slang-memory-arena.h"
#include "slang-artifact-associated.h"
#include "slang-artifact-diagnostic-util.h"
#include "slang-artifact-util.h"
#include "slang-com-helper.h"
#include "slang-com-ptr.h"

namespace Slang
{

class ArtifactDiagnostics : public ComBaseObject, public IArtifactDiagnostics
{
public:
    typedef ArtifactDiagnostics ThisType;

    SLANG_COM_BASE_IUNKNOWN_ALL

    // ICastable
    SLANG_NO_THROW void* SLANG_MCALL castAs(const Guid& guid) SLANG_OVERRIDE;
    // IClonable
    SLANG_NO_THROW virtual void* SLANG_MCALL clone(const Guid& intf) SLANG_OVERRIDE;
    // IDiagnostic
    SLANG_NO_THROW virtual const Diagnostic* SLANG_MCALL getAt(Index i) SLANG_OVERRIDE
    {
        return &m_diagnostics[i];
    }
    SLANG_NO_THROW virtual Count SLANG_MCALL getCount() SLANG_OVERRIDE
    {
        return m_diagnostics.getCount();
    }
    SLANG_NO_THROW virtual void SLANG_MCALL add(const Diagnostic& diagnostic) SLANG_OVERRIDE;
    SLANG_NO_THROW virtual void SLANG_MCALL removeAt(Index i) SLANG_OVERRIDE
    {
        m_diagnostics.removeAt(i);
    }
    SLANG_NO_THROW virtual SlangResult SLANG_MCALL getResult() SLANG_OVERRIDE { return m_result; }
    SLANG_NO_THROW virtual void SLANG_MCALL setResult(SlangResult res) SLANG_OVERRIDE
    {
        m_result = res;
    }
    SLANG_NO_THROW virtual void SLANG_MCALL setRaw(const CharSlice& slice) SLANG_OVERRIDE;
    SLANG_NO_THROW virtual void SLANG_MCALL appendRaw(const CharSlice& slice) SLANG_OVERRIDE;
    SLANG_NO_THROW virtual TerminatedCharSlice SLANG_MCALL getRaw() SLANG_OVERRIDE
    {
        return SliceUtil::asTerminatedCharSlice(m_raw);
    }
    SLANG_NO_THROW virtual void SLANG_MCALL reset() SLANG_OVERRIDE;
    SLANG_NO_THROW virtual Count SLANG_MCALL getCountAtLeastSeverity(Diagnostic::Severity severity)
        SLANG_OVERRIDE;
    SLANG_NO_THROW virtual Count SLANG_MCALL getCountBySeverity(Diagnostic::Severity severity)
        SLANG_OVERRIDE;
    SLANG_NO_THROW virtual bool SLANG_MCALL hasOfAtLeastSeverity(Diagnostic::Severity severity)
        SLANG_OVERRIDE;
    SLANG_NO_THROW virtual Count SLANG_MCALL getCountByStage(
        Diagnostic::Stage stage,
        Count outCounts[Int(Diagnostic::Severity::CountOf)]) SLANG_OVERRIDE;
    SLANG_NO_THROW virtual void SLANG_MCALL removeBySeverity(Diagnostic::Severity severity)
        SLANG_OVERRIDE;
    SLANG_NO_THROW virtual void SLANG_MCALL maybeAddNote(const CharSlice& in) SLANG_OVERRIDE;
    SLANG_NO_THROW virtual void SLANG_MCALL requireErrorDiagnostic() SLANG_OVERRIDE;
    SLANG_NO_THROW virtual void SLANG_MCALL calcSummary(ISlangBlob** outBlob) SLANG_OVERRIDE;
    SLANG_NO_THROW virtual void SLANG_MCALL calcSimplifiedSummary(ISlangBlob** outBlob)
        SLANG_OVERRIDE;

    /// Default ctor
    ArtifactDiagnostics()
        : ComBaseObject()
    {
    }
    /// Copy ctor
    ArtifactDiagnostics(const ThisType& rhs);

    /// Create
    static ComPtr<IArtifactDiagnostics> create()
    {
        return ComPtr<IArtifactDiagnostics>(new ThisType);
    }

protected:
    void* getInterface(const Guid& uuid);
    void* getObject(const Guid& uuid);

    SliceAllocator m_allocator;

    List<Diagnostic> m_diagnostics;
    SlangResult m_result = SLANG_OK;

    // Raw diagnostics
    StringBuilder m_raw;
};

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!! ArtifactPostEmitMetadata !!!!!!!!!!!!!!!!!!!!!!!!!! */

struct ShaderBindingRange
{
    slang::ParameterCategory category = slang::ParameterCategory::None;
    UInt spaceIndex = 0;
    UInt registerIndex = 0;
    UInt registerCount = 0; // 0 for unsized

    bool isInfinite() const { return registerCount == 0; }

    bool containsBinding(slang::ParameterCategory _category, UInt _spaceIndex, UInt _registerIndex)
        const
    {
        return category == _category && spaceIndex == _spaceIndex &&
               registerIndex <= _registerIndex &&
               (isInfinite() || registerCount + registerIndex > _registerIndex);
    }

    bool intersectsWith(const ShaderBindingRange& other) const
    {
        if (category != other.category || spaceIndex != other.spaceIndex)
            return false;

        const bool leftIntersection =
            (registerIndex < other.registerIndex + other.registerCount) || other.isInfinite();
        const bool rightIntersection =
            (other.registerIndex < registerIndex + registerCount) || isInfinite();

        return leftIntersection && rightIntersection;
    }

    bool adjacentTo(const ShaderBindingRange& other) const
    {
        if (category != other.category || spaceIndex != other.spaceIndex)
            return false;

        const bool leftIntersection =
            (registerIndex <= other.registerIndex + other.registerCount) || other.isInfinite();
        const bool rightIntersection =
            (other.registerIndex <= registerIndex + registerCount) || isInfinite();

        return leftIntersection && rightIntersection;
    }

    void mergeWith(const ShaderBindingRange other)
    {
        UInt newRegisterIndex = Math::Min(registerIndex, other.registerIndex);

        if (other.isInfinite())
            registerCount = 0;
        else if (!isInfinite())
            registerCount = Math::Max(
                                registerIndex + registerCount,
                                other.registerIndex + other.registerCount) -
                            newRegisterIndex;

        registerIndex = newRegisterIndex;
    }

    static bool isUsageTracked(slang::ParameterCategory category)
    {
        switch (category)
        {
        case slang::ConstantBuffer:
        case slang::ShaderResource:
        case slang::UnorderedAccess:
        case slang::SamplerState:
        case slang::DescriptorTableSlot:
        case slang::VaryingInput:
        case slang::VaryingOutput:
        case slang::SpecializationConstant:
            return true;
        default:
            return false;
        }
    }
};

class ArtifactPostEmitMetadata : public ComBaseObject, public IArtifactPostEmitMetadata
{
public:
    typedef ArtifactPostEmitMetadata ThisType;

    SLANG_CLASS_GUID(0x6f82509f, 0xe48b, 0x4b83, {0xa3, 0x84, 0x5d, 0x70, 0x83, 0x19, 0x83, 0xcc})

    SLANG_COM_BASE_IUNKNOWN_ALL

    // ICastable
    SLANG_NO_THROW void* SLANG_MCALL castAs(const Guid& guid) SLANG_OVERRIDE;

    // IArtifactPostEmitMetadata
    SLANG_NO_THROW virtual Slice<ShaderBindingRange> SLANG_MCALL getUsedBindingRanges()
        SLANG_OVERRIDE;
    SLANG_NO_THROW virtual Slice<String> SLANG_MCALL getExportedFunctionMangledNames()
        SLANG_OVERRIDE;

    // IMetadata
    SLANG_NO_THROW virtual SlangResult isParameterLocationUsed(
        SlangParameterCategory category, // is this a `t` register? `s` register?
        SlangUInt spaceIndex,            // `space` for D3D12, `set` for Vulkan
        SlangUInt registerIndex,         // `register` for D3D12, `binding` for Vulkan
        bool& outUsed) SLANG_OVERRIDE;

    void* getInterface(const Guid& uuid);
    void* getObject(const Guid& uuid);

    static ComPtr<IArtifactPostEmitMetadata> create()
    {
        return ComPtr<IArtifactPostEmitMetadata>(new ThisType);
    }

    List<ShaderBindingRange> m_usedBindings;
    List<String> m_exportedFunctionMangledNames;
};

} // namespace Slang

#endif
