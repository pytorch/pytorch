// cpu-shader-object-layout.h
#pragma once
#include "cpu-base.h"

namespace gfx
{
using namespace Slang;

namespace cpu
{

struct BindingRangeInfo
{
    slang::BindingType bindingType;
    Index count;
    Index baseIndex; // Flat index for sub-objects
    Index subObjectIndex;

    // TODO: The `uniformOffset` field should be removed,
    // since it cannot be supported by the Slang reflection
    // API once we fix some design issues.
    //
    // It is only being used today for pre-allocation of sub-objects
    // for constant buffers and parameter blocks (which should be
    // deprecated/removed anyway).
    //
    // Note: We would need to bring this field back, plus
    // a lot of other complexity, if we ever want to support
    // setting of resources/buffers directly by a binding
    // range index and array index.
    //
    Index uniformOffset; // Uniform offset for a resource typed field.

    bool isSpecializable;
};

struct SubObjectRangeInfo
{
    RefPtr<ShaderObjectLayoutImpl> layout;
    Index bindingRangeIndex;
};

class ShaderObjectLayoutImpl : public ShaderObjectLayoutBase
{
public:
    // TODO: Once memory lifetime stuff is handled, there is
    // no specific need to even track binding or sub-object
    // ranges for CPU.

    size_t m_size = 0;
    List<SubObjectRangeInfo> subObjectRanges;
    List<BindingRangeInfo> m_bindingRanges;

    Index m_subObjectCount = 0;
    Index m_resourceCount = 0;

    ShaderObjectLayoutImpl(
        RendererBase* renderer,
        slang::ISession* session,
        slang::TypeLayoutReflection* layout);

    size_t getSize();
    Index getResourceCount() const;
    Index getSubObjectCount() const;
    List<SubObjectRangeInfo>& getSubObjectRanges();
    BindingRangeInfo getBindingRange(Index index);
    Index getBindingRangeCount() const;
};

class EntryPointLayoutImpl : public ShaderObjectLayoutImpl
{
private:
    slang::EntryPointLayout* m_entryPointLayout = nullptr;

public:
    EntryPointLayoutImpl(
        RendererBase* renderer,
        slang::ISession* session,
        slang::EntryPointLayout* entryPointLayout)
        : ShaderObjectLayoutImpl(renderer, session, entryPointLayout->getTypeLayout())
        , m_entryPointLayout(entryPointLayout)
    {
    }

    const char* getEntryPointName();
};

class RootShaderObjectLayoutImpl : public ShaderObjectLayoutImpl
{
public:
    slang::ProgramLayout* m_programLayout = nullptr;
    List<RefPtr<EntryPointLayoutImpl>> m_entryPointLayouts;

    RootShaderObjectLayoutImpl(
        RendererBase* renderer,
        slang::ISession* session,
        slang::ProgramLayout* programLayout);

    int getKernelIndex(UnownedStringSlice kernelName);
    void getKernelThreadGroupSize(int kernelIndex, UInt* threadGroupSizes);

    EntryPointLayoutImpl* getEntryPoint(Index index);
};

} // namespace cpu
} // namespace gfx
