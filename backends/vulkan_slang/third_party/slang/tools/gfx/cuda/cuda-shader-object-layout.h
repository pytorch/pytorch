// cuda-shader-object-layout.h
#pragma once
#include "cuda-base.h"

namespace gfx
{
#ifdef GFX_ENABLE_CUDA
using namespace Slang;

namespace cuda
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
    List<SubObjectRangeInfo> subObjectRanges;
    List<BindingRangeInfo> m_bindingRanges;

    Index m_subObjectCount = 0;
    Index m_resourceCount = 0;

    ShaderObjectLayoutImpl(
        RendererBase* renderer,
        slang::ISession* session,
        slang::TypeLayoutReflection* layout);

    Index getResourceCount() const;
    Index getSubObjectCount() const;
    List<SubObjectRangeInfo>& getSubObjectRanges();
    BindingRangeInfo getBindingRange(Index index);
    Index getBindingRangeCount() const;
};

class RootShaderObjectLayoutImpl : public ShaderObjectLayoutImpl
{
public:
    slang::ProgramLayout* programLayout = nullptr;
    List<RefPtr<ShaderObjectLayoutImpl>> entryPointLayouts;
    RootShaderObjectLayoutImpl(RendererBase* renderer, slang::ProgramLayout* inProgramLayout);

    int getKernelIndex(UnownedStringSlice kernelName);

    void getKernelThreadGroupSize(int kernelIndex, UInt* threadGroupSizes);
};

} // namespace cuda
#endif
} // namespace gfx
