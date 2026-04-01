// simple-render-pass-layout.h
#pragma once

// Implementation of a dummy render pass layout object that stores and holds its
// desc value. Used by targets that does not expose an API object for the render pass
// concept.

#include "core/slang-basic.h"
#include "core/slang-com-object.h"
#include "slang-gfx.h"

namespace gfx
{

class SimpleRenderPassLayout : public IRenderPassLayout, public Slang::ComObject
{
public:
    SLANG_COM_OBJECT_IUNKNOWN_ALL
    IRenderPassLayout* getInterface(const Slang::Guid& guid);

public:
    Slang::ShortList<TargetAccessDesc> m_renderTargetAccesses;
    TargetAccessDesc m_depthStencilAccess;
    bool m_hasDepthStencil;
    void init(const IRenderPassLayout::Desc& desc);
};

} // namespace gfx
