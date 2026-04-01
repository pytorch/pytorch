// debug-render-pass.h
#pragma once
#include "debug-base.h"

namespace gfx
{
using namespace Slang;

namespace debug
{

class DebugRenderPassLayout : public DebugObject<IRenderPassLayout>
{
public:
    SLANG_COM_OBJECT_IUNKNOWN_ALL;

public:
    IRenderPassLayout* getInterface(const Slang::Guid& guid);
};

} // namespace debug
} // namespace gfx
