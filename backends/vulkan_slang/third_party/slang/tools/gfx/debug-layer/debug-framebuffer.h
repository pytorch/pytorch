// debug-framebuffer.h
#pragma once
#include "debug-base.h"

namespace gfx
{
using namespace Slang;

namespace debug
{

class DebugFramebuffer : public DebugObject<IFramebuffer>
{
public:
    SLANG_COM_OBJECT_IUNKNOWN_ALL;

public:
    IFramebuffer* getInterface(const Slang::Guid& guid);
};

class DebugFramebufferLayout : public DebugObject<IFramebufferLayout>
{
public:
    SLANG_COM_OBJECT_IUNKNOWN_ALL;

public:
    IFramebufferLayout* getInterface(const Slang::Guid& guid);
};

} // namespace debug
} // namespace gfx
